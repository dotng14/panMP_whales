#!/usr/bin/env python3
"""Identify peaks that repeat with a fixed period in a 1-D energy series.

Reads a time series from an ``.npz`` (keys ``t`` and ``band_energy``, or
``t`` and ``y``) or from two columns in a CSV. Finds peaks, ranks candidate
periods by autocorrelation, scores how regularly peak gaps match each
period, and optionally writes a figure and a JSON summary.

Example::

    python -m analysis.dict_HMM.find_songs.repeating_peaks \\
        analysis/dict_HMM/find_songs/three_songs/song_20230427_015045_band_energy_50_500.npz \\
        --t0 560 --t1 720 --out-dir analysis/dict_HMM/find_songs/three_songs
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.signal import correlate, find_peaks


@dataclass
class PeriodHit:
    """One candidate fixed period."""

    period_s: float
    acf: float
    n_gaps_near: int
    n_gaps: int
    longest_run: int
    median_gap_s: float | None
    runs: list[tuple[float, float, int]]  # (t_start, t_end, n_intervals)


@dataclass
class RepeatingPeaksResult:
    """Peaks and the ranked fixed-period pattern."""

    dt: float
    t0: float
    t1: float
    n_samples: int
    n_peaks: int
    peak_times: list[float]
    peak_heights: list[float]
    gaps: list[float]
    periods: list[PeriodHit]
    best_period_s: float | None


def load_series(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(t, y)`` from ``.npz`` or two-column CSV."""
    path = Path(path)
    if path.suffix == ".npz":
        data = np.load(path)
        if "t" not in data:
            raise ValueError(f"{path} needs array 't'")
        if "band_energy" in data:
            y = np.asarray(data["band_energy"], dtype=np.float64)
        elif "y" in data:
            y = np.asarray(data["y"], dtype=np.float64)
        elif "ratio" in data:
            y = np.asarray(data["ratio"], dtype=np.float64)
        else:
            raise ValueError(f"{path} needs 'band_energy', 'y', or 'ratio'")
        return np.asarray(data["t"], dtype=np.float64), y
    if path.suffix == ".csv":
        arr = np.loadtxt(path, delimiter=",", ndmin=2)
        if arr.shape[1] < 2:
            raise ValueError(f"{path} needs at least two columns (t, y)")
        return arr[:, 0].astype(np.float64), arr[:, 1].astype(np.float64)
    raise ValueError(f"unsupported input {path} (use .npz or .csv)")


def smooth(y: np.ndarray, dt: float, seconds: float) -> np.ndarray:
    """Moving average of width ``seconds``."""
    win = max(1, int(round(seconds / dt)))
    if win == 1:
        return np.asarray(y, dtype=np.float64)
    kernel = np.ones(win, dtype=np.float64) / win
    return np.convolve(np.asarray(y, dtype=np.float64), kernel, mode="same")


def detect_peaks(
    t: np.ndarray,
    y: np.ndarray,
    *,
    dt: float,
    min_separation_s: float = 1.5,
    height_percentile: float = 75.0,
    prominence_percentile: float = 65.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Peak times and heights on a smoothed series ``y``."""
    height = float(np.percentile(y, height_percentile))
    prominence = float(np.percentile(y, prominence_percentile))
    distance = max(1, int(round(min_separation_s / dt)))
    idx, _ = find_peaks(y, height=height, prominence=prominence, distance=distance)
    return t[idx], y[idx]


def autocorrelation_periods(
    y: np.ndarray,
    dt: float,
    *,
    min_period_s: float = 2.0,
    max_period_s: float = 60.0,
    top_k: int = 8,
) -> list[tuple[float, float]]:
    """Return ``[(period_s, acf), ...]`` ranked by autocorrelation."""
    x = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    if x.size < 8 or float(np.dot(x, x)) <= 0.0:
        return []
    acf = correlate(x, x, mode="full")
    acf = acf[x.size - 1 :]
    acf = acf / acf[0]
    lo = max(1, int(round(min_period_s / dt)))
    hi = min(acf.size - 1, int(round(max_period_s / dt)))
    if hi <= lo:
        return []
    seg = acf[lo:hi]
    peaks, _ = find_peaks(seg, height=0.05, distance=max(1, int(round(min_period_s / dt))))
    if peaks.size == 0:
        # fall back to the single strongest lag
        j = int(np.argmax(seg)) + lo
        return [(j * dt, float(acf[j]))]
    lags = peaks + lo
    order = np.argsort(acf[lags])[::-1]
    out: list[tuple[float, float]] = []
    for i in lags[order[:top_k]]:
        out.append((float(i * dt), float(acf[i])))
    return out


def _near_period(values: np.ndarray, period_s: float, tol: float) -> np.ndarray:
    """Boolean mask: each value is within ``tol`` of ``period_s`` or ``2*period_s``."""
    return (np.abs(values - period_s) / period_s <= tol) | (
        np.abs(values - 2.0 * period_s) / (2.0 * period_s) <= tol
    )


def _gap_support(
    peak_t: np.ndarray,
    period_s: float,
    *,
    tol: float = 0.15,
    min_run: int = 3,
) -> tuple[int, int, int, list[tuple[float, float, int]], np.ndarray]:
    """Match single gaps, two-gap sums, and contiguous runs to ``period_s``."""
    if peak_t.size < 2:
        return 0, 0, 0, [], np.array([])
    gaps = np.diff(peak_t)
    single = _near_period(gaps, period_s, tol)
    pair = np.zeros(gaps.size, dtype=bool)
    if gaps.size >= 2:
        pair_vals = gaps[:-1] + gaps[1:]
        pair_hit = np.abs(pair_vals - period_s) / period_s <= tol
        # mark both gaps of each hitting pair
        for i, ok in enumerate(pair_hit):
            if ok:
                pair[i] = pair[i + 1] = True
    near = single | pair
    runs: list[tuple[float, float, int]] = []
    best = cur = 0
    start: int | None = None
    for i, ok in enumerate(near):
        if ok:
            if start is None:
                start = i
            cur += 1
            best = max(best, cur)
        else:
            if start is not None and cur >= min_run:
                runs.append((float(peak_t[start]), float(peak_t[start + cur]), int(cur)))
            start = None
            cur = 0
    if start is not None and cur >= min_run:
        runs.append((float(peak_t[start]), float(peak_t[start + cur]), int(cur)))
    longest = max(best, max((r[2] for r in runs), default=0))
    return int(near.sum()), int(gaps.size), int(longest), runs, gaps[near]


def score_period(peak_t: np.ndarray, period_s: float, *, tol: float = 0.15) -> PeriodHit:
    n_near, n_gaps, longest, runs, near_gaps = _gap_support(peak_t, period_s, tol=tol)
    return PeriodHit(
        period_s=float(period_s),
        acf=0.0,
        n_gaps_near=n_near,
        n_gaps=n_gaps,
        longest_run=int(longest),
        median_gap_s=float(np.median(near_gaps)) if near_gaps.size else None,
        runs=runs,
    )


def find_repeating_peaks(
    t: np.ndarray,
    y: np.ndarray,
    *,
    t0: float | None = None,
    t1: float | None = None,
    smooth_s: float = 0.4,
    min_separation_s: float = 1.5,
    height_percentile: float = 75.0,
    prominence_percentile: float = 65.0,
    min_period_s: float = 2.0,
    max_period_s: float = 60.0,
    top_k: int = 8,
    gap_tol: float = 0.15,
) -> RepeatingPeaksResult:
    """Detect peaks and rank fixed periods that explain their spacing."""
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if t.size != y.size or t.size < 8:
        raise ValueError("t and y must have the same length >= 8")
    order = np.argsort(t)
    t, y = t[order], y[order]
    dt = float(np.median(np.diff(t)))
    if dt <= 0:
        raise ValueError("non-positive time step")

    lo = float(t[0] if t0 is None else t0)
    hi = float(t[-1] if t1 is None else t1)
    mask = (t >= lo) & (t <= hi)
    tt, yy = t[mask], y[mask]
    if tt.size < 8:
        raise ValueError(f"fewer than 8 samples in [{lo}, {hi}]")

    yy_s = smooth(yy, dt, smooth_s)
    peak_t, peak_h = detect_peaks(
        tt,
        yy_s,
        dt=dt,
        min_separation_s=min_separation_s,
        height_percentile=height_percentile,
        prominence_percentile=prominence_percentile,
    )
    gaps = np.diff(peak_t).tolist() if peak_t.size >= 2 else []

    acf_periods = autocorrelation_periods(
        yy_s, dt, min_period_s=min_period_s, max_period_s=max_period_s, top_k=top_k
    )
    hits: list[PeriodHit] = []
    for period_s, acf in acf_periods:
        hit = score_period(peak_t, period_s, tol=gap_tol)
        hit.acf = acf
        hits.append(hit)

    median_gap = float(np.median(gaps)) if gaps else 0.0

    def rank_key(h: PeriodHit) -> tuple:
        # Prefer phrase-scale periods (≳ 1.5× median unit gap) with strong ACF
        # and gap support; still keep shorter unit periods in the list.
        phrase = 1.0 if median_gap <= 0.0 or h.period_s >= 1.5 * median_gap else 0.0
        return (phrase, h.acf, h.longest_run, h.n_gaps_near)

    hits_sorted = sorted(hits, key=rank_key, reverse=True)
    best = hits_sorted[0].period_s if hits_sorted else None

    return RepeatingPeaksResult(
        dt=dt,
        t0=lo,
        t1=hi,
        n_samples=int(tt.size),
        n_peaks=int(peak_t.size),
        peak_times=peak_t.tolist(),
        peak_heights=peak_h.tolist(),
        gaps=gaps,
        periods=hits_sorted,
        best_period_s=best,
    )


def plot_result(
    t: np.ndarray,
    y: np.ndarray,
    result: RepeatingPeaksResult,
    out_png: Path,
    *,
    smooth_s: float = 0.4,
) -> None:
    """Energy, ACF markers, and fold at the best period."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask = (t >= result.t0) & (t <= result.t1)
    tt, yy = t[mask], y[mask]
    yy_s = smooth(yy, result.dt, smooth_s)
    peak_t = np.asarray(result.peak_times)
    peak_h = np.asarray(result.peak_heights)
    best = result.best_period_s

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 8.5),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [1.5, 1.0, 1.1]},
    )
    axes[0].plot(tt, yy_s, color="#1f4e79", lw=0.8)
    if peak_t.size:
        axes[0].plot(peak_t, peak_h, "o", color="0.45", ms=3.5, label="peaks")
    if best is not None and peak_t.size:
        # phrase-spaced subset
        dist = max(1, int(round(0.7 * best / result.dt)))
        idx, _ = find_peaks(yy_s, height=np.percentile(yy_s, 80), distance=dist)
        ph_t = tt[idx]
        axes[0].plot(ph_t, yy_s[idx], "D", color="#c0392b", ms=5, label=f"~{best:.1f}s peaks")
        if ph_t.size:
            grid = ph_t[0] + np.arange(0, int((tt[-1] - ph_t[0]) / best) + 2) * best
            for g in grid:
                if tt[0] <= g <= tt[-1]:
                    axes[0].axvline(g, color="#c0392b", alpha=0.2, lw=0.9)
    axes[0].set_xlim(result.t0, result.t1)
    axes[0].set_ylabel("Series")
    title = "Repeating peaks"
    if best is not None:
        title += f" · best period ≈ {best:.2f} s"
    axes[0].set_title(title)
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)

    # ACF panel
    x = yy_s - yy_s.mean()
    acf = correlate(x, x, mode="full")
    acf = acf[x.size - 1 :] / acf[x.size - 1]
    max_lag = min(acf.size - 1, int(round(60.0 / result.dt)))
    lags = np.arange(max_lag + 1) * result.dt
    axes[1].plot(lags, acf[: max_lag + 1], color="#1f4e79", lw=0.9)
    for hit in result.periods[:5]:
        axes[1].axvline(hit.period_s, color="#c0392b", ls="--", lw=0.8)
        axes[1].text(
            hit.period_s,
            min(0.95, hit.acf + 0.03),
            f"{hit.period_s:.1f}s",
            rotation=90,
            fontsize=8,
            va="bottom",
        )
    axes[1].set_xlim(0, min(60.0, lags[-1] if lags.size else 60.0))
    axes[1].set_ylabel("ACF")
    axes[1].set_xlabel("Lag (s)")
    axes[1].set_title("Autocorrelation")

    # Fold
    if best is not None and best > 2 * result.dt:
        n_cycles = int((tt[-1] - tt[0]) / best)
        grid = np.linspace(0, best, max(8, int(round(best / result.dt))))
        fold = []
        for k in range(max(n_cycles, 0)):
            a = tt[0] + k * best
            b = a + best
            mm = (tt >= a) & (tt < b)
            if mm.sum() < 10:
                continue
            fold.append(np.interp(grid, tt[mm] - a, yy_s[mm]))
        if fold:
            fold_arr = np.asarray(fold)
            im = axes[2].imshow(
                fold_arr,
                aspect="auto",
                origin="lower",
                cmap="magma",
                extent=(0, best, 0, len(fold_arr)),
                interpolation="nearest",
            )
            axes[2].set_xlabel(f"Phase within {best:.1f} s (s)")
            axes[2].set_ylabel("Cycle")
            axes[2].set_title(f"Folded at {best:.1f} s — {len(fold_arr)} cycles")
            fig.colorbar(im, ax=axes[2], fraction=0.02, pad=0.01)
        else:
            axes[2].text(0.5, 0.5, "no cycles to fold", ha="center", transform=axes[2].transAxes)
            axes[2].set_axis_off()
    else:
        axes[2].text(0.5, 0.5, "no period found", ha="center", transform=axes[2].transAxes)
        axes[2].set_axis_off()

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m analysis.dict_HMM.find_songs.repeating_peaks",
        description="Find peaks that repeat with a fixed period in a 1-D series.",
    )
    p.add_argument("series", type=Path, help=".npz with t+band_energy/y/ratio, or CSV t,y")
    p.add_argument("--t0", type=float, default=None, help="start time (default: first sample)")
    p.add_argument("--t1", type=float, default=None, help="end time (default: last sample)")
    p.add_argument("--smooth-s", type=float, default=0.4, help="smoothing window in seconds")
    p.add_argument("--min-separation-s", type=float, default=1.5, help="minimum peak spacing")
    p.add_argument("--height-percentile", type=float, default=75.0)
    p.add_argument("--prominence-percentile", type=float, default=65.0)
    p.add_argument("--min-period-s", type=float, default=2.0)
    p.add_argument("--max-period-s", type=float, default=60.0)
    p.add_argument("--top-k", type=int, default=8, help="ACF periods to keep")
    p.add_argument("--gap-tol", type=float, default=0.15, help="relative tolerance for gap≈period")
    p.add_argument("--out-dir", type=Path, default=None, help="write summary JSON and figure here")
    p.add_argument("--plot", action="store_true", help="write repeating_peaks.png")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    t, y = load_series(args.series)
    result = find_repeating_peaks(
        t,
        y,
        t0=args.t0,
        t1=args.t1,
        smooth_s=args.smooth_s,
        min_separation_s=args.min_separation_s,
        height_percentile=args.height_percentile,
        prominence_percentile=args.prominence_percentile,
        min_period_s=args.min_period_s,
        max_period_s=args.max_period_s,
        top_k=args.top_k,
        gap_tol=args.gap_tol,
    )

    payload = asdict(result)
    print(
        json.dumps(
            {
                "n_peaks": result.n_peaks,
                "best_period_s": result.best_period_s,
                "periods": [
                    {
                        "period_s": h.period_s,
                        "acf": h.acf,
                        "n_gaps_near": h.n_gaps_near,
                        "longest_run": h.longest_run,
                    }
                    for h in result.periods[:5]
                ],
            },
            indent=2,
        )
    )

    if args.out_dir is not None or args.plot:
        out_dir = Path(args.out_dir) if args.out_dir is not None else Path(args.series).parent
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "repeating_peaks.json").write_text(json.dumps(payload, indent=2))
        if args.plot or args.out_dir is not None:
            plot_result(t, y, result, out_dir / "repeating_peaks.png", smooth_s=args.smooth_s)
            print(f"wrote {out_dir / 'repeating_peaks.json'}")
            print(f"wrote {out_dir / 'repeating_peaks.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
