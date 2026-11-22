#!/usr/bin/env python3
"""Random Site H wavs: 0–500 Hz energy ratio at 0.2 s, per-minute ACF peak/shift, 3-panel plots.

ACF peaks are restricted to lags of 1–20 s. For each file write one full-file plot.
"""

from __future__ import annotations

import csv
import math
import random
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import correlate, find_peaks

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from analysis.dict_HMM.find_songs.siteH_all_songs.hires_energy_ratio import (  # noqa: E402
    WINDOW_SEC,
    layout_for,
    open_wav,
    read_samples,
    window_frames,
)

OUT_DIR = Path(__file__).resolve().parent / "random_ratio_acf"
MANIFEST = Path(__file__).resolve().parent / "manifest.csv"
SONGS_CSV = Path(__file__).resolve().parent / "conforming_songs.csv"
DT = 0.2
MINUTE = 60.0
LO_HZ = 0.0
HI_HZ = 500.0
SPEC_MAX_HZ = 2500.0
LAG_LO_S = 1.0
LAG_HI_S = 20.0
# One spectrogram column per second keeps song-length panels readable.
SPEC_COL_SEC = 1.0


def load_song_intervals() -> dict[str, list[tuple[float, float]]]:
    """Map wav_path -> [(start_sec, end_sec), ...] from conforming_songs.csv."""
    out: dict[str, list[tuple[float, float]]] = {}
    if not SONGS_CSV.is_file():
        return out
    with SONGS_CSV.open() as handle:
        for row in csv.DictReader(handle):
            path = row["wav_path"]
            a = float(row["start_sec"])
            b = float(row["end_sec"])
            if b > a:
                out.setdefault(path, []).append((a, b))
    for path in out:
        out[path].sort()
    return out


def eligible_wavs() -> tuple[list[Path], list[Path]]:
    """Return (all eligible wavs, wavs known to contain songs via status ok_song)."""
    all_paths: list[Path] = []
    song_paths: list[Path] = []
    with MANIFEST.open() as handle:
        for row in csv.DictReader(handle):
            if row["status"] == "fail":
                continue
            path = Path(row["wav_path"])
            if not path.is_file():
                continue
            all_paths.append(path)
            if row["status"] == "ok_song":
                song_paths.append(path)
    return all_paths, song_paths


def band_rows(layout: dict) -> tuple[int, int]:
    """Inclusive row indices for [LO_HZ, HI_HZ] inside the whitened spectrogram."""
    rate = layout["rate"]
    fftl = layout["fftl"]
    row0 = layout["row0"]
    blo = max(int(round(LO_HZ * fftl / rate)), row0)
    bhi = min(int(round(HI_HZ * fftl / rate)), row0 + layout["nrows"] - 1)
    return blo - row0, bhi - row0


def minute_acf(ratio: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-minute max ACF and lag, restricted to [LAG_LO_S, LAG_HI_S]."""
    n_bins = ratio.size
    n_min = max(1, int(math.floor(n_bins * DT / MINUTE)))
    max_acf = np.full(n_min, np.nan)
    shift_s = np.full(n_min, np.nan)
    t_min = (np.arange(n_min) + 0.5) * MINUTE
    samples_per_min = int(round(MINUTE / DT))
    lo = max(1, int(round(LAG_LO_S / DT)))
    hi = min(int(round(LAG_HI_S / DT)), samples_per_min - 1)
    for m in range(n_min):
        a = m * samples_per_min
        b = min((m + 1) * samples_per_min, n_bins)
        x = ratio[a:b]
        if x.size < hi + 2:
            continue
        x = x - x.mean()
        if float(np.dot(x, x)) <= 0:
            continue
        acf = correlate(x, x, mode="full")
        acf = acf[x.size - 1 :] / acf[x.size - 1]
        hi_m = min(hi, acf.size - 1)
        if hi_m < lo:
            continue
        seg = acf[lo : hi_m + 1]
        peaks, _ = find_peaks(seg, height=0.0)
        if peaks.size:
            j = int(peaks[np.argmax(seg[peaks])]) + lo
        else:
            j = int(np.argmax(seg)) + lo
        max_acf[m] = float(acf[j])
        shift_s[m] = float(j * DT)
    return t_min, max_acf, shift_s


def process(path: Path, *, has_songs: bool, song_intervals: list[tuple[float, float]] | None = None) -> dict:
    rate, data_off, n_samples = open_wav(path)
    layout = layout_for(rate)
    b0, b1 = band_rows(layout)
    duration = n_samples / rate
    n_bins = max(1, int(math.ceil(duration / DT)))
    band = np.zeros(n_bins, dtype=np.float64)
    overall = np.zeros(n_bins, dtype=np.float64)
    seg = max(1, int(round(DT * rate)))
    win = WINDOW_SEC * rate

    row_hz = rate / layout["fftl"]
    n_freq = min(layout["nrows"], max(1, int(round(SPEC_MAX_HZ / row_hz)) - layout["row0"] + 1))
    n_freq = max(n_freq, b1 + 1)
    n_cols = max(1, int(math.ceil(duration / SPEC_COL_SEC)))
    col_sec = duration / n_cols
    spec = np.zeros((n_freq, n_cols), dtype=np.float64)
    spec_n = np.zeros(n_cols, dtype=np.float64)

    n_win = int(math.ceil(n_samples / win))
    for w in range(n_win):
        start = w * win
        if start >= n_samples:
            break
        n = min(win, n_samples - start)
        samples = read_samples(path, data_off, n_samples, start, n)
        smooth, centers = window_frames(samples, layout)
        if centers.size == 0:
            continue
        abs_center = start + centers
        energy = smooth ** 2
        overall_e = energy.sum(axis=0)
        band_e = energy[b0 : b1 + 1].sum(axis=0)
        bins = np.clip((abs_center // seg).astype(int), 0, n_bins - 1)
        np.add.at(band, bins, band_e)
        np.add.at(overall, bins, overall_e)

        cols = np.clip((abs_center / rate / col_sec).astype(int), 0, n_cols - 1)
        piece = energy[:n_freq]
        for c in np.unique(cols):
            mask = cols == c
            spec[:, c] += piece[:, mask].sum(axis=1)
            spec_n[c] += float(mask.sum())

    ratio = np.divide(band, overall, out=np.zeros_like(band), where=overall > 0)
    t_ratio = np.arange(n_bins) * DT
    t_min, max_acf, shift_s = minute_acf(ratio)

    with np.errstate(invalid="ignore", divide="ignore"):
        spec_mean = np.divide(spec, spec_n, out=np.zeros_like(spec), where=spec_n > 0)

    return {
        "path": path,
        "has_songs": has_songs,
        "song_label": "KNOWN SONGS" if has_songs else "NO KNOWN SONGS",
        "song_intervals": list(song_intervals or []),
        "rate": rate,
        "duration": duration,
        "t_ratio": t_ratio,
        "ratio": ratio,
        "t_min": t_min,
        "max_acf": max_acf,
        "shift_s": shift_s,
        "spec": spec_mean,
        "f_hz": (layout["row0"] + np.arange(n_freq)) * row_hz,
        "col_sec": col_sec,
    }


def _slice_spec(spec: np.ndarray, col_sec: float, t0: float, t1: float) -> tuple[np.ndarray, float, float]:
    c0 = max(0, int(math.floor(t0 / col_sec)))
    c1 = min(spec.shape[1], max(c0 + 1, int(math.ceil(t1 / col_sec))))
    return spec[:, c0:c1], c0 * col_sec, c1 * col_sec


def plot_window(
    data: dict,
    out_png: Path,
    *,
    t0: float,
    t1: float,
    title_extra: str,
) -> None:
    path: Path = data["path"]
    f_hz = data["f_hz"]
    duration = data["duration"]
    t1 = min(t1, duration)
    if t1 <= t0:
        return

    spec_w, s0, s1 = _slice_spec(data["spec"], data["col_sec"], t0, t1)
    disp = np.maximum(spec_w, 0.0) ** (1.0 / 3.0)
    vmax = float(np.percentile(disp, 99.9)) if disp.size else 1.0
    if not vmax > 0:
        vmax = 1.0

    rmask = (data["t_ratio"] >= t0) & (data["t_ratio"] < t1)
    mmask = (data["t_min"] >= t0) & (data["t_min"] < t1)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 9),
        layout="constrained",
        sharex=True,
        gridspec_kw={"height_ratios": [1.6, 1.0, 1.2]},
    )
    ax0, ax1, ax2 = axes
    ax0.imshow(
        disp,
        origin="lower",
        aspect="auto",
        cmap="jet",
        vmin=0.0,
        vmax=vmax,
        extent=(s0 / 60.0, s1 / 60.0, f_hz[0], f_hz[-1]),
        interpolation="nearest",
    )
    ax0.set_ylabel("Hz")
    ax0.set_ylim(0, min(SPEC_MAX_HZ, f_hz[-1]))
    song_label = data.get("song_label", "UNKNOWN")
    ax0.set_title(
        f"{path.name}\n"
        f"{song_label}  ·  {title_extra}  ·  0–500 Hz / total  ·  ACF lag {LAG_LO_S:.0f}–{LAG_HI_S:.0f} s"
    )
    badge_color = "#1b7f3a" if data.get("has_songs") else "#7a7a7a"
    ax0.text(
        0.01,
        0.97,
        song_label,
        transform=ax0.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color="white",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=badge_color, edgecolor="none", alpha=0.92),
    )

    ax1.plot(data["t_ratio"][rmask] / 60.0, data["ratio"][rmask], color="#1f4e79", lw=0.7)
    ax1.set_ylabel("Energy ratio\n0–500 Hz / total")
    ax1.set_ylim(0, 1)

    ax2.plot(data["t_min"][mmask] / 60.0, data["max_acf"][mmask], color="#1f4e79", lw=1.0, marker="o", ms=3, label="max ACF")
    ax2.set_ylabel("Max ACF")
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel("Minutes from file start")
    ax2b = ax2.twinx()
    ax2b.plot(
        data["t_min"][mmask] / 60.0,
        data["shift_s"][mmask],
        color="#b85c38",
        lw=0.9,
        marker="s",
        ms=3,
        alpha=0.85,
        label="shift (s)",
    )
    ax2b.set_ylabel("Shift (s)", color="#b85c38")
    ax2b.set_ylim(LAG_LO_S - 0.5, LAG_HI_S + 0.5)
    ax2.set_xlim(t0 / 60.0, t1 / 60.0)
    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper right", frameon=False, fontsize=9)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=120)
    plt.close(fig)


def write_all_plots(data: dict, out_png: Path) -> int:
    """Write full-file plot, npz, and one plot per conforming song. Return PNG count."""
    path: Path = data["path"]
    duration = data["duration"]
    plot_window(
        data,
        out_png,
        t0=0.0,
        t1=duration,
        title_extra=f"{duration / 3600:.2f} h full file",
    )
    intervals = data.get("song_intervals") or []
    np.savez_compressed(
        out_png.with_suffix(".npz"),
        t_ratio=data["t_ratio"],
        ratio=data["ratio"],
        t_min=data["t_min"],
        max_acf=data["max_acf"],
        shift_s=data["shift_s"],
        duration=np.array([duration]),
        rate=np.array([data["rate"]]),
        lag_lo_s=np.array([LAG_LO_S]),
        lag_hi_s=np.array([LAG_HI_S]),
        has_songs=np.array([bool(data.get("has_songs"))]),
        song_label=np.array(data.get("song_label", "")),
        song_starts=np.array([a for a, _ in intervals], dtype=np.float64),
        song_ends=np.array([b for _, b in intervals], dtype=np.float64),
        wav=np.array(str(path)),
    )
    n = 1
    for i, (a, b) in enumerate(intervals, start=1):
        a = max(0.0, float(a))
        b = min(duration, float(b))
        if b <= a:
            continue
        name = f"{path.stem}.song{i:03d}_{int(a)}-{int(b)}s.ratio_acf.png"
        plot_window(
            data,
            out_png.parent / name,
            t0=a,
            t1=b,
            title_extra=f"single song {i}/{len(intervals)} · {a / 60:.1f}–{b / 60:.1f} min ({(b - a) / 60:.1f} min)",
        )
        n += 1
    return n


def out_name(path: Path) -> str:
    return f"{path.stem}.ratio_acf.png"


def is_full_file_plot(name: str) -> bool:
    """True for stem.ratio_acf.png, false for .songNNN_ or legacy five-minute names."""
    import re as _re

    return not _re.search(r"\.song\d{3}_", name) and not _re.search(r"\.m\d{4}-\d{4}\.", name)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wavs, song_wavs = eligible_wavs()
    if not wavs:
        raise SystemExit("no wavs in manifest")
    song_set = set(song_wavs)
    song_intervals = load_song_intervals()
    rng = random.Random()
    done = {p.name for p in OUT_DIR.glob("*.ratio_acf.png") if is_full_file_plot(p.name)}
    n_files = len(done)
    n_pngs = len(list(OUT_DIR.glob("*.ratio_acf.png")))
    n_from_song = n_from_any = 0
    print(
        f"eligible {len(wavs)} song_known {len(song_wavs)} "
        f"with_intervals {len(song_intervals)} files_done {n_files} pngs {n_pngs} out {OUT_DIR}",
        flush=True,
    )
    print(
        f"ACF lag {LAG_LO_S:.0f}–{LAG_HI_S:.0f}s; full-file + per-song plots; "
        "pick 50% any / 50% ok_song",
        flush=True,
    )
    while True:
        unseen = [w for w in wavs if not (OUT_DIR / out_name(w)).exists()]
        if not unseen:
            print(f"all {len(wavs)} files already plotted; stopping", flush=True)
            break
        # Prefer files with timed conforming songs so single-song plots can be made.
        unseen_with_songs = [w for w in unseen if str(w) in song_intervals]
        unseen_ok_song = [w for w in unseen if w in song_set]
        song_pool = unseen_with_songs or unseen_ok_song
        # Coin flip so ~half of new files are known songs.
        if song_pool and rng.random() < 0.5:
            path = rng.choice(song_pool)
            pool = "song"
        else:
            path = rng.choice(unseen)
            pool = "any"
        out_png = OUT_DIR / out_name(path)
        t0 = time.time()
        intervals = song_intervals.get(str(path), [])
        has_songs = path in song_set or bool(intervals)
        print(
            f"start [{pool}] {'SONGS' if has_songs else 'no-songs'} "
            f"n_songs={len(intervals)} {path.name}",
            flush=True,
        )
        try:
            data = process(path, has_songs=has_songs, song_intervals=intervals)
            n = write_all_plots(data, out_png)
        except Exception as exc:  # noqa: BLE001
            print(f"FAIL {path.name}: {exc}", flush=True)
            continue
        n_files += 1
        n_pngs += n
        if pool == "song":
            n_from_song += 1
        else:
            n_from_any += 1
        print(
            f"wrote {out_png.name} + {n - 1} song plots  minutes {data['max_acf'].size}  "
            f"{time.time() - t0:.1f}s  files {n_files} pngs {n_pngs} "
            f"(any {n_from_any} song {n_from_song})",
            flush=True,
        )


if __name__ == "__main__":
    main()
