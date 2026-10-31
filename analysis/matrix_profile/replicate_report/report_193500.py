"""Song-structure report for the 2 kHz, 10-hour 193500 recording.

Follows the MATLAB experiment (profiles193500, motifLengths193500,
motifsVarLen193500, songRelLocation193500, songOrder193500,
bandEnergy193500, minimaRelLocation193500). Nearest-neighbor distances
come from the translated matrix profile. Results are cached under
``cache_193500`` because a full length scan is expensive.
"""

from __future__ import annotations

import json
import os
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import fftconvolve, find_peaks
from scipy.stats import ks_2samp, wasserstein_distance

try:
    from analysis.matrix_profile import mpx_fast
except ImportError:  # run from this directory: another ``analysis`` may shadow ours
    for _name in [k for k in sys.modules if k == "analysis" or k.startswith("analysis.")]:
        del sys.modules[_name]
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from analysis.matrix_profile import mpx_fast

CACHE = Path(__file__).resolve().parent / "cache_193500"
WAV = Path(r"C:\Users\hoang\Downloads\SOCAL_H_76_230426_193500_df100.x.x.wav")
RATE = 10.0
WHALE_MFCC = Path(r"C:\Users\hoang\project\stumpy\Whales_Song")

SCAN_MINUTES = (2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
PHRASE_MINUTES = 1.0
SONG_MINUTES = 7.0

MOTIF_COLUMN = 2
ENERGY_COLUMN = 1
COARSE_PAIRS = ((2, 4), (4, 8), (8, 16))
FINE_PAIRS = ((6, 7), (7, 8), (8, 9), (9, 10))

THREADS = max(1, os.cpu_count() or 1)


def _matlab_mfcc(x, fs):
    import sys

    sys.path.insert(0, str(WHALE_MFCC))
    from whale_mfcc import matlab_mfcc

    return matlab_mfcc(
        x,
        fs,
        window_sec=0.1,
        hop_sec=0.1,
        num_coeffs=13,
        chunk_frames=20000,
        dtype=np.float64,
        verbose=True,
    )


def load_features(wav_path=WAV):
    """Return ``(c1, c0)`` at 10 Hz.

    Motif profiles use cepstral coefficient c1. The slow energy curve uses c0,
    coefficient 0: a prominence
    of 0.01 on that curve is the setting that yields the report's minima. The
    appended log-energy column is too small after the 30-minute mean is removed
    for that same threshold. Both arrays are copies, so later searches cannot
    write back into the cached coefficients.
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / "coeffs_10hz.npz"
    if path.exists():
        coeffs = np.load(path)["coeffs"]
    else:
        import scipy.io.wavfile as wavfile

        fs, x = wavfile.read(wav_path)
        if fs != 2000:
            raise RuntimeError(f"expected 2000 Hz audio, found {fs}")
        coeffs = _matlab_mfcc(np.asarray(x), int(fs))
        np.savez(path, coeffs=coeffs, fs=fs)
    if coeffs.shape[1] <= MOTIF_COLUMN:
        raise RuntimeError(
            f"MFCC cache has {coeffs.shape[1]} columns, need coefficient 2"
        )
    motif = np.ascontiguousarray(coeffs[:, MOTIF_COLUMN], dtype=np.float64)
    energy = np.ascontiguousarray(coeffs[:, ENERGY_COLUMN], dtype=np.float64)
    return motif, energy


def profile_for(series, minutes, rate=RATE):
    """Non-overlapping matrix profile at a motif length of ``minutes``."""
    CACHE.mkdir(parents=True, exist_ok=True)
    samples = int(round(minutes * 60.0 * rate))
    path = CACHE / f"profile_c1_{samples}.npz"
    if path.exists():
        cached = np.load(path)
        return cached["mp"], int(cached["w"])
    print(f"matrix profile: {minutes:g} min ({samples} samples)", flush=True)
    mp, _mpi = mpx_fast(series, samples, samples, nthreads=THREADS)
    np.savez(path, mp=mp, w=samples)
    return mp, samples


def correlation(mp, w):
    r = 1.0 - np.asarray(mp, dtype=np.float64) ** 2 / (2.0 * w)
    r[~np.isfinite(r)] = -np.inf
    return r


def corr_to_seed(series, seed, w):
    """Pearson correlation of every window with ``series[seed:seed + w]``.

    The query is a copy. Subtracting its mean must not change ``series``.
    """
    q = np.array(series[seed : seed + w], dtype=np.float64, copy=True)
    q -= q.mean()
    qnorm = np.sqrt(np.dot(q, q))
    dots = fftconvolve(series, q[::-1], mode="valid")
    c1 = np.cumsum(series)
    c2 = np.cumsum(series * series)
    s = c1[w - 1 :] - np.concatenate(([0.0], c1[: len(c1) - w]))
    s2 = c2[w - 1 :] - np.concatenate(([0.0], c2[: len(c2) - w]))
    energy = np.maximum(s2 - s * s / w, 0.0)
    return dots / (np.sqrt(energy) * qnorm)


def collect_copies(
    series,
    seed,
    w,
    rate,
    start_thr=0.2,
    step=0.01,
    min_thr=-1.0,
    max_gap_sec=1200,
    max_overlap=0.1,
    max_copies=10000,
):
    """One motif group, lowering the copy threshold until no 20-minute hole remains."""
    prof = corr_to_seed(series, seed, w)
    prof[~np.isfinite(prof)] = -np.inf
    min_sep = max(1, int(round((1.0 - max_overlap) * w)))
    order = np.argsort(prof)[::-1]
    accepted = []
    blocked = np.zeros(len(prof), dtype=bool)
    for idx in order:
        if prof[idx] < min_thr or len(accepted) >= max_copies:
            break
        if blocked[idx]:
            continue
        accepted.append(int(idx))
        lo = max(0, idx - min_sep + 1)
        blocked[lo : idx + min_sep] = True
    accepted = np.asarray(accepted, dtype=np.int64)
    acc_r = prof[accepted]
    n = len(series)
    thr = start_thr
    while True:
        pos = (
            np.sort(accepted[acc_r >= thr])
            if len(accepted)
            else np.array([], dtype=np.int64)
        )
        if len(pos):
            edges = np.concatenate(([0], pos, [n - w]))
            stretch = float(np.diff(edges).max())
        else:
            stretch = float(n)
        if stretch <= max_gap_sec * rate or thr - step < min_thr:
            break
        thr = round(thr - step, 10)
    pos = np.sort(accepted[acc_r >= thr])
    return {
        "threshold": float(thr),
        "starts": pos.astype(np.int64),
        "seed_r": float(prof[seed]) if 0 <= seed < len(prof) else float("nan"),
    }


def best_seed(mp, w):
    r = correlation(mp, w)
    seed = int(np.argmax(r))
    return seed, float(r[seed])


def start_intervals_min(starts, rate):
    if len(starts) < 2:
        return np.array([])
    return np.diff(np.sort(starts)) / rate / 60.0


def midpoint_sec(starts, w, rate):
    return (np.sort(starts) + w / 2.0) / rate


def compare_pair(starts_a, w_a, starts_b, w_b, rate, within_sec=60.0):
    """KS D, Wasserstein shift in minutes, and shared-midpoint count.

    ``starts_a`` is the shorter motif. A shorter occurrence is shared when its
    midpoint lies within ``within_sec`` of some longer occurrence.
    """
    gap_a = start_intervals_min(starts_a, rate)
    gap_b = start_intervals_min(starts_b, rate)
    ks = (
        float(ks_2samp(gap_a, gap_b, method="asymp").statistic)
        if len(gap_a) and len(gap_b)
        else float("nan")
    )
    shift = (
        float(wasserstein_distance(gap_a, gap_b))
        if len(gap_a) and len(gap_b)
        else float("nan")
    )
    mid_a = midpoint_sec(starts_a, w_a, rate)
    mid_b = midpoint_sec(starts_b, w_b, rate)
    if len(mid_b) == 0:
        shared = 0
    else:
        shared = 0
        j = 0
        for t in mid_a:
            while j < len(mid_b) and mid_b[j] < t - within_sec:
                j += 1
            k = j
            hit = False
            if k < len(mid_b) and mid_b[k] <= t + within_sec:
                hit = True
            if not hit:
                continue
            shared += 1
    return {
        "n_a": int(len(starts_a)),
        "n_b": int(len(starts_b)),
        "ks": ks,
        "w_min": shift,
        "shared": int(shared),
    }


def song_boundaries_sec(midpoints_sec):
    """Halfway between consecutive 7-minute centres. 53 centres -> 51 songs."""
    m = np.sort(np.asarray(midpoints_sec, dtype=np.float64))
    bounds = (m[:-1] + m[1:]) / 2.0
    lengths_min = np.diff(bounds) / 60.0
    return bounds, lengths_min


def assign_song(times_sec, bounds):
    """Return song index and relative location, or -1 if outside the songs."""
    times = np.asarray(times_sec, dtype=np.float64)
    k = np.searchsorted(bounds, times, side="right") - 1
    rel = np.full(len(times), np.nan)
    inside = (k >= 0) & (k < len(bounds) - 1)
    kk = k[inside]
    rel[inside] = (times[inside] - bounds[kk]) / (bounds[kk + 1] - bounds[kk])
    k = k.astype(np.int64)
    k[~inside] = -1
    return k, rel


def circular_stats(locations):
    loc = np.asarray(locations, dtype=np.float64)
    loc = loc[np.isfinite(loc)]
    theta = 2.0 * np.pi * loc
    C = np.cos(theta).mean()
    S = np.sin(theta).mean()
    R = float(np.hypot(C, S))
    mean = float((np.arctan2(S, C) % (2.0 * np.pi)) / (2.0 * np.pi))
    return mean, R


def rayleigh_p(locations):
    """CircStat ``circ_rtest`` p-value, including the small-sample correction."""
    loc = np.asarray(locations, dtype=np.float64)
    loc = loc[np.isfinite(loc)]
    n = len(loc)
    mean, R = circular_stats(loc)
    z = n * R * R
    p = np.exp(-z) * (
        1.0
        + (2.0 * z - z**2) / (4.0 * n)
        - (24.0 * z - 132.0 * z**2 + 76.0 * z**3 - 9.0 * z**4) / (288.0 * n**2)
    )
    return mean, R, float(p), n


def phrase_groups(series, mp, w, r_min=0.35, max_copies=30, max_groups=30):
    """Greedy groups. Seeds and copies need ``r_min``, with no overlaps."""
    r = correlation(mp, w)
    taken = np.zeros(len(r), dtype=bool)
    groups = []
    order = np.argsort(r)[::-1]
    for seed in order:
        if len(groups) >= max_groups:
            break
        seed = int(seed)
        if not np.isfinite(r[seed]) or r[seed] < r_min:
            break
        if taken[seed]:
            continue
        prof = corr_to_seed(series, seed, w)
        prof[~np.isfinite(prof)] = -np.inf
        prof[taken] = -np.inf
        blocked = taken.copy()
        copies = []
        for idx in np.argsort(prof)[::-1]:
            idx = int(idx)
            if prof[idx] < r_min:
                break
            if blocked[idx]:
                continue
            copies.append(idx)
            lo = max(0, idx - w + 1)
            blocked[lo : idx + w] = True
        others = [idx for idx in copies if idx != seed]
        if not others:
            taken[seed] = True
            continue
        for idx in copies:
            lo = max(0, idx - w + 1)
            taken[lo : idx + w] = True
        groups.append(
            {
                "seed": seed,
                "seed_r": float(prof[others[0]]),
                "starts": np.asarray(copies[:max_copies], dtype=np.int64),
                "n_matches": len(copies),
            }
        )
    return groups


def _merge_runs(labels):
    out = []
    for lab in labels:
        if not out or out[-1] != lab:
            out.append(int(lab))
    return out


def song_sequences(group_of_song, cut):
    """Groups in each song, ordered after moving the origin to ``cut``."""
    seqs = []
    for items in group_of_song:
        if not items:
            seqs.append([])
            continue
        items = sorted(items, key=lambda gr: (gr[1] - cut) % 1.0)
        seqs.append(_merge_runs([g for g, _rel in items]))
    return seqs


def largest_empty_cut(locations):
    loc = np.sort(np.asarray(locations, dtype=np.float64) % 1.0)
    if len(loc) < 2:
        return 0.0
    gaps = np.diff(loc)
    wrap = loc[0] + 1.0 - loc[-1]
    k = int(np.argmax(gaps))
    if wrap >= gaps[k]:
        return float((loc[-1] + wrap / 2.0) % 1.0)
    return float(loc[k] + gaps[k] / 2.0)


def _agrees(seq, order_pos):
    last = -1
    seen = set()
    for g in seq:
        if g in seen:
            return False
        seen.add(g)
        pos = order_pos[g]
        if pos <= last:
            return False
        last = pos
    return True


def consensus(seqs, labels):
    """Order of ``labels`` agreed with by the most songs that have 2+ groups."""
    usable = [s for s in seqs if len(set(s)) >= 2]
    best_n = -1
    best = None
    for perm in permutations(range(len(labels))):
        pos = {i: p for p, i in enumerate(perm)}
        n_ok = sum(_agrees(s, pos) for s in usable)
        if n_ok > best_n:
            best_n = n_ok
            best = perm
    named = [labels[i] for i in best]
    return named, int(best_n), len(usable)


def movmean(x, k):
    """MATLAB ``movmean`` (centered, shrinking endpoints)."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    c = np.empty(n + 1, dtype=np.float64)
    c[0] = 0.0
    np.cumsum(x, out=c[1:])
    idx = np.arange(n)
    if k % 2 == 0:
        a = np.maximum(idx - k // 2, 0)
        b = np.minimum(idx + k // 2, n)
    else:
        half = k // 2
        a = np.maximum(idx - half, 0)
        b = np.minimum(idx + half + 1, n)
    return (c[b] - c[a]) / (b - a)


def energy_minima(energy, rate, bounds):
    """Slow energy curve: subtract a 30-minute mean, then a 4-minute Gaussian."""
    win = int(round(1800 * rate))
    detrended = energy - movmean(energy, win)
    sigma = 240.0 * rate
    smooth = gaussian_filter1d(detrended, sigma=sigma, truncate=2.5, mode="nearest")
    distance = int(round(480.0 * rate))
    peaks, _props = find_peaks(-smooth, prominence=0.01, distance=distance)
    times = peaks / rate
    song, rel = assign_song(times, bounds)
    inside = song >= 0
    intervals = np.diff(times) / 60.0
    mean, R, p, n_loc = rayleigh_p(rel[inside])
    return {
        "n_minima": int(len(peaks)),
        "n_inside": int(np.sum(inside)),
        "median_interval_min": (
            float(np.median(intervals)) if len(intervals) else float("nan")
        ),
        "circular_mean": mean,
        "concentration": R,
        "rayleigh_p": p,
        "n_locations": n_loc,
        "times_sec": times,
        "relative": rel,
    }


def _scan_one(series, minutes, rate):
    mp, w = profile_for(series, minutes, rate)
    seed, seed_r = best_seed(mp, w)
    copies = collect_copies(series, seed, w, rate)
    return {
        "minutes": minutes,
        "w": w,
        "best_r": seed_r,
        "threshold": copies["threshold"],
        "starts": copies["starts"],
    }


def _pair_table(scans, pairs, rate):
    table = []
    for shorter, longer in pairs:
        compared = compare_pair(
            scans[shorter]["starts"],
            scans[shorter]["w"],
            scans[longer]["starts"],
            scans[longer]["w"],
            rate,
        )
        compared["pair"] = f"{shorter:g} vs {longer:g}"
        table.append(compared)
    return table


def build_report(wav_path=WAV, rate=RATE):
    motif, energy = load_features(wav_path)
    scans = {m: _scan_one(motif, m, rate) for m in SCAN_MINUTES}
    phrase_mp, phrase_w = profile_for(motif, PHRASE_MINUTES, rate)

    song = scans[SONG_MINUTES]
    centres = midpoint_sec(song["starts"], song["w"], rate)
    bounds, lengths_min = song_boundaries_sec(centres)

    groups = phrase_groups(motif, phrase_mp, phrase_w)
    kept = [g for g in groups if len(g["starts"]) >= 8]
    labels = [f"G{i}" for i in range(1, len(kept) + 1)]

    mids = []
    raw_rel = []
    per_song = [[] for _ in range(max(len(bounds) - 1, 0))]
    group_locs = []
    for gi, g in enumerate(kept):
        times = midpoint_sec(g["starts"], phrase_w, rate)
        song_i, rel = assign_song(times, bounds)
        locs = []
        for s, rv, t in zip(song_i, rel, times):
            mids.append(t)
            raw_rel.append(rv)
            if s < 0:
                continue
            locs.append(float(rv))
            per_song[s].append((gi, float(rv)))
        songs_hit = len({int(s) for s in song_i if s >= 0})
        mean, conc = circular_stats(locs)
        group_locs.append(
            {
                "label": labels[gi],
                "occurrences": int(len(g["starts"])),
                "songs": songs_hit,
                "seed_r": g["seed_r"],
                "circular_mean": mean,
                "concentration": conc,
            }
        )

    finite_rel = np.asarray([v for v in raw_rel if np.isfinite(v)], dtype=np.float64)
    cut = largest_empty_cut(finite_rel)
    seqs = song_sequences(per_song, cut)
    order, n_agree, n_usable = consensus(seqs, labels)

    coarse_pairs = _pair_table(scans, COARSE_PAIRS, rate)
    fine_pairs = _pair_table(scans, FINE_PAIRS, rate)
    minima = energy_minima(energy, rate, bounds)

    report = {
        "n_frames": int(len(motif)),
        "motif_coefficient": 1,
        "best_r": {str(m): scans[m]["best_r"] for m in SCAN_MINUTES},
        "n_copies": {str(m): int(len(scans[m]["starts"])) for m in SCAN_MINUTES},
        "thresholds": {str(m): scans[m]["threshold"] for m in SCAN_MINUTES},
        "coarse_pairs": coarse_pairs,
        "fine_pairs": fine_pairs,
        "coarse_2_vs_4": coarse_pairs[0],
        "fine_6_vs_7": fine_pairs[0],
        "n_song_markers": int(len(song["starts"])),
        "n_songs": int(len(lengths_min)),
        "song_length_median_min": (
            float(np.median(lengths_min)) if len(lengths_min) else float("nan")
        ),
        "song_length_min_min": (
            float(lengths_min.min()) if len(lengths_min) else float("nan")
        ),
        "song_length_max_min": (
            float(lengths_min.max()) if len(lengths_min) else float("nan")
        ),
        "n_phrase_groups": len(groups),
        "n_kept": len(kept),
        "groups": group_locs,
        "order_cut": cut,
        "consensus_order": order,
        "songs_agree": n_agree,
        "songs_compared": n_usable,
        "minima": {
            k: minima[k]
            for k in (
                "n_minima",
                "n_inside",
                "median_interval_min",
                "circular_mean",
                "concentration",
                "rayleigh_p",
                "n_locations",
            )
        },
    }
    CACHE.mkdir(parents=True, exist_ok=True)
    (CACHE / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    report = build_report()
    print(json.dumps(report, indent=2))
