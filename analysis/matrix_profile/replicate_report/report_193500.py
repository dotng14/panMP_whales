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
RATE = 10.0
WHALE_MFCC = Path(r"C:\Users\hoang\project\stumpy\Whales_Song")

_WAV_CANDIDATES = (
    Path(r"C:\Users\hoang\Downloads\SOCAL_H_76_230426_193500_df100.x.x.wav"),
    WHALE_MFCC / "SOCAL_H_76_230426_193500_df100.x.x.wav",
)


def resolve_wav(wav_path=None) -> Path:
    """Return an existing 193500 WAV path."""
    if wav_path is not None:
        path = Path(wav_path)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    for path in _WAV_CANDIDATES:
        if path.exists():
            return path
    raise FileNotFoundError(
        "193500 WAV not found; tried:\n" + "\n".join(str(p) for p in _WAV_CANDIDATES)
    )


WAV = resolve_wav()

SCAN_MINUTES = (2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
FINE_MINUTES = (6, 7, 8, 9, 10)
COARSE_MINUTES = (2, 4, 8, 16)
PHRASE_MINUTES = 1.0
SONG_MINUTES = 7.0

MOTIF_COLUMN = 2
ENERGY_COLUMN = 1
COARSE_PAIRS = ((2, 4), (4, 8), (8, 16))
FINE_PAIRS = ((6, 7), (7, 8), (8, 9), (9, 10))

# Prefer Audio Toolbox MFCC when present (coeffs_matlab_10hz.mat / .npz).
MATLAB_COEFFS_MAT = CACHE / "coeffs_matlab_10hz.mat"
MATLAB_COEFFS_NPZ = CACHE / "coeffs_matlab_10hz.npz"
PROFILE_PREFIX = "profile_c1"  # overridden to profile_ml_c1 when using MATLAB MFCC

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


def _load_matlab_coeffs():
    """Load Audio Toolbox MFCC cache written by ``run_mfcc_full.m``."""
    global PROFILE_PREFIX
    if MATLAB_COEFFS_NPZ.exists():
        coeffs = np.load(MATLAB_COEFFS_NPZ)["coeffs"]
        PROFILE_PREFIX = "profile_ml_c1"
        return np.asarray(coeffs, dtype=np.float64)
    if MATLAB_COEFFS_MAT.exists():
        try:
            import scipy.io as sio

            data = sio.loadmat(MATLAB_COEFFS_MAT)
            coeffs = np.asarray(data["coeffs"], dtype=np.float64)
            fs = int(np.asarray(data.get("fs", 2000)).flat[0])
        except NotImplementedError:
            import h5py

            with h5py.File(MATLAB_COEFFS_MAT, "r") as f:
                raw = np.array(f["coeffs"])
                coeffs = raw.T if raw.shape[0] < raw.shape[1] else raw
                fs = int(np.array(f["fs"]).flat[0]) if "fs" in f else 2000
            coeffs = np.asarray(coeffs, dtype=np.float64)
        np.savez(MATLAB_COEFFS_NPZ, coeffs=coeffs, fs=fs)
        PROFILE_PREFIX = "profile_ml_c1"
        return coeffs
    return None


def load_features(wav_path=None):
    """Return ``(motif_coeff, energy_coeff)`` at 10 Hz.

    Prefers MATLAB Audio Toolbox MFCC (``coeffs_matlab_10hz``) when available.
    Motif profiles use column index 2 (cepstral c1 after appended log-energy).
    Energy minima use column index 1 (c0). Both arrays are copies.
    """
    global PROFILE_PREFIX
    CACHE.mkdir(parents=True, exist_ok=True)
    coeffs = _load_matlab_coeffs()
    if coeffs is None:
        PROFILE_PREFIX = "profile_c1"
        path = CACHE / "coeffs_10hz.npz"
        if path.exists():
            coeffs = np.load(path)["coeffs"]
        else:
            import scipy.io.wavfile as wavfile

            fs, x = wavfile.read(resolve_wav(wav_path))
            if fs != 2000:
                raise RuntimeError(f"expected 2000 Hz audio, found {fs}")
            coeffs = _matlab_mfcc(np.asarray(x), int(fs))
            np.savez(path, coeffs=coeffs, fs=fs)
    if coeffs.shape[1] <= MOTIF_COLUMN:
        raise RuntimeError(
            f"MFCC cache has {coeffs.shape[1]} columns, need coefficient 2"
        )
    print(
        f"features: shape={coeffs.shape} profile_prefix={PROFILE_PREFIX}",
        flush=True,
    )
    motif = np.ascontiguousarray(coeffs[:, MOTIF_COLUMN], dtype=np.float64)
    energy = np.ascontiguousarray(coeffs[:, ENERGY_COLUMN], dtype=np.float64)
    return motif, energy


def profile_for(series, minutes, rate=RATE):
    """Non-overlapping matrix profile at a motif length of ``minutes``."""
    CACHE.mkdir(parents=True, exist_ok=True)
    samples = int(round(minutes * 60.0 * rate))
    path = CACHE / f"{PROFILE_PREFIX}_{samples}.npz"
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
    """Greedy groups. Seeds and copies need ``r_min``, with no overlaps.

    ``taken`` blocks accepted group copies (and a single failed seed index),
    matching the MATLAB search. A separate ``tried`` mask skips re-FFTing
    every one-sample shift of a seed that was already evaluated.
    """
    r = correlation(mp, w)
    taken = np.zeros(len(r), dtype=bool)
    tried = np.zeros(len(r), dtype=bool)
    groups = []
    order = np.argsort(r)[::-1]
    attempts = 0
    for seed in order:
        if len(groups) >= max_groups:
            break
        seed = int(seed)
        if not np.isfinite(r[seed]) or r[seed] < r_min:
            break
        if taken[seed] or tried[seed]:
            continue
        attempts += 1
        if attempts % 25 == 1:
            print(
                f"  phrase attempt {attempts}: seed={seed} r={r[seed]:.3f} "
                f"groups={len(groups)}",
                flush=True,
            )
        # Do not retry one-sample shifts of this seed.
        lo = max(0, seed - w + 1)
        tried[lo : seed + w] = True
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
            clo = max(0, idx - w + 1)
            blocked[clo : idx + w] = True
        others = [idx for idx in copies if idx != seed]
        if not others:
            taken[seed] = True
            continue
        for idx in copies[:max_copies]:
            clo = max(0, idx - w + 1)
            taken[clo : idx + w] = True
        groups.append(
            {
                "seed": seed,
                "seed_r": float(prof[others[0]]),
                "starts": np.asarray(copies[:max_copies], dtype=np.int64),
                "n_matches": len(copies),
            }
        )
    print(f"  phrase groups done: {len(groups)} (attempts={attempts})", flush=True)
    return groups


def dedupe_phrase_groups(series, groups, w, corr_max=0.5, min_copies=8, max_keep=7):
    """Keep diverse groups: drop a later seed if it correlates with an earlier kept seed.

    Discovery order is already by matrix-profile match strength. MATLAB's write-up
    keeps seven groups with at least eight occurrences (G1–G7).
    """
    kept = []
    for g in groups:
        if len(g["starts"]) < min_copies:
            continue
        seed = int(g["seed"])
        duplicate = False
        for prev in kept:
            # Correlation of this seed window with the previous group's seed.
            a = np.asarray(series[seed : seed + w], dtype=np.float64)
            b = np.asarray(
                series[int(prev["seed"]) : int(prev["seed"]) + w], dtype=np.float64
            )
            a = a - a.mean()
            b = b - b.mean()
            denom = np.sqrt(np.dot(a, a) * np.dot(b, b))
            rho = float(np.dot(a, b) / denom) if denom > 0 else 1.0
            if rho >= corr_max:
                duplicate = True
                break
        if duplicate:
            continue
        kept.append(g)
        if len(kept) >= max_keep:
            break
    return kept


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


def length_summary(scan, rate=RATE):
    """Per-length row for the fine/coarse motif tables."""
    starts = scan["starts"]
    gaps = start_intervals_min(starts, rate)
    n = int(len(starts))
    under = int(np.sum(gaps < 10.0)) if len(gaps) else 0
    if len(gaps):
        # Hazen / MATLAB-like percentiles (matches the published IQR rounding).
        q1, med, q3 = np.percentile(gaps, [25, 50, 75], method="hazen")
        median_iqr = f"{med:.1f} ({q1:.1f}–{q3:.1f})"
        median = float(med)
        iqr = (float(q1), float(q3))
    else:
        median_iqr = "—"
        median = float("nan")
        iqr = (float("nan"), float("nan"))
    return {
        "minutes": float(scan["minutes"]),
        "occurrences": n,
        "threshold": float(scan["threshold"]),
        "best_r": float(scan["best_r"]),
        "median_interval_min": median,
        "iqr": iqr,
        "median_iqr": median_iqr,
        "under_10": under,
        "n_gaps": int(len(gaps)),
        "gaps": gaps,
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
        n_a, n_b = compared["n_a"], compared["n_b"]
        base = max(n_a, n_b)
        compared["pair"] = f"{shorter:g} vs {longer:g}"
        compared["count_diff_pct"] = (
            float("nan") if base == 0 else 100.0 * abs(n_a - n_b) / base
        )
        table.append(compared)
    return table


def pairwise_order(seqs, labels, pairs=None):
    """Count how often label A precedes B within the same song."""
    if pairs is None:
        pairs = [
            ("G6", "G2"),
            ("G1", "G4"),
            ("G6", "G1"),
            ("G6", "G3"),
            ("G1", "G2"),
        ]
    index = {lab: i for i, lab in enumerate(labels)}
    rows = []
    for earlier, later in pairs:
        if earlier not in index or later not in index:
            continue
        ia, ib = index[earlier], index[later]
        n_ok = 0
        n_both = 0
        for seq in seqs:
            if ia not in seq or ib not in seq:
                continue
            n_both += 1
            if seq.index(ia) < seq.index(ib):
                n_ok += 1
        # Two-sided binomial against 0.5; indicative only (report note).
        from math import comb

        if n_both == 0:
            p = float("nan")
        else:
            tail = sum(comb(n_both, k) for k in range(n_ok, n_both + 1))
            p = min(1.0, 2.0 * tail / (2**n_both))
        rows.append(
            {
                "earlier": earlier,
                "later": later,
                "n_ok": n_ok,
                "n_both": n_both,
                "p": float(p),
            }
        )
    return rows


def _circular_agrees(seq, pos, nlab):
    """True if ``seq`` is increasing along the cycle ``pos`` (any start)."""
    present = [g for g in seq if g in pos]
    if len(set(present)) < 2:
        return False
    vals = [pos[g] for g in present]
    # Lift into a linear order starting at the first group's position.
    start = vals[0]
    lifted = [(v - start) % nlab for v in vals]
    return all(lifted[i] < lifted[i + 1] for i in range(len(lifted) - 1))


def circular_consensus(seqs, labels):
    """Best cyclic order: most songs agree up to rotation."""
    usable = [s for s in seqs if len(set(s)) >= 2]
    best_n = -1
    best = None
    nlab = len(labels)
    if nlab == 0:
        return [], 0, 0
    for perm in permutations(range(nlab)):
        pos = {lab: i for i, lab in enumerate(perm)}
        n_ok = sum(1 for seq in usable if _circular_agrees(seq, pos, nlab))
        if n_ok > best_n:
            best_n = n_ok
            best = perm
    named = [labels[i] for i in best]
    return named, int(best_n), len(usable)


def shuffle_null(seqs, labels, n_shuffles=100, seed=0):
    """Repeat linear/circular consensus after shuffling order inside each song."""
    rng = np.random.default_rng(seed)
    linear_scores = []
    circular_scores = []
    for _ in range(n_shuffles):
        shuffled = []
        for seq in seqs:
            if len(seq) < 2:
                shuffled.append(list(seq))
                continue
            arr = list(seq)
            rng.shuffle(arr)
            shuffled.append(arr)
        _order, n_lin, _n = consensus(shuffled, labels)
        _corder, n_circ, _nc = circular_consensus(shuffled, labels)
        linear_scores.append(n_lin)
        circular_scores.append(n_circ)
    return {
        "n_shuffles": int(n_shuffles),
        "linear_mean": float(np.mean(linear_scores)),
        "linear_max": int(np.max(linear_scores)) if linear_scores else 0,
        "circular_mean": float(np.mean(circular_scores)),
        "circular_max": int(np.max(circular_scores)) if circular_scores else 0,
        "linear_scores": [int(x) for x in linear_scores],
        "circular_scores": [int(x) for x in circular_scores],
    }


def marker_offset_stats(minima_times_sec, marker_centres_sec):
    """Offset of each energy minimum from the nearest 7-min marker centre."""
    times = np.asarray(minima_times_sec, dtype=np.float64)
    centres = np.asarray(marker_centres_sec, dtype=np.float64)
    if len(times) == 0 or len(centres) == 0:
        return {
            "n_inside_marker": 0,
            "median_offset_min": float("nan"),
            "iqr_offset_min": (float("nan"), float("nan")),
            "n_before_centre": 0,
        }
    half = 3.5 * 60.0  # half of a 7-min marker
    offsets = []
    inside = 0
    before = 0
    for t in times:
        j = int(np.argmin(np.abs(centres - t)))
        off = t - centres[j]
        offsets.append(off)
        if abs(off) <= half:
            inside += 1
        if off < 0:
            before += 1
    offsets = np.asarray(offsets) / 60.0
    q1, med, q3 = np.percentile(offsets, [25, 50, 75])
    return {
        "n_inside_marker": int(inside),
        "median_offset_min": float(med),
        "iqr_offset_min": (float(q1), float(q3)),
        "n_before_centre": int(before),
        "offsets_min": offsets,
    }


def build_report(wav_path=None, rate=RATE, n_shuffles=100):
    wav_path = resolve_wav(wav_path)
    motif, energy = load_features(wav_path)
    scans = {}
    for m in SCAN_MINUTES:
        print(f"scan {m:g} min", flush=True)
        scans[m] = _scan_one(motif, m, rate)
    print("phrase profile 1 min", flush=True)
    phrase_mp, phrase_w = profile_for(motif, PHRASE_MINUTES, rate)

    song = scans[SONG_MINUTES]
    centres = midpoint_sec(song["starts"], song["w"], rate)
    bounds, lengths_min = song_boundaries_sec(centres)

    print("phrase groups", flush=True)
    groups = phrase_groups(motif, phrase_mp, phrase_w)
    kept = [g for g in groups if len(g["starts"]) >= 8][:7]
    labels = [f"G{i}" for i in range(1, len(kept) + 1)]

    mids = []
    raw_rel = []
    per_song = [[] for _ in range(max(len(bounds) - 1, 0))]
    group_locs = []
    kept_duration_h = 0.0
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
        kept_duration_h += len(g["starts"]) * (phrase_w / rate) / 3600.0
        group_locs.append(
            {
                "label": labels[gi],
                "occurrences": int(len(g["starts"])),
                "songs": songs_hit,
                "seed_r": g["seed_r"],
                "circular_mean": mean,
                "concentration": conc,
                "relative_locations": locs,
                "times_sec": times.tolist(),
            }
        )

    all_duration_h = sum(len(g["starts"]) for g in groups) * (phrase_w / rate) / 3600.0

    finite_rel = np.asarray([v for v in raw_rel if np.isfinite(v)], dtype=np.float64)
    cut = largest_empty_cut(finite_rel)
    seqs = song_sequences(per_song, cut)
    print("consensus order", flush=True)
    order, n_agree, n_usable = consensus(seqs, labels)
    circ_order, n_circ_agree, n_circ_usable = circular_consensus(seqs, labels)
    print(f"shuffle null ({n_shuffles})", flush=True)
    shuffle = shuffle_null(seqs, labels, n_shuffles=n_shuffles)
    # p-value: fraction of shuffles with score >= observed (min 1/n for 100 shuffles).
    lin_ge = sum(1 for s in shuffle["linear_scores"] if s >= n_agree)
    circ_ge = sum(1 for s in shuffle["circular_scores"] if s >= n_circ_agree)
    shuffle["linear_p"] = max(1.0 / max(n_shuffles, 1), lin_ge / max(n_shuffles, 1))
    shuffle["circular_p"] = max(1.0 / max(n_shuffles, 1), circ_ge / max(n_shuffles, 1))
    # Fixed random order baseline: mean agreements of one random perm.
    rng = np.random.default_rng(1)
    fixed_scores = []
    for _ in range(200):
        perm = list(range(len(labels)))
        rng.shuffle(perm)
        pos = {i: p for p, i in enumerate(perm)}
        fixed_scores.append(sum(_agrees(s, pos) for s in seqs if len(set(s)) >= 2))
    shuffle["fixed_random_linear_mean"] = float(np.mean(fixed_scores)) if fixed_scores else float("nan")

    pair_orders = pairwise_order(seqs, labels)

    coarse_pairs = _pair_table(scans, COARSE_PAIRS, rate)
    fine_pairs = _pair_table(scans, FINE_PAIRS, rate)
    fine_lengths = [length_summary(scans[m], rate) for m in FINE_MINUTES]
    coarse_lengths = [length_summary(scans[m], rate) for m in COARSE_MINUTES]
    # Drop non-JSON gaps arrays from the serialized summaries.
    fine_lengths_json = [
        {k: v for k, v in row.items() if k != "gaps"} for row in fine_lengths
    ]
    coarse_lengths_json = [
        {k: v for k, v in row.items() if k != "gaps"} for row in coarse_lengths
    ]

    minima = energy_minima(energy, rate, bounds)
    offsets = marker_offset_stats(minima["times_sec"], centres)
    intervals = np.diff(minima["times_sec"]) / 60.0
    minima_extra = {
        "median_interval_min": float(np.median(intervals)) if len(intervals) else float("nan"),
        "iqr_interval_min": (
            tuple(float(x) for x in np.percentile(intervals, [25, 75]))
            if len(intervals)
            else (float("nan"), float("nan"))
        ),
        "relative_median": (
            float(np.nanmedian(minima["relative"]))
            if np.any(np.isfinite(minima["relative"]))
            else float("nan")
        ),
        "relative_iqr": (
            tuple(
                float(x)
                for x in np.nanpercentile(
                    minima["relative"][np.isfinite(minima["relative"])], [25, 75]
                )
            )
            if np.any(np.isfinite(minima["relative"]))
            else (float("nan"), float("nan"))
        ),
        "songs_with_minimum": int(np.sum(np.isfinite(minima["relative"]))),
        **{k: offsets[k] for k in (
            "n_inside_marker",
            "median_offset_min",
            "iqr_offset_min",
            "n_before_centre",
        )},
    }

    # JSON-safe group locs (drop bulky arrays from the written file's main view
    # but keep them available on the returned object via a separate key).
    group_locs_json = [
        {k: v for k, v in g.items() if k not in ("relative_locations", "times_sec")}
        for g in group_locs
    ]

    report = {
        "n_frames": int(len(motif)),
        "motif_coefficient": 2,
        "best_r": {str(m): scans[m]["best_r"] for m in SCAN_MINUTES},
        "n_copies": {str(m): int(len(scans[m]["starts"])) for m in SCAN_MINUTES},
        "thresholds": {str(m): scans[m]["threshold"] for m in SCAN_MINUTES},
        "coarse_pairs": coarse_pairs,
        "fine_pairs": fine_pairs,
        "coarse_2_vs_4": coarse_pairs[0],
        "fine_6_vs_7": fine_pairs[0],
        "fine_lengths": fine_lengths_json,
        "coarse_lengths": coarse_lengths_json,
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
        "phrase_coverage_kept_h": kept_duration_h,
        "phrase_coverage_all_h": all_duration_h,
        "groups": group_locs_json,
        "order_cut": cut,
        "consensus_order": order,
        "songs_agree": n_agree,
        "songs_compared": n_usable,
        "circular_order": circ_order,
        "circular_songs_agree": n_circ_agree,
        "circular_songs_compared": n_circ_usable,
        "pairwise_order": pair_orders,
        "shuffle": {
            k: shuffle[k]
            for k in (
                "n_shuffles",
                "linear_mean",
                "linear_max",
                "circular_mean",
                "circular_max",
                "linear_p",
                "circular_p",
                "fixed_random_linear_mean",
                "linear_scores",
                "circular_scores",
            )
        },
        "minima": {
            **{
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
            **minima_extra,
        },
        # Runtime-only (not written to the slim JSON below when huge):
        "_fine_gaps": {str(int(r["minutes"])): r["gaps"].tolist() for r in fine_lengths},
        "_group_details": group_locs,
        "_marker_centres_sec": centres.tolist(),
        "_song_bounds_sec": bounds.tolist(),
        "_minima_times_sec": minima["times_sec"].tolist(),
        "_minima_relative": minima["relative"].tolist(),
    }

    def _jsonable(obj):
        if isinstance(obj, dict):
            return {k: _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonable(v) for v in obj]
        if isinstance(obj, (np.floating, float)):
            return float(obj)
        if isinstance(obj, (np.integer, int)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    CACHE.mkdir(parents=True, exist_ok=True)
    slim = {
        k: _jsonable(v)
        for k, v in report.items()
        if not str(k).startswith("_")
    }
    # Keep shuffle score histograms out of the on-disk JSON (large, PDF-only).
    if "shuffle" in slim:
        slim["shuffle"] = {
            k: v
            for k, v in slim["shuffle"].items()
            if k not in ("linear_scores", "circular_scores")
        }
    (CACHE / "report.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    # Full payload with arrays for the PDF writer.
    np.savez(
        CACHE / "report_arrays.npz",
        marker_centres_sec=centres,
        song_bounds_sec=bounds,
        minima_times_sec=minima["times_sec"],
        minima_relative=minima["relative"],
    )
    return report


if __name__ == "__main__":
    report = build_report()
    slim = {k: v for k, v in report.items() if not str(k).startswith("_")}
    print(json.dumps(slim, indent=2))
