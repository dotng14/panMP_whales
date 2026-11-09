"""Diagnosis against Payne & McVay's song properties, and a test of the song-boundary idea.

    python analysis/song_hmm_phrase/diagnostics/boundary_checks.py

1. Theme lengths per song in the CSV  vs  the models' minimum theme length.
2. Session gaps decoded by each model  vs  the rule "no pause longer than 1 min inside a session".
3. Song-scale repetition in the audio: similarity of phrase windows at lag L (minutes).
   A song repeats, so the audio should be most similar to itself one song later.
4. Oracle: if song boundaries were known (CSV song starts), split each song into its 6 themes
   in order (optimal 6-segment split of the phrase descriptors) and compare with the CSV themes.
Needs the outputs of both experiments (train + report).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from analysis.song_hmm.labels import load_labels  # noqa: E402
from analysis.song_hmm.train import CSV_PATH, SPLIT_S  # noqa: E402
from analysis.song_hmm_phrase.metrics import unit_rows  # noqa: E402
from analysis.song_hmm_phrase.train import WORK  # noqa: E402

EXP1_WORK = Path(__file__).resolve().parents[2] / "song_hmm" / "work"
N = 36000


def runs(mask: np.ndarray) -> np.ndarray:
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0])).reshape(-1, 2)
    return e[:, 1] - e[:, 0]


def segment_ordered(x: np.ndarray, k: int) -> np.ndarray:
    """Optimal split of x (n, d) into k contiguous segments minimising within-segment squared error."""
    n = len(x)
    c1 = np.vstack([np.zeros(x.shape[1]), np.cumsum(x, 0)])
    c2 = np.r_[0.0, np.cumsum((x * x).sum(1))]

    def cost(a, b):  # segment [a, b)
        s = c1[b] - c1[a]
        return c2[b] - c2[a] - (s @ s) / (b - a)

    best = np.full((k + 1, n + 1), np.inf)
    arg = np.zeros((k + 1, n + 1), dtype=int)
    best[0, 0] = 0.0
    for j in range(1, k + 1):
        for b in range(j, n + 1):
            vals = [best[j - 1, a] + cost(a, b) for a in range(j - 1, b)]
            i = int(np.argmin(vals))
            best[j, b], arg[j, b] = vals[i], i + j - 1
    lab = np.empty(n, dtype=int)
    b = n
    for j in range(k, 0, -1):
        a = arg[j, b]
        lab[a:b] = j - 1
        b = a
    return lab


def main() -> None:
    L = load_labels(CSV_PATH, N)

    print("1. CSV seconds per theme per song (median [IQR]); models' minimum theme length: Exp 1 none (≥ 3 s phrase), Exp 2 20 s")
    th_s = []
    for a, b in L.song_bounds.astype(int):
        th_s.append(np.bincount(L.theme[a:b][L.theme[a:b] >= 0], minlength=6))
    th_s = np.array(th_s)
    for k in range(6):
        v = th_s[:, k]
        print(f"   T{k + 1}: {np.median(v):6.0f} s  [{np.percentile(v, 25):.0f}-{np.percentile(v, 75):.0f}]   songs with < 20 s: {np.mean(v < 20):.0%}")

    print("\n2. decoded gaps (background runs) inside the recording, test half; a session ends only after a pause > 60 s")
    for name, path in (
        ("Exp 1", EXP1_WORK / "test" / "posteriors.npz"),
        ("Exp 2", WORK / "exp2" / "test" / "decoded.npz"),
        ("Exp 2b", WORK / "exp2b" / "test" / "decoded.npz"),
    ):
        d = np.load(path)
        if "session" in d:
            bg = ~d["session"]
        else:
            from analysis.song_hmm.topology import IS_BG

            bg = IS_BG[d["path"]]
        r = runs(bg)
        print(f"   {name:7s} gaps: {len(r):3d}   ≤ 60 s: {np.sum(r <= 60):3d}   61-120 s: {np.sum((r > 60) & (r <= 120)):3d}   > 120 s: {np.sum(r > 120):3d}")

    f = np.load(WORK / "phrase_features.npz")
    print("\n3. song-scale repetition: mean cosine of phrase windows L minutes apart (2 s steps)")
    lags_min = np.arange(1, 31)
    for half in ("train", "test"):
        z = unit_rows(f[f"z_{half}"])
        sims = np.array([np.mean(np.sum(z[: -lg * 30] * z[lg * 30 :], 1)) for lg in lags_min])
        best = lags_min[4:][np.argmax(sims[4:])]  # ignore < 5 min
        print(f"   {half}: " + " ".join(f"{m}:{s:.3f}" for m, s in zip(lags_min, sims, strict=True) if m in (1, 3, 5, 7, 9, 10, 11, 12, 13, 14, 15, 17, 20, 25, 30)))
        print(f"          best lag ≥ 5 min: {best} min;  CSV median song length {np.median(np.diff(L.song_bounds, axis=1)) / 60:.1f} min")
    # local period, 40-min blocks
    z = unit_rows(np.vstack([f["z_train"], f["z_test"]]))
    t = np.r_[f["t_train"], f["t_test"]]
    per = []
    for s0 in range(0, N - 2400, 1200):
        i0, i1 = np.searchsorted(t, s0), np.searchsorted(t, s0 + 2400)
        blk = z[i0:i1]
        lg = np.arange(150, 600)  # 5-20 min in steps
        sims = [np.mean(np.sum(blk[:-g] * blk[g:], 1)) for g in lg if g < len(blk) - 100]
        per.append(lg[int(np.argmax(sims))] * 2 / 60)
    print(f"   local song period in 40-min blocks (min): median {np.median(per):.1f}, IQR {np.percentile(per, 25):.1f}-{np.percentile(per, 75):.1f}")

    print("\n4. oracle: CSV song boundaries given, each song split into 6 ordered segments from the phrase descriptors alone")
    for half, off in (("train", 0), ("test", SPLIT_S)):
        z = f[f"z_{half}"]
        t = f[f"t_{half}"] + 6.0
        csv_w = L.theme[np.clip(t.astype(int), 0, N - 1)]
        hit = tot = 0
        rand_hit = 0
        for a, b in L.song_bounds:
            if a < off or b > off + 18000:
                continue
            i0, i1 = np.searchsorted(t, a), np.searchsorted(t, b)
            if i1 - i0 < 12:
                continue
            lab = segment_ordered(z[i0:i1], 6)
            y = csv_w[i0:i1]
            hit += np.sum(lab == y)
            tot += len(y)
            # baseline: equal-length 6-way split in order
            rand_hit += np.sum(np.minimum((np.arange(len(y)) * 6) // len(y), 5) == y)
        print(f"   {half}: theme accuracy with known boundaries {hit / tot:.3f}   (equal-length ordered split {rand_hit / tot:.3f})")


if __name__ == "__main__":
    main()


def phase_check() -> None:
    """5. Do CSV song starts sit at the same point of the audio's repeating cycle?

    For each pair of consecutive CSV songs, slide the first 3 min after start n over the
    region around start n+1 (+-5 min) and find the offset that matches best.
    If the CSV marks the same place in every song, the best offset is near 0
    and the audio's song length (start_{n+1} + offset - start_n) equals the CSV's.
    """
    L = load_labels(CSV_PATH, N)
    f = np.load(WORK / "phrase_features.npz")
    z = unit_rows(np.vstack([f["z_train"], f["z_test"]]))
    t = np.r_[f["t_train"], f["t_test"]] + 6.0
    starts = L.song_bounds[:, 0]
    w = 90  # windows = 3 min
    offs, csv_len, audio_len, peak = [], [], [], []
    for s0, s1 in zip(starts[:-1], starts[1:], strict=True):
        i0, i1 = np.searchsorted(t, s0), np.searchsorted(t, s1)
        if (s0 < SPLIT_S) != (s1 + 300 < SPLIT_S) or i1 + 150 + w > len(z) or i1 - 150 < 0:
            continue  # stay inside one half
        ref = z[i0 : i0 + w]
        sims = [np.mean(np.sum(ref * z[i1 + d : i1 + d + w], 1)) for d in range(-150, 151)]
        d = int(np.argmax(sims)) - 150
        offs.append(d * 2)
        csv_len.append((s1 - s0) / 60)
        audio_len.append((s1 + 2 * d - s0) / 60)
        peak.append(max(sims))
    offs = np.array(offs)
    print("\n5. phase check: best offset of the next CSV song start (s), searched over +-300 s")
    print(f"   pairs {len(offs)}; |offset| <= 15 s: {np.mean(np.abs(offs) <= 15):.0%};  <= 60 s: {np.mean(np.abs(offs) <= 60):.0%};  median |offset| {np.median(np.abs(offs)):.0f} s")
    print(f"   CSV song length median {np.median(csv_len):.1f} min;  audio-aligned song length median {np.median(audio_len):.1f} min;  "
          f"corr(CSV length, audio length) {np.corrcoef(csv_len, audio_len)[0, 1]:.2f};  mean best similarity {np.mean(peak):.3f}")


if __name__ == "__main__":
    phase_check()
