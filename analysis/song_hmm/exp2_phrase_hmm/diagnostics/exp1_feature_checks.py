"""Why Experiment 1 failed: can per-second spectral features recognise the CSV units and themes at all?

    python analysis/song_hmm/exp2_phrase_hmm/diagnostics/exp1_feature_checks.py

Needs scikit-learn (not a project dependency; diagnostics only). Trains a
logistic-regression classifier on the first 5 h and scores the last 5 h:
  1. unit name (U1..U11) per second, with the labels shifted by -20..+20 s
     (a misaligned CSV would peak at a non-zero lag)
  2. theme from 1 s / 10 s / 30 s windows, for several feature sets
Results quoted in reports/exp2_phrase_hmm.md (Experiment 1, "Why it fell short").
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from analysis.song_hmm.exp1_slot_hmm.labels import load_labels  # noqa: E402
from analysis.song_hmm.exp1_slot_hmm.topology import SLOT_UNITS  # noqa: E402
from analysis.song_hmm.exp1_slot_hmm.train import CSV_PATH, WAV_PATH  # noqa: E402

N, SPLIT = 36000, 18000


def spectrogram_bands(n_bands: int) -> np.ndarray:
    """Frame log-power (0.1 s frames) in log bands 30-1000 Hz, minus a per-half 10th-percentile floor."""
    sr, audio = wavfile.read(WAV_PATH, mmap=True)
    hop, nfft = 200, 512
    freqs = np.fft.rfftfreq(nfft, 1 / sr)
    edges = np.geomspace(30, 1000, n_bands + 1)
    idx = np.searchsorted(edges, freqs, side="right") - 1
    ok = (idx >= 0) & (idx < n_bands)
    m = np.zeros((freqs.size, n_bands), np.float32)
    m[np.flatnonzero(ok), idx[ok]] = 1
    m = m[:, m.sum(0) > 0]
    m /= m.sum(0)
    out = np.empty((N * 10, m.shape[1]), np.float32)
    win = np.hanning(nfft).astype(np.float32)
    for f0 in range(0, N * 10, 18000):
        f1 = min(N * 10, f0 + 18000)
        need = (f1 - f0 - 1) * hop + nfft
        seg = np.asarray(audio[f0 * hop : f0 * hop + need], np.float32)
        seg = np.r_[seg, np.zeros(need - seg.size, np.float32)]
        fr = np.lib.stride_tricks.sliding_window_view(seg, nfft)[::hop][: f1 - f0]
        out[f0:f1] = np.log(np.abs(np.fft.rfft(fr * win, axis=1)) ** 2 @ m + 1e-6)
    for sl in (slice(0, SPLIT * 10), slice(SPLIT * 10, None)):
        out[sl] -= np.percentile(out[sl][::10], 10, axis=0)
    return out


def score(x: np.ndarray, y: np.ndarray, w: int) -> tuple[float, float]:
    k = N // w
    xw = x[: k * w].reshape(k, w, -1).mean(1)
    yw = np.array([np.bincount(y[i * w : (i + 1) * w] + 1, minlength=y.max() + 2).argmax() - 1 for i in range(k)])
    tr = np.arange(k) < SPLIT // w
    xs = StandardScaler().fit(xw[tr]).transform(xw)
    if xs.shape[1] > 60:
        xs = PCA(60, random_state=0).fit(xs[tr]).transform(xs)
    clf = LogisticRegression(max_iter=2000, C=0.1, class_weight="balanced").fit(xs[tr], yw[tr])
    p = clf.predict(xs[~tr])
    return float(np.mean(p == yw[~tr])), float(balanced_accuracy_score(yw[~tr], p))


def main() -> None:
    labels = load_labels(CSV_PATH, N)
    b16, b48 = spectrogram_bands(16), spectrogram_bands(48)
    per_s = {
        "16 bands, 1 s mean (Experiment 1)": b16.reshape(N, 10, -1).mean(1),
        "48 bands, 1 s mean + std": np.hstack([b48.reshape(N, 10, -1).mean(1), b48.reshape(N, 10, -1).std(1)]),
        "16 bands x 10 frames (contour)": b16.reshape(N, -1),
    }
    print("theme classification on the test half: accuracy / balanced accuracy (chance balanced 0.17)")
    for name, x in per_s.items():
        print(f"  {name:36s}", "  ".join(f"{w:2d} s: {a:.2f}/{b:.2f}" for w in (1, 10, 30) for a, b in [score(x, labels.theme, w)]))
    units = np.array([int(SLOT_UNITS[s][1:]) if s >= 0 else -1 for s in labels.slot])
    x = per_s["48 bands, 1 s mean + std"]
    print("unit-name classification per second, labels shifted by lag (balanced accuracy; chance 0.125)")
    for lag in (-20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20):
        print(f"  lag {lag:+3d} s: {score(x, np.roll(units, lag), 1)[1]:.3f}")


if __name__ == "__main__":
    main()
