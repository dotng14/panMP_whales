import numpy as np
from scipy.io import wavfile
from scipy.signal import chirp

from analysis.song_hmm_phrase.decode import Decoded
from analysis.song_hmm_phrase.features import STATS, band_spectrogram, build, descriptors, whiten
from analysis.song_hmm_phrase.metrics import boundary_novelty, novelty_curve, novelty_peaks, theme_coherence, unit_rows

SR = 2000


def _noise(seconds, seed=0):
    return np.random.default_rng(seed).normal(0, 50, seconds * SR)


def _write(path, a):
    wavfile.write(path, SR, np.clip(a, -32000, 32000).astype(np.int16))


def test_shapes_and_split(tmp_path):
    a = _noise(600)
    path = tmp_path / "a.wav"
    _write(path, a)
    f = build(path, 300)
    nb = f["band_edges"].shape[0]
    n_win = (300 - 12) // 2 + 1
    assert f["z_train"].shape == (n_win, len(STATS) * nb)
    assert f["X_train"].shape == (n_win, 20) and f["X_test"].shape == (n_win, 20)
    assert f["t_train"][-1] + 12 <= 300 and f["t_test"][0] >= 300  # no window crosses the split


def test_train_stats_ignore_test_half(tmp_path):
    a = _noise(600, 1)
    b = a.copy()
    b[400 * SR :] *= 30
    pa, pb = tmp_path / "a.wav", tmp_path / "b.wav"
    _write(pa, a)
    _write(pb, b)
    fa, fb = build(pa, 300), build(pb, 300)
    for k in ("mu", "sigma", "basis", "X_train"):
        np.testing.assert_allclose(fa[k], fb[k])


def test_upsweep_and_downsweep():
    t = np.arange(int(0.5 * SR)) / SR
    up = chirp(t, 200, 0.5, 600, method="logarithmic")
    down = chirp(t, 600, 0.5, 200, method="logarithmic")
    a = _noise(240, 2)
    for s in range(0, 120):
        a[s * SR : s * SR + up.size] += 3000 * up
    for s in range(120, 240):
        a[s * SR : s * SR + down.size] += 3000 * down
    fps = 20
    s = whiten(band_spectrogram(a, SR, 240 * fps), fps)
    starts, v = descriptors(s, fps)
    nb = s.shape[1]
    up_stat = v[:, 3 * nb : 4 * nb].sum(1)
    down_stat = v[:, 4 * nb : 5 * nb].sum(1)
    first, second = starts / fps + 12 <= 120, starts / fps >= 120
    assert np.all(up_stat[first] > down_stat[first])
    assert np.all(down_stat[second] > up_stat[second])


def _blocks(sizes, d=30, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.standard_normal((len(sizes), d)) * 3
    z = np.vstack([c + rng.standard_normal((n, d)) for c, n in zip(centres, sizes, strict=True)])
    lab = np.concatenate([np.full(n, i % 6) for i, n in enumerate(sizes)])
    return unit_rows(z), lab


def test_coherence_and_novelty_on_planted_blocks():
    zn, lab = _blocks([80, 80, 80, 80])
    good = theme_coherence(zn, lab)["coherence"]
    shuffled = theme_coherence(zn, np.roll(lab, 40))["coherence"]
    assert good > 0.5 and good > shuffled + 0.3
    peaks = novelty_peaks(novelty_curve(zn))
    res = boundary_novelty(lab, peaks)
    assert res["recall"] == 1.0 and res["precision"] == 1.0


def test_decoded_songs_and_boundaries():
    theme = np.array([3] * 10 + [4] * 10 + [5] * 10 + [0] * 10 + [1] * 10 + [2] * 10 + [3] * 10 + [4] * 10 + [5] * 10 + [-1] * 10)
    dec = Decoded("x", theme, theme >= 0, np.array([30]))
    assert [(s.start_s, s.end_s, s.complete) for s in dec.songs] == [(0, 30, False), (30, 90, True)]
    assert dec.sessions == [(0, 90)]
