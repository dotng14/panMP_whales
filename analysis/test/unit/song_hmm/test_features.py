import numpy as np
from scipy.io import wavfile

from analysis.song_hmm.features import FEATURE_NAMES, build_features

SR = 2000
LEVEL = FEATURE_NAMES.index("level")
MOD = FEATURE_NAMES.index("modulation")


def _noise(seconds, seed=0):
    return np.random.default_rng(seed).normal(0.0, 100.0, seconds * SR)


def _write(path, audio):
    wavfile.write(path, SR, np.clip(audio, -32000, 32000).astype(np.int16))


def test_tone_bursts_raise_level_and_modulation(tmp_path):
    seconds = 1200
    audio = _noise(seconds)
    t = np.arange(SR) / SR
    burst = 3000.0 * np.sin(2 * np.pi * 300.0 * t) * (t < 0.5)  # 0.5 s on, 0.5 s off
    song = list(range(300, 320)) + list(range(900, 920))
    for s in song:
        audio[s * SR : (s + 1) * SR] += burst
    path = tmp_path / "burst.wav"
    _write(path, audio)
    feats = build_features(path, split_s=600)
    raw = feats["raw"]
    quiet = np.setdiff1d(np.arange(seconds), song)
    assert raw[song, LEVEL].min() > np.percentile(raw[quiet, LEVEL], 99)
    assert raw[song, MOD].min() > np.percentile(raw[quiet, MOD], 99)


def test_train_stats_ignore_the_test_half(tmp_path):
    seconds = 1200
    audio = _noise(seconds, seed=1)
    a, b = tmp_path / "a.wav", tmp_path / "b.wav"
    _write(a, audio)
    changed = audio.copy()
    changed[700 * SR :] *= 20.0
    _write(b, changed)
    fa, fb = build_features(a, 600), build_features(b, 600)
    np.testing.assert_allclose(fa["mean"], fb["mean"])
    np.testing.assert_allclose(fa["std"], fb["std"])
    np.testing.assert_allclose(fa["x"][:600], fb["x"][:600])
