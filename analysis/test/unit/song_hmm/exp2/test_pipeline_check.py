import numpy as np

from analysis.song_hmm_phrase import features as F
from analysis.song_hmm_phrase.pipeline_check.check import (
    SR,
    istft_snr,
    modulation_spectrum,
    patch_similarity,
    tone_burst_frame,
)


def test_stft_framing_is_invertible():
    x = np.random.default_rng(0).normal(0, 1, 60 * SR)
    assert istft_snr(x) > 40.0


def test_band_pooling_keeps_all_power_in_range():
    m, _ = F.band_matrix(SR)
    freqs = np.fft.rfftfreq(F.NFFT, 1 / SR)
    inside = (freqs >= F.F_LO) & (freqs < F.F_HI)
    assert np.all(m[inside].sum(1) == 1) and np.all(m[~inside].sum(1) == 0)
    p = np.random.default_rng(1).random((10, freqs.size))
    np.testing.assert_allclose((p @ m).sum(1), p[:, inside].sum(1), rtol=1e-5)


def test_tone_burst_lands_in_its_frame():
    got, want = tone_burst_frame()
    assert abs(got - want) <= 1


def test_modulation_spectrum_finds_a_10hz_pulse_train():
    t = np.arange(60 * SR) / SR
    x = np.sin(2 * np.pi * 400 * t) * ((t * 10) % 1 < 0.3) + np.random.default_rng(2).normal(0, 0.05, t.size)
    f, p = modulation_spectrum(x)
    band = (f > 2) & (f < 30)
    assert abs(f[band][np.argmax(p[band])] - 10.0) < 0.3


def test_shift_tolerant_patch_similarity_finds_a_repeat():
    rng = np.random.default_rng(3)
    w = 240
    sw = np.abs(rng.normal(0, 1, (6000, 20))).astype(np.float32)
    motif = np.abs(rng.normal(0, 3, (w, 20))).astype(np.float32)
    sw[1000 : 1000 + w] += motif
    sw[4030 : 4030 + w] += motif  # same phrase 30 frames (1.5 s) off the window grid
    cands = np.arange(0, 6000 - w, 40)
    sim = patch_similarity(sw, np.array([1000]), cands, w)
    far = np.abs(cands - 1000) > 400
    best = cands[far][np.argmax(sim[0, far])]
    assert abs(best - 4030) <= 40

