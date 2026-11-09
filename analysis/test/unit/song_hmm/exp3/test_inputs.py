import numpy as np
from scipy.signal import chirp

from analysis.song_hmm.exp3.inputs.pcen import pcen
from analysis.song_hmm.exp3.inputs.spectrogram import SR, normalise, spectrogram
from analysis.song_hmm.exp3.inputs.units import detect
from analysis.song_hmm.exp3.inputs.window_features import x2


def test_pcen_matches_librosa_after_warmup():
    librosa = __import__("pytest").importorskip("librosa")
    e = np.abs(np.random.default_rng(0).normal(1, 0.5, (4000, 8))) * 5
    ours = pcen(e, 40)
    ref = librosa.pcen(e.T, sr=40, hop_length=1, time_constant=2.0, gain=0.98, bias=2.0, power=0.5, eps=1e-6).T
    np.testing.assert_allclose(ours[1000:], ref[1000:], atol=1e-4)  # librosa starts its smoother at 1, we at frame 0


def test_x2_ignores_a_time_shift():
    rng = np.random.default_rng(1)
    phrase = rng.random((80, 30))  # 4 s phrase at 20 fps
    sw = np.tile(phrase, (20, 1)).astype(np.float32)  # repeating phrase
    a = x2(sw, 20, 12.0).raw
    # windows starting half a phrase apart see the same repeating pattern, shifted
    np.testing.assert_allclose(a[0], a[1], rtol=0.05, atol=0.05)


def _audio_with_events(seconds=60):
    x = np.random.default_rng(2).normal(0, 20, seconds * SR).astype(np.float32)
    t = np.arange(int(1.0 * SR)) / SR
    up = chirp(t, 200, 1.0, 400, method="logarithmic") * 2000  # rises one octave in 1 s
    x[10 * SR : 11 * SR] += up
    tp = np.arange(int(1.0 * SR)) / SR
    pulses = np.sin(2 * np.pi * 600 * tp) * ((tp * 8) % 1 < 0.4) * 2000  # 8 Hz pulse train at 600 Hz
    x[30 * SR : 31 * SR] += pulses
    return x


def test_units_found_with_their_shape():
    x = _audio_with_events()
    s = spectrogram(x, "S2", 60)
    v = normalise(s, "N2")
    u = detect(v, s.fps, s.edges, t0=s.centre_s)
    rise = u[(u["start_s"] > 9.5) & (u["start_s"] < 10.6)]
    assert len(rise) == 1
    r = rise[0]
    assert 0.8 < r["duration_s"] < 1.3
    assert 8 < r["slope_st_per_s"] < 16  # one octave = 12 semitones per second
    assert 180 < r["start_hz"] < 260 and 330 < r["end_hz"] < 450
    pul = u[(u["start_s"] > 29.5) & (u["start_s"] < 30.6)]
    assert len(pul) == 1  # pulses close into one unit
    assert 500 < pul[0]["peak_hz"] < 700
    assert 6 < pul[0]["pulse_hz"] < 10
