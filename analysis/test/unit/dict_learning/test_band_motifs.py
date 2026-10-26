"""5-band profiles, summed distances, and dictionary seeds from motif pairs."""

from __future__ import annotations


import numpy as np
import pytest


from analysis.bands import BAND_WIDTH_HZ, FREQ_MAX_HZ, N_BANDS, band_edges, band_energy  # noqa: E402
from analysis.dict_learning import seeds_from_motif_pairs  # noqa: E402
from analysis.matrix_profile import mpx_channels, mpx_fast  # noqa: E402
from analysis.matrix_profile import closest_pairs  # noqa: E402


def test_band_edges_cover_0_to_500_hz():
    edges = band_edges()
    assert len(edges) == N_BANDS == 5
    assert edges[0] == (0, BAND_WIDTH_HZ)
    assert edges[-1] == (FREQ_MAX_HZ - BAND_WIDTH_HZ, FREQ_MAX_HZ)
    assert all(hi - lo == BAND_WIDTH_HZ for lo, hi in edges)


def test_tone_lands_in_its_100_hz_band():
    sample_rate = 8000
    frame_rate = 8
    time = np.arange(sample_rate) / sample_rate
    tone = np.sin(2 * np.pi * 250 * time)
    energy = band_energy(tone, sample_rate, frame_rate)
    assert energy.shape == (N_BANDS, frame_rate)
    assert int(np.argmax(energy.mean(axis=1))) == 2  # 200–300 Hz
    above = np.sin(2 * np.pi * 750 * time)
    assert band_energy(above, sample_rate, frame_rate).sum() < 0.05 * energy.sum()


def _zdist(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean()
    y = y - y.mean()
    nx = np.linalg.norm(x)
    ny = np.linalg.norm(y)
    if nx == 0.0 and ny == 0.0:
        return 0.0
    if nx == 0.0 or ny == 0.0:
        return float(np.sqrt(2.0 * len(x)))
    corr = float(np.dot(x, y) / (nx * ny))
    corr = min(1.0, max(-1.0, corr))
    return float(np.sqrt(2.0 * len(x) * (1.0 - corr)))


def test_channel_distance_is_the_sum_of_band_distances():
    rng = np.random.default_rng(0)
    window = 16
    series = rng.standard_normal((3, 80))
    distance, neighbor = mpx_channels(series, window, window, nthreads=2)
    nprof = series.shape[1] - window + 1
    for i in (0, 7, nprof // 2, nprof - 1):
        j = int(neighbor[i])
        expected = sum(
            _zdist(series[b, i : i + window], series[b, j : j + window])
            for b in range(series.shape[0])
        )
        assert distance[i] == pytest.approx(expected, rel=1e-6, abs=1e-6)
        assert abs(j - i) >= window


def test_one_band_matches_the_scalar_profile():
    rng = np.random.default_rng(1)
    series = rng.standard_normal(120)
    scalar, scalar_idx = mpx_fast(series, 16, 16, nthreads=1)
    stacked, stacked_idx = mpx_channels(series.reshape(1, -1), 16, 16, nthreads=1)
    np.testing.assert_allclose(stacked, scalar, rtol=1e-6, atol=1e-6)
    assert np.array_equal(stacked_idx, scalar_idx)


def test_motif_pair_becomes_a_band_by_width_seed():
    width = 4
    profile = np.zeros((N_BANDS, 30))
    profile[:, 2:6] = 1.0
    profile[:, 10:14] = 3.0
    distance = np.full(profile.shape[1], np.nan)
    neighbor = np.zeros(profile.shape[1], dtype=int)
    distance[2] = 0.2
    neighbor[2] = 10
    pairs = closest_pairs(distance, neighbor, width, k=1)
    seeds = seeds_from_motif_pairs(profile, pairs, width, sample_rate=2000, frame_rate=100)
    atom = seeds.dictionary.templates[0]
    assert atom.data.shape == (N_BANDS, width)
    np.testing.assert_allclose(atom.data, 2.0)
    assert [(m.t_start, m.window_id) for m in seeds.matchings] == [
        (2, "band_profile"),
        (10, "band_profile"),
    ]
    assert seeds.window.data.shape == profile.shape
    assert seeds.window.end_freq == FREQ_MAX_HZ
