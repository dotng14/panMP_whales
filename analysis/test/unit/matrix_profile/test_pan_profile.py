"""pan_profile, length_profile, pan_top_pairs, and the signal builders."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.matrix_profile import (
    length_profile,
    mpx_channels,
    mpx_fast,
    mpxForPan,
    pan_lengths,
    pan_minlag,
    pan_profile,
    pan_top_pairs,
    panMP,
)
from analysis.matrix_profile.signals import (
    _band_rows,
    default_band_edges,
    load_gpl,
    load_series,
)
from analysis.test.synthetic import planted_series, write_gpl_folder


def test_pan_lengths_and_minlag():
    assert pan_lengths(8, 20, 4) == [8, 12, 16, 20]
    assert pan_lengths(8, 8, 4) == [8]
    assert pan_minlag(4) == 4
    assert pan_minlag(10) == 8


@pytest.mark.parametrize("args", [(3, 8, 1), (8, 4, 1), (8, 16, 0)])
def test_pan_lengths_rejects_bad_sweeps(args):
    with pytest.raises(ValueError):
        pan_lengths(*args)


def test_length_profile_engines_agree():
    series = planted_series(n=300, starts=(40, 200))
    fast = length_profile(series, 16, engine="fast")
    ref = length_profile(series, 16, engine="reference")
    np.testing.assert_allclose(fast[0], ref[0], atol=1e-8)
    np.testing.assert_allclose(fast[0], mpx_fast(series, pan_minlag(16), 16)[0])
    np.testing.assert_allclose(ref[0], mpxForPan(series, pan_minlag(16), 16)[0])


def test_length_profile_rows():
    rows = planted_series(n=300, rows=2, starts=(40, 200))
    distance, _ = length_profile(rows, 16)
    np.testing.assert_allclose(distance, mpx_channels(rows, pan_minlag(16), 16)[0])
    with pytest.raises(ValueError, match="single row"):
        length_profile(rows, 16, engine="reference")
    with pytest.raises(ValueError, match="engine"):
        length_profile(rows[0], 16, engine="gpu")


def test_one_row_of_a_2d_signal_is_the_1d_signal():
    series = planted_series(n=300, starts=(40, 200))
    a, _ = length_profile(series, 16)
    b, _ = length_profile(series[np.newaxis, :], 16)
    np.testing.assert_array_equal(a, b)


def test_pan_profile_surface():
    series = planted_series(n=400, starts=(40, 220))
    profile = pan_profile(series, [16, 24])
    assert profile.lengths == [16, 24]
    assert profile.surface.shape == (2, 400)
    assert np.nanmax(profile.surface) == pytest.approx(1.0)
    assert np.nanmin(profile.surface) == pytest.approx(0.0)
    # Positions past the last window of a length are NaN.
    assert np.isnan(profile.surface[1, 400 - 24 + 1 :]).all()
    assert np.isfinite(profile.surface[1, : 400 - 24 + 1]).all()
    # The planted repeat is the strongest motif.
    peak = int(np.nanargmax(profile.surface[0]))
    assert min(abs(peak - 40), abs(peak - 220)) <= 4


def test_pan_profile_rejects_short_and_empty():
    with pytest.raises(ValueError, match="too short"):
        pan_profile(np.zeros(29), [16])
    with pytest.raises(ValueError, match="at least one"):
        pan_profile(np.zeros(30), [])
    with pytest.raises(ValueError, match="1-D or"):
        pan_profile(np.zeros((2, 2, 50)), [8])


def test_pan_top_pairs_finds_planted_copies():
    series = planted_series(n=600, length=24, starts=(40, 220, 420))
    pairs = pan_top_pairs(pan_profile(series, [24]), k=2)
    assert len(pairs) == 2
    planted = (40, 220, 420)
    for pair in pairs:
        assert pair["length"] == 24
        assert pair["start_a"] < pair["start_b"]
        assert any(abs(pair["start_a"] - p) <= 4 for p in planted)
        assert any(abs(pair["start_b"] - p) <= 4 for p in planted)
    # A later pair starts outside the copies already chosen; its neighbour
    # may repeat one of them (top_pairs_by_zscore blocks only the start).
    first = {pairs[0]["start_a"], pairs[0]["start_b"]}
    assert len({pairs[1]["start_a"], pairs[1]["start_b"]} - first) >= 1


def test_panMP_fast_engine_matches_reference():
    series = planted_series(n=300, starts=(40, 200))
    kwargs = dict(plot=False, MinMotifLen=8, MaxMotifLen=24, StepSize=8, verbose=False)
    np.testing.assert_allclose(
        panMP(series, engine="fast", **kwargs), panMP(series, engine="reference", **kwargs),
        atol=1e-8, equal_nan=True,
    )


def test_panMP_rejects_a_matrix():
    with pytest.raises(ValueError, match="one series"):
        panMP(np.zeros((2, 600)), plot=False)


def test_panMP_saves_plot(tmp_path):
    path = tmp_path / "pan.png"
    panMP(planted_series(n=300, starts=(40, 200)), plot=True, MinMotifLen=8, MaxMotifLen=16, StepSize=8,
          engine="fast", plot_path=path, verbose=False)
    assert path.stat().st_size > 0


def test_default_band_edges():
    assert default_band_edges(10.0, 230.0) == [10.0, 110.0, 210.0, 230.0]
    assert default_band_edges(0.0, 200.0) == [0.0, 100.0, 200.0]


def test_band_rows_sum_rows_within_edges():
    spec = np.arange(12, dtype=float).reshape(4, 3)
    freqs = np.array([0.0, 50.0, 100.0, 150.0])
    out = _band_rows(spec, freqs, [0.0, 100.0, 150.0])
    # [0, 100) holds rows 0-1; the last band [100, 150] is closed and holds rows 2-3.
    np.testing.assert_array_equal(out, [spec[0] + spec[1], spec[2] + spec[3]])


def test_load_series_shapes(tmp_path):
    np.save(tmp_path / "a.npy", np.arange(5.0))
    assert load_series(tmp_path / "a.npy").data.shape == (1, 5)
    np.save(tmp_path / "b.npy", np.zeros((3, 5)))
    assert load_series(tmp_path / "b.npy").data.shape == (3, 5)


def test_load_gpl_energy_is_the_sum_of_full_rows(tmp_path):
    folder = write_gpl_folder(tmp_path / "gpl", n_windows=2)
    full = load_gpl(folder.root, "full").data
    energy = load_gpl(folder.root, "energy").data
    np.testing.assert_allclose(energy[0], full.sum(axis=0))
    bands = load_gpl(folder.root, "bands", band_edges=[10.0, 60.0, 130.0]).data
    np.testing.assert_allclose(bands.sum(axis=0), full.sum(axis=0))


def test_load_gpl_rate_averages_frames(tmp_path):
    folder = write_gpl_folder(tmp_path / "gpl", n_windows=1)
    native = load_gpl(folder.root, "energy").data[0]
    binned = load_gpl(folder.root, "energy", rate=10.0).data[0]
    # Frame i is centred at (i + 1) / 100 s, so the 0.5-0.6 s bin averages frames 49-58.
    assert binned.shape == (24,)
    assert binned[5] == pytest.approx(native[49:59].mean(), rel=1e-9)
