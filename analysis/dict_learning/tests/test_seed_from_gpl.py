"""Tests for energy-ranked seed_from_GPL and MP helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.dict_learning.code.io import Detection, global_samples_to_time_bins  # noqa: E402
from analysis.dict_learning.code.mp import mp_step, residual_energy, run_mp_loop  # noqa: E402
from analysis.dict_learning.code.core import Dictionary, Segment, Template, Window  # noqa: E402
from analysis.dict_learning.code.mp import score_shifts  # noqa: E402
from analysis.dict_learning.code.learn.seed_from_gpl import (  # noqa: E402
    Seed,
    detections_to_segments,
    seed_from_gpl,
)

# Tiny positive epsilon: peel until no improving step (eps must be > 0).
EPS_PEEL = 1e-12


def test_global_samples_to_time_bins_basic():
    assert global_samples_to_time_bins(
        1, 20, win_start_sample=1, overlap=10, n_cols=100
    ) == (0, 2)
    assert (
        global_samples_to_time_bins(1, 5, win_start_sample=100, overlap=10, n_cols=50)
        is None
    )


def test_score_and_mp_step():
    f, t, w = 4, 30, 5
    tmpl = np.ones((f, w), dtype=np.float64)
    residual = np.zeros((f, t), dtype=np.float64)
    residual[:, 7 : 7 + w] = 2.5 * tmpl
    scores = score_shifts(tmpl, residual)
    assert int(np.argmax(np.abs(scores))) == 7
    d = Dictionary()
    d.add(Template(template_id=0, data=tmpl))
    stack = {"w0": residual.copy()}
    e0 = residual_energy(stack)
    _new_stack, step = mp_step(stack, d)
    assert step.matching is not None
    assert step.matching.t_start == 7
    assert abs(step.matching.amplitude - 2.5) < 1e-9
    assert step.energy_after < e0


def _make_window(data: np.ndarray, wid: str = "w0", overlap: int = 10) -> Window:
    cols = data.shape[1]
    return Window(
        window_id=wid,
        data=data,
        start_sample=1,
        end_sample=1 + cols * overlap - 1,
        overlap=overlap,
    )


def test_seed_picks_min_residual_among_all_candidates():
    """Trials every candidate; selects the unit that leaves the lowest residual."""
    f, t, w = 4, 60, 5
    pattern = np.ones((f, w), dtype=np.float64) * 2.0
    residual = np.zeros((f, t), dtype=np.float64)
    for t0 in (5, 20, 35):
        residual[:, t0 : t0 + w] += pattern

    spike = np.ones((f, 3), dtype=np.float64) * 3.0
    residual[:, 50:53] += spike

    window = _make_window(residual)
    true_seg = Segment(
        segment_id="true",
        data=pattern.copy(),
        window_id="w0",
        t_start=5,
        t_end=5 + w,
        global_start_sample=1,
        global_end_sample=2,
        detection_index=0,
    )
    spike_seg = Segment(
        segment_id="spike",
        data=spike.copy(),
        window_id="w0",
        t_start=50,
        t_end=53,
        global_start_sample=1,
        global_end_sample=2,
        detection_index=1,
    )
    assert spike_seg.energy > true_seg.energy
    assert 3 * true_seg.energy > spike_seg.energy

    result = seed_from_gpl(
        [window],
        [true_seg, spike_seg],
        eps_step_fraction=EPS_PEEL,
        min_repeats=1,
    )
    assert isinstance(result, Seed)
    assert result.is_selected()
    assert result.get_selected_segment_id() == "true"
    assert result.get_n_candidates_tried() == 2
    assert result.get_energy_after() < result.get_energy_before()
    deltas = result.get_mp_delta_energies()
    assert len(deltas) >= 1
    assert all(d >= 0 for d in deltas)
    assert abs(sum(deltas) - (result.get_mp_energies()[0] - result.get_mp_energies()[-1])) < 1e-6


def test_seed_min_repeats_rejects_one_shot():
    """min_repeats filters out candidates that do not repeat enough."""
    f, t, w = 4, 50, 5
    pattern = np.ones((f, w), dtype=np.float64) * 2.0
    residual = np.zeros((f, t), dtype=np.float64)
    for t0 in (5, 20, 35):
        residual[:, t0 : t0 + w] += pattern
    spike = np.ones((f, 3), dtype=np.float64) * 3.0
    residual[:, 45:48] += spike
    window = _make_window(residual)
    pattern_seg = Segment(
        segment_id="pattern",
        data=pattern.copy(),
        window_id="w0",
        t_start=5,
        t_end=5 + w,
        global_start_sample=1,
        global_end_sample=2,
        detection_index=0,
    )
    spike_seg = Segment(
        segment_id="spike",
        data=spike.copy(),
        window_id="w0",
        t_start=45,
        t_end=48,
        global_start_sample=1,
        global_end_sample=2,
        detection_index=1,
    )
    result = seed_from_gpl(
        [window],
        [pattern_seg, spike_seg],
        eps_step_fraction=EPS_PEEL,
        min_repeats=3,
    )
    assert result.is_selected()
    assert result.get_selected_segment_id() == "pattern"
    assert len(result.get_matchings()) >= 3


def test_seed_empty_candidates():
    result = seed_from_gpl(
        [_make_window(np.zeros((2, 10)))],
        [],
        eps_step_fraction=0.2,
    )
    assert not result.is_selected()
    assert result.get_n_candidates_tried() == 0


def test_detections_to_segments_from_csv_times():
    data = np.zeros((2, 20), dtype=np.float64)
    data[:, 5:8] = 1.0
    window = _make_window(data, overlap=10)
    dets = [Detection(start_time=51, end_time=80, index=0)]
    segs = detections_to_segments([window], dets)
    assert len(segs) == 1
    assert segs[0].t_start == 5 and segs[0].t_end == 8


def test_mp_loop_stops_on_eps_step_fraction():
    """After a strong first step, a weak second step is rejected by the fraction rule."""
    f, t, w = 3, 40, 4
    tmpl = np.ones((f, w), dtype=np.float64)
    residual = np.zeros((f, t), dtype=np.float64)
    residual[:, 5 : 5 + w] = 10.0 * tmpl
    residual[:, 20 : 20 + w] = 1.0 * tmpl
    d = Dictionary()
    d.add(Template(template_id=0, data=tmpl))
    loop = run_mp_loop({"w0": residual}, d, eps_step_fraction=0.5)
    assert len(loop.matchings) == 1
    assert loop.stopped_reason == "eps_step_fraction"
    deltas = [loop.energies[i] - loop.energies[i + 1] for i in range(len(loop.energies) - 1)]
    assert len(deltas) == 1
    loop2 = run_mp_loop({"w0": residual}, d, eps_step_fraction=EPS_PEEL)
    assert len(loop2.matchings) >= 2


def test_mp_loop_rejects_nonpositive_eps():
    d = Dictionary()
    d.add(Template(template_id=0, data=np.ones((2, 3))))
    try:
        run_mp_loop({"w0": np.zeros((2, 10))}, d, eps_step_fraction=0.0)
    except ValueError as exc:
        assert "eps_step_fraction" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-positive eps")


def test_mp_loop_disallows_overlapping_matchings():
    tmpl = np.array([[1.0, -1.0, 2.0, -2.0]])
    residual = np.zeros((1, 40), dtype=np.float64)
    residual[:, 5:9] += 3.0 * tmpl
    residual[:, 7:11] += 2.0 * tmpl
    residual[:, 25:29] += 2.5 * tmpl
    dictionary = Dictionary()
    dictionary.add(Template(template_id=0, data=tmpl))

    loop = run_mp_loop(
        {"w0": residual},
        dictionary,
        eps_step_fraction=EPS_PEEL,
        disallow_overlap=True,
    )

    intervals = sorted((m.t_start, m.t_start + tmpl.shape[1]) for m in loop.matchings)
    assert len(intervals) >= 2
    assert all(
        right_start >= left_end
        for (_, left_end), (right_start, _) in zip(intervals, intervals[1:])
    )


if __name__ == "__main__":
    test_global_samples_to_time_bins_basic()
    test_score_and_mp_step()
    test_seed_picks_min_residual_among_all_candidates()
    test_seed_min_repeats_rejects_one_shot()
    test_seed_empty_candidates()
    test_detections_to_segments_from_csv_times()
    test_mp_loop_stops_on_eps_step_fraction()
    test_mp_loop_rejects_nonpositive_eps()
    test_mp_loop_disallows_overlapping_matchings()
    print("test_seed_from_gpl: OK")
