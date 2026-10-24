"""Tests for low-coverage seed-window selection."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.dict_learning.code.learn.learn_dictionary import (  # noqa: E402
    learn_averaged_templates_from_gpl,
)
from analysis.dict_learning.code.core import Segment, Window  # noqa: E402
from analysis.dict_learning.code.core import (  # noqa: E402
    filter_candidates_by_windows,
    select_low_coverage_seed_windows,
    window_uncovered_fraction,
)


def _window(wid: str, data: np.ndarray) -> Window:
    cols = data.shape[1]
    return Window(
        window_id=wid,
        data=data,
        start_sample=1,
        end_sample=1 + cols * 10 - 1,
        overlap=10,
    )


def _seg(seg_id: str, wid: str, data: np.ndarray, t0: int, det: int) -> Segment:
    return Segment(
        segment_id=seg_id,
        data=data.copy(),
        window_id=wid,
        t_start=t0,
        t_end=t0 + data.shape[1],
        global_start_sample=1,
        global_end_sample=2,
        detection_index=det,
    )


def test_window_uncovered_fraction():
    residual = {"w0": np.zeros((2, 100)), "w1": np.zeros((2, 50))}
    covered = {"w0": [(0, 40), (30, 60)]}  # union length 60
    assert abs(window_uncovered_fraction("w0", residual, covered) - 0.4) < 1e-9
    assert abs(window_uncovered_fraction("w1", residual, covered) - 1.0) < 1e-9


def test_select_low_coverage_prefers_uncovered_window():
    residual = {
        "w0": np.zeros((2, 100)),
        "w1": np.zeros((2, 100)),
        "w2": np.zeros((2, 100)),
    }
    patch = np.ones((2, 5), dtype=np.float64)
    candidates = [
        _seg("a", "w0", patch, 0, 0),
        _seg("b", "w1", patch * 2, 10, 1),  # higher energy
        _seg("c", "w2", patch, 20, 2),
    ]
    covered = {"w0": [(0, 80)], "w1": [(0, 80)]}  # w2 fully uncovered
    anchor, seed_ids = select_low_coverage_seed_windows(
        residual, candidates, covered, neighborhood=0
    )
    assert anchor == "w2"
    assert seed_ids == {"w2"}


def test_select_low_coverage_neighborhood():
    residual = {f"w{i}": np.zeros((2, 40)) for i in range(5)}
    patch = np.ones((2, 4), dtype=np.float64)
    candidates = [_seg(f"s{i}", f"w{i}", patch, 0, i) for i in range(5)]
    covered = {f"w{i}": [(0, 30)] for i in range(5) if i != 2}
    anchor, seed_ids = select_low_coverage_seed_windows(
        residual,
        candidates,
        covered,
        window_order=[f"w{i}" for i in range(5)],
        neighborhood=1,
    )
    assert anchor == "w2"
    assert seed_ids == {"w1", "w2", "w3"}


def test_select_tie_break_by_energy():
    residual = {"w0": np.zeros((2, 40)), "w1": np.zeros((2, 40))}
    weak = np.ones((2, 3), dtype=np.float64)
    strong = np.ones((2, 3), dtype=np.float64) * 5
    candidates = [
        _seg("weak", "w0", weak, 0, 0),
        _seg("strong", "w1", strong, 0, 1),
    ]
    # Both uncovered equally → prefer higher energy (w1)
    anchor, seed_ids = select_low_coverage_seed_windows(
        residual, candidates, covered_intervals={}, neighborhood=0
    )
    assert anchor == "w1"
    assert seed_ids == {"w1"}


def test_filter_candidates_by_windows():
    patch = np.ones((2, 3))
    cands = [
        _seg("a", "w0", patch, 0, 0),
        _seg("b", "w1", patch, 0, 1),
        _seg("c", "w0", patch, 5, 2),
    ]
    out = filter_candidates_by_windows(cands, {"w0"})
    assert [s.segment_id for s in out] == ["a", "c"]


def test_learn_low_coverage_seeds_from_uncovered_window():
    """With w0 already covered, low_coverage should seed from w1."""
    f, t = 4, 80
    pattern = np.zeros((f, 5), dtype=np.float64)
    pattern[0, :] = 3.0
    pattern[1, :] = -3.0

    def _filled(starts: tuple[int, ...]) -> np.ndarray:
        data = np.zeros((f, t), dtype=np.float64)
        for t0 in starts:
            data[:, t0 : t0 + 5] += pattern
        return data

    # Same motif in both windows; enough repeats for min_repeats=2.
    w0 = _window("w0", _filled((5, 25, 45)))
    w1 = _window("w1", _filled((8, 28, 48)))
    seg_w0 = _seg("seed0", "w0", pattern, 5, 0)
    seg_w1 = _seg("seed1", "w1", pattern, 8, 1)

    # Pretend w0 is already largely covered so low_coverage picks w1.
    occupied = {"w0": [(0, 70)]}
    result = learn_averaged_templates_from_gpl(
        [w0, w1],
        [seg_w0, seg_w1],
        n_rounds=1,
        eps_step_fraction=1e-12,
        min_repeats=2,
        seed_window_policy="low_coverage",
        seed_window_neighborhood=0,
        occupied_intervals=occupied,
        disallow_overlap=True,
        show_progress=False,
        n_workers=1,
    )
    assert len(result.get_templates()) == 1
    logs = result.get_round_logs()
    assert len(logs) == 1
    assert logs[0].seed_window_policy == "low_coverage"
    assert logs[0].seed_window_id == "w1"
    assert logs[0].n_candidates == 1
    assert logs[0].n_candidates_global == 2
    # Matches should still be found on the full stack (both windows possible).
    assert logs[0].averaged_n_matchings >= 2


def test_learn_global_policy_unchanged_default():
    f, t = 4, 60
    pattern = np.zeros((f, 4), dtype=np.float64)
    pattern[0, :] = 2.0
    pattern[1, :] = -2.0
    data = np.zeros((f, t), dtype=np.float64)
    for t0 in (5, 20, 35):
        data[:, t0 : t0 + 4] += pattern
    window = _window("w0", data)
    seg = _seg("s0", "w0", pattern, 5, 0)
    result = learn_averaged_templates_from_gpl(
        [window],
        [seg],
        n_rounds=1,
        eps_step_fraction=1e-12,
        min_repeats=2,
        show_progress=False,
        n_workers=1,
    )
    assert len(result.get_templates()) == 1
    assert result.get_round_logs()[0].seed_window_policy == "global"


if __name__ == "__main__":
    test_window_uncovered_fraction()
    test_select_low_coverage_prefers_uncovered_window()
    test_select_low_coverage_neighborhood()
    test_select_tie_break_by_energy()
    test_filter_candidates_by_windows()
    test_learn_low_coverage_seeds_from_uncovered_window()
    test_learn_global_policy_unchanged_default()
    print("ok")
