"""Tests for the multi-template residual-deflation learner."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.dict_learning.code.learn.learn_dictionary import (  # noqa: E402
    DictionaryLearningError,
    DictionaryRefinementResult,
    LearnedDictionary,
    NoMatchingPatchesError,
    learn_averaged_templates_from_gpl,
    learn_from_gpl,
    refine_dictionary_by_averaging,
    refine_seed_by_averaging,
)
from analysis.dict_learning.code.learn.seed_from_gpl import Seed  # noqa: E402
from analysis.dict_learning.code.core import Dictionary, Matching, Segment, Template, Window  # noqa: E402


def _make_window(data: np.ndarray, wid: str = "w0", overlap: int = 10) -> Window:
    cols = data.shape[1]
    return Window(
        window_id=wid,
        data=data,
        start_sample=1,
        end_sample=1 + cols * overlap - 1,
        overlap=overlap,
    )


def _segment(seg_id: str, data: np.ndarray, t0: int, det_index: int) -> Segment:
    return Segment(
        segment_id=seg_id,
        data=data.copy(),
        window_id="w0",
        t_start=t0,
        t_end=t0 + data.shape[1],
        global_start_sample=1,
        global_end_sample=2,
        detection_index=det_index,
    )


def _two_pattern_window() -> tuple[Window, Segment, Segment, np.ndarray, np.ndarray]:
    """Window containing two disjoint-frequency repeating patterns.

    Pattern A (rows 0-1, ±3, width 5) repeats 3×; pattern B (rows 2-3, ±2,
    width 4) repeats 2×. Disjoint rows keep the normalized correlation between
    the patterns at zero, and both patterns (hence the whole window) are
    zero-mean so the mean-centering in Seed.from_gpl is a no-op and exact
    deflation still drives the residual to zero.
    """
    f, t = 4, 120
    pattern_a = np.zeros((f, 5), dtype=np.float64)
    pattern_a[0, :] = 3.0
    pattern_a[1, :] = -3.0
    pattern_b = np.zeros((f, 4), dtype=np.float64)
    pattern_b[2, :] = 2.0
    pattern_b[3, :] = -2.0

    residual = np.zeros((f, t), dtype=np.float64)
    for t0 in (5, 25, 45):
        residual[:, t0 : t0 + 5] += pattern_a
    for t0 in (70, 90):
        residual[:, t0 : t0 + 4] += pattern_b

    window = _make_window(residual)
    seg_a = _segment("segA", pattern_a, t0=5, det_index=0)
    seg_b = _segment("segB", pattern_b, t0=70, det_index=1)
    return window, seg_a, seg_b, pattern_a, pattern_b


def test_learn_finds_two_templates_with_all_occurrences():
    window, seg_a, seg_b, pattern_a, pattern_b = _two_pattern_window()
    assert seg_a.energy > seg_b.energy

    result = learn_from_gpl(
        [window],
        [seg_a, seg_b],
        max_templates=5,
        min_seed_delta_fraction=0.1,
        eps_step_fraction=1e-12,  # peel each template until no improvement
    )
    assert isinstance(result, LearnedDictionary)
    templates = result.get_templates()
    assert len(templates) == 2

    # Round order follows residual energy: A (louder) first, then B.
    assert templates[0].source_segment_id == "segA"
    assert templates[1].source_segment_id == "segB"
    assert templates[0].template_id != templates[1].template_id

    assert len(result.get_matchings_for(templates[0].template_id)) == 3
    assert len(result.get_matchings_for(templates[1].template_id)) == 2

    # Both patterns are exact copies, so deflation explains everything.
    assert result.get_energy_after() < 1e-9 * result.get_energy_before()


def test_learn_second_template_seeded_from_residual():
    """Round 2 candidates come from the residual: segA's slot must be spent."""
    window, seg_a, seg_b, _pa, _pb = _two_pattern_window()
    result = learn_from_gpl(
        [window],
        [seg_a, seg_b],
        max_templates=2,
        min_seed_delta_fraction=0.0,
        eps_step_fraction=1e-12,
    )
    seeds = result.get_seeds()
    assert len(seeds) == 2
    # After round 1 removed pattern A exactly, only segB survives re-cropping.
    assert seeds[1].get_n_candidates_tried() == 1
    assert seeds[1].get_selected_segment_id() == "segB"


def test_min_seed_delta_fraction_stops_rounds():
    window, seg_a, seg_b, _pa, _pb = _two_pattern_window()
    # ΔE round 1 (3×A) = 270, round 2 (2×B) = 64 → 64 < 0.5 × 270 stops round 2.
    result = learn_from_gpl(
        [window],
        [seg_a, seg_b],
        max_templates=5,
        min_seed_delta_fraction=0.5,
        eps_step_fraction=1e-12,
    )
    assert len(result.get_templates()) == 1
    assert result.get_templates()[0].source_segment_id == "segA"


def test_max_templates_caps_rounds():
    window, seg_a, seg_b, _pa, _pb = _two_pattern_window()
    result = learn_from_gpl(
        [window],
        [seg_a, seg_b],
        max_templates=1,
        eps_step_fraction=1e-12,
    )
    assert len(result.get_templates()) == 1


def test_learn_empty_detections():
    result = learn_from_gpl([_make_window(np.zeros((2, 10)))], [])
    assert len(result) == 0
    assert result.get_templates() == []
    assert result.get_matchings() == []


def test_template_ids_are_sequential():
    window, seg_a, seg_b, _pa, _pb = _two_pattern_window()
    result = learn_from_gpl(
        [window],
        [seg_a, seg_b],
        eps_step_fraction=1e-12,
    )
    ids = [t.template_id for t in result.get_templates()]
    assert ids == list(range(len(ids)))


def test_refine_seed_by_averaging():
    """Averaged template re-finds every occurrence and shrinks the residual."""
    window, seg_a, _seg_b, _pa, _pb = _two_pattern_window()
    seed = Seed.from_gpl(
        [window],
        [seg_a],
        eps_step_fraction=1e-12,
    )
    assert seed.is_selected()
    assert len(seed.get_matchings()) == 3

    refined = refine_seed_by_averaging(
        [window],
        seed,
        eps_step_fraction=1e-12,
    )
    assert refined.is_selected()
    template = refined.get_template()
    assert template is not None
    assert template.data.shape == seg_a.data.shape
    # The averaged template matches the same three occurrences of pattern A.
    assert len(refined.get_matchings()) >= 3
    assert refined.get_energy_after() < refined.get_energy_before()


def test_refine_dictionary_by_averaging_repeats_multi_atom_mp():
    window, _seg_a, _seg_b, pattern_a, pattern_b = _two_pattern_window()
    dictionary = Dictionary()
    dictionary.add(Template(template_id=0, data=pattern_a.copy()))
    dictionary.add(Template(template_id=1, data=pattern_b.copy()))
    initial_matchings = [
        *(Matching(0, "w0", t0, 1.0) for t0 in (5, 25, 45)),
        *(Matching(1, "w0", t0, 1.0) for t0 in (70, 90)),
    ]

    result = refine_dictionary_by_averaging(
        [window],
        dictionary,
        initial_matchings,
        iter=2,
        eps_step_fraction=1e-12,
    )

    assert isinstance(result, DictionaryRefinementResult)
    assert result.iterations_completed == 2
    assert len(result.dictionary) == 2
    assert len(result.matchings) == 5
    assert {m.template_id for m in result.matchings} == {0, 1}
    assert np.allclose(result.dictionary.templates[0].data, pattern_a)
    assert np.allclose(result.dictionary.templates[1].data, pattern_b)
    assert result.energies[-1] < 1e-9 * result.energies[0]


def test_refine_dictionary_requires_matches_for_each_atom():
    window, _seg_a, _seg_b, pattern_a, pattern_b = _two_pattern_window()
    dictionary = Dictionary()
    dictionary.add(Template(template_id=0, data=pattern_a))
    dictionary.add(Template(template_id=1, data=pattern_b))

    try:
        refine_dictionary_by_averaging(
            [window],
            dictionary,
            [Matching(0, "w0", 5, 1.0)],
        )
    except NoMatchingPatchesError as exc:
        assert exc.template_id == 1
        assert "template_id=1" in str(exc)
        # Remains catchable via the base class and legacy ValueError.
        assert isinstance(exc, DictionaryLearningError)
        assert isinstance(exc, ValueError)
    else:
        raise AssertionError(
            "expected missing template matches to raise NoMatchingPatchesError"
        )


def _loud_and_faint_window() -> tuple[Window, np.ndarray, np.ndarray]:
    """Loud pattern repeated 3×, plus one faint pattern MP stops before reaching."""
    f, t = 4, 120
    loud = np.zeros((f, 5), dtype=np.float64)
    loud[0, :] = 3.0
    loud[1, :] = -3.0
    faint = np.zeros((f, 4), dtype=np.float64)
    faint[2, :] = 0.05
    faint[3, :] = -0.05

    data = np.zeros((f, t), dtype=np.float64)
    for t0 in (5, 25, 45):
        data[:, t0 : t0 + 5] += loud
    data[:, 70:74] += faint
    return _make_window(data), loud, faint


def _loud_faint_dictionary(loud: np.ndarray, faint: np.ndarray) -> Dictionary:
    dictionary = Dictionary()
    dictionary.add(Template(template_id=0, data=loud.copy()))
    dictionary.add(Template(template_id=1, data=faint.copy()))
    return dictionary


def test_refine_drops_faint_atom_without_pinning():
    window, loud, faint = _loud_and_faint_window()
    labels = [
        *(Matching(0, "w0", t0, 1.0) for t0 in (5, 25, 45)),
        Matching(1, "w0", 70, 1.0),
    ]

    # One pass: the eps stopping rule fires before the faint atom is placed, so
    # its labeled patch is silently dropped from the matchings.
    one_pass = refine_dictionary_by_averaging(
        [window], _loud_faint_dictionary(loud, faint), labels, iter=1
    )
    assert {m.template_id for m in one_pass.matchings} == {0}

    # Two passes: the now-unmatched atom has nothing to average.
    try:
        refine_dictionary_by_averaging(
            [window], _loud_faint_dictionary(loud, faint), labels, iter=2
        )
    except NoMatchingPatchesError as exc:
        assert exc.template_id == 1
    else:
        raise AssertionError("expected the starved faint atom to raise")


def test_pinned_matchings_keep_faint_atom_and_its_patch():
    window, loud, faint = _loud_and_faint_window()
    pinned = [Matching(1, "w0", 70, 1.0)]
    labels = [*(Matching(0, "w0", t0, 1.0) for t0 in (5, 25, 45)), *pinned]

    result = refine_dictionary_by_averaging(
        [window],
        _loud_faint_dictionary(loud, faint),
        labels,
        iter=2,
        pinned_matchings=pinned,
    )

    assert result.iterations_completed == 2
    assert result.n_pinned_matchings == 1
    assert len(result.dictionary) == 2
    kept = [m for m in result.matchings if m.template_id == 1]
    assert [m.t_start for m in kept] == [70]
    assert {m.template_id for m in result.matchings} == {0, 1}
    got = result.dictionary.templates[1].data.ravel()
    ref = faint.ravel()
    corr = float(np.dot(got, ref) / (np.linalg.norm(got) * np.linalg.norm(ref)))
    assert abs(corr) > 0.99


def test_pinned_matching_is_not_reassigned_to_a_competing_atom():
    """Two atoms of identical shape: MP hands every patch to the first one."""
    window, loud, _faint = _loud_and_faint_window()
    dictionary = Dictionary()
    dictionary.add(Template(template_id=0, data=loud.copy()))
    dictionary.add(Template(template_id=1, data=loud.copy()))
    pinned = [Matching(1, "w0", 45, 1.0)]
    labels = [*(Matching(0, "w0", t0, 1.0) for t0 in (5, 25)), *pinned]

    unpinned = refine_dictionary_by_averaging(
        [window], dictionary, labels, iter=1, eps_step_fraction=1e-12
    )
    assert (1, 45) not in {(m.template_id, m.t_start) for m in unpinned.matchings}

    result = refine_dictionary_by_averaging(
        [window],
        dictionary,
        labels,
        iter=1,
        eps_step_fraction=1e-12,
        pinned_matchings=pinned,
    )
    assert (1, 45) in {(m.template_id, m.t_start) for m in result.matchings}
    # The pinned span is off limits, so atom 0 cannot double-claim it.
    assert (0, 45) not in {(m.template_id, m.t_start) for m in result.matchings}


def test_learn_averaged_templates_from_gpl_rounds():
    window, seg_a, seg_b, _pa, _pb = _two_pattern_window()
    result = learn_averaged_templates_from_gpl(
        [window],
        [seg_a, seg_b],
        n_windows=1,
        n_rounds=2,
        eps_step_fraction=1e-12,
        min_repeats=2,
    )
    assert isinstance(result, LearnedDictionary)
    seeds = result.get_seeds()
    assert len(seeds) == 2
    # First round should explain the stronger pattern A at least 3 occurrences.
    assert len(seeds[0].get_matchings()) >= 3
    # Outer residual-deflation should reduce energy round-to-round.
    assert seeds[1].get_energy_before() < seeds[0].get_energy_before()
    assert result.get_energy_after() < result.get_energy_before()


def test_learn_averaged_filters_by_template_width():
    window, seg_a, seg_b, _pa, _pb = _two_pattern_window()
    # Pattern A width=5, B width=4 — keep only A.
    result = learn_averaged_templates_from_gpl(
        [window],
        [seg_a, seg_b],
        n_rounds=2,
        eps_step_fraction=1e-12,
        min_repeats=2,
        min_template_width=5,
        max_template_width=5,
    )
    templates = result.get_templates()
    assert len(templates) == 1
    assert templates[0].source_segment_id == "segA"


def test_learn_averaged_window_offset():
    window0, seg_a, seg_b, _pa, _pb = _two_pattern_window()
    quiet = np.zeros_like(window0.data)
    window1 = _make_window(quiet, wid="w1")
    # Skip the quiet first window; learn on the patterned second.
    result = learn_averaged_templates_from_gpl(
        [window1, window0],
        [seg_a, seg_b],
        n_windows=1,
        window_offset=1,
        n_rounds=1,
        eps_step_fraction=1e-12,
        min_repeats=2,
    )
    assert len(result.get_templates()) >= 1
    assert all(m.window_id == "w0" for m in result.get_matchings())


def test_learn_averaged_round_logs_and_coverage():
    from analysis.dict_learning.code.learn.learn_dictionary import LearningRoundLog

    window, seg_a, seg_b, _pa, _pb = _two_pattern_window()
    seen: list[LearningRoundLog] = []
    result = learn_averaged_templates_from_gpl(
        [window],
        [seg_a, seg_b],
        n_rounds=2,
        eps_step_fraction=1e-12,
        min_repeats=2,
        on_round_complete=seen.append,
    )
    logs = result.get_round_logs()
    assert len(logs) == len(seen) == 2
    assert result.get_stop_reason() == "max_rounds"
    assert logs[0].seed_n_matchings >= 1
    assert logs[0].averaged_n_matchings >= 1
    assert logs[0].delta_energy > 0
    assert logs[0].n_windows_matched_round >= 1
    assert logs[0].n_windows_matched_cumulative >= 1
    assert 0.0 < logs[0].window_coverage_fraction <= 1.0
    assert 0.0 < logs[0].time_coverage_fraction <= 1.0
    assert logs[1].cumulative_delta_energy >= logs[0].cumulative_delta_energy
    assert logs[1].n_windows_matched_cumulative >= logs[0].n_windows_matched_cumulative
    assert "seed_matches=" in logs[0].summary_line()


def test_save_round_template_each_outer_iteration(tmp_path=None):
    from pathlib import Path

    out = Path(tmp_path) if tmp_path is not None else Path("/tmp/test_round_templates")
    if out.exists():
        import shutil

        shutil.rmtree(out)
    window, seg_a, seg_b, _pa, _pb = _two_pattern_window()
    result = learn_averaged_templates_from_gpl(
        [window],
        [seg_a, seg_b],
        n_rounds=2,
        eps_step_fraction=1e-12,
        min_repeats=2,
        template_out_dir=out,
    )
    assert len(result) == 2
    assert (out / "round_000" / "template.npy").is_file()
    assert (out / "round_001" / "template.npy").is_file()
    assert (out / "manifest.json").is_file()
    loaded = np.load(out / "round_000" / "template.npy")
    assert loaded.shape == result.get_templates()[0].data.shape
    assert np.allclose(loaded, result.get_templates()[0].data)


def test_learn_averaged_stops_below_min_repeats_without_round_cap():
    """With no n_rounds cap, learning stops when no seed meets min_repeats."""
    window, seg_a, seg_b, _pa, _pb = _two_pattern_window()
    # Pattern B has only 2 occurrences, so min_repeats=3 accepts only A.
    result = learn_averaged_templates_from_gpl(
        [window],
        [seg_a, seg_b],
        n_rounds=None,
        eps_step_fraction=1e-12,
        min_repeats=3,
    )
    assert len(result.get_templates()) == 1
    assert result.get_templates()[0].source_segment_id == "segA"
    assert result.get_stop_reason() == "below_min_repeats"


if __name__ == "__main__":
    test_learn_finds_two_templates_with_all_occurrences()
    test_learn_second_template_seeded_from_residual()
    test_min_seed_delta_fraction_stops_rounds()
    test_max_templates_caps_rounds()
    test_learn_empty_detections()
    test_template_ids_are_sequential()
    test_refine_seed_by_averaging()
    test_refine_dictionary_by_averaging_repeats_multi_atom_mp()
    test_refine_dictionary_requires_matches_for_each_atom()
    test_refine_drops_faint_atom_without_pinning()
    test_pinned_matchings_keep_faint_atom_and_its_patch()
    test_pinned_matching_is_not_reassigned_to_a_competing_atom()
    test_learn_averaged_templates_from_gpl_rounds()
    test_learn_averaged_filters_by_template_width()
    test_learn_averaged_window_offset()
    test_learn_averaged_round_logs_and_coverage()
    test_save_round_template_each_outer_iteration()
    test_learn_averaged_stops_below_min_repeats_without_round_cap()
    print("test_learn_dictionary: OK")
