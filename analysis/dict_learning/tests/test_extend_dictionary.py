"""Tests for single-stage dictionary extend (coverage-gated grow)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.dict_learning.code.core.objects import (  # noqa: E402
    Dictionary,
    Segment,
    Window,
)
from analysis.dict_learning.code.learn.extend_dictionary import (  # noqa: E402
    extend_dictionary_for_wav,
    extend_dictionary_on_windows,
)
from analysis.dict_learning.code.learn.learn_dictionary import (  # noqa: E402
    learn_averaged_templates_from_gpl,
)
from database import catalog, index_existing  # noqa: E402


def _make_window(data: np.ndarray, wid: str) -> Window:
    cols = data.shape[1]
    return Window(
        window_id=wid,
        data=data,
        start_sample=1,
        end_sample=1 + cols * 10 - 1,
        overlap=10,
    )


def _seg(seg_id: str, wid: str, data: np.ndarray, t0: int, det_index: int) -> Segment:
    return Segment(
        segment_id=seg_id,
        data=data.copy(),
        window_id=wid,
        t_start=t0,
        t_end=t0 + data.shape[1],
        global_start_sample=1,
        global_end_sample=2,
        detection_index=det_index,
    )


def _pattern_a() -> np.ndarray:
    p = np.zeros((4, 5), dtype=np.float64)
    p[0, :] = 3.0
    p[1, :] = -3.0
    return p


def _pattern_b() -> np.ndarray:
    p = np.zeros((4, 4), dtype=np.float64)
    p[2, :] = 2.5
    p[3, :] = -2.5
    return p


def _windows_and_segments():
    """Four windows: A repeats on w0/w1; B only on w2/w3."""
    pa, pb = _pattern_a(), _pattern_b()
    windows: list[Window] = []
    segments: list[Segment] = []
    det = 0
    for i, wid in enumerate(["w0", "w1", "w2", "w3"]):
        data = np.zeros((4, 80), dtype=np.float64)
        if i < 2:
            for t0 in (5, 25, 45):
                data[:, t0 : t0 + 5] += pa
                segments.append(_seg(f"{wid}_A{t0}", wid, pa, t0, det))
                det += 1
        else:
            for t0 in (10, 30, 50):
                data[:, t0 : t0 + 4] += pb
                segments.append(_seg(f"{wid}_B{t0}", wid, pb, t0, det))
                det += 1
        windows.append(_make_window(data, wid))
    return windows, segments, pa, pb


def test_template_id_start_and_occupied_intervals():
    windows, segments, _pa, _pb = _windows_and_segments()
    early = windows[:2]
    early_segs = [s for s in segments if s.window_id in {"w0", "w1"}]
    first = learn_averaged_templates_from_gpl(
        early,
        early_segs,
        n_rounds=1,
        eps_step_fraction=1e-12,
        min_repeats=2,
        disallow_overlap=True,
        show_progress=False,
    )
    assert len(first) == 1
    assert first.get_templates()[0].template_id == 0

    residual = first.get_residual_after()
    assert residual is not None
    tmpl = first.get_templates()[0]
    occupied: dict[str, list[tuple[int, int]]] = {}
    for m in first.get_matchings():
        occupied.setdefault(m.window_id, []).append(
            (m.t_start, m.t_start + tmpl.width)
        )
    second = learn_averaged_templates_from_gpl(
        residual,
        early_segs,
        windows_for_units=early,
        n_rounds=1,
        eps_step_fraction=1e-12,
        min_repeats=2,
        disallow_overlap=True,
        occupied_intervals=occupied,
        template_id_start=1,
        show_progress=False,
    )
    assert len(second) == 0 or all(
        t.template_id >= 1 for t in second.get_templates()
    )


def test_peel_only_when_coverage_already_met():
    windows, segments, _pa, _pb = _windows_and_segments()
    early = windows[:2]
    early_segs = [s for s in segments if s.window_id in {"w0", "w1"}]
    first = learn_averaged_templates_from_gpl(
        early,
        early_segs,
        n_rounds=1,
        eps_step_fraction=1e-12,
        min_repeats=2,
        show_progress=False,
    )
    parent = Dictionary()
    for t in first.get_templates():
        parent.add(t)

    result = extend_dictionary_on_windows(
        early,
        early_segs,
        dictionary=parent,
        min_coverage=0.01,
        max_new_atoms=5,
        eps_step_fraction=1e-12,
        min_repeats=2,
        min_template_width=None,
        max_template_width=None,
        refine_iters=1,
        show_progress=False,
    )
    assert result.n_atoms_added == 0
    assert result.coverage >= 0.01
    assert result.stop_reason == "min_coverage"
    assert len(result.dictionary) == len(parent)


def test_coverage_loop_adds_one_atom_then_refines():
    windows, segments, pa, pb = _windows_and_segments()
    result = extend_dictionary_on_windows(
        windows,
        segments,
        dictionary=None,
        min_coverage=0.9,
        max_new_atoms=4,
        eps_step_fraction=1e-12,
        min_repeats=2,
        min_template_width=None,
        max_template_width=None,
        refine_iters=1,
        show_progress=False,
    )
    assert result.n_atoms_added >= 1
    assert len(result.dictionary) == result.n_atoms_added
    coverages = [
        float(log["coverage"])
        for log in result.stage_logs
        if log.get("event") == "atom_added"
    ]
    assert coverages
    assert all(c2 >= c1 - 1e-9 for c1, c2 in zip(coverages, coverages[1:]))

    mats = [t.data for t in result.dictionary]
    has_a = any(float(np.sum(m[0:2] ** 2)) > float(np.sum(m[2:4] ** 2)) for m in mats)
    has_b = any(float(np.sum(m[2:4] ** 2)) > float(np.sum(m[0:2] ** 2)) for m in mats)
    assert has_a
    # Late windows may add B depending on coverage; if coverage hit early, B optional.
    if result.coverage < 0.9:
        assert has_b or result.stop_reason in {
            "below_min_repeats",
            "max_new_atoms",
            "no_candidates",
        }


def test_stop_when_grow_cannot_add_atom():
    windows, segments, _pa, _pb = _windows_and_segments()
    # Only three A placements on w0; require 4 repeats so grow never qualifies.
    result = extend_dictionary_on_windows(
        windows[:1],
        [s for s in segments if s.window_id == "w0"],
        dictionary=None,
        min_coverage=0.9,
        max_new_atoms=3,
        eps_step_fraction=1e-12,
        min_repeats=4,
        min_template_width=None,
        max_template_width=None,
        refine_iters=1,
        show_progress=False,
    )
    assert result.coverage < 0.9
    assert result.n_atoms_added == 0
    assert result.stop_reason in {"below_min_repeats", "no_candidates", "missing_residual"}


def _make_two_wav_library(tmp: Path) -> tuple[Path, Path, Path]:
    """Two WAVs with independent GPL segmentations (cross-WAV extend fixture)."""
    root = tmp / "lib"
    wavs: list[Path] = []
    for stem, t_slots in (
        ("WAV_A_20230427", (5, 25, 45)),
        ("WAV_B_20230428", (10, 30, 50)),
    ):
        wav = root / f"{stem}.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        wav.write_bytes(b"RIFF....WAVE")
        wavs.append(wav)
        npy_dir = root / "result" / f"{stem}.gpl" / "output" / "spectrogram_npy"
        npy_dir.mkdir(parents=True)
        n_freq, n_time, overlap = 4, 120, 10
        data = np.zeros((n_freq, n_time))
        for t0 in t_slots:
            data[0, t0 : t0 + 5] = 3.0
            data[1, t0 : t0 + 5] = -3.0
        win = npy_dir / f"spwhiten_{stem}_w000001.npy"
        np.save(win, data)
        win.with_suffix(".json").write_text(
            json.dumps(
                {
                    "start_sample": 1,
                    "end_sample": n_time * overlap,
                    "overlap": overlap,
                    "fftl": 8,
                    "sample_freq": 100.0,
                    "cols": n_time,
                    "rows": n_freq,
                }
            )
        )
        # Sample times ≈ 10 * t0 + 1 for start (matches build_dictionary fixture style).
        rows = []
        for t0 in t_slots:
            start = t0 * overlap + 1
            end = start + 5 * overlap - 1
            rows.append(f"{start},{end}")
        (npy_dir.parent / f"CSV_{stem}.csv").write_text(
            "start_time,end_time\n" + "\n".join(rows) + "\n"
        )
        (npy_dir.parent / "panel.bin").write_bytes(b"\x00\x00\x00\x02{}")
    return root, wavs[0], wavs[1]


def test_extend_catalog_lineage_and_cross_wav(tmp_path: Path, monkeypatch) -> None:
    root, wav_a, wav_b = _make_two_wav_library(tmp_path)
    db = tmp_path / "catalog.sqlite"
    index_existing.reindex(db, wav_roots=[root], result_dirname="result")

    import analysis.dict_learning.code.learn.extend_dictionary as extend_mod

    stamps = iter(["20260101T000001Z", "20260101T000002Z", "20260101T000003Z"])
    monkeypatch.setattr(extend_mod, "utc_run_stamp", lambda: next(stamps))

    out_a = tmp_path / "out_a"
    parent_summary = extend_dictionary_for_wav(
        db,
        str(wav_a),
        n_windows=1,
        out_dir=out_a,
        name="parent_a",
        min_coverage=0.01,
        max_new_atoms=1,
        eps_step_fraction=1e-12,
        min_repeats=2,
        min_template_width=None,
        max_template_width=None,
        refine_iters=1,
        show_progress=False,
    )
    parent_id = parent_summary["dictionary_id"]
    assert parent_id is not None
    assert parent_summary["source"] == "seed"
    assert parent_summary["parent_dictionary_id"] is None

    with catalog.connect(db) as conn:
        seg_a = parent_summary["segmentation_id"]
        wav_b_row = catalog.find_wav_file(conn, str(wav_b))
        assert wav_b_row is not None
        segs_b = catalog.list_gpl_segmentations_for_wav(conn, wav_b_row["wav_id"])
        assert segs_b
        seg_b = segs_b[0]["segmentation_id"]
        assert seg_b != seg_a

    out_b = tmp_path / "out_b"
    child = extend_dictionary_for_wav(
        db,
        str(wav_b),
        input_dictionary_id=int(parent_id),
        n_windows=1,
        out_dir=out_b,
        name="child_b",
        min_coverage=0.01,
        max_new_atoms=1,
        eps_step_fraction=1e-12,
        min_repeats=2,
        min_template_width=None,
        max_template_width=None,
        refine_iters=1,
        show_progress=False,
    )
    assert child["source"] == "refinement"
    assert child["parent_dictionary_id"] == parent_id
    assert child["input_dictionary_id"] == parent_id
    assert child["segmentation_id"] == seg_b

    with catalog.connect(db) as conn:
        lineage = catalog.get_dictionary_lineage(conn, int(child["dictionary_id"]))
        assert len(lineage) >= 2
        assert lineage[0]["dictionary_id"] == parent_id
        assert lineage[-1]["dictionary_id"] == child["dictionary_id"]
        assert lineage[-1]["parent_dictionary_id"] == parent_id
        wins = catalog.list_dictionary_windows(conn, int(child["dictionary_id"]))
        assert wins
        tool = catalog.get_tool_run(conn, int(child["tool_run_id"]))
        assert tool is not None
        assert tool["tool"] == "extend_dictionary"
        assert tool["input_dictionary_id"] == parent_id


def test_drop_unmatched_atoms_keeps_only_matched():
    """Atoms with no matchings are removed; ids and order of the rest survive."""
    from analysis.dict_learning.code.core.objects import Dictionary, Matching, Template
    from analysis.dict_learning.code.learn.extend_dictionary import _drop_unmatched_atoms

    dictionary = Dictionary()
    for tid in (0, 1, 2):
        dictionary.add(Template(template_id=tid, data=np.ones((2, 3))))

    kept, dropped = _drop_unmatched_atoms(
        dictionary, [Matching(0, "w0", 0, 1.0), Matching(2, "w0", 9, 1.0)]
    )
    assert dropped == [1]
    assert [t.template_id for t in kept] == [0, 2]


def test_drop_unmatched_atoms_is_a_noop_when_all_matched():
    from analysis.dict_learning.code.core.objects import Dictionary, Matching, Template
    from analysis.dict_learning.code.learn.extend_dictionary import _drop_unmatched_atoms

    dictionary = Dictionary()
    dictionary.add(Template(template_id=5, data=np.ones((2, 3))))
    kept, dropped = _drop_unmatched_atoms(dictionary, [Matching(5, "w0", 0, 1.0)])
    assert dropped == []
    assert kept is dictionary


def test_extend_leaves_no_unmatched_atoms():
    """End of an extend stage: every surviving atom owns at least one matching."""
    windows, segments, _pa, _pb = _windows_and_segments()
    result = extend_dictionary_on_windows(
        windows,
        segments,
        dictionary=None,
        min_coverage=0.9,
        max_new_atoms=4,
        eps_step_fraction=1e-12,
        min_repeats=2,
        min_template_width=None,
        max_template_width=None,
        refine_iters=1,
        show_progress=False,
    )
    matched = {m.template_id for m in result.matchings}
    orphans = [t.template_id for t in result.dictionary if t.template_id not in matched]
    assert not orphans, f"atoms with no matchings survived: {orphans}"
