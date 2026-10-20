"""End-to-end test: averaged-templates dictionary for one WAV → catalog."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.dict_learning.code.learn import build_dictionary as build_dictionary_module  # noqa: E402
from analysis.dict_learning.code.learn.build_dictionary import (  # noqa: E402
    apply_refine_iters,
    build_dictionary_for_wav,
)
from analysis.dict_learning.code.learn.learn_dictionary import (  # noqa: E402
    LearnedDictionary,
    NoMatchingPatchesError,
)
from analysis.dict_learning.code.core.objects import Matching, Template, Window  # noqa: E402
from analysis.dict_learning.code.learn.seed_from_gpl import Seed  # noqa: E402
from database import catalog, index_existing  # noqa: E402


def _make_library(tmp: Path) -> tuple[Path, Path, str]:
    stem = "SOCAL_20230427_003345"
    root = tmp / "lib"
    wav = root / f"{stem}.wav"
    wav.parent.mkdir(parents=True)
    wav.write_bytes(b"RIFF....WAVE")

    npy_dir = root / "result" / f"{stem}.gpl" / "output" / "spectrogram_npy"
    npy_dir.mkdir(parents=True)

    n_freq, n_time, overlap = 4, 120, 10
    data = np.zeros((n_freq, n_time))
    for t0 in (5, 25, 45):
        data[0, t0 : t0 + 5] = 3.0
        data[1, t0 : t0 + 5] = -3.0
    for t0 in (70, 90):
        data[2, t0 : t0 + 4] = 2.0
        data[3, t0 : t0 + 4] = -2.0

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
    (npy_dir.parent / f"CSV_{stem}.csv").write_text(
        "start_time,end_time\n51,100\n701,740\n"
    )
    # Empty panel.bin so catalog can record panel_path (header not required for this test).
    (npy_dir.parent / "panel.bin").write_bytes(b"\x00\x00\x00\x02{}")
    return root, wav, stem


def test_build_dictionary_for_wav(tmp_path: Path, monkeypatch) -> None:
    root, wav, stem = _make_library(tmp_path)
    db = tmp_path / "catalog.sqlite"
    index_existing.reindex(db, wav_roots=[root], result_dirname="result")

    with catalog.connect(db) as conn:
        segs = catalog.list_gpl_segmentations_for_wav(conn, catalog.list_wav_files(conn)[0]["wav_id"])
        assert segs[0]["panel_path"] is not None

    stamps = iter(["20260101T000001Z", "20260101T000002Z"])
    monkeypatch.setattr(build_dictionary_module, "utc_run_stamp", lambda: next(stamps))

    out_dir = tmp_path / "dict_out"
    summary = build_dictionary_for_wav(
        db,
        str(wav),
        n_rounds=2,
        n_windows=10,
        out_dir=out_dir,
        eps_step_fraction=1e-12,
        min_repeats=2,
    )

    assert summary["n_windows"] == 1
    assert summary["n_templates"] >= 1
    assert summary["n_matchings"] >= 1
    assert summary["n_rounds_logged"] >= 1
    assert summary["stop_reason"] is not None
    assert summary["energy_after"] <= summary["energy_before"]
    assert summary["run_stamp"] == "20260101T000001Z"
    assert summary["base_out_dir"] == str(out_dir.resolve())
    run_out = Path(summary["out_dir"])
    assert run_out == out_dir / "20260101T000001Z"
    assert run_out.is_dir()
    assert (run_out / "template_000.npy").is_file()
    assert (run_out / "manifest.json").is_file()
    assert not (out_dir / "manifest.json").is_file()
    manifest = json.loads((run_out / "manifest.json").read_text())
    assert len(manifest["round_logs"]) == summary["n_rounds_logged"]
    widths = {row["template_id"]: row["shape"][1] for row in manifest["templates"]}
    intervals_by_window: dict[str, list[tuple[int, int]]] = {}
    for matching in manifest["matchings"]:
        start = matching["t_start"]
        intervals_by_window.setdefault(matching["window_id"], []).append(
            (start, start + widths[matching["template_id"]])
        )
    for intervals in intervals_by_window.values():
        intervals.sort()
        assert all(
            right_start >= left_end
            for (_, left_end), (right_start, _) in zip(intervals, intervals[1:])
        )

    paths_run1: list[str]
    with catalog.connect(db, read_only=True) as conn:
        d = catalog.get_dictionary(conn, summary["dictionary_id"])
        assert d["source"] == "seed"
        assert d["segmentation_id"] == summary["segmentation_id"]
        assert catalog.get_dictionary_for_segmentation(conn, summary["segmentation_id"]) is not None
        members = catalog.list_dictionary_templates(conn, d["dictionary_id"])
        assert len(members) == summary["n_templates"]
        assert all(Path(m["npy_path"]).is_file() for m in members)
        assert all(Path(m["npy_path"]).parent == run_out for m in members)
        paths_run1 = [m["npy_path"] for m in members]
        run = catalog.list_learning_runs(conn)[0]
        assert run["run_id"] == summary["run_id"]
        assert run["status"] == "completed"
        assert Path(run["out_dir"]) == run_out
        logs = catalog.list_learning_round_logs(conn, summary["run_id"])
        assert len(logs) == summary["n_rounds_logged"]
        assert logs[0]["averaged_n_matchings"] >= 1
        assert logs[0]["seed_n_matchings"] >= 1
        assert logs[0]["delta_energy"] is not None

    summary2 = build_dictionary_for_wav(
        db,
        stem,
        n_rounds=2,
        n_windows=10,
        out_dir=out_dir,
        eps_step_fraction=1e-12,
        min_repeats=2,
    )
    assert summary2["run_id"] != summary["run_id"]
    assert summary2["run_stamp"] == "20260101T000002Z"
    run_out2 = Path(summary2["out_dir"])
    assert run_out2 == out_dir / "20260101T000002Z"
    assert run_out2.is_dir()
    assert run_out.is_dir()
    stamped = sorted(p.name for p in out_dir.iterdir() if p.is_dir())
    assert stamped == ["20260101T000001Z", "20260101T000002Z"]
    for path in paths_run1:
        assert Path(path).is_file()
    with catalog.connect(db, read_only=True) as conn:
        assert len(catalog.list_dictionaries(conn)) == 2
        # Prior run's round logs remain; new run has its own logs.
        assert len(catalog.list_learning_round_logs(conn, summary["run_id"])) >= 1
        assert len(catalog.list_learning_round_logs(conn, summary2["run_id"])) >= 1
        members1 = catalog.list_dictionary_templates(conn, summary["dictionary_id"])
        assert [m["npy_path"] for m in members1] == paths_run1


def test_apply_refine_iters_keeps_unrefined_result_on_lost_matches(monkeypatch) -> None:
    """A NoMatchingPatchesError during refinement must not lose the learned dictionary."""
    data = np.zeros((4, 30))
    data[0, 5:10] = 1.0
    window = Window(
        window_id="w0", data=data, start_sample=1, end_sample=300, overlap=10
    )
    template = Template(template_id=0, data=data[:, 5:10].copy())
    seed = Seed(
        template=template,
        matchings=[Matching(0, "w0", 5, 1.0)],
        residual_after={"w0": data.copy()},
        energy_before=10.0,
        energy_after=4.0,
    )
    result = LearnedDictionary(
        seeds=[seed], energy_before=10.0, energy_after=4.0, stop_reason="completed"
    )

    def _raise(*args, **kwargs):
        raise NoMatchingPatchesError(0, iteration=1)

    monkeypatch.setattr(
        build_dictionary_module, "refine_dictionary_by_averaging", _raise
    )
    kept = apply_refine_iters(
        [window],
        result,
        refine_iters=2,
        eps_step_fraction=0.2,
        normalize=True,
    )

    assert len(kept.get_templates()) == 1
    assert kept.get_energy_after() == 4.0
    assert kept.get_stop_reason().startswith("refine_aborted")
    assert "template_id=0" in kept.get_stop_reason()


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        test_build_dictionary_for_wav(Path(d) / "a")
    print("test_build_dictionary: OK")
