"""``python -m analysis.dict_learning pan-build`` end to end."""

from __future__ import annotations

import json
from contextlib import closing

import numpy as np
import pytest

from analysis.dict_learning.code.cli import main
from analysis.test.catalog_setup import downgrade_tool_runs, pursue
from analysis.test.synthetic import read_json, run_module
from database import catalog

# The synthetic motifs are 12 columns of 10 ms. --n-min 2 and --refine-iters 1
# keep a run short, as in the randomized-build tests.
FAST = ["--lengths-s", "0.12", "--m-per-outer", "3", "--n-min", "2", "--refine-iters", "1",
        "--workers", "1", "--max-outer", "2", "--min-coverage", "1.0"]


def pan_build(gpl_dir, out, *extra):
    return main(["pan-build", "--gpl-dir", str(gpl_dir), "--out", str(out), *FAST, *extra])


def test_writes_a_build_folder_with_its_seeds(gpl_dir, tmp_path, capsys):
    out = tmp_path / "pan"
    assert pan_build(gpl_dir, out, "--verbose") == 0
    summary = read_json(out / "summary.json")
    assert summary["command"] == "pan-build"
    assert summary["seeder"] == {
        "signal": "pca", "pca_rows": 8, "band_edges": None, "lengths_s": [0.12],
        "lengths_columns": [12], "decimate": 1, "copy_threshold": 0.5, "min_copies": 3,
        "dedupe_corr": 0.9,
    }
    assert "epsilon" not in summary and "seed" not in summary
    assert summary["n_atoms"] == len(summary["atoms"]) > 0
    for atom in summary["atoms"]:
        assert np.load(out / atom["path"]).shape == (12, 12)
        assert atom["source_segment_id"].startswith("pan:")
    passes = [json.loads(line) for line in (out / "pan_seeds.jsonl").read_text().splitlines()]
    logs = (out / "outer_logs.jsonl").read_text().splitlines()
    assert len(passes) == len(logs) == summary["n_outer_completed"] == 2
    # Two seeds of the second pass refine to the shape of a third and are folded into it.
    assert [json.loads(line)["n_merged"] for line in logs] == [0, 2]
    first = passes[0]
    assert first["signal_rows"] == 8 and first["signal_columns"] == 720
    assert len(first["picks"]) == 3
    assert all(len(pick["copies"]) >= 3 for pick in first["picks"])
    assert "outer 1:" in capsys.readouterr().err


def test_finds_the_planted_motifs(gpl_folder, tmp_path):
    out = tmp_path / "pan"
    assert pan_build(gpl_folder.root, out, "--max-outer", "1", "--signal", "full") == 0
    atoms = [np.load(out / a["path"]) for a in read_json(out / "summary.json")["atoms"]]
    for motif in gpl_folder.motifs:
        unit = motif / np.linalg.norm(motif)
        assert max(float(np.sum(atom * unit)) for atom in atoms) > 0.9


def test_matching_pursuit_and_catalog_read_it(gpl_dir, tmp_path, capsys):
    db = tmp_path / "catalog.sqlite"
    out = tmp_path / "pan"
    assert pan_build(gpl_dir, out, "--catalog", str(db)) == 0
    assert '"kind": "pan-build"' in capsys.readouterr().out
    assert pursue(gpl_dir, out, tmp_path / "mp", "--catalog", str(db)) == 0
    with closing(catalog.connect(db, read_only=True)) as conn:
        tools = sorted(r["tool"] for r in catalog.list_tool_runs(conn))
        assert tools == ["matching_pursuit", "pan_build"]


def test_registers_in_a_catalog_made_before_pan_build(gpl_dir, tmp_path):
    db = tmp_path / "catalog.sqlite"
    first = tmp_path / "pan_a"
    assert pan_build(gpl_dir, first, "--max-outer", "1") == 0
    with closing(catalog.connect(db)) as conn, conn:
        conn.execute(
            "INSERT INTO tool_runs (tool, out_dir, created_at) VALUES ('randomized_build', 'elsewhere', 'now')"
        )
    downgrade_tool_runs(db)
    result = run_module("database", "register", "--db", str(db), str(first))
    assert result.returncode == 0, result.stderr
    with closing(catalog.connect(db, read_only=True)) as conn:
        tools = sorted(r["tool"] for r in catalog.list_tool_runs(conn))
        assert tools == ["pan_build", "randomized_build"]


@pytest.mark.parametrize(
    "extra, rows",
    [
        (["--signal", "energy"], 1),
        (["--signal", "bands", "--band-edges", "10,50,90,130"], 3),
        (["--signal", "pca", "--pca-rows", "4"], 4),
    ],
)
def test_search_signals(gpl_dir, tmp_path, extra, rows):
    out = tmp_path / "pan"
    assert pan_build(gpl_dir, out, *extra) == 0
    first = json.loads((out / "pan_seeds.jsonl").read_text().splitlines()[0])
    assert first["signal_rows"] == rows
    # A one-row signal cannot tell the band-separated motifs apart, so some
    # pairs point at copies already taken.
    if rows == 1:
        assert first["n_masked"] > 0


def test_decimated_search(gpl_dir, tmp_path):
    out = tmp_path / "pan"
    assert pan_build(gpl_dir, out, "--decimate", "2", "--max-outer", "1") == 0
    first = json.loads((out / "pan_seeds.jsonl").read_text().splitlines()[0])
    assert first["signal_columns"] == 360
    assert all(atom["width"] == 12 for atom in read_json(out / "summary.json")["atoms"])


def test_takes_every_pair_it_can(gpl_dir, tmp_path):
    # Asking for more seeds than the signal holds: the pass ends when every
    # start is masked or used, and loose duplicate and copy rules keep it going.
    out = tmp_path / "pan"
    assert pan_build(gpl_dir, out, "--m-per-outer", "100", "--min-copies", "2", "--max-outer", "1",
                     "--dedupe-corr", "0.3") == 0
    first = json.loads((out / "pan_seeds.jsonl").read_text().splitlines()[0])
    assert first["n_duplicates"] > 0
    assert first["n_pairs_tried"] < 25 * 100


def test_no_pair_with_enough_copies_exits_1(gpl_dir, tmp_path, capsys):
    out = tmp_path / "pan"
    assert pan_build(gpl_dir, out, "--m-per-outer", "1", "--min-copies", "50") == 1
    summary = read_json(out / "summary.json")
    assert summary["stop_reason"] == "empty_pool"
    first = json.loads((out / "pan_seeds.jsonl").read_text().splitlines()[0])
    assert first["n_pairs_tried"] == 25 and first["n_too_few_copies"] > 0
    assert "no atoms survived" in capsys.readouterr().err


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--lengths-s", "0"], "--lengths-s must be positive"),
        (["--lengths-s", ","], "--lengths-s must be positive"),
        (["--lengths-s", "0.03"], "at least 4 search samples"),
        (["--lengths-s", "5"], "too short for motif length"),
        (["--decimate", "0"], "decimate must be >= 1"),
        (["--copy-threshold", "1"], "copy threshold"),
        (["--min-copies", "1"], "min copies"),
        (["--dedupe-corr", "0"], "dedupe correlation"),
        (["--pca-rows", "13"], "PCA rows must be between 1 and 12"),
        (["--post-refine-max-relative-distance", "2"], "must be in [0, 1]"),
    ],
)
def test_bad_input_exit_2(gpl_dir, tmp_path, capsys, extra, message):
    assert pan_build(gpl_dir, tmp_path / "pan", *extra) == 2
    assert message in capsys.readouterr().err


def test_sidecars_without_sample_rate_exit_2(gpl_variant, tmp_path, capsys):
    root = gpl_variant(sidecar_sample_freq=False)
    assert pan_build(root, tmp_path / "pan") == 2
    assert "needs sample_freq" in capsys.readouterr().err


def test_no_detection_csv_needed(gpl_variant, tmp_path):
    assert pan_build(gpl_variant(detections="none"), tmp_path / "pan", "--max-outer", "1") == 0


def test_python_dash_m_entry_point(gpl_dir, tmp_path):
    out = tmp_path / "pan"
    result = run_module("analysis.dict_learning", "pan-build", "--gpl-dir", str(gpl_dir), "--out", str(out),
                        *FAST, "--max-outer", "1")
    assert result.returncode == 0, result.stderr
    assert read_json(out / "summary.json")["command"] == "pan-build"
