"""``--overlap limited`` on both builds and on matching-pursuit, end to end."""

from __future__ import annotations

import json
from contextlib import closing
from itertools import combinations

import numpy as np
import pytest

from analysis.dict_learning.code.cli import main
from analysis.dict_learning.code.mp import OverlapRule
from analysis.test.catalog_setup import pursue
from analysis.test.synthetic import read_json
from database import catalog

BUILDS = {
    "randomized-build": ["--m-per-outer", "3", "--n-min", "2", "--refine-iters", "1", "--seed", "0"],
    # Two lengths, so 6-column atoms can sit inside 12-column ones.
    "pan-build": ["--lengths-s", "0.06,0.12", "--m-per-outer", "4", "--n-min", "2", "--refine-iters", "1"],
}
COMMON = ["--workers", "1", "--max-outer", "2", "--min-coverage", "1.0"]


def _overlaps(folder, widths) -> tuple[int, int]:
    """(pairs of placements in one window that overlap at all, pairs the rule forbids)."""
    rule = OverlapRule()
    by_window: dict[str, list[tuple[int, int]]] = {}
    for row in json.loads((folder / "matchings.json").read_text()):
        start = int(row["t_start"])
        by_window.setdefault(row["window_id"], []).append((start, start + widths[int(row["template_id"])]))
    touching = forbidden = 0
    for spans in by_window.values():
        for (a0, a1), (b0, b1) in combinations(spans, 2):
            touching += min(a1, b1) > max(a0, b0)
            forbidden += not rule.allows(a0, a1, b0, b1)
    return touching, forbidden


@pytest.mark.parametrize("command", BUILDS)
def test_builds_and_pursuit_keep_to_the_rule(command, gpl_dir, tmp_path):
    runs = {}
    for mode in ("free", "limited"):
        out = tmp_path / f"{command}_{mode}"
        argv = [command, "--gpl-dir", str(gpl_dir), "--out", str(out), *BUILDS[command], *COMMON, "--overlap", mode]
        assert main(argv) == 0
        summary = read_json(out / "summary.json")
        assert summary["overlap"] == mode
        assert ("max_partial_overlap" in summary) == (mode == "limited")
        widths = {int(a["template_id"]): int(a["width"]) for a in summary["atoms"]}
        runs[mode] = _overlaps(out, widths)
        for mp_mode in ("per_window", "global"):
            mp_out = tmp_path / f"{command}_{mode}_{mp_mode}"
            assert pursue(gpl_dir, out, mp_out, "--overlap", mode, "--mp-mode", mp_mode) == 0
            assert read_json(mp_out / "summary.json")["overlap"] == mode
            assert read_json(mp_out / "summary.json")["n_matchings"] > 0
            if mode == "limited":
                assert _overlaps(mp_out, widths)[1] == 0
    # Limited placements never break the rule (free ones may; the unit tests
    # show a case), and they still overlap where the rule allows.
    assert runs["limited"][1] == 0
    assert runs["limited"][0] <= runs["free"][0]


def test_no_overlap_at_all(gpl_dir, tmp_path):
    # Two lengths, so nesting would be possible if it were allowed.
    out = tmp_path / "pan_none"
    argv = ["pan-build", "--gpl-dir", str(gpl_dir), "--out", str(out), *BUILDS["pan-build"], *COMMON,
            "--overlap", "limited", "--max-partial-overlap", "0", "--max-nested-ratio", "0"]
    assert main(argv) == 0
    summary = read_json(out / "summary.json")
    assert (summary["max_partial_overlap"], summary["max_nested_ratio"]) == (0.0, 0.0)
    widths = {int(a["template_id"]): int(a["width"]) for a in summary["atoms"]}
    assert summary["n_matchings"] > 0 and _overlaps(out, widths)[0] == 0


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--max-partial-overlap", "1"], "max partial overlap must be in [0, 1)"),
        (["--max-nested-ratio", "-0.1"], "max nested ratio must be in [0, 1)"),
    ],
)
def test_bad_limits_exit_2(gpl_dir, tmp_path, capsys, extra, message):
    argv = ["pan-build", "--gpl-dir", str(gpl_dir), "--out", str(tmp_path / "pan"), *BUILDS["pan-build"], *COMMON,
            "--overlap", "limited", *extra]
    assert main(argv) == 2
    assert message in capsys.readouterr().err
    assert pursue(gpl_dir, tmp_path / "pan", tmp_path / "mp", "--overlap", "limited", *extra) == 2


def test_a_second_stage_from_a_no_overlap_dictionary(gpl_dir, tmp_path, capsys):
    db = tmp_path / "catalog.sqlite"
    first = tmp_path / "stage1"
    argv = ["pan-build", "--gpl-dir", str(gpl_dir), "--out", str(first), *BUILDS["pan-build"], *COMMON,
            "--max-outer", "1", "--overlap", "limited", "--max-partial-overlap", "0", "--max-nested-ratio", "0"]
    assert main(argv) == 0
    stage1 = read_json(first / "summary.json")
    for command in BUILDS:
        out = tmp_path / f"stage2_{command}"
        argv = [command, "--gpl-dir", str(gpl_dir), "--out", str(out), *BUILDS[command], *COMMON,
                "--overlap", "limited", "--init-dictionary", str(first), "--catalog", str(db)]
        assert main(argv) == 0
        summary = read_json(out / "summary.json")
        assert summary["init_dictionary"] == str(first.resolve())
        assert summary["n_atoms"] >= stage1["n_atoms"]
        assert summary["time_coverage_fraction"] >= stage1["time_coverage_fraction"]
        # The first stage's atoms come through unchanged.
        for atom in stage1["atoms"]:
            np.testing.assert_array_equal(np.load(out / atom["path"]), np.load(first / atom["path"]))
    assert '"kind": "pan-build"' in capsys.readouterr().out
    with closing(catalog.connect(db, read_only=True)) as conn:
        rows = {r["out_dir"]: dict(r) for r in conn.execute("SELECT * FROM dictionaries").fetchall()}
        parent = rows[str(first.resolve())]
        assert parent["source"] == "seed"
        for command in BUILDS:
            child = rows[str((tmp_path / f"stage2_{command}").resolve())]
            assert child["source"] == "refinement" and child["parent_dictionary_id"] == parent["dictionary_id"]
            assert json.loads(child["params_json"])["init_dictionary"] == str(first.resolve())


def test_init_dictionary_that_does_not_fit_exit_2(gpl_dir, tmp_path, capsys):
    first = tmp_path / "stage1"
    assert main(["pan-build", "--gpl-dir", str(gpl_dir), "--out", str(first), *BUILDS["pan-build"], *COMMON,
                 "--max-outer", "1"]) == 0
    run = ["pan-build", "--gpl-dir", str(gpl_dir), "--out", str(tmp_path / "stage2"), *BUILDS["pan-build"], *COMMON,
           "--init-dictionary", str(first)]
    # Placements in windows this run does not load.
    assert main([*run, "--n-windows", "1"]) == 2
    assert "placement windows of" in capsys.readouterr().err
    # Atoms with other frequency rows.
    atom = read_json(first / "summary.json")["atoms"][0]["path"]
    np.save(first / atom, np.ones((5, 12)))
    assert main(run) == 2
    assert "different frequency rows" in capsys.readouterr().err
