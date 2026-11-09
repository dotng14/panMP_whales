"""``python -m analysis.dict_learning split`` end to end."""

from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from analysis.dict_learning.code.cli import load_windows, main
from analysis.test.catalog_setup import downgrade_tool_runs, pursue
from analysis.test.synthetic import read_json, run_module
from database import catalog


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


def mixed_dictionary(gpl_folder, out: Path) -> Path:
    """A dictionary whose atom 0 is the mean of motifs 0 and 1, placed on every copy of both.

    Atom 1 is motif 2 on its own copies. Splitting atom 0 must give motifs 0
    and 1 back, each with its own copies.
    """
    windows = load_windows(gpl_folder.root, None, 0)
    ids = [w.window_id for w in windows]
    m0, m1, m2 = gpl_folder.motifs
    atoms = {0: _unit(m0 + m1), 1: _unit(m2)}
    (out / "atoms").mkdir(parents=True)
    rows, listed = [], []
    for template_id, data in atoms.items():
        path = f"atoms/template_{template_id:03d}.npy"
        np.save(out / path, data)
        listed.append({"template_id": template_id, "width": data.shape[1], "n_freq": data.shape[0],
                       "n_components": 1, "n_patches": 0, "source_segment_id": f"hand:{template_id}", "path": path})
    for w, k, c in gpl_folder.copies:
        rows.append({"template_id": 0 if k in (0, 1) else 1, "window_id": ids[w], "t_start": c, "amplitude": 1.0})
    (out / "matchings.json").write_text(json.dumps(rows))
    summary = {"command": "randomized-build", "gpl_dir": str(gpl_folder.root.resolve()), "n_windows": len(ids),
               "window_ids": ids, "window_offset": 0, "n_atoms": 2, "n_matchings": len(rows), "atoms": listed}
    (out / "summary.json").write_text(json.dumps(summary))
    return out


def split(gpl_folder, dictionary, out, *extra) -> int:
    return main(["split", "--gpl-dir", str(gpl_folder.root), "--dictionary", str(dictionary), "--out", str(out),
                 "--seed", "0", *extra])


@pytest.fixture
def mixed(gpl_folder, tmp_path):
    return mixed_dictionary(gpl_folder, tmp_path / "mixed")


def test_splits_a_mixed_atom_into_its_two_motifs(gpl_folder, mixed, tmp_path, capsys):
    out = tmp_path / "split"
    assert split(gpl_folder, mixed, out, "--template", "0") == 0
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed["new_template_ids"] == [0, 2]
    summary = read_json(out / "summary.json")
    assert summary["command"] == "split" and summary["accepted"] and summary["relative_reduction"] > 0.5
    assert [a["template_id"] for a in summary["atoms"]] == [0, 1, 2]
    assert summary["atoms"][2]["source_segment_id"] == "split:hand:0:b"
    atoms = {a["template_id"]: np.load(out / a["path"]) for a in summary["atoms"]}
    np.testing.assert_array_equal(atoms[1], np.load(mixed / "atoms/template_001.npy"))
    # Each half is one planted motif and holds exactly that motif's copies.
    rows = read_json(out / "matchings.json")
    windows = [w.window_id for w in load_windows(gpl_folder.root, None, 0)]
    for half in (0, 2):
        motif = int(np.argmax([abs(float(np.sum(atoms[half] * _unit(m)))) for m in gpl_folder.motifs]))
        assert motif in (0, 1)
        assert abs(float(np.sum(atoms[half] * _unit(gpl_folder.motifs[motif])))) > 0.99
        planted = {(windows[w], c) for w, k, c in gpl_folder.copies if k == motif}
        assert {(r["window_id"], r["t_start"]) for r in rows if r["template_id"] == half} == planted
    assert sum(a["n_patches"] for a in summary["atoms"]) == len(rows) == len(gpl_folder.copies)


def test_the_split_is_a_dictionary_for_matching_pursuit_and_the_catalog(gpl_folder, mixed, tmp_path, capsys):
    db = tmp_path / "catalog.sqlite"
    out = tmp_path / "split"
    assert split(gpl_folder, mixed, out, "--template", "0", "--catalog", str(db)) == 0
    assert '"kind": "split"' in capsys.readouterr().out
    assert pursue(gpl_folder.root, out, tmp_path / "mp") == 0
    with closing(catalog.connect(db, read_only=True)) as conn:
        tools = sorted(r["tool"] for r in catalog.list_tool_runs(conn))
        assert tools == ["randomized_build", "split"]
        rows = {r["out_dir"]: r for r in conn.execute("SELECT * FROM dictionaries").fetchall()}
        child, parent = rows[str(out.resolve())], rows[str(mixed.resolve())]
        assert child["source"] == "edit" and child["parent_dictionary_id"] == parent["dictionary_id"]
        assert parent["source"] == "seed"
        run = conn.execute("SELECT input_dictionary_id FROM tool_runs WHERE tool = 'split'").fetchone()
        assert run["input_dictionary_id"] == parent["dictionary_id"]


def test_split_of_a_split_and_patches_from_a_matching_pursuit_run(gpl_folder, mixed, tmp_path):
    first = tmp_path / "split"
    assert split(gpl_folder, mixed, first, "--template", "0") == 0
    mp = tmp_path / "mp"
    assert pursue(gpl_folder.root, first, mp) == 0
    second = tmp_path / "split2"
    assert split(gpl_folder, first, second, "--template", "2", "--matchings", str(mp),
                 "--min-relative-reduction", "0") == 0
    summary = read_json(second / "summary.json")
    assert summary["matchings_from"] == str(mp.resolve()) and summary["new_template_ids"] == [2, 3]
    assert summary["atoms"][3]["source_segment_id"] == "split:split:hand:0:b:b"


def test_a_one_motif_atom_is_not_split(gpl_folder, mixed, tmp_path, capsys):
    out = tmp_path / "split"
    assert split(gpl_folder, mixed, out, "--template", "1", "--min-relative-reduction", "0.9") == 1
    summary = read_json(out / "summary.json")
    assert summary["accepted"] is False and summary["atoms"] == [] and summary["new_template_ids"] == []
    assert not (out / "matchings.json").exists()
    assert "split rejected" in capsys.readouterr().err


def test_too_few_patches_for_two_sides_is_rejected(gpl_folder, mixed, tmp_path):
    assert split(gpl_folder, mixed, tmp_path / "split", "--template", "0", "--min-size", "50") == 1


def test_no_refine_rounds_still_splits(gpl_folder, mixed, tmp_path):
    # The first assignment from the ± starts already separates the two motifs.
    assert split(gpl_folder, mixed, tmp_path / "split", "--template", "0", "--max-refine", "0") == 0


def test_registers_in_a_catalog_made_before_split(gpl_folder, mixed, tmp_path):
    db = tmp_path / "catalog.sqlite"
    with closing(catalog.connect(db)) as conn, conn:
        conn.execute("INSERT INTO tool_runs (tool, out_dir, created_at) VALUES ('randomized_build', 'elsewhere', 'now')")
    downgrade_tool_runs(db)
    out = tmp_path / "split"
    assert split(gpl_folder, mixed, out, "--template", "0") == 0
    result = run_module("database", "register", "--db", str(db), str(out))
    assert result.returncode == 0, result.stderr
    with closing(catalog.connect(db, read_only=True)) as conn:
        assert sorted(r["tool"] for r in catalog.list_tool_runs(conn)) == ["randomized_build", "randomized_build", "split"]


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--template", "7"], "template 7 not in"),
        (["--template", "0", "--epsilon", "-1"], "epsilon must be >= 0"),
        (["--template", "0", "--n-trials", "0"], "n_trials must be >= 1"),
        (["--template", "0", "--max-refine", "-1"], "max_refine must be >= 0"),
        (["--template", "0", "--min-size", "0"], "min_size must be >= 1"),
        (["--template", "0", "--min-relative-reduction", "2"], "min_relative_reduction must be in [0, 1]"),
    ],
)
def test_bad_input_exit_2(gpl_folder, mixed, tmp_path, capsys, extra, message):
    assert split(gpl_folder, mixed, tmp_path / "split", *extra) == 2
    assert message in capsys.readouterr().err


def test_inputs_that_do_not_fit_exit_2(gpl_folder, mixed, tmp_path, capsys):
    out = tmp_path / "split"
    # No dictionary there.
    assert split(gpl_folder, tmp_path / "nowhere", out, "--template", "0") == 2
    assert "no summary.json" in capsys.readouterr().err
    # A matching-pursuit folder of another dictionary.
    other = mixed_dictionary(gpl_folder, tmp_path / "other")
    mp = tmp_path / "mp_other"
    assert pursue(gpl_folder.root, other, mp) == 0
    assert split(gpl_folder, mixed, out, "--template", "0", "--matchings", str(mp)) == 2
    assert "is not a matching-pursuit run of" in capsys.readouterr().err
    # A folder with no matchings.json.
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "summary.json").write_text(json.dumps({"dictionary": str(mixed.resolve())}))
    assert split(gpl_folder, mixed, out, "--template", "0", "--matchings", str(empty)) == 2
    assert "no matchings.json" in capsys.readouterr().err
    # Placements in windows the GPL folder does not have.
    rows = read_json(mixed / "matchings.json")
    rows[0]["window_id"] = "spwhiten_elsewhere_w9"
    (mixed / "matchings.json").write_text(json.dumps(rows))
    assert split(gpl_folder, mixed, out, "--template", "0") == 2
    assert "placement windows are not in" in capsys.readouterr().err
    # An atom with no placements at all.
    (mixed / "matchings.json").write_text(json.dumps([r for r in rows[1:] if r["template_id"] != 1]))
    assert split(gpl_folder, mixed, out, "--template", "1") == 2
    assert "has no placements" in capsys.readouterr().err


def test_atoms_with_other_frequency_rows_exit_2(gpl_folder, mixed, tmp_path, capsys):
    np.save(mixed / "atoms/template_001.npy", np.ones((5, 12)))
    assert split(gpl_folder, mixed, tmp_path / "split", "--template", "0") == 2
    assert "different frequency rows" in capsys.readouterr().err


def test_python_dash_m_entry_point(gpl_folder, mixed, tmp_path):
    out = tmp_path / "split"
    result = run_module("analysis.dict_learning", "split", "--gpl-dir", str(gpl_folder.root), "--dictionary",
                        str(mixed), "--template", "0", "--out", str(out), "--seed", "0")
    assert result.returncode == 0, result.stderr
    assert read_json(out / "summary.json")["accepted"] is True
