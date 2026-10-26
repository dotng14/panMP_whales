"""``python -m analysis.dict_learning matching-pursuit`` end to end."""

from __future__ import annotations

import json

import numpy as np
import pytest

from analysis.dict_learning.code.cli import main
from analysis.test.synthetic import read_json, run_module

BUILD = ["--m-per-outer", "3", "--n-min", "2", "--refine-iters", "1", "--seed", "0",
         "--workers", "1", "--max-outer", "2", "--min-coverage", "1.0"]


@pytest.fixture(scope="module")
def dictionary_dir(gpl_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("dict") / "rank1"
    assert main(["randomized-build", "--gpl-dir", str(gpl_dir), "--out", str(out), *BUILD]) == 0
    return out


@pytest.fixture(scope="module")
def subspace_dictionary_dir(gpl_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("dict") / "k2"
    args = ["randomized-build", "--gpl-dir", str(gpl_dir), "--out", str(out), *BUILD,
            "--n-components", "2"]
    assert main(args) == 0
    return out


def pursue(gpl_dir, dictionary, out, *extra):
    return main(["matching-pursuit", "--gpl-dir", str(gpl_dir), "--dictionary", str(dictionary),
                 "--out", str(out), "--workers", "1", *extra])


def test_places_atoms_and_reports_energy(gpl_dir, dictionary_dir, tmp_path, capsys):
    out = tmp_path / "mp"
    assert pursue(gpl_dir, dictionary_dir, out, "--mp-mode", "global", "--loop-stats") == 0
    summary = read_json(out / "summary.json")
    assert summary["n_atoms"] == read_json(dictionary_dir / "summary.json")["n_atoms"]
    assert summary["energy_after"] < summary["energy_before"]
    assert 0.0 < summary["time_coverage_fraction"] <= 1.0
    assert summary["n_windows_total"] == 3
    assert len(read_json(out / "matchings.json")) == summary["n_matchings"] > 0
    assert (out / "loop_stats.json").is_file()
    assert '"n_matchings"' in capsys.readouterr().out


def test_per_window_pool_with_limits(gpl_dir, dictionary_dir, tmp_path):
    out = tmp_path / "mp"
    code = pursue(gpl_dir, dictionary_dir, out, "--mp-mode", "per_window", "--workers", "2",
                  "--max-repeats", "2", "--disallow-overlap", "--max-relative-distance", "0.9",
                  "--progress")
    assert code == 0
    summary = read_json(out / "summary.json")
    assert summary["n_matchings"] <= 2 * summary["n_windows_total"]


def test_subspace_dictionary(gpl_dir, subspace_dictionary_dir, tmp_path):
    out = tmp_path / "mp"
    assert pursue(gpl_dir, subspace_dictionary_dir, out, "--eps-step-fraction", "0.001") == 0
    matchings = read_json(out / "matchings.json")
    assert matchings and all(len(m["coeffs"]) == 2 for m in matchings)
    gated = tmp_path / "gated"
    assert pursue(gpl_dir, subspace_dictionary_dir, gated, "--max-relative-distance", "0.9") == 0
    strict = tmp_path / "strict"
    assert pursue(gpl_dir, subspace_dictionary_dir, strict, "--max-relative-distance", "0") == 0
    assert read_json(strict / "summary.json")["n_matchings"] == 0


@pytest.mark.parametrize("mode", ["global", "per_window"])
def test_gate_that_rejects_everything(gpl_dir, dictionary_dir, tmp_path, mode):
    out = tmp_path / "mp"
    assert pursue(gpl_dir, dictionary_dir, out, "--mp-mode", mode, "--max-relative-distance", "0") == 0
    summary = read_json(out / "summary.json")
    assert summary["n_matchings"] == 0
    assert summary["stop_reason"] == "no_improvement"


def test_per_window_runs_until_nothing_qualifies(gpl_dir, dictionary_dir, tmp_path):
    out = tmp_path / "mp"
    assert pursue(gpl_dir, dictionary_dir, out, "--mp-mode", "per_window",
                  "--max-relative-distance", "0.5", "--eps-step-fraction", "1e-12") == 0
    summary = read_json(out / "summary.json")
    assert summary["n_matchings"] > 0
    assert summary["stop_reason"] == "completed"


def test_global_mode_limits(gpl_dir, dictionary_dir, tmp_path):
    out = tmp_path / "mp"
    assert pursue(gpl_dir, dictionary_dir, out, "--mp-mode", "global", "--disallow-overlap",
                  "--max-relative-distance", "0.5", "--eps-step-fraction", "1e-12") == 0
    assert read_json(out / "summary.json")["stop_reason"] == "no_improvement"
    capped = tmp_path / "capped"
    assert pursue(gpl_dir, dictionary_dir, capped, "--mp-mode", "global", "--max-repeats", "3") == 0
    summary = read_json(capped / "summary.json")
    assert summary["stop_reason"] == "max_repeats"
    assert summary["n_matchings"] == 3


def test_window_slice(gpl_dir, dictionary_dir, tmp_path):
    out = tmp_path / "mp"
    assert pursue(gpl_dir, dictionary_dir, out, "--n-windows", "1", "--window-offset", "2") == 0
    assert read_json(out / "summary.json")["n_windows_total"] == 1


def _copy_dictionary(src, dst, mutate):
    dst.mkdir(parents=True)
    summary = read_json(src / "summary.json")
    (dst / "atoms").mkdir()
    for atom in summary["atoms"]:
        (dst / atom["path"]).write_bytes((src / atom["path"]).read_bytes())
    mutate(summary, dst)
    (dst / "summary.json").write_text(json.dumps(summary))
    return dst


def test_dictionary_problems_exit_2(gpl_dir, dictionary_dir, tmp_path, capsys):
    cases = {
        "no summary": (tmp_path / "nothing", "no summary.json"),
        "no atoms": (
            _copy_dictionary(dictionary_dir, tmp_path / "noatoms",
                             lambda s, d: s.update(atoms=[])),
            "lists no atoms",
        ),
        "missing file": (
            _copy_dictionary(dictionary_dir, tmp_path / "missing",
                             lambda s, d: (d / s["atoms"][0]["path"]).unlink()),
            "atom file missing",
        ),
        "wrong rows": (
            _copy_dictionary(dictionary_dir, tmp_path / "rows",
                             lambda s, d: np.save(d / s["atoms"][0]["path"], np.ones((3, 12)))),
            "frequency rows",
        ),
    }
    (tmp_path / "nothing").mkdir()
    for name, (directory, message) in cases.items():
        assert pursue(gpl_dir, directory, tmp_path / name) == 2, name
        assert message in capsys.readouterr().err, name


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--max-relative-distance", "2"], "must be in [0, 1]"),
        (["--eps-step-fraction", "0"], "eps_step_fraction must be > 0"),
        (["--max-repeats", "0"], "max_repeats must be >= 1"),
    ],
)
def test_bad_arguments_exit_2(gpl_dir, dictionary_dir, tmp_path, capsys, extra, message):
    assert pursue(gpl_dir, dictionary_dir, tmp_path / "mp", *extra) == 2
    assert message in capsys.readouterr().err


def test_python_dash_m_entry_point(gpl_dir, dictionary_dir, tmp_path):
    out = tmp_path / "mp"
    proc = run_module("analysis.dict_learning", "matching-pursuit", "--gpl-dir", str(gpl_dir),
                      "--dictionary", str(dictionary_dir), "--out", str(out), "--workers", "1")
    assert proc.returncode == 0, proc.stderr
    assert (out / "summary.json").is_file()
