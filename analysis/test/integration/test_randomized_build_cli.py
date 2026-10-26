"""``python -m analysis.dict_learning randomized-build`` end to end."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.dict_learning.code import mp as mp_module
from analysis.dict_learning.code.cli import main
from analysis.test.synthetic import read_json, run_module

# Small, fast settings that still find the planted motifs.
FAST = ["--m-per-outer", "3", "--n-min", "2", "--refine-iters", "1", "--seed", "0", "--workers", "1"]


def build(gpl_dir, out, *extra):
    return main(["randomized-build", "--gpl-dir", str(gpl_dir), "--out", str(out), *FAST, *extra])


def test_writes_dictionary_in_report_layout(gpl_dir, tmp_path, capsys):
    out = tmp_path / "dict"
    code = build(gpl_dir, out, "--max-outer", "2", "--min-coverage", "1.0", "--mp-mode", "global",
                 "--verbose", "--loop-stats")
    assert code == 0
    summary = read_json(out / "summary.json")
    assert summary["stop_reason"] == "max_outer"
    assert summary["n_outer_completed"] == 2
    assert summary["n_windows"] == 3
    assert summary["sec_per_bin"] == pytest.approx(0.01)
    assert summary["n_atoms"] == len(summary["atoms"]) > 0
    for atom in summary["atoms"]:
        data = np.load(out / atom["path"])
        assert data.shape == (atom["n_freq"], atom["width"])
        assert atom["n_patches"] >= 2
    matchings = read_json(out / "matchings.json")
    assert len(matchings) == summary["n_matchings"]
    assert {"template_id", "window_id", "t_start", "amplitude", "energy"} <= set(matchings[0])
    logs = (out / "outer_logs.jsonl").read_text().splitlines()
    assert len(logs) == 2
    assert read_json(out / "loop_stats.json")["counts"]["mp.loop.global"] >= 1
    captured = capsys.readouterr()
    assert "outer 1:" in captured.err
    assert '"stop_reason": "max_outer"' in captured.out


def test_stops_at_coverage(gpl_dir, tmp_path):
    out = tmp_path / "dict"
    assert build(gpl_dir, out, "--min-coverage", "0.05", "--max-outer", "5") == 0
    summary = read_json(out / "summary.json")
    assert summary["stop_reason"] == "coverage"
    assert summary["time_coverage_fraction"] > 0.05


def test_stops_when_seed_pool_runs_out(gpl_dir, tmp_path):
    # Every crop must leave over 95% of its energy unexplained, so the pool
    # empties once the first pass explains the planted copies.
    out = tmp_path / "dict"
    assert build(gpl_dir, out, "--epsilon", "0.95", "--min-coverage", "1.0", "--max-outer", "20") == 0
    assert read_json(out / "summary.json")["stop_reason"] == "empty_pool"


def test_no_survivors_exits_1(gpl_dir, tmp_path, capsys):
    # Two refine iterations: the first prunes every atom, the second has none left.
    out = tmp_path / "dict"
    assert build(gpl_dir, out, "--n-min", "1000", "--refine-iters", "2") == 1
    summary = read_json(out / "summary.json")
    assert summary["stop_reason"] == "no_survivors"
    assert summary["atoms"] == []
    assert "no atoms survived" in capsys.readouterr().err


def test_strict_post_refine_gate_drops_every_placement(gpl_dir, tmp_path):
    out = tmp_path / "dict"
    assert build(gpl_dir, out, "--post-refine-max-relative-distance", "0") == 1
    summary = read_json(out / "summary.json")
    assert summary["stop_reason"] == "no_survivors"


def test_rebuild_gate_and_single_patch_atoms(gpl_dir, tmp_path):
    # --n-min 1 keeps atoms placed only once, so some are rebuilt from one patch.
    out = tmp_path / "dict"
    assert build(gpl_dir, out, "--max-relative-distance", "0.5", "--n-min", "1",
                 "--max-outer", "2", "--min-coverage", "1.0") == 0
    summary = read_json(out / "summary.json")
    assert summary["max_relative_distance"] == 0.5
    assert min(atom["n_patches"] for atom in summary["atoms"]) >= 1


def test_rebuild_gate_of_one_keeps_every_patch(gpl_dir, tmp_path):
    out = tmp_path / "dict"
    assert build(gpl_dir, out, "--max-relative-distance", "1", "--max-outer", "1") == 0


def test_missing_sample_rate_leaves_sec_per_bin_empty(gpl_variant, tmp_path):
    out = tmp_path / "dict"
    assert build(gpl_variant(sidecar_sample_freq=False), out, "--max-outer", "1") == 0
    assert read_json(out / "summary.json")["sec_per_bin"] is None


def test_subspace_atoms(gpl_dir, tmp_path):
    out = tmp_path / "dict"
    assert build(gpl_dir, out, "--n-components", "2", "--max-outer", "1") == 0
    summary = read_json(out / "summary.json")
    assert all(atom["n_components"] == 2 for atom in summary["atoms"])
    data = np.load(out / summary["atoms"][0]["path"])
    assert data.ndim == 3 and data.shape[0] == 2
    assert "coeffs" in read_json(out / "matchings.json")[0]


def test_gates_and_width_limits_off(gpl_dir, tmp_path):
    out = tmp_path / "dict"
    code = build(gpl_dir, out, "--max-outer", "1", "--post-refine-max-relative-distance", "none",
                 "--max-relative-distance", "none", "--min-template-width", "none",
                 "--max-template-width", "none")
    assert code == 0
    summary = read_json(out / "summary.json")
    assert summary["post_refine_max_relative_distance"] is None
    assert summary["min_template_width"] is None


def test_per_window_mode_with_worker_pool(gpl_dir, tmp_path):
    out = tmp_path / "dict"
    assert build(gpl_dir, out, "--max-outer", "1", "--mp-mode", "per_window", "--workers", "2") == 0
    assert read_json(out / "summary.json")["n_atoms"] > 0


def test_window_slice_and_progress(gpl_dir, tmp_path, capsys):
    out = tmp_path / "dict"
    assert build(gpl_dir, out, "--n-windows", "2", "--window-offset", "1", "--max-outer", "1",
                 "--progress") == 0
    summary = read_json(out / "summary.json")
    assert summary["n_windows"] == 2
    assert summary["window_offset"] == 1


def test_workers_and_mode_from_environment(gpl_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("DICT_LEARNING_MP_WORKERS", "2")
    monkeypatch.setenv("DICT_LEARNING_MP_MODE", "per_window")
    out = tmp_path / "dict"
    args = ["randomized-build", "--gpl-dir", str(gpl_dir), "--out", str(out),
            "--m-per-outer", "3", "--n-min", "2", "--refine-iters", "1", "--seed", "0",
            "--max-outer", "1"]
    assert main(args) == 0
    monkeypatch.delenv("DICT_LEARNING_MP_WORKERS")
    monkeypatch.setenv("DICT_LEARNING_MP_MODE", "global")
    assert main(args) == 0


def test_default_worker_count(gpl_dir, tmp_path, monkeypatch):
    monkeypatch.delenv("DICT_LEARNING_MP_WORKERS", raising=False)
    monkeypatch.delenv("DICT_LEARNING_MP_MODE", raising=False)
    out = tmp_path / "dict"
    args = ["randomized-build", "--gpl-dir", str(gpl_dir), "--out", str(out),
            "--m-per-outer", "3", "--n-min", "2", "--refine-iters", "1", "--seed", "0",
            "--max-outer", "1", "--mp-mode", "per_window"]
    assert main(args) == 0


@pytest.mark.parametrize(
    "name, value, message",
    [
        ("DICT_LEARNING_MP_MODE", "bogus", "mp_mode must be 'global' or 'per_window'"),
        ("DICT_LEARNING_MP_COMPUTE", "bogus", "mp_compute must be cpu|cuda|auto"),
    ],
)
def test_bad_environment_exit_2(gpl_dir, tmp_path, monkeypatch, capsys, name, value, message):
    monkeypatch.setenv(name, value)
    assert build(gpl_dir, tmp_path / "dict", "--max-outer", "1") == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("compute", ["cpu", "auto"])
def test_compute_backend(gpl_dir, tmp_path, monkeypatch, compute):
    monkeypatch.delenv("DICT_LEARNING_MP_COMPUTE", raising=False)
    out = tmp_path / "dict"
    assert build(gpl_dir, out, "--max-outer", "1", "--compute", compute) == 0


@pytest.mark.gpu
def test_cuda_backend(gpl_dir, tmp_path, monkeypatch):
    if not mp_module.cupy_available():
        pytest.skip("no usable CUDA device")
    monkeypatch.delenv("DICT_LEARNING_MP_COMPUTE", raising=False)
    out = tmp_path / "dict"
    assert build(gpl_dir, out, "--max-outer", "1", "--compute", "cuda") == 0
    # Subspace atoms score shifts without normalizing.
    assert build(gpl_dir, tmp_path / "k2", "--max-outer", "1", "--compute", "cuda",
                 "--n-components", "2") == 0


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--window-offset", "-1"], "--window-offset must be >= 0"),
        (["--n-windows", "0"], "--n-windows must be >= 1"),
        (["--window-offset", "9"], "past the last of 3 windows"),
        (["--max-relative-distance", "1.5"], "--max-relative-distance must be in [0, 1]"),
        (["--post-refine-max-relative-distance", "-0.1"], "must be in [0, 1]"),
        (["--m-per-outer", "0"], "m_per_outer must be >= 1"),
        (["--min-coverage", "0"], "min_coverage must be in (0, 1]"),
        (["--max-outer", "0"], "max_outer must be >= 1"),
        (["--n-min", "0"], "n_min must be >= 1"),
        (["--epsilon", "-1"], "epsilon must be >= 0"),
        (["--refine-iters", "-1"], "iter must be >= 0"),
        (["--n-components", "0"], "n_components must be >= 1"),
    ],
)
def test_bad_arguments_exit_2(gpl_dir, tmp_path, capsys, extra, message):
    assert build(gpl_dir, tmp_path / "dict", *extra) == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    "variant, message",
    [
        ({"detections": "none"}, "no CSV_*.csv detection file"),
        ({"detections": "empty"}, "has no rows"),
    ],
)
def test_bad_detections_exit_2(gpl_variant, tmp_path, capsys, variant, message):
    assert build(gpl_variant(**variant), tmp_path / "dict") == 2
    assert message in capsys.readouterr().err


def test_missing_or_empty_gpl_dir_exit_2(tmp_path, capsys):
    assert build(tmp_path / "missing", tmp_path / "dict") == 2
    assert "GPL directory not found" in capsys.readouterr().err
    empty = tmp_path / "empty"
    empty.mkdir()
    assert build(empty, tmp_path / "dict") == 2
    assert "no whitened spectrogram windows" in capsys.readouterr().err


def test_narrow_detections_give_no_atoms(gpl_variant, tmp_path):
    out = tmp_path / "dict"
    assert build(gpl_variant(detections="narrow"), out) == 1
    assert read_json(out / "summary.json")["stop_reason"] == "empty_pool"


def test_python_dash_m_entry_point(gpl_dir, tmp_path):
    out = tmp_path / "dict"
    proc = run_module("analysis.dict_learning", "randomized-build", "--gpl-dir", str(gpl_dir),
                      "--out", str(out), *FAST, "--max-outer", "1")
    assert proc.returncode == 0, proc.stderr
    assert (out / "summary.json").is_file()
