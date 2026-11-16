"""Randomized-build grid for Exp7 (wav23)."""

from __future__ import annotations

from typing import Any

# Each config: run_id + CLI kwargs for randomized-build (beyond gpl/out/windows/seed/workers)
GRID: list[dict[str, Any]] = [
    {
        "run_id": "R_defaults",
        "label": "Package defaults",
        "min_coverage": 0.2,
        "max_outer": 10,
        "disallow_overlap": False,
        "n_min": 10,
        "epsilon": 0.9,
    },
    {
        # Same knobs as R_defaults, but keep growing until higher coverage.
        "run_id": "R_defaults_hicov",
        "label": "Defaults + higher coverage target",
        "min_coverage": 0.85,
        "max_outer": 40,
        "disallow_overlap": False,
        "n_min": 10,
        "epsilon": 0.9,
    },
    {
        # Just clear coverage_lt_0.4; keep other defaults.
        "run_id": "R_defaults_cov04",
        "label": "Defaults + min_coverage=0.4",
        "min_coverage": 0.4,
        "max_outer": 20,
        "disallow_overlap": False,
        "n_min": 10,
        "epsilon": 0.9,
    },
    {
        "run_id": "R_exp6_like",
        "label": "Exp6-like knobs",
        "min_coverage": 0.95,
        "max_outer": 40,
        "disallow_overlap": True,
        "n_min": 10,
        "epsilon": 0.9,
        "m_per_outer": 5,
    },
    {
        "run_id": "R_fewer_atoms",
        "label": "Fewer atoms (n_min=25)",
        "min_coverage": 0.5,
        "max_outer": 20,
        "disallow_overlap": True,
        "n_min": 25,
        "epsilon": 0.9,
    },
    {
        "run_id": "R_strict_eps",
        "label": "Stricter create epsilon=0.95",
        "min_coverage": 0.5,
        "max_outer": 20,
        "disallow_overlap": True,
        "n_min": 10,
        "epsilon": 0.95,
    },
    {
        "run_id": "R_mod_overlap",
        "label": "Moderate coverage, overlap allowed",
        "min_coverage": 0.5,
        "max_outer": 15,
        "disallow_overlap": False,
        "n_min": 10,
        "epsilon": 0.9,
    },
    {
        "run_id": "R_mod_noov",
        "label": "Moderate coverage, no overlap",
        "min_coverage": 0.5,
        "max_outer": 15,
        "disallow_overlap": True,
        "n_min": 10,
        "epsilon": 0.9,
    },
    {
        "run_id": "R_narrow_w",
        "label": "Narrow widths 8–14",
        "min_coverage": 0.5,
        "max_outer": 15,
        "disallow_overlap": True,
        "n_min": 10,
        "epsilon": 0.9,
        "min_template_width": 8,
        "max_template_width": 14,
    },
    {
        "run_id": "R_wide_w",
        "label": "Wide widths 12–30",
        "min_coverage": 0.5,
        "max_outer": 15,
        "disallow_overlap": True,
        "n_min": 10,
        "epsilon": 0.9,
        "min_template_width": 12,
        "max_template_width": 30,
    },
]
