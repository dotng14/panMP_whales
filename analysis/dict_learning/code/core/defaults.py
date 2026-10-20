"""Shared pipeline defaults for averaged-template dictionary learning.

Callers (CLI, ``build_dictionary``, preprocess, notebooks) should import these
rather than hardcoding knobs. Function signatures on
``learn_averaged_templates_from_gpl`` keep their own defaults for backward
compatibility; argparse and production entrypoints use the values here.
"""

from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:
    """Positive int from the environment, falling back to ``default``."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


# Matches analysis/dict_learning/notebooks/seed_from_gpl_LeiData_setup.py (on-disk knobs)
# and the README worked example / repeat_builder tuning.
EPS_STEP_FRACTION = 0.01
MAX_ATOMS_PER_ROUND = 10  # legacy catalog/CLI alias; MP stops via eps_step_fraction
# Scope of a learning run, overridable via env (see backend/env.example).
# One outer round yields one template, so N_ROUNDS is the dictionary size.
# 30 windows / 10 templates: widened from the original 10/1 so later
# (residual-deflation) templates have enough underlying data to find more than
# one or two repeats each.
N_WINDOWS = _env_int("GPL_REPEATS_N_WINDOWS", 30)
N_ROUNDS = _env_int("GPL_REPEATS_N_ROUNDS", 10)
NORMALIZE = True
# Candidate GPL patch width (time bins); None = no filter.
MIN_TEMPLATE_WIDTH: int | None = None
MAX_TEMPLATE_WIDTH: int | None = 24
WINDOW_OFFSET = 0
# How each atom is rebuilt from its matched patches: "pca" (leading principal
# component / largest eigenvector) or "average" (element-wise mean).
TEMPLATE_METHOD = "pca"
# Number of PCA modes per atom. ``1`` is rank-1 MP (catalog-compatible).
# ``k>1`` uses subspace MP; builds persist to a manifest only (no catalog schema).
N_COMPONENTS = 1
# Closeness gate shared by MP acceptance and template rebuilds: reject a patch
# when more than this fraction of its energy is left unexplained by the atom
# (0 = exact match only, 1 = accept everything, None = gate off).
MAX_RELATIVE_DISTANCE: float | None = None
# Optional multi-atom refine after outer seed→average rounds (build_dictionary).
REFINE_ITERS = 0
# Extend-dictionary stage: stop adding atoms once time-bin coverage reaches this.
MIN_COVERAGE = 0.9
# Minimum matchings required in both the seed trial and the averaged re-MP.
MIN_REPEATS = 3
# Optional Mallat cap: stop after this many accepted matchings (None = no cap).
# In per_window mode the cap applies per window.
MAX_REPEATS: int | None = None
# Seed candidate window policy for learn_averaged_templates_from_gpl:
#   "global" — trial every residual GPL candidate (legacy)
#   "low_coverage" — seed from least-covered window (+ neighborhood); match on full stack
SEED_WINDOW_POLICY = "global"
SEED_WINDOW_NEIGHBORHOOD = 0
# Incremental builder: initial span; max span is caller-required (None = use pool size).
INITIAL_N_WINDOWS = N_WINDOWS
MAX_N_WINDOWS: int | None = None
# Incremental builder candidate width band (time bins).
INCREMENTAL_MIN_TEMPLATE_WIDTH: int | None = 12
INCREMENTAL_MAX_TEMPLATE_WIDTH: int | None = 24

# Matching Pursuit parallelism / mode / compute.
# Override with DICT_LEARNING_MP_WORKERS / DICT_LEARNING_MP_MODE /
# DICT_LEARNING_MP_COMPUTE (or GPL_COMPUTE=cuda), or CLI flags.
def _default_mp_workers() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return max(1, int(os.cpu_count() or 1))


MP_WORKERS = _default_mp_workers()
# per_window: independent Mallat per window (default). global: classic stack Mallat.
MP_MODE = "per_window"
# auto: CUDA when CuPy works and problem is large enough; else CPU (Numba/NumPy).
MP_COMPUTE = "auto"
