"""Setup for seed_from_gpl_LeiData.ipynb: path discovery and plot knobs.

Learner knobs live in :mod:`analysis.dict_learning.code.core.defaults` and are exposed by
``python -m analysis.dict_learning.code.learn.learn_dictionary``.
"""

from pathlib import Path
import os
import sys

try:
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt

import numpy as np


def find_repo() -> Path:
    """Locate repo root containing analysis/dict_learning (cwd-independent)."""
    here = Path.cwd().resolve()
    for cand in [here, *here.parents]:
        if (cand / "analysis" / "dict_learning" / "code" / "core" / "objects.py").is_file():
            return cand
        if (cand / "analysis" / "dict_learning" / "code" / "learn" / "seed_from_gpl.py").is_file():
            return cand
    raise FileNotFoundError(
        "Could not find gplrev3-webserver root from cwd=" + str(here)
    )


REPO = find_repo()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from analysis.dict_learning.code.core import defaults
from analysis.dict_learning.code.io.io_gpl import (
    load_window,
    load_detections_csv,
    detection_overlaps_window,
)
from analysis.dict_learning.code.learn.seed_from_gpl import detections_to_segments, mean_center
from analysis.dict_learning.code.learn.learn_dictionary import learn_averaged_templates_from_gpl

_env_roots = [Path(p.strip()) for p in os.environ.get("WAV_LIBRARY_ROOTS", "").split(",") if p.strip()]
_default_roots = [
    Path("/home/yfreund/whale_processing"),
    Path.home() / "whale_processing",
    Path.home() / "projects" / "whale_processing",
]
LIBRARY_ROOTS = _env_roots or _default_roots

LEI_DATA_ROOT = next(
    (
        root / "result" / "LeiData"
        for root in LIBRARY_ROOTS
        if (root / "result" / "LeiData").is_dir()
    ),
    None,
)
if LEI_DATA_ROOT is None:
    raise FileNotFoundError(
        "No LeiData root found under LIBRARY_ROOTS. "
        "Set WAV_LIBRARY_ROOTS to your whale_processing root on this machine."
    )

LEI_GPL_DIRS = sorted(
    p
    for p in LEI_DATA_ROOT.glob("*.gpl/output")
    if (p / "spectrogram_npy").is_dir()
)
if not LEI_GPL_DIRS:
    raise FileNotFoundError(f"No *.gpl/output dirs with spectrogram_npy under {LEI_DATA_ROOT}")

# Shared learner knobs (see analysis.dict_learning.code.core.defaults)
EPS_STEP_FRACTION = defaults.EPS_STEP_FRACTION
MIN_REPEATS = defaults.MIN_REPEATS
N_WINDOWS = defaults.N_WINDOWS
N_OUTER_ITERS = defaults.N_ROUNDS

# Notebook-only
MAX_TEMPLATE_WIDTH = 24
N_PLOT_WINDOWS = 2
N_PLOT_ROUNDS = 2
TEMPLATE_GRID = (10, 10)

TEMPLATE_OUT_DIR = LEI_DATA_ROOT / "dict_learning_templates" / f"n{N_WINDOWS}_r{N_OUTER_ITERS or 'dyn'}"

print(f"Python: {sys.executable}")
print(f"Repo: {REPO}")
print(f"LEI_DATA_ROOT: {LEI_DATA_ROOT}")
print(f"LeiData GPL dirs: {len(LEI_GPL_DIRS)}")
for d in LEI_GPL_DIRS:
    print(f"  {d.parent.name}")
print(f"N_WINDOWS={N_WINDOWS}  N_OUTER_ITERS={N_OUTER_ITERS} MIN_REPEATS={MIN_REPEATS} (from defaults)")
print(f"TEMPLATE_OUT_DIR: {TEMPLATE_OUT_DIR}")
