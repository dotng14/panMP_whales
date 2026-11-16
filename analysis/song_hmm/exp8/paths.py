"""Paths for Exp8 theme–slot HMM on Exp7 wav23 dictionaries."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXP8 = Path(__file__).resolve().parent
EXP7 = EXP8.parent / "exp7"
WORK = EXP8 / "work"
ARMS = WORK / "arms"
NOTEBOOKS = EXP8 / "notebooks"
REPORTS = EXP8.parents[0] / "reports"

GPL_WAV23 = Path("/home/yfreund/whale_processing/result/LeiData/wav23.gpl/output")

ARM_IDS = ("R_defaults", "R_defaults_cov04")
TRAIN_T1_S = 3600.0  # 1 h train / rest test
DURATION_S = 4500.0
N_FIVE_MIN_PANELS = 15
WINDOWS_PER_FIVE_MIN = 4
JS_W_S = 60.0
JS_N_TRIALS = 50

PY = "/opt/conda/bin/python"


def arm_dir(run_id: str) -> Path:
    return ARMS / run_id


def exp7_run_dir(run_id: str) -> Path:
    return EXP7 / "work" / "runs" / run_id


def ensure_work_dirs() -> None:
    for sub in (WORK, ARMS, NOTEBOOKS, WORK / "figures"):
        sub.mkdir(parents=True, exist_ok=True)
    for run_id in ARM_IDS:
        d = arm_dir(run_id)
        for sub in (
            d,
            d / "sequences",
            d / "models",
            d / "results",
            d / "figures" / "motifs_by_5min",
            d / "figures" / "themes_by_5min",
        ):
            sub.mkdir(parents=True, exist_ok=True)
