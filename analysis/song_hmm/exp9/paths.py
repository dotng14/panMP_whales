"""Paths for Exp9 many-theme HMM (self-loops) on Exp7/Exp8 wav23 arms."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXP9 = Path(__file__).resolve().parent
EXP8 = EXP9.parent / "exp8"
EXP7 = EXP9.parent / "exp7"
WORK = EXP9 / "work"
ARMS = WORK / "arms"
NOTEBOOKS = EXP9 / "notebooks"
REPORTS = EXP9.parents[0] / "reports"

GPL_WAV23 = Path("/home/yfreund/whale_processing/result/LeiData/wav23.gpl/output")

ARM_IDS = ("R_defaults", "R_defaults_cov04")
TRAIN_T1_S = 3600.0
JS_W_S = 60.0
JS_N_TRIALS = 50
L_MAX = 10
THEME_GRID = (8, 10, 12, 14, 16)

PY = "/opt/conda/bin/python"


def arm_dir(run_id: str) -> Path:
    return ARMS / run_id


def exp8_arm_dir(run_id: str) -> Path:
    return EXP8 / "work" / "arms" / run_id


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
