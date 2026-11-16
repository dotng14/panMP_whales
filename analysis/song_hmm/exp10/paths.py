"""Paths for Exp10: R_defaults pipeline on 193500."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXP10 = Path(__file__).resolve().parent
WORK = EXP10 / "work"
NOTEBOOKS = EXP10 / "notebooks"
REPORTS = EXP10.parents[0] / "reports"

LEIDATA = Path("/home/yfreund/whale_processing/LeiData")
GPL_ROOT = Path("/home/yfreund/whale_processing/result/LeiData")

WAV_193500 = LEIDATA / "SOCAL_H_76_230426_193500_df100.x.x.wav"
GPL_193500 = GPL_ROOT / "SOCAL_H_76_230426_193500_df100.x.x.gpl" / "output"

N_WINDOWS = 480
WINDOWS_PER_HOUR = 48
TRAIN_HOURS = 7
TEST_HOURS = 3
TRAIN_N_WINDOWS = TRAIN_HOURS * WINDOWS_PER_HOUR  # 336
TEST_N_WINDOWS = TEST_HOURS * WINDOWS_PER_HOUR  # 144
TRAIN_T0_S = 0.0
TRAIN_T1_S = float(TRAIN_HOURS * 3600)  # 25200
TEST_T0_S = TRAIN_T1_S
TEST_T1_S = float((TRAIN_HOURS + TEST_HOURS) * 3600)  # 36000

WINDOWS_PER_FIVE_MIN = 4  # ~75 s windows
N_TEST_FIVE_MIN_PANELS = TEST_HOURS * 12  # 36
N_TRAIN_SAMPLE_PANELS = 6

JS_W_S = 60.0
JS_N_TRIALS = 50

PY = "/opt/conda/bin/python"

# R_defaults knobs (Exp7)
R_DEFAULTS = {
    "run_id": "R_defaults",
    "min_coverage": 0.2,
    "max_outer": 10,
    "disallow_overlap": False,
    "n_min": 10,
    "epsilon": 0.9,
}


def hour_file_id(hour: int) -> str:
    return f"193500_h{hour:02d}"


def ensure_work_dirs() -> None:
    for sub in (
        WORK,
        WORK / "dict",
        WORK / "mp",
        WORK / "sequences",
        WORK / "models",
        WORK / "results",
        WORK / "figures" / "motifs_by_5min",
        WORK / "figures" / "themes_by_5min",
        NOTEBOOKS,
    ):
        sub.mkdir(parents=True, exist_ok=True)
