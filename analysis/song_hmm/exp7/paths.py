"""Paths for Exp7 dictionary evaluation on wav23."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXP7 = Path(__file__).resolve().parent
WORK = EXP7 / "work"
RUNS = WORK / "runs"
EXPORTS = WORK / "exports"
NOTEBOOKS = EXP7 / "notebooks"
REPORTS = EXP7.parents[0] / "reports"

LEIDATA = Path("/home/yfreund/whale_processing/LeiData")
GPL_ROOT = Path("/home/yfreund/whale_processing/result/LeiData")

WAV_WAV23 = LEIDATA / "wav23.wav"
GPL_WAV23 = GPL_ROOT / "wav23.gpl" / "output"

N_WINDOWS = 60
DURATION_S = 4500.0  # 1.25 h
WINDOWS_PER_FIVE_MIN = 4  # ~75 s windows
N_FIVE_MIN_PANELS = 15  # 4500 / 300

PY = "/opt/conda/bin/python"


def run_dir(run_id: str) -> Path:
    return RUNS / run_id


def ensure_work_dirs() -> None:
    for sub in (WORK, RUNS, EXPORTS, NOTEBOOKS, WORK / "figures"):
        sub.mkdir(parents=True, exist_ok=True)
