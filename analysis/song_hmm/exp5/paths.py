"""Paths and 7 h / 3 h split for Experiment 5 (193500 only)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXP5 = Path(__file__).resolve().parent
WORK = EXP5 / "work"
REFERENCE = EXP5 / "reference"
NOTEBOOKS = EXP5 / "notebooks"
REPORTS = EXP5.parents[0] / "reports"

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

GAP_S = (1.5, 3.0)
GAP_L = (3.0, 60.0)
BREAK_S = 60.0


@dataclass(frozen=True)
class FileSpec:
    file_id: str
    gpl_dir: Path
    window_offset: int
    n_windows: int
    role: str  # train | test
    time_offset_s: float = 0.0


def hour_file(hour: int) -> FileSpec:
    """1-based hour index 1..10."""
    if not 1 <= hour <= 10:
        raise ValueError(hour)
    role = "train" if hour <= TRAIN_HOURS else "test"
    return FileSpec(
        file_id=f"193500_h{hour:02d}",
        gpl_dir=GPL_193500,
        window_offset=(hour - 1) * WINDOWS_PER_HOUR,
        n_windows=WINDOWS_PER_HOUR,
        role=role,
        time_offset_s=float((hour - 1) * 3600),
    )


def all_hour_files() -> list[FileSpec]:
    return [hour_file(h) for h in range(1, 11)]


def train_hour_files() -> list[FileSpec]:
    return [hour_file(h) for h in range(1, TRAIN_HOURS + 1)]


def test_hour_files() -> list[FileSpec]:
    return [hour_file(h) for h in range(TRAIN_HOURS + 1, TRAIN_HOURS + TEST_HOURS + 1)]


def ensure_work_dirs() -> None:
    for sub in (
        WORK,
        WORK / "dict",
        WORK / "mp",
        WORK / "sequences",
        WORK / "features",
        WORK / "models" / "A",
        WORK / "models" / "B",
        WORK / "results" / "A",
        WORK / "results" / "B",
        WORK / "figures" / "A",
        WORK / "figures" / "B",
        REFERENCE,
        NOTEBOOKS,
    ):
        sub.mkdir(parents=True, exist_ok=True)
