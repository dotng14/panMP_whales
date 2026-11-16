"""Paths and per-wav half-split manifests for Experiment 4."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXP4 = Path(__file__).resolve().parent
WORK = EXP4 / "work"
REFERENCE = EXP4 / "reference"
NOTEBOOKS = EXP4 / "notebooks"

LEIDATA = Path("/home/yfreund/whale_processing/LeiData")
GPL_ROOT = Path("/home/yfreund/whale_processing/result/LeiData")

WAV_193500 = LEIDATA / "SOCAL_H_76_230426_193500_df100.x.x.wav"
WAV_014845 = LEIDATA / "SOCAL_H_76_230427_014845.x.wav"
WAV_014845_10KHZ = LEIDATA / "SOCAL_H_76_230427_014845.x_10kHz.wav"
WAV_014845_2KHZ = WORK / "audio" / "SOCAL_H_76_230427_014845.x_2kHz.wav"

GPL_193500 = GPL_ROOT / "SOCAL_H_76_230426_193500_df100.x.x.gpl" / "output"
GPL_014845 = WORK / "gpl_014845_2kHz"

N_WINDOWS_193500 = 480
WINDOWS_PER_HOUR = 48
N_HOURS = 10
TRAIN_HOURS = range(1, 6)
TEST_HOURS = range(6, 11)

OFFSET_014845_IN_193500_S = 6 * 3600 + 13 * 60 + 45

GAP_S = (1.5, 3.0)
GAP_L = (3.0, 60.0)
BREAK_S = 60.0

WAV_IDS = ("193500", "014845")


@dataclass(frozen=True)
class FileSpec:
    """One processing unit (1-hour slice or a half of 014845)."""

    file_id: str
    wav_id: str
    gpl_dir: Path
    window_offset: int
    n_windows: int
    role: str  # train | eval
    time_offset_s: float = 0.0


@dataclass(frozen=True)
class WavSplit:
    """Train / eval half for one wav."""

    wav_id: str
    gpl_dir: Path
    train: list[FileSpec]
    eval: list[FileSpec]


def hour_file(hour: int) -> FileSpec:
    if not 1 <= hour <= N_HOURS:
        raise ValueError(f"hour must be 1..{N_HOURS}, got {hour}")
    role = "train" if hour in TRAIN_HOURS else "eval"
    return FileSpec(
        file_id=f"193500_h{hour:02d}",
        wav_id="193500",
        gpl_dir=GPL_193500,
        window_offset=(hour - 1) * WINDOWS_PER_HOUR,
        n_windows=WINDOWS_PER_HOUR,
        role=role,
        time_offset_s=float((hour - 1) * 3600),
    )


def all_hour_files() -> list[FileSpec]:
    return [hour_file(h) for h in range(1, N_HOURS + 1)]


def train_files() -> list[FileSpec]:
    return [hour_file(h) for h in TRAIN_HOURS]


def eval_files_193500() -> list[FileSpec]:
    return [hour_file(h) for h in TEST_HOURS]


def split_014845(n_windows: int | None = None) -> WavSplit:
    """Half / half by window count on the 2 kHz GPL."""
    if n_windows is None:
        # discover lazily: caller may pass after GPL exists
        n_windows = 30
    n_train = n_windows // 2
    n_eval = n_windows - n_train
    train = [
        FileSpec(
            file_id="014845_train",
            wav_id="014845",
            gpl_dir=GPL_014845,
            window_offset=0,
            n_windows=n_train,
            role="train",
            time_offset_s=0.0,
        )
    ]
    eval_specs = [
        FileSpec(
            file_id="014845_eval",
            wav_id="014845",
            gpl_dir=GPL_014845,
            window_offset=n_train,
            n_windows=n_eval,
            role="eval",
            time_offset_s=float(n_train * 75.0),
        )
    ]
    return WavSplit("014845", GPL_014845, train, eval_specs)


def split_193500() -> WavSplit:
    return WavSplit(
        "193500",
        GPL_193500,
        [hour_file(h) for h in TRAIN_HOURS],
        [hour_file(h) for h in TEST_HOURS],
    )


def results_dir(wav_id: str) -> Path:
    return WORK / "results" / wav_id


def ensure_work_dirs() -> None:
    for sub in (
        WORK,
        WORK / "audio",
        WORK / "dict",
        WORK / "mp",
        WORK / "sequences",
        WORK / "models",
        WORK / "results" / "193500",
        WORK / "results" / "014845",
        WORK / "figures" / "193500",
        WORK / "figures" / "014845",
        REFERENCE,
    ):
        sub.mkdir(parents=True, exist_ok=True)
