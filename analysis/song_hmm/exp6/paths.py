"""Paths for Exp6 — writes only under exp6/work; reads Exp5 work as inputs."""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXP6 = Path(__file__).resolve().parent
WORK = EXP6 / "work"
EXPORTS = WORK / "exports"
REPORTS = EXP6.parents[0] / "reports"

# Exp5 artifacts (read-only)
EXP5 = EXP6.parent / "exp5"
EXP5_WORK = EXP5 / "work"
EXP5_SEQUENCES = EXP5_WORK / "sequences"
EXP5_MP = EXP5_WORK / "mp"
EXP5_DICT = EXP5_WORK / "dict" / "train_w336_s0_noov"
EXP5_FIGURES_30MIN = EXP5_WORK / "figures" / "motifs_by_30min"
EXP5_SUMMARY = EXP5_WORK / "results" / "summary.json"
EXP5_EVAL_A = EXP5_WORK / "results" / "A" / "eval.json"

WAV_STEM = "SOCAL_H_76_230426_193500_df100.x.x"
MOTIF_SEGMENTS_NAME = f"{WAV_STEM}.motif_segments.csv"
PLACEMENTS_CSV_NAME = f"placements_{WAV_STEM}.csv"


def ensure_work_dirs() -> None:
    EXPORTS.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
