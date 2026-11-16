"""Exp6: export aa32a7f-compatible CSVs from Exp5 Approach A artifacts (read-only)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import paths
from .export_csvs import (
    assign_song_theme,
    load_all_exp5_sessions,
    load_eval_a_themes,
    theme_intervals_from_eval,
    tokens_to_placements,
    wav_start_unix,
    write_motif_segments_csv,
    write_placements_csv,
    write_placements_txt,
    write_segments_csv,
    write_segments_motifs_only,
)
from .paths import (
    EXP5_DICT,
    EXP5_EVAL_A,
    EXP5_FIGURES_30MIN,
    EXP5_MP,
    EXP5_SEQUENCES,
    EXP5_SUMMARY,
    EXP5_WORK,
    EXPORTS,
    MOTIF_SEGMENTS_NAME,
    PLACEMENTS_CSV_NAME,
    WAV_STEM,
    WORK,
    ensure_work_dirs,
)


def stage0() -> dict:
    ensure_work_dirs()
    n_panels = len(list(EXP5_FIGURES_30MIN.glob("panel_*.png"))) if EXP5_FIGURES_30MIN.is_dir() else 0
    checks = {
        "exp5_work": str(EXP5_WORK),
        "dict_summary": str(EXP5_DICT / "summary.json"),
        "dict_ok": (EXP5_DICT / "summary.json").is_file(),
        "dict_name": EXP5_DICT.name,
        "sequences_dir": str(EXP5_SEQUENCES),
        "n_sequence_files": len(list(EXP5_SEQUENCES.glob("193500_h*.json"))),
        "eval_a": str(EXP5_EVAL_A),
        "eval_a_ok": EXP5_EVAL_A.is_file(),
        "summary": str(EXP5_SUMMARY),
        "summary_ok": EXP5_SUMMARY.is_file(),
        "mp_dir": str(EXP5_MP),
        "figures_30min": str(EXP5_FIGURES_30MIN),
        "n_30min_panels": n_panels,
        "wav_stem": WAV_STEM,
        "origin_unix": wav_start_unix(WAV_STEM),
    }
    if not checks["dict_ok"]:
        raise FileNotFoundError(f"missing Exp5 dictionary summary: {EXP5_DICT / 'summary.json'}")
    if checks["n_sequence_files"] < 1:
        raise FileNotFoundError(f"no Exp5 sequences under {EXP5_SEQUENCES}")
    if not checks["eval_a_ok"]:
        raise FileNotFoundError(f"missing Exp5 Approach A eval: {EXP5_EVAL_A}")
    (WORK / "file_manifest.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))
    return checks


def stage_export_placements() -> dict:
    ensure_work_dirs()
    sessions = load_all_exp5_sessions(EXP5_SEQUENCES)
    placements = tokens_to_placements(sessions)
    origin = wav_start_unix(WAV_STEM)

    txt = EXPORTS / "placements.txt"
    csv_path = EXPORTS / PLACEMENTS_CSV_NAME
    motifs_only = EXPORTS / "segments_motifs_only.csv"
    write_placements_txt(txt, placements)
    write_placements_csv(csv_path, placements)
    write_segments_motifs_only(motifs_only, origin_unix=origin, placements=placements)

    summary = {
        "n_sessions": len(sessions),
        "n_placements": len(placements),
        "origin_unix": origin,
        "wrote": [str(txt), str(csv_path), str(motifs_only)],
    }
    (WORK / "export_placements_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def stage_export_structure() -> dict:
    ensure_work_dirs()
    sessions = load_all_exp5_sessions(EXP5_SEQUENCES)
    placements = tokens_to_placements(sessions)
    origin = wav_start_unix(WAV_STEM)
    theme_dicts = load_eval_a_themes(EXP5_EVAL_A)
    themes = theme_intervals_from_eval(theme_dicts)
    rows, songs = assign_song_theme(placements, themes, origin_unix=origin)

    motif_csv = EXPORTS / MOTIF_SEGMENTS_NAME
    segments_csv = EXPORTS / "segments.csv"
    write_motif_segments_csv(motif_csv, rows)
    # Motifs that fall inside Approach A decoded theme spans (test hours)
    structured_starts = {round(r.start_unix - origin, 6) for r in rows}
    motifs_in_structure = [
        p for p in placements if round(p.start_sec, 6) in structured_starts
    ]
    write_segments_csv(
        segments_csv,
        origin_unix=origin,
        motifs=motifs_in_structure,
        themes=themes,
        songs=songs,
    )

    summary = {
        "n_placements_all": len(placements),
        "n_motif_segment_rows": len(rows),
        "n_theme_visits": len(themes),
        "n_songs": len(songs),
        "origin_unix": origin,
        "note": "Approach A decode covers test hours only; motif_segments rows are placements inside those theme spans",
        "wrote": [str(motif_csv), str(segments_csv)],
    }
    (WORK / "export_structure_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage",
        choices=["0", "export-placements", "export-structure", "export", "all"],
    )
    args = p.parse_args(argv)

    if args.stage in ("0", "all", "export", "export-placements", "export-structure"):
        if args.stage == "0" or args.stage == "all":
            stage0()
        elif args.stage in ("export", "export-placements", "export-structure"):
            # still verify inputs
            stage0()

    if args.stage in ("export-placements", "export", "all"):
        stage_export_placements()
    if args.stage in ("export-structure", "export", "all"):
        stage_export_structure()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
