"""Exp7: dictionary evaluation gate on wav23."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from analysis.dict_learning.code.cli import load_windows
from analysis.song_hmm.exp4.sequence import (
    Event,
    Session,
    clean_events,
    energy_threshold,
    events_to_sessions,
    load_matchings,
    load_template_widths,
    placements_to_events,
    save_sequences,
)
from analysis.song_hmm.exp6.export_csvs import Placement, tokens_to_placements, write_placements_csv

from . import paths
from .grid_configs import GRID
from .metrics_dict import load_sessions, score_run, write_leaderboard
from .paths import (
    EXPORTS,
    GPL_WAV23,
    N_WINDOWS,
    PY,
    RUNS,
    WAV_WAV23,
    WORK,
    ensure_work_dirs,
    run_dir,
)
from .plot_panels import plot_five_min_panels


def _run(cmd: list[str]) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def stage0() -> dict:
    ensure_work_dirs()
    meta = {
        "wav": str(WAV_WAV23),
        "wav_ok": WAV_WAV23.is_file(),
        "gpl": str(GPL_WAV23),
        "gpl_ok": GPL_WAV23.is_dir(),
        "n_windows_expected": N_WINDOWS,
    }
    if not meta["wav_ok"]:
        raise FileNotFoundError(WAV_WAV23)
    if not meta["gpl_ok"]:
        raise FileNotFoundError(GPL_WAV23)
    wins = load_windows(GPL_WAV23, None, 0)
    meta["n_windows_found"] = len(wins)
    (WORK / "file_manifest.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return meta


def stage_inventory() -> None:
    ensure_work_dirs()
    src = Path(__file__).resolve().parent / "inventory.md"
    dst = WORK / "inventory.md"
    shutil.copy2(src, dst)
    print(f"copied inventory → {dst}")


def _build_randomized(cfg: dict[str, Any], *, force: bool = False) -> Path:
    rd = run_dir(cfg["run_id"])
    dict_out = rd / "dict"
    if (dict_out / "summary.json").is_file() and not force:
        print(f"dict exists: {dict_out}")
        return dict_out
    if force and dict_out.exists():
        shutil.rmtree(dict_out)
    dict_out.mkdir(parents=True, exist_ok=True)
    cmd = [
        PY, "-m", "analysis.dict_learning", "randomized-build",
        "--gpl-dir", str(GPL_WAV23),
        "--out", str(dict_out),
        "--n-windows", str(N_WINDOWS),
        "--window-offset", "0",
        "--seed", "0",
        "--workers", "8",
        "--verbose",
        "--loop-stats",
        "--min-coverage", str(cfg["min_coverage"]),
        "--max-outer", str(cfg["max_outer"]),
        "--n-min", str(cfg.get("n_min", 10)),
        "--epsilon", str(cfg.get("epsilon", 0.9)),
    ]
    if cfg.get("m_per_outer") is not None:
        cmd.extend(["--m-per-outer", str(cfg["m_per_outer"])])
    if cfg.get("min_template_width") is not None:
        cmd.extend(["--min-template-width", str(cfg["min_template_width"])])
    if cfg.get("max_template_width") is not None:
        cmd.extend(["--max-template-width", str(cfg["max_template_width"])])
    if cfg.get("disallow_overlap"):
        cmd.append("--disallow-overlap")
    t0 = time.time()
    _run(cmd)
    (rd / "dict_timing.json").write_text(
        json.dumps({"build_s": time.time() - t0, "knobs": cfg}, indent=2)
    )
    return dict_out


def _mp_place(dictionary: Path, rd: Path, *, disallow_overlap: bool, force: bool = False) -> Path:
    mp_out = rd / "mp"
    if (mp_out / "matchings.json").is_file() and not force:
        print(f"MP exists: {mp_out}")
        return mp_out
    if force and mp_out.exists():
        shutil.rmtree(mp_out)
    cmd = [
        PY, "-m", "analysis.dict_learning", "matching-pursuit",
        "--gpl-dir", str(GPL_WAV23),
        "--dictionary", str(dictionary),
        "--out", str(mp_out),
        "--n-windows", str(N_WINDOWS),
        "--window-offset", "0",
        "--workers", "8",
    ]
    if disallow_overlap:
        cmd.append("--disallow-overlap")
    t0 = time.time()
    _run(cmd)
    (mp_out / "timing.json").write_text(json.dumps({"mp_s": time.time() - t0}, indent=2))
    return mp_out


def _sequences_from_mp(dictionary: Path, mp_dir: Path, rd: Path) -> Path:
    seq_path = rd / "sequences" / "wav23.json"
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    widths = load_template_widths(dictionary)
    windows = load_windows(GPL_WAV23, N_WINDOWS, 0)
    events = placements_to_events(load_matchings(mp_dir / "matchings.json"), windows, widths)
    emin = energy_threshold(events)
    cleaned = clean_events(events, energy_min=emin)
    sessions = events_to_sessions(cleaned)
    save_sequences(
        seq_path,
        sessions,
        meta={
            "file_id": "wav23",
            "n_events": len(cleaned),
            "n_sessions": len(sessions),
            "energy_min": emin,
            "n_raw_events": len(events),
        },
    )
    (rd / "sequences" / "energy_min.json").write_text(json.dumps({"energy_min": emin}, indent=2))
    print(f"sequences: {len(cleaned)} events, {len(sessions)} sessions → {seq_path}")
    return seq_path


def stage_grid(*, force: bool = False, only: str | None = None) -> None:
    ensure_work_dirs()
    configs = GRID if only is None else [c for c in GRID if c["run_id"] == only]
    if only and not configs:
        raise SystemExit(f"unknown run_id {only!r}")
    for cfg in configs:
        print(f"\n=== GRID {cfg['run_id']}: {cfg['label']} ===", flush=True)
        rd = run_dir(cfg["run_id"])
        rd.mkdir(parents=True, exist_ok=True)
        (rd / "config.json").write_text(json.dumps(cfg, indent=2))
        dictionary = _build_randomized(cfg, force=force)
        mp = _mp_place(dictionary, rd, disallow_overlap=bool(cfg.get("disallow_overlap")), force=force)
        _sequences_from_mp(dictionary, mp, rd)


def _motifs_to_sessions(out_dir: Path) -> Path:
    """Convert learn_from_motifs matchings → sequence JSON."""
    summary = json.loads((out_dir / "summary.json").read_text())
    sec = float(summary.get("sec_per_bin") or 0.1)
    matchings = json.loads((out_dir / "matchings.json").read_text())
    # widths from templates on disk
    widths: dict[int, int] = {}
    for p in sorted(out_dir.glob("template_*.npy")):
        tid = int(p.stem.split("_")[1])
        import numpy as np

        arr = np.load(p)
        widths[tid] = int(arr.shape[-1])
    events: list[Event] = []
    for row in matchings:
        tid = int(row["template_id"])
        w = widths.get(tid, 10)
        t0 = float(row["t_start"]) * sec
        t1 = t0 + w * sec
        amp = float(row.get("amplitude") or 0.0)
        events.append(
            Event(
                t0=t0,
                t1=t1,
                motif=tid,
                energy=abs(amp),
                amplitude=amp,
                window_id=str(row.get("window_id") or "band_profile"),
            )
        )
    events.sort(key=lambda e: (e.t0, -e.energy))
    emin = energy_threshold(events)
    cleaned = clean_events(events, energy_min=emin)
    sessions = events_to_sessions(cleaned)
    seq_path = out_dir / "sequences" / "wav23.json"
    seq_path.parent.mkdir(parents=True, exist_ok=True)
    save_sequences(
        seq_path,
        sessions,
        meta={
            "file_id": "wav23",
            "method": "learn_from_motifs",
            "n_events": len(cleaned),
            "n_sessions": len(sessions),
            "energy_min": emin,
            "sec_per_bin": sec,
        },
    )
    # summary-shaped dict for scoring
    (out_dir / "dict_summary_for_score.json").write_text(
        json.dumps(
            {
                "n_atoms": summary.get("n_atoms"),
                "time_coverage_fraction": summary.get("coverage"),
                "coverage": summary.get("coverage"),
                "stop_reason": summary.get("stop_reason"),
            },
            indent=2,
        )
    )
    print(f"motifs sequences: {len(cleaned)} events, {len(sessions)} sessions")
    return seq_path


# Retuned after L_motifs_wav23_v1_loose: one atom + max_relative_distance=0.859
# accepted dissimilar patches (median rd≈0.77) and hit coverage=0.5 immediately.
MOTIFS_LENGTHS = "2,3,5"
MOTIFS_MIN_COVERAGE = 0.85
MOTIFS_MAX_RELATIVE_DISTANCE = 0.5


def stage_motifs(*, force: bool = False, max_rounds: int = 30) -> Path:
    ensure_work_dirs()
    rd = run_dir("L_motifs_wav23")
    out = rd / "dict"
    if (out / "summary.json").is_file() and not force:
        print(f"learn_from_motifs exists: {out}")
        if not (out / "sequences" / "wav23.json").is_file():
            _motifs_to_sessions(out)
        return out
    if force:
        if out.exists():
            shutil.rmtree(out)
        figs = rd / "figures"
        if figs.exists():
            shutil.rmtree(figs)
        for stale in ("metrics.json", "motifs_timing.json"):
            p = rd / stale
            if p.is_file():
                p.unlink()
    out.mkdir(parents=True, exist_ok=True)
    cfg = {
        "run_id": "L_motifs_wav23",
        "method": "learn_from_motifs",
        "lengths": MOTIFS_LENGTHS,
        "min_coverage": MOTIFS_MIN_COVERAGE,
        "max_relative_distance": MOTIFS_MAX_RELATIVE_DISTANCE,
        "max_rounds": max_rounds,
        "note": "retuned: tight match gate + high coverage stop (v1 loose archived)",
    }
    (rd / "config.json").write_text(json.dumps(cfg, indent=2))
    cmd = [
        PY, "-m", "analysis.dict_learning.code.learn.learn_from_motifs",
        "--gpl-dir", str(GPL_WAV23),
        "--out", str(out),
        "--lengths", MOTIFS_LENGTHS,
        "--min-coverage", str(MOTIFS_MIN_COVERAGE),
        "--max-relative-distance", str(MOTIFS_MAX_RELATIVE_DISTANCE),
        "--max-rounds", str(max_rounds),
        "--nthreads", "8",
    ]
    t0 = time.time()
    _run(cmd)
    (rd / "motifs_timing.json").write_text(json.dumps({"s": time.time() - t0}, indent=2))
    _motifs_to_sessions(out)
    return out


def _iter_run_ids() -> list[str]:
    ids = [c["run_id"] for c in GRID] + ["L_motifs_wav23"]
    return [i for i in ids if run_dir(i).is_dir()]


def stage_score() -> dict:
    ensure_work_dirs()
    rows: list[dict] = []
    for run_id in _iter_run_ids():
        rd = run_dir(run_id)
        cfg_path = rd / "config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
        method = cfg.get("method") or "randomized-build"
        if run_id.startswith("L_"):
            method = "learn_from_motifs"
            summary_path = rd / "dict" / "dict_summary_for_score.json"
            if not summary_path.is_file():
                summary_path = rd / "dict" / "summary.json"
            seq_path = rd / "dict" / "sequences" / "wav23.json"
        else:
            summary_path = rd / "dict" / "summary.json"
            seq_path = rd / "sequences" / "wav23.json"
        if not seq_path.is_file():
            print(f"skip score {run_id}: no sequences")
            continue
        summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
        sessions = load_sessions(seq_path)
        row = score_run(
            run_id=run_id,
            method=method,
            dict_summary=summary,
            sessions=sessions,
            knobs=cfg,
        )
        (rd / "metrics.json").write_text(json.dumps(row, indent=2))
        rows.append(row)
        print(
            f"{run_id}: atoms={row.get('n_atoms')} cov={row.get('time_coverage_fraction')} "
            f"period3={row.get('frac_local_period3_repeat')} ready={row.get('phrase_ready')}"
        )
    board = write_leaderboard(WORK / "leaderboard.json", rows)
    print(json.dumps({"n_runs": board["n_runs"], "winner": (board.get("winner") or {}).get("run_id")}, indent=2))
    return board


def _seq_path_for_run(run_id: str) -> Path | None:
    rd = run_dir(run_id)
    if run_id.startswith("L_"):
        cand = rd / "dict" / "sequences" / "wav23.json"
    else:
        cand = rd / "sequences" / "wav23.json"
    return cand if cand.is_file() else None


def write_motifs_csv(path: Path, sessions: list[Session]) -> int:
    """motif,start_sec,end_sec,energy — one row per motif placement."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["motif", "start_sec", "end_sec", "energy"])
        for s in sessions:
            for t in s.tokens:
                if t.kind != "motif" or t.motif is None or t.t1 <= t.t0:
                    continue
                w.writerow(
                    [
                        int(t.motif),
                        f"{float(t.t0):.10f}",
                        f"{float(t.t1):.10f}",
                        f"{float(t.energy):.10g}",
                    ]
                )
                n += 1
    return n


def stage_export_csvs(*, only: str | None = None) -> dict[str, Any]:
    """Write motif placement CSVs under work/exports/<run_id>/ and each run dir."""
    ensure_work_dirs()
    summary: dict[str, Any] = {"runs": {}}
    for run_id in _iter_run_ids():
        if only and run_id != only:
            continue
        seq_path = _seq_path_for_run(run_id)
        if seq_path is None:
            print(f"skip export {run_id}: no sequences")
            continue
        sessions = load_sessions(seq_path)
        placements = tokens_to_placements(sessions)
        rd = run_dir(run_id)
        export_dir = EXPORTS / run_id
        export_dir.mkdir(parents=True, exist_ok=True)
        motifs_csv = export_dir / "motifs.csv"
        placements_csv = export_dir / "placements.csv"
        n = write_motifs_csv(motifs_csv, sessions)
        write_placements_csv(placements_csv, placements)
        # Also keep a copy next to the run artifacts
        write_motifs_csv(rd / "motifs.csv", sessions)
        write_placements_csv(rd / "placements.csv", placements)
        summary["runs"][run_id] = {
            "n_motifs": n,
            "motifs_csv": str(motifs_csv),
            "placements_csv": str(placements_csv),
        }
        print(f"export {run_id}: {n} motifs → {motifs_csv}")
    (EXPORTS / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def stage_panels() -> None:
    ensure_work_dirs()
    for run_id in _iter_run_ids():
        seq_path = _seq_path_for_run(run_id)
        if seq_path is None:
            continue
        sessions = load_sessions(seq_path)
        placements = tokens_to_placements(sessions)
        out = run_dir(run_id) / "figures" / "motifs_by_5min"
        if out.exists() and len(list(out.glob("panel_*.png"))) >= 15:
            print(f"panels exist: {out}")
            continue
        print(f"panels {run_id} …", flush=True)
        paths = plot_five_min_panels(placements, out_dir=out)
        print(f"  wrote {len(paths)}")


def stage_report() -> Path:
    ensure_work_dirs()
    board = json.loads((WORK / "leaderboard.json").read_text()) if (WORK / "leaderboard.json").is_file() else {"ranked": []}
    ranked = board.get("ranked") or []
    winner = board.get("winner")
    lines = [
        "# Exp7: dictionary evaluation on wav23",
        "",
        "**Recording:** `wav23.wav` (200 kHz, 1.25 h), GPL 60 windows.",
        "",
        "Phrase-readiness gate before any theme HMM. Methods: `randomized-build` grid + `learn_from_motifs`.",
        "",
        "## Leaderboard",
        "",
        "| Rank | run_id | method | atoms | cov | period3 | same | multi-phrase | gate | ready |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|---|",
    ]
    for i, r in enumerate(ranked, 1):
        lines.append(
            "| {i} | `{id}` | {m} | {a} | {c} | {p3} | {sm} | {mu} | {g} | {ready} |".format(
                i=i,
                id=r.get("run_id"),
                m=r.get("method"),
                a=r.get("n_atoms"),
                c=_fmt(r.get("time_coverage_fraction")),
                p3=_fmt(r.get("frac_local_period3_repeat")),
                sm=_fmt(r.get("frac_consecutive_same_label")),
                mu=_fmt(r.get("frac_phrases_len_ge2")),
                g=r.get("gate_pass_hint"),
                ready="yes" if r.get("phrase_ready") else "no",
            )
        )
    lines.extend(["", "## Go / no-go", ""])
    n_ready = sum(1 for r in ranked if r.get("phrase_ready"))
    if n_ready and winner:
        lines.append(
            f"**Go** — choose `{winner.get('run_id')}` "
            f"(method={winner.get('method')}, atoms={winner.get('n_atoms')}, "
            f"period3={_fmt(winner.get('frac_local_period3_repeat'))})."
        )
        lines.append("")
        lines.append(f"Dict path: `work/runs/{winner.get('run_id')}/dict/`")
    else:
        lines.append(
            "**No-go** — no run is phrase-ready (gate fail and/or coverage < 0.4 and/or too many atoms). "
            "Do **not** advance to theme HMM. Next: dict redesign (e.g. panMP→MP batch), not another Exp5-style theme model."
        )
    lines.extend(
        [
            "",
            "## Arms: good / bad",
            "",
        ]
    )
    for r in ranked:
        sk = r.get("skepticism") or {}
        flags = [k for k, v in sk.items() if v]
        lines.append(
            f"- `{r.get('run_id')}` ({r.get('method')}): "
            f"atoms={r.get('n_atoms')}, cov={_fmt(r.get('time_coverage_fraction'))}, "
            f"period3={_fmt(r.get('frac_local_period3_repeat'))}, "
            f"flags={flags or 'none'}"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- Runs: `analysis/song_hmm/exp7/work/runs/`",
            "- Leaderboard: `analysis/song_hmm/exp7/work/leaderboard.json`",
            "- Panels: `…/runs/<id>/figures/motifs_by_5min/`",
            "- Notebook: `analysis/song_hmm/exp7/notebooks/dict_compare.ipynb`",
            "- Inventory: `analysis/song_hmm/exp7/inventory.md`",
            "",
        ]
    )
    text = "\n".join(lines)
    (WORK / "report.md").write_text(text)
    report = paths.REPORTS / "exp7_dict_evaluation.md"
    report.write_text(text)
    print(f"wrote {WORK / 'report.md'} and {report}")
    _write_notebook(ranked)
    return report


def _fmt(x: Any) -> str:
    if x is None:
        return "—"
    try:
        return f"{float(x):.3f}"
    except (TypeError, ValueError):
        return str(x)


def _write_notebook(ranked: list[dict]) -> None:
    """Slim notebook: leaderboard + markdown images for top runs."""
    cells: list[dict] = [
        {
            "cell_type": "markdown",
            "id": "md0",
            "metadata": {},
            "source": [
                "# Exp7 — dict compare on wav23\n",
                "\n",
                "Images load from disk. Kernel: **Python (panMP / ensemble)**.\n",
            ],
        }
    ]
    # leaderboard markdown
    lb = ["## Leaderboard\n", "\n", "See `../work/leaderboard.json` and `../work/report.md`.\n", "\n"]
    for i, r in enumerate(ranked[:8], 1):
        lb.append(
            f"{i}. `{r.get('run_id')}` — atoms={r.get('n_atoms')}, "
            f"period3={_fmt(r.get('frac_local_period3_repeat'))}, "
            f"ready={r.get('phrase_ready')}\n"
        )
    cells.append({"cell_type": "markdown", "id": "md_lb", "metadata": {}, "source": lb})

    for r in ranked[:3]:
        rid = r.get("run_id")
        fig = Path(__file__).resolve().parent / "work" / "runs" / str(rid) / "figures" / "motifs_by_5min"
        src = [f"## `{rid}`\n", "\n"]
        pngs = sorted(fig.glob("panel_*.png"))[:6] if fig.is_dir() else []
        if not pngs:
            src.append("(no panels yet)\n")
        for p in pngs:
            rel = f"../work/runs/{rid}/figures/motifs_by_5min/{p.name}"
            src.append(f"### {p.name}\n\n![{p.name}]({rel})\n\n")
        cells.append({"cell_type": "markdown", "id": f"md_{rid}", "metadata": {}, "source": src})

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (panMP / ensemble)",
                "language": "python",
                "name": "panmp",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }
    paths.NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    (paths.NOTEBOOKS / "dict_compare.ipynb").write_text(json.dumps(nb, indent=1) + "\n")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage",
        choices=[
            "0",
            "inventory",
            "grid",
            "motifs",
            "score",
            "panels",
            "export",
            "report",
            "all",
        ],
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--only", default=None, help="grid/export: only this run_id")
    p.add_argument("--max-rounds", type=int, default=30)
    args = p.parse_args(argv)

    if args.stage in ("0", "all"):
        stage0()
    if args.stage in ("inventory", "all"):
        stage_inventory()
    if args.stage in ("grid", "all"):
        stage_grid(force=args.force, only=args.only)
    if args.stage in ("motifs", "all"):
        stage_motifs(force=args.force, max_rounds=args.max_rounds)
    if args.stage in ("score", "all"):
        stage_score()
    if args.stage in ("panels", "all"):
        stage_panels()
    if args.stage in ("export", "all"):
        stage_export_csvs(only=args.only)
    if args.stage in ("report", "all"):
        stage_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
