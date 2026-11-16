"""Exp9: many-theme HMM with slot self-loops on Exp7 wav23 dictionaries."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from analysis.song_hmm.exp4.sequence import save_sequences, vocabulary
from analysis.song_hmm.exp5.evaluate import (
    js_at_times_motifs,
    random_position_js_null,
    summarize_js,
)
from analysis.song_hmm.exp6.export_csvs import tokens_to_placements
from analysis.song_hmm.exp7.metrics_dict import load_sessions
from analysis.song_hmm.exp7.plot_panels import plot_five_min_panels
from analysis.song_hmm.exp8.plot_themes import plot_five_min_theme_panels
from analysis.song_hmm.exp8.split import split_sessions

from .paths import (
    ARM_IDS,
    EXP7,
    EXP8,
    GPL_WAV23,
    JS_N_TRIALS,
    JS_W_S,
    L_MAX,
    NOTEBOOKS,
    REPORTS,
    THEME_GRID,
    TRAIN_T1_S,
    WORK,
    arm_dir,
    ensure_work_dirs,
    exp7_run_dir,
    exp8_arm_dir,
)
from .theme_slot_many import (
    decode_theme_slot,
    save_theme_slot_many,
    select_theme_slot_many,
)


def _arms(only: str | None) -> tuple[str, ...]:
    if only is None:
        return ARM_IDS
    if only not in ARM_IDS:
        raise SystemExit(f"unknown arm {only!r}; choose from {ARM_IDS}")
    return (only,)


def stage_sync(*, only: str | None = None) -> None:
    """Prefer Exp8 synced arms; fall back to Exp7 runs."""
    ensure_work_dirs()
    for run_id in _arms(only):
        dst = arm_dir(run_id)
        src8 = exp8_arm_dir(run_id)
        src7 = exp7_run_dir(run_id)
        src = src8 if (src8 / "sequences" / "wav23.json").is_file() else src7
        seq_src = src / "sequences" / "wav23.json"
        if not seq_src.is_file():
            raise FileNotFoundError(seq_src)
        shutil.copy2(seq_src, dst / "sequences" / "wav23.json")
        for name in ("motifs.csv", "placements.csv", "config.json", "metrics.json"):
            p = src / name
            if p.is_file():
                shutil.copy2(p, dst / name)
            elif name in ("motifs.csv", "placements.csv"):
                alt = EXP7 / "work" / "exports" / run_id / name
                if alt.is_file():
                    shutil.copy2(alt, dst / name)
        # Copy motif figures from Exp8 if present (optional speed-up)
        fig8 = src8 / "figures" / "motifs_by_5min"
        fig9 = dst / "figures" / "motifs_by_5min"
        if fig8.is_dir() and any(fig8.glob("panel_*.png")):
            for png in fig8.glob("panel_*.png"):
                shutil.copy2(png, fig9 / png.name)
        print(f"sync {run_id} ← {src}")


def _write_motif_notebook(run_id: str, n_panels: int = 15) -> Path:
    lines = [
        f"# Exp9 — motifs (`{run_id}`)\n\n",
        "Same placements as Exp8 / Exp7. GPL display + motif bars.\n\n",
        "Kernel: **Python (panMP / ensemble)**\n\n",
        "## Panels\n\n",
    ]
    for i in range(1, n_panels + 1):
        rel = f"../work/arms/{run_id}/figures/motifs_by_5min/panel_{i:02d}.png"
        lines.append(f"### panel_{i:02d} ({(i-1)*5}–{i*5} min)\n")
        lines.append(f"![p{i:02d}]({rel})\n\n")
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (panMP / ensemble)",
                "language": "python",
                "name": "python3",
            }
        },
        "cells": [{"cell_type": "markdown", "metadata": {}, "source": lines}],
    }
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    path = NOTEBOOKS / f"motifs_{run_id}.ipynb"
    path.write_text(json.dumps(nb, indent=1) + "\n")
    return path


def stage_motifs(*, only: str | None = None) -> None:
    ensure_work_dirs()
    for run_id in _arms(only):
        dst = arm_dir(run_id)
        out = dst / "figures" / "motifs_by_5min"
        existing = sorted(out.glob("panel_*.png"))
        if len(existing) >= 15:
            print(f"motifs panels exist: {out}")
            _write_motif_notebook(run_id, n_panels=len(existing))
            continue
        sessions = load_sessions(dst / "sequences" / "wav23.json")
        placements = tokens_to_placements(sessions)
        print(f"motifs panels {run_id} …", flush=True)
        paths = plot_five_min_panels(placements, out_dir=out, gpl_dir=GPL_WAV23)
        nb = _write_motif_notebook(run_id, n_panels=len(paths))
        print(f"  wrote {len(paths)} + {nb}")


def stage_hmm(*, only: str | None = None) -> None:
    ensure_work_dirs()
    for run_id in _arms(only):
        dst = arm_dir(run_id)
        sessions = load_sessions(dst / "sequences" / "wav23.json")
        train_sess, test_sess = split_sessions(sessions, cut_s=TRAIN_T1_S)
        save_sequences(
            dst / "sequences" / "train.json",
            train_sess,
            meta={"role": "train", "cut_s": TRAIN_T1_S, "run_id": run_id},
        )
        save_sequences(
            dst / "sequences" / "test.json",
            test_sess,
            meta={"role": "test", "cut_s": TRAIN_T1_S, "run_id": run_id},
        )
        vocab = vocabulary(train_sess)
        print(
            f"{run_id}: vocab={len(vocab)} theme_grid={THEME_GRID} L_max={L_MAX}",
            flush=True,
        )
        model, spec, meta = select_theme_slot_many(
            train_sess, vocab, theme_grid=THEME_GRID, l_max=L_MAX
        )
        save_theme_slot_many(dst / "models" / "theme_slot.npz", model, spec, meta)
        print(
            json.dumps(
                {
                    k: meta[k]
                    for k in meta
                    if k not in ("medoids", "state_names", "vocab")
                },
                indent=2,
            )
        )

        themes: list[dict] = []
        bounds: list[float] = []
        for s in test_sess:
            d = decode_theme_slot(model, s, meta["vocab"], spec)
            themes.extend(d.themes)
            bounds.extend(d.boundaries)
        bounds = sorted(set(bounds))

        t_lo = float(TRAIN_T1_S) + float(JS_W_S)
        t_hi = max(float(s.t1) for s in test_sess) - float(JS_W_S)
        if t_hi <= t_lo:
            t_lo = float(TRAIN_T1_S)
            t_hi = max(float(s.t1) for s in test_sess)
        null = random_position_js_null(
            test_sess,
            t_lo=t_lo,
            t_hi=t_hi,
            n_times=max(len(bounds), 1),
            n_trials=JS_N_TRIALS,
            seed=0,
            W=JS_W_S,
        )
        js_pred = js_at_times_motifs(test_sess, bounds, W=JS_W_S) if bounds else []
        js_sum = summarize_js(js_pred, null["flat_random_js"])
        js_sum["random_null_mean_of_trials"] = null["mean_of_trial_means"]
        js_sum["random_null_p90_of_trials"] = null["p90_of_trial_means"]
        if js_sum.get("mean_js_pred") is not None and null["p90_of_trial_means"] is not None:
            js_sum["pred_below_random_p90"] = bool(
                js_sum["mean_js_pred"] <= null["p90_of_trial_means"]
            )

        result = {
            "run_id": run_id,
            "experiment": "exp9",
            "split": {
                "train_t1": TRAIN_T1_S,
                "train_n_tokens": sum(len(s.tokens) for s in train_sess),
                "test_n_tokens": sum(len(s.tokens) for s in test_sess),
                "test_t0": min(float(s.t0) for s in test_sess),
                "test_t1": max(float(s.t1) for s in test_sess),
            },
            "boundaries": bounds,
            "themes": themes,
            "js_summary": js_sum,
            "js_random_null_trials": {
                k: null[k]
                for k in (
                    "n_times",
                    "n_trials",
                    "t_lo",
                    "t_hi",
                    "mean_of_trial_means",
                    "std_of_trial_means",
                    "p90_of_trial_means",
                )
            },
            "model_meta": {
                k: meta[k]
                for k in (
                    "kind",
                    "n_themes",
                    "slot_counts",
                    "l_max",
                    "self_loops",
                    "train_score",
                    "selection",
                    "n_phrases",
                    "n_train_seqs",
                )
                if k in meta
            },
            "note": "Exp9 many-theme + self-loops; train 1h / test rest; JS null on test",
        }
        out = dst / "results" / "eval.json"
        out.write_text(json.dumps(result, indent=2))
        print(
            f"{run_id}: T={meta.get('n_themes')} bounds={len(bounds)} "
            f"js_pred={js_sum.get('mean_js_pred')} js_rand={js_sum.get('mean_js_random')} → {out}"
        )


def _theme_table_md(themes: list[dict], *, max_rows: int = 40) -> list[str]:
    lines = [
        "| theme | t0 (s) | t1 (s) | t0 (min) | t1 (min) | n_tokens |\n",
        "|---|---:|---:|---:|---:|---:|\n",
    ]
    for th in themes[:max_rows]:
        t0, t1 = float(th["t0"]), float(th["t1"])
        lines.append(
            f"| {th.get('theme')} | {t0:.1f} | {t1:.1f} | {t0/60:.2f} | {t1/60:.2f} | "
            f"{th.get('n_tokens')} |\n"
        )
    if len(themes) > max_rows:
        lines.append(f"\n_… {len(themes) - max_rows} more in eval.json_\n")
    return lines


def _write_theme_notebook(run_id: str, eval_path: Path, panel_paths: list[Path]) -> Path:
    data = json.loads(eval_path.read_text())
    themes = data.get("themes") or []
    js = data.get("js_summary") or {}
    meta = data.get("model_meta") or {}
    head = [
        f"# Exp9 — themes (`{run_id}`)\n\n",
        "Many-theme HMM with slot self-loops. Yellow = theme boundaries (test).\n\n",
        "Kernel: **Python (panMP / ensemble)**\n\n",
        "## Model / split\n\n",
        f"- Train `[0, {TRAIN_T1_S:.0f})` s; `L_max`={meta.get('l_max')}; self_loops={meta.get('self_loops')}\n",
        f"- `n_themes`={meta.get('n_themes')}, `slot_counts`={meta.get('slot_counts')}\n",
        f"- Boundaries={len(data.get('boundaries') or [])}, visits={len(themes)}\n",
        f"- mean JS pred={js.get('mean_js_pred')}, random={js.get('mean_js_random')}, "
        f"pred/random={js.get('mean_js_pred_over_random')}\n\n",
        "## Theme intervals (test)\n\n",
    ]
    head.extend(_theme_table_md(themes))
    head.append("\n## Panels\n\n")
    body: list[str] = []
    for p in panel_paths:
        try:
            idx = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            idx = 0
        rel = f"../work/arms/{run_id}/figures/themes_by_5min/{p.name}"
        body.append(f"### {p.name} ({(idx-1)*5}–{idx*5} min)\n![{p.stem}]({rel})\n\n")
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (panMP / ensemble)",
                "language": "python",
                "name": "python3",
            }
        },
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": head},
            {"cell_type": "markdown", "metadata": {}, "source": body},
        ],
    }
    path = NOTEBOOKS / f"themes_{run_id}.ipynb"
    path.write_text(json.dumps(nb, indent=1) + "\n")
    return path


def stage_themes(*, only: str | None = None) -> None:
    ensure_work_dirs()
    for run_id in _arms(only):
        dst = arm_dir(run_id)
        eval_path = dst / "results" / "eval.json"
        if not eval_path.is_file():
            raise FileNotFoundError(f"run hmm first: {eval_path}")
        data = json.loads(eval_path.read_text())
        themes = data.get("themes") or []
        bounds = [float(b) for b in (data.get("boundaries") or [])]
        sessions = load_sessions(dst / "sequences" / "wav23.json")
        placements = tokens_to_placements(sessions)
        out = dst / "figures" / "themes_by_5min"
        print(f"theme panels {run_id} …", flush=True)
        paths = plot_five_min_theme_panels(
            placements, themes, bounds, out_dir=out, gpl_dir=GPL_WAV23
        )
        nb = _write_theme_notebook(run_id, eval_path, paths)
        print(f"  wrote {len(paths)} + {nb}")


def _fmt(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def stage_report() -> Path:
    ensure_work_dirs()
    rows9: list[dict] = []
    for run_id in ARM_IDS:
        ev = arm_dir(run_id) / "results" / "eval.json"
        if ev.is_file():
            rows9.append({"run_id": run_id, "eval": json.loads(ev.read_text())})

    lines = [
        "# Exp9: many-theme HMM (self-loops) on wav23",
        "",
        "**Recording:** wav23 (1.25 h). **Arms:** `R_defaults`, `R_defaults_cov04`.",
        "",
        "**Split:** train `[0, 3600)` s, test remainder.",
        "",
        "## Architecture vs Exp8",
        "",
        "```text",
        "slot1 → slot2 → … → slot_Lmax   (L_max=10, shared)",
        "  each slot: self-loop allowed",
        "  slot_Lmax → slot1          (phrase repeat)",
        "  slot_Lmax → next.slot1     (next theme; never revisit)",
        "```",
        "",
        f"- Theme grid `T ∈ {list(THEME_GRID)}` (BIC on train).",
        "- Themes never revisit; one phrase type per theme; phrase may repeat inside theme.",
        "",
        "## Exp9 results (test JS vs random)",
        "",
        "| run_id | n_themes | L_max | n_bounds | mean JS pred | mean JS random | pred/random | "
        "null trial mean | null p90 | looks_like_random | pred_below_p90 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for r in rows9:
        ev = r["eval"]
        js = ev.get("js_summary") or {}
        null = ev.get("js_random_null_trials") or {}
        mm = ev.get("model_meta") or {}
        lines.append(
            "| `{id}` | {nt} | {lm} | {nb} | {jp} | {jr} | {ratio} | {tm} | {p90} | {llr} | {pbp} |".format(
                id=r["run_id"],
                nt=mm.get("n_themes"),
                lm=mm.get("l_max"),
                nb=len(ev.get("boundaries") or []),
                jp=_fmt(js.get("mean_js_pred")),
                jr=_fmt(js.get("mean_js_random")),
                ratio=_fmt(js.get("mean_js_pred_over_random")),
                tm=_fmt(null.get("mean_of_trial_means")),
                p90=_fmt(null.get("p90_of_trial_means")),
                llr=js.get("looks_like_random"),
                pbp=js.get("pred_below_random_p90"),
            )
        )

    lines.extend(
        [
            "",
            "## Comparison vs Exp8",
            "",
            "| run_id | Exp8 T | Exp8 JS pred | Exp8 pred/rand | Exp9 T | Exp9 JS pred | Exp9 pred/rand |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for run_id in ARM_IDS:
        e8p = EXP8 / "work" / "arms" / run_id / "results" / "eval.json"
        e9p = arm_dir(run_id) / "results" / "eval.json"
        e8 = json.loads(e8p.read_text()) if e8p.is_file() else {}
        e9 = json.loads(e9p.read_text()) if e9p.is_file() else {}
        j8 = e8.get("js_summary") or {}
        j9 = e9.get("js_summary") or {}
        m8 = e8.get("model_meta") or {}
        m9 = e9.get("model_meta") or {}
        lines.append(
            f"| `{run_id}` | {m8.get('n_themes')} | {_fmt(j8.get('mean_js_pred'))} | "
            f"{_fmt(j8.get('mean_js_pred_over_random'))} | {m9.get('n_themes')} | "
            f"{_fmt(j9.get('mean_js_pred'))} | {_fmt(j9.get('mean_js_pred_over_random'))} |"
        )

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `analysis/song_hmm/exp9/work/arms/<run_id>/`",
            "- Model: `…/models/theme_slot.npz`",
            "- Eval: `…/results/eval.json`",
            "- Notebooks: `analysis/song_hmm/exp9/notebooks/`",
            "",
        ]
    )
    text = "\n".join(lines)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out1 = REPORTS / "exp9_varlen_many_theme.md"
    out2 = WORK / "report.md"
    out1.write_text(text)
    out2.write_text(text)
    print(f"wrote {out1} and {out2}")
    return out1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage",
        choices=["sync", "motifs", "hmm", "themes", "report", "all"],
    )
    p.add_argument("--only", default=None)
    args = p.parse_args(argv)
    if args.stage in ("sync", "all"):
        stage_sync(only=args.only)
    if args.stage in ("motifs", "all"):
        stage_motifs(only=args.only)
    if args.stage in ("hmm", "all"):
        stage_hmm(only=args.only)
    if args.stage in ("themes", "all"):
        stage_themes(only=args.only)
    if args.stage in ("report", "all"):
        stage_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
