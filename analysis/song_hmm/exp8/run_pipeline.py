"""Exp8: theme–slot HMM on Exp7 wav23 dictionaries (R_defaults, R_defaults_cov04)."""

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
from analysis.song_hmm.exp5.theme_slot import (
    decode_theme_slot,
    save_theme_slot,
    select_theme_slot,
)
from analysis.song_hmm.exp6.export_csvs import tokens_to_placements
from analysis.song_hmm.exp7.metrics_dict import load_sessions
from analysis.song_hmm.exp7.plot_panels import plot_five_min_panels

from . import paths
from .paths import (
    ARM_IDS,
    EXP7,
    GPL_WAV23,
    JS_N_TRIALS,
    JS_W_S,
    NOTEBOOKS,
    REPORTS,
    TRAIN_T1_S,
    WORK,
    arm_dir,
    ensure_work_dirs,
    exp7_run_dir,
)
from .plot_themes import plot_five_min_theme_panels
from .split import split_sessions


def _arms(only: str | None) -> tuple[str, ...]:
    if only is None:
        return ARM_IDS
    if only not in ARM_IDS:
        raise SystemExit(f"unknown arm {only!r}; choose from {ARM_IDS}")
    return (only,)


def stage_sync(*, only: str | None = None) -> None:
    ensure_work_dirs()
    for run_id in _arms(only):
        src = exp7_run_dir(run_id)
        dst = arm_dir(run_id)
        seq_src = src / "sequences" / "wav23.json"
        if not seq_src.is_file():
            raise FileNotFoundError(seq_src)
        shutil.copy2(seq_src, dst / "sequences" / "wav23.json")
        for name in ("motifs.csv", "placements.csv", "config.json", "metrics.json"):
            p = src / name
            if p.is_file():
                shutil.copy2(p, dst / name)
            elif name in ("motifs.csv", "placements.csv"):
                # Fall back to Exp7 exports/
                alt = EXP7 / "work" / "exports" / run_id / name
                if alt.is_file():
                    shutil.copy2(alt, dst / name)
        print(f"sync {run_id} ← {src}")


def _write_motif_notebook(run_id: str, n_panels: int = 15) -> Path:
    lines = [
        f"# Exp8 — motifs (`{run_id}`)\n",
        "\n",
        "GPL **display** spectrogram (10–1000 Hz, jet) + motif bars. Full **1.25 h**.\n",
        "\n",
        "Kernel: **Python (panMP / ensemble)**\n",
        "\n",
        f"Source dict: Exp7 `{run_id}`. CSV: `../work/arms/{run_id}/motifs.csv`\n",
        "\n",
        "## Panels\n",
        "\n",
    ]
    for i in range(1, n_panels + 1):
        t0 = (i - 1) * 5
        t1 = i * 5
        rel = f"../work/arms/{run_id}/figures/motifs_by_5min/panel_{i:02d}.png"
        lines.append(f"### panel_{i:02d} ({t0}–{t1} min)\n")
        lines.append(f"![p{i:02d}]({rel})\n")
        lines.append("\n")
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (panMP / ensemble)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": lines},
        ],
    }
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    path = NOTEBOOKS / f"motifs_{run_id}.ipynb"
    path.write_text(json.dumps(nb, indent=1) + "\n")
    return path


def stage_motifs(*, only: str | None = None) -> None:
    ensure_work_dirs()
    for run_id in _arms(only):
        dst = arm_dir(run_id)
        seq = dst / "sequences" / "wav23.json"
        if not seq.is_file():
            raise FileNotFoundError(f"run sync first: {seq}")
        sessions = load_sessions(seq)
        placements = tokens_to_placements(sessions)
        out = dst / "figures" / "motifs_by_5min"
        print(f"motifs panels {run_id} …", flush=True)
        paths = plot_five_min_panels(placements, out_dir=out, gpl_dir=GPL_WAV23)
        nb = _write_motif_notebook(run_id, n_panels=len(paths))
        print(f"  wrote {len(paths)} panels + {nb}")


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
            f"{run_id}: train_sessions={len(train_sess)} test_sessions={len(test_sess)} "
            f"vocab={len(vocab)}",
            flush=True,
        )
        model, spec, meta = select_theme_slot(train_sess, vocab)
        model_path = dst / "models" / "theme_slot.npz"
        save_theme_slot(model_path, model, spec, meta)
        print(
            json.dumps(
                {k: meta[k] for k in meta if k not in ("medoids", "state_names", "vocab")},
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

        # JS null on test span inset by W
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
        js_rand = null["flat_random_js"]
        js_sum = summarize_js(js_pred, js_rand)
        js_sum["random_null_mean_of_trials"] = null["mean_of_trial_means"]
        js_sum["random_null_p90_of_trials"] = null["p90_of_trial_means"]
        if js_sum.get("mean_js_pred") is not None and null["p90_of_trial_means"] is not None:
            js_sum["pred_below_random_p90"] = bool(
                js_sum["mean_js_pred"] <= null["p90_of_trial_means"]
            )

        result = {
            "run_id": run_id,
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
                for k in ("n_themes", "slot_counts", "train_score", "selection", "n_phrases", "n_train_seqs")
                if k in meta
            },
            "note": "Approach A theme–slot; train 1h / test rest; JS null = random positions on test",
        }
        out = dst / "results" / "eval.json"
        out.write_text(json.dumps(result, indent=2))
        print(
            f"{run_id}: themes={len(themes)} bounds={len(bounds)} "
            f"js_pred={js_sum.get('mean_js_pred')} js_rand={js_sum.get('mean_js_random')} "
            f"→ {out}"
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
        lines.append(f"\n_… {len(themes) - max_rows} more rows in `eval.json`_\n")
    return lines


def _write_theme_notebook(run_id: str, eval_path: Path, panel_paths: list[Path]) -> Path:
    data = json.loads(eval_path.read_text())
    themes = data.get("themes") or []
    js = data.get("js_summary") or {}
    split = data.get("split") or {}
    meta = data.get("model_meta") or {}

    head: list[str] = [
        f"# Exp8 — themes (`{run_id}`)\n",
        "\n",
        "Yellow lines = decoded theme boundaries on test. Motif bars below spectrogram.\n",
        "\n",
        "Kernel: **Python (panMP / ensemble)**\n",
        "\n",
        "## Model / split\n",
        "\n",
        f"- Train `[0, {TRAIN_T1_S:.0f})` s; test from {split.get('test_t0')} s\n",
        f"- `n_themes`={meta.get('n_themes')}, `slot_counts`={meta.get('slot_counts')}\n",
        f"- Boundaries={len(data.get('boundaries') or [])}, theme visits={len(themes)}\n",
        f"- mean JS pred={js.get('mean_js_pred')}, mean JS random={js.get('mean_js_random')}, "
        f"pred/random={js.get('mean_js_pred_over_random')}\n",
        "\n",
        "## Theme intervals (test)\n",
        "\n",
    ]
    head.extend(_theme_table_md(themes))
    head.append("\n## Panels\n\n")

    # Prefer test panels (13–15 = minutes 60–75) first, then list all written
    cells = [{"cell_type": "markdown", "metadata": {}, "source": head}]
    body: list[str] = []
    for p in panel_paths:
        # panel_NN.png → minutes
        try:
            idx = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            idx = 0
        t0, t1 = (idx - 1) * 5, idx * 5
        rel = f"../work/arms/{run_id}/figures/themes_by_5min/{p.name}"
        body.append(f"### {p.name} ({t0}–{t1} min)\n")
        body.append(f"![{p.stem}]({rel})\n\n")
    cells.append({"cell_type": "markdown", "metadata": {}, "source": body})

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python (panMP / ensemble)",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
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
        # Full file + emphasize test: write all 15
        print(f"theme panels {run_id} …", flush=True)
        paths = plot_five_min_theme_panels(
            placements,
            themes,
            bounds,
            out_dir=out,
            gpl_dir=GPL_WAV23,
        )
        nb = _write_theme_notebook(run_id, eval_path, paths)
        print(f"  wrote {len(paths)} panels + {nb}")


def _fmt(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def stage_report() -> Path:
    ensure_work_dirs()
    rows: list[dict] = []
    for run_id in ARM_IDS:
        ev = arm_dir(run_id) / "results" / "eval.json"
        met = arm_dir(run_id) / "metrics.json"
        if not ev.is_file():
            continue
        data = json.loads(ev.read_text())
        exp7 = json.loads(met.read_text()) if met.is_file() else {}
        rows.append({"run_id": run_id, "eval": data, "exp7": exp7})

    lines = [
        "# Exp8: theme–slot HMM on Exp7 wav23 dictionaries",
        "",
        "**Recording:** `wav23.wav` (200 kHz, 1.25 h), GPL 60 windows.",
        "",
        "**Arms:** `R_defaults` and `R_defaults_cov04` (Exp7 randomized-build dictionaries).",
        "",
        "**Split:** train `[0, 3600)` s (1 h), test `[3600, end)` (~15 min).",
        "",
        "## HMM architecture (Approach A theme–slot)",
        "",
        "Categorical HMM (`hmmlearn.CategoricalHMM`) over motif/gap tokens.",
        "",
        "- Each theme = **one phrase type** = left-to-right slot chain `slot1 → … → L` (no skips, no self-loops).",
        "- At last slot: **phrase-repeat** → `slot1` (same theme), or **next-theme** → next theme’s `slot1` (never revisit).",
        "- Final theme: phrase-repeat only.",
        "- `T ∈ {3..8}` chosen by **BIC** on train; per-theme `L_t` from phrase medoids (clipped [2,10]).",
        "",
        "```text",
        "Theme1 ──next──► Theme2 ──next──► … ──next──► ThemeT",
        "  ↑ phrase_repeat    ↑ phrase_repeat              ↑ phrase_repeat",
        "```",
        "",
        "## Dict provenance (Exp7)",
        "",
        "| run_id | atoms | cov | period3 | gate | phrase_ready |",
        "|---|---:|---:|---:|---|---|",
    ]
    for r in rows:
        e7 = r["exp7"]
        lines.append(
            f"| `{r['run_id']}` | {e7.get('n_atoms')} | {_fmt(e7.get('time_coverage_fraction'))} | "
            f"{_fmt(e7.get('frac_local_period3_repeat'))} | {e7.get('gate_pass_hint')} | "
            f"{e7.get('phrase_ready')} |"
        )

    lines.extend(
        [
            "",
            "## Results (test JS vs random)",
            "",
            "| run_id | n_themes | slot_counts | n_bounds | mean JS pred | mean JS random | pred/random | "
            "null trial mean | null p90 | looks_like_random | pred_below_p90 |",
            "|---|---:|---|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for r in rows:
        ev = r["eval"]
        js = ev.get("js_summary") or {}
        null = ev.get("js_random_null_trials") or {}
        mm = ev.get("model_meta") or {}
        lines.append(
            "| `{id}` | {nt} | `{sc}` | {nb} | {jp} | {jr} | {ratio} | {tm} | {p90} | {llr} | {pbp} |".format(
                id=r["run_id"],
                nt=mm.get("n_themes"),
                sc=mm.get("slot_counts"),
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

    lines.extend(["", "## Comparison", ""])
    if len(rows) >= 2:
        a, b = rows[0], rows[1]
        ja = (a["eval"].get("js_summary") or {}).get("mean_js_pred")
        jb = (b["eval"].get("js_summary") or {}).get("mean_js_pred")
        ra = (a["eval"].get("js_summary") or {}).get("mean_js_pred_over_random")
        rb = (b["eval"].get("js_summary") or {}).get("mean_js_pred_over_random")
        lines.append(
            f"- `{a['run_id']}`: mean JS pred={_fmt(ja)}, pred/random={_fmt(ra)}, "
            f"T={(a['eval'].get('model_meta') or {}).get('n_themes')}."
        )
        lines.append(
            f"- `{b['run_id']}`: mean JS pred={_fmt(jb)}, pred/random={_fmt(rb)}, "
            f"T={(b['eval'].get('model_meta') or {}).get('n_themes')}."
        )
        if ja is not None and jb is not None:
            better = a["run_id"] if ja >= jb else b["run_id"]
            lines.append(
                f"- Higher mean JS at predicted boundaries: **`{better}`** "
                "(larger = sharper motif-histogram change at boundary)."
            )
    else:
        lines.append("_Incomplete arms — re-run `hmm`._")

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- Arms: `analysis/song_hmm/exp8/work/arms/<run_id>/`",
            "- Motif CSVs: `…/motifs.csv`",
            "- Model: `…/models/theme_slot.npz`",
            "- Eval: `…/results/eval.json`",
            "- Notebooks: `analysis/song_hmm/exp8/notebooks/`",
            "",
            "## Note",
            "",
            "Train and test are time-adjacent segments of the same 1.25 h file (no separate recording).",
            "Exp9 (many themes + slot self-loops) is planned after Exp8 — not run here.",
            "",
        ]
    )

    text = "\n".join(lines)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out1 = REPORTS / "exp8_theme_hmm_wav23.md"
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
    p.add_argument("--only", default=None, help="only this arm id")
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
