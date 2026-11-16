"""Exp10: R_defaults dictionary + theme–slot HMM on 193500."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from analysis.dict_learning.code.cli import load_windows
from analysis.song_hmm.exp4.sequence import (
    clean_events,
    energy_threshold,
    events_to_sessions,
    load_matchings,
    load_sequences,
    load_template_widths,
    placements_to_events,
    save_sequences,
    vocabulary,
)
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
from analysis.song_hmm.exp6.export_csvs import Placement, tokens_to_placements, write_placements_csv
from analysis.song_hmm.exp8.plot_themes import plot_five_min_theme_panels

from .paths import (
    GPL_193500,
    JS_N_TRIALS,
    JS_W_S,
    N_TEST_FIVE_MIN_PANELS,
    N_TRAIN_SAMPLE_PANELS,
    N_WINDOWS,
    NOTEBOOKS,
    PY,
    R_DEFAULTS,
    REPORTS,
    TEST_HOURS,
    TEST_N_WINDOWS,
    TEST_T0_S,
    TEST_T1_S,
    TRAIN_HOURS,
    TRAIN_N_WINDOWS,
    TRAIN_T1_S,
    WAV_193500,
    WINDOWS_PER_HOUR,
    WORK,
    ensure_work_dirs,
    hour_file_id,
)


def _run(cmd: list[str]) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True)


def stage0() -> dict:
    ensure_work_dirs()
    meta = {
        "wav": str(WAV_193500),
        "wav_ok": WAV_193500.is_file(),
        "gpl": str(GPL_193500),
        "gpl_ok": GPL_193500.is_dir(),
        "n_windows_expected": N_WINDOWS,
        "train_n_windows": TRAIN_N_WINDOWS,
        "test_n_windows": TEST_N_WINDOWS,
        "knobs": R_DEFAULTS,
    }
    if not meta["wav_ok"]:
        raise FileNotFoundError(WAV_193500)
    if not meta["gpl_ok"]:
        raise FileNotFoundError(GPL_193500)
    wins = load_windows(GPL_193500, None, 0)
    meta["n_windows_found"] = len(wins)
    (WORK / "file_manifest.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return meta


def stage_dict(*, force: bool = False, workers: int = 8) -> Path:
    ensure_work_dirs()
    out = WORK / "dict" / "R_defaults"
    if (out / "summary.json").is_file() and not force:
        print(f"dict exists: {out}")
        return out
    if force and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = dict(R_DEFAULTS)
    (WORK / "dict" / "config.json").write_text(json.dumps(cfg, indent=2))
    cmd = [
        PY, "-m", "analysis.dict_learning", "randomized-build",
        "--gpl-dir", str(GPL_193500),
        "--out", str(out),
        "--n-windows", str(TRAIN_N_WINDOWS),
        "--window-offset", "0",
        "--seed", "0",
        "--workers", str(workers),
        "--verbose",
        "--loop-stats",
        "--min-coverage", str(cfg["min_coverage"]),
        "--max-outer", str(cfg["max_outer"]),
        "--n-min", str(cfg["n_min"]),
        "--epsilon", str(cfg["epsilon"]),
    ]
    t0 = time.time()
    _run(cmd)
    (WORK / "dict" / "timing.json").write_text(
        json.dumps({"build_s": time.time() - t0, "knobs": cfg}, indent=2)
    )
    return out


def stage_mp(*, force: bool = False, workers: int = 8) -> Path:
    ensure_work_dirs()
    dictionary = WORK / "dict" / "R_defaults"
    if not (dictionary / "summary.json").is_file():
        raise FileNotFoundError(f"run dict first: {dictionary}")
    widths = load_template_widths(dictionary)
    train_events = []

    for hour in range(1, TRAIN_HOURS + TEST_HOURS + 1):
        file_id = hour_file_id(hour)
        out = WORK / "mp" / file_id
        offset = (hour - 1) * WINDOWS_PER_HOUR
        if force and out.exists():
            shutil.rmtree(out)
        if (out / "matchings.json").is_file() and not force:
            print(f"MP exists: {out}")
        else:
            cmd = [
                PY, "-m", "analysis.dict_learning", "matching-pursuit",
                "--gpl-dir", str(GPL_193500),
                "--dictionary", str(dictionary),
                "--out", str(out),
                "--workers", str(workers),
                "--n-windows", str(WINDOWS_PER_HOUR),
                "--window-offset", str(offset),
            ]
            # R_defaults allows overlap — do not pass --disallow-overlap
            t0 = time.time()
            _run(cmd)
            (out / "timing.json").write_text(json.dumps({"mp_s": time.time() - t0}, indent=2))

        windows = load_windows(GPL_193500, WINDOWS_PER_HOUR, offset)
        events = placements_to_events(
            load_matchings(out / "matchings.json"), windows, widths
        )
        if hour <= TRAIN_HOURS:
            train_events.extend(events)

    emin = energy_threshold(train_events)
    (WORK / "sequences" / "energy_min.json").write_text(
        json.dumps({"energy_min": emin}, indent=2)
    )

    all_placements = []
    for hour in range(1, TRAIN_HOURS + TEST_HOURS + 1):
        file_id = hour_file_id(hour)
        offset = (hour - 1) * WINDOWS_PER_HOUR
        role = "train" if hour <= TRAIN_HOURS else "test"
        windows = load_windows(GPL_193500, WINDOWS_PER_HOUR, offset)
        events = clean_events(
            placements_to_events(
                load_matchings(WORK / "mp" / file_id / "matchings.json"),
                windows,
                widths,
            ),
            energy_min=emin,
        )
        sessions = events_to_sessions(events)
        save_sequences(
            WORK / "sequences" / f"{file_id}.json",
            sessions,
            meta={
                "file_id": file_id,
                "role": role,
                "n_events": len(events),
                "n_sessions": len(sessions),
                "energy_min": emin,
                "window_offset": offset,
                "n_windows": WINDOWS_PER_HOUR,
            },
        )
        print(f"{file_id}: {len(events)} events, {len(sessions)} sessions ({role})")
        for s in sessions:
            for t in s.tokens:
                if t.kind == "motif" and t.motif is not None and t.t1 > t.t0:
                    all_placements.append(
                        Placement(
                            motif=int(t.motif),
                            start_sec=float(t.t0),
                            end_sec=float(t.t1),
                        )
                    )

    all_placements.sort(key=lambda p: (p.start_sec, p.end_sec, p.motif))
    write_placements_csv(WORK / "placements.csv", all_placements)
    # motifs.csv with energy
    import csv

    with (WORK / "motifs.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["motif", "start_sec", "end_sec", "energy"])
        for hour in range(1, TRAIN_HOURS + TEST_HOURS + 1):
            sess, _ = load_sequences(WORK / "sequences" / f"{hour_file_id(hour)}.json")
            for s in sess:
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
    print(f"wrote {WORK / 'motifs.csv'} and placements.csv ({len(all_placements)} placements)")
    return dictionary


def _load_split_sessions() -> tuple[list, list]:
    train, test = [], []
    for path in sorted((WORK / "sequences").glob("193500_h*.json")):
        sess, meta = load_sequences(path)
        if meta.get("role") == "train":
            train.extend(sess)
        else:
            test.extend(sess)
    return train, test


def _load_all_placements():
    train, test = _load_split_sessions()
    return tokens_to_placements(train + test)


def stage_motifs() -> None:
    ensure_work_dirs()
    placements = _load_all_placements()
    out = WORK / "figures" / "motifs_by_5min"
    # Train sample: panels 1..6 (0–30 min)
    # Test: panels for hours 8–10 → global panel index 85..120
    # Panel k covers minutes (k-1)*5 .. k*5; hour 8 starts at minute 420 → panel 85
    train_idxs = list(range(1, N_TRAIN_SAMPLE_PANELS + 1))
    test_start_panel = TRAIN_HOURS * 12 + 1  # 85
    test_idxs = list(range(test_start_panel, test_start_panel + N_TEST_FIVE_MIN_PANELS))
    # plot_five_min_panels always writes 1..n_panels sequentially — write custom loop
    from analysis.song_hmm.exp7.plot_panels import stitch_gpl_display
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    import numpy as np

    out.mkdir(parents=True, exist_ok=True)
    colors = plt.cm.tab20.colors
    written: list[Path] = []
    for panel in train_idxs + test_idxs:
        offset = (panel - 1) * 4
        t0_min = (panel - 1) * 5
        t1_min = panel * 5
        fig, (ax_s, ax_m) = plt.subplots(
            2, 1, figsize=(14, 4.4),
            gridspec_kw={"height_ratios": [3.4, 0.9], "hspace": 0.08},
            sharex=True,
        )
        spec, t0, t1, f0, f1 = stitch_gpl_display(
            GPL_193500, window_offset=offset, n_windows=4,
            freq_lo_hz=10.0, freq_hi_hz=1000.0,
        )
        finite = spec[np.isfinite(spec)]
        vmax = float(np.percentile(finite, 99.5)) if finite.size else 1.0
        vmin = float(np.percentile(finite, 1.0)) if finite.size else 0.0
        ax_s.imshow(
            spec, origin="lower", aspect="auto",
            extent=[t0 / 60.0, t1 / 60.0, f0, f1], cmap="jet",
            vmin=vmin, vmax=max(vmax, vmin + 1e-6), interpolation="nearest",
        )
        role = "train-sample" if panel in train_idxs else "test"
        ax_s.set_ylabel("freq (Hz)")
        ax_s.set_title(f"Panel {panel:03d}: {t0_min}–{t1_min} min ({role})")
        ax_s.tick_params(labelbottom=False)
        in_panel = [p for p in placements if p.end_sec >= t0 and p.start_sec <= t1]
        seen: dict[int, tuple] = {}
        ax_m.set_xlim(t0 / 60.0, t1 / 60.0)
        ax_m.set_ylim(0, 1)
        ax_m.set_yticks([])
        ax_m.set_xlabel("time (min from wav start)")
        ax_m.set_ylabel("motifs")
        for p in in_panel:
            if p.motif not in seen:
                seen[p.motif] = colors[len(seen) % len(colors)]
            x0 = max(p.start_sec, t0) / 60.0
            x1 = min(p.end_sec, t1) / 60.0
            if x1 > x0:
                ax_m.axvspan(x0, x1, ymin=0.15, ymax=0.85, color=seen[p.motif], alpha=0.95, lw=0)
        ax_m.set_title(f"{len(in_panel)} placements, {len(seen)} ids", fontsize=9, loc="left")
        if seen:
            handles = [
                Patch(facecolor=c, edgecolor="none", label=f"M{m}")
                for m, c in sorted(seen.items())[:12]
            ]
            ax_m.legend(handles=handles, loc="upper right", ncol=6, fontsize=6, framealpha=0.7)
        path = out / f"panel_{panel:03d}.png"
        fig.savefig(path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        written.append(path)
        print(f"  {path.name}", flush=True)

    # Notebook
    lines = [
        "# Exp10 — motifs (R_defaults on 193500)\n\n",
        "GPL display 10–1000 Hz + motif bars. Train sample (0–30 min) + full test (7–10 h).\n\n",
        "Kernel: **Python (panMP / ensemble)**\n\n",
        f"CSV: `../work/motifs.csv`\n\n",
        "## Train sample\n\n",
    ]
    for panel in train_idxs:
        rel = f"../work/figures/motifs_by_5min/panel_{panel:03d}.png"
        lines.append(f"### panel_{panel:03d} ({(panel-1)*5}–{panel*5} min)\n")
        lines.append(f"![p{panel:03d}]({rel})\n\n")
    lines.append("## Test hours (7–10 h)\n\n")
    for panel in test_idxs:
        rel = f"../work/figures/motifs_by_5min/panel_{panel:03d}.png"
        lines.append(f"### panel_{panel:03d} ({(panel-1)*5}–{panel*5} min)\n")
        lines.append(f"![p{panel:03d}]({rel})\n\n")
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
    path = NOTEBOOKS / "motifs_R_defaults.ipynb"
    path.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {len(written)} panels + {path}")


def stage_hmm() -> dict:
    ensure_work_dirs()
    train_sess, test_sess = _load_split_sessions()
    if not train_sess or not test_sess:
        raise SystemExit("missing train/test sequences; run mp first")
    vocab = vocabulary(train_sess)
    print(f"Approach A: vocab={len(vocab)} train_sess={len(train_sess)} test_sess={len(test_sess)}")
    model, spec, meta = select_theme_slot(train_sess, vocab)
    save_theme_slot(WORK / "models" / "theme_slot.npz", model, spec, meta)
    print(json.dumps({k: meta[k] for k in meta if k not in ("medoids", "state_names", "vocab")}, indent=2))

    themes, bounds = [], []
    for s in test_sess:
        d = decode_theme_slot(model, s, meta["vocab"], spec)
        themes.extend(d.themes)
        bounds.extend(d.boundaries)
    bounds = sorted(set(bounds))

    t_lo = TEST_T0_S + JS_W_S
    t_hi = TEST_T1_S - JS_W_S
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

    summary = json.loads((WORK / "dict" / "R_defaults" / "summary.json").read_text())
    result = {
        "run_id": "R_defaults",
        "file": "193500",
        "split": {
            "train_hours": TRAIN_HOURS,
            "test_hours": TEST_HOURS,
            "train_t1": TRAIN_T1_S,
            "test_t0": TEST_T0_S,
            "test_t1": TEST_T1_S,
        },
        "dict": {
            "n_atoms": summary.get("n_atoms"),
            "time_coverage_fraction": summary.get("time_coverage_fraction"),
            "stop_reason": summary.get("stop_reason"),
            "knobs": R_DEFAULTS,
        },
        "boundaries": bounds,
        "themes": themes,
        "js_summary": js_sum,
        "js_random_null_trials": {
            k: null[k]
            for k in (
                "n_times", "n_trials", "t_lo", "t_hi",
                "mean_of_trial_means", "std_of_trial_means", "p90_of_trial_means",
            )
        },
        "model_meta": {
            k: meta[k]
            for k in ("n_themes", "slot_counts", "train_score", "selection", "n_phrases", "n_train_seqs")
            if k in meta
        },
        "note": "Exp10 R_defaults on 193500; Exp8-style theme–slot; JS null on test 3h",
    }
    out = WORK / "results" / "eval.json"
    out.write_text(json.dumps(result, indent=2))
    print(
        f"themes={len(themes)} bounds={len(bounds)} "
        f"js_pred={js_sum.get('mean_js_pred')} js_rand={js_sum.get('mean_js_random')} → {out}"
    )
    return result


def stage_themes() -> None:
    ensure_work_dirs()
    eval_path = WORK / "results" / "eval.json"
    if not eval_path.is_file():
        raise FileNotFoundError("run hmm first")
    data = json.loads(eval_path.read_text())
    themes = data.get("themes") or []
    bounds = [float(b) for b in (data.get("boundaries") or [])]
    placements = _load_all_placements()
    out = WORK / "figures" / "themes_by_5min"
    # Test panels only: global panel indices 85..120
    test_start = TRAIN_HOURS * 12 + 1
    test_idxs = list(range(test_start, test_start + N_TEST_FIVE_MIN_PANELS))
    print(f"theme panels test indices {test_idxs[0]}..{test_idxs[-1]} …", flush=True)
    paths = plot_five_min_theme_panels(
        placements,
        themes,
        bounds,
        out_dir=out,
        gpl_dir=GPL_193500,
        panel_indices=test_idxs,
    )

    js = data.get("js_summary") or {}
    mm = data.get("model_meta") or {}
    head = [
        "# Exp10 — themes (R_defaults on 193500)\n\n",
        "Test hours 7–10. Yellow = theme boundaries.\n\n",
        "Kernel: **Python (panMP / ensemble)**\n\n",
        f"- `n_themes`={mm.get('n_themes')}, `slot_counts`={mm.get('slot_counts')}\n",
        f"- Boundaries={len(bounds)}, visits={len(themes)}\n",
        f"- mean JS pred={js.get('mean_js_pred')}, random={js.get('mean_js_random')}, "
        f"pred/random={js.get('mean_js_pred_over_random')}\n\n",
        "## Theme intervals (test)\n\n",
        "| theme | t0 (min) | t1 (min) | n_tokens |\n",
        "|---|---:|---:|---:|\n",
    ]
    for th in themes[:60]:
        t0, t1 = float(th["t0"]), float(th["t1"])
        head.append(
            f"| {th.get('theme')} | {t0/60:.2f} | {t1/60:.2f} | {th.get('n_tokens')} |\n"
        )
    if len(themes) > 60:
        head.append(f"\n_… {len(themes)-60} more in eval.json_\n")
    head.append("\n## Panels\n\n")
    body = []
    for p in paths:
        try:
            idx = int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            idx = 0
        rel = f"../work/figures/themes_by_5min/{p.name}"
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
    path = NOTEBOOKS / "themes_R_defaults.ipynb"
    path.write_text(json.dumps(nb, indent=1) + "\n")
    print(f"wrote {len(paths)} panels + {path}")


def _fmt(x: Any) -> str:
    if x is None:
        return "—"
    if isinstance(x, float):
        return f"{x:.4g}"
    return str(x)


def stage_report() -> Path:
    ensure_work_dirs()
    ev_path = WORK / "results" / "eval.json"
    data = json.loads(ev_path.read_text()) if ev_path.is_file() else {}
    js = data.get("js_summary") or {}
    null = data.get("js_random_null_trials") or {}
    mm = data.get("model_meta") or {}
    dd = data.get("dict") or {}
    lines = [
        "# Exp10: R_defaults pipeline on 193500",
        "",
        "**Recording:** `SOCAL_H_76_230426_193500` (~10 h, 2 kHz GPL df100).",
        "",
        "**Arm:** R_defaults only (`min_coverage=0.2`, `max_outer=10`, overlap allowed, `n_min=10`, `epsilon=0.9`).",
        "",
        "**Split:** train 0–7 h, test 7–10 h.",
        "",
        "## HMM architecture",
        "",
        "Exp8 Approach A theme–slot categorical HMM (`T∈{3..8}`, no self-loops).",
        "Phrase-repeat at last slot → slot1; next-theme → next theme slot1 (never revisit).",
        "",
        "## Dictionary",
        "",
        f"- atoms={dd.get('n_atoms')}, coverage={_fmt(dd.get('time_coverage_fraction'))}, "
        f"stop={dd.get('stop_reason')}",
        "",
        "## Results (test JS vs random)",
        "",
        f"| metric | value |",
        f"|---|---|",
        f"| n_themes (selected) | {mm.get('n_themes')} |",
        f"| slot_counts | `{mm.get('slot_counts')}` |",
        f"| n_boundaries | {len(data.get('boundaries') or [])} |",
        f"| n_theme_visits | {len(data.get('themes') or [])} |",
        f"| mean JS pred | {_fmt(js.get('mean_js_pred'))} |",
        f"| mean JS random | {_fmt(js.get('mean_js_random'))} |",
        f"| pred/random | {_fmt(js.get('mean_js_pred_over_random'))} |",
        f"| null trial mean | {_fmt(null.get('mean_of_trial_means'))} |",
        f"| null p90 | {_fmt(null.get('p90_of_trial_means'))} |",
        f"| looks_like_random | {js.get('looks_like_random')} |",
        f"| pred_below_random_p90 | {js.get('pred_below_random_p90')} |",
        "",
        "## Artifacts",
        "",
        "- Dict: `analysis/song_hmm/exp10/work/dict/R_defaults/`",
        "- Sequences: `…/sequences/193500_h*.json`",
        "- Motifs CSV: `…/work/motifs.csv`",
        "- Model: `…/models/theme_slot.npz`",
        "- Eval: `…/results/eval.json`",
        "- Notebooks: `analysis/song_hmm/exp10/notebooks/`",
        "",
        "## Note vs Exp8",
        "",
        "Same R_defaults knobs and theme–slot model family as Exp8 on wav23, but different recording (193500, 10 h, 7/3 split).",
        "",
    ]
    text = "\n".join(lines)
    REPORTS.mkdir(parents=True, exist_ok=True)
    out1 = REPORTS / "exp10_r_defaults_193500.md"
    out2 = WORK / "report.md"
    out1.write_text(text)
    out2.write_text(text)
    print(f"wrote {out1} and {out2}")
    return out1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage",
        choices=["0", "dict", "mp", "motifs", "hmm", "themes", "report", "all"],
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)

    if args.stage in ("0", "all"):
        stage0()
    if args.stage in ("dict", "all"):
        stage_dict(force=args.force, workers=args.workers)
    if args.stage in ("mp", "all"):
        stage_mp(force=args.force, workers=args.workers)
    if args.stage in ("motifs", "all"):
        stage_motifs()
    if args.stage in ("hmm", "all"):
        stage_hmm()
    if args.stage in ("themes", "all"):
        stage_themes()
    if args.stage in ("report", "all"):
        stage_report()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
