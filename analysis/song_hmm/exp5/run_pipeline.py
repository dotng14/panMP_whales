"""Run Experiment 5 stages: 0 data → 1A dict/MP/seq → 1B features → 2 models → 3 eval."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from analysis.dict_learning.code.cli import load_windows
from analysis.song_hmm.exp4.baselines import b1_boundaries
from analysis.song_hmm.exp4.marks import propose_provisional_marks, save_marks
from analysis.song_hmm.exp4.sequence import (
    Session,
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
from analysis.song_hmm.exp4.spectrogram import plot_eval_slices

from . import paths
from .evaluate import (
    js_at_times_features,
    js_at_times_motifs,
    js_novelty_baseline_motifs,
    random_position_js_null,
    score_boundaries,
    summarize_js,
)
from .features_b import build_features, load_features, save_features
from .gate import motif_gate, save_gate
from .paths import (
    GPL_193500,
    TRAIN_N_WINDOWS,
    TRAIN_T0_S,
    TRAIN_T1_S,
    TEST_N_WINDOWS,
    TEST_T0_S,
    TEST_T1_S,
    WAV_193500,
    WORK,
    all_hour_files,
    ensure_work_dirs,
    train_hour_files,
)
from .theme_slot import (
    decode_theme_slot,
    load_theme_slot,
    save_theme_slot,
    select_theme_slot,
)
from .theme_slot_gauss import (
    decode_theme_slot_gauss,
    load_theme_slot_gauss,
    save_theme_slot_gauss,
    select_theme_slot_gauss,
)

PY = sys.executable
REPO = paths.REPO


def _run(cmd: list[str]) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO))


def stage0() -> dict:
    ensure_work_dirs()
    if not WAV_193500.is_file():
        raise FileNotFoundError(WAV_193500)
    if not GPL_193500.is_dir():
        raise FileNotFoundError(GPL_193500)
    windows = load_windows(GPL_193500, None, 0)
    meta = {
        "wav": str(WAV_193500),
        "gpl": str(GPL_193500),
        "n_windows": len(windows),
        "shape": list(windows[0].data.shape),
        "sample_freq": windows[0].sample_freq,
        "train_windows": f"[0, {TRAIN_N_WINDOWS})",
        "test_windows": f"[{TRAIN_N_WINDOWS}, {TRAIN_N_WINDOWS + TEST_N_WINDOWS})",
        "train_s": [TRAIN_T0_S, TRAIN_T1_S],
        "test_s": [TEST_T0_S, TEST_T1_S],
    }
    files = [{**f.__dict__, "gpl_dir": str(f.gpl_dir)} for f in all_hour_files()]
    (WORK / "file_manifest.json").write_text(json.dumps({"meta": meta, "files": files}, indent=2))
    print(json.dumps(meta, indent=2))
    return meta


# Strict non-overlap, high coverage dictionary (Approach A front-end).
DICT_NOOV_NAME = f"train_w{TRAIN_N_WINDOWS}_s0_noov"


def stage1a_dictionary(
    *,
    seed: int = 0,
    workers: int = 8,
    force: bool = False,
    no_overlap: bool = True,
    min_coverage: float = 0.95,
    max_outer: int = 40,
) -> Path:
    ensure_work_dirs()
    name = DICT_NOOV_NAME if no_overlap else f"train_w{TRAIN_N_WINDOWS}_s{seed}"
    out = WORK / "dict" / name
    if (out / "summary.json").is_file() and not force:
        print(f"dictionary exists: {out}")
        return out
    if force and out.exists():
        import shutil

        shutil.rmtree(out)
    t0 = time.time()
    cmd = [
        PY, "-m", "analysis.dict_learning", "randomized-build",
        "--gpl-dir", str(GPL_193500), "--out", str(out),
        "--n-windows", str(TRAIN_N_WINDOWS), "--window-offset", "0",
        "--seed", str(seed), "--loop-stats", "--verbose",
        "--workers", str(workers),
        "--max-outer", str(max_outer), "--m-per-outer", "5",
        "--min-coverage", str(min_coverage),
    ]
    if no_overlap:
        cmd.append("--disallow-overlap")
    _run(cmd)
    (WORK / "dict" / "timing.json").write_text(
        json.dumps(
            {
                "build_s": time.time() - t0,
                "dict": str(out),
                "disallow_overlap": no_overlap,
                "min_coverage": min_coverage,
            },
            indent=2,
        )
    )
    return out


def stage1a_mp(
    dictionary: Path,
    *,
    workers: int = 8,
    force: bool = False,
    no_overlap: bool = True,
) -> None:
    ensure_work_dirs()
    for spec in all_hour_files():
        out = WORK / "mp" / spec.file_id
        if force and out.exists():
            import shutil

            shutil.rmtree(out)
        if (out / "matchings.json").is_file():
            print(f"MP exists: {out}")
            continue
        cmd = [
            PY, "-m", "analysis.dict_learning", "matching-pursuit",
            "--gpl-dir", str(spec.gpl_dir), "--dictionary", str(dictionary),
            "--out", str(out), "--workers", str(workers),
            "--n-windows", str(spec.n_windows), "--window-offset", str(spec.window_offset),
        ]
        if no_overlap:
            cmd.append("--disallow-overlap")
        t0 = time.time()
        _run(cmd)
        (out / "timing.json").write_text(json.dumps({"mp_s": time.time() - t0}, indent=2))


def stage1a_sequences(dictionary: Path) -> None:
    ensure_work_dirs()
    widths = load_template_widths(dictionary)
    train_events = []
    for spec in train_hour_files():
        mp = WORK / "mp" / spec.file_id / "matchings.json"
        if not mp.is_file():
            continue
        windows = load_windows(spec.gpl_dir, spec.n_windows, spec.window_offset)
        train_events.extend(placements_to_events(load_matchings(mp), windows, widths))
    emin = energy_threshold(train_events)
    (WORK / "sequences" / "energy_min.json").write_text(json.dumps({"energy_min": emin}, indent=2))

    for spec in all_hour_files():
        mp = WORK / "mp" / spec.file_id / "matchings.json"
        if not mp.is_file():
            print(f"skip sequences, no MP: {spec.file_id}")
            continue
        windows = load_windows(spec.gpl_dir, spec.n_windows, spec.window_offset)
        events = clean_events(
            placements_to_events(load_matchings(mp), windows, widths), energy_min=emin
        )
        sessions = events_to_sessions(events)
        save_sequences(
            WORK / "sequences" / f"{spec.file_id}.json",
            sessions,
            meta={
                "file_id": spec.file_id,
                "role": spec.role,
                "n_events": len(events),
                "n_sessions": len(sessions),
                "energy_min": emin,
                "window_offset": spec.window_offset,
                "n_windows": spec.n_windows,
            },
        )
        print(f"{spec.file_id}: {len(events)} events, {len(sessions)} sessions")


def _load_split_sessions() -> tuple[list[Session], list[Session]]:
    train, test = [], []
    for path in sorted((WORK / "sequences").glob("193500_h*.json")):
        if path.name.endswith("_phrases.json"):
            continue
        sess, meta = load_sequences(path)
        if meta.get("role") == "train":
            train.extend(sess)
        else:
            test.extend(sess)
    return train, test


def stage1a_gate() -> dict:
    train, _ = _load_split_sessions()
    gate = motif_gate(train)
    save_gate(WORK / "sequences" / "motif_gate_train.json", gate)
    print(json.dumps(gate, indent=2))
    return gate


def stage1b_features() -> None:
    ensure_work_dirs()
    train_path = WORK / "features" / "train.npz"
    test_path = WORK / "features" / "test.npz"
    if train_path.is_file() and test_path.is_file():
        print("features exist")
        return
    print("building train features …", flush=True)
    train = build_features(
        GPL_193500, window_offset=0, n_windows=TRAIN_N_WINDOWS, hop_s=1.0, n_bands=16
    )
    save_features(
        train_path,
        train,
        meta={"role": "train", "window_offset": 0, "n_windows": TRAIN_N_WINDOWS},
    )
    print("building test features (train mean/std frozen) …", flush=True)
    test = build_features(
        GPL_193500,
        window_offset=TRAIN_N_WINDOWS,
        n_windows=TEST_N_WINDOWS,
        hop_s=1.0,
        n_bands=16,
        mean=train.mean,
        std=train.std,
    )
    save_features(
        test_path,
        test,
        meta={"role": "test", "window_offset": TRAIN_N_WINDOWS, "n_windows": TEST_N_WINDOWS},
    )
    print(f"train frames={len(train.X)} test frames={len(test.X)} dim={train.X.shape[1]}")


def stage2_models() -> None:
    ensure_work_dirs()
    train_sess, _ = _load_split_sessions()
    if not train_sess:
        raise SystemExit("no train sessions; run stage 1A sequences")

    vocab = vocabulary(train_sess)
    print(f"Approach A: vocab={len(vocab)} sessions={len(train_sess)}", flush=True)
    model_a, spec_a, meta_a = select_theme_slot(train_sess, vocab)
    save_theme_slot(WORK / "models" / "A" / "theme_slot.npz", model_a, spec_a, meta_a)
    print(json.dumps({k: meta_a[k] for k in meta_a if k != "medoids"}, indent=2))

    train_feat, _ = load_features(WORK / "features" / "train.npz")
    print(f"Approach B: frames={len(train_feat.X)}", flush=True)
    model_b, spec_b, meta_b = select_theme_slot_gauss(train_feat.X, train_feat.times)
    save_theme_slot_gauss(WORK / "models" / "B" / "theme_slot.npz", model_b, spec_b, meta_b)
    print(json.dumps(meta_b, indent=2))


def stage3_eval() -> dict:
    ensure_work_dirs()
    train_sess, test_sess = _load_split_sessions()
    marks = propose_provisional_marks(train_sess, test_sess)
    save_marks(marks, paths.REFERENCE / "marks_193500_provisional.json")
    eval_marks = marks.get("R2") or []

    # --- Approach A ---
    model_a, spec_a, meta_a = load_theme_slot(WORK / "models" / "A" / "theme_slot.npz")
    vocab = meta_a["vocab"]
    themes_a, bounds_a = [], []
    for s in test_sess:
        d = decode_theme_slot(model_a, s, vocab, spec_a)
        themes_a.extend(d.themes)
        bounds_a.extend(d.boundaries)
    bounds_a = sorted(set(bounds_a))
    base_a = js_novelty_baseline_motifs(test_sess)
    t_lo = TEST_T0_S + 60
    t_hi = TEST_T1_S - 60
    null_a = random_position_js_null(
        test_sess,
        t_lo=t_lo,
        t_hi=t_hi,
        n_times=max(len(bounds_a), 1),
        n_trials=50,
        seed=0,
    )
    random_times = null_a["flat_random_times"]
    js_pred = js_at_times_motifs(test_sess, bounds_a)
    js_rand = null_a["flat_random_js"]
    js_sum = summarize_js(js_pred, js_rand)
    js_sum["random_null_mean_of_trials"] = null_a["mean_of_trial_means"]
    js_sum["random_null_p90_of_trials"] = null_a["p90_of_trial_means"]
    if js_sum.get("mean_js_pred") is not None and null_a["p90_of_trial_means"] is not None:
        js_sum["pred_below_random_p90"] = bool(
            js_sum["mean_js_pred"] <= null_a["p90_of_trial_means"]
        )
    hours = (TEST_T1_S - TEST_T0_S) / 3600.0
    result_a = {
        "boundaries": bounds_a,
        "themes": themes_a,
        "js_summary": js_sum,
        "js_random_null_trials": {
            k: null_a[k]
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
        "baseline_js_novelty_boundaries": base_a,
        "scores_vs_provisional_marks": score_boundaries(bounds_a, eval_marks, hours=hours),
        "baseline_scores": score_boundaries(base_a, eval_marks, hours=hours),
        "n_provisional_marks": len(eval_marks),
        "model_meta": {k: meta_a[k] for k in ("n_themes", "slot_counts", "train_score", "selection") if k in meta_a},
        "note": "marks are provisional (from novelty); JS null = random positions (50 trials)",
    }
    (WORK / "results" / "A" / "eval.json").write_text(json.dumps(result_a, indent=2))

    # figures A
    plot_eval_slices(
        GPL_193500,
        themes_a,
        bounds_a,
        window_offset=TRAIN_N_WINDOWS,
        n_windows=TEST_N_WINDOWS,
        slice_s=600.0,
        title_prefix="Exp5 A test / theme-slot",
        out_dir=WORK / "figures" / "A",
    )

    # --- Approach B ---
    model_b, spec_b, meta_b = load_theme_slot_gauss(WORK / "models" / "B" / "theme_slot.npz")
    test_feat, _ = load_features(WORK / "features" / "test.npz")
    d_b = decode_theme_slot_gauss(model_b, test_feat.X, test_feat.times, spec_b)
    bounds_b = d_b.boundaries
    js_pred_b = js_at_times_features(test_feat.X, test_feat.times, bounds_b)
    js_rand_b = js_at_times_features(test_feat.X, test_feat.times, random_times)
    result_b = {
        "boundaries": bounds_b,
        "themes": d_b.themes,
        "js_summary": summarize_js(js_pred_b, js_rand_b),
        "scores_vs_provisional_marks": score_boundaries(bounds_b, eval_marks, hours=hours),
        "n_provisional_marks": len(eval_marks),
        "model_meta": {k: meta_b[k] for k in ("n_themes", "slot_counts", "train_score", "selection") if k in meta_b},
        "note": "marks are provisional (from novelty); label as such until hand marks exist",
    }
    (WORK / "results" / "B" / "eval.json").write_text(json.dumps(result_b, indent=2))
    plot_eval_slices(
        GPL_193500,
        d_b.themes,
        bounds_b,
        window_offset=TRAIN_N_WINDOWS,
        n_windows=TEST_N_WINDOWS,
        slice_s=600.0,
        title_prefix="Exp5 B test / theme-slot",
        out_dir=WORK / "figures" / "B",
    )

    summary = {
        "A": {
            "n_boundaries": len(bounds_a),
            "n_themes": len(themes_a),
            "js": result_a["js_summary"],
            "scores": result_a["scores_vs_provisional_marks"],
            "model": result_a["model_meta"],
        },
        "B": {
            "n_boundaries": len(bounds_b),
            "n_themes": len(d_b.themes),
            "js": result_b["js_summary"],
            "scores": result_b["scores_vs_provisional_marks"],
            "model": result_b["model_meta"],
        },
    }
    (WORK / "results" / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def stage_export_motif_csvs(dictionary: Path | None = None) -> dict:
    """Write aa32a7f-style placement CSVs for all cleaned motif tokens (10 h)."""
    from analysis.song_hmm.exp1_slot_hmm.labels import recording_start_unix
    from analysis.song_hmm.exp6.export_csvs import (
        tokens_to_placements,
        write_placements_csv,
        write_placements_txt,
        write_segments_motifs_only,
    )

    ensure_work_dirs()
    export_dir = WORK / "exports"
    export_dir.mkdir(parents=True, exist_ok=True)
    stem = "SOCAL_H_76_230426_193500_df100.x.x"
    origin = float(recording_start_unix(stem))

    sessions = []
    for path in sorted((WORK / "sequences").glob("193500_h*.json")):
        if path.name.endswith("_phrases.json"):
            continue
        sess, _ = load_sequences(path)
        sessions.extend(sess)
    placements = tokens_to_placements(sessions)
    write_placements_txt(export_dir / "placements.txt", placements)
    write_placements_csv(export_dir / f"placements_{stem}.csv", placements)
    write_segments_motifs_only(
        export_dir / "segments_motifs_only.csv",
        origin_unix=origin,
        placements=placements,
    )
    # Catalog-style motif_segments without theme/song labels yet (theme=0, song=1)
    motif_csv = export_dir / f"{stem}.motif_segments.csv"
    import csv as _csv

    with motif_csv.open("w", newline="", encoding="utf-8") as fh:
        w = _csv.writer(fh)
        w.writerow(
            ["motif", "song", "theme", "phrase_id", "phrase_count", "start_unix", "end_unix"]
        )
        for p in placements:
            w.writerow(
                [
                    p.motif,
                    1,
                    0,
                    0,
                    1,
                    f"{origin + p.start_sec:.6f}",
                    f"{origin + p.end_sec:.6f}",
                ]
            )
    summary = {
        "n_placements": len(placements),
        "origin_unix": origin,
        "dictionary": str(dictionary) if dictionary else None,
        "exports": str(export_dir),
    }
    (export_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage",
        choices=[
            "0", "1a", "1b", "2", "3", "all",
            "dict", "mp", "seq", "gate", "export-motifs",
        ],
    )
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true", help="Rebuild dict/MP even if present")
    p.add_argument(
        "--min-coverage",
        type=float,
        default=0.95,
        help="Dictionary build coverage target (no-overlap recipe)",
    )
    p.add_argument("--max-outer", type=int, default=40)
    args = p.parse_args(argv)

    if args.stage in ("0", "all"):
        stage0()

    dictionary = WORK / "dict" / DICT_NOOV_NAME
    if args.stage in ("1a", "dict", "all"):
        dictionary = stage1a_dictionary(
            seed=args.seed,
            workers=args.workers,
            force=args.force,
            no_overlap=True,
            min_coverage=args.min_coverage,
            max_outer=args.max_outer,
        )
    if args.stage in ("1a", "mp", "all"):
        if not (dictionary / "summary.json").is_file():
            # fall back to latest noov if stage skipped
            if not dictionary.is_dir():
                raise SystemExit(f"missing dictionary {dictionary}")
        stage1a_mp(
            dictionary,
            workers=args.workers,
            force=args.force,
            no_overlap=True,
        )
    if args.stage in ("1a", "seq", "all"):
        if not (dictionary / "summary.json").is_file():
            raise SystemExit(f"missing dictionary {dictionary}")
        stage1a_sequences(dictionary)
        stage1a_gate()
        stage_export_motif_csvs(dictionary)
    if args.stage == "gate":
        stage1a_gate()
    if args.stage == "export-motifs":
        stage_export_motif_csvs(dictionary if (dictionary / "summary.json").is_file() else None)
    if args.stage in ("1b", "all"):
        stage1b_features()
    if args.stage in ("2", "all"):
        stage2_models()
    if args.stage in ("3", "all"):
        stage3_eval()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
