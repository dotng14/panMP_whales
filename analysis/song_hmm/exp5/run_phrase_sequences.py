"""Build timed A→B→C phrase sequences + random-position JS null (Exp5/6 next step)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analysis.song_hmm.exp4.sequence import load_sequences
from analysis.song_hmm.exp5.evaluate import js_at_times_motifs, random_position_js_null, summarize_js
from analysis.song_hmm.exp5.paths import TEST_T0_S, TEST_T1_S, WORK
from analysis.song_hmm.exp5.phrase_sequence import (
    build_all_phrase_sequences,
    save_phrase_sequences,
    summarize_sequences,
    write_phrase_csv,
)


def _load_by_file() -> dict[str, list]:
    by_file: dict[str, list] = {}
    for path in sorted((WORK / "sequences").glob("193500_h*.json")):
        if path.name.endswith("_phrases.json"):
            continue
        sess, _ = load_sequences(path)
        by_file[path.stem] = sess
    return by_file


def main() -> int:
    out_dir = WORK / "phrase_sequences"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_file = _load_by_file()
    seqs = build_all_phrase_sequences(by_file, cut="period", max_edit=0.34)
    summary = summarize_sequences(seqs)
    save_phrase_sequences(out_dir / "phrase_sequences_period.json", seqs, summary)
    write_phrase_csv(out_dir / "phrase_units_period.csv", seqs)

    seqs_gap = build_all_phrase_sequences(by_file, cut="gap", max_edit=0.34)
    summary_gap = summarize_sequences(seqs_gap)
    save_phrase_sequences(out_dir / "phrase_sequences_gap.json", seqs_gap, summary_gap)
    write_phrase_csv(out_dir / "phrase_units_gap.csv", seqs_gap)

    # Random-position JS null on test hours (same n as current Approach A boundaries if present)
    eval_a = WORK / "results" / "A" / "eval.json"
    n_bounds = 20
    bounds: list[float] = []
    if eval_a.is_file():
        data = json.loads(eval_a.read_text())
        bounds = list(data.get("boundaries") or [])
        n_bounds = max(len(bounds), 1)

    test_sess = []
    for fid, sessions in by_file.items():
        # hours 8–10 = test (h08,h09,h10) — file_id 193500_h08 etc.
        hour = int(fid.split("_h")[-1])
        if hour >= 8:
            test_sess.extend(sessions)

    null = random_position_js_null(
        test_sess,
        t_lo=TEST_T0_S + 60,
        t_hi=TEST_T1_S - 60,
        n_times=n_bounds,
        n_trials=50,
        seed=0,
    )
    js_pred = js_at_times_motifs(test_sess, bounds) if bounds else []
    js_sum = summarize_js(js_pred, null["flat_random_js"])
    # also: is pred mean inside the random trial distribution?
    if js_sum.get("mean_js_pred") is not None and null["mean_of_trial_means"] is not None:
        js_sum["random_null_mean_of_trials"] = null["mean_of_trial_means"]
        js_sum["random_null_p90_of_trials"] = null["p90_of_trial_means"]
        js_sum["pred_below_random_p90"] = bool(
            js_sum["mean_js_pred"] <= null["p90_of_trial_means"]
        )

    report = {
        "phrase_period": summary,
        "phrase_gap": summary_gap,
        "js_random_null": {
            "n_bounds": n_bounds,
            "summarize_vs_one_random_draw": js_sum,
            "multi_trial": {k: null[k] for k in null if k not in ("flat_random_times", "flat_random_js")},
        },
        "paths": {
            "sequences_period": str(out_dir / "phrase_sequences_period.json"),
            "csv_period": str(out_dir / "phrase_units_period.csv"),
            "sequences_gap": str(out_dir / "phrase_sequences_gap.json"),
            "csv_gap": str(out_dir / "phrase_units_gap.csv"),
        },
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
