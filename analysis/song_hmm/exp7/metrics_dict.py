"""Phrase-readiness metrics for Exp7 dictionary runs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

from analysis.song_hmm.exp4.phrases import cut_phrases
from analysis.song_hmm.exp4.sequence import Session, load_sequences
from analysis.song_hmm.exp5.gate import motif_gate
from analysis.song_hmm.exp5.phrase_sequence import (
    build_all_phrase_sequences,
    summarize_sequences,
)


def load_sessions(sequences_path: Path) -> list[Session]:
    """Load sessions from a single JSON or a directory of *.json."""
    path = Path(sequences_path)
    sessions: list[Session] = []
    if path.is_file():
        sess, _ = load_sequences(path)
        sessions.extend(sess)
        return sessions
    for p in sorted(path.glob("*.json")):
        if p.name in ("energy_min.json", "gate.json") or p.name.endswith("_meta.json"):
            continue
        if "gate" in p.name:
            continue
        sess, _ = load_sequences(p)
        sessions.extend(sess)
    return sessions


def score_run(
    *,
    run_id: str,
    method: str,
    dict_summary: dict[str, Any] | None,
    sessions: Sequence[Session],
    knobs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute coverage-style + phrase-readiness metrics for one run."""
    n_atoms = None
    coverage = None
    stop_reason = None
    if dict_summary:
        n_atoms = dict_summary.get("n_atoms")
        coverage = dict_summary.get("time_coverage_fraction", dict_summary.get("coverage"))
        stop_reason = dict_summary.get("stop_reason")

    motif_ids: list[int] = []
    for s in sessions:
        for t in s.tokens:
            if t.kind == "motif" and t.motif is not None:
                motif_ids.append(int(t.motif))
    ctr = Counter(motif_ids)
    n_placements = len(motif_ids)
    n_unique = len(ctr)
    rare = sum(1 for c in ctr.values() if c <= 5)
    frac_rare = (rare / n_unique) if n_unique else None

    phrases = []
    for s in sessions:
        phrases.extend(cut_phrases(s))
    lens = [len(p.motifs) for p in phrases]
    n_phrases = len(phrases)
    n_multi = sum(1 for L in lens if L >= 2)
    frac_multi = (n_multi / n_phrases) if n_phrases else None
    median_len = float(sorted(lens)[len(lens) // 2]) if lens else None

    gate = motif_gate(list(sessions)) if sessions else {}
    by_file = {"all": list(sessions)}
    seqs = build_all_phrase_sequences(by_file, cut="period", max_edit=0.5)
    phrase_sum = summarize_sequences(seqs)

    period3 = phrase_sum.get("frac_local_period3_repeat")
    same = phrase_sum.get("frac_consecutive_same_label")
    n_types = phrase_sum.get("n_phrase_types")

    atoms_flag = n_atoms is not None and int(n_atoms) > 80
    gate_fail = not bool(gate.get("gate_pass_hint"))
    # Single motif id → consecutive-same / type stats look perfect but are not multi-motif phrases
    degenerate_alphabet = n_unique is not None and int(n_unique) <= 1
    cov_low = coverage is not None and float(coverage) < 0.4
    # period3 is reported only — not a gate (phrase length / period unknown a priori)
    phrase_ready = not (
        atoms_flag or gate_fail or degenerate_alphabet or cov_low
    )

    return {
        "run_id": run_id,
        "method": method,
        "knobs": knobs or {},
        "n_atoms": n_atoms,
        "time_coverage_fraction": coverage,
        "stop_reason": stop_reason,
        "n_placements": n_placements,
        "n_unique_motifs": n_unique,
        "frac_atoms_rare_le5": frac_rare,
        "n_phrases_gap": n_phrases,
        "frac_phrases_len_ge2": frac_multi,
        "median_phrase_len": median_len,
        "gate_pass_hint": gate.get("gate_pass_hint"),
        "median_d1_consecutive": gate.get("median_d1_consecutive"),
        "median_d1_at_js_peaks": gate.get("median_d1_at_js_peaks"),
        "n_phrase_types_period_edit05": n_types,
        "frac_consecutive_same_label": same,
        "frac_local_period3_repeat": period3,
        "skepticism": {
            "n_atoms_gt_80": atoms_flag,
            "gate_fail": gate_fail,
            "coverage_lt_0.4": cov_low,
            "degenerate_single_motif_alphabet": degenerate_alphabet,
        },
        "phrase_ready": phrase_ready,
    }


def rank_leaderboard(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort: phrase_ready first; demote degenerate alphabets; then multi-motif / same / cov."""

    def key(r: dict[str, Any]) -> tuple:
        deg = bool((r.get("skepticism") or {}).get("degenerate_single_motif_alphabet"))
        return (
            0 if r.get("phrase_ready") else 1,
            1 if deg else 0,
            -(float(r.get("frac_phrases_len_ge2") or 0.0)),
            -(float(r.get("frac_consecutive_same_label") or 0.0)),
            -(float(r.get("time_coverage_fraction") or 0.0)),
            r.get("run_id") or "",
        )

    return sorted(rows, key=key)


def write_leaderboard(path: Path, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    ranked = rank_leaderboard(rows)
    payload = {
        "n_runs": len(ranked),
        "n_phrase_ready": sum(1 for r in ranked if r.get("phrase_ready")),
        "ranked": ranked,
        "winner": ranked[0] if ranked else None,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return payload
