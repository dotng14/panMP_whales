"""Motif-sequence gate metrics on train (Approach A)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from analysis.song_hmm.exp4.baselines import b1_boundaries
from analysis.song_hmm.exp4.phrases import cut_phrases, normalised_edit
from analysis.song_hmm.exp4.sequence import Session


def motif_gate(sessions: Sequence[Session]) -> dict:
    phrases = []
    for s in sessions:
        phrases.extend(cut_phrases(s))
    d1_run: list[float] = []
    for i in range(1, len(phrases)):
        # consecutive phrases in time
        if phrases[i].t0 - phrases[i - 1].t1 < 30.0:
            d1_run.append(normalised_edit(phrases[i - 1].motifs, phrases[i].motifs))
    peaks = b1_boundaries(sessions)
    d1_peak: list[float] = []
    for p in peaks:
        # phrases immediately before/after peak
        before = [ph for ph in phrases if ph.t1 <= p]
        after = [ph for ph in phrases if ph.t0 >= p]
        if before and after:
            d1_peak.append(normalised_edit(before[-1].motifs, after[0].motifs))

    n_motif_tokens = sum(
        1 for s in sessions for t in s.tokens if t.kind == "motif"
    )
    coverage_proxy = float(n_motif_tokens)  # absolute; GPL call coverage needs detections

    out = {
        "n_sessions": len(sessions),
        "n_phrases": len(phrases),
        "n_motif_tokens": n_motif_tokens,
        "median_d1_consecutive": float(np.median(d1_run)) if d1_run else None,
        "median_d1_at_js_peaks": float(np.median(d1_peak)) if d1_peak else None,
        "n_d1_consecutive": len(d1_run),
        "n_d1_peaks": len(d1_peak),
        "gate_pass_hint": None,
        "coverage_proxy_n_motif_tokens": coverage_proxy,
    }
    if out["median_d1_consecutive"] is not None and out["median_d1_at_js_peaks"] is not None:
        out["gate_pass_hint"] = bool(
            out["median_d1_consecutive"] < out["median_d1_at_js_peaks"]
        )
    return out


def save_gate(path: Path, gate: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(gate, indent=2))
