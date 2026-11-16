"""Build timed phrase-type sequences (A→B→C A→B→C …) from motif tokens.

Raw motif ids are too fine-grained for theme HMMs. Here we:
1. Cut the motif stream into repeating units (best self-similarity period, or gaps)
2. Cluster units by normalised edit distance into types labeled A, B, C, …
3. Keep each unit's absolute time span and underlying motif ids
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from analysis.song_hmm.exp4.phrases import Phrase, cut_phrases, cut_phrases_by_period, normalised_edit
from analysis.song_hmm.exp4.sequence import Session


def _letter(i: int) -> str:
    """0→A, 25→Z, 26→AA, …"""
    if i < 0:
        raise ValueError(i)
    chars: list[str] = []
    n = i
    while True:
        chars.append(chr(ord("A") + (n % 26)))
        n = n // 26 - 1
        if n < 0:
            break
    return "".join(reversed(chars))


@dataclass
class PhraseUnit:
    """One occurrence of a phrase type in time."""

    label: str  # A, B, C, …
    t0: float
    t1: float
    motifs: tuple[int, ...]
    cluster_id: int
    file_id: str | None = None


@dataclass
class PhraseSequence:
    """Timed symbolic sequence for one session / file."""

    file_id: str
    units: list[PhraseUnit]
    period_hint: int | None
    cut: str  # "period" | "gap"

    def as_labels(self) -> list[str]:
        return [u.label for u in self.units]

    def as_string(self, sep: str = " ") -> str:
        return sep.join(self.as_labels())

    def with_times(self) -> list[dict]:
        return [
            {
                "label": u.label,
                "t0": u.t0,
                "t1": u.t1,
                "motifs": list(u.motifs),
            }
            for u in self.units
        ]


def best_period(motif_ids: Sequence[int], max_p: int = 12) -> tuple[int, float]:
    ids = list(motif_ids)
    if len(ids) < 4:
        return 1, 1.0
    best_p, best_score = 1, 1.0
    for p in range(2, min(max_p, len(ids) // 2) + 1):
        score = normalised_edit(ids[:-p], ids[p:])
        if score < best_score:
            best_score, best_p = score, p
    return best_p, float(best_score)


def cluster_motif_tuples(
    phrases: Sequence[Phrase],
    *,
    max_edit: float = 0.34,
) -> list[int]:
    """Greedy cluster: assign each phrase to first medoid with edit ≤ max_edit."""
    cluster_ids: list[int] = []
    medoids: list[tuple[int, ...]] = []
    for ph in phrases:
        assigned = None
        for cid, med in enumerate(medoids):
            if normalised_edit(ph.motifs, med) <= max_edit:
                assigned = cid
                break
        if assigned is None:
            assigned = len(medoids)
            medoids.append(tuple(ph.motifs))
        cluster_ids.append(assigned)
    return cluster_ids


def session_to_phrase_sequence(
    session: Session,
    *,
    file_id: str = "",
    cut: str = "period",
    max_edit: float = 0.34,
    max_p: int = 12,
) -> PhraseSequence:
    if cut == "gap":
        phrases = cut_phrases(session)
        period_hint = None
    else:
        motifs = [t for t in session.tokens if t.kind == "motif" and t.motif is not None]
        ids = [int(t.motif) for t in motifs]
        period_hint, _ = best_period(ids, max_p=max_p)
        phrases = cut_phrases_by_period(session, max_p=max_p)

    # drop empty / singleton-noise optional: keep all for time tracking
    cluster_ids = cluster_motif_tuples(phrases, max_edit=max_edit)
    units = [
        PhraseUnit(
            label=_letter(cid),
            t0=ph.t0,
            t1=ph.t1,
            motifs=tuple(ph.motifs),
            cluster_id=cid,
            file_id=file_id or None,
        )
        for ph, cid in zip(phrases, cluster_ids)
    ]
    return PhraseSequence(file_id=file_id, units=units, period_hint=period_hint, cut=cut)


def build_all_phrase_sequences(
    sessions_by_file: dict[str, list[Session]],
    *,
    cut: str = "period",
    max_edit: float = 0.34,
) -> list[PhraseSequence]:
    out: list[PhraseSequence] = []
    for file_id, sessions in sorted(sessions_by_file.items()):
        for i, sess in enumerate(sessions):
            fid = file_id if len(sessions) == 1 else f"{file_id}_s{i}"
            out.append(
                session_to_phrase_sequence(sess, file_id=fid, cut=cut, max_edit=max_edit)
            )
    return out


def summarize_sequences(seqs: Sequence[PhraseSequence]) -> dict:
    n_units = sum(len(s.units) for s in seqs)
    labels = [u.label for s in seqs for u in s.units]
    uniq = sorted(set(labels), key=lambda x: (len(x), x))
    # consecutive repeat rate: fraction of i where label[i]==label[i-1]
    same = 0
    tot = 0
    for s in seqs:
        labs = s.as_labels()
        for a, b in zip(labs, labs[1:]):
            tot += 1
            if a == b:
                same += 1
    # look for literal ABCABC-style: period-3 on labels
    period3_hits = 0
    period3_checks = 0
    for s in seqs:
        labs = s.as_labels()
        if len(labs) < 6:
            continue
        for i in range(len(labs) - 5):
            period3_checks += 1
            if labs[i : i + 3] == labs[i + 3 : i + 6]:
                period3_hits += 1
    return {
        "n_sequences": len(seqs),
        "n_units": n_units,
        "n_phrase_types": len(uniq),
        "phrase_types": uniq[:40],
        "frac_consecutive_same_label": (same / tot) if tot else None,
        "frac_local_period3_repeat": (period3_hits / period3_checks) if period3_checks else None,
        "period_hints": [s.period_hint for s in seqs],
        "example_strings": [
            {"file_id": s.file_id, "n": len(s.units), "labels": s.as_string(), "first_times": s.with_times()[:8]}
            for s in seqs[:3]
        ],
    }


def save_phrase_sequences(path: Path, seqs: Sequence[PhraseSequence], summary: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "summary": summary,
        "sequences": [
            {
                "file_id": s.file_id,
                "cut": s.cut,
                "period_hint": s.period_hint,
                "labels": s.as_labels(),
                "units": [asdict(u) for u in s.units],
            }
            for s in seqs
        ],
    }
    path.write_text(json.dumps(payload, indent=2))


def write_phrase_csv(path: Path, seqs: Sequence[PhraseSequence]) -> None:
    """One row per phrase unit: file, label, t0, t1, motifs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["file_id,label,t0,t1,motifs\n"]
    for s in seqs:
        for u in s.units:
            motifs = "-".join(str(m) for m in u.motifs)
            lines.append(f"{s.file_id},{u.label},{u.t0:.6f},{u.t1:.6f},{motifs}\n")
    path.write_text("".join(lines))
