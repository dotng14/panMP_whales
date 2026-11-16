"""Phrase cutting and option-2 repetition features."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from .sequence import Session, Token, motif_string


@dataclass
class Phrase:
    t0: float
    t1: float
    motifs: tuple[int, ...]
    gap_before: float  # seconds since previous phrase end (0 for first)


@dataclass
class PhraseFeatures:
    d1: float
    d2: float
    dmin3: float
    gap: float
    len_ratio: float
    t0: float
    session_start: bool


def levenshtein(a: Sequence[int], b: Sequence[int]) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            ins = cur[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def normalised_edit(a: Sequence[int], b: Sequence[int]) -> float:
    denom = max(len(a), len(b), 1)
    return levenshtein(a, b) / denom


def cut_phrases(session: Session) -> list[Phrase]:
    """Cut phrases at GAP_S / GAP_L tokens."""
    phrases: list[Phrase] = []
    cur: list[Token] = []
    last_end: float | None = None
    for tok in session.tokens:
        if tok.kind in ("GAP_S", "GAP_L"):
            if cur:
                motifs = motif_string(cur)
                if motifs:
                    gap = 0.0 if last_end is None else cur[0].t0 - last_end
                    phrases.append(
                        Phrase(t0=cur[0].t0, t1=cur[-1].t1, motifs=motifs, gap_before=max(gap, 0.0))
                    )
                    last_end = cur[-1].t1
                cur = []
            continue
        if tok.kind == "motif":
            cur.append(tok)
    if cur:
        motifs = motif_string(cur)
        if motifs:
            gap = 0.0 if last_end is None else cur[0].t0 - last_end
            phrases.append(
                Phrase(t0=cur[0].t0, t1=cur[-1].t1, motifs=motifs, gap_before=max(gap, 0.0))
            )
    return phrases


def cut_phrases_by_period(session: Session, max_p: int = 15) -> list[Phrase]:
    """Fallback: cut a motif run at its best self-similarity lag."""
    motifs = [t for t in session.tokens if t.kind == "motif"]
    if len(motifs) < 4:
        return cut_phrases(session)
    ids = [int(t.motif) for t in motifs if t.motif is not None]
    best_p, best_score = 1, 1.0
    for p in range(2, min(max_p, len(ids) // 2) + 1):
        score = normalised_edit(ids[:-p], ids[p:])
        if score < best_score:
            best_score, best_p = score, p
    phrases: list[Phrase] = []
    last_end: float | None = None
    for i in range(0, len(motifs), best_p):
        chunk = motifs[i : i + best_p]
        m = motif_string(chunk)
        if not m:
            continue
        gap = 0.0 if last_end is None else chunk[0].t0 - last_end
        phrases.append(Phrase(t0=chunk[0].t0, t1=chunk[-1].t1, motifs=m, gap_before=max(gap, 0.0)))
        last_end = chunk[-1].t1
    return phrases


def phrase_feature_rows(phrases: Sequence[Phrase]) -> list[PhraseFeatures]:
    rows: list[PhraseFeatures] = []
    for i, ph in enumerate(phrases):
        if i == 0:
            rows.append(
                PhraseFeatures(
                    d1=1.0, d2=1.0, dmin3=1.0, gap=0.0, len_ratio=0.0, t0=ph.t0, session_start=True
                )
            )
            continue
        prev = phrases[i - 1]
        d1 = normalised_edit(ph.motifs, prev.motifs)
        d2 = normalised_edit(ph.motifs, phrases[i - 2].motifs) if i >= 2 else 1.0
        recent = phrases[max(0, i - 3) : i]
        dmin3 = min(normalised_edit(ph.motifs, r.motifs) for r in recent)
        gap = float(np.log1p(ph.gap_before))
        len_ratio = float(np.log(max(len(ph.motifs), 1) / max(len(prev.motifs), 1)))
        rows.append(
            PhraseFeatures(
                d1=d1, d2=d2, dmin3=dmin3, gap=gap, len_ratio=len_ratio, t0=ph.t0, session_start=False
            )
        )
    return rows


def features_matrix(rows: Sequence[PhraseFeatures]) -> NDArray[np.floating]:
    return np.asarray(
        [[r.d1, r.d2, r.dmin3, r.gap, r.len_ratio] for r in rows], dtype=float
    )


def standardise(
    X: NDArray[np.floating],
    mean: NDArray[np.floating] | None = None,
    std: NDArray[np.floating] | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    if mean is None:
        mean = X.mean(axis=0)
    if std is None:
        std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (X - mean) / std, mean, std


def phrases_to_jsonable(phrases: Sequence[Phrase]) -> list[dict]:
    return [{**asdict(p), "motifs": list(p.motifs)} for p in phrases]
