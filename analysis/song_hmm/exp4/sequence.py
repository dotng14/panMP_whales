"""Placements → events → motif token sequences with gap tokens."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .paths import BREAK_S, GAP_L, GAP_S


GAP_TOKENS = ("GAP_S", "GAP_L", "BREAK")


@dataclass
class Event:
    t0: float
    t1: float
    motif: int
    energy: float
    amplitude: float
    window_id: str


@dataclass
class Token:
    kind: str  # "motif" | "GAP_S" | "GAP_L" | "BREAK"
    t0: float
    t1: float
    motif: int | None = None
    energy: float = 0.0


@dataclass
class Session:
    tokens: list[Token]
    t0: float
    t1: float


def _window_index(windows: Sequence[Any]) -> dict[str, Any]:
    return {w.window_id: w for w in windows}


def load_matchings(path: Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text())


def load_template_widths(dictionary_dir: Path) -> dict[int, int]:
    summary = json.loads((Path(dictionary_dir) / "summary.json").read_text())
    return {int(a["template_id"]): int(a["width"]) for a in summary.get("atoms") or []}


def placements_to_events(
    matchings: Sequence[dict[str, Any]],
    windows: Sequence[Any],
    template_widths: dict[int, int],
    *,
    drop_negative_amp: bool = True,
) -> list[Event]:
    """Convert MP placements to absolute-time events."""
    by_id = _window_index(windows)
    events: list[Event] = []
    for row in matchings:
        wid = row["window_id"]
        w = by_id.get(wid)
        if w is None:
            continue
        amp = float(row["amplitude"])
        if drop_negative_amp and amp < 0:
            continue
        tid = int(row["template_id"])
        width = int(template_widths.get(tid, 10))
        sec_per_bin = float(w.overlap) / float(w.sample_freq) if w.sample_freq else 0.0
        t0 = (int(w.start_sample) - 1) / float(w.sample_freq) + int(row["t_start"]) * sec_per_bin
        t1 = t0 + width * sec_per_bin
        events.append(
            Event(
                t0=float(t0),
                t1=float(t1),
                motif=tid,
                energy=float(row.get("energy") or 0.0),
                amplitude=amp,
                window_id=str(wid),
            )
        )
    events.sort(key=lambda e: (e.t0, -e.energy))
    return events


def energy_threshold(events: Sequence[Event], *, q_low: float = 0.2) -> float:
    """Valley-style threshold: keep events above the low quantile of energy."""
    if not events:
        return 0.0
    energies = np.array([e.energy for e in events], dtype=float)
    return float(np.quantile(energies[energies > 0], q_low)) if np.any(energies > 0) else 0.0


def clean_events(
    events: Sequence[Event],
    *,
    energy_min: float | None = None,
    overlap_frac: float = 0.5,
) -> list[Event]:
    kept = list(events)
    if energy_min is not None:
        kept = [e for e in kept if e.energy >= energy_min]
    kept.sort(key=lambda e: (e.t0, -e.energy))
    out: list[Event] = []
    for e in kept:
        drop = False
        for prev in out:
            overlap = min(e.t1, prev.t1) - max(e.t0, prev.t0)
            if overlap <= 0:
                continue
            shorter = min(e.t1 - e.t0, prev.t1 - prev.t0)
            if shorter > 0 and overlap / shorter >= overlap_frac:
                if e.energy <= prev.energy:
                    drop = True
                    break
                # replace weaker earlier event
                out.remove(prev)
                break
        if not drop:
            out.append(e)
    out.sort(key=lambda e: e.t0)
    return out


def events_to_sessions(
    events: Sequence[Event],
    *,
    gap_s: tuple[float, float] = GAP_S,
    gap_l: tuple[float, float] = GAP_L,
    break_s: float = BREAK_S,
) -> list[Session]:
    if not events:
        return []
    tokens: list[Token] = []
    for i, e in enumerate(events):
        if i > 0:
            gap = e.t0 - events[i - 1].t1
            if gap > break_s:
                tokens.append(Token(kind="BREAK", t0=events[i - 1].t1, t1=e.t0))
            elif gap >= gap_l[0]:
                tokens.append(Token(kind="GAP_L", t0=events[i - 1].t1, t1=e.t0))
            elif gap >= gap_s[0]:
                tokens.append(Token(kind="GAP_S", t0=events[i - 1].t1, t1=e.t0))
        tokens.append(
            Token(kind="motif", t0=e.t0, t1=e.t1, motif=e.motif, energy=e.energy)
        )

    sessions: list[Session] = []
    cur: list[Token] = []
    for tok in tokens:
        if tok.kind == "BREAK":
            if cur:
                sessions.append(Session(tokens=cur, t0=cur[0].t0, t1=cur[-1].t1))
                cur = []
            continue
        cur.append(tok)
    if cur:
        sessions.append(Session(tokens=cur, t0=cur[0].t0, t1=cur[-1].t1))
    return sessions


def vocabulary(sessions: Sequence[Session]) -> list[str]:
    motifs = sorted({t.motif for s in sessions for t in s.tokens if t.kind == "motif" and t.motif is not None})
    return [f"M{m}" for m in motifs] + list(GAP_TOKENS)


def encode_session(session: Session, vocab: Sequence[str]) -> np.ndarray:
    index = {v: i for i, v in enumerate(vocab)}
    ids: list[int] = []
    for t in session.tokens:
        if t.kind == "motif":
            key = f"M{t.motif}"
        else:
            key = t.kind
        if key in index:
            ids.append(index[key])
    return np.asarray(ids, dtype=np.int64)


def session_to_dict(session: Session) -> dict[str, Any]:
    return {
        "t0": session.t0,
        "t1": session.t1,
        "tokens": [asdict(t) for t in session.tokens],
    }


def save_sequences(path: Path, sessions: Sequence[Session], meta: dict[str, Any] | None = None) -> None:
    payload = {
        "meta": meta or {},
        "sessions": [session_to_dict(s) for s in sessions],
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def load_sequences(path: Path) -> tuple[list[Session], dict[str, Any]]:
    payload = json.loads(Path(path).read_text())
    sessions: list[Session] = []
    for s in payload.get("sessions") or []:
        tokens = [Token(**t) for t in s["tokens"]]
        sessions.append(Session(tokens=tokens, t0=float(s["t0"]), t1=float(s["t1"])))
    return sessions, payload.get("meta") or {}


def motif_string(tokens: Iterable[Token]) -> tuple[int, ...]:
    return tuple(t.motif for t in tokens if t.kind == "motif" and t.motif is not None)
