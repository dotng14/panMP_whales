"""Write aa32a7f / catalog-compatible placement and song-structure CSVs."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from analysis.song_hmm.exp1_slot_hmm.labels import recording_start_unix
from analysis.song_hmm.exp4.sequence import Session, Token, load_sequences


@dataclass
class Placement:
    motif: int
    start_sec: float
    end_sec: float
    unexplained: float | None = None


@dataclass
class MotifSegmentRow:
    motif: int
    song: int
    theme: int
    phrase_id: int
    phrase_count: int
    start_unix: float
    end_unix: float


@dataclass
class Interval:
    index: int  # motif id, theme id (1-based), or song number
    start_sec: float
    end_sec: float


def wav_start_unix(stem: str) -> float:
    return float(recording_start_unix(stem))


def tokens_to_placements(sessions: Sequence[Session]) -> list[Placement]:
    """Motif tokens from cleaned Exp5 sequences → placement rows."""
    out: list[Placement] = []
    for s in sessions:
        for t in s.tokens:
            if t.kind != "motif" or t.motif is None:
                continue
            if t.t1 <= t.t0:
                continue
            out.append(Placement(motif=int(t.motif), start_sec=float(t.t0), end_sec=float(t.t1)))
    out.sort(key=lambda p: (p.start_sec, p.end_sec, p.motif))
    return out


def load_all_exp5_sessions(sequences_dir: Path) -> list[Session]:
    sessions: list[Session] = []
    for path in sorted(sequences_dir.glob("193500_h*.json")):
        if path.name.endswith("_phrases.json"):
            continue
        sess, _ = load_sequences(path)
        sessions.extend(sess)
    return sessions


def attach_unexplained_from_mp(
    placements: Sequence[Placement],
    mp_dir: Path,
) -> list[Placement]:
    """Best-effort: match (motif, approx start) to MP rows for unexplained."""
    # Build lookup of motif -> list of (start_sec estimate not available without windows);
    # instead index by motif_id and unexplained from matchings; fill when unique motif times
    # are not recoverable cheaply. Use window-free approach: collect unexplained by
    # (template_id) average — not ideal. Better: leave None unless we can join.
    #
    # Practical join: matchings have motif/template_id but times need windows.
    # Skip heavy reload; leave unexplained blank unless a sidecar exists.
    unexplained_by_motif: dict[int, list[float]] = {}
    for path in sorted(mp_dir.glob("*/matchings.json")):
        try:
            rows = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for row in rows:
            tid = int(row.get("template_id", row.get("motif_id", -1)))
            u = row.get("unexplained")
            if u is None:
                continue
            unexplained_by_motif.setdefault(tid, []).append(float(u))
    # Without exact time join, do not invent per-placement values.
    return list(placements)


def write_placements_txt(path: Path, placements: Sequence[Placement]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"{p.motif} {p.start_sec:.10f} {p.end_sec:.10f}\n" for p in placements]
    path.write_text("".join(lines))


def write_placements_csv(path: Path, placements: Sequence[Placement]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["motif", "start_sec", "end_sec", "unexplained"])
        for p in placements:
            u = "" if p.unexplained is None else f"{p.unexplained:.6g}"
            w.writerow([p.motif, f"{p.start_sec:.10f}", f"{p.end_sec:.10f}", u])


def write_motif_segments_csv(path: Path, rows: Sequence[MotifSegmentRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(
            ["motif", "song", "theme", "phrase_id", "phrase_count", "start_unix", "end_unix"]
        )
        for r in rows:
            w.writerow(
                [
                    r.motif,
                    r.song,
                    r.theme,
                    r.phrase_id,
                    r.phrase_count,
                    f"{r.start_unix:.6f}",
                    f"{r.end_unix:.6f}",
                ]
            )


def write_segments_csv(
    path: Path,
    *,
    origin_unix: float,
    motifs: Sequence[Placement],
    themes: Sequence[Interval],
    songs: Sequence[Interval],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["type", "segment index", "start_unix", "end_unix"])
        for p in motifs:
            w.writerow(
                [
                    "motif",
                    p.motif,
                    f"{origin_unix + p.start_sec:.6f}",
                    f"{origin_unix + p.end_sec:.6f}",
                ]
            )
        for th in themes:
            w.writerow(
                [
                    "theme",
                    th.index,
                    f"{origin_unix + th.start_sec:.6f}",
                    f"{origin_unix + th.end_sec:.6f}",
                ]
            )
        for sg in songs:
            w.writerow(
                [
                    "song",
                    sg.index,
                    f"{origin_unix + sg.start_sec:.6f}",
                    f"{origin_unix + sg.end_sec:.6f}",
                ]
            )


def write_segments_motifs_only(
    path: Path,
    *,
    origin_unix: float,
    placements: Sequence[Placement],
) -> None:
    write_segments_csv(path, origin_unix=origin_unix, motifs=placements, themes=[], songs=[])


def _theme_id(label: Any) -> int:
    """Parse 'T3' / 3 / '3' → 1-based theme int."""
    if isinstance(label, (int, float)):
        return int(label)
    s = str(label).strip()
    if s.startswith("T") or s.startswith("t"):
        return int(s[1:].split("_")[0].split(".")[0])
    return int(float(s))


def theme_intervals_from_eval(themes: Sequence[dict]) -> list[Interval]:
    out: list[Interval] = []
    for th in themes:
        tid = _theme_id(th["theme"])
        t0, t1 = float(th["t0"]), float(th["t1"])
        if t1 > t0:
            out.append(Interval(index=tid, start_sec=t0, end_sec=t1))
    out.sort(key=lambda x: (x.start_sec, x.end_sec))
    return out


def song_intervals_from_themes(themes: Sequence[Interval]) -> list[Interval]:
    """New song at every visit of theme 1 (and first segment if it is not theme 1)."""
    if not themes:
        return []
    songs: list[Interval] = []
    song_id = 0
    cur_start: float | None = None
    cur_end: float | None = None

    def flush() -> None:
        nonlocal song_id, cur_start, cur_end
        if cur_start is None or cur_end is None:
            return
        song_id += 1
        songs.append(Interval(index=song_id, start_sec=cur_start, end_sec=cur_end))
        cur_start, cur_end = None, None

    for th in themes:
        if th.index == 1:
            flush()
            cur_start, cur_end = th.start_sec, th.end_sec
        else:
            if cur_start is None:
                # leading non-T1 material: start song 1 here
                cur_start, cur_end = th.start_sec, th.end_sec
            else:
                cur_end = max(cur_end or th.end_sec, th.end_sec)
    flush()
    return songs


def assign_song_theme(
    placements: Sequence[Placement],
    themes: Sequence[Interval],
    *,
    origin_unix: float,
) -> tuple[list[MotifSegmentRow], list[Interval]]:
    """Assign each placement to a theme interval; songs from theme-1 returns."""
    songs = song_intervals_from_themes(themes)
    rows: list[MotifSegmentRow] = []
    for p in placements:
        theme = _theme_at(themes, p.start_sec)
        song = _song_at(songs, p.start_sec)
        if theme is None:
            # placement outside decoded theme spans (e.g. train-only hours): skip structure rows
            continue
        if song is None:
            song = 1
        rows.append(
            MotifSegmentRow(
                motif=p.motif,
                song=song,
                theme=theme,
                phrase_id=theme,
                phrase_count=1,
                start_unix=origin_unix + p.start_sec,
                end_unix=origin_unix + p.end_sec,
            )
        )
    return rows, songs


def _theme_at(themes: Sequence[Interval], t: float) -> int | None:
    for th in themes:
        if th.start_sec <= t < th.end_sec:
            return th.index
    # allow end-inclusive on last interval
    for th in reversed(themes):
        if th.start_sec <= t <= th.end_sec:
            return th.index
    return None


def _song_at(songs: Sequence[Interval], t: float) -> int | None:
    for sg in songs:
        if sg.start_sec <= t < sg.end_sec:
            return sg.index
    for sg in reversed(songs):
        if sg.start_sec <= t <= sg.end_sec:
            return sg.index
    return None


def load_eval_a_themes(eval_path: Path) -> list[dict]:
    data = json.loads(eval_path.read_text())
    return list(data.get("themes") or [])
