"""Unit tests for motif sequence and phrases."""

from __future__ import annotations

from analysis.song_hmm.exp4.phrases import cut_phrases, normalised_edit, phrase_feature_rows
from analysis.song_hmm.exp4.sequence import Event, Session, Token, clean_events, events_to_sessions


def test_gap_tokens_and_sessions():
    events = [
        Event(0.0, 1.0, 0, 1.0, 1.0, "w"),
        Event(2.0, 3.0, 1, 1.0, 1.0, "w"),  # gap 1.0 — no token
        Event(5.0, 6.0, 0, 1.0, 1.0, "w"),  # gap 2.0 — GAP_S
        Event(12.0, 13.0, 2, 1.0, 1.0, "w"),  # gap 6 — GAP_L
        Event(100.0, 101.0, 0, 1.0, 1.0, "w"),  # BREAK
    ]
    sessions = events_to_sessions(events)
    assert len(sessions) == 2
    kinds = [t.kind for t in sessions[0].tokens]
    assert "GAP_S" in kinds and "GAP_L" in kinds


def test_overlap_cleanup():
    events = [
        Event(0.0, 2.0, 0, 1.0, 1.0, "w"),
        Event(0.5, 2.5, 1, 2.0, 1.0, "w"),  # overlaps, higher energy
    ]
    kept = clean_events(events, energy_min=0.0)
    assert len(kept) == 1 and kept[0].motif == 1


def test_phrases_and_features():
    tokens = [
        Token("motif", 0, 1, 0, 1),
        Token("motif", 1, 2, 1, 1),
        Token("GAP_S", 2, 4),
        Token("motif", 4, 5, 0, 1),
        Token("motif", 5, 6, 1, 1),
    ]
    s = Session(tokens, 0, 6)
    phrases = cut_phrases(s)
    assert len(phrases) == 2
    assert normalised_edit(phrases[0].motifs, phrases[1].motifs) == 0.0
    feats = phrase_feature_rows(phrases)
    assert feats[1].d1 == 0.0
