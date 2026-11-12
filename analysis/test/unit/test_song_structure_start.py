"""Where a song-structure CSV's times start: the WAV stem, else ``wav_starts.json``."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from app.services import song_structure

START = datetime(2023, 4, 27, 1, 11, 15, tzinfo=timezone.utc).timestamp()


def test_start_from_the_stem():
    assert song_structure.wav_start_unix("SOCAL_H_76_230427_011115.x") == START


def test_start_from_wav_starts_json_for_a_stem_without_a_date(tmp_path, monkeypatch):
    starts = tmp_path / "wav_starts.json"
    monkeypatch.setattr(song_structure, "WAV_STARTS_FILE", starts)
    assert song_structure.wav_start_unix("wav23") is None
    starts.write_text(json.dumps({"wav23": "230427_011115"}))
    assert song_structure.wav_start_unix("wav23") == START
    assert song_structure.wav_start_unix("other") is None
