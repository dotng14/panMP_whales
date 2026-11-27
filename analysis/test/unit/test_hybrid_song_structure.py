"""Hybrid song structure: only conforming windows become songs, every window keeps its themes."""

from __future__ import annotations

import csv
from pathlib import Path

from app.services import hybrid_song_service

FIELDS = [
    "start_sec",
    "end_sec",
    "min_margin",
    "shuffles_at_or_above",
    "theme_starts_sec",
    "theme_minutes",
]


def _write_decoded(out: Path, rows: list[dict]) -> None:
    with (out / "decoded_songs.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_themes_are_kept_from_non_conforming_windows(tmp_path):
    _write_decoded(tmp_path, [
        {
            "start_sec": 0,
            "end_sec": 1980,
            "min_margin": 5,
            "shuffles_at_or_above": 0,
            "theme_starts_sec": "0/1800/1860/1920",
            "theme_minutes": "30/1/1/1",
        },
        {
            "start_sec": 1980,
            "end_sec": 3900,
            "min_margin": -30,
            "shuffles_at_or_above": -1,
            "theme_starts_sec": "1980/3720/3780/3840",
            "theme_minutes": "29/1/1/1",
        },
    ])

    payload = hybrid_song_service._build_song_structure_json(
        tmp_path, "SOCAL_H_76_230426_054845_df20.x.wav"
    )

    assert [(s["song"], s["startSec"]) for s in payload["songs"]] == [(1, 0.0)]
    starts = [t["startSec"] for t in payload["themes"]]
    assert starts == sorted(starts)
    assert 1980.0 in starts  # theme from the rejected window is still present
    assert len(payload["themes"]) == 8
    assert payload["version"] == hybrid_song_service.SONG_STRUCTURE_VERSION


def test_stale_cached_song_structure_is_rebuilt_on_load(tmp_path):
    _write_decoded(tmp_path, [
        {
            "start_sec": 0,
            "end_sec": 1980,
            "min_margin": 5,
            "shuffles_at_or_above": 0,
            "theme_starts_sec": "0/1800/1860/1920",
            "theme_minutes": "30/1/1/1",
        },
        {
            "start_sec": 1980,
            "end_sec": 3900,
            "min_margin": -30,
            "shuffles_at_or_above": -1,
            "theme_starts_sec": "1980/3720/3780/3840",
            "theme_minutes": "29/1/1/1",
        },
    ])
    # Simulate an older cached JSON written without a version (only 4 themes).
    (tmp_path / "song_structure.json").write_text(
        '{"wavStartUnix": 0, "units": [], "songs": [{"song": 1, "startSec": 0, "endSec": 1980}], "themes": [{"theme": 0, "startSec": 0, "endSec": 1800}]}'
    )

    rebuilt = hybrid_song_service.load_or_rebuild_song_structure(
        tmp_path, "SOCAL_H_76_230426_054845_df20.x.wav"
    )
    assert rebuilt is not None
    assert rebuilt["version"] == hybrid_song_service.SONG_STRUCTURE_VERSION
    assert len(rebuilt["themes"]) == 8

