"""Song-structure CSVs: where their times start, the long layout, and the Parquet copy."""

from __future__ import annotations

import json
from datetime import datetime, timezone

from app.services import song_structure
from database import catalog

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


LONG_CSV = """type,segment index,start_unix,end_unix
motif,26,{a},{b}
motif,35,{c},{d}
theme,0,{t0},{t1}
theme,1,{t1},{t2}
song,1,{t0},{t2}
"""


def test_long_layout_gives_theme_spans_and_tags_each_motif_by_its_start(tmp_path):
    t = lambda sec: f"{START + sec:.6f}"  # noqa: E731
    path = tmp_path / "SOCAL_H_76_230427_011115.x.motif_segments.csv"
    # Motif 35 straddles the theme boundary at 30 s; it starts in theme 0.
    path.write_text(LONG_CSV.format(a=t(1), b=t(2), c=t(29.5), d=t(30.7), t0=t(0), t1=t(30), t2=t(90)))

    payload = song_structure.load_song_structure_csv(path, path.name.split(".motif")[0])

    assert payload["themes"] == [
        {"theme": 0, "startSec": 0.0, "endSec": 30.0},
        {"theme": 1, "startSec": 30.0, "endSec": 90.0},
    ]
    assert payload["songs"] == [{"song": 1, "startSec": 0.0, "endSec": 90.0}]
    assert [(u["motif"], u["song"], u["theme"], u["phraseId"]) for u in payload["units"]] == [
        ("26", 1, 0, None),
        ("35", 1, 0, None),
    ]
    assert catalog.read_motif_csv_counts(path) == (2, 1)


def test_catalog_keeps_a_parquet_copy_that_reads_the_same_as_the_csv(tmp_path):
    t = lambda sec: f"{START + sec:.6f}"  # noqa: E731
    path = tmp_path / "SOCAL_H_76_230427_011115.x.motif_segments.csv"
    path.write_text(LONG_CSV.format(a=t(1), b=t(2), c=t(29.5), d=t(30.7), t0=t(0), t1=t(30), t2=t(90)))
    stem = path.name.split(".motif")[0]

    with catalog.connect(tmp_path / "db" / "c.sqlite") as conn:
        csv_id = catalog.upsert_annotation_csv(conn, path)
        row = catalog.get_annotation_csv(conn, csv_id)

    assert row["parquet_path"] == str(tmp_path / "db" / "segments" / f"{csv_id}.parquet")
    from_parquet = song_structure.load_song_structure_parquet(row["parquet_path"], stem)
    assert from_parquet == song_structure.load_song_structure_csv(path, stem)
