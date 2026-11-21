"""Universal site + wall-clock start helpers."""

from __future__ import annotations

from database.recording_when import (
    normalize_start_prefix,
    parse_recording_when,
    start_filter_sql,
    yymmdd_hhmmss_to_start,
)


def test_yymmdd_hhmmss_to_start():
    assert yymmdd_hhmmss_to_start("230426", "193500") == "2023-04-26 19:35:00"
    assert yymmdd_hhmmss_to_start("230426") == "2023-04-26"
    assert yymmdd_hhmmss_to_start(None, "193500") is None


def test_normalize_iso_and_partials():
    assert normalize_start_prefix("2023-04-26 19:35:00") == "2023-04-26 19:35:00"
    assert normalize_start_prefix("2023-04-26T19:35:00") == "2023-04-26 19:35:00"
    assert normalize_start_prefix("2023-04-26") == "2023-04-26"
    assert normalize_start_prefix("2023-04-26 19") == "2023-04-26 19"
    assert normalize_start_prefix("2023-04-26 19:35") == "2023-04-26 19:35"
    assert normalize_start_prefix("not-a-date") is None


def test_parse_iso_with_default_site():
    got = parse_recording_when("2023-04-26 19:35:00")
    assert got.site == "H"
    assert got.start == "2023-04-26 19:35:00"


def test_parse_site_prefix():
    got = parse_recording_when("H 2023-04-26")
    assert got.site == "H"
    assert got.start == "2023-04-26"


def test_parse_legacy_socal_stem():
    got = parse_recording_when("SOCAL_H_76_230426_193500_df100.x.x.wav")
    assert got.site == "H"
    assert got.start == "2023-04-26 19:35:00"


def test_parse_partial_socal_date_only():
    got = parse_recording_when("SOCAL_H_76_230426")
    assert got.site == "H"
    assert got.start == "2023-04-26"


def test_start_filter_sql_equality_vs_prefix():
    frag, arg = start_filter_sql("2023-04-26 19:35:00")
    assert frag == "start_time = ?"
    assert arg == "2023-04-26 19:35:00"

    frag, arg = start_filter_sql("2023-04-26")
    assert frag == "start_time LIKE ?"
    assert arg == "2023-04-26%"

    frag, arg = start_filter_sql("2023-04-26T19:35:00")
    assert frag == "start_time = ?"
    assert arg == "2023-04-26 19:35:00"
