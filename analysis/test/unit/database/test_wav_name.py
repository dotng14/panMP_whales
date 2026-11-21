"""SOCAL wav-name parser matrix (database.wav_name)."""

from __future__ import annotations

import pytest

from database.wav_name import parse_wav_name


@pytest.mark.parametrize(
    "name, expected",
    [
        (
            "SOCAL_H_76_230426_193500_df100.x.x.wav",
            {
                "family": "SOCAL",
                "site": "H",
                "deployment": "76",
                "disk": None,
                "date": "230426",
                "time": "193500",
                "df": "100",
                "channel": "x.x",
                "resample": None,
            },
        ),
        (
            "SOCAL_H_76_230427_014845.x.wav",
            {
                "family": "SOCAL",
                "site": "H",
                "deployment": "76",
                "date": "230427",
                "time": "014845",
                "df": "native",
                "channel": "x",
                "resample": None,
            },
        ),
        (
            "SOCAL_H_76_230427_014845.x_10kHz.wav",
            {
                "family": "SOCAL",
                "site": "H",
                "deployment": "76",
                "date": "230427",
                "time": "014845",
                "df": "native",
                "channel": "x",
                "resample": "10kHz",
            },
        ),
        (
            "SOCAL_H_75_HE_221016_231900_df20.x.wav",
            {
                "family": "SOCAL",
                "site": "H",
                "deployment": "75",
                "date": "221016",
                "time": "231900",
                "df": "20",
                "channel": "x",
            },
        ),
        (
            "SOCAL45H_DL37_120131_042345.df20.x.wav",
            {
                "family": "SOCAL",
                "site": "H",
                "deployment": "45",
                "disk": "DL37",
                "date": "120131",
                "time": "042345",
                "df": "20",
                "channel": "x",
            },
        ),
        (
            "SOCAL27H_DL19_disk01_080803_173000.df100.x.wav",
            {
                "family": "SOCAL",
                "site": "H",
                "deployment": "27",
                "disk": "DL19",
                "date": "080803",
                "time": "173000",
                "df": "100",
                "channel": "x",
            },
        ),
        (
            "SOCAL18H_d01_070723_163700.d100.x.wav",
            {
                "family": "SOCAL",
                "site": "H",
                "deployment": "18",
                "disk": "d01",
                "date": "070723",
                "time": "163700",
                "df": "100",
                "channel": "x",
            },
        ),
        (
            "SOCAL38H_01_100410_080359.df100.x.wav",
            {
                "family": "SOCAL",
                "site": "H",
                "deployment": "38",
                "disk": "01",
                "date": "100410",
                "time": "080359",
                "df": "100",
                "channel": "x",
            },
        ),
        (
            "SOCAL_H_76_230426",
            {
                "family": "SOCAL",
                "site": "H",
                "deployment": "76",
                "date": "230426",
                "time": None,
                "df": None,
                "channel": None,
            },
        ),
        (
            "path/to/SOCAL_H_76_230426_054845_df20.x.wav",
            {
                "family": "SOCAL",
                "site": "H",
                "deployment": "76",
                "date": "230426",
                "time": "054845",
                "df": "20",
                "channel": "x",
            },
        ),
    ],
)
def test_parse_socal_matrix(name, expected):
    got = parse_wav_name(name)
    assert got is not None
    for key, value in expected.items():
        assert getattr(got, key) == value, key


def test_partial_date_time_in_paste():
    got = parse_wav_name("SOCAL_H_76_230426_193500")
    assert got is not None
    assert (got.date, got.time, got.df, got.channel) == ("230426", "193500", None, None)


def test_non_socal_returns_none():
    assert parse_wav_name("wav23.wav") is None
    assert parse_wav_name("CHNMS_NO_01_231105_154200_df32.x.wav") is None


def test_socal_prefix_only():
    got = parse_wav_name("SOCAL")
    assert got is not None and got.family == "SOCAL" and got.deployment is None
