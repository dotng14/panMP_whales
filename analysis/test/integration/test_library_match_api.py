"""Component facets / match API for the recording picker."""

from __future__ import annotations

import os

os.environ.setdefault("GPL_COMPUTE", "cpu")

from contextlib import closing

import pytest
from fastapi.testclient import TestClient

from analysis.test.synthetic import write_wav
from app import config
from app.main import app
from database import catalog


@pytest.fixture
def client(tmp_path, monkeypatch):
    library = tmp_path / "library"
    library.mkdir()
    write_wav(library / "SOCAL_H_76_230426_054845_df20.x.wav")
    write_wav(library / "SOCAL_H_76_230426_193500_df100.x.x.wav")
    write_wav(library / "SOCAL_H_76_230427_014845.x.wav")
    write_wav(library / "SOCAL_H_81_260203_170123_df20.x.wav")
    write_wav(library / "wav23.wav")

    monkeypatch.setattr(config, "CATALOG_DB", tmp_path / "catalog.sqlite")
    monkeypatch.setattr(config, "CATALOG_RUNS_ROOTS", [])
    monkeypatch.setattr(config, "CATALOG_CSV_ROOTS", [])
    monkeypatch.setattr(config, "GPL_RESULT_DIRNAME", "result")
    roots = config.WAV_LIBRARY_ROOTS
    saved = list(roots)
    roots[:] = [library]
    yield TestClient(app)
    roots[:] = saved


def test_match_all_and_socal_filter(client):
    all_body = client.get("/api/library/match").json()
    assert all_body["total"] == 5
    names = {e["name"] for e in all_body["entries"]}
    assert "wav23.wav" in names

    body = client.get("/api/library/match", params={"family": "SOCAL", "deployment": "76"}).json()
    assert body["total"] == 3
    assert all("SOCAL_H_76" in e["name"] for e in body["entries"])
    assert all(e["fileId"] and e["components"]["deployment"] == "76" for e in body["entries"])


def test_match_cascades_to_one_file(client):
    body = client.get(
        "/api/library/match",
        params={
            "family": "SOCAL",
            "deployment": "76",
            "date": "230426",
            "time": "193500",
            "df": "100",
        },
    ).json()
    assert body["total"] == 1
    assert body["entries"][0]["name"] == "SOCAL_H_76_230426_193500_df100.x.x.wav"


def test_facets_shrink_with_filters(client):
    open_facets = client.get("/api/library/facets", params={"family": "SOCAL"}).json()["facets"]
    deployments = {v["value"] for v in open_facets["deployment"]}
    assert deployments >= {"76", "81"}

    narrow = client.get(
        "/api/library/facets",
        params={"family": "SOCAL", "deployment": "76", "date": "230426"},
    ).json()["facets"]
    times = {v["value"]: v["count"] for v in narrow["time"]}
    assert times == {"054845": 1, "193500": 1}
    dfs = {v["value"] for v in narrow["df"]}
    assert dfs == {"20", "100"}


def test_substring_q_for_non_socal(client):
    body = client.get("/api/library/match", params={"q": "wav23"}).json()
    assert body["total"] == 1
    assert body["entries"][0]["name"] == "wav23.wav"


def test_components_persisted_on_index(client, tmp_path):
    client.get("/api/library/match")  # triggers ensure index + commit
    with closing(catalog.connect(config.CATALOG_DB, read_only=True)) as conn:
        row = conn.execute(
            "SELECT family, site, deployment, df, channel FROM wav_files WHERE stem LIKE ?",
            ("%193500_df100%",),
        ).fetchone()
        assert dict(row) == {
            "family": "SOCAL",
            "site": "H",
            "deployment": "76",
            "df": "100",
            "channel": "x.x",
        }
        blank = conn.execute(
            "SELECT family FROM wav_files WHERE stem = ?", ("wav23",)
        ).fetchone()
        assert blank["family"] == ""
