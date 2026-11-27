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
    from app.services import library_service

    library = tmp_path / "library"
    library.mkdir()
    write_wav(library / "SOCAL_H_76_230426_054845_df20.x.wav")
    write_wav(library / "SOCAL_H_76_230426_193500_df100.x.x.wav")
    write_wav(library / "SOCAL_H_76_230427_014845.x.wav")
    write_wav(library / "SOCAL_H_81_260203_170123_df20.x.wav")
    write_wav(library / "wav23.wav")

    db = tmp_path / "catalog.sqlite"
    monkeypatch.setattr(config, "CATALOG_DB", db)
    monkeypatch.setattr(library_service, "CATALOG_DB", db)
    monkeypatch.setattr(config, "CATALOG_RUNS_ROOTS", [])
    monkeypatch.setattr(config, "CATALOG_CSV_ROOTS", [])
    monkeypatch.setattr(config, "GPL_RESULT_DIRNAME", "result")
    monkeypatch.setattr(library_service, "GPL_RESULT_DIRNAME", "result")
    # Fresh catalog per test — clear the one-shot ready flag.
    library_service._catalog_ready = False
    roots = config.WAV_LIBRARY_ROOTS
    saved = list(roots)
    roots[:] = [library]
    yield TestClient(app)
    roots[:] = saved
    library_service._catalog_ready = False


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
            "SELECT family, site, deployment, df, channel, start_time FROM wav_files WHERE stem LIKE ?",
            ("%193500_df100%",),
        ).fetchone()
        assert dict(row) == {
            "family": "SOCAL",
            "site": "H",
            "deployment": "76",
            "df": "100",
            "channel": "x.x",
            "start_time": "2023-04-26 19:35:00",
        }
        blank = conn.execute(
            "SELECT family FROM wav_files WHERE stem = ?", ("wav23",)
        ).fetchone()
        assert blank["family"] == ""


def test_match_by_site_and_start_prefix(client):
    day = client.get(
        "/api/library/match",
        params={"site": "H", "start": "2023-04-26"},
    ).json()
    assert day["total"] == 2
    assert {e["startTime"] for e in day["entries"]} == {
        "2023-04-26 05:48:45",
        "2023-04-26 19:35:00",
    }

    exact = client.get(
        "/api/library/match",
        params={"site": "H", "start": "2023-04-26 19:35:00"},
    ).json()
    assert exact["total"] == 1
    assert exact["entries"][0]["name"] == "SOCAL_H_76_230426_193500_df100.x.x.wav"
    assert exact["entries"][0]["startTime"] == "2023-04-26 19:35:00"


def test_match_start_with_t_separator(client):
    body = client.get(
        "/api/library/match",
        params={"site": "H", "start": "2023-04-26T19:35:00"},
    ).json()
    assert body["total"] == 1
    assert body["entries"][0]["startTime"] == "2023-04-26 19:35:00"


def test_siteh_auto_index_skipped(tmp_path, monkeypatch):
    """Site H under /mnt/sdb (or /mnt/sda) must not be rglob'd by ensure_wav_component_index."""
    from pathlib import Path
    from unittest.mock import MagicMock

    from app.services import library_service

    fake_root = MagicMock()
    fake_root.is_dir.return_value = True
    fake_root.is_absolute.return_value = True
    fake_root.__str__.return_value = "/mnt/sdb/SiteH"
    fake_root.resolve.return_value = Path("/mnt/sdb/SiteH")

    monkeypatch.setattr(config, "CATALOG_DB", tmp_path / "catalog.sqlite")
    roots = config.WAV_LIBRARY_ROOTS
    saved = list(roots)
    roots[:] = [fake_root]
    calls: list = []

    def boom(conn, root, *, result_dirname):
        calls.append(root)
        raise AssertionError("must not rglob Site H")

    monkeypatch.setattr(library_service, "index_wav_root", boom)

    with closing(catalog.connect(config.CATALOG_DB)) as conn:
        library_service.ensure_wav_component_index(conn)
        conn.commit()

    assert calls == []
    roots[:] = saved


def test_siteh_cache_uses_local_fallback(tmp_path, monkeypatch):
    """External /mnt/sdb/SiteH caches on local SSD (no PermissionError / exFAT writes)."""
    from pathlib import Path

    from app.services import gpl_cache

    fallback = tmp_path / "gpl_cache"
    monkeypatch.setattr(gpl_cache, "GPL_FALLBACK_CACHE_DIR", fallback)
    roots = config.WAV_LIBRARY_ROOTS
    saved = list(roots)
    roots[:] = [Path("/mnt/sdb/SiteH")]
    try:
        wav = Path("/mnt/sdb/SiteH/SOCAL_H_76/SOCAL_H_76_230426_193500_df100.x.x.wav")
        got = gpl_cache.cache_dir_for(wav)
        assert got == fallback / "SiteH" / "SOCAL_H_76" / "SOCAL_H_76_230426_193500_df100.x.x.gpl"
    finally:
        roots[:] = saved


def test_match_filters_by_cached_status(client, monkeypatch):
    from app.services import library_service

    monkeypatch.setattr(
        library_service,
        "is_gpl_cached",
        lambda p: "230426_193500" in p.name,
    )

    ready = client.get("/api/library/match", params={"site": "H", "cached": "true"}).json()
    assert ready["total"] == 1
    assert all(e["cached"] is True for e in ready["entries"])
    assert "230426_193500" in ready["entries"][0]["name"]

    not_ready = client.get("/api/library/match", params={"site": "H", "cached": "false"}).json()
    assert not_ready["total"] == 3
    assert all(e["cached"] is False for e in not_ready["entries"])
    assert all("230426_193500" not in e["name"] for e in not_ready["entries"])

