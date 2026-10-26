"""The catalog through the backend's HTTP API (FastAPI ``TestClient``).

The app is imported as ``app`` (``backend/`` is on ``pythonpath``). Each test
points ``app.config`` at a private copy of the session catalog and at a
one-file WAV library.
"""

from __future__ import annotations

import os

# Importing the app loads backend/.env, which may ask GPL_Python for CUDA at
# import time. These tests never compute a spectrogram; an exported value wins.
os.environ.setdefault("GPL_COMPUTE", "cpu")

import asyncio
import shutil
import sqlite3
import uuid
from contextlib import closing

import pytest
from fastapi.testclient import TestClient

from analysis.test.catalog_setup import CATALOG_STEM
from analysis.test.synthetic import write_wav
from app import config
from app.main import app
from app.routes import jobs as jobs_route
from app.routes import library as library_route
from app.services import job_store, pipeline_service
from app.services.gpl_cache import cache_dir_for
from app.services.library_service import encode_file_id


@pytest.fixture
def client(catalog_runs, tmp_path, monkeypatch):
    library = tmp_path / "library"
    library.mkdir()
    write_wav(library / f"{CATALOG_STEM}.wav")
    monkeypatch.setattr(config, "CATALOG_DB", catalog_runs.copy_db(tmp_path))
    monkeypatch.setattr(config, "CATALOG_RUNS_ROOTS", [catalog_runs.root / "runs"])
    monkeypatch.setattr(config, "CATALOG_CSV_ROOTS", [catalog_runs.csv.parent])
    monkeypatch.setattr(config, "GPL_RESULT_DIRNAME", "result")
    # library_service holds the same list object, so replace its contents.
    roots = config.WAV_LIBRARY_ROOTS
    saved = list(roots)
    roots[:] = [library]
    yield TestClient(app)
    roots[:] = saved


@pytest.fixture
def file_id(client) -> str:
    return encode_file_id(0, f"{CATALOG_STEM}.wav")


def ids(client):
    """``(build id, mp id, csv id)`` from the picker list."""
    items = client.get("/api/catalog/dictionaries").json()["dictionaries"]
    mp = next(d for d in items if d["source"] == "matching")
    csv_id = client.get("/api/catalog/csvs").json()["csvs"][0]["csv_id"]
    return mp["parent_dictionary_id"], mp["dictionary_id"], csv_id


def test_picker_lists_dictionaries_and_csvs(client, file_id):
    body = client.get("/api/library/analyses", params={"fileId": file_id}).json()
    assert body["stem"] == CATALOG_STEM and body["gplCached"] is False
    kinds = [(i["kind"], i.get("source")) for i in body["items"]]
    assert kinds == [("dictionary", "matching"), ("dictionary", "seed"), ("csv", None)]
    mp, built, csv = body["items"]
    assert mp["parentId"] == built["id"]
    assert built["nTemplates"] > 0 and built["nMatchings"] > 0 and built["nWindows"] == 3
    assert 0 < built["timeCoverage"] <= 1
    assert (csv["nRows"], csv["nSongs"], csv["named"], csv["displayName"]) == (6, 2, False, csv["name"])


def test_without_a_catalog(client, file_id, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "CATALOG_DB", tmp_path / "none.sqlite")
    assert client.get("/api/library/analyses", params={"fileId": file_id}).json()["items"] == []
    res = client.get("/api/catalog/dictionaries")
    assert res.status_code == 404 and "POST /api/catalog/reindex" in res.json()["detail"]


def test_rename_dictionary_csv_and_atom(client, file_id):
    built, _, csv_id = ids(client)
    named = client.patch(f"/api/catalog/dictionaries/{built}", json={"label": "  first build "}).json()
    assert (named["label"], named["named"], named["displayName"]) == ("first build", True, "first build")
    cleared = client.patch(f"/api/catalog/dictionaries/{built}", json={"label": ""}).json()
    assert (cleared["label"], cleared["named"]) == (None, False)
    assert client.patch(f"/api/catalog/csvs/{csv_id}", json={"label": "HMM v1"}).json()["displayName"] == "HMM v1"
    items = client.get("/api/library/analyses", params={"fileId": file_id}).json()["items"]
    assert items[-1]["label"] == "HMM v1"

    atom = client.get(f"/api/catalog/dictionaries/{built}/templates").json()["templates"][0]["template_id"]
    res = client.patch(f"/api/catalog/dictionaries/{built}/templates/{atom}", json={"label": "upsweep"})
    assert res.json() == {"dictionaryId": built, "templateId": atom, "label": "upsweep"}
    placements = client.get(f"/api/catalog/dictionaries/{built}/placements").json()["placements"]
    assert {p["label"] for p in placements if p["templateId"] == atom} == {"upsweep"}

    assert client.patch("/api/catalog/dictionaries/999", json={"label": "x"}).status_code == 404
    assert client.patch("/api/catalog/csvs/999", json={"label": "x"}).status_code == 404
    res = client.patch(f"/api/catalog/dictionaries/{built}/templates/999", json={"label": "x"})
    assert res.status_code == 404 and "has no atom 999" in res.json()["detail"]


def test_placements(client):
    _, mp, _ = ids(client)
    body = client.get(f"/api/catalog/dictionaries/{mp}/placements").json()
    assert body["dictionary"]["source"] == "matching" and body["nWithoutTime"] == 0
    starts = [p["startSec"] for p in body["placements"]]
    assert starts == sorted(starts) and len(starts) == body["dictionary"]["nMatchings"]
    assert all(p["endSec"] > p["startSec"] for p in body["placements"])


def test_reads(client, catalog_runs):
    built, mp, csv_id = ids(client)
    assert client.get("/api/catalog/wavs").json() == {"wavs": []}
    assert client.get("/api/catalog/wavs/1/segmentations").json() == {"segmentations": []}
    run = client.get("/api/catalog/runs").json()["runs"][0]
    run_id = run["run_id"]
    assert run["out_dir"] == str(catalog_runs.build)
    templates = client.get(f"/api/catalog/runs/{run_id}/templates").json()["templates"]
    assert len(client.get(f"/api/catalog/runs/{run_id}/rounds").json()["rounds"]) == run["n_outer_completed"]
    tid = templates[0]["template_id"]
    assert client.get(f"/api/catalog/runs/{run_id}/templates/{tid}").json()["template_pk"] == templates[0]["template_pk"]
    data = client.get(f"/api/catalog/runs/{run_id}/templates/{tid}", params={"include": "data"})
    assert data.content.startswith(b"\x93NUMPY")
    assert data.headers["content-disposition"].endswith(f'run{run_id}_template{tid}.npy"')
    for path in ("/api/catalog/runs/999/templates", "/api/catalog/runs/999/rounds",
                 f"/api/catalog/runs/{run_id}/templates/999"):
        assert client.get(path).status_code == 404, path

    assert client.get(f"/api/catalog/dictionaries/{mp}").json()["parent_dictionary_id"] == built
    assert len(client.get(f"/api/catalog/dictionaries/{mp}/templates").json()["templates"]) == len(templates)
    assert len(client.get(f"/api/catalog/dictionaries/{mp}/windows").json()["windows"]) == 3
    lineage = client.get(f"/api/catalog/dictionaries/{mp}/lineage").json()["lineage"]
    assert [d["dictionary_id"] for d in lineage] == [built, mp]
    for suffix in ("", "/templates", "/windows", "/lineage", "/matchings", "/placements"):
        assert client.get(f"/api/catalog/dictionaries/999{suffix}").status_code == 404, suffix

    matchings = client.get(f"/api/catalog/dictionaries/{built}/matchings").json()["matchings"]
    first = matchings[0]
    by_atom = client.get(f"/api/catalog/dictionaries/{built}/matchings",
                         params={"template_pk": first["template_pk"]}).json()["matchings"]
    assert by_atom and all(m["template_pk"] == first["template_pk"] for m in by_atom)
    by_window = client.get(f"/api/catalog/dictionaries/{built}/matchings",
                           params={"window_id": first["window_id"]}).json()["matchings"]
    assert by_window and all(m["window_id"] == first["window_id"] for m in by_window)
    assert client.get(f"/api/catalog/matchings/{first['id']}").json()["t_start"] == first["t_start"]
    assert client.get("/api/catalog/matchings/99999").status_code == 404

    tool_runs = client.get("/api/catalog/tool-runs").json()["toolRuns"]
    assert [t["tool"] for t in tool_runs] == ["randomized_build", "matching_pursuit"]
    assert len(client.get("/api/catalog/tool-runs", params={"learning_run_id": run_id}).json()["toolRuns"]) == 1
    assert client.get(f"/api/catalog/tool-runs/{tool_runs[1]['tool_run_id']}").json()["input_dictionary_id"] == built
    assert client.get("/api/catalog/tool-runs/999").status_code == 404

    assert client.get(f"/api/catalog/csvs/{csv_id}").json()["wav_stem"] == CATALOG_STEM
    assert client.get("/api/catalog/csvs/999").status_code == 404


def test_template_file_missing(client):
    with closing(sqlite3.connect(config.CATALOG_DB)) as conn, conn:
        conn.execute("UPDATE templates SET npy_path = '/nonexistent/atom.npy'")
    run_id = client.get("/api/catalog/runs").json()["runs"][0]["run_id"]
    tid = client.get(f"/api/catalog/runs/{run_id}/templates").json()["templates"][0]["template_id"]
    res = client.get(f"/api/catalog/runs/{run_id}/templates/{tid}", params={"include": "data"})
    assert res.status_code == 404 and "Template file missing" in res.json()["detail"]


def test_feedback(client):
    built, _, _ = ids(client)
    matchings = client.get(f"/api/catalog/dictionaries/{built}/matchings").json()["matchings"]
    target = matchings[0]["id"]
    res = client.put(f"/api/catalog/matchings/{target}/feedback", json={"label": "good", "note": "clear"})
    assert (res.json()["feedback"], res.json()["feedback_note"]) == ("good", "clear")

    def count(feedback):
        url = f"/api/catalog/dictionaries/{built}/matchings"
        return len(client.get(url, params={"feedback": feedback}).json()["matchings"])

    assert (count("good"), count("bad"), count("unlabeled")) == (1, 0, len(matchings) - 1)
    assert client.put(f"/api/catalog/matchings/{target}/feedback", json={"label": "bad"}).json()["feedback"] == "bad"
    assert client.put("/api/catalog/matchings/99999/feedback", json={"label": "good"}).status_code == 404
    assert client.delete(f"/api/catalog/matchings/{target}/feedback").json() == {"matchingId": target, "deleted": True}
    assert client.delete(f"/api/catalog/matchings/{target}/feedback").json()["deleted"] is False
    assert client.delete("/api/catalog/matchings/99999/feedback").status_code == 404


def test_reindex(client, catalog_runs):
    indexed = client.post("/api/catalog/reindex").json()["indexed"]
    assert indexed["wavs"] == 1 and indexed["runs"] == 2 and indexed["csvs"] == 1
    assert indexed["skipped"] == []
    wavs = client.get("/api/catalog/wavs").json()["wavs"]
    assert [w["stem"] for w in wavs] == [CATALOG_STEM]


def test_song_structure_from_a_catalog_csv(client, catalog_runs, tmp_path):
    job_id = str(uuid.uuid4())
    job_store.create_job(job_id, f"{CATALOG_STEM}.wav", tmp_path)
    _, _, csv_id = ids(client)
    body = client.get(f"/api/jobs/{job_id}/song-structure", params={"csvId": csv_id}).json()
    assert len(body["units"]) == 6 and [s["song"] for s in body["songs"]] == [1, 2]
    assert body["units"][0]["startSec"] == pytest.approx(0.0)
    # Without csvId: the CSV named after the WAV under backend/data/song_structure (none here).
    assert client.get(f"/api/jobs/{job_id}/song-structure").status_code == 404
    assert client.get(f"/api/jobs/{job_id}/song-structure", params={"csvId": 999}).status_code == 404
    with closing(sqlite3.connect(config.CATALOG_DB)) as conn, conn:
        conn.execute("UPDATE annotation_csvs SET csv_path = '/nonexistent.motif_segments.csv'")
    res = client.get(f"/api/jobs/{job_id}/song-structure", params={"csvId": csv_id})
    assert res.status_code == 404 and "CSV file missing" in res.json()["detail"]


# ── GPL runs register themselves in the catalog ──────────────────────────────


def _segmentations(client) -> list[dict]:
    wavs = client.get("/api/catalog/wavs").json()["wavs"]
    assert [w["stem"] for w in wavs] == [CATALOG_STEM]
    return client.get(f"/api/catalog/wavs/{wavs[0]['wav_id']}/segmentations").json()["segmentations"]


def _library_wav(tmp_path):
    return tmp_path / "library" / f"{CATALOG_STEM}.wav"


def test_picker_says_whether_gpl_exists(client, file_id, monkeypatch):
    monkeypatch.setattr(library_route, "is_gpl_cached", lambda wav: True)
    assert client.get("/api/library/analyses", params={"fileId": file_id}).json()["gplCached"] is True


def test_gpl_run_registers_in_a_new_catalog(client, catalog_runs, tmp_path, monkeypatch):
    """The slow path: GPL runs into the cache, then the WAV and its output enter the catalog."""
    monkeypatch.setattr(config, "CATALOG_DB", tmp_path / "fresh" / "catalog.sqlite")
    wav = _library_wav(tmp_path)
    output = cache_dir_for(wav) / "output"
    output.mkdir(parents=True)
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "output").symlink_to(output, target_is_directory=True)
    (job_dir / "input.wav").symlink_to(wav)

    class FakeOrchestrator:
        def __init__(self, out_dir):
            self.out_dir = out_dir

        def process(self, _wav):
            shutil.copytree(catalog_runs.gpl, self.out_dir, dirs_exist_ok=True)

    import analysis.GPL_Python.process_v4 as process_v4

    monkeypatch.setattr(process_v4, "ProcessingOrchestrator", FakeOrchestrator)
    monkeypatch.setattr(pipeline_service, "build_manifest_from_outputs", lambda out, job_id: {"jobId": job_id})
    monkeypatch.setattr(pipeline_service, "finalize_gpl_cache", lambda *a, **k: None)
    job = job_store.create_job(str(uuid.uuid4()), wav.name, job_dir)
    job.library_wav_path = wav

    async def run():
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, pipeline_service._run_job_sync, job, loop)

    asyncio.run(run())
    assert job.status == "complete"
    [seg] = _segmentations(client)
    assert seg["output_dir"] == str(output.resolve()) and seg["n_windows"] == 3
    # Running again updates the same row.
    asyncio.run(run())
    assert len(_segmentations(client)) == 1


def test_registration_failure_does_not_fail_the_job(client, tmp_path, monkeypatch):
    def boom(_wav):
        raise RuntimeError("locked")

    monkeypatch.setattr(pipeline_service, "register_gpl_in_catalog", boom)
    monkeypatch.setattr(pipeline_service, "build_manifest_from_outputs", lambda out, job_id: {"jobId": job_id})
    monkeypatch.setattr(pipeline_service, "finalize_gpl_cache", lambda *a, **k: None)

    import analysis.GPL_Python.process_v4 as process_v4

    monkeypatch.setattr(process_v4, "ProcessingOrchestrator", lambda out: type("P", (), {"process": lambda self, w: None})())
    job = job_store.create_job(str(uuid.uuid4()), "x.wav", tmp_path)
    job.library_wav_path = _library_wav(tmp_path)

    async def run():
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, pipeline_service._run_job_sync, job, loop)

    asyncio.run(run())
    assert job.status == "complete"


def test_viewing_a_cached_file_registers_it(client, file_id, catalog_runs, tmp_path, monkeypatch):
    """The fast path backfills caches made before jobs registered themselves."""
    cache = cache_dir_for(_library_wav(tmp_path))
    shutil.copytree(catalog_runs.gpl, cache / "output")
    (cache / "manifest.json").write_text("{}")
    monkeypatch.setattr(jobs_route, "is_gpl_cached", lambda wav: True)
    assert client.get("/api/catalog/wavs").json()["wavs"] == []
    assert client.post("/api/jobs/from-library", json={"fileId": file_id}).status_code == 200
    [seg] = _segmentations(client)
    assert seg["n_windows"] == 3

    # A catalog failure is logged; the file still opens.
    monkeypatch.setattr(jobs_route, "register_gpl_in_catalog", lambda wav: 1 / 0)
    assert client.post("/api/jobs/from-library", json={"fileId": file_id}).status_code == 200
