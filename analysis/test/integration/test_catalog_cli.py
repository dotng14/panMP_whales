"""Catalog registration through the CLIs: ``--catalog`` on the dict_learning commands and ``python -m database``."""

from __future__ import annotations

import json
import os
import shutil
from contextlib import closing
from pathlib import Path

import pytest

from analysis.test.catalog_setup import CATALOG_START_UNIX, CATALOG_STEM, build, pursue
from analysis.test.synthetic import read_json, run_module, write_gpl_folder, write_motif_csv, write_wav
from database import catalog
from database.__main__ import main as db_main


def query(db: Path, fn, *args, **kwargs):
    with closing(catalog.connect(db, read_only=True)) as conn:
        return fn(conn, *args, **kwargs)


def edit_json(path: Path, mutate) -> None:
    """Load a JSON file, let ``mutate`` change it in place, write it back."""
    data = read_json(path)
    mutate(data)
    path.write_text(json.dumps(data))


def copy_run(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


def test_cli_registers_build_and_mp(catalog_runs):
    r = catalog_runs
    found = query(r.db, catalog.list_analyses_for_stem, CATALOG_STEM)
    mp, built = found["dictionaries"]
    assert (built["source"], mp["source"]) == ("seed", "matching")
    assert mp["parent_dictionary_id"] == built["dictionary_id"]
    summary = read_json(r.build / "summary.json")
    assert built["n_templates"] == mp["n_templates"] == summary["n_atoms"]
    assert built["n_matchings"] == summary["n_matchings"]
    assert built["n_windows"] == mp["n_windows"] == 3
    assert built["wav_stem"] == CATALOG_STEM
    assert [c["name"] for c in found["csvs"]] == [r.csv.name]
    assert found["csvs"][0]["n_rows"] == 6 and found["csvs"][0]["n_songs"] == 2

    # Placement times: (window start sample - 1) / fs + t_start * hop / fs.
    matchings = query(r.db, catalog.list_matchings, built["dictionary_id"])
    second = next(m for m in matchings if m["window_id"].endswith("_w000002_s000002401_e000004800"))
    assert second["start_sec"] == pytest.approx(2.4 + second["t_start"] * 0.01)
    assert all(m["end_sec"] > m["start_sec"] for m in matchings)

    run = query(r.db, catalog.list_learning_runs)[0]
    assert len(query(r.db, catalog.list_learning_round_logs, run["run_id"])) == summary["n_outer_completed"]
    assert [t["tool"] for t in query(r.db, catalog.list_tool_runs)] == ["randomized_build", "matching_pursuit"]
    windows = query(r.db, catalog.list_windows_for_segmentation, built["segmentation_id"])
    assert [w["window_index"] for w in windows] == [1, 2, 3]
    assert windows[0]["abs_start_time"] is None  # the WAV is not indexed, so its start is unknown
    lineage = query(r.db, catalog.get_dictionary_lineage, mp["dictionary_id"])
    assert [d["dictionary_id"] for d in lineage] == [built["dictionary_id"], mp["dictionary_id"]]


def test_mp_registers_its_dictionary_first(catalog_runs, tmp_path):
    r = catalog_runs
    db = tmp_path / "fresh.sqlite"
    assert pursue(r.gpl, r.build, tmp_path / "mp", "--catalog", str(db)) == 0
    # A gate that rejects every placement: the dictionary is registered with none.
    assert pursue(r.gpl, r.build, tmp_path / "none", "--max-relative-distance", "0", "--catalog", str(db)) == 0
    dictionaries = query(db, catalog.list_dictionaries)
    assert [d["source"] for d in dictionaries] == ["matching", "matching", "seed"]
    assert dictionaries[0]["n_matchings"] == 0


def test_window_without_sample_rate_gets_no_time(catalog_runs, gpl_variant, tmp_path):
    gpl = gpl_variant(sidecar_sample_freq=False, stem="nofreq")
    db = tmp_path / "c.sqlite"
    assert pursue(gpl, catalog_runs.build, tmp_path / "mp", "--catalog", str(db)) == 0
    mp = query(db, catalog.list_dictionaries)[0]
    assert all(m["start_sec"] is None for m in query(db, catalog.list_matchings, mp["dictionary_id"]))


def test_reregistering_keeps_ids_and_labels(catalog_runs, tmp_path, capsys):
    r = catalog_runs
    db = r.copy_db(tmp_path)
    before = query(db, catalog.list_dictionaries)
    with closing(catalog.connect(db)) as conn:
        catalog.set_dictionary_label(conn, before[1]["dictionary_id"], "first build")
        conn.commit()
    assert db_main(["register", "--db", str(db), str(r.build)]) == 0
    assert db_main(["register", "--db", str(db), "--label", "best mp", str(r.mp)]) == 0
    assert db_main(["register", "--db", str(db), "--label", "HMM v1", str(r.csv)]) == 0
    other_csv = Path(shutil.copy(r.csv, tmp_path / "other.csv"))
    assert db_main(["register", "--db", str(db), str(other_csv)]) == 0
    lines = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [line["kind"] for line in lines] == ["randomized-build", "matching-pursuit", "csv", "csv"]
    after = query(db, catalog.list_dictionaries)
    assert [d["dictionary_id"] for d in after] == [d["dictionary_id"] for d in before]
    assert [d["label"] for d in after] == ["best mp", "first build"]
    other, first = query(db, catalog.list_annotation_csvs)
    assert first["label"] == "HMM v1"
    assert (other["name"], other["wav_stem"]) == ("other.csv", "other")


def test_registers_older_and_edited_folders(catalog_runs, tmp_path, monkeypatch):
    r = catalog_runs
    # An older build folder: no window list, no outer log, gpl_dir relative to the folder.
    old = copy_run(r.build, tmp_path / "runs" / "old")
    (old / "outer_logs.jsonl").unlink()
    edit_json(old / "summary.json", lambda s: (s.pop("window_ids"), s.update(gpl_dir=os.path.relpath(r.gpl, old))))
    db = tmp_path / "c.sqlite"
    assert db_main(["register", "--db", str(db), "--label", "old build", str(old)]) == 0
    built = query(db, catalog.list_dictionaries)[0]
    assert built["label"] == "old build"
    assert built["n_windows"] == len({m["window_id"] for m in read_json(old / "matchings.json")})
    assert built["wav_stem"] == CATALOG_STEM
    assert query(db, catalog.list_learning_round_logs, built["learning_run_id"]) == []
    assert all(m["start_sec"] is not None for m in query(db, catalog.list_matchings, built["dictionary_id"]))

    # gpl_dir relative to the working directory.
    cwd_rel = copy_run(r.build, tmp_path / "runs" / "cwd")
    monkeypatch.chdir(r.root)
    edit_json(cwd_rel / "summary.json", lambda s: s.update(gpl_dir=os.path.relpath(r.gpl, r.root)))
    assert db_main(["register", "--db", str(db), str(cwd_rel)]) == 0
    assert query(db, catalog.list_dictionaries)[0]["segmentation_id"] == built["segmentation_id"]

    # GPL folder gone and window ids of another kind: no times, no WAV stem.
    moved = copy_run(r.build, tmp_path / "runs" / "moved")
    edit_json(moved / "summary.json", lambda s: s.update(gpl_dir=str(tmp_path / "gone"), window_ids=["custom"]))
    fresh = tmp_path / "fresh.sqlite"
    assert db_main(["register", "--db", str(fresh), str(moved)]) == 0
    lost = query(fresh, catalog.list_dictionaries)[0]
    assert lost["wav_stem"] is None and lost["segmentation_id"] is None
    assert all(m["start_sec"] is None for m in query(fresh, catalog.list_matchings, lost["dictionary_id"]))

    # Re-registering after an atom was dropped removes it.
    def drop_atom(s):
        s["atoms"] = s["atoms"][1:]
    first = read_json(old / "summary.json")["atoms"][0]["template_id"]
    edit_json(old / "summary.json", drop_atom)
    edit_json(old / "matchings.json", lambda rows: rows.__setitem__(
        slice(None), [m for m in rows if m["template_id"] != first]))
    assert db_main(["register", "--db", str(db), str(old)]) == 0
    templates = query(db, catalog.list_templates, built["learning_run_id"])
    assert first not in [t["template_id"] for t in templates]


def test_non_gpl_window_names(catalog_runs, tmp_path):
    """A flat folder whose windows are not named spwhiten_<stem>_w…: indexed, but no stem."""
    flat = tmp_path / "flat"
    flat.mkdir()
    for i, path in enumerate(sorted((catalog_runs.gpl / "spectrogram_npy").glob("spwhiten_*_e*[0-9].npy"))):
        shutil.copy(path, flat / f"spwhiten_x{i}.npy")
        shutil.copy(path.with_suffix(".json"), flat / f"spwhiten_x{i}.json")
    db = tmp_path / "c.sqlite"
    assert pursue(flat, catalog_runs.build, tmp_path / "mp", "--catalog", str(db)) == 0
    mp = query(db, catalog.list_dictionaries)[0]
    assert mp["wav_stem"] is None
    windows = query(db, catalog.list_windows_for_segmentation, mp["segmentation_id"])
    assert len(windows) == 3 and all(w["window_index"] is None for w in windows)


@pytest.mark.parametrize(
    "case, message",
    [
        ("no folder", "run folder not found"),
        ("no summary", "missing summary.json"),
        ("bad json", "is not valid JSON"),
        ("other command", "is not from randomized-build or matching-pursuit"),
        ("no atom list", "has no atom list"),
        ("atom file missing", "atom file missing"),
        ("unknown atom", "which the dictionary lacks"),
        ("mp without dictionary", "does not name its dictionary folder"),
        ("mp dictionary unknown", "is not in the catalog and has no summary.json"),
        ("csv columns", "is missing columns: end_unix"),
    ],
)
def test_register_rejects_bad_input(catalog_runs, tmp_path, capsys, case, message):
    r = catalog_runs
    target = tmp_path / "item"
    if case == "no summary":
        target.mkdir()
    elif case == "bad json":
        target.mkdir()
        (target / "summary.json").write_text("{")
    elif case == "other command":
        target.mkdir()
        (target / "summary.json").write_text(json.dumps({"command": "pan"}))
    elif case == "no atom list":
        target.mkdir()
        (target / "summary.json").write_text(json.dumps({"command": "randomized-build"}))
    elif case == "atom file missing":
        copy_run(r.build, target)
        next((target / "atoms").glob("*.npy")).unlink()
    elif case == "unknown atom":
        copy_run(r.build, target)
        edit_json(target / "matchings.json", lambda rows: rows[0].update(template_id=999))
    elif case == "mp without dictionary":
        copy_run(r.mp, target)
        edit_json(target / "summary.json", lambda s: s.pop("dictionary"))
    elif case == "mp dictionary unknown":
        copy_run(r.mp, target)
        (tmp_path / "empty").mkdir()
        edit_json(target / "summary.json", lambda s: s.update(dictionary=str(tmp_path / "empty")))
    elif case == "csv columns":
        target = tmp_path / "x.motif_segments.csv"
        target.write_text("motif,song,theme,phrase_id,phrase_count,start_unix\n")
    assert db_main(["register", "--db", str(tmp_path / "c.sqlite"), str(target)]) == 1
    assert message in capsys.readouterr().err


def test_catalog_flag_reports_registration_errors(catalog_runs, tmp_path, capsys):
    """A hand-made dictionary folder works for MP but cannot be registered: exit 2."""
    handmade = copy_run(catalog_runs.build, tmp_path / "handmade")
    edit_json(handmade / "summary.json", lambda s: s.pop("command"))
    code = pursue(catalog_runs.gpl, handmade, tmp_path / "mp", "--catalog", str(tmp_path / "c.sqlite"))
    assert code == 2
    assert "could not register" in capsys.readouterr().err


def test_index_scans_roots(tmp_path, monkeypatch, capsys):
    stem = "SOCAL_I_230426_193500"
    library = tmp_path / "library"
    (library / "site").mkdir(parents=True)
    write_wav(library / "site" / f"{stem}.wav")
    write_wav(library / "site" / "nocache.wav")
    (library / "result").mkdir()
    write_wav(library / "result" / "stray.wav")  # inside the cache folder: not a library WAV
    cache = library / "result" / "site" / f"{stem}.gpl" / "output"
    write_gpl_folder(cache, stem=stem)
    (cache / "panel.bin").write_bytes(b"panel")
    # Recording starts as YYYYMMDD_HHMMSS, and two stems whose date does not parse.
    for other in ("A_20230426_193500", "B_2304269_193500", "C_231399_193500"):
        write_wav(library / "site" / f"{other}.wav")
        write_gpl_folder(library / "result" / "site" / f"{other}.gpl" / "output", stem=other, n_windows=1)

    runs = tmp_path / "runs"
    assert build(cache, runs / "b_build") == 0
    assert pursue(cache, runs / "b_build", runs / "a_mp") == 0  # sorts first; indexed after the build
    for name, text in {"junk_json": "{", "junk_list": "[]", "other": json.dumps({"command": "pan"}),
                       "broken": json.dumps({"command": "randomized-build"})}.items():
        (runs / name).mkdir()
        (runs / name / "summary.json").write_text(text)
    csvs = tmp_path / "csvs"
    good = write_motif_csv(csvs / f"{stem}.motif_segments.csv", start_unix=CATALOG_START_UNIX)
    (csvs / "bad.motif_segments.csv").write_text("motif\n")
    (csvs / "notes.csv").write_text("ignored\n")

    db = tmp_path / "c.sqlite"
    missing = str(tmp_path / "missing")
    capsys.readouterr()  # drop the CLI output above
    code = db_main(["index", "--db", str(db), "--wav-root", str(library), "--runs-root", str(runs),
                    "--csv-root", str(csvs), "--wav-root", missing, "--runs-root", missing, "--csv-root", missing])
    out, err = capsys.readouterr()
    assert code == 1
    assert json.loads(out) == {"wavs": 5, "gpl_segmentations": 4, "windows": 6, "runs": 2, "csvs": 1,
                               "csvs_pruned": 0}
    assert "broken: " in err and "bad.motif_segments.csv: " in err

    wavs = query(db, catalog.list_wav_files)
    indexed = next(w for w in wavs if w["stem"] == stem)
    segmentation = query(db, catalog.list_gpl_segmentations_for_wav, indexed["wav_id"])[0]
    assert segmentation["n_windows"] == 3 and segmentation["panel_path"].endswith("panel.bin")
    windows = query(db, catalog.list_windows_for_segmentation, segmentation["segmentation_id"])
    assert windows[0]["abs_start_time"] == "2023-04-26T19:35:00+00:00"
    assert windows[1]["abs_start_time"] == "2023-04-26T19:35:02.400000+00:00"
    starts = {}
    for w in wavs:
        for seg in query(db, catalog.list_gpl_segmentations_for_wav, w["wav_id"]):
            starts[w["stem"]] = query(db, catalog.list_windows_for_segmentation, seg["segmentation_id"])[0]["abs_start_time"]
    assert starts["A_20230426_193500"] == "2023-04-26T19:35:00+00:00"
    assert starts["B_2304269_193500"] is None and starts["C_231399_193500"] is None
    found = query(db, catalog.list_analyses_for_stem, stem)
    assert [d["source"] for d in found["dictionaries"]] == ["matching", "seed"]
    assert [c["csv_path"] for c in found["csvs"]] == [str(good)]

    # A run on a GPL folder outside the cache is linked to the indexed WAV by stem.
    other = write_gpl_folder(tmp_path / "other_gpl", stem="nocache").root
    assert build(other, tmp_path / "other_run", "--catalog", str(db)) == 0
    nocache = next(w for w in wavs if w["stem"] == "nocache")
    assert len(query(db, catalog.list_gpl_segmentations_for_wav, nocache["wav_id"])) == 1

    # Environment fallbacks; a deleted CSV is pruned.
    shutil.rmtree(runs / "broken")
    (csvs / "bad.motif_segments.csv").unlink()
    good.unlink()
    monkeypatch.setenv("WAV_LIBRARY_ROOTS", str(library))
    monkeypatch.setenv("CATALOG_RUNS_ROOTS", str(runs))
    monkeypatch.setenv("CATALOG_CSV_ROOTS", str(csvs))
    capsys.readouterr()
    assert db_main(["index", "--db", str(db)]) == 0
    totals = json.loads(capsys.readouterr().out)
    assert totals["runs"] == 2 and totals["csvs"] == 0 and totals["csvs_pruned"] == 1


def test_python_dash_m_database(catalog_runs, tmp_path):
    db = tmp_path / "c.sqlite"
    proc = run_module("database", "register", "--db", str(db), str(catalog_runs.build))
    assert proc.returncode == 0, proc.stderr
    assert json.loads(proc.stdout)["kind"] == "randomized-build"
