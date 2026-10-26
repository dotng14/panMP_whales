"""database.catalog called directly: helpers, validation, registration of hand-made run folders."""

from __future__ import annotations

import json
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest

from analysis.test.synthetic import write_gpl_folder, write_motif_csv
from database import catalog, index_existing

STEM = "SOCAL_U_230426_193500"


def make_build(root: Path, gpl: Path, *, n_atoms: int = 2) -> Path:
    """A randomized-build folder written by hand: atoms, two placements, one outer pass."""
    out = root / "build"
    (out / "atoms").mkdir(parents=True)
    atoms = []
    for tid in range(n_atoms):
        rel = f"atoms/template_{tid:03d}.npy"
        np.save(out / rel, np.ones((12, 10)))
        atoms.append({"template_id": tid, "width": 10, "n_freq": 12, "n_components": 1,
                      "n_patches": 1, "source_segment_id": None, "path": rel})
    window = sorted(p.stem for p in (gpl / "spectrogram_npy").glob(f"spwhiten_{STEM}_w000001*e*[0-9].json"))[0]
    (out / "summary.json").write_text(json.dumps({
        "command": "randomized-build", "gpl_dir": str(gpl), "window_ids": [window], "atoms": atoms,
        "n_atoms": n_atoms, "n_matchings": 2, "stop_reason": "coverage", "n_outer_completed": 1,
    }))
    (out / "matchings.json").write_text(json.dumps([
        {"template_id": 0, "window_id": window, "t_start": 5, "amplitude": 1.0, "energy": 1.0},
        {"template_id": n_atoms - 1, "window_id": window, "t_start": 50, "amplitude": 0.5, "energy": 0.25,
         "coeffs": [0.5, 0.1]},
    ]))
    (out / "outer_logs.jsonl").write_text(json.dumps({"outer": 0, "n_atoms": n_atoms}) + "\n\n")
    return out


@pytest.fixture
def conn(tmp_path):
    with closing(catalog.connect(tmp_path / "c.sqlite")) as connection:
        yield connection


@pytest.fixture
def registered(conn, tmp_path):
    gpl = write_gpl_folder(tmp_path / "gpl", stem=STEM, n_windows=2).root
    out = make_build(tmp_path, gpl)
    result = catalog.register_run_dir(conn, out)
    return result, out


def test_window_id_parts():
    assert catalog.window_id_parts("spwhiten_a_b_w000012_s000000001_e000002400") == ("a_b", 12)
    assert catalog.window_id_parts("spwhiten_clip_w3") == ("clip", 3)
    assert catalog.window_id_parts("custom") is None


@pytest.mark.parametrize(
    "stem, expected",
    [
        ("SOCAL_H_76_230426_193500_df100", "2023-04-26T19:35:00+00:00"),
        ("x_20230426_193500", "2023-04-26T19:35:00+00:00"),
        ("x_2304269_193500", None),
        ("x_231399_193500", None),
        ("x_231399_193500_230426_193501", "2023-04-26T19:35:01+00:00"),
        ("clip", None),
    ],
)
def test_wav_start_datetime(stem, expected):
    got = catalog.wav_start_datetime(stem)
    assert (got.isoformat() if got else None) == expected


def test_window_abs_times():
    start = catalog.wav_start_datetime("x_230426_193500")
    assert catalog._window_abs_times(start, 1001, 2000, 1000.0) == (
        "2023-04-26T19:35:01+00:00", "2023-04-26T19:35:02+00:00")
    assert catalog._window_abs_times(None, 1, 2, 1000.0) == (None, None)
    assert catalog._window_abs_times(start, 1, 2, None) == (None, None)
    assert catalog._window_abs_times(start, None, 2, 1000.0) == (None, None)


def test_labels_and_names():
    assert catalog.csv_stem("/a/x.y.motif_segments.csv") == "x.y"
    assert catalog.csv_stem("other.csv") == "other"
    assert catalog._normalize_user_label("  ") is None
    assert catalog._normalize_user_label(None) is None
    assert catalog._normalize_user_label(" a ") == "a"
    assert catalog.is_named(" a ") and not catalog.is_named("  ") and not catalog.is_named(None)


def test_read_only_needs_an_existing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        catalog.connect(tmp_path / "missing.sqlite", read_only=True)


def test_register_and_query(conn, registered, tmp_path):
    result, out = registered
    dictionary_id = result["dictionary_id"]
    row = catalog.get_dictionary(conn, dictionary_id)
    assert (row["source"], row["wav_stem"], row["n_templates"], row["n_matchings"], row["n_windows"]) == (
        "seed", STEM, 2, 2, 1)
    matchings = catalog.list_matchings(conn, dictionary_id)
    assert [m["start_sec"] for m in matchings] == pytest.approx([0.05, 0.5])
    assert matchings[1]["coeffs_json"] == "[0.5, 0.1]"
    assert len(catalog.list_learning_round_logs(conn, result["run_id"])) == 1
    assert catalog.list_dictionaries_for_stem(conn, STEM)[0]["dictionary_id"] == dictionary_id
    assert catalog.list_dictionaries_for_stem(conn, "other") == []
    assert catalog.list_dictionary_windows(conn, dictionary_id)[0]["window_index"] == 1
    assert catalog.get_learning_run(conn, result["run_id"])["out_dir"] == str(out)
    assert catalog.get_tool_run(conn, result["tool_run_id"])["output_dictionary_id"] == dictionary_id
    assert len(catalog.list_tool_runs(conn, learning_run_id=result["run_id"])) == 1


def test_membership_and_window_sets(conn, registered):
    dictionary_id = registered[0]["dictionary_id"]
    members = catalog.list_dictionary_templates(conn, dictionary_id)
    catalog.set_dictionary_template_label(conn, dictionary_id, members[1]["template_id"], " up ")
    catalog.set_dictionary_templates(conn, dictionary_id, [members[1]["template_pk"]])
    kept = catalog.list_dictionary_templates(conn, dictionary_id)
    assert [(m["template_pk"], m["label"], m["position"]) for m in kept] == [(members[1]["template_pk"], "up", 0)]
    assert catalog.set_dictionary_windows(conn, dictionary_id, ["a", "a", "", "b"]) == 2
    with pytest.raises(KeyError):
        catalog.set_dictionary_template_label(conn, dictionary_id, 99, "x")


def test_labels_and_feedback_validation(conn, registered):
    dictionary_id = registered[0]["dictionary_id"]
    with pytest.raises(KeyError):
        catalog.set_dictionary_label(conn, 999, "x")
    with pytest.raises(KeyError):
        catalog.set_annotation_csv_label(conn, 999, "x")
    with pytest.raises(ValueError, match="feedback filter"):
        catalog.list_matchings(conn, dictionary_id, feedback="maybe")
    matching_id = catalog.list_matchings(conn, dictionary_id)[0]["id"]
    with pytest.raises(ValueError, match="label must be"):
        catalog.set_matching_feedback(conn, matching_id, "maybe")
    with pytest.raises(KeyError):
        catalog.set_matching_feedback(conn, 999, "good")
    assert catalog.set_matching_feedback(conn, matching_id, "good")["feedback"] == "good"
    assert catalog.delete_matching_feedback(conn, matching_id) is True
    assert catalog.get_dictionary_lineage(conn, 999) == []


def test_internal_invariants(conn):
    with pytest.raises(ValueError, match="source must be"):
        catalog._upsert_dictionary(conn, out_dir="x", source="other", parent_dictionary_id=None)
    with pytest.raises(ValueError, match="seed dictionaries"):
        catalog._upsert_dictionary(conn, out_dir="x", source="seed", parent_dictionary_id=1)
    with pytest.raises(ValueError, match="seed dictionaries"):
        catalog._upsert_dictionary(conn, out_dir="x", source="matching", parent_dictionary_id=None)
    with pytest.raises(ValueError, match="tool must be"):
        catalog._upsert_tool_run(conn, tool="other", out_dir="x")


def test_csvs(conn, tmp_path):
    path = write_motif_csv(tmp_path / f"{STEM}.motif_segments.csv", start_unix=0.0, n_songs=3)
    csv_id = catalog.upsert_annotation_csv(conn, path, label="v1")
    assert catalog.upsert_annotation_csv(conn, path) == csv_id  # keeps the id and label
    row = catalog.get_annotation_csv(conn, csv_id)
    assert (row["wav_stem"], row["n_rows"], row["n_songs"], row["label"]) == (STEM, 9, 3, "v1")
    assert catalog.list_analyses_for_stem(conn, STEM)["csvs"][0]["csv_id"] == csv_id
    with pytest.raises(catalog.CatalogError, match="CSV not found"):
        catalog.upsert_annotation_csv(conn, tmp_path / "missing.csv")
    path.unlink()
    assert catalog.prune_missing_annotation_csvs(conn) == 1
    assert catalog.list_annotation_csvs(conn) == []


def test_reindex_counts(tmp_path):
    gpl = write_gpl_folder(tmp_path / "gpl", stem=STEM, n_windows=1).root
    make_build(tmp_path / "runs", gpl)
    totals = index_existing.reindex(tmp_path / "c.sqlite", runs_roots=[tmp_path / "runs"])
    assert (totals["runs"], totals["skipped"]) == (1, [])
    assert index_existing.gpl_output_dir(tmp_path / "lib" / "a" / "x.wav", tmp_path / "lib", "result") == (
        tmp_path / "lib" / "result" / "a" / "x.gpl" / "output")
