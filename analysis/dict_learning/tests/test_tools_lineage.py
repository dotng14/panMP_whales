"""Tests for elementary dict tools and catalog lineage persistence."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from analysis.dict_learning.code.core.objects import Dictionary, Matching, Template, Window
from analysis.dict_learning.code.tools import (
    average_patches,
    average_templates,
    matching_pursuit,
    merge_templates,
    remove_template,
)
from database import catalog


def _toy_window(window_id: str = "w0", *, n_freq: int = 4, n_time: int = 40) -> Window:
    rng = np.random.default_rng(0)
    data = rng.normal(size=(n_freq, n_time)).astype(np.float64)
    # Plant a clear atom so MP finds it.
    atom = np.zeros((n_freq, 5), dtype=np.float64)
    atom[1:3, :] = 3.0
    data[:, 10:15] += atom
    data[:, 25:30] += atom
    return Window(
        window_id=window_id,
        data=data,
        start_sample=1,
        end_sample=n_time,
        overlap=1,
    )


def test_average_patches_and_templates_pure():
    window = _toy_window()
    tmpl = Template(template_id=0, data=window.data[:, 10:15].copy())
    dictionary = Dictionary()
    dictionary.add(tmpl)
    matchings = [
        Matching(template_id=0, window_id="w0", t_start=10, amplitude=1.0),
        Matching(template_id=0, window_id="w0", t_start=25, amplitude=1.0),
    ]
    avg = average_patches({"w0": window.data}, matchings, width=5)
    assert avg.shape == (4, 5)
    result = average_templates({"w0": window.data}, dictionary, matchings)
    assert len(result.dictionary) == 1
    assert result.dictionary.templates[0].data.shape == (4, 5)


def test_matching_pursuit_and_catalog_lineage(tmp_path: Path):
    window = _toy_window()
    tmpl = Template(template_id=0, data=window.data[:, 10:15].copy())
    dictionary = Dictionary()
    dictionary.add(tmpl)

    db = tmp_path / "cat.sqlite"
    out = tmp_path / "mp_out"
    out.mkdir()
    np.save(out / "template_000.npy", tmpl.data)

    with catalog.connect(db) as conn:
        run_id = catalog.start_learning_run(
            conn, label="t", out_dir=out, params={}, n_windows=1
        )
        pks = catalog.append_templates(
            conn,
            [
                {
                    "template_id": 0,
                    "npy_path": out / "template_000.npy",
                    "n_freq": 4,
                    "width": 5,
                }
            ],
            run_id=run_id,
        )
        input_id = catalog.create_dictionary(
            conn, name="seed0", source="seed", template_pks=pks
        )
        conn.commit()

        mp = matching_pursuit(
            {window.window_id: window.data},
            dictionary,
            eps_step_fraction=0.01,
            conn=conn,
            input_dictionary_id=input_id,
            learning_run_id=run_id,
            out_dir=out,
            name="mp1",
        )
        conn.commit()

        assert mp.tool_run_id is not None
        assert mp.output_dictionary_id is not None
        out_dict = catalog.get_dictionary(conn, mp.output_dictionary_id)
        assert out_dict["source"] == "matching"
        assert out_dict["parent_dictionary_id"] == input_id
        matchings = catalog.list_matchings(conn, mp.output_dictionary_id)
        assert len(matchings) >= 1
        tool = catalog.get_tool_run(conn, mp.tool_run_id)
        assert tool["tool"] == "matching_pursuit"
        # Same template pks reused.
        assert catalog._dictionary_template_pks(conn, mp.output_dictionary_id) == pks


def test_average_persist_and_remove_merge(tmp_path: Path):
    window = _toy_window()
    t0 = Template(template_id=0, data=window.data[:, 10:15].copy())
    t1 = Template(template_id=1, data=window.data[:, 25:30].copy())
    dictionary = Dictionary()
    dictionary.add(t0)
    dictionary.add(t1)
    matchings = [
        Matching(template_id=0, window_id="w0", t_start=10, amplitude=1.0),
        Matching(template_id=0, window_id="w0", t_start=25, amplitude=1.0),
        Matching(template_id=1, window_id="w0", t_start=10, amplitude=1.0),
        Matching(template_id=1, window_id="w0", t_start=25, amplitude=1.0),
    ]

    db = tmp_path / "cat.sqlite"
    out = tmp_path / "avg"
    out.mkdir()
    np.save(out / "t0.npy", t0.data)
    np.save(out / "t1.npy", t1.data)

    with catalog.connect(db) as conn:
        run_id = catalog.start_learning_run(
            conn, label="t", out_dir=out, params={}, n_windows=1
        )
        pks = catalog.append_templates(
            conn,
            [
                {"template_id": 0, "npy_path": out / "t0.npy", "n_freq": 4, "width": 5},
                {"template_id": 1, "npy_path": out / "t1.npy", "n_freq": 4, "width": 5},
            ],
            run_id=run_id,
        )
        input_id = catalog.create_dictionary(
            conn, name="seed", source="seed", template_pks=pks
        )
        conn.commit()

        avg = average_templates(
            {window.window_id: window.data},
            dictionary,
            matchings,
            conn=conn,
            input_dictionary_id=input_id,
            out_dir=out / "avg_out",
            learning_run_id=run_id,
            name="avg1",
        )
        conn.commit()
        assert avg.output_dictionary_id is not None
        assert len(avg.new_template_pks) == 2
        parents = catalog.list_template_parents(conn, avg.new_template_pks[0])
        assert len(parents) == 1
        assert parents[0]["parent_template_pk"] == pks[0]
        avg_dict = catalog.get_dictionary(conn, avg.output_dictionary_id)
        assert avg_dict["source"] == "refinement"

        # Meta-dictionary still has original atoms after remove.
        n_before = conn.execute("SELECT COUNT(*) AS c FROM templates").fetchone()["c"]
        rm = remove_template(
            avg.dictionary,
            template_id=0,
            conn=conn,
            input_dictionary_id=avg.output_dictionary_id,
            learning_run_id=run_id,
        )
        conn.commit()
        n_after = conn.execute("SELECT COUNT(*) AS c FROM templates").fetchone()["c"]
        assert n_after == n_before
        assert len(rm.dictionary) == 1
        rm_dict = catalog.get_dictionary(conn, rm.output_dictionary_id)
        assert rm_dict["source"] == "edit"
        assert len(catalog._dictionary_template_pks(conn, rm.output_dictionary_id)) == 1

        # Merge the two original seed atoms (use seed dict + matchings).
        merge_out = out / "merge"
        merge_out.mkdir()
        merged = merge_templates(
            {window.window_id: window.data},
            dictionary,
            matchings,
            0,
            1,
            conn=conn,
            input_dictionary_id=input_id,
            out_dir=merge_out,
            learning_run_id=run_id,
        )
        conn.commit()
        assert merged.merged_template_pk is not None
        mparents = catalog.list_template_parents(conn, merged.merged_template_pk)
        assert {p["parent_template_pk"] for p in mparents} == set(pks)
        mdict = catalog.get_dictionary(conn, merged.output_dictionary_id)
        assert mdict["source"] == "edit"
        assert len(catalog._dictionary_template_pks(conn, merged.output_dictionary_id)) == 1


def test_migration_creates_tool_tables(tmp_path: Path):
    db = tmp_path / "fresh.sqlite"
    with catalog.connect(db) as conn:
        names = {
            r[0]
            for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert "tool_runs" in names
        assert "template_parents" in names
        # Expanded source CHECK accepts matching/edit.
        run_id = catalog.start_learning_run(
            conn, label="x", out_dir=tmp_path / "o", params={}
        )
        (tmp_path / "o").mkdir(exist_ok=True)
        npy = tmp_path / "o" / "t.npy"
        np.save(npy, np.zeros((2, 2)))
        pks = catalog.append_templates(
            conn,
            [{"template_id": 0, "npy_path": npy, "n_freq": 2, "width": 2}],
            run_id=run_id,
        )
        seed_id = catalog.create_dictionary(
            conn, name="s", source="seed", template_pks=pks
        )
        mid = catalog._insert_dictionary(
            conn,
            name="m",
            source="matching",
            parent_dictionary_id=seed_id,
        )
        assert mid > seed_id
