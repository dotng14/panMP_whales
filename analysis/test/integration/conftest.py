"""A catalog built through the CLIs, shared by the catalog CLI and API tests."""

from __future__ import annotations

from contextlib import closing

import pytest

from analysis.test.catalog_setup import CATALOG_START_UNIX, CATALOG_STEM, CatalogRuns, build, pursue
from analysis.test.synthetic import write_gpl_folder, write_motif_csv
from database import catalog


@pytest.fixture(scope="session")
def catalog_runs(tmp_path_factory) -> CatalogRuns:
    """A randomized build and a matching-pursuit run registered with ``--catalog``, plus one CSV."""
    root = tmp_path_factory.mktemp("catalog")
    gpl = write_gpl_folder(root / "gpl", stem=CATALOG_STEM).root
    db = root / "catalog.sqlite"
    build_dir, mp_dir = root / "runs" / "b_build", root / "runs" / "a_mp"
    assert build(gpl, build_dir, "--catalog", str(db)) == 0
    assert pursue(gpl, build_dir, mp_dir, "--catalog", str(db)) == 0
    csv_path = write_motif_csv(root / "csv" / f"{CATALOG_STEM}.motif_segments.csv", start_unix=CATALOG_START_UNIX)
    with closing(catalog.connect(db)) as conn:
        catalog.upsert_annotation_csv(conn, csv_path)
        conn.commit()
    return CatalogRuns(root=root, db=db, gpl=gpl, build=build_dir, mp=mp_dir, csv=csv_path)
