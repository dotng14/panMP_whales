"""Helpers for the catalog tests: run the dict_learning CLIs, then register their output."""

from __future__ import annotations

import shutil
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from analysis.dict_learning.code.cli import main as dict_learning_main

# A WAV stem with a recording start (2023-04-26 19:35:00 UTC), so CSV times resolve.
CATALOG_STEM = "SOCAL_T_230426_193500"
CATALOG_START_UNIX = 1682537700.0
BUILD_ARGS = ["--m-per-outer", "3", "--n-min", "2", "--refine-iters", "1", "--seed", "0",
              "--workers", "1", "--max-outer", "2", "--min-coverage", "1.0"]


@dataclass
class CatalogRuns:
    root: Path
    db: Path
    gpl: Path
    build: Path
    mp: Path
    csv: Path

    def copy_db(self, dest: Path) -> Path:
        """A private copy for tests that write to the catalog."""
        return Path(shutil.copy(self.db, dest / "catalog.sqlite"))


def build(gpl: Path, out: Path, *extra: str) -> int:
    return dict_learning_main(["randomized-build", "--gpl-dir", str(gpl), "--out", str(out), *BUILD_ARGS, *extra])


def pursue(gpl: Path, dictionary: Path, out: Path, *extra: str) -> int:
    return dict_learning_main(["matching-pursuit", "--gpl-dir", str(gpl), "--dictionary", str(dictionary),
                               "--out", str(out), "--workers", "1", *extra])


def downgrade_tool_runs(db: Path) -> None:
    """Give ``tool_runs`` the CHECK it had before ``pan_build`` and ``split`` existed, keeping its rows."""
    with closing(sqlite3.connect(db)) as raw:
        ddl = raw.execute("SELECT sql FROM sqlite_master WHERE name = 'tool_runs'").fetchone()[0]
        columns = ddl.split("(", 1)[1].replace("'pan_build', ", "").replace(", 'split'", "")
        raw.executescript(
            f"""
            PRAGMA foreign_keys = OFF;
            BEGIN;
            CREATE TABLE old_tool_runs ({columns};
            INSERT INTO old_tool_runs SELECT * FROM tool_runs;
            DROP TABLE tool_runs;
            ALTER TABLE old_tool_runs RENAME TO tool_runs;
            COMMIT;
            """
        )
