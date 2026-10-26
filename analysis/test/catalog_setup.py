"""Helpers for the catalog tests: run the dict_learning CLIs, then register their output."""

from __future__ import annotations

import shutil
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
