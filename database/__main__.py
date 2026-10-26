"""Command line for the catalog::

    python -m database index --db catalog.sqlite --wav-root /data/recordings \\
        --runs-root runs --csv-root backend/data/song_structure
    python -m database register --db catalog.sqlite runs/build_a runs/build_a_mp x.motif_segments.csv

``index`` scans roots (see ``index_existing``). ``register`` adds the given
``randomized-build`` / ``matching-pursuit`` folders and song-structure CSVs.
Exit codes: 0 success, 1 something was skipped, 2 bad arguments.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import closing
from pathlib import Path

from . import catalog, index_existing


def _env_paths(name: str) -> list[Path]:
    return [Path(p.strip()) for p in os.environ.get(name, "").split(",") if p.strip()]


def run_index(args: argparse.Namespace) -> int:
    totals = index_existing.reindex(
        args.db,
        wav_roots=args.wav_root or _env_paths("WAV_LIBRARY_ROOTS"),
        runs_roots=args.runs_root or _env_paths("CATALOG_RUNS_ROOTS"),
        csv_roots=args.csv_root or _env_paths("CATALOG_CSV_ROOTS"),
        result_dirname=args.result_dirname,
    )
    skipped = totals.pop("skipped")
    print(json.dumps(totals))
    for item in skipped:
        print(f"skipped {item}", file=sys.stderr)
    return 1 if skipped else 0


def run_register(args: argparse.Namespace) -> int:
    failed = False
    with closing(catalog.connect(args.db)) as conn:
        for path in args.paths:
            try:
                with conn:
                    if path.is_file():
                        result = {"kind": "csv", "csv_id": catalog.upsert_annotation_csv(conn, path, label=args.label)}
                    else:
                        result = catalog.register_run_dir(conn, path, label=args.label)
            except catalog.CatalogError as exc:
                print(f"error: {exc}", file=sys.stderr)
                failed = True
                continue
            print(json.dumps({"path": str(path), **result}))
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m database", description="SQLite catalog of runs and CSVs.")
    sub = parser.add_subparsers(dest="command", required=True)

    index = sub.add_parser("index", help="Scan WAV, run and CSV roots into the catalog")
    index.add_argument("--db", type=Path, required=True, help="Catalog SQLite file (created if missing)")
    index.add_argument("--wav-root", type=Path, action="append", default=[],
                       help="WAV library root (repeatable; default: $WAV_LIBRARY_ROOTS)")
    index.add_argument("--runs-root", type=Path, action="append", default=[],
                       help="Folder holding run output folders (repeatable; default: $CATALOG_RUNS_ROOTS)")
    index.add_argument("--csv-root", type=Path, action="append", default=[],
                       help="Folder holding *.motif_segments.csv (repeatable; default: $CATALOG_CSV_ROOTS)")
    index.add_argument("--result-dirname", default=os.environ.get("GPL_RESULT_DIRNAME", "result").strip() or "result",
                       help="GPL cache folder under each WAV root")
    index.set_defaults(func=run_index)

    register = sub.add_parser("register", help="Add run output folders and song-structure CSVs")
    register.add_argument("--db", type=Path, required=True, help="Catalog SQLite file (created if missing)")
    register.add_argument("--label", default=None, help="User label for every registered item")
    register.add_argument("paths", type=Path, nargs="+", help="Run folders or *.motif_segments.csv files")
    register.set_defaults(func=run_register)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
