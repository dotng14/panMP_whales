#!/usr/bin/env python3
"""Hang-safe Site H catalog ingest (manifest + ``skip_stat``).

Does **not** ``rglob`` / ``stat`` ``/mnt/sda/SiteH`` (exFAT metadata I/O can
hang or raise I/O errors). Prefer this over ``python -m database index`` for
Site H. Paths come from a CSV on a local filesystem; upserts use
``skip_stat=True`` so the volume need not be readable at index time.

When ``/mnt/sda`` is down, skip this script and ship the picker from whatever
rows are already in the catalog (``start_time`` backfills from stem date/time).

Default manifest: hybrid-run CSV with ~6800 absolute ``wav_path`` values.
"""

from __future__ import annotations

import csv
import sys
import time
from contextlib import closing
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "backend")]

from database import catalog  # noqa: E402

SITEH = Path("/mnt/sda/SiteH")
DEFAULT_DB = REPO / "backend" / "data" / "catalog.sqlite"
DEFAULT_MANIFEST = Path(
    "/home/yfreund/projects/panMP_whales/analysis/dict_HMM/find_songs/"
    "siteH_all_songs/manifest.csv"
)


def main() -> int:
    # Usage: index_siteh.py [manifest.csv] [catalog.sqlite]
    manifest = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MANIFEST
    db = Path(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_DB
    if not manifest.is_file():
        print(f"manifest not found: {manifest}", flush=True)
        return 2

    siteh = str(SITEH)
    rows = list(csv.DictReader(manifest.open()))
    paths = []
    for row in rows:
        raw = (row.get("wav_path") or "").strip()
        if raw.endswith(".wav"):
            paths.append(Path(raw))
    # unique, stable order
    paths = sorted(set(paths), key=lambda p: str(p))
    print(f"manifest {manifest}: {len(paths)} unique wav paths", flush=True)
    print(f"catalog {db}", flush=True)

    t0 = time.time()
    n_ok = 0
    with closing(catalog.connect(db)) as conn:
        for i, wav in enumerate(paths, 1):
            catalog.upsert_wav_file(
                conn, library_root=siteh, abs_path=wav, skip_stat=True
            )
            n_ok += 1
            if i % 500 == 0 or i == len(paths):
                conn.commit()
                print(f"  {i}/{len(paths)} upserted ({time.time() - t0:.1f}s)", flush=True)
        conn.commit()
        total = conn.execute(
            "SELECT COUNT(*) FROM wav_files WHERE library_root = ?", (siteh,)
        ).fetchone()[0]
        socal = conn.execute(
            "SELECT COUNT(*) FROM wav_files WHERE library_root = ? AND family = 'SOCAL'",
            (siteh,),
        ).fetchone()[0]

    print(
        f"done in {time.time() - t0:.1f}s: upserted={n_ok} "
        f"SiteH_rows={total} SOCAL={socal}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
