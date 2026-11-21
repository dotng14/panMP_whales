#!/usr/bin/env python3
"""Index Site H WAVs into the catalog from a known path manifest.

Does **not** ``rglob`` ``/mnt/sda/SiteH`` (exFAT listing there can hang for
minutes and return zero files). Uses the hybrid-run manifest that already
lists ~6800 absolute paths under SiteH.
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
DB = REPO / "backend" / "data" / "catalog.sqlite"
DEFAULT_MANIFEST = Path(
    "/home/yfreund/projects/panMP_whales/analysis/dict_HMM/find_songs/"
    "siteH_all_songs/manifest.csv"
)


def main() -> int:
    manifest = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_MANIFEST
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

    t0 = time.time()
    n_ok = 0
    with closing(catalog.connect(DB)) as conn:
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
