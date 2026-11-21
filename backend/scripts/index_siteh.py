#!/usr/bin/env python3
"""Index /mnt/sda/SiteH one deployment folder at a time into the catalog.

Avoids ``SiteH.iterdir()`` (can hang on this exFAT volume); uses the known
deployment folder names from siteH_deployments.md instead.
"""

from __future__ import annotations

import sys
import time
from contextlib import closing
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO), str(REPO / "backend")]

from database import catalog  # noqa: E402
from database.index_existing import index_wav_root  # noqa: E402

SITEH = Path("/mnt/sda/SiteH")
DB = REPO / "backend" / "data" / "catalog.sqlite"

# Prefer recent deployments first so the picker fills with useful files sooner.
DEPLOYMENTS = [
    "SOCAL_H_81", "SOCAL_H_80", "SOCAL_H_79", "SOCAL_H_78", "SOCAL_H_77", "SOCAL_H_76",
    "SOCAL_H_75_HE", "SOCAL_H_74_HE", "SOCAL_H_73_HE", "SOCAL_H_72_HE", "SOCAL_H_71",
    "SOCAL_H_70", "SOCAL_H_69", "SOCAL_H_68", "SOCAL_H_67_HE", "SOCAL_H_66", "SOCAL_H_65",
    "SOCAL_H_64", "SOCAL_H_63", "SOCAL_H_62", "SOCAL_H_61", "SOCAL_H_59", "SOCAL_H_58",
    "SOCAL57H", "SOCAL56H", "SOCAL55H", "SOCAL54H", "SOCAL53H", "SOCAL52H", "SOCAL51H",
    "SOCAL50H", "SOCAL49H", "SOCAL48H", "SOCAL47H", "SOCAL46H", "SOCAL45H", "SOCAL44H",
    "SOCAL41H", "SOCAL40H", "SOCAL38H", "SOCAL37H", "SOCAL36H", "SOCAL35H", "SOCAL34H",
    "SOCAL32H", "SOCAL31H", "SOCAL30H", "SOCAL29H", "SOCAL27H", "SOCAL26H", "SOCAL18H",
]


def main() -> int:
    siteh = str(SITEH.resolve())
    subs = [SITEH / name for name in DEPLOYMENTS if (SITEH / name).is_dir()]
    print(f"indexing {len(subs)} reachable deployments under {SITEH}", flush=True)

    for i, sub in enumerate(subs, 1):
        t1 = time.time()
        print(f"[{i}/{len(subs)}] indexing {sub.name} …", flush=True)
        try:
            with closing(catalog.connect(DB)) as conn:
                counts = index_wav_root(conn, sub, result_dirname="result")
                sub_s = str(sub.resolve())
                rows = conn.execute(
                    "SELECT wav_id, abs_path FROM wav_files WHERE library_root = ?",
                    (sub_s,),
                ).fetchall()
                for row in rows:
                    abs_path = Path(row["abs_path"])
                    rel = str(abs_path.relative_to(SITEH.resolve()))
                    conn.execute(
                        "UPDATE wav_files SET library_root = ?, rel_path = ? WHERE wav_id = ?",
                        (siteh, rel, row["wav_id"]),
                    )
                conn.commit()
                n = conn.execute(
                    "SELECT COUNT(*) FROM wav_files WHERE library_root = ?", (siteh,)
                ).fetchone()[0]
            print(
                f"[{i}/{len(subs)}] {sub.name} ok in {time.time() - t1:.1f}s "
                f"counts={counts} SiteH_total={n}",
                flush=True,
            )
        except Exception as exc:
            print(f"[{i}/{len(subs)}] FAILED {sub.name}: {exc}", flush=True)
    print("done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
