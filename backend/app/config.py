import logging
import os
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).parent.parent
load_dotenv(BACKEND_DIR / ".env")

TMP_DIR = BACKEND_DIR / "tmp" / "jobs"
TMP_DIR.mkdir(parents=True, exist_ok=True)

_extra_origins = os.environ.get("CORS_ORIGINS", "")
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    *[o.strip() for o in _extra_origins.split(",") if o.strip()],
]
MAX_UPLOAD_MB = 500

WAV_LIBRARY_ROOTS: list[Path] = [
    Path(p.strip())
    for p in os.environ.get("WAV_LIBRARY_ROOTS", "").split(",")
    if p.strip()
]

# Subfolder under each library root where all processed caches are collected,
# mirroring the source tree: <root>/<GPL_RESULT_DIRNAME>/<rel>.gpl/
GPL_RESULT_DIRNAME = os.environ.get("GPL_RESULT_DIRNAME", "result").strip() or "result"

# Local fallback cache dir for read-only / external mounts (e.g. /mnt/sdb/SiteH)
# where writing <root>/result fails with PermissionError or risks exFAT hangs.
GPL_FALLBACK_CACHE_DIR = Path(
    os.environ.get("GPL_FALLBACK_CACHE_DIR", "").strip()
    or BACKEND_DIR / "data" / "gpl_cache"
)

logger = logging.getLogger(__name__)
for root in WAV_LIBRARY_ROOTS:
    # Skip exists() on external /mnt/sd* mounts — metadata I/O can hang uvicorn startup.
    if str(root).startswith("/mnt/sd"):
        continue
    if not root.exists():
        logger.warning("WAV_LIBRARY_ROOTS entry does not exist: %s", root)


def _paths(raw: str) -> list[Path]:
    return [Path(p.strip()) for p in raw.split(",") if p.strip()]


# SQLite catalog of dictionaries, MP runs and song-structure CSVs (database/).
# This repo's own file; never point it at gplrev3's catalog.
CATALOG_DB = Path(os.environ.get("CATALOG_DB", "").strip() or BACKEND_DIR / "data" / "catalog.sqlite")
# Folders scanned by POST /api/catalog/reindex for randomized-build / matching-pursuit output.
CATALOG_RUNS_ROOTS: list[Path] = _paths(os.environ.get("CATALOG_RUNS_ROOTS", ""))
# Folders scanned for *.motif_segments.csv.
CATALOG_CSV_ROOTS: list[Path] = _paths(
    os.environ.get("CATALOG_CSV_ROOTS", "") or str(BACKEND_DIR / "data" / "song_structure")
)
# 20-year minute-energy scan results (monthly_hourly.md, hourly/*.png, overview plots).
SCAN_RESULTS_DIR = Path(
    os.environ.get("SCAN_RESULTS_DIR", "").strip()
    or "/home/yfreund/whale_processing/12TB_results"
)
