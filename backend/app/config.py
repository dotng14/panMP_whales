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

logger = logging.getLogger(__name__)
for root in WAV_LIBRARY_ROOTS:
    if not root.exists():
        logger.warning("WAV_LIBRARY_ROOTS entry does not exist: %s", root)
