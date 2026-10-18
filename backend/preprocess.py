#!/usr/bin/env python3
"""Thin shim: offline preprocess lives in analysis.GPL_Python.preprocess.

Passes ``WAV_LIBRARY_ROOTS`` / ``CATALOG_DB`` from the backend env into the
analysis CLI. Prefer::

    python -m analysis.GPL_Python.preprocess …
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).parent.resolve()
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

# Load backend .env so WAV_LIBRARY_ROOTS / CATALOG_DB are visible.
try:
    from dotenv import load_dotenv

    load_dotenv(_BACKEND_DIR / ".env")
except ImportError:
    pass

# Ensure repo root is importable for analysis + database packages.
_REPO = _BACKEND_DIR.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from analysis.GPL_Python.preprocess import main  # noqa: E402


if __name__ == "__main__":
    # If --library-root omitted, inject WAV_LIBRARY_ROOTS entries.
    argv = list(sys.argv[1:])
    if "--library-root" not in argv:
        for root in os.environ.get("WAV_LIBRARY_ROOTS", "").split(","):
            root = root.strip()
            if root:
                argv.extend(["--library-root", root])
    if "--db" not in argv and os.environ.get("CATALOG_DB", "").strip():
        argv.extend(["--db", os.environ["CATALOG_DB"].strip()])
    raise SystemExit(main(argv))
