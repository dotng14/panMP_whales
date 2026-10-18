import subprocess

from fastapi import APIRouter

from app.config import BACKEND_DIR

router = APIRouter()


def _read_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(BACKEND_DIR),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:  # noqa: BLE001 - health must never fail on a missing git
        return "unknown"


# Resolved at import, NOT on first request: this must report the commit whose
# code is actually loaded in this process. Reading it lazily would pick up a
# later `git pull` and report a sha the running process never loaded - hiding
# exactly the pulled-but-not-restarted drift this exists to catch.
RUNNING_COMMIT = _read_commit()


@router.get("/api/health")
async def health() -> dict:
    return {"status": "ok", "commit": RUNNING_COMMIT}
