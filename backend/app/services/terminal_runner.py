"""Subprocess entry point for one terminal command.

Run as ``python -m app.services.terminal_runner <result_json_path>`` with a JSON
payload on stdin. It executes the command in this *separate* process and writes
``{"ok": bool, "lines": [...]}`` to the result path. Running out-of-process is
deliberate: the analysis is CPU-bound Python that holds the GIL for minutes, so
doing it in a thread inside the web server would starve the event loop and hang
every other request. A child process leaves the server fully responsive.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# Pin the BLAS/OpenMP pools to one thread each, before anything can import
# NumPy — OpenBLAS reads these only at library load time.
#
# Matching pursuit parallelises over candidate trials with its own thread pool,
# and each trial's PCA rebuild calls into LAPACK. OpenBLAS keeps ONE global
# thread pool (this build: MAX_THREADS=24), so N candidate threads calling it at
# once oversubscribe it many times over — which is what made extend die with a
# native access violation partway through a round instead of finishing.
for _var in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")

# `analysis` and `database` live at the repo root, one level above backend/ (the
# child's cwd); make them importable even when the package is not pip-installed.
_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m app.services.terminal_runner <result_path>", file=sys.stderr)
        return 2
    result_path = Path(sys.argv[1])

    from app.services.command_service import PROGRESS_PREFIX, CommandContext, run_action

    sys.stdin.reconfigure(encoding="utf-8")
    payload = json.loads(sys.stdin.read())

    def _emit(progress: dict) -> None:
        # One JSON dict per line behind the sentinel; the parent tails stdout for
        # these to drive the form's live progress + ETA readout.
        sys.stdout.write(PROGRESS_PREFIX + json.dumps(progress) + "\n")
        sys.stdout.flush()

    ctx = CommandContext(
        stem=payload["stem"],
        segmentation_id=payload.get("segmentation_id"),
        dictionary_id=payload.get("dictionary_id"),
        current_template=payload.get("current_template"),
        source_wav=payload.get("source_wav"),
        progress_callback=_emit,
    )
    result = run_action(ctx, payload.get("action") or "", payload.get("params") or {})
    result_path.write_text(
        json.dumps(
            {
                "ok": result.ok,
                "lines": result.lines,
                "new_dictionary_id": result.new_dictionary_id,
            }
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
