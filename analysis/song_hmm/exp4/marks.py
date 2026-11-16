"""Hand marks and provisional marks from B1 novelty."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from .baselines import b1_boundaries
from .paths import REFERENCE
from .sequence import Session


def marks_path(name: str = "marks.json") -> Path:
    return REFERENCE / name


def save_marks(marks: dict[str, list[float]], path: Path | None = None) -> Path:
    path = path or marks_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(marks, indent=2))
    return path


def load_marks(path: Path | None = None) -> dict[str, list[float]]:
    path = path or marks_path()
    if not path.is_file():
        return {"R0": [], "R1": [], "R2": [], "A_to_B": []}
    return json.loads(path.read_text())


def propose_provisional_marks(
    train_sessions: Sequence[Session],
    test_sessions: Sequence[Session],
) -> dict[str, list[float]]:
    """Propose provisional theme-change times from B1 until hand marks exist."""
    train_bounds = b1_boundaries(train_sessions, min_sep_s=90.0)
    test_bounds = b1_boundaries(test_sessions, min_sep_s=90.0)
    # R0: first few in hours 1–5; R1: rest of train; R2: test
    r0 = train_bounds[:3]
    r1 = train_bounds[3:]
    r2 = test_bounds
    a_to_b = r0[:1] + r1[:4]
    return {
        "R0": r0,
        "R1": r1,
        "R2": r2,
        "A_to_B": a_to_b,
        "note": "provisional from B1 novelty; replace with hand marks on the spectrogram",
    }
