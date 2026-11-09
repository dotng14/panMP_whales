"""Which atoms of one build have no counterpart in another, and where they are placed.

python new_atoms.py BUILD REFERENCE [--threshold 0.8]

An atom of BUILD is "new" when its best shifted cosine (``shape_similarity``)
with every atom of REFERENCE is below the threshold. For each atom it prints
the width, placements, the best match in REFERENCE, and how many placements
fall in windows REFERENCE did not use.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from analysis.dict_learning.code.learn.shape_match import shape_similarity


def _first(atom: np.ndarray) -> np.ndarray:
    """A (k, F, W) subspace atom compares by its first component."""
    return atom[0] if atom.ndim == 3 else atom


def _atoms(folder: Path) -> dict[int, np.ndarray]:
    summary = json.loads((folder / "summary.json").read_text())
    return {int(a["template_id"]): _first(np.load(folder / a["path"])) for a in summary["atoms"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("build", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--threshold", type=float, default=0.8)
    args = parser.parse_args()

    atoms, reference = _atoms(args.build), _atoms(args.reference)
    seen = set(json.loads((args.reference / "summary.json").read_text())["window_ids"])
    rows = json.loads((args.build / "matchings.json").read_text())
    report = []
    for template_id, data in sorted(atoms.items()):
        best_id, best = max(((rid, shape_similarity(data, ref)) for rid, ref in reference.items()), key=lambda x: x[1])
        placed = [r for r in rows if int(r["template_id"]) == template_id]
        outside = sum(r["window_id"] not in seen for r in placed)
        report.append({
            "template_id": template_id, "width": int(data.shape[1]), "placements": len(placed),
            "outside_reference_windows": outside, "best_reference_id": best_id, "best_similarity": round(best, 3),
            "new": best < args.threshold,
        })
    new = [r for r in report if r["new"]]
    print(json.dumps({"n_atoms": len(report), "n_new": len(new), "atoms": report}, indent=1))


if __name__ == "__main__":
    main()
