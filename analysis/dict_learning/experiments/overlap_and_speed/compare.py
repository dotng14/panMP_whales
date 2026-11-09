"""Compare dictionary build folders: speed, coverage, overlap, atom similarity.

python compare.py NAME=FOLDER [NAME=FOLDER ...]   (FOLDER.log holds /usr/bin/time -v output)
"""
import json, re, sys
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np

from analysis.dict_learning.code.learn.shape_match import shape_similarity
from analysis.dict_learning.code.mp import OverlapRule

RULE = OverlapRule()


def _first(atom: np.ndarray) -> np.ndarray:
    """A (k, F, W) subspace atom compares by its first component."""
    return atom[0] if atom.ndim == 3 else atom


def wall_and_rss(log: Path):
    text = log.read_text() if log.is_file() else ""
    m = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", text)
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", text)
    secs = None
    if m:
        parts = [float(p) for p in m.group(1).split(":")]
        secs = sum(p * 60 ** i for i, p in enumerate(reversed(parts)))
    return secs, (int(rss.group(1)) / 1e6 if rss else None)


def stats(folder: Path) -> dict:
    s = json.loads((folder / "summary.json").read_text())
    rows = json.loads((folder / "matchings.json").read_text())
    rows = rows if isinstance(rows, list) else rows["matchings"]
    widths = {int(a["template_id"]): int(a["width"]) for a in s["atoms"]}
    atoms = [_first(np.load(folder / a["path"])) for a in s["atoms"]]
    by_window = {}
    for r in rows:
        t = int(r["t_start"])
        by_window.setdefault(r["window_id"], []).append((t, t + widths[int(r["template_id"])], int(r["template_id"])))
    pairs = any_overlap = same_atom = nested = forbidden = 0
    placed_overlapping = set()
    placed_cols = covered = 0
    for wid, spans in by_window.items():
        for i, j in combinations(range(len(spans)), 2):
            (a0, a1, ta), (b0, b1, tb) = spans[i], spans[j]
            if min(a1, b1) <= max(a0, b0):
                continue
            any_overlap += 1
            placed_overlapping.update({(wid, i), (wid, j)})
            same_atom += ta == tb
            nested += (a0 <= b0 and b1 <= a1) or (b0 <= a0 and a1 <= b1)
            forbidden += not RULE.allows(a0, a1, b0, b1)
        cover = Counter(c for a, b, _ in spans for c in range(a, b))
        placed_cols += sum(cover.values()); covered += len(cover)
    sims = [shape_similarity(a, b) for a, b in combinations(atoms, 2)]
    logs = [json.loads(l) for l in (folder / "outer_logs.jsonl").read_text().splitlines()]
    wall, rss = wall_and_rss(folder.with_suffix(".log"))
    return {
        "wall_s": wall, "rss_gb": rss,
        "coverage": s["time_coverage_fraction"],
        "energy_explained": 1 - s["energy_after"] / s["energy_before"],
        "atoms": len(atoms), "placements": len(rows),
        "windows_matched": len(by_window),
        "overlapping_pairs": any_overlap, "same_atom_pairs": same_atom, "nested_pairs": nested,
        "forbidden_pairs": forbidden,
        "placements_overlapping": len(placed_overlapping) / max(1, len(rows)),
        "cover_per_column": placed_cols / max(1, covered),
        "max_atom_similarity": max(sims) if sims else 0.0,
        "atom_pairs_over_0.8": sum(x > 0.8 for x in sims),
        "coverage_per_pass": [round(l["time_coverage_fraction"], 3) for l in logs],
        "widths": sorted(widths.values()),
    }


if __name__ == "__main__":
    out = {}
    for arg in sys.argv[1:]:
        name, path = arg.split("=", 1)
        out[name] = stats(Path(path))
    print(json.dumps(out, indent=1))
