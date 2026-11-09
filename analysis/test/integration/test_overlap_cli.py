"""``--overlap limited`` on both builds and on matching-pursuit, end to end."""

from __future__ import annotations

import json
from itertools import combinations

import pytest

from analysis.dict_learning.code.cli import main
from analysis.dict_learning.code.mp import OverlapRule
from analysis.test.catalog_setup import pursue
from analysis.test.synthetic import read_json

BUILDS = {
    "randomized-build": ["--m-per-outer", "3", "--n-min", "2", "--refine-iters", "1", "--seed", "0"],
    # Two lengths, so 6-column atoms can sit inside 12-column ones.
    "pan-build": ["--lengths-s", "0.06,0.12", "--m-per-outer", "4", "--n-min", "2", "--refine-iters", "1"],
}
COMMON = ["--workers", "1", "--max-outer", "2", "--min-coverage", "1.0"]


def _overlaps(folder, widths) -> tuple[int, int]:
    """(pairs of placements in one window that overlap at all, pairs the rule forbids)."""
    rule = OverlapRule()
    by_window: dict[str, list[tuple[int, int]]] = {}
    for row in json.loads((folder / "matchings.json").read_text()):
        start = int(row["t_start"])
        by_window.setdefault(row["window_id"], []).append((start, start + widths[int(row["template_id"])]))
    touching = forbidden = 0
    for spans in by_window.values():
        for (a0, a1), (b0, b1) in combinations(spans, 2):
            touching += min(a1, b1) > max(a0, b0)
            forbidden += not rule.allows(a0, a1, b0, b1)
    return touching, forbidden


@pytest.mark.parametrize("command", BUILDS)
def test_builds_and_pursuit_keep_to_the_rule(command, gpl_dir, tmp_path):
    runs = {}
    for mode in ("free", "limited"):
        out = tmp_path / f"{command}_{mode}"
        argv = [command, "--gpl-dir", str(gpl_dir), "--out", str(out), *BUILDS[command], *COMMON, "--overlap", mode]
        assert main(argv) == 0
        summary = read_json(out / "summary.json")
        assert summary["overlap"] == mode
        widths = {int(a["template_id"]): int(a["width"]) for a in summary["atoms"]}
        runs[mode] = _overlaps(out, widths)
        for mp_mode in ("per_window", "global"):
            mp_out = tmp_path / f"{command}_{mode}_{mp_mode}"
            assert pursue(gpl_dir, out, mp_out, "--overlap", mode, "--mp-mode", mp_mode) == 0
            assert read_json(mp_out / "summary.json")["overlap"] == mode
            assert read_json(mp_out / "summary.json")["n_matchings"] > 0
            if mode == "limited":
                assert _overlaps(mp_out, widths)[1] == 0
    # Limited placements never break the rule (free ones may; the unit tests
    # show a case), and they still overlap where the rule allows.
    assert runs["limited"][1] == 0
    assert runs["limited"][0] <= runs["free"][0]
