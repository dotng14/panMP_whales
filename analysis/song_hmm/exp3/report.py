"""Experiment 3 report.

    python -m analysis.song_exp3.report

Reads work/eval.json (run evaluate first), work/models/results.json, the screening
files, and Experiment 2's and the pipeline check's saved numbers. Writes reports/exp3_inputs_and_models.md
and figures/. Refuses to write a report without the three required sections.
"""

from __future__ import annotations

import json

from analysis.report_paths import write_report

from .evaluate import hand_labels
from .figures import all_figures
from .models.common import HERE, WORK
from .report_text import REQUIRED, render

PHRASE = HERE.parent / "song_hmm_phrase"


def load(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def main() -> None:
    data = {
        "eval": load(WORK / "eval.json"),
        "results": load(WORK / "models" / "results.json", {}),
        "inputs_used": load(WORK / "models" / "inputs_used.json", {}),
        "screen": load(WORK / "inputs" / "screen.json", {}),
        "units": load(WORK / "inputs" / "units_screen.json", {}),
        "exp2_report": load(PHRASE / "work" / "report.json", {}),
        "pipeline_check": load(PHRASE / "pipeline_check" / "work" / "results.json", {}),
        "boundary_checks": (PHRASE / "diagnostics" / "boundary_checks.txt").read_text(encoding="utf-8") if (PHRASE / "diagnostics" / "boundary_checks.txt").exists() else "",
        "pairs": load(WORK / "pairs" / "judgments.json"),
        "pairs_key": load(WORK / "pairs" / "key.json"),
    }
    if data["eval"] is None:
        raise SystemExit("run python -m analysis.song_exp3.evaluate first")
    hand = hand_labels()
    all_figures(data["eval"], data["results"], hand)
    text = render(data)
    missing = [h for h in REQUIRED if h not in text]
    if missing:
        raise RuntimeError(f"report is missing required sections: {missing}")
    print(f"wrote {write_report(text, HERE, HERE / 'report.md')}")


if __name__ == "__main__":
    main()
