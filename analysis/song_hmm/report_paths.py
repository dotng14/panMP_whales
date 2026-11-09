"""All experiment reports live in the repository's reports/ folder.

Each report generator still builds its markdown with links relative to its own
package folder (figures/, plan.md, ../song_hmm/...). ``write_report`` rewrites
those links so they work from reports/, and points links to an old report
location at the report's new file.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPORTS = REPO / "reports"
ANALYSIS = REPO / "analysis"

# where each report used to be  ->  its file in reports/
MOVED = {
    ANALYSIS / "song_hmm" / "report.md": "exp1_slot_hmm.md",
    ANALYSIS / "song_hmm_phrase" / "report.md": "exp2_phrase_hmm.md",
    ANALYSIS / "song_hmm_phrase" / "pipeline_check" / "report.md": "exp2_pipeline_check.md",
    ANALYSIS / "song_exp3" / "report.md": "exp3_inputs_and_models.md",
    ANALYSIS / "song_exp3" / "best_combination.md": "exp3_best_combination.md",
}

_LINK = re.compile(r"\]\(([^)\s]+)\)")


def relink(text: str, src_dir: Path, dst_dir: Path = REPORTS) -> str:
    """Rewrite relative markdown link targets written for ``src_dir`` so they work from ``dst_dir``."""
    src_dir, dst_dir = Path(src_dir).resolve(), Path(dst_dir).resolve()

    def fix(m: re.Match) -> str:
        target = m.group(1)
        if re.match(r"^[a-z]+:", target) or target.startswith("#"):
            return m.group(0)  # absolute URL or same-page anchor
        path, _, anchor = target.partition("#")
        full = (src_dir / path).resolve()
        if full in MOVED:
            full = REPORTS / MOVED[full]
        rel = os.path.relpath(full, dst_dir).replace(os.sep, "/")
        if target.endswith("/") and not rel.endswith("/"):
            rel += "/"
        return f"]({rel}{'#' + anchor if anchor else ''})"

    return _LINK.sub(fix, text)


def write_report(text: str, src_dir: Path, old_path: Path) -> Path:
    """Write a report generated in ``src_dir`` to its file in reports/ and return that path."""
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / MOVED[Path(old_path).resolve()]
    out.write_text(relink(text, src_dir), encoding="utf-8")
    return out
