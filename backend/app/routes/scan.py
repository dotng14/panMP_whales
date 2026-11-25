"""Serve 20-year minute-energy scan results (monthly hourly plots + overviews)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from app.config import SCAN_RESULTS_DIR

router = APIRouter(prefix="/api/scan", tags=["scan"])

_MONTH_RE = re.compile(r"^\d{4}-\d{2}$")
_SECTION_RE = re.compile(
    r"<!--\s*month:(\d{4}-\d{2})\s*-->(.*?)<!--\s*/month:\1\s*-->",
    re.DOTALL,
)
_STATS_RE = re.compile(
    r"(\d+)\s+recordings?\s+contribute\.\s+(\d+)\s+hours?\s+contain\s+audio\s+\((\d+)\s+minutes?\)\."
    r"\s+Across\s+those\s+minutes\s+the\s+mean\s+energy\s+ratio\s+is\s+([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
    r"\s+and\s+the\s+mean\s+0\.1[–-]1\s+Hz\s+FFT\s+fraction\s+is\s+([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
)

_OVERVIEW_FILES: list[tuple[str, str, str]] = [
    ("monthly_scatter", "Monthly scatter", "monthly_scatter.png"),
    ("minute_scatter", "Minute scatter", "minute_scatter.png"),
    ("minute_contour2d", "2D contour", "minute_contour2d.png"),
]
_OVERVIEW_BY_NAME = {fname: fname for _, _, fname in _OVERVIEW_FILES}

_summary_cache: tuple[float, list[dict[str, Any]]] | None = None


def _parse_months(md_path: Path) -> list[dict[str, Any]]:
    global _summary_cache
    mtime = md_path.stat().st_mtime
    if _summary_cache is not None and _summary_cache[0] == mtime:
        return _summary_cache[1]

    text = md_path.read_text(encoding="utf-8", errors="replace")
    hourly_dir = md_path.parent / "hourly"
    months: list[dict[str, Any]] = []
    for match in _SECTION_RE.finditer(text):
        month = match.group(1)
        body = match.group(2)
        if not (hourly_dir / f"{month}.png").is_file():
            continue
        entry: dict[str, Any] = {
            "month": month,
            "imageUrl": f"/api/scan/hourly/{month}.png",
        }
        stats = _STATS_RE.search(body)
        if stats:
            entry["recordings"] = int(stats.group(1))
            entry["hours"] = int(stats.group(2))
            entry["minutes"] = int(stats.group(3))
            entry["meanEnergyRatio"] = float(stats.group(4))
            entry["meanFftFraction"] = float(stats.group(5))
        months.append(entry)

    _summary_cache = (mtime, months)
    return months


@router.get("/summary")
def get_scan_summary() -> dict[str, Any]:
    root = SCAN_RESULTS_DIR
    md_path = root / "monthly_hourly.md"
    if not root.is_dir():
        return {"available": False, "months": [], "overviews": []}

    months = _parse_months(md_path) if md_path.is_file() else []
    overviews = [
        {"id": oid, "label": label, "imageUrl": f"/api/scan/overview/{fname}"}
        for oid, label, fname in _OVERVIEW_FILES
        if (root / fname).is_file()
    ]
    return {
        "available": bool(months or overviews),
        "months": months,
        "overviews": overviews,
    }


@router.get("/hourly/{month}.png")
def get_hourly_image(month: str) -> FileResponse:
    if not _MONTH_RE.match(month):
        raise HTTPException(status_code=400, detail="Invalid month format (expected YYYY-MM)")
    path = SCAN_RESULTS_DIR / "hourly" / f"{month}.png"
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"No hourly plot for {month}")
    return FileResponse(path, media_type="image/png")


@router.get("/overview/{filename}")
def get_overview_image(filename: str) -> FileResponse:
    safe_name = _OVERVIEW_BY_NAME.get(filename)
    if safe_name is None:
        raise HTTPException(status_code=404, detail="Unknown overview plot")
    path = SCAN_RESULTS_DIR / safe_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Overview plot not found: {safe_name}")
    return FileResponse(path, media_type="image/png")
