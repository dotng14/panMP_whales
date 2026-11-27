"""Serve 20-year minute-energy scan results (monthly hourly plots + overviews)."""
from __future__ import annotations

import io
import re
import struct
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

from app.config import SCAN_RESULTS_DIR
from app.services.hybrid_song_service import HYBRID_MODEL_BIN

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


# ── 2023 Hybrid Motif Dictionary (from song_model.bin) ───────────────────────

_HOP_SEC = 1536.0 / 12500.0
_HZ_PER_ROW = 12500.0 / 2048.0
_dict_cache: tuple[float, dict[int, np.ndarray], list[dict[str, Any]], int] | None = None
_png_cache: dict[tuple[int, str], bytes] = {}
_MOTIF_VIEWS = ("thumb", "full")


def _load_hybrid_dictionary() -> tuple[dict[int, np.ndarray], list[dict[str, Any]], int] | None:
    global _dict_cache
    if not HYBRID_MODEL_BIN.is_file():
        return None
    mtime = HYBRID_MODEL_BIN.stat().st_mtime
    if _dict_cache is not None and _dict_cache[0] == mtime:
        return _dict_cache[1], _dict_cache[2], _dict_cache[3]

    with HYBRID_MODEL_BIN.open("rb") as f:
        magic = f.read(8)
        if magic != b"SONGMDL1":
            return None
        n_motifs, n_comp, n_rows, _n_themes = struct.unpack("<4i", f.read(16))
        ids = np.frombuffer(f.read(n_motifs * 4), dtype="<i4").tolist()
        widths = np.frombuffer(f.read(n_motifs * 4), dtype="<i4").tolist()
        means: dict[int, np.ndarray] = {}
        motifs: list[dict[str, Any]] = []
        for mid, w in zip(ids, widths):
            mean_bytes = n_rows * w * 8
            basis_bytes = n_comp * n_rows * w * 8
            mean_arr = np.frombuffer(f.read(mean_bytes), dtype="<f8").reshape((n_rows, w)).copy()
            f.seek(basis_bytes, 1)
            means[int(mid)] = mean_arr
            motifs.append({
                "id": int(mid),
                "widthCols": int(w),
                "durationSec": round(w * _HOP_SEC, 2),
                "imageUrl": f"/api/scan/hybrid-dictionary/{int(mid)}.png?view=thumb",
                "zoomImageUrl": f"/api/scan/hybrid-dictionary/{int(mid)}.png?view=full",
            })

    _png_cache.clear()
    _dict_cache = (mtime, means, motifs, int(n_rows))
    return means, motifs, int(n_rows)


@router.get("/hybrid-dictionary")
def get_hybrid_dictionary() -> dict[str, Any]:
    loaded = _load_hybrid_dictionary()
    if loaded is None:
        return {"available": False, "motifs": []}
    _means, motifs, n_rows = loaded
    return {
        "available": True,
        "model": "2023",
        "nRows": n_rows,
        "freqLo": round(2 * 12500.0 / 2048.0, 1),
        "freqHi": round((2 + n_rows - 1) * 12500.0 / 2048.0, 1),
        "motifs": motifs,
    }


def _render_motif_png(motif_id: int, arr: np.ndarray, view: str) -> bytes:
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    n_rows, n_cols = arr.shape
    # Power in dB; a per-motif colour range keeps the structure visible.
    db = 10.0 * np.log10(np.maximum(arr, 1e-9))
    vmin = float(np.percentile(db, 2.0))
    vmax = float(np.percentile(db, 99.8))
    if vmax <= vmin:
        vmax = vmin + 1.0

    duration = n_cols * _HOP_SEC
    f_lo = 2 * _HZ_PER_ROW
    f_hi = (2 + n_rows - 1) * _HZ_PER_ROW
    buf = io.BytesIO()

    if view == "thumb":
        fig = plt.figure(figsize=(1.6, 1.84), dpi=100)
        ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
        ax.imshow(db, origin="lower", aspect="auto", cmap="jet",
                  vmin=vmin, vmax=vmax, interpolation="bilinear",
                  interpolation_stage="data")
        ax.set_axis_off()
        fig.savefig(buf, format="png")
    else:
        fig, ax = plt.subplots(figsize=(6.4, 5.0), dpi=130, facecolor="white")
        # Bicubic interpolation on the dB values gives a smooth gradient instead of
        # hard blocks (only ~13 time frames per motif).
        im = ax.imshow(db, origin="lower", aspect="auto", cmap="jet",
                       vmin=vmin, vmax=vmax, interpolation="bicubic",
                       interpolation_stage="data",
                       extent=(0.0, duration, f_lo, f_hi))
        ax.set_xlabel("time within motif (s)")
        ax.set_ylabel("frequency (Hz)")
        ax.set_title(f"Motif M{motif_id}: {duration:.2f} s, {n_cols} frames", fontsize=11)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("power (dB, relative)")
        fig.tight_layout()
        fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


@router.get("/hybrid-dictionary/{motif_id}.png")
def get_hybrid_motif_png(motif_id: int, view: str = "thumb") -> Response:
    if view not in _MOTIF_VIEWS:
        raise HTTPException(status_code=400, detail="view must be 'thumb' or 'full'")
    loaded = _load_hybrid_dictionary()
    if loaded is None:
        raise HTTPException(status_code=404, detail="Hybrid dictionary model not available")
    means, _motifs, _n_rows = loaded
    arr = means.get(motif_id)
    if arr is None:
        raise HTTPException(status_code=404, detail=f"Motif {motif_id} not found")

    key = (motif_id, view)
    if key not in _png_cache:
        _png_cache[key] = _render_motif_png(motif_id, arr, view)

    return Response(content=_png_cache[key], media_type="image/png")

