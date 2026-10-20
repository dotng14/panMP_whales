"""Build the frontend RepeatMatches payload from catalog rows + panel.bin."""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from analysis.GPL_Python.gpl_outputs import clip_intensity_percentile, display_from_spwhiten
from analysis.GPL_Python.panel_bin import read_panel_header
from analysis.dict_learning.code.core.utils import patch_template_distance
from database import catalog

REPEATS_PAYLOAD_VERSION = 7
_AVERAGE_MAGNITUDE_PERCENTILE = 99.9
_DISTANCE_EPS = 1e-12


def _average_raster(template_data: np.ndarray) -> dict:
    clipped, _, _ = clip_intensity_percentile(
        template_data, hi_percentile=_AVERAGE_MAGNITUDE_PERCENTILE
    )
    display = display_from_spwhiten(clipped)
    norm_max = float(np.percentile(display, _AVERAGE_MAGNITUDE_PERCENTILE))
    if norm_max <= 0:
        norm_max = 1.0
    magnitude_u8 = np.clip(display / norm_max * 255.0, 0, 255).astype(np.uint8)
    rows, cols = display.shape
    return {
        "rows": int(rows),
        "cols": int(cols),
        "normMax": norm_max,
        "magnitude": base64.b64encode(magnitude_u8.tobytes()).decode("ascii"),
    }


def chunk_distance_norm(
    patch: np.ndarray,
    template_data: np.ndarray,
) -> float | None:
    """Unexplained energy fraction after LS fit of ``template_data`` to ``patch``.

    Returns ``None`` when the patch has ~zero energy (undefined relative distance).
    """
    residual, patch_energy, _explained = patch_template_distance(
        patch, template_data, return_parts=True
    )
    if patch_energy <= _DISTANCE_EPS:
        return None
    return float(np.clip(float(residual) / float(patch_energy), 0.0, 1.0))


def sort_chunks_by_fit(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sort chunks best-match-first by ``distanceNorm`` (undefined distances last)."""
    return sorted(
        chunks,
        key=lambda c: c["distanceNorm"] if c.get("distanceNorm") is not None else math.inf,
    )


def _crop_patch_from_window(
    window: np.ndarray,
    *,
    t_start: int,
    width: int,
    n_freq: int,
) -> np.ndarray | None:
    """Crop ``(n_freq, width)`` from a window, zero-padding a short tail."""
    if window.ndim != 2 or width < 1 or n_freq < 1:
        return None
    cols = int(window.shape[1])
    if t_start < 0 or t_start >= cols:
        return None
    avail = min(width, cols - t_start)
    if avail < 1:
        return None
    freq = min(n_freq, int(window.shape[0]))
    patch = np.zeros((n_freq, width), dtype=np.float64)
    patch[:freq, :avail] = np.asarray(
        window[:freq, t_start : t_start + avail], dtype=np.float64
    )
    return patch


def build_repeats_payload_from_catalog(
    conn,
    *,
    segmentation_id: int,
    dictionary_id: int | None = None,
) -> dict[str, Any]:
    """Map catalog templates/matchings into the existing frontend repeats JSON."""
    seg = catalog.get_segmentation(conn, segmentation_id)
    if seg is None:
        raise KeyError(f"segmentation {segmentation_id} not found")
    panel_path = Path(seg["panel_path"]) if seg.get("panel_path") else None
    if panel_path is None or not panel_path.is_file():
        # Fall back to conventional location under output_dir
        panel_path = Path(seg["output_dir"]) / "panel.bin"
    panel_header = read_panel_header(panel_path)
    if panel_header is None:
        raise FileNotFoundError(f"panel.bin unreadable: {panel_path}")

    if dictionary_id is None:
        d = catalog.get_dictionary_for_segmentation(conn, segmentation_id)
        if d is None:
            return {"version": REPEATS_PAYLOAD_VERSION, "templates": []}
        dictionary_id = int(d["dictionary_id"])

    members = catalog.list_dictionary_templates(conn, dictionary_id)
    member_tids = [int(mem["template_id"]) for mem in members]
    # Ignore legacy display_label_by_template_id maps. They drifted +1 across
    # rebuilds (e.g. {1:2,2:3,3:4}) and disagreed with overlay/index labels.
    # Seeds stay 0-based in the catalog; show them as tid+1. Rebuilds that
    # already store UI labels as template_id show tid as-is.
    zero_based = bool(member_tids) and min(member_tids) == 0
    matchings = catalog.list_matchings(conn, dictionary_id)
    windows = catalog.list_windows_for_segmentation(conn, segmentation_id)
    window_offsets: dict[str, int] = {}
    window_metadata: dict[str, dict] = {}
    window_arrays: dict[str, np.ndarray] = {}
    total_source_cols = 0
    for w in windows:
        wid = w["window_id"]
        # Prefer sidecar JSON if present for cols/overlap
        meta: dict[str, Any] = {
            "cols": None,
            "overlap": w.get("overlap"),
            "sample_freq": w.get("sample_freq"),
            "start_sample": w.get("start_sample"),
            "end_sample": w.get("end_sample"),
        }
        if w.get("json_path") and Path(w["json_path"]).is_file():
            meta.update(json.loads(Path(w["json_path"]).read_text()))
        arr = None
        if w.get("npy_path") and Path(w["npy_path"]).is_file():
            arr = catalog.load_template_array(w["npy_path"])
            if meta.get("cols") is None:
                meta["cols"] = int(arr.shape[1])
        if meta.get("cols") is None:
            continue
        window_offsets[wid] = total_source_cols
        window_metadata[wid] = meta
        if arr is not None:
            window_arrays[wid] = np.asarray(arr, dtype=np.float64)
        total_source_cols += int(meta["cols"])

    display_cols = int(panel_header["timeCols"])
    stride = max(1, math.ceil(total_source_cols / display_cols)) if total_source_cols else 1

    matchings_by_tid: dict[int, list] = {}
    for m in matchings:
        # Join natural template_id via members
        tid = None
        for mem in members:
            if mem["template_pk"] == m["template_pk"]:
                tid = int(mem["template_id"])
                break
        if tid is None:
            continue
        matchings_by_tid.setdefault(tid, []).append(m)

    any_meta = next(iter(window_metadata.values()), {})
    stft_overlap = int(any_meta.get("overlap") or 1)
    sample_freq = float(any_meta.get("sample_freq") or 1.0)

    template_payloads = []
    for mem in members:
        tid = int(mem["template_id"])
        display_tid = tid + 1 if zero_based else tid
        data = catalog.load_template_array(mem["npy_path"])
        template_width = int(data.shape[-1])
        n_freq = int(data.shape[-2]) if data.ndim >= 2 else int(data.shape[0])
        chunks = []
        seen: set[tuple[int, int]] = set()
        for m in matchings_by_tid.get(tid, []):
            wid = m.get("window_id")
            if wid not in window_offsets or wid not in window_metadata:
                continue
            metadata = window_metadata[wid]
            window_cols = int(metadata["cols"])
            t_start = int(m["t_start"] or 0)
            if t_start >= window_cols:
                continue
            time_bin_start = window_offsets[wid] + t_start
            window_bin_end = window_offsets[wid] + window_cols
            source_end = min(window_bin_end, time_bin_start + template_width)
            overlap = int(metadata.get("overlap") or stft_overlap)
            window_end_sample = int(metadata["end_sample"])
            source_start_sample = int(metadata["start_sample"]) + t_start * overlap
            source_end_sample = min(
                window_end_sample,
                source_start_sample + (source_end - time_bin_start) * overlap - 1,
            )
            key = (source_start_sample, source_end_sample)
            if key in seen:
                continue
            seen.add(key)
            start_col = min(display_cols - 1, time_bin_start // stride)
            end_col = min(display_cols, max(start_col + 1, math.ceil(source_end / stride)))
            chunk: dict[str, Any] = {
                "id": f"template-{tid}:{source_start_sample}:{source_end_sample}",
                "sourceStartCol": start_col,
                "sourceEndCol": end_col,
                "sourceStartSample": source_start_sample,
                "sourceEndSample": source_end_sample,
                "windowId": wid,
                "windowStartCol": t_start,
                "amplitude": m.get("amplitude"),
                "originalTemplateNumber": display_tid,
                # Geometry for remapping to a different template width on reassignment.
                "timeBinStart": time_bin_start,
                "windowBinEnd": window_bin_end,
                "windowEndSample": window_end_sample,
                "overlap": overlap,
                "stride": stride,
                "displayCols": display_cols,
            }
            window_arr = window_arrays.get(wid)
            if window_arr is not None:
                patch = _crop_patch_from_window(
                    window_arr,
                    t_start=t_start,
                    width=template_width,
                    n_freq=n_freq,
                )
                if patch is not None:
                    try:
                        dist = chunk_distance_norm(patch, data)
                    except ValueError:
                        dist = None
                    if dist is not None:
                        chunk["distanceNorm"] = dist
            chunks.append(chunk)
        # Order patches by fit to the template atom (best match first) so the
        # reviewer sees closest-to-average repeats first, worst last.
        chunks = sort_chunks_by_fit(chunks)
        # Keep empty templates so numbering stays contiguous in the navigator
        # (e.g. template 4 with 0 chunks still appears as a slot).
        average = _average_raster(data)
        average["durationSec"] = template_width * stft_overlap / sample_freq
        member_label = mem.get("label")
        named = bool(str(member_label).strip()) if member_label is not None else False
        template_payloads.append(
            {
                "id": f"template-{display_tid}",
                "templateNumber": display_tid,
                "name": str(member_label).strip() if named else None,
                "sourceSegmentId": mem.get("source_segment_id"),
                "widthBins": template_width,
                "chunks": chunks,
                "average": average,
            }
        )
    template_payloads.sort(key=lambda t: int(t["templateNumber"]))
    return {"version": REPEATS_PAYLOAD_VERSION, "templates": template_payloads}
