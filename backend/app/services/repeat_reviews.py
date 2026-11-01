"""Repeat chunk reassignment reviews — flat repeat_reviews.json + staged labels.

When a chunk is staged toward a catalog template with known ``widthBins``, its
display/sample span is remapped to that destination width (same start anchors).
``remap_chunk_to_template_width`` must match
``frontend/src/utils/repeatGrouping.ts::remapChunkToTemplateWidth``.

Chunks stay under their original catalog template until submit-rebuild; only the
staged label and remapped span change in the GET payload.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

REPEATS_RESPONSE_VERSION = 7


def _template_number(template: dict[str, Any]) -> int:
    number = template.get("templateNumber")
    if isinstance(number, int):
        return number
    template_id = str(template.get("id", ""))
    if template_id.startswith("template-"):
        suffix = template_id.removeprefix("template-")
        if suffix.isdigit():
            return int(suffix)
    return 0


def load_reviews(output_dir: Path) -> dict[str, dict[str, int]]:
    """Read flat ``{ chunkId: { targetTemplateId } }``; missing file → ``{}``.

    Legacy nested ``{ templateId: { chunkId: { incorrect } } }`` entries are
    ignored — there is no mapping from ``incorrect`` to ``targetTemplateId``.
    """
    review_path = output_dir / "repeat_reviews.json"
    if not review_path.exists():
        return {}
    raw = json.loads(review_path.read_text())
    if not isinstance(raw, dict):
        return {}
    flat: dict[str, dict[str, int]] = {}
    for key, value in raw.items():
        if not isinstance(value, dict):
            continue
        if "targetTemplateId" in value:
            flat[str(key)] = {"targetTemplateId": int(value["targetTemplateId"])}
    return flat


def save_review(
    output_dir: Path,
    chunk_id: str,
    target_template_id: int,
    original_template_number: int,
) -> None:
    """Persist one reassignment; remove the entry when target equals original."""
    reviews = load_reviews(output_dir)
    if target_template_id == original_template_number:
        reviews.pop(chunk_id, None)
    else:
        reviews[chunk_id] = {"targetTemplateId": target_template_id}
    review_path = output_dir / "repeat_reviews.json"
    temporary_path = review_path.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(reviews, indent=2))
    temporary_path.replace(review_path)


def clear_reviews(output_dir: Path) -> None:
    review_path = output_dir / "repeat_reviews.json"
    if review_path.exists():
        review_path.write_text("{}\n")


def staged_target_template_number(
    chunk: dict[str, Any],
    reviews: dict[str, dict[str, int]],
) -> int:
    review = reviews.get(chunk["id"])
    if review is not None:
        return int(review["targetTemplateId"])
    return int(chunk["originalTemplateNumber"])


def _resolve_width_bins(meta: dict[str, Any] | None) -> int | None:
    """Prefer explicit ``widthBins``; fall back to ``average.cols`` for older payloads."""
    if not meta:
        return None
    width = meta.get("widthBins")
    if width is not None:
        w = int(width)
        return w if w >= 1 else None
    average = meta.get("average")
    if isinstance(average, dict) and average.get("cols") is not None:
        w = int(average["cols"])
        return w if w >= 1 else None
    return None


def remap_chunk_to_template_width(
    chunk: dict[str, Any],
    width_bins: int | None,
) -> dict[str, Any]:
    """Resize chunk span to ``width_bins`` keeping start anchors; no-op if impossible.

    Uses the same formulas as ``repeats_from_catalog`` for end col/sample.
    Chunk ``id`` is left unchanged.
    """
    if width_bins is None or int(width_bins) < 1:
        return chunk
    width_bins = int(width_bins)
    time_bin_start = chunk.get("timeBinStart")
    window_bin_end = chunk.get("windowBinEnd")
    overlap = chunk.get("overlap")
    stride = chunk.get("stride")
    if (
        time_bin_start is None
        or window_bin_end is None
        or overlap is None
        or stride is None
    ):
        return chunk
    time_bin_start = int(time_bin_start)
    window_bin_end = int(window_bin_end)
    overlap = int(overlap)
    stride = max(1, int(stride))
    new_end_bin = min(window_bin_end, time_bin_start + width_bins)
    if new_end_bin <= time_bin_start:
        return chunk

    display_cols = chunk.get("displayCols")
    start_col = int(chunk["sourceStartCol"])
    end_col = max(start_col + 1, math.ceil(new_end_bin / stride))
    if display_cols is not None:
        end_col = min(int(display_cols), end_col)

    source_start_sample = int(chunk["sourceStartSample"])
    span_bins = new_end_bin - time_bin_start
    source_end_sample = source_start_sample + span_bins * overlap - 1
    window_end_sample = chunk.get("windowEndSample")
    if window_end_sample is not None:
        source_end_sample = min(int(window_end_sample), source_end_sample)

    out = dict(chunk)
    out["sourceEndCol"] = end_col
    out["sourceEndSample"] = source_end_sample
    return out


def group_chunks_by_effective_template(
    templates: list[dict[str, Any]],
    *,
    pinned_template_numbers: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Regroup chunks by staged label; keep every original template number.

    Emptied originals (e.g. all of template 2 moved to 1) remain as slots with
    ``chunks: []`` so navigation stays contiguous when later templates exist.
    """
    catalog_meta: dict[int, dict[str, Any]] = {}
    all_chunks: list[dict[str, Any]] = []

    for template in templates:
        tid = _template_number(template)
        catalog_meta[tid] = {
            "sourceSegmentId": template.get("sourceSegmentId"),
            "average": template.get("average"),
            "widthBins": _resolve_width_bins(template),
            "name": template.get("name"),
        }
        for chunk in template.get("chunks", []):
            item = dict(chunk)
            item.setdefault("originalTemplateNumber", tid)
            if "stagedTemplateNumber" not in item:
                item["stagedTemplateNumber"] = int(item["originalTemplateNumber"])
            all_chunks.append(item)

    original_numbers = sorted(
        {_template_number(template) for template in templates}
        | {int(n) for n in (pinned_template_numbers or [])}
    )
    original_set = set(original_numbers)

    buckets: dict[int, list[dict[str, Any]]] = {n: [] for n in original_numbers}
    width_by_number = {
        tid: meta.get("widthBins") for tid, meta in catalog_meta.items()
    }

    for chunk in all_chunks:
        bucket = int(chunk.get("stagedTemplateNumber", chunk["originalTemplateNumber"]))
        remapped = remap_chunk_to_template_width(chunk, width_by_number.get(bucket))
        buckets.setdefault(bucket, []).append(remapped)

    synthetic = sorted(n for n in buckets if n not in original_set)
    leading_zero = [0] if 0 in synthetic else []
    trailing = [n for n in synthetic if n != 0]
    navigable = [*leading_zero, *original_numbers, *trailing]

    result: list[dict[str, Any]] = []
    for number in navigable:
        meta = catalog_meta.get(number, {})
        width = meta.get("widthBins")
        result.append(
            {
                "id": f"template-{number}",
                "templateNumber": number,
                "name": meta.get("name"),
                "sourceSegmentId": meta.get("sourceSegmentId"),
                "widthBins": width,
                "chunks": buckets.get(number, []),
                "average": meta.get("average"),
            }
        )
    return result


def stamp_reviews_onto_templates(
    raw_templates: list[dict[str, Any]],
    reviews: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Keep original template grouping; expose staged labels and remap widths.

    Chunks stay under their catalog template until submit-rebuild. Staged
    destination widths are applied in place so the reviewer can preview crop
    without chunks jumping buckets.
    """
    width_by_number: dict[int, int | None] = {}
    for template in raw_templates:
        tid = _template_number(template)
        width_by_number[tid] = _resolve_width_bins(template)

    template_payloads: list[dict[str, Any]] = []
    for template in raw_templates:
        tid = _template_number(template)
        stamped_chunks = []
        for chunk in template.get("chunks", []):
            stamped = dict(chunk)
            stamped.setdefault("originalTemplateNumber", tid)
            staged = staged_target_template_number(stamped, reviews)
            stamped["stagedTemplateNumber"] = staged
            dest_width = width_by_number.get(staged)
            stamped = remap_chunk_to_template_width(stamped, dest_width)
            stamped_chunks.append(stamped)
        stamped_template = dict(template)
        if stamped_template.get("widthBins") is None:
            stamped_template["widthBins"] = _resolve_width_bins(stamped_template)
        stamped_template["chunks"] = stamped_chunks
        template_payloads.append(stamped_template)
    return {"version": REPEATS_RESPONSE_VERSION, "templates": template_payloads}


def find_chunk_in_raw(
    raw_templates: list[dict[str, Any]],
    chunk_id: str,
) -> dict[str, Any] | None:
    for template in raw_templates:
        tid = _template_number(template)
        for chunk in template.get("chunks", []):
            if chunk["id"] == chunk_id:
                result = dict(chunk)
                result.setdefault("originalTemplateNumber", tid)
                return result
    return None


def all_chunk_ids(raw_templates: list[dict[str, Any]]) -> set[str]:
    return {
        chunk["id"]
        for template in raw_templates
        for chunk in template.get("chunks", [])
    }
