"""Shared helpers for dictionary learning (coverage, residual crops, copies)."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .objects import Dictionary, Matching, Segment, Template, Window
from ..learn.seed_from_gpl import mean_center


def as_stack(
    stack: Sequence[Window] | dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if isinstance(stack, dict):
        return {k: np.asarray(v, dtype=np.float64) for k, v in stack.items()}
    return {w.window_id: mean_center(w.data) for w in stack}


def copy_dictionary(dictionary: Dictionary) -> Dictionary:
    out = Dictionary()
    for template in dictionary:
        out.add(
            Template(
                template_id=template.template_id,
                data=np.array(template.data, dtype=np.float64, copy=True),
                source_segment_id=template.source_segment_id,
                variances=(
                    None
                    if template.variances is None
                    else np.array(template.variances, dtype=np.float64, copy=True)
                ),
            )
        )
    return out


def union_interval_length(intervals: list[tuple[int, int]]) -> int:
    """Length of the union of half-open ``[start, end)`` integer intervals."""
    if not intervals:
        return 0
    ordered = sorted((int(a), int(b)) for a, b in intervals if b > a)
    if not ordered:
        return 0
    total = 0
    cur_start, cur_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
        else:
            total += cur_end - cur_start
            cur_start, cur_end = start, end
    total += cur_end - cur_start
    return total


def coverage_from_matchings(
    stack: dict[str, np.ndarray],
    matchings: Sequence[Matching],
    *,
    template_widths: dict[int, int] | None = None,
    template_width: int | None = None,
) -> tuple[int, int, int, int]:
    """Return ``(n_windows_matched, n_windows_total, n_bins_covered, n_bins_total)``.

    Prefer ``template_widths[template_id]``; fall back to ``template_width`` then ``1``.
    """
    n_windows_total = len(stack)
    n_bins_total = sum(int(arr.shape[1]) for arr in stack.values())
    by_window: dict[str, list[tuple[int, int]]] = {}
    for matching in matchings:
        wid = matching.window_id
        if wid not in stack:
            continue
        if template_widths is not None:
            width = int(template_widths.get(matching.template_id, template_width or 1))
        else:
            width = int(template_width if template_width is not None else 1)
        t_width = stack[wid].shape[1]
        start = max(0, int(matching.t_start))
        end = min(t_width, start + width)
        if end > start:
            by_window.setdefault(wid, []).append((start, end))
    n_windows_matched = len(by_window)
    n_bins_covered = sum(union_interval_length(ivs) for ivs in by_window.values())
    return n_windows_matched, n_windows_total, n_bins_covered, n_bins_total


def crop_candidates_from_residual(
    base_segments: Sequence[Segment],
    residual: dict[str, np.ndarray],
) -> list[Segment]:
    """Re-crop each candidate patch from the current residual at its fixed location."""
    candidates: list[Segment] = []
    for seg in base_segments:
        window = residual.get(seg.window_id)
        if window is None:
            continue
        patch = np.array(window[:, seg.t_start : seg.t_end], copy=True)
        if patch.size == 0 or patch.shape[1] == 0:
            continue
        if float(np.sum(patch**2)) <= 0.0:
            continue
        # tools.py historically also required matching width; keep that guard.
        if patch.shape[1] != seg.data.shape[1]:
            continue
        candidates.append(
            Segment(
                segment_id=seg.segment_id,
                data=patch,
                window_id=seg.window_id,
                t_start=seg.t_start,
                t_end=seg.t_end,
                global_start_sample=seg.global_start_sample,
                global_end_sample=seg.global_end_sample,
                detection_index=seg.detection_index,
                boundary_cross_flag=seg.boundary_cross_flag,
            )
        )
    return candidates


def window_uncovered_fraction(
    window_id: str,
    residual: dict[str, np.ndarray],
    covered_intervals: dict[str, list[tuple[int, int]]] | None,
) -> float:
    """Fraction of time bins in ``window_id`` not covered by ``covered_intervals``."""
    arr = residual.get(window_id)
    if arr is None:
        return 0.0
    total = int(arr.shape[1])
    if total <= 0:
        return 0.0
    covered = union_interval_length(list((covered_intervals or {}).get(window_id, [])))
    covered = min(max(covered, 0), total)
    return 1.0 - (covered / float(total))


def select_low_coverage_seed_windows(
    residual: dict[str, np.ndarray],
    candidates: Sequence[Segment],
    covered_intervals: dict[str, list[tuple[int, int]]] | None,
    *,
    window_order: Sequence[str] | None = None,
    neighborhood: int = 0,
) -> tuple[str | None, set[str]]:
    """Pick the least-covered window that still has candidates (+ optional neighbors).

    Returns ``(anchor_window_id, seed_window_ids)``. When no candidate windows
    exist, returns ``(None, set())``.

    Tie-break among equal uncovered fractions: highest max candidate energy in
    that window (prefers a strong local motif when coverage is tied, e.g. round 0).

    ``neighborhood`` includes ±N windows in ``window_order`` (residual key order
    by default) around the anchor.
    """
    if neighborhood < 0:
        raise ValueError(f"neighborhood must be >= 0, got {neighborhood}")

    by_window: dict[str, list[Segment]] = {}
    for seg in candidates:
        by_window.setdefault(seg.window_id, []).append(seg)
    if not by_window:
        return None, set()

    order = list(window_order) if window_order is not None else list(residual.keys())
    index = {wid: i for i, wid in enumerate(order)}

    def _score(wid: str) -> tuple[float, float]:
        energies = [float(s.energy) for s in by_window[wid]]
        return (
            window_uncovered_fraction(wid, residual, covered_intervals),
            max(energies) if energies else 0.0,
        )

    anchor = max(by_window.keys(), key=_score)
    seed_ids: set[str] = {anchor}
    if neighborhood > 0 and anchor in index:
        i0 = index[anchor]
        for j in range(i0 - neighborhood, i0 + neighborhood + 1):
            if 0 <= j < len(order):
                wid = order[j]
                if wid in by_window:
                    seed_ids.add(wid)
    # Include neighbor windows even if they currently have no candidates only
    # when they appear in by_window — seed pool is candidate-driven.
    return anchor, seed_ids


def filter_candidates_by_windows(
    candidates: Sequence[Segment],
    window_ids: set[str],
) -> list[Segment]:
    """Keep candidates whose ``window_id`` is in ``window_ids``."""
    if not window_ids:
        return []
    return [s for s in candidates if s.window_id in window_ids]


def patch_template_distance(
    patch: np.ndarray,
    template: np.ndarray | Template,
    *,
    return_parts: bool = False,
) -> float | tuple[float, float, float]:
    """Distance = residual energy after least-squares fit of ``template`` to ``patch``.

    For a patch $x$ and template subspace $\\mathcal{V}$ (rank-1 or $k$-mode)::

        d(x, V) = ||x - Π_V x||_F² = ||x||_F² − ||Π_V x||_F²

    * Rank-1 ``(F, W)``: optimal amplitude
      ``a = ⟨x, v⟩ / ||v||²``, explained energy ``a² ||v||² = ⟨x, v⟩² / ||v||²``.
    * Subspace ``(k, F, W)`` with orthonormal modes (as stored by PCA atoms):
      explained energy ``Σ_m ⟨x, v_m⟩²``.

    Parameters
    ----------
    patch
        Array shaped ``(F, W)``.
    template
        ``Template``, rank-1 ``(F, W)``, or subspace ``(k, F, W)``.
    return_parts
        If true, return ``(distance, patch_energy, explained_energy)``.

    Returns
    -------
    float or tuple
        Residual energy (distance), optionally with ``||x||²`` and explained energy.
    """
    x = np.asarray(patch, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"patch must be 2-D (F, W), got shape {x.shape}")

    if isinstance(template, Template):
        basis = np.asarray(template.basis, dtype=np.float64)
    else:
        arr = np.asarray(template, dtype=np.float64)
        if arr.ndim == 2:
            basis = arr[np.newaxis, ...]
        elif arr.ndim == 3:
            basis = arr
        else:
            raise ValueError(
                f"template must be (F, W) or (k, F, W), got shape {arr.shape}"
            )

    if basis.shape[1:] != x.shape:
        raise ValueError(
            f"patch shape {x.shape} does not match template spatial shape "
            f"{basis.shape[1:]}"
        )

    patch_energy = float(np.sum(x * x))
    k = int(basis.shape[0])
    if k == 1:
        v = basis[0]
        v_energy = float(np.sum(v * v))
        if v_energy <= 0.0:
            explained = 0.0
        else:
            inner = float(np.sum(x * v))
            explained = (inner * inner) / v_energy
    else:
        # Orthonormal subspace modes: ||Π x||² = Σ ⟨x, v_m⟩²
        explained = 0.0
        for m in range(k):
            inner = float(np.sum(x * basis[m]))
            explained += inner * inner

    explained = min(explained, patch_energy)  # numerical guard
    distance = patch_energy - explained
    if return_parts:
        return distance, patch_energy, explained
    return distance


def template_energy(template: Template | np.ndarray) -> float:
    """Frobenius energy of a template / subspace basis."""
    if isinstance(template, Template):
        arr = np.asarray(template.basis, dtype=np.float64)
    else:
        arr = np.asarray(template, dtype=np.float64)
    return float(np.sum(arr * arr))


def patch_template_relative_distance(
    patch: np.ndarray,
    template: np.ndarray | Template,
) -> float:
    """Fraction of patch energy left unexplained by ``template``.

    ``r(x, V) = d(x, V) / ||x||_F²`` on ``[0, 1]``, with ``d`` from
    :func:`patch_template_distance`. Scale-free: for a rank-1 atom this is
    ``1 − cos²`` of the angle between patch and template, so one threshold
    applies across atoms, amplitudes, and dictionaries.

    Empty patches (zero energy) score ``1.0`` — nothing to explain.
    """
    distance, patch_energy, _ = patch_template_distance(
        patch, template, return_parts=True
    )
    if patch_energy <= 0.0:
        return 1.0
    return float(distance / patch_energy)


def filter_matchings_by_relative_distance(
    signal: Sequence[Window] | dict[str, np.ndarray],
    matchings: Sequence[Matching],
    template: Template | np.ndarray,
    *,
    width: int | None = None,
    n_freq: int | None = None,
    max_relative_distance: float,
    min_keep: int = 1,
) -> list[Matching]:
    """Keep matchings whose patch is close enough to ``template``.

    A matching is kept when
    :func:`patch_template_relative_distance` ``<= max_relative_distance``.
    If fewer than ``min_keep`` survive, the closest ``min_keep`` patches are
    retained so a strict threshold cannot starve a rebuild.

    ``max_relative_distance`` is on ``[0, 1]``; ``1`` keeps all matchings. The
    gate is skipped when ``template`` has zero energy (nothing to measure
    against, e.g. the zero placeholders in ``match_to_dictionary``).
    """
    tau = float(max_relative_distance)
    if not (0.0 <= tau <= 1.0):
        raise ValueError(f"max_relative_distance must be in [0, 1], got {tau}")
    if min_keep < 1:
        raise ValueError(f"min_keep must be >= 1, got {min_keep}")
    items = list(matchings)
    if not items or tau >= 1.0:
        return items
    if template_energy(template) <= 0.0:
        return items

    if isinstance(template, Template):
        w = int(width) if width is not None else int(template.width)
        freq = int(n_freq) if n_freq is not None else int(template.n_freq)
        ref = template
    else:
        arr = np.asarray(template, dtype=np.float64)
        if arr.ndim == 2:
            freq, w = (int(n_freq) if n_freq is not None else int(arr.shape[0])), (
                int(width) if width is not None else int(arr.shape[1])
            )
        elif arr.ndim == 3:
            freq, w = (int(n_freq) if n_freq is not None else int(arr.shape[1])), (
                int(width) if width is not None else int(arr.shape[2])
            )
        else:
            raise ValueError(f"template must be (F,W) or (k,F,W), got {arr.shape}")
        ref = arr

    stack = as_stack(signal)
    scored: list[tuple[float, Matching]] = []
    for matching in items:
        arr = stack.get(matching.window_id)
        if arr is None:
            continue
        patch = np.asarray(
            arr[:freq, int(matching.t_start) : int(matching.t_start) + w],
            dtype=np.float64,
        )
        if patch.shape != (freq, w):
            # Skip incomplete edge crops rather than inventing padding here.
            continue
        r = patch_template_relative_distance(patch, ref)
        scored.append((r, matching))

    if not scored:
        return items[: max(min_keep, 1)] if items else []

    kept = [m for r, m in scored if r <= tau]
    if len(kept) >= min_keep:
        return kept
    # Too few after the cut: keep the closest min_keep.
    order = np.argsort(np.asarray([r for r, _ in scored], dtype=np.float64))
    n_take = min(int(min_keep), len(scored))
    return [scored[int(i)][1] for i in order[:n_take]]

