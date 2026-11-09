"""Shifted cosine between two atom shapes."""

from __future__ import annotations

import numpy as np

# Two shapes overlap by at least half the narrower width when compared.
MIN_SHAPE_OVERLAP = 0.5


def best_shift(a: np.ndarray, b: np.ndarray, min_overlap: float = MIN_SHAPE_OVERLAP) -> tuple[float, int]:
    """Best cosine between two ``(F, W)`` shapes over time shifts, and its shift.

    Every shift where the two overlap by at least ``min_overlap`` of the
    narrower width is tried; the cosine is taken over the overlapping columns
    only, so a call matches a longer or shifted cut of itself. At the returned
    ``shift``, column ``c`` of ``a`` lines up with column ``c + shift`` of
    ``b``: a copy of ``b`` starting at ``t`` holds ``a`` starting at
    ``t + shift``.
    """
    wa, wb = a.shape[1], b.shape[1]
    need = max(1, int(np.ceil(min_overlap * min(wa, wb))))
    best, best_at = 0.0, 0
    for shift in range(-(wa - need), wb - need + 1):
        lo, hi = max(0, shift), min(wb, shift + wa)
        part_b = b[:, lo:hi]
        part_a = a[:, lo - shift : hi - shift]
        denom = float(np.linalg.norm(part_a) * np.linalg.norm(part_b))
        if denom > 0.0:  # no-pipeline-branch: seeds and refined atoms have energy at every shift
            cosine = float(np.sum(part_a * part_b)) / denom
            if cosine > best:
                best, best_at = cosine, shift
    return best, best_at


def shape_similarity(a: np.ndarray, b: np.ndarray, min_overlap: float = MIN_SHAPE_OVERLAP) -> float:
    """Best shifted cosine of :func:`best_shift`."""
    return best_shift(a, b, min_overlap)[0]
