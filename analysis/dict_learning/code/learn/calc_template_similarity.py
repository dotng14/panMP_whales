"""Template similarity from response correlation on a fixed matching.

Templates are unit-``L2`` rank-1 atoms. For a patch ``P`` and template ``T``:

    alpha(T, P) = <T, P>

Given a **caller-supplied** matching (patch locations only — no matching pursuit
here), form the response of every atom on every extracted patch and set
similarity to the Pearson correlation of those response vectors. High
correlation means the atoms "fire" together on the same patches.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..core.objects import Dictionary, Matching, Template, Window
from ..core.utils import as_stack
from ..tools import _extract_patches


@dataclass
class TemplateSimilarityResult:
    """Pairwise Pearson correlations of unit-L2 template–patch responses."""

    template_ids: list[int]
    similarity: np.ndarray  # (n, n); S[i,j] = corr(alpha_i, alpha_j)
    n_patches: int
    alphas: np.ndarray  # (n, M) raw responses
    params: dict[str, Any] = field(default_factory=dict)


def _center_pad_rank1(data: np.ndarray, width: int) -> np.ndarray:
    """Center-align a ``(F, W)`` atom into ``(F, width)`` (crop or zero-pad)."""
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"expected rank-1 (F, W), got shape {arr.shape}")
    freq, w = int(arr.shape[0]), int(arr.shape[1])
    if w == width:
        return np.array(arr, dtype=np.float64, copy=True)
    canvas = np.zeros((freq, width), dtype=np.float64)
    if w < width:
        left = (width - w) // 2
        canvas[:, left : left + w] = arr
    else:
        left = (w - width) // 2
        canvas[:, :] = arr[:, left : left + width]
    return canvas


def _unit_templates(
    dictionary: Dictionary, width: int
) -> tuple[list[int], np.ndarray]:
    """Return ``(template_ids, stack)`` with shape ``(n, F, W)``, unit Frobenius norm."""
    templates = list(dictionary)
    if not templates:
        raise ValueError("dictionary must contain at least one atom")

    ids: list[int] = []
    padded: list[np.ndarray] = []
    n_freq: int | None = None
    for template in templates:
        if int(template.n_components) != 1:
            raise ValueError(
                "calc_template_similarity requires rank-1 atoms; "
                f"template_id={template.template_id} has n_components="
                f"{template.n_components}"
            )
        data = _center_pad_rank1(template.as_rank1, width)
        if n_freq is None:
            n_freq = int(data.shape[0])
        elif int(data.shape[0]) != n_freq:
            raise ValueError(
                f"all atoms must share n_freq={n_freq}, "
                f"template_id={template.template_id} has {data.shape[0]}"
            )
        norm = float(np.linalg.norm(data))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError(
                f"template_id={template.template_id} has zero energy after padding"
            )
        ids.append(int(template.template_id))
        padded.append(data / norm)
    return ids, np.stack(padded, axis=0)


def calc_template_similarity(
    stack: Sequence[Window] | dict[str, np.ndarray],
    dictionary: Dictionary,
    matchings: Sequence[Matching],
) -> TemplateSimilarityResult:
    """Pearson correlation of unit-L2 template responses on a fixed matching.

    Patches are taken only from ``matchings`` (window_id, t_start). Matching
    ``template_id`` / amplitude are ignored for scoring: every atom is dotted
    with every extracted patch. Does **not** run matching pursuit.
    """
    if not dictionary or len(dictionary) == 0:
        raise ValueError("dictionary must contain at least one atom")
    if not matchings:
        raise ValueError("matchings must be non-empty")

    originals = as_stack(stack)
    if not originals:
        raise ValueError("stack is empty")

    templates = list(dictionary)
    common_width = max(int(t.width) for t in templates)
    if common_width < 1:
        raise ValueError(f"common_width must be >= 1, got {common_width}")

    template_ids, unit = _unit_templates(dictionary, common_width)
    n_freq = int(unit.shape[1])
    patches = _extract_patches(
        originals, matchings, width=common_width, n_freq=n_freq
    )
    # Drop pure-empty crops so constant-zero locations do not inflate M.
    usable = [p for p in patches if float(np.sum(p * p)) > 0.0]
    if len(usable) < 2:
        raise ValueError(
            f"need at least 2 non-empty patches from matchings, got {len(usable)}"
        )

    n = len(template_ids)
    m = len(usable)
    # A[i, m] = <T_i, P_m>
    patch_stack = np.stack(usable, axis=0).reshape(m, -1)  # (M, F*W)
    templates_flat = unit.reshape(n, -1)  # (n, F*W)
    alphas = templates_flat @ patch_stack.T  # (n, M)

    # corrcoef treats rows as variables when rowvar=True (default).
    with np.errstate(invalid="ignore", divide="ignore"):
        similarity = np.corrcoef(alphas)
    if similarity.ndim == 0:
        # Single variable edge case (should not happen with n >= 1 and M >= 2)
        similarity = np.array([[1.0]], dtype=np.float64)
    else:
        similarity = np.asarray(similarity, dtype=np.float64)

    # corrcoef leaves nan row/col when a response is constant; force self=1 when
    # variance is positive, leave nan on dead rows.
    row_std = np.std(alphas, axis=1)
    for i in range(n):
        if row_std[i] > 0.0 and np.isfinite(similarity[i, i]):
            similarity[i, i] = 1.0
        elif row_std[i] <= 0.0:
            similarity[i, :] = np.nan
            similarity[:, i] = np.nan

    return TemplateSimilarityResult(
        template_ids=template_ids,
        similarity=similarity,
        n_patches=m,
        alphas=alphas,
        params={
            "common_width": common_width,
            "n_freq": n_freq,
            "n_atoms": n,
            "n_matchings_input": len(matchings),
            "n_patches_extracted": len(patches),
        },
    )


__all__ = ["TemplateSimilarityResult", "calc_template_similarity"]
