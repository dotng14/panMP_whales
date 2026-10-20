"""Group-mean template similarity on a fixed labeled matching.

Partition patches by ``Matching.template_id``. For the group of patches assigned
to atom A and unit-``L2`` atoms T_B:

    mu(A, B) = mean over P in G_A of <T_B, P>

Similarity (asymmetric):

    S(A, B) = mu(A, B) / mu(A, A)

when the self-response is significant. Near 1 means B scores about as well
as A on the patches assigned to A. No matching pursuit is performed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..core.objects import Dictionary, Matching, Template, Window
from ..core.utils import as_stack
from ..tools import _extract_patches


@dataclass
class TemplateGroupSimilarityResult:
    """Asymmetric group-mean similarity on a labeled matching."""

    template_ids: list[int]
    similarity: np.ndarray  # (n, n) S[A,B] = mu(A,B)/mu(A,A); nan if A insignificant
    mean_alpha: np.ndarray  # (n, n) mu(A,B)
    group_sizes: np.ndarray  # (n,) |G_A|
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
                "calc_template_group_similarity requires rank-1 atoms; "
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


def calc_template_group_similarity(
    stack: Sequence[Window] | dict[str, np.ndarray],
    dictionary: Dictionary,
    matchings: Sequence[Matching],
) -> TemplateGroupSimilarityResult:
    """Asymmetric group-mean similarity from a labeled matching.

    Patches are partitioned by ``Matching.template_id``. Matchings whose
    ``template_id`` is not in the dictionary, or that crop to empty energy, are
    skipped. Does **not** run matching pursuit.
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
    n = len(template_ids)
    id_to_row = {tid: i for i, tid in enumerate(template_ids)}
    templates_flat = unit.reshape(n, -1)  # (n, F*W)

    # Per-group sums of outer products: sum_B alpha(B,P) for P in G_A.
    group_sums = np.zeros((n, n), dtype=np.float64)
    group_sizes = np.zeros(n, dtype=np.int64)
    n_skipped_unknown = 0
    n_skipped_empty = 0

    for matching in matchings:
        tid = int(matching.template_id)
        row = id_to_row.get(tid)
        if row is None:
            n_skipped_unknown += 1
            continue
        patches = _extract_patches(
            originals,
            [matching],
            width=common_width,
            n_freq=n_freq,
        )
        if not patches:
            n_skipped_empty += 1
            continue
        patch = patches[0]
        if float(np.sum(patch * patch)) <= 0.0:
            n_skipped_empty += 1
            continue
        # alpha_j = <T_j, P> for all j
        alphas = templates_flat @ patch.reshape(-1)
        group_sums[row] += alphas
        group_sizes[row] += 1

    if int(np.sum(group_sizes)) == 0:
        raise ValueError(
            "no usable patches after filtering unknown template_ids and empty crops"
        )

    mean_alpha = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        if group_sizes[i] > 0:
            mean_alpha[i] = group_sums[i] / float(group_sizes[i])

    # Scale floor for the self-response; inactive rows stay nan.
    finite_mu = mean_alpha[np.isfinite(mean_alpha)]
    scale = float(np.max(np.abs(finite_mu))) if finite_mu.size else 0.0
    eps = 1e-12 * max(1.0, scale)

    similarity = np.full((n, n), np.nan, dtype=np.float64)
    for i in range(n):
        self_mu = mean_alpha[i, i]
        if not np.isfinite(self_mu) or self_mu <= eps:
            continue
        similarity[i] = mean_alpha[i] / self_mu
        similarity[i, i] = 1.0

    return TemplateGroupSimilarityResult(
        template_ids=template_ids,
        similarity=similarity,
        mean_alpha=mean_alpha,
        group_sizes=group_sizes,
        params={
            "common_width": common_width,
            "n_freq": n_freq,
            "n_atoms": n,
            "n_matchings_input": len(matchings),
            "n_patches_used": int(np.sum(group_sizes)),
            "n_skipped_unknown_template": n_skipped_unknown,
            "n_skipped_empty": n_skipped_empty,
            "eps": eps,
        },
    )


__all__ = ["TemplateGroupSimilarityResult", "calc_template_group_similarity"]
