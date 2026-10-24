"""Compare atoms by the patches they explain."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..core import EPS_STEP_FRACTION, MAX_RELATIVE_DISTANCE
from ..core import Dictionary, Matching, Template, Window
from ..core import as_stack
from ..mp import (
    IntervalMap,
    copy_residual_stack,
    inflate_matching,
    residual_energy,
)
from ..tools import matching_pursuit


@dataclass
class DivergenceResult:
    """Pairwise asymmetric divergences over dictionary atoms."""

    template_ids: list[int]
    divergences: np.ndarray  # (n, n); D[i, j] = D(A_i, B_j); nan if A insignificant
    significant: np.ndarray  # (n,) bool
    energy_residual: float
    energy_without: np.ndarray  # (n,) E(R[-A])
    delta_energy: np.ndarray  # (n,) E(R[-A]) - E(R)
    joint_matchings: list[Matching]
    min_delta_fraction: float
    params: dict[str, Any] = field(default_factory=dict)


def _copy_template(template: Template) -> Template:
    return Template(
        template_id=int(template.template_id),
        data=np.array(template.data, dtype=np.float64, copy=True),
        source_segment_id=template.source_segment_id,
        variances=(
            None
            if template.variances is None
            else np.array(template.variances, dtype=np.float64, copy=True)
        ),
    )


def _singleton_dictionary(template: Template) -> Dictionary:
    d = Dictionary()
    d.add(_copy_template(template))
    return d


def _inflate_matchings(
    residual: dict[str, np.ndarray],
    matchings: Sequence[Matching],
    template: Template,
) -> None:
    """In-place: add back each matching of ``template`` on ``residual``."""
    basis = np.ascontiguousarray(template.basis, dtype=np.float64)
    tid = int(template.template_id)
    for matching in matchings:
        if int(matching.template_id) != tid:
            continue
        window = residual.get(matching.window_id)
        if window is None:
            continue
        inflate_matching(window, matching, basis)


def calc_divergences(
    stack: Sequence[Window] | dict[str, np.ndarray],
    dictionary: Dictionary,
    *,
    eps_step_fraction: float = EPS_STEP_FRACTION,
    normalize: bool = True,
    disallow_overlap: bool = True,
    max_relative_distance: float | None = MAX_RELATIVE_DISTANCE,
    min_delta_fraction: float = 1e-6,
    show_progress: bool | None = None,
    mp_mode: str | None = None,
    n_workers: int | None = None,
    algorithm: str = "greedy",
    lambda_fraction: float = 0.2,
    respect_occupancy: bool = False,
) -> DivergenceResult:
    """Compute all pairwise atom divergences on a spectrogram stack.

    Cost is one joint multi-atom MP plus ``O(n²)`` single-atom MPs for an
    ``n``-atom dictionary.

    ``algorithm='dp'`` peels with the exact non-overlapping solver.

    ``respect_occupancy=True`` blocks the ``B`` re-peel from spans that atoms
    other than ``A`` still hold in the joint solution, so ``B`` may only claim
    what ``A`` vacated. Without it ``B`` can stack on top of another atom's
    placement and drive the residual below ``E(R)``, which shows up as a
    negative divergence unrelated to how well ``B`` substitutes for ``A``.
    """
    if algorithm not in ("greedy", "dp"):
        raise ValueError(f"algorithm must be 'greedy' or 'dp', got {algorithm!r}")
    if algorithm == "dp":
        disallow_overlap = True
    if float(min_delta_fraction) < 0.0:
        raise ValueError(
            f"min_delta_fraction must be >= 0, got {min_delta_fraction}"
        )
    if not dictionary or len(dictionary) == 0:
        raise ValueError("dictionary must contain at least one atom")

    original = as_stack(stack)
    if not original:
        raise ValueError("stack is empty")
    e_stack = residual_energy(original)

    templates = list(dictionary)
    template_ids = [int(t.template_id) for t in templates]
    n = len(templates)
    by_id = {int(t.template_id): t for t in templates}

    joint = matching_pursuit(
        original,
        dictionary,
        eps_step_fraction=eps_step_fraction,
        normalize=normalize,
        disallow_overlap=disallow_overlap,
        show_progress=show_progress,
        progress_desc="div-joint-mp",
        mp_mode=mp_mode,
        n_workers=n_workers,
        max_relative_distance=max_relative_distance,
        algorithm=algorithm,
        lambda_fraction=lambda_fraction,
    )
    residual_r = copy_residual_stack(joint.loop.residual_stack)
    joint_matchings = list(joint.loop.matchings)
    e_r = residual_energy(residual_r)

    matchings_by_tid: dict[int, list[Matching]] = {tid: [] for tid in template_ids}
    for matching in joint_matchings:
        tid = int(matching.template_id)
        if tid in matchings_by_tid:
            matchings_by_tid[tid].append(matching)

    widths = {int(t.template_id): int(t.width) for t in templates}
    spans_by_tid: dict[int, dict[str, list[tuple[int, int]]]] = {}
    if respect_occupancy:
        for matching in joint_matchings:
            tid = int(matching.template_id)
            width = widths.get(tid)
            if width is None:
                continue
            per_window = spans_by_tid.setdefault(tid, {})
            per_window.setdefault(matching.window_id, []).append(
                (int(matching.t_start), int(matching.t_start) + width)
            )

    divergences = np.full((n, n), np.nan, dtype=np.float64)
    significant = np.zeros(n, dtype=bool)
    energy_without = np.zeros(n, dtype=np.float64)
    delta_energy = np.zeros(n, dtype=np.float64)

    min_delta = float(min_delta_fraction) * float(e_stack)
    # Absolute floor so a near-zero stack (or float noise) never divides.
    min_delta = max(min_delta, 1e-12 * max(float(e_stack), 1.0))

    for i, atom_a in enumerate(templates):
        tid_a = int(atom_a.template_id)
        r_wo = copy_residual_stack(residual_r)
        _inflate_matchings(r_wo, matchings_by_tid[tid_a], atom_a)
        e_wo = residual_energy(r_wo)
        delta = float(e_wo - e_r)
        energy_without[i] = e_wo
        delta_energy[i] = delta

        if delta < min_delta or delta <= 0.0:
            significant[i] = False
            continue
        significant[i] = True

        occupied_other: IntervalMap | None = None
        if respect_occupancy:
            occupied_other = {}
            for tid_other, per_window in spans_by_tid.items():
                if tid_other == tid_a:
                    continue
                for window_id, spans in per_window.items():
                    occupied_other.setdefault(window_id, []).extend(spans)

        for j, atom_b in enumerate(templates):
            r_ab = copy_residual_stack(r_wo)
            mp_b = matching_pursuit(
                r_ab,
                _singleton_dictionary(atom_b),
                eps_step_fraction=eps_step_fraction,
                normalize=normalize,
                disallow_overlap=disallow_overlap,
                show_progress=False,
                progress_desc=f"div-mp B={atom_b.template_id}",
                mp_mode=mp_mode,
                n_workers=n_workers,
                max_relative_distance=max_relative_distance,
                algorithm=algorithm,
                lambda_fraction=lambda_fraction,
                occupied_intervals=occupied_other,
            )
            e_ab = residual_energy(mp_b.loop.residual_stack)
            divergences[i, j] = float((e_ab - e_r) / delta)

    params = {
        "eps_step_fraction": eps_step_fraction,
        "normalize": normalize,
        "disallow_overlap": disallow_overlap,
        "max_relative_distance": max_relative_distance,
        "min_delta_fraction": float(min_delta_fraction),
        "n_atoms": n,
        "n_joint_matchings": len(joint_matchings),
        "energy_stack": float(e_stack),
        "mp_mode": mp_mode,
        "n_workers": n_workers,
        "algorithm": algorithm,
        "lambda_fraction": float(lambda_fraction),
        "respect_occupancy": bool(respect_occupancy),
    }
    return DivergenceResult(
        template_ids=template_ids,
        divergences=divergences,
        significant=significant,
        energy_residual=float(e_r),
        energy_without=energy_without,
        delta_energy=delta_energy,
        joint_matchings=joint_matchings,
        min_delta_fraction=float(min_delta_fraction),
        params=params,
    )


__all__ = ["DivergenceResult", "calc_divergences"]

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..core import Dictionary, Matching, Template, Window
from ..core import as_stack
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

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..core import Dictionary, Matching, Template, Window
from ..core import as_stack
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
