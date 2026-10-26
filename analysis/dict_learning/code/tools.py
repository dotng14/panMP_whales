"""Elementary dictionary-learning tools.

Tools
-----
A  :func:`matching_pursuit` — Mallat MP on a dictionary + residual
B  :func:`average_templates` / :func:`rebuild_template_from_matchings` —
   rebuild atoms from their matched patches (mean, leading PC, or top-k PCA)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .mp import IntervalMap, MpLoopResult, run_mp_loop
from .core import Dictionary, Matching, Template, Window


@dataclass
class MatchingPursuitResult:
    loop: MpLoopResult = field(default_factory=lambda: MpLoopResult({}, [], [], ""))
    dictionary: Dictionary = field(default_factory=Dictionary)


@dataclass
class AverageTemplatesResult:
    dictionary: Dictionary = field(default_factory=Dictionary)
    parent_template_ids: list[int] = field(default_factory=list)


from .core import as_stack as _as_stack
from .core import copy_dictionary as _copy_dictionary
from .core import filter_matchings_by_relative_distance


def _extract_patches(
    signal: Sequence[Window] | dict[str, np.ndarray],
    matchings: Sequence[Matching],
    *,
    width: int,
    n_freq: int | None = None,
) -> list[np.ndarray]:
    """Crop the ``(n_freq, width)`` patch under every matching location.

    When a matching's native crop is narrower/wider than ``width``, the patch is
    center-aligned into a ``(n_freq, width)`` canvas (crop or zero-pad). Shared
    by :func:`average_patches` and :func:`pca_patches`.
    """
    stack = _as_stack(signal)
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")  # no-pipeline-cover: atoms always have a width
    patches: list[np.ndarray] = []
    for matching in matchings:
        arr = stack.get(matching.window_id)
        if arr is None:  # no-pipeline-cover: matchings always point into the stack
            continue
        freq = int(n_freq) if n_freq is not None else int(arr.shape[0])
        raw = arr[:freq, matching.t_start : matching.t_start + width]
        if raw.shape[1] != width or raw.shape[0] != freq:  # no-pipeline-cover: MP places whole atoms
            # Center-align into fixed canvas (supports merge of unequal widths).
            canvas = np.zeros((freq, width), dtype=np.float64)
            src = arr[:freq, matching.t_start :]
            take = min(src.shape[1], width)
            if take <= 0:
                continue
            # Prefer the matching's intended span when available.
            intended = arr[:freq, matching.t_start : matching.t_start + max(width, 1)]
            src_w = intended.shape[1]
            if src_w == 0:
                continue
            if src_w == width:
                canvas[:, :] = intended
            elif src_w < width:
                left = (width - src_w) // 2
                canvas[:, left : left + src_w] = intended
            else:
                left = (src_w - width) // 2
                canvas[:, :] = intended[:, left : left + width]
            patches.append(canvas)
            continue
        patches.append(np.asarray(raw, dtype=np.float64))
    return patches


def _filtered_matchings(
    signal: Sequence[Window] | dict[str, np.ndarray],
    matchings: Sequence[Matching],
    *,
    width: int,
    n_freq: int | None,
    reference_template: Template | np.ndarray | None,
    max_relative_distance: float | None,
    min_keep: int,
) -> list[Matching]:
    """Apply the relative-distance gate when it is both requested and possible."""
    used = list(matchings)
    if max_relative_distance is None:
        return used
    if reference_template is None:  # no-pipeline-cover: rebuilds always pass the atom
        raise ValueError(
            "max_relative_distance requires reference_template "
            "(the template used to score patch distances)"
        )
    return filter_matchings_by_relative_distance(
        signal,
        used,
        reference_template,
        width=width,
        n_freq=n_freq,
        max_relative_distance=float(max_relative_distance),
        min_keep=max(1, min(int(min_keep), len(used) or 1)),
    )


def average_patches(  # no-pipeline-cover: averaging rebuild; the randomized build uses PCA
    signal: Sequence[Window] | dict[str, np.ndarray],
    matchings: Sequence[Matching],
    *,
    width: int,
    n_freq: int | None = None,
    reference_template: Template | np.ndarray | None = None,
    max_relative_distance: float | None = None,
) -> np.ndarray:
    """Average patches at matching locations into one template array.

    When a matching's native crop is narrower/wider than ``width``, the patch is
    center-aligned into a ``(n_freq, width)`` canvas (crop or zero-pad).

    When ``max_relative_distance`` is set together with ``reference_template``,
    only patches close enough to that template are averaged (see
    :func:`filter_matchings_by_relative_distance`).
    """
    used = _filtered_matchings(
        signal,
        matchings,
        width=width,
        n_freq=n_freq,
        reference_template=reference_template,
        max_relative_distance=max_relative_distance,
        min_keep=1,
    )
    patches = _extract_patches(signal, used, width=width, n_freq=n_freq)
    if not patches:
        raise ValueError("no matched patches could be extracted from the signal")
    return np.mean(np.stack(patches, axis=0), axis=0)


def pca_basis(
    signal: Sequence[Window] | dict[str, np.ndarray],
    matchings: Sequence[Matching],
    *,
    width: int,
    n_freq: int | None = None,
    n_components: int = 1,
    reference_template: Template | np.ndarray | None = None,
    max_relative_distance: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-``k`` orthonormal PCA basis of the matched patches.

    Returns ``(basis, variances)`` where ``basis`` has shape
    ``(k, n_freq, width)`` — the leading ``k`` left singular vectors (top
    eigenvectors of the patch scatter matrix ``sum_i P_i P_i^T``) reshaped to
    patch geometry — and ``variances`` is length ``k`` with ``lambda_m / n``
    (per-patch energy along each component), in descending order.

    Each basis vector is unit norm and sign-oriented to correlate positively
    with the mean patch (deterministic). ``k`` is clamped to the number of
    available patches. Falls back to the (normalized) mean direction when the
    patches carry no variance.

    When ``max_relative_distance`` is set (in ``[0, 1]``) and
    ``reference_template`` is provided, only patches leaving at most that
    fraction of their energy unexplained by the template enter the scatter
    matrix (see :func:`filter_matchings_by_relative_distance`).
    """
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")  # no-pipeline-cover: the build checks n_components first
    used = _filtered_matchings(
        signal,
        matchings,
        width=width,
        n_freq=n_freq,
        reference_template=reference_template,
        max_relative_distance=max_relative_distance,
        min_keep=int(n_components),
    )
    patches = _extract_patches(signal, used, width=width, n_freq=n_freq)
    if not patches:
        raise ValueError("no matched patches could be extracted from the signal")  # no-pipeline-cover: atoms are only rebuilt from their placements
    freq, w = patches[0].shape
    data = np.stack([p.reshape(-1) for p in patches], axis=0)  # (n, D)
    n = data.shape[0]
    mean_vec = np.mean(data, axis=0)
    k = min(int(n_components), n)

    def _oriented(vec: np.ndarray) -> np.ndarray:
        return -vec if float(np.dot(vec, mean_vec)) < 0.0 else vec

    if n == 1:
        vec = data[0]
        norm = float(np.linalg.norm(vec))
        unit = vec / norm if norm > 0.0 else vec
        basis = np.zeros((k, freq, w), dtype=np.float64)
        basis[0] = _oriented(unit).reshape(freq, w)
        variances = np.zeros(k, dtype=np.float64)
        variances[0] = norm * norm
        return basis, variances

    # Small n x n Gram matrix: D (= n_freq*width) is typically far larger than
    # the patch count n, so eigh on X X^T is faster and lighter than full SVD.
    gram = data @ data.T  # (n, n)
    eigvals, eigvecs = np.linalg.eigh(gram)  # ascending
    order = np.argsort(eigvals)[::-1][:k]  # top-k, descending

    basis = np.zeros((k, freq, w), dtype=np.float64)
    variances = np.zeros(k, dtype=np.float64)
    for i, idx in enumerate(order):
        lam = float(eigvals[idx])
        if lam <= 0.0:  # no-pipeline-cover: degenerate patch sets with no variance
            # No variance left along this / further components: pad with the
            # (normalized) mean direction for slot 0, zeros otherwise.
            if i == 0:
                norm = float(np.linalg.norm(mean_vec))
                if norm > 0.0:
                    basis[i] = (mean_vec / norm).reshape(freq, w)
            break
        s = float(np.sqrt(lam))
        u = eigvecs[:, idx]
        v = (data.T @ u) / s  # unit-norm right singular vector (length D)
        basis[i] = _oriented(v).reshape(freq, w)
        variances[i] = lam / n
    return basis, variances


def pca_patches(
    signal: Sequence[Window] | dict[str, np.ndarray],
    matchings: Sequence[Matching],
    *,
    width: int,
    n_freq: int | None = None,
    reference_template: Template | np.ndarray | None = None,
    max_relative_distance: float | None = None,
) -> np.ndarray:
    """Leading principal component of matched patches (max energy-reduction dir).

    Unlike :func:`average_patches` (the mean, which cancels when patches are
    misaligned in phase), this returns the first left singular vector of the
    stacked patch matrix — i.e. the top eigenvector of the patch scatter matrix
    ``sum_i P_i P_i^T``. With per-location optimal amplitudes that direction
    maximizes the energy removed by a single rank-1 template (Eckart–Young).

    The vector is oriented to correlate positively with the mean patch (a
    deterministic sign convention) and scaled to the RMS projection
    ``sigma_1 / sqrt(n)`` so it carries a data-like magnitude. Under MP template
    normalization the scale is immaterial; only the direction drives matching.
    """
    basis, variances = pca_basis(
        signal,
        matchings,
        width=width,
        n_freq=n_freq,
        n_components=1,
        reference_template=reference_template,
        max_relative_distance=max_relative_distance,
    )
    # sigma_1 / sqrt(n) == sqrt(variance_1); gives PC1 a data-like magnitude.
    return float(np.sqrt(max(variances[0], 0.0))) * basis[0]


def combine_patches(
    signal: Sequence[Window] | dict[str, np.ndarray],
    matchings: Sequence[Matching],
    *,
    width: int,
    n_freq: int | None = None,
    method: str = "pca",
    reference_template: Template | np.ndarray | None = None,
    max_relative_distance: float | None = None,
) -> np.ndarray:
    """Dispatch to :func:`average_patches` or :func:`pca_patches` by ``method``."""
    if method == "average":  # no-pipeline-cover: the randomized build rebuilds with PCA
        return average_patches(
            signal,
            matchings,
            width=width,
            n_freq=n_freq,
            reference_template=reference_template,
            max_relative_distance=max_relative_distance,
        )
    if method == "pca":
        return pca_patches(
            signal,
            matchings,
            width=width,
            n_freq=n_freq,
            reference_template=reference_template,
            max_relative_distance=max_relative_distance,
        )
    raise ValueError(  # no-pipeline-cover: the randomized build rebuilds with PCA
        f"unknown template method: {method!r} (expected 'average' or 'pca')"
    )

def matching_pursuit(
    residual: Sequence[Window] | dict[str, np.ndarray],
    dictionary: Dictionary,
    *,
    eps_step_fraction: float = 0.5,
    max_repeats: int | None = None,
    normalize: bool = True,
    disallow_overlap: bool = False,
    occupied_intervals: IntervalMap | None = None,
    energy_floor: float | None = None,
    show_progress: bool | None = None,
    progress_desc: str = "mp",
    mp_mode: str | None = None,
    n_workers: int | None = None,
    max_relative_distance: float | None = None,
) -> MatchingPursuitResult:
    """A: Mallat matching pursuit.

    ``max_relative_distance`` rejects placements that leave more than that
    fraction of the patch energy unexplained by the atom.
    """
    loop = run_mp_loop(
        _as_stack(residual),
        dictionary,
        energy_floor=energy_floor,
        eps_step_fraction=eps_step_fraction,
        max_repeats=max_repeats,
        normalize=normalize,
        disallow_overlap=disallow_overlap,
        occupied_intervals=occupied_intervals,
        show_progress=show_progress,
        progress_desc=progress_desc,
        mp_mode=mp_mode,
        n_workers=n_workers,
        max_relative_distance=max_relative_distance,
    )
    return MatchingPursuitResult(loop=loop, dictionary=_copy_dictionary(dictionary))


def rebuild_template_from_matchings(
    signal: Sequence[Window] | dict[str, np.ndarray],
    template: Template,
    matchings: Sequence[Matching],
    *,
    n_components: int = 1,
    method: str = "pca",
    max_relative_distance: float | None = None,
) -> Template:
    """Rebuild one atom from its matched patches.

    ``n_components == 1`` uses :func:`combine_patches` (``average`` or ``pca``).
    ``n_components > 1`` uses :func:`pca_basis` and stores a ``(k, F, W)`` basis.

    When ``max_relative_distance`` is set, patches leaving more than that
    fraction of their energy unexplained by the current ``template`` are
    dropped before the rebuild — for averaging as well as PCA.
    """
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")  # no-pipeline-cover: the build checks n_components first
    patches = [m for m in matchings if m.template_id == template.template_id]
    if not patches:  # no-pipeline-cover: callers pass the atom's own placements
        # Callers may pass already-filtered matchings without template_id set
        # consistently; fall back to all provided matchings.
        patches = list(matchings)
    if not patches:  # no-pipeline-cover: callers pass the atom's own placements
        raise ValueError(
            f"template_id {template.template_id} has no matchings to rebuild"
        )
    # Gate once here so every rebuild path (average, PC1, k-subspace) sees the
    # same surviving patches.
    patches = _filtered_matchings(
        signal,
        patches,
        width=template.width,
        n_freq=template.n_freq,
        reference_template=template,
        max_relative_distance=max_relative_distance,
        min_keep=int(n_components),
    )
    if n_components == 1:
        data = combine_patches(
            signal,
            patches,
            width=template.width,
            n_freq=template.n_freq,
            method=method,
        )
        return Template(
            template_id=template.template_id,
            data=data,
            source_segment_id=template.source_segment_id,
        )
    basis, variances = pca_basis(
        signal,
        patches,
        width=template.width,
        n_freq=template.n_freq,
        n_components=n_components,
    )
    return Template(
        template_id=template.template_id,
        data=basis,
        source_segment_id=template.source_segment_id,
        variances=variances,
    )


def average_templates(  # no-pipeline-cover: used by refine_dictionary_by_averaging (learn_from_motifs)
    signal: Sequence[Window] | dict[str, np.ndarray],
    dictionary: Dictionary,
    matchings: Sequence[Matching],
    *,
    method: str = "pca",
    n_components: int = 1,
    max_relative_distance: float | None = None,
) -> AverageTemplatesResult:
    """B: Rebuild every atom from its matched patches (no re-MP).

    ``method="average"`` (default) uses the per-atom mean patch; ``method="pca"``
    uses the leading principal component (see :func:`pca_patches`). Both preserve
    each atom's ``template_id`` and lineage. When ``n_components > 1``, each atom
    is rebuilt as a top-``k`` PCA subspace via :func:`pca_basis` (``method`` is
    ignored). ``max_relative_distance`` gates which patches contribute, for
    every rebuild method (see :func:`rebuild_template_from_matchings`).
    """
    if len(dictionary) == 0:
        raise ValueError("dictionary must contain at least one atom")
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    by_template: dict[int, list[Matching]] = {}
    for matching in matchings:
        by_template.setdefault(matching.template_id, []).append(matching)

    averaged = Dictionary()
    parent_ids: list[int] = []
    for template in dictionary:
        patches = by_template.get(template.template_id, [])
        if not patches:
            raise ValueError(
                f"template_id {template.template_id} has no matchings to average"
            )
        rebuilt = rebuild_template_from_matchings(
            signal,
            template,
            patches,
            n_components=n_components,
            method=method,
            max_relative_distance=max_relative_distance,
        )
        averaged.add(rebuilt)
        parent_ids.append(template.template_id)

    return AverageTemplatesResult(dictionary=averaged, parent_template_ids=parent_ids)


