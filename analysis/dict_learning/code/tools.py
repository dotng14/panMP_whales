"""Elementary dictionary-learning tools with optional catalog persistence.

Tools
-----
A  :func:`matching_pursuit` — Mallat MP on a dictionary + residual
B  :func:`average_templates` / :func:`average_patches` — pure 1–1 averaging
   (:func:`pca_templates` / :func:`pca_patches` — leading-PC variant)
C  :func:`build_dictionary_from_detections` — greedy GPL single-atom growth
D  :func:`remove_template` — membership edit (append-only new dictionary)
E  :func:`merge_templates` — average detections of two atoms into one
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from database import catalog

from .io.io_gpl import Detection, load_gpl_inputs
from .mp.mp_step import (
    IntervalMap,
    MpLoopResult,
    copy_residual_stack,
    residual_energy,
    run_mp_loop,
    single_template_dictionary,
)
from .core.objects import Dictionary, Matching, Segment, Template, Window
from .learn.seed_from_gpl import Seed, detections_to_segments, mean_center


@dataclass
class ToolResult:
    """Common optional catalog ids attached to tool outputs."""

    tool_run_id: int | None = None
    output_dictionary_id: int | None = None
    input_dictionary_id: int | None = None


@dataclass
class MatchingPursuitResult(ToolResult):
    loop: MpLoopResult = field(default_factory=lambda: MpLoopResult({}, [], [], ""))
    dictionary: Dictionary = field(default_factory=Dictionary)


@dataclass
class AverageTemplatesResult(ToolResult):
    dictionary: Dictionary = field(default_factory=Dictionary)
    parent_template_ids: list[int] = field(default_factory=list)
    new_template_pks: list[int] = field(default_factory=list)


@dataclass
class EditDictionaryResult(ToolResult):
    dictionary: Dictionary = field(default_factory=Dictionary)
    removed_template_ids: list[int] = field(default_factory=list)
    merged_template_id: int | None = None
    merged_template_pk: int | None = None


from .core.utils import as_stack as _as_stack
from .core.utils import copy_dictionary as _copy_dictionary
from .core.utils import filter_matchings_by_relative_distance


def _matching_rows(matchings: Sequence[Matching]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for m in matchings:
        row: dict[str, Any] = {
            "template_id": m.template_id,
            "window_id": m.window_id,
            "t_start": m.t_start,
            "amplitude": float(m.amplitude),
        }
        coeffs = np.asarray(m.coeffs, dtype=np.float64).reshape(-1)
        if coeffs.size > 1:
            row["coeffs"] = coeffs.tolist()
        rows.append(row)
    return rows


def _save_template_npy(
    out_dir: Path,
    template: Template,
    *,
    prefix: str = "template",
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{prefix}_{template.template_id:03d}.npy"
    # Persist the stored array (F,W or k,F,W) as-is.
    np.save(path, template.data)
    return path


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
        raise ValueError(f"width must be >= 1, got {width}")
    patches: list[np.ndarray] = []
    for matching in matchings:
        arr = stack.get(matching.window_id)
        if arr is None:
            continue
        freq = int(n_freq) if n_freq is not None else int(arr.shape[0])
        raw = arr[:freq, matching.t_start : matching.t_start + width]
        if raw.shape[1] == width and raw.shape[0] == freq:
            patches.append(np.asarray(raw, dtype=np.float64))
            continue
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
    if reference_template is None:
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


def average_patches(
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
        raise ValueError(f"n_components must be >= 1, got {n_components}")
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
        raise ValueError("no matched patches could be extracted from the signal")
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
        if lam <= 0.0:
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
    if method == "average":
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
    raise ValueError(
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
    algorithm: str = "greedy",
    lambda_fraction: float = 0.2,
    conn: sqlite3.Connection | None = None,
    input_dictionary_id: int | None = None,
    out_dir: Path | str | None = None,
    parent_tool_run_id: int | None = None,
    learning_run_id: int | None = None,
    segmentation_id: int | None = None,
    name: str | None = None,
    params: dict[str, Any] | None = None,
) -> MatchingPursuitResult:
    """A: Mallat matching pursuit. Optionally persist matchings on a dict snapshot.

    ``max_relative_distance`` rejects placements that leave more than that
    fraction of the patch energy unexplained by the atom.

    ``algorithm='dp'`` runs the exact non-overlapping solver instead of the
    greedy loop, with ``lambda_fraction`` controlling sparsity.
    """
    resid = _as_stack(residual)
    e0 = residual_energy(resid)
    loop = run_mp_loop(
        resid,
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
        algorithm=algorithm,
        lambda_fraction=lambda_fraction,
    )
    result = MatchingPursuitResult(
        loop=loop,
        dictionary=_copy_dictionary(dictionary),
        input_dictionary_id=input_dictionary_id,
    )
    if conn is not None:
        if input_dictionary_id is None:
            raise ValueError("input_dictionary_id required when conn is set")
        tool_params = {
            "eps_step_fraction": eps_step_fraction,
            "max_repeats": max_repeats,
            "normalize": normalize,
            "disallow_overlap": disallow_overlap,
            "mp_mode": mp_mode,
            "n_workers": n_workers,
            "max_relative_distance": max_relative_distance,
            "algorithm": algorithm,
            "lambda_fraction": lambda_fraction,
            **(params or {}),
        }
        window_ids = sorted({m.window_id for m in loop.matchings if m.window_id})
        tool_run_id, output_dictionary_id = catalog.persist_matching_pursuit_result(
            conn,
            input_dictionary_id=input_dictionary_id,
            matchings=_matching_rows(loop.matchings),
            window_ids=window_ids,
            name=name,
            learning_run_id=learning_run_id,
            segmentation_id=segmentation_id,
            params=tool_params,
            energy_before=e0,
            energy_after=loop.energies[-1] if loop.energies else e0,
            stop_reason=loop.stopped_reason,
            out_dir=out_dir,
            parent_tool_run_id=parent_tool_run_id,
        )
        result.tool_run_id = tool_run_id
        result.output_dictionary_id = output_dictionary_id
    return result


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
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    patches = [m for m in matchings if m.template_id == template.template_id]
    if not patches:
        # Callers may pass already-filtered matchings without template_id set
        # consistently; fall back to all provided matchings.
        patches = list(matchings)
    if not patches:
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


def average_templates(
    signal: Sequence[Window] | dict[str, np.ndarray],
    dictionary: Dictionary,
    matchings: Sequence[Matching],
    *,
    method: str = "pca",
    n_components: int = 1,
    max_relative_distance: float | None = None,
    conn: sqlite3.Connection | None = None,
    input_dictionary_id: int | None = None,
    input_template_pks: Sequence[int] | None = None,
    out_dir: Path | str | None = None,
    parent_tool_run_id: int | None = None,
    learning_run_id: int | None = None,
    segmentation_id: int | None = None,
    name: str | None = None,
    params: dict[str, Any] | None = None,
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
    if conn is not None and n_components > 1:
        raise ValueError(
            "catalog persistence for average_templates requires n_components=1"
        )
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

    result = AverageTemplatesResult(
        dictionary=averaged,
        parent_template_ids=parent_ids,
        input_dictionary_id=input_dictionary_id,
    )
    if conn is not None:
        if input_dictionary_id is None:
            raise ValueError("input_dictionary_id required when conn is set")
        parent_pks = list(input_template_pks or [])
        if not parent_pks:
            parent_pks = catalog._dictionary_template_pks(conn, input_dictionary_id)
        if len(parent_pks) != len(averaged):
            raise ValueError(
                "input_template_pks length must match dictionary membership"
            )
        stamp = Path(out_dir) if out_dir is not None else None
        prefix = "pca" if method == "pca" or n_components > 1 else "avg"
        template_rows: list[dict[str, Any]] = []
        for template in averaged:
            if stamp is not None:
                npy_path = _save_template_npy(stamp, template, prefix=prefix)
            else:
                raise ValueError("out_dir required when persisting average_templates")
            template_rows.append(
                {
                    "template_id": template.template_id,
                    "npy_path": npy_path,
                    "n_freq": template.n_freq,
                    "width": template.width,
                    "source_segment_id": template.source_segment_id,
                }
            )
        tool_run_id, output_dictionary_id, new_pks = (
            catalog.persist_average_templates_result(
                conn,
                input_dictionary_id=input_dictionary_id,
                templates=template_rows,
                parent_template_pks=parent_pks,
                name=name,
                learning_run_id=learning_run_id,
                segmentation_id=segmentation_id,
                params=params,
                out_dir=out_dir,
                parent_tool_run_id=parent_tool_run_id,
            )
        )
        result.tool_run_id = tool_run_id
        result.output_dictionary_id = output_dictionary_id
        result.new_template_pks = new_pks
    return result


def pca_templates(
    signal: Sequence[Window] | dict[str, np.ndarray],
    dictionary: Dictionary,
    matchings: Sequence[Matching],
    **kwargs: Any,
) -> AverageTemplatesResult:
    """B (PCA variant): rebuild each atom from its patches' leading PC.

    Thin wrapper over :func:`average_templates` with ``method="pca"``; accepts
    the same keyword arguments (including catalog persistence).
    """
    if "method" in kwargs:
        raise TypeError("pca_templates does not accept a 'method' argument")
    return average_templates(signal, dictionary, matchings, method="pca", **kwargs)


from .core.utils import crop_candidates_from_residual as _crop_candidates_from_residual


def build_dictionary_from_detections(
    signal: Sequence[Window] | dict[str, np.ndarray],
    detections: Sequence[Detection] | Sequence[Segment],
    *,
    windows_for_units: Sequence[Window] | None = None,
    eps_step_fraction: float = 0.5,
    min_repeats: int = 1,
    n_rounds: int | None = None,
    max_templates: int | None = None,
    min_seed_delta_fraction: float = 0.0,
    normalize: bool = True,
    disallow_overlap: bool = False,
    show_progress: bool | None = None,
    mp_mode: str | None = None,
    n_workers: int | None = None,
    max_relative_distance: float | None = None,
    conn: sqlite3.Connection | None = None,
    segmentation_id: int | None = None,
    out_dir: Path | str | None = None,
    learning_run_id: int | None = None,
    parent_tool_run_id: int | None = None,
    name: str | None = None,
    params: dict[str, Any] | None = None,
) -> Seed | Any:
    """C: Greedy GPL seeding via single-atom MP (uses A for each peel).

    When ``conn`` is set, persists each accepted round as a seed/refinement
    dictionary with matchings. Returns a :class:`~.learn_dictionary.LearnedDictionary`
    (imported lazily to avoid cycles).
    """
    from .learn.learn_dictionary import LearnedDictionary
    from .mp.progress import range_progress

    rounds_cap = n_rounds if n_rounds is not None else max_templates
    if rounds_cap is not None and rounds_cap < 1:
        raise ValueError(f"n_rounds/max_templates must be >= 1, got {rounds_cap}")

    if isinstance(signal, dict):
        residual = {k: np.asarray(v, dtype=np.float64) for k, v in signal.items()}
        unit_windows = list(windows_for_units or [])
    else:
        residual = {w.window_id: mean_center(w.data) for w in signal}
        unit_windows = (
            list(windows_for_units) if windows_for_units is not None else list(signal)
        )

    energy_before = residual_energy(residual)
    if detections and isinstance(detections[0], Segment):
        base_segments = list(detections)  # type: ignore[arg-type]
    elif detections:
        if not unit_windows:
            raise ValueError(
                "windows_for_units (or Window stack) required with Detection inputs"
            )
        base_segments = detections_to_segments(unit_windows, detections)  # type: ignore[arg-type]
    else:
        base_segments = []

    seeds: list[Seed] = []
    first_delta: float | None = None
    parent_dictionary_id: int | None = None
    stamp = Path(out_dir) if out_dir is not None else None
    root_tool_run_id: int | None = None
    if conn is not None:
        root_tool_run_id = catalog.insert_tool_run(
            conn,
            tool="build_dictionary_from_detections",
            parent_tool_run_id=parent_tool_run_id,
            segmentation_id=segmentation_id,
            learning_run_id=learning_run_id,
            params={
                "eps_step_fraction": eps_step_fraction,
                "min_repeats": min_repeats,
                "n_rounds": rounds_cap,
                "min_seed_delta_fraction": min_seed_delta_fraction,
                **(params or {}),
            },
            energy_before=energy_before,
            out_dir=out_dir,
        )

    from .mp.mp_parallel import allocate_nested_workers, map_parallel, resolve_mp_mode, resolve_mp_workers
    from .mp.mp_workers import (
        CandidateTrialPayload,
        init_trial_worker_shared,
        payload_by_segment_id,
        rematerialize_candidate_trial,
        run_candidate_trial,
        slim_payloads_for_process_pool,
        trial_sort_key_energy_then_segment,
    )

    mode = resolve_mp_mode(mp_mode)
    cand_workers = resolve_mp_workers(n_workers)

    stop_reason = "completed"
    for template_id in range_progress(
        rounds_cap, desc="from_det rounds", show_progress=show_progress, unit="round"
    ):
        candidates = _crop_candidates_from_residual(base_segments, residual)
        if not candidates:
            stop_reason = "no_candidates"
            break
        # Trial all candidates by energy (same spirit as Seed.from_gpl).
        ordered = sorted(candidates, key=lambda s: s.energy, reverse=True)
        outer_workers, inner_workers = allocate_nested_workers(
            len(ordered), total_workers=cand_workers
        )
        payloads = [
            CandidateTrialPayload(
                segment_id=seg.segment_id,
                template_id=template_id,
                template_data=np.array(seg.data, copy=True),
                residual=residual,
                eps_step_fraction=eps_step_fraction,
                normalize=normalize,
                disallow_overlap=disallow_overlap,
                occupied_intervals=None,
                mp_mode=mode,
                n_workers=inner_workers,
                max_relative_distance=max_relative_distance,
            )
            for seg in ordered
        ]
        trial_results = map_parallel(
            run_candidate_trial,
            slim_payloads_for_process_pool(payloads)
            if outer_workers > 1 and len(payloads) > 1
            else payloads,
            n_workers=outer_workers,
            prefer_threads=False,
            initializer=init_trial_worker_shared
            if outer_workers > 1 and len(payloads) > 1
            else None,
            initargs=(residual, None)
            if outer_workers > 1 and len(payloads) > 1
            else (),
        )
        best_tr = None
        best_key = None
        for tr in trial_results:
            if tr.n_matchings < min_repeats:
                continue
            key = trial_sort_key_energy_then_segment(tr)
            if best_tr is None or key < best_key:
                best_tr = tr
                best_key = key
        if best_tr is None:
            stop_reason = "below_min_repeats"
            break
        winner = rematerialize_candidate_trial(
            payload_by_segment_id(payloads, best_tr.segment_id)  # type: ignore[arg-type]
        )
        tmpl = Template(
            template_id=winner.template_id,
            data=np.array(winner.template_data, copy=True),
            source_segment_id=winner.segment_id,
        )
        best = Seed(
            template=tmpl,
            matchings=winner.matchings,
            residual_after=winner.residual_stack,
            energy_before=residual_energy(residual),
            energy_after=winner.energy_after,
            n_candidates_tried=1,
            selected_segment_id=winner.segment_id,
            candidates=candidates,
            mp_energies=list(winner.energies),
        )
        if not best.is_selected() or not winner.residual_stack:
            stop_reason = "missing_residual" if not winner.residual_stack else "below_min_repeats"
            break
        delta = best.get_delta_energy_total()
        if delta <= 0.0:
            stop_reason = "no_improvement"
            break
        if first_delta is None:
            first_delta = delta
        elif min_seed_delta_fraction > 0 and delta < min_seed_delta_fraction * first_delta:
            stop_reason = "min_seed_delta_fraction"
            break
        seeds.append(best)
        tmpl = best.get_template()
        assert tmpl is not None
        next_residual = best.get_residual_after()
        if next_residual is None:
            stop_reason = "missing_residual"
            break

        if conn is not None and stamp is not None:
            npy_path = _save_template_npy(stamp / f"round_{template_id:03d}", tmpl)
            _tr, out_dict_id, _pks = catalog.persist_build_from_detections_result(
                conn,
                templates=[
                    {
                        "template_id": tmpl.template_id,
                        "npy_path": npy_path,
                        "n_freq": tmpl.n_freq,
                        "width": tmpl.width,
                        "source_segment_id": tmpl.source_segment_id,
                        "round_index": template_id,
                        "n_matchings": len(best.get_matchings()),
                    }
                ],
                matchings=_matching_rows(best.get_matchings()),
                window_ids=sorted({m.window_id for m in best.get_matchings()}),
                name=(name or "from_detections") + f"#r{template_id:03d}",
                parent_dictionary_id=parent_dictionary_id,
                learning_run_id=learning_run_id,
                segmentation_id=segmentation_id,
                params=params,
                energy_before=best.get_energy_before(),
                energy_after=best.get_energy_after(),
                out_dir=stamp / f"round_{template_id:03d}",
                parent_tool_run_id=root_tool_run_id,
                round_index=template_id,
            )
            parent_dictionary_id = out_dict_id

        residual = next_residual
    else:
        if rounds_cap is not None:
            stop_reason = "max_rounds"

    energy_after = residual_energy(residual)
    if conn is not None and root_tool_run_id is not None:
        catalog.finish_tool_run(
            conn,
            root_tool_run_id,
            output_dictionary_id=parent_dictionary_id,
            energy_before=energy_before,
            energy_after=energy_after,
            stop_reason=stop_reason,
            out_dir=out_dir,
        )

    return LearnedDictionary(
        seeds=seeds,
        energy_before=energy_before,
        energy_after=energy_after,
        residual_after=residual,
        stop_reason=stop_reason,
        tool_run_id=root_tool_run_id,
        output_dictionary_id=parent_dictionary_id,
    )


def remove_template(
    dictionary: Dictionary,
    template_id: int | None = None,
    *,
    template_pk: int | None = None,
    conn: sqlite3.Connection | None = None,
    input_dictionary_id: int | None = None,
    parent_tool_run_id: int | None = None,
    learning_run_id: int | None = None,
    segmentation_id: int | None = None,
    name: str | None = None,
    params: dict[str, Any] | None = None,
) -> EditDictionaryResult:
    """D: New dictionary = input membership minus one atom (meta-dict unchanged)."""
    removed_pk: int | None = template_pk
    removed_tid = template_id
    if conn is not None and input_dictionary_id is not None and removed_pk is None:
        if removed_tid is None:
            raise ValueError("template_id or template_pk required")
        for pk in catalog._dictionary_template_pks(conn, input_dictionary_id):
            row = catalog.get_template_by_pk(conn, pk)
            if row and int(row["template_id"]) == int(removed_tid):
                removed_pk = pk
                break
        if removed_pk is None:
            raise ValueError(
                f"template_id {removed_tid} not in dictionary {input_dictionary_id}"
            )

    if removed_tid is None and removed_pk is not None and conn is not None:
        row = catalog.get_template_by_pk(conn, removed_pk)
        if row is None:
            raise KeyError(f"template_pk {removed_pk} not found")
        removed_tid = int(row["template_id"])
    if removed_tid is None:
        raise ValueError("template_id or template_pk required")

    new_dict = Dictionary()
    found = False
    for template in dictionary:
        if template.template_id == removed_tid:
            found = True
            continue
        new_dict.add(
            Template(
                template_id=template.template_id,
                data=np.array(template.data, dtype=np.float64, copy=True),
                source_segment_id=template.source_segment_id,
            )
        )
    if not found and conn is None:
        raise ValueError(f"template_id {removed_tid} not in dictionary")

    result = EditDictionaryResult(
        dictionary=new_dict,
        removed_template_ids=[removed_tid],
        input_dictionary_id=input_dictionary_id,
    )
    if conn is not None:
        if input_dictionary_id is None or removed_pk is None:
            raise ValueError(
                "input_dictionary_id and resolvable template_pk required when conn set"
            )
        tool_run_id, output_dictionary_id = catalog.persist_remove_template_result(
            conn,
            input_dictionary_id=input_dictionary_id,
            removed_template_pk=removed_pk,
            name=name,
            learning_run_id=learning_run_id,
            segmentation_id=segmentation_id,
            params=params,
            parent_tool_run_id=parent_tool_run_id,
        )
        result.tool_run_id = tool_run_id
        result.output_dictionary_id = output_dictionary_id
    return result


def merge_templates(
    signal: Sequence[Window] | dict[str, np.ndarray],
    dictionary: Dictionary,
    matchings: Sequence[Matching],
    template_a: int,
    template_b: int,
    *,
    conn: sqlite3.Connection | None = None,
    input_dictionary_id: int | None = None,
    out_dir: Path | str | None = None,
    parent_tool_run_id: int | None = None,
    learning_run_id: int | None = None,
    segmentation_id: int | None = None,
    name: str | None = None,
    params: dict[str, Any] | None = None,
    new_template_id: int | None = None,
) -> EditDictionaryResult:
    """E: Merge two atoms by averaging all of their detections into one new atom."""
    if template_a == template_b:
        raise ValueError("template_a and template_b must differ")
    by_id = {t.template_id: t for t in dictionary}
    if template_a not in by_id or template_b not in by_id:
        raise ValueError("both templates must be members of dictionary")
    ta, tb = by_id[template_a], by_id[template_b]
    pooled = [
        m for m in matchings if m.template_id in (template_a, template_b)
    ]
    if not pooled:
        raise ValueError("no matchings for either template to merge")
    out_width = max(ta.width, tb.width)
    n_freq = ta.n_freq if ta.n_freq == tb.n_freq else max(ta.n_freq, tb.n_freq)
    avg = average_patches(signal, pooled, width=out_width, n_freq=n_freq)
    merged_id = (
        new_template_id
        if new_template_id is not None
        else max(dictionary.order, default=-1) + 1
    )
    merged = Template(
        template_id=merged_id,
        data=avg,
        source_segment_id=f"merge:{template_a}+{template_b}",
    )
    new_dict = Dictionary()
    for template in dictionary:
        if template.template_id in (template_a, template_b):
            continue
        new_dict.add(
            Template(
                template_id=template.template_id,
                data=np.array(template.data, dtype=np.float64, copy=True),
                source_segment_id=template.source_segment_id,
            )
        )
    new_dict.add(merged)

    result = EditDictionaryResult(
        dictionary=new_dict,
        removed_template_ids=[template_a, template_b],
        merged_template_id=merged_id,
        input_dictionary_id=input_dictionary_id,
    )
    if conn is not None:
        if input_dictionary_id is None:
            raise ValueError("input_dictionary_id required when conn is set")
        if out_dir is None:
            raise ValueError("out_dir required when persisting merge_templates")
        stamp = Path(out_dir)
        npy_path = _save_template_npy(stamp, merged, prefix="merged")
        # Resolve parent pks from input membership order.
        parent_pks: list[int] = []
        remaining: list[int] = []
        for pk in catalog._dictionary_template_pks(conn, input_dictionary_id):
            row = catalog.get_template_by_pk(conn, pk)
            if row is None:
                continue
            tid = int(row["template_id"])
            if tid in (template_a, template_b):
                parent_pks.append(pk)
            else:
                remaining.append(pk)
        if len(parent_pks) != 2:
            raise ValueError(
                f"expected both templates in dictionary membership, found {parent_pks}"
            )
        # Stable parent order matching template_a, template_b.
        ordered_parents: list[int] = []
        for want in (template_a, template_b):
            for pk in parent_pks:
                row = catalog.get_template_by_pk(conn, pk)
                if row and int(row["template_id"]) == want:
                    ordered_parents.append(pk)
                    break
        tool_run_id, output_dictionary_id, merged_pk = (
            catalog.persist_merge_templates_result(
                conn,
                input_dictionary_id=input_dictionary_id,
                merged_template={
                    "template_id": merged_id,
                    "npy_path": npy_path,
                    "n_freq": merged.n_freq,
                    "width": merged.width,
                    "source_segment_id": merged.source_segment_id,
                    "n_matchings": len(pooled),
                },
                parent_template_pks=ordered_parents,
                remaining_template_pks=remaining,
                name=name,
                learning_run_id=learning_run_id,
                segmentation_id=segmentation_id,
                params={
                    "template_a": template_a,
                    "template_b": template_b,
                    **(params or {}),
                },
                out_dir=out_dir,
                parent_tool_run_id=parent_tool_run_id,
            )
        )
        result.tool_run_id = tool_run_id
        result.output_dictionary_id = output_dictionary_id
        result.merged_template_pk = merged_pk
    return result
