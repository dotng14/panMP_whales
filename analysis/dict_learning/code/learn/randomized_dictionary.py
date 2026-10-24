"""Randomized dictionary building (CreateDictionary + RefineDictionary).

Implements the architecture in ``documentation/Docs/randomized-dictionary.tex``:

1. ``create_dictionary`` — sample poorly-explained (high relative residual)
   GPL patches and turn each into a unit-norm order-1 atom. A crop stays in
   the pool iff ``‖r‖ / ‖x‖ > epsilon`` (residual vs original span).
2. ``refine_dictionary`` — assign patches to atoms with whole-dictionary MP,
   rebuild each atom by top-``K`` PCA, drop atoms with too few patches
   (default ``iter=3``). After refine finishes, drop placements with
   unexplained energy fraction ``dnorm > 0.5``.
3. ``learn_randomized_outer`` — residual outer loop: each pass samples
   ``m_per_outer`` new seeds, refines them, merges survivors into an
   accumulating dictionary, and stops when time coverage exceeds a target
   (or a max-outer / stall guard fires).

Typical use::

    d0 = create_dictionary(stack, detections, m=10)
    result = refine_dictionary(stack, d0, n_components=1)  # default: 3 iters

    # Coverage-gated residual outer loop:
    outer = learn_randomized_outer(
        windows, detections,
        m_per_outer=5,
        n_min=10,
        post_refine_max_relative_distance=0.9,
        min_coverage=0.5,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from ..core import defaults
from ..core import Dictionary, Matching, Segment, Template, Window
from ..core import (
    as_stack,
    copy_dictionary,
    coverage_from_matchings,
    patch_template_relative_distance,
)
from ..io import Detection
from ..mp import refit_and_deflate, residual_energy
from ..tools import matching_pursuit, rebuild_template_from_matchings
from .seed_from_gpl import detections_to_segments

# Default post-refine gate: drop placements with unexplained energy fraction
# ``dnorm(x, A) > POST_REFINE_MAX_RELATIVE_DISTANCE``.
POST_REFINE_MAX_RELATIVE_DISTANCE = 0.5
DEFAULT_REFINE_ITERS = 3
DEFAULT_OUTER_M = 5
DEFAULT_OUTER_N_MIN = 10
DEFAULT_OUTER_POST_REFINE_TAU = 0.9
DEFAULT_OUTER_MIN_COVERAGE = 0.5
DEFAULT_OUTER_MAX_OUTER = 100
DEFAULT_CREATE_EPSILON = 0.1
DEFAULT_MIN_TEMPLATE_WIDTH: int | None = 10
DEFAULT_MAX_TEMPLATE_WIDTH: int | None = 20


@dataclass
class RandomizedCreateResult:
    """Output of :func:`create_dictionary`."""

    dictionary: Dictionary
    pool_size: int = 0
    n_sampled: int = 0
    residual: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class RandomizedRefineResult:
    """Output of :func:`refine_dictionary`."""

    dictionary: Dictionary
    matchings: list[Matching] = field(default_factory=list)
    residual_after: dict[str, np.ndarray] = field(default_factory=dict)
    energies: list[float] = field(default_factory=list)
    iterations_completed: int = 0
    n_pruned: int = 0
    n_distance_filtered: int = 0


@dataclass
class RandomizedOuterPassLog:
    """Telemetry for one residual outer pass of :func:`learn_randomized_outer`."""

    outer: int
    pool_size: int
    n_sampled: int
    n_atoms_this_pass: int
    n_matchings_this_pass: int
    n_pruned: int
    n_distance_filtered: int
    n_atoms: int
    n_matchings: int
    time_coverage_fraction: float
    window_coverage_fraction: float
    n_bins_covered: int
    n_bins_total: int
    n_windows_matched: int
    n_windows_total: int
    energy_before: float | None = None
    energy_after: float | None = None


@dataclass
class RandomizedOuterResult:
    """Output of :func:`learn_randomized_outer`."""

    dictionary: Dictionary
    matchings: list[Matching] = field(default_factory=list)
    residual_after: dict[str, np.ndarray] = field(default_factory=dict)
    original_stack: dict[str, np.ndarray] = field(default_factory=dict)
    outer_logs: list[RandomizedOuterPassLog] = field(default_factory=list)
    time_coverage_fraction: float = 0.0
    window_coverage_fraction: float = 0.0
    n_outer_completed: int = 0
    stop_reason: str = ""
    energy_before: float | None = None
    energy_after: float | None = None

    @property
    def n_atoms(self) -> int:
        return len(self.dictionary)

    @property
    def n_matchings(self) -> int:
        return len(self.matchings)


def _crop_matching_patch(
    stack: dict[str, np.ndarray],
    matching: Matching,
    *,
    n_freq: int,
    width: int,
) -> np.ndarray | None:
    arr = stack.get(matching.window_id)
    if arr is None:
        return None
    patch = np.asarray(
        arr[:n_freq, int(matching.t_start) : int(matching.t_start) + width],
        dtype=np.float64,
    )
    if patch.shape != (n_freq, width):
        return None
    return patch


def filter_matchings_by_atom_distance(
    stack: dict[str, np.ndarray],
    dictionary: Dictionary,
    matchings: Sequence[Matching],
    *,
    max_relative_distance: float,
) -> tuple[list[Matching], int]:
    """Keep placements with ``dnorm(x, A) <= max_relative_distance`` (strict).

    Unlike :func:`~..core.filter_matchings_by_relative_distance`, this
    never rescues far patches to satisfy a ``min_keep`` quota.
    Returns ``(kept_matchings, n_filtered)``.
    """
    tau = float(max_relative_distance)
    if not (0.0 <= tau <= 1.0):
        raise ValueError(f"max_relative_distance must be in [0, 1], got {tau}")
    by_id = {int(t.template_id): t for t in dictionary}
    kept: list[Matching] = []
    n_filtered = 0
    for matching in matchings:
        atom = by_id.get(int(matching.template_id))
        if atom is None:
            n_filtered += 1
            continue
        patch = _crop_matching_patch(
            stack,
            matching,
            n_freq=int(atom.n_freq),
            width=int(atom.width),
        )
        if patch is None:
            n_filtered += 1
            continue
        if patch_template_relative_distance(patch, atom) > tau:
            n_filtered += 1
            continue
        kept.append(matching)
    return kept, n_filtered


def dictionary_explained_energy(
    patch: np.ndarray,
    dictionary: Dictionary,
) -> float:
    """``g_D(x)``: max explained energy of ``patch`` over atoms in ``dictionary``.

    Returns ``0.0`` when the dictionary is empty.
    """
    if len(dictionary) == 0:
        return 0.0
    x = np.asarray(patch, dtype=np.float64)
    best = 0.0
    for atom in dictionary:
        w = atom.width
        if x.shape[1] != w or x.shape[0] != atom.n_freq:
            continue
        basis = atom.basis
        explained = 0.0
        for m in range(basis.shape[0]):
            inner = float(np.sum(x * basis[m]))
            explained += inner * inner
        if explained > best:
            best = explained
    return float(best)


def _crop_patch(
    stack: dict[str, np.ndarray],
    window_id: str,
    t_start: int,
    t_end: int,
) -> np.ndarray | None:
    arr = stack.get(window_id)
    if arr is None:
        return None
    patch = np.asarray(arr[:, int(t_start) : int(t_end)], dtype=np.float64)
    if patch.size == 0 or patch.shape[1] <= 0:
        return None
    return patch


def _partition_to_segments(
    residual: dict[str, np.ndarray],
    partition: Sequence[Detection] | Sequence[Segment],
    windows_for_units: Sequence[Window] | None,
) -> list[Segment]:
    if not partition:
        return []
    if isinstance(partition[0], Segment):
        base = list(partition)  # type: ignore[arg-type]
    else:
        if not windows_for_units:
            raise ValueError(
                "windows_for_units (or a Window stack) required with Detection inputs"
            )
        base = detections_to_segments(windows_for_units, partition)  # type: ignore[arg-type]
    # Re-crop from the residual at fixed locations.
    out: list[Segment] = []
    for seg in base:
        patch = _crop_patch(residual, seg.window_id, seg.t_start, seg.t_end)
        if patch is None:
            continue
        out.append(
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
    return out


def _pool_segments(
    segments: Sequence[Segment],
    *,
    original: dict[str, np.ndarray],
    min_template_width: int | None,
    max_template_width: int | None,
    epsilon: float,
    parent: Dictionary | None = None,
    alpha: float | None = None,
) -> list[Segment]:
    """Width / relative residual energy / optional poorly-explained gate.

    Keep a residual crop ``r`` at the same span as original crop ``x`` iff
    ``‖r‖ / ‖x‖ > epsilon`` (and ``‖x‖ > 0``).
    """
    pool: list[Segment] = []
    for seg in segments:
        w = int(seg.data.shape[1])
        if min_template_width is not None and w < min_template_width:
            continue
        if max_template_width is not None and w > max_template_width:
            continue
        orig = _crop_patch(original, seg.window_id, seg.t_start, seg.t_end)
        if orig is None or orig.shape != seg.data.shape:
            continue
        orig_norm = float(np.linalg.norm(orig))
        if orig_norm <= 0.0:
            continue
        resid_norm = float(np.linalg.norm(seg.data))
        if not (resid_norm / orig_norm > float(epsilon)):
            continue
        if parent is not None and alpha is not None:
            if dictionary_explained_energy(seg.data, parent) > float(alpha):
                continue
        pool.append(seg)
    return pool


def create_dictionary(
    residual: Sequence[Window] | dict[str, np.ndarray],
    partition: Sequence[Detection] | Sequence[Segment],
    m: int,
    *,
    windows_for_units: Sequence[Window] | None = None,
    original: Sequence[Window] | dict[str, np.ndarray] | None = None,
    epsilon: float = DEFAULT_CREATE_EPSILON,
    min_template_width: int | None = DEFAULT_MIN_TEMPLATE_WIDTH,
    max_template_width: int | None = DEFAULT_MAX_TEMPLATE_WIDTH,
    template_id_start: int = 0,
    parent: Dictionary | None = None,
    alpha: float | None = None,
    rng: np.random.Generator | int | None = None,
) -> RandomizedCreateResult:
    """CreateDictionary: random order-1 atoms from residual partition patches.

    Parameters
    ----------
    residual
        Spectrogram residual (Windows are mean-centered once via ``as_stack``).
    partition
        GPL detections or pre-built segments (the time partition).
    m
        Target number of seed atoms (sample size).
    original
        Stack used for the denominator ``‖x‖`` in the relative energy gate.
        Defaults to ``residual`` (so a first-shot create keeps any non-silent
        crop whose residual equals the original).
    epsilon
        Keep a patch in the pool iff ``‖r‖ / ‖x‖ > epsilon``, where ``r`` is
        the residual crop and ``x`` is the same span on ``original``.
    parent, alpha
        Optional growth gate: keep only patches with ``g_parent(x) ≤ alpha``.
    """
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    if epsilon < 0.0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon}")

    stack = as_stack(residual)
    original_stack = as_stack(original) if original is not None else stack
    unit_windows: Sequence[Window] | None = windows_for_units
    if unit_windows is None and not isinstance(residual, dict):
        unit_windows = list(residual)  # type: ignore[arg-type]

    segments = _partition_to_segments(stack, partition, unit_windows)
    pool = _pool_segments(
        segments,
        original=original_stack,
        min_template_width=min_template_width,
        max_template_width=max_template_width,
        epsilon=epsilon,
        parent=parent,
        alpha=alpha,
    )
    if not pool:
        return RandomizedCreateResult(dictionary=Dictionary(), pool_size=0, residual=stack)

    gen = (
        rng
        if isinstance(rng, np.random.Generator)
        else np.random.default_rng(rng)
    )
    n_take = min(int(m), len(pool))
    indices = gen.choice(len(pool), size=n_take, replace=False)
    sampled = [pool[int(i)] for i in indices]

    dictionary = Dictionary()
    for offset, seg in enumerate(sampled):
        data = np.asarray(seg.data, dtype=np.float64)
        norm = float(np.linalg.norm(data))
        if norm <= 0.0:
            continue
        dictionary.add(
            Template(
                template_id=template_id_start + offset,
                data=data / norm,
                source_segment_id=seg.segment_id,
            )
        )
    return RandomizedCreateResult(
        dictionary=dictionary,
        pool_size=len(pool),
        n_sampled=len(dictionary),
        residual=stack,
    )


def refine_dictionary(
    residual: Sequence[Window] | dict[str, np.ndarray],
    dictionary: Dictionary,
    *,
    iter: int = DEFAULT_REFINE_ITERS,
    n_components: int = defaults.N_COMPONENTS,
    n_min: int = defaults.MIN_REPEATS,
    eps_step_fraction: float = defaults.EPS_STEP_FRACTION,
    max_repeats: int | None = defaults.MAX_REPEATS,
    max_relative_distance: float | None = defaults.MAX_RELATIVE_DISTANCE,
    post_refine_max_relative_distance: float | None = POST_REFINE_MAX_RELATIVE_DISTANCE,
    normalize: bool | None = None,
    mp_mode: str | None = None,
    n_workers: int | None = None,
    show_progress: bool | None = False,
) -> RandomizedRefineResult:
    """RefineDictionary: whole-dictionary MP, top-K PCA rebuild, prune.

    Each of the ``iter`` iterations (default 3):

    1. ``MP(R, D)`` assigns every accepted patch to exactly one atom.
    2. Atoms with fewer than ``max(n_min, K)`` patches are dropped.
    3. Surviving atoms are rebuilt as the top-``K`` principal subspace of
       their assigned patches (cropped from the *input* residual, not the
       deflated one).

    After all iterations complete, placements with unexplained energy fraction
    ``dnorm(x, A) > post_refine_max_relative_distance`` (default ``0.5``) are
    eliminated. Atoms that then fall below ``n_min`` placements are dropped.
    The returned residual is re-deflated from the input residual using only the
    kept matchings. Pass ``post_refine_max_relative_distance=None`` to skip
    that gate.

    Overlaps are allowed (``disallow_overlap=False``).
    """
    if iter < 0:
        raise ValueError(f"iter must be >= 0, got {iter}")
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    if n_min < 1:
        raise ValueError(f"n_min must be >= 1, got {n_min}")
    if len(dictionary) == 0:
        stack0 = as_stack(residual)
        return RandomizedRefineResult(
            dictionary=Dictionary(),
            residual_after={k: np.array(v, copy=True) for k, v in stack0.items()},
            energies=[residual_energy(stack0)],
        )

    original = as_stack(residual)
    current = copy_dictionary(dictionary)
    matchings: list[Matching] = []
    residual_after = {k: np.array(v, copy=True) for k, v in original.items()}
    energies = [residual_energy(original)]
    n_pruned_total = 0
    n_distance_filtered = 0
    completed = 0

    # Subspace atoms: force global MP without per-template amplitude normalize.
    eff_normalize = False if n_components > 1 else (True if normalize is None else normalize)
    eff_mode = "global" if n_components > 1 else mp_mode
    min_keep = max(int(n_min), int(n_components))

    for _ in range(iter):
        if len(current) == 0:
            break
        mp = matching_pursuit(
            original,
            current,
            eps_step_fraction=eps_step_fraction,
            max_repeats=max_repeats,
            normalize=eff_normalize,
            disallow_overlap=False,
            show_progress=show_progress,
            progress_desc="refine mp",
            mp_mode=eff_mode,
            n_workers=n_workers,
            max_relative_distance=max_relative_distance,
        )
        matchings = list(mp.loop.matchings)
        residual_after = {
            k: np.array(v, copy=True) for k, v in mp.loop.residual_stack.items()
        }
        energies.append(residual_energy(residual_after))

        rebuilt = Dictionary()
        survivors: set[int] = set()
        for atom in current:
            atom_matchings = [m for m in matchings if m.template_id == atom.template_id]
            if len(atom_matchings) < min_keep:
                n_pruned_total += 1
                continue
            new_atom = rebuild_template_from_matchings(
                original,
                atom,
                atom_matchings,
                n_components=n_components,
                method="pca",
                max_relative_distance=max_relative_distance,
            )
            rebuilt.add(new_atom)
            survivors.add(atom.template_id)

        matchings = [m for m in matchings if m.template_id in survivors]
        current = rebuilt
        completed += 1

    # Post-refine: drop placements with dnorm(x, A) > tau, then re-prune atoms.
    if (
        post_refine_max_relative_distance is not None
        and len(current) > 0
        and matchings
    ):
        matchings, n_distance_filtered = filter_matchings_by_atom_distance(
            original,
            current,
            matchings,
            max_relative_distance=float(post_refine_max_relative_distance),
        )
        kept_dict = Dictionary()
        survivors = set()
        for atom in current:
            atom_matchings = [
                m for m in matchings if int(m.template_id) == int(atom.template_id)
            ]
            if len(atom_matchings) < min_keep:
                n_pruned_total += 1
                continue
            kept_dict.add(atom)
            survivors.add(int(atom.template_id))
        matchings = [m for m in matchings if int(m.template_id) in survivors]
        current = kept_dict
        residual_after = {k: np.array(v, copy=True) for k, v in original.items()}
        if matchings and len(current) > 0:
            matchings, _occupied = refit_and_deflate(
                residual_after,
                current,
                matchings,
                normalize=eff_normalize,
            )
        energies.append(residual_energy(residual_after))

    return RandomizedRefineResult(
        dictionary=current,
        matchings=matchings,
        residual_after=residual_after,
        energies=energies,
        iterations_completed=completed,
        n_pruned=n_pruned_total,
        n_distance_filtered=n_distance_filtered,
    )


def build_randomized_dictionary(
    residual: Sequence[Window] | dict[str, np.ndarray],
    partition: Sequence[Detection] | Sequence[Segment],
    m: int,
    *,
    refine_iters: int = DEFAULT_REFINE_ITERS,
    n_components: int = defaults.N_COMPONENTS,
    n_min: int = defaults.MIN_REPEATS,
    epsilon: float = DEFAULT_CREATE_EPSILON,
    min_template_width: int | None = DEFAULT_MIN_TEMPLATE_WIDTH,
    max_template_width: int | None = DEFAULT_MAX_TEMPLATE_WIDTH,
    eps_step_fraction: float = defaults.EPS_STEP_FRACTION,
    max_relative_distance: float | None = defaults.MAX_RELATIVE_DISTANCE,
    post_refine_max_relative_distance: float | None = POST_REFINE_MAX_RELATIVE_DISTANCE,
    rng: np.random.Generator | int | None = None,
    show_progress: bool | None = False,
) -> RandomizedRefineResult:
    """CreateDictionary then RefineDictionary (convenience wrapper)."""
    created = create_dictionary(
        residual,
        partition,
        m,
        epsilon=epsilon,
        min_template_width=min_template_width,
        max_template_width=max_template_width,
        rng=rng,
    )
    return refine_dictionary(
        created.residual,
        created.dictionary,
        iter=refine_iters,
        n_components=n_components,
        n_min=n_min,
        eps_step_fraction=eps_step_fraction,
        max_relative_distance=max_relative_distance,
        post_refine_max_relative_distance=post_refine_max_relative_distance,
        show_progress=show_progress,
    )


def _reindex_dictionary(
    dictionary: Dictionary, matchings: Sequence[Matching]
) -> tuple[Dictionary, list[Matching]]:
    """Keep atoms with ≥1 matching and renumber template ids contiguously from 0."""
    counts: dict[int, int] = {}
    for matching in matchings:
        tid = int(matching.template_id)
        counts[tid] = counts.get(tid, 0) + 1
    survivors = [t for t in dictionary if counts.get(int(t.template_id), 0) > 0]
    id_map = {int(t.template_id): i for i, t in enumerate(survivors)}
    new_dict = Dictionary()
    for template in survivors:
        new_dict.add(
            Template(
                template_id=id_map[int(template.template_id)],
                data=np.asarray(template.data, dtype=np.float64).copy(),
                source_segment_id=template.source_segment_id,
                variances=(
                    None
                    if template.variances is None
                    else np.asarray(template.variances).copy()
                ),
            )
        )
    new_matchings = [
        Matching(
            template_id=id_map[int(m.template_id)],
            window_id=str(m.window_id),
            t_start=int(m.t_start),
            amplitude=float(m.amplitude),
            coeffs=None if m.coeffs is None else np.asarray(m.coeffs).copy(),
        )
        for m in matchings
        if int(m.template_id) in id_map
    ]
    return new_dict, new_matchings


def _offset_dictionary(
    dictionary: Dictionary, matchings: Sequence[Matching], start_id: int
) -> tuple[Dictionary, list[Matching]]:
    """Shift all template ids by ``start_id`` (used when merging outer passes)."""
    new_dict = Dictionary()
    for template in dictionary:
        new_dict.add(
            Template(
                template_id=start_id + int(template.template_id),
                data=np.asarray(template.data, dtype=np.float64).copy(),
                source_segment_id=template.source_segment_id,
                variances=(
                    None
                    if template.variances is None
                    else np.asarray(template.variances).copy()
                ),
            )
        )
    new_matchings = [
        Matching(
            template_id=start_id + int(m.template_id),
            window_id=str(m.window_id),
            t_start=int(m.t_start),
            amplitude=float(m.amplitude),
            coeffs=None if m.coeffs is None else np.asarray(m.coeffs).copy(),
        )
        for m in matchings
    ]
    return new_dict, new_matchings


def _merge_dictionaries(left: Dictionary, right: Dictionary) -> Dictionary:
    out = Dictionary()
    for template in left:
        out.add(template)
    for template in right:
        out.add(template)
    return out


def _coverage_fractions(
    stack: dict[str, np.ndarray],
    dictionary: Dictionary,
    matchings: Sequence[Matching],
) -> tuple[float, float, int, int, int, int]:
    widths = {int(t.template_id): int(t.width) for t in dictionary}
    n_win_m, n_win_t, n_bins_c, n_bins_t = coverage_from_matchings(
        stack, matchings, template_widths=widths
    )
    time_cov = (n_bins_c / n_bins_t) if n_bins_t else 0.0
    win_cov = (n_win_m / n_win_t) if n_win_t else 0.0
    return time_cov, win_cov, n_bins_c, n_bins_t, n_win_m, n_win_t


def learn_randomized_outer(
    residual: Sequence[Window] | dict[str, np.ndarray],
    partition: Sequence[Detection] | Sequence[Segment],
    *,
    m_per_outer: int = DEFAULT_OUTER_M,
    min_coverage: float = DEFAULT_OUTER_MIN_COVERAGE,
    max_outer: int = DEFAULT_OUTER_MAX_OUTER,
    refine_iters: int = DEFAULT_REFINE_ITERS,
    n_components: int = defaults.N_COMPONENTS,
    n_min: int = DEFAULT_OUTER_N_MIN,
    epsilon: float = DEFAULT_CREATE_EPSILON,
    min_template_width: int | None = DEFAULT_MIN_TEMPLATE_WIDTH,
    max_template_width: int | None = DEFAULT_MAX_TEMPLATE_WIDTH,
    eps_step_fraction: float = defaults.EPS_STEP_FRACTION,
    max_relative_distance: float | None = defaults.MAX_RELATIVE_DISTANCE,
    post_refine_max_relative_distance: float | None = DEFAULT_OUTER_POST_REFINE_TAU,
    windows_for_units: Sequence[Window] | None = None,
    rng: np.random.Generator | int | None = None,
    show_progress: bool | None = False,
    progress_callback: Callable[[RandomizedOuterPassLog], None] | None = None,
) -> RandomizedOuterResult:
    """Coverage-gated residual outer loop over Create→Refine.

    Each outer pass:

    1. Sample ``m_per_outer`` random seed atoms from the current residual
       partition (CreateDictionary).
    2. Refine those seeds with ``n_min`` patch pruning and post-refine
       ``dnorm`` gate ``post_refine_max_relative_distance``.
    3. Merge survivors into an accumulating dictionary / matching set.
    4. Peel the residual with the pass matchings and continue.

    Stops when cumulative time coverage exceeds ``min_coverage``, when
    ``max_outer`` passes complete, when a pass adds no surviving atoms, or
    when the seed pool is empty.

    Default recipe (loose distance, stricter repeats, small random batches)::

        learn_randomized_outer(
            windows, detections,
            m_per_outer=5,
            n_min=10,
            post_refine_max_relative_distance=0.9,
            min_coverage=0.5,
        )
    """
    if m_per_outer < 1:
        raise ValueError(f"m_per_outer must be >= 1, got {m_per_outer}")
    if not (0.0 < float(min_coverage) <= 1.0):
        raise ValueError(f"min_coverage must be in (0, 1], got {min_coverage}")
    if max_outer < 1:
        raise ValueError(f"max_outer must be >= 1, got {max_outer}")
    if n_min < 1:
        raise ValueError(f"n_min must be >= 1, got {n_min}")

    unit_windows: Sequence[Window] | None = windows_for_units
    if unit_windows is None and not isinstance(residual, dict):
        unit_windows = list(residual)  # type: ignore[arg-type]

    original = as_stack(residual)
    residual_stack = {k: np.array(v, copy=True) for k, v in original.items()}
    gen = (
        rng
        if isinstance(rng, np.random.Generator)
        else np.random.default_rng(rng)
    )

    accum = Dictionary()
    accum_matchings: list[Matching] = []
    outer_logs: list[RandomizedOuterPassLog] = []
    energy_start: float | None = None
    stop_reason = "max_outer"

    for outer in range(1, int(max_outer) + 1):
        created = create_dictionary(
            residual_stack,
            partition,
            m_per_outer,
            windows_for_units=unit_windows,
            original=original,
            epsilon=epsilon,
            min_template_width=min_template_width,
            max_template_width=max_template_width,
            rng=gen,
        )
        if created.n_sampled == 0 or len(created.dictionary) == 0:
            stop_reason = "empty_pool"
            break

        refined = refine_dictionary(
            residual_stack,
            created.dictionary,
            iter=refine_iters,
            n_components=n_components,
            n_min=n_min,
            eps_step_fraction=eps_step_fraction,
            max_relative_distance=max_relative_distance,
            post_refine_max_relative_distance=post_refine_max_relative_distance,
            show_progress=show_progress,
        )
        if energy_start is None and refined.energies:
            energy_start = refined.energies[0]

        if len(refined.dictionary) == 0:
            stop_reason = "no_survivors"
            break

        new_dict, new_match = _reindex_dictionary(refined.dictionary, refined.matchings)
        new_dict, new_match = _offset_dictionary(new_dict, new_match, start_id=len(accum))
        accum = _merge_dictionaries(accum, new_dict)
        accum_matchings = list(accum_matchings) + list(new_match)
        accum, accum_matchings = _reindex_dictionary(accum, accum_matchings)

        time_cov, win_cov, n_bins_c, n_bins_t, n_win_m, n_win_t = _coverage_fractions(
            original, accum, accum_matchings
        )
        log = RandomizedOuterPassLog(
            outer=outer,
            pool_size=created.pool_size,
            n_sampled=created.n_sampled,
            n_atoms_this_pass=len(refined.dictionary),
            n_matchings_this_pass=len(refined.matchings),
            n_pruned=refined.n_pruned,
            n_distance_filtered=refined.n_distance_filtered,
            n_atoms=len(accum),
            n_matchings=len(accum_matchings),
            time_coverage_fraction=time_cov,
            window_coverage_fraction=win_cov,
            n_bins_covered=n_bins_c,
            n_bins_total=n_bins_t,
            n_windows_matched=n_win_m,
            n_windows_total=n_win_t,
            energy_before=refined.energies[0] if refined.energies else None,
            energy_after=refined.energies[-1] if refined.energies else None,
        )
        outer_logs.append(log)
        if progress_callback is not None:
            progress_callback(log)

        residual_stack = refined.residual_after

        if time_cov > float(min_coverage):
            stop_reason = "coverage"
            break
    else:
        stop_reason = "max_outer"

    time_cov, win_cov, *_ = _coverage_fractions(original, accum, accum_matchings)
    energy_after = (
        outer_logs[-1].energy_after if outer_logs else residual_energy(residual_stack)
    )
    return RandomizedOuterResult(
        dictionary=accum,
        matchings=accum_matchings,
        residual_after=residual_stack,
        original_stack=original,
        outer_logs=outer_logs,
        time_coverage_fraction=time_cov,
        window_coverage_fraction=win_cov,
        n_outer_completed=len(outer_logs),
        stop_reason=stop_reason,
        energy_before=energy_start,
        energy_after=energy_after,
    )
