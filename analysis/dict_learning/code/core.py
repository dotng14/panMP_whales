"""Windows, templates, matchings, and how much of a patch an atom leaves unexplained."""

from __future__ import annotations

import os


def _env_int(name: str, default: int) -> int:  # no-pipeline-cover: reads GPL_REPEATS_* knobs, used only by GPL_Python/preprocess.py
    """Positive int from the environment, falling back to ``default``."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return value if value > 0 else default


# Matches analysis/dict_learning/notebooks/seed_from_gpl_LeiData_setup.py (on-disk knobs)
# and the README worked example / repeat_builder tuning.
EPS_STEP_FRACTION = 0.01
MAX_ATOMS_PER_ROUND = 10  # legacy catalog/CLI alias; MP stops via eps_step_fraction
# Scope of a learning run, overridable via env (see backend/env.example).
# One outer round yields one template, so N_ROUNDS is the dictionary size.
# 30 windows / 10 templates: widened from the original 10/1 so later
# (residual-deflation) templates have enough underlying data to find more than
# one or two repeats each.
N_WINDOWS = _env_int("GPL_REPEATS_N_WINDOWS", 30)
N_ROUNDS = _env_int("GPL_REPEATS_N_ROUNDS", 10)
NORMALIZE = True
# Candidate GPL patch width (time bins); None = no filter.
MIN_TEMPLATE_WIDTH: int | None = None
MAX_TEMPLATE_WIDTH: int | None = 24
WINDOW_OFFSET = 0
# How each atom is rebuilt from its matched patches: "pca" (leading principal
# component / largest eigenvector) or "average" (element-wise mean).
TEMPLATE_METHOD = "pca"
# Number of PCA modes per atom. ``1`` is rank-1 MP (catalog-compatible).
# ``k>1`` uses subspace MP; builds persist to a manifest only (no catalog schema).
N_COMPONENTS = 1
# Closeness gate shared by MP acceptance and template rebuilds: reject a patch
# when more than this fraction of its energy is left unexplained by the atom
# (0 = exact match only, 1 = accept everything, None = gate off).
MAX_RELATIVE_DISTANCE: float | None = None
# Optional multi-atom refine after outer seed→average rounds (build_dictionary).
REFINE_ITERS = 0
# Extend-dictionary stage: stop adding atoms once time-bin coverage reaches this.
MIN_COVERAGE = 0.9
# Minimum matchings required in both the seed trial and the averaged re-MP.
MIN_REPEATS = 3
# Optional Mallat cap: stop after this many accepted matchings (None = no cap).
# In per_window mode the cap applies per window.
MAX_REPEATS: int | None = None
# Seed candidate window policy for learn_averaged_templates_from_gpl:
#   "global" — trial every residual GPL candidate (legacy)
#   "low_coverage" — seed from least-covered window (+ neighborhood); match on full stack
SEED_WINDOW_POLICY = "global"
SEED_WINDOW_NEIGHBORHOOD = 0
# Incremental builder: initial span; max span is caller-required (None = use pool size).
INITIAL_N_WINDOWS = N_WINDOWS
MAX_N_WINDOWS: int | None = None
# Incremental builder candidate width band (time bins).
INCREMENTAL_MIN_TEMPLATE_WIDTH: int | None = 12
INCREMENTAL_MAX_TEMPLATE_WIDTH: int | None = 24

# Matching Pursuit parallelism / mode / compute.
# Override with DICT_LEARNING_MP_WORKERS / DICT_LEARNING_MP_MODE /
# DICT_LEARNING_MP_COMPUTE (or GPL_COMPUTE=cuda), or CLI flags.
def _default_mp_workers() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):  # no-pipeline-cover: platforms without sched_getaffinity
        return max(1, int(os.cpu_count() or 1))


MP_WORKERS = _default_mp_workers()
# per_window: independent Mallat per window (default). global: classic stack Mallat.
MP_MODE = "per_window"
# auto: CUDA when CuPy works and problem is large enough; else CPU (Numba/NumPy).
MP_COMPUTE = "auto"

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Window:
    """One spectrogram tile (GPL process window) with metadata."""

    window_id: str
    data: np.ndarray  # shape (F, T)
    start_sample: int  # file-global 1-based
    end_sample: int
    overlap: int  # STFT hop samples
    fftl: int = 0
    sample_freq: float = 0.0
    start_freq: float = 0.0
    end_freq: float = 0.0
    npy_path: Path | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_freq(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_time(self) -> int:
        return int(self.data.shape[1])


@dataclass
class Segment:
    """Spectrogram patch cropped from a window (one GPL detection overlap)."""

    segment_id: str
    data: np.ndarray  # (F, W)
    window_id: str
    t_start: int
    t_end: int  # exclusive
    global_start_sample: int
    global_end_sample: int
    detection_index: int = -1
    boundary_cross_flag: int = 0

    @property
    def energy(self) -> float:  # no-pipeline-cover: kept for notebooks; the CLIs never read it
        """Frobenius energy ‖U‖²_F of the patch."""
        return float(np.sum(np.asarray(self.data, dtype=np.float64) ** 2))


@dataclass
class Template:
    """Dictionary atom: rank-1 ``(F, W)`` or subspace ``(k, F, W)`` basis."""

    template_id: int
    data: np.ndarray  # (F, W) or (k, F, W)
    source_segment_id: str | None = None
    variances: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float64)
        if self.data.ndim not in (2, 3):  # no-pipeline-cover: library input check
            raise ValueError(
                f"Template.data must be (F, W) or (k, F, W), got shape {self.data.shape}"
            )
        if self.variances is not None:
            self.variances = np.asarray(self.variances, dtype=np.float64)

    @property
    def n_components(self) -> int:
        return 1 if self.data.ndim == 2 else int(self.data.shape[0])

    @property
    def basis(self) -> np.ndarray:
        """Orthonormal modes as ``(k, F, W)`` (view/copy-safe contiguous)."""
        if self.data.ndim == 2:
            return np.ascontiguousarray(self.data[np.newaxis, ...])
        return np.ascontiguousarray(self.data)

    @property
    def as_rank1(self) -> np.ndarray:  # no-pipeline-cover: library helper; the CLIs never call it
        """``(F, W)`` slice; requires ``n_components == 1``."""
        if self.n_components != 1:
            raise ValueError(
                f"as_rank1 requires n_components=1, got {self.n_components}"
            )
        return self.basis[0]

    @property
    def width(self) -> int:
        return int(self.data.shape[-1])

    @property
    def n_freq(self) -> int:
        return int(self.data.shape[-2])


@dataclass
class Matching:
    """One activation of a template in a window.

    Construct with ``amplitude=`` (rank-1) and/or ``coeffs=`` (length ``k``).
    """

    template_id: int
    window_id: str
    t_start: int
    amplitude: float | None = None
    coeffs: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.coeffs is not None:
            c = np.asarray(self.coeffs, dtype=np.float64).reshape(-1)
        elif self.amplitude is not None:
            c = np.asarray([float(self.amplitude)], dtype=np.float64)
        else:
            raise ValueError("Matching requires amplitude= or coeffs=")  # no-pipeline-cover: library input check
        object.__setattr__(self, "coeffs", c)
        if c.size == 1:
            object.__setattr__(self, "amplitude", float(c[0]))
        else:
            object.__setattr__(self, "amplitude", float(np.linalg.norm(c)))

    @property
    def energy(self) -> float:
        c = np.asarray(self.coeffs, dtype=np.float64)
        return float(np.dot(c, c))

    @property
    def n_components(self) -> int:
        return int(np.asarray(self.coeffs).size)


@dataclass
class Dictionary:
    """Active set of templates keyed by template_id."""

    templates: dict[int, Template] = field(default_factory=dict)
    order: list[int] = field(default_factory=list)

    def add(self, template: Template) -> None:
        tid = template.template_id
        if tid not in self.templates:  # no-pipeline-branch: the CLIs never re-add a template id
            self.order.append(tid)
        self.templates[tid] = template

    def __iter__(self):
        for tid in self.order:
            yield self.templates[tid]

    def __len__(self) -> int:
        return len(self.order)

    @property
    def max_n_components(self) -> int:
        if not self.order:  # no-pipeline-cover: the CLIs never ask an empty dictionary
            return 1
        return max(t.n_components for t in self)

def mean_center(arr: np.ndarray) -> np.ndarray:
    """Subtract the global mean so ``arr`` has zero mean (float64 copy)."""
    a = np.asarray(arr, dtype=np.float64)
    return a - float(np.mean(a))


from typing import Sequence

import numpy as np


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
    if not intervals:  # no-pipeline-cover: only called with at least one interval
        return 0
    ordered = sorted((int(a), int(b)) for a, b in intervals if b > a)
    if not ordered:  # no-pipeline-cover: callers pass only non-empty intervals
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
        if wid not in stack:  # no-pipeline-cover: matchings come from the same stack
            continue
        if template_widths is not None:
            width = int(template_widths.get(matching.template_id, template_width or 1))
        else:
            width = int(template_width if template_width is not None else 1)  # no-pipeline-cover: the CLIs always pass template_widths
        t_width = stack[wid].shape[1]
        start = max(0, int(matching.t_start))
        end = min(t_width, start + width)
        if end > start:  # no-pipeline-branch: placements always lie inside their window
            by_window.setdefault(wid, []).append((start, end))
    n_windows_matched = len(by_window)
    n_bins_covered = sum(union_interval_length(ivs) for ivs in by_window.values())
    return n_windows_matched, n_windows_total, n_bins_covered, n_bins_total


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
        raise ValueError(f"patch must be 2-D (F, W), got shape {x.shape}")  # no-pipeline-cover: library input check

    if isinstance(template, Template):
        basis = np.asarray(template.basis, dtype=np.float64)
    else:  # no-pipeline-cover: library input: array template; the CLIs pass Template atoms
        arr = np.asarray(template, dtype=np.float64)
        if arr.ndim == 2:
            basis = arr[np.newaxis, ...]
        elif arr.ndim == 3:
            basis = arr
        else:
            raise ValueError(
                f"template must be (F, W) or (k, F, W), got shape {arr.shape}"
            )

    if basis.shape[1:] != x.shape:  # no-pipeline-cover: library input check
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
            explained = 0.0  # no-pipeline-cover: atoms always have energy
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
    return distance  # no-pipeline-cover: the CLIs only ask for the parts


def template_energy(template: Template | np.ndarray) -> float:
    """Frobenius energy of a template / subspace basis."""
    if isinstance(template, Template):
        arr = np.asarray(template.basis, dtype=np.float64)
    else:  # no-pipeline-cover: library input: array template
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
        return 1.0  # no-pipeline-cover: placed patches always have energy
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
        raise ValueError(f"max_relative_distance must be in [0, 1], got {tau}")  # no-pipeline-cover: the CLIs check the range first
    if min_keep < 1:
        raise ValueError(f"min_keep must be >= 1, got {min_keep}")  # no-pipeline-cover: callers pass min_keep >= 1
    items = list(matchings)
    if not items or tau >= 1.0:
        return items
    if template_energy(template) <= 0.0:  # no-pipeline-cover: atoms always have energy
        return items

    if isinstance(template, Template):
        w = int(width) if width is not None else int(template.width)
        freq = int(n_freq) if n_freq is not None else int(template.n_freq)
        ref = template
    else:  # no-pipeline-cover: library input: array template; the CLIs pass Template atoms
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
        if arr is None:  # no-pipeline-cover: matchings come from the same stack
            continue
        patch = np.asarray(
            arr[:freq, int(matching.t_start) : int(matching.t_start) + w],
            dtype=np.float64,
        )
        if patch.shape != (freq, w):  # no-pipeline-cover: MP places whole atoms
            # Skip incomplete edge crops rather than inventing padding here.
            continue
        r = patch_template_relative_distance(patch, ref)
        scored.append((r, matching))

    if not scored:  # no-pipeline-cover: follows from the two checks above
        return items[: max(min_keep, 1)] if items else []

    kept = [m for r, m in scored if r <= tau]
    if len(kept) >= min_keep:
        return kept
    # Too few after the cut: keep the closest min_keep.
    order = np.argsort(np.asarray([r for r, _ in scored], dtype=np.float64))  # no-pipeline-cover: MP placed these patches under the same gate and atom
    n_take = min(int(min_keep), len(scored))  # no-pipeline-cover: MP placed these patches under the same gate and atom
    return [scored[int(i)][1] for i in order[:n_take]]  # no-pipeline-cover: MP placed these patches under the same gate and atom

class defaults:
    """Knobs accessed as ``defaults.NAME``."""


for _name, _value in list(globals().items()):
    if _name.isupper():
        setattr(defaults, _name, _value)
del _name, _value
