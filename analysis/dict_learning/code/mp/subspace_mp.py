"""Subspace matching pursuit — compatibility wrappers over unified MP.

The production path in :mod:`.mp_step` now scores ``k>1`` atoms by projection
energy and deflates with the full subspace projection. This module keeps the
historical ``SubspaceAtom`` / ``SubspaceMatching`` names as thin aliases so
existing notebooks and the compare harness keep working.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .mp_step import MpLoopResult, copy_residual_stack, residual_energy, run_mp_loop
from ..core.objects import Dictionary, Matching, Template, Window

IntervalMap = dict[str, list[tuple[int, int]]]


@dataclass
class SubspaceAtom:
    """A dictionary atom spanning the top-``k`` PCA modes of a template."""

    template_id: int
    basis: np.ndarray
    variances: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.basis = np.ascontiguousarray(self.basis, dtype=np.float64)
        if self.basis.ndim != 3:
            raise ValueError(
                f"basis must be (k, n_freq, width), got shape {self.basis.shape}"
            )

    @property
    def n_components(self) -> int:
        return int(self.basis.shape[0])

    @property
    def n_freq(self) -> int:
        return int(self.basis.shape[1])

    @property
    def width(self) -> int:
        return int(self.basis.shape[2])

    def to_template(self) -> Template:
        data = self.basis[0] if self.n_components == 1 else self.basis
        return Template(
            template_id=self.template_id,
            data=data,
            variances=self.variances,
        )


@dataclass
class SubspaceMatching:
    """One activation of a subspace atom: ``k`` coefficients at a location."""

    template_id: int
    window_id: str
    t_start: int
    coeffs: np.ndarray

    @property
    def energy(self) -> float:
        c = np.asarray(self.coeffs, dtype=np.float64)
        return float(np.dot(c, c))

    def to_matching(self) -> Matching:
        return Matching(
            template_id=self.template_id,
            window_id=self.window_id,
            t_start=self.t_start,
            coeffs=np.asarray(self.coeffs, dtype=np.float64),
        )

    @classmethod
    def from_matching(cls, matching: Matching) -> SubspaceMatching:
        return cls(
            template_id=matching.template_id,
            window_id=matching.window_id,
            t_start=matching.t_start,
            coeffs=np.asarray(matching.coeffs, dtype=np.float64).reshape(-1),
        )


@dataclass
class SubspaceStepResult:
    matching: SubspaceMatching | None
    energy_before: float
    energy_after: float
    delta_energy: float


@dataclass
class SubspaceMpLoopResult:
    residual_stack: dict[str, np.ndarray]
    matchings: list[SubspaceMatching] = field(default_factory=list)
    energies: list[float] = field(default_factory=list)
    stopped_reason: str = ""


def build_subspace_atom(
    signal,
    template_id: int,
    matchings,
    *,
    width: int,
    n_freq: int | None = None,
    n_components: int = 1,
) -> SubspaceAtom:
    """Build a :class:`SubspaceAtom` from a template's matched patches via PCA."""
    from ..tools import pca_basis

    basis, variances = pca_basis(
        signal, matchings, width=width, n_freq=n_freq, n_components=n_components
    )
    return SubspaceAtom(template_id=template_id, basis=basis, variances=variances)


def _atoms_to_dictionary(atoms: list[SubspaceAtom]) -> Dictionary:
    d = Dictionary()
    for atom in atoms:
        d.add(atom.to_template())
    return d


def subspace_mp_step(
    resid: dict[str, np.ndarray],
    atoms: list[SubspaceAtom],
    *,
    occupied_intervals: IntervalMap | None = None,
    pc1_screen: bool = False,
    screen_top: int = 8,
) -> SubspaceStepResult:
    """One global subspace-MP iteration (delegates to :func:`.mp_step.mp_step`)."""
    from .mp_step import mp_step

    stack, step = mp_step(
        resid,
        _atoms_to_dictionary(atoms),
        normalize=False,
        occupied_intervals=occupied_intervals,
        pc1_screen=pc1_screen,
        screen_top=screen_top,
    )
    # mp_step returns a new map only when input was not already a dict of views;
    # mutate caller's resid in place to preserve prior prototype semantics.
    for wid, arr in stack.items():
        resid[wid] = arr
    matching = (
        None if step.matching is None else SubspaceMatching.from_matching(step.matching)
    )
    return SubspaceStepResult(
        matching=matching,
        energy_before=step.energy_before,
        energy_after=step.energy_after,
        delta_energy=step.delta_energy,
    )


def run_subspace_mp_loop(
    residual_stack: dict[str, np.ndarray] | list[Window],
    atoms: list[SubspaceAtom],
    *,
    energy_floor: float | None = None,
    eps_step_fraction: float = 0.5,
    disallow_overlap: bool = False,
    occupied_intervals: IntervalMap | None = None,
    pc1_screen: bool = False,
    screen_top: int = 8,
    show_progress: bool | None = None,
    progress_desc: str = "subspace mp",
) -> SubspaceMpLoopResult:
    """Repeat global subspace-MP steps (delegates to :func:`.mp_step.run_mp_loop`)."""
    result: MpLoopResult = run_mp_loop(
        residual_stack,
        _atoms_to_dictionary(atoms),
        energy_floor=energy_floor,
        eps_step_fraction=eps_step_fraction,
        normalize=False,
        disallow_overlap=disallow_overlap,
        occupied_intervals=occupied_intervals,
        show_progress=show_progress,
        progress_desc=progress_desc,
        mp_mode="global",
        pc1_screen=pc1_screen,
        screen_top=screen_top,
    )
    return SubspaceMpLoopResult(
        residual_stack=result.residual_stack,
        matchings=[SubspaceMatching.from_matching(m) for m in result.matchings],
        energies=result.energies,
        stopped_reason=result.stopped_reason,
    )


# Re-export helpers used by older call sites / tests.
__all__ = [
    "SubspaceAtom",
    "SubspaceMatching",
    "SubspaceStepResult",
    "SubspaceMpLoopResult",
    "build_subspace_atom",
    "subspace_mp_step",
    "run_subspace_mp_loop",
    "copy_residual_stack",
    "residual_energy",
]
