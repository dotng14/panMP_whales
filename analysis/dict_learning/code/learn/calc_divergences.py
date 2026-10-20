"""Asymmetric residual-energy divergences between dictionary atoms.

Joint MP peels the stack with the full dictionary to residual ``R``. For atoms
``A`` and ``B``:

1. Inflate (add back) matchings of ``A`` to form ``R[-A]``.
2. If ``ΔA = E(R[-A]) − E(R)`` is negligible, row ``A`` is insignificant.
3. Otherwise MP with only ``B`` on ``R[-A]`` yields ``R[-A+B]``.
4. Divergence
   ``D(A,B) = (E(R[-A+B]) − E(R)) / (E(R[-A]) − E(R))``.

``D ≈ 0`` means ``B`` re-explains almost what ``A`` held (``A`` is close to
``B`` as a substitute). The measure is not symmetric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from ..core.defaults import EPS_STEP_FRACTION, MAX_RELATIVE_DISTANCE
from ..core.objects import Dictionary, Matching, Template, Window
from ..core.utils import as_stack
from ..mp.mp_step import (
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
