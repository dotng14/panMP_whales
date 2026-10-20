"""Picklable workers for parallel GPL candidate MP trials."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from .mp_step import residual_energy, run_mp_loop, single_template_dictionary
from ..core.objects import Matching, Template

# Set once per process-pool worker via :func:`init_trial_worker_shared` so each
# candidate task need not pickle a full residual stack through the task queue.
_SHARED_RESIDUAL: dict[str, np.ndarray] | None = None
_SHARED_OCCUPIED: dict[str, list[tuple[int, int]]] | None = None


def init_trial_worker_shared(
    residual: dict[str, np.ndarray],
    occupied_intervals: dict[str, list[tuple[int, int]]] | None = None,
) -> None:
    """ProcessPool initializer: share one residual copy per worker process."""
    global _SHARED_RESIDUAL, _SHARED_OCCUPIED
    _SHARED_RESIDUAL = residual
    _SHARED_OCCUPIED = occupied_intervals


def clear_trial_worker_shared() -> None:
    """Drop shared worker state (tests / sequential cleanup)."""
    global _SHARED_RESIDUAL, _SHARED_OCCUPIED
    _SHARED_RESIDUAL = None
    _SHARED_OCCUPIED = None


def _resolve_residual(
    residual: dict[str, np.ndarray] | None,
) -> dict[str, np.ndarray]:
    if residual is not None:
        return residual
    if _SHARED_RESIDUAL is None:
        raise RuntimeError(
            "trial residual missing: pass residual= on the payload or call "
            "init_trial_worker_shared in the worker process"
        )
    return _SHARED_RESIDUAL


def _resolve_occupied(
    occupied_intervals: dict[str, list[tuple[int, int]]] | None,
    *,
    residual_on_payload: bool,
) -> dict[str, list[tuple[int, int]]] | None:
    if residual_on_payload or occupied_intervals is not None:
        return occupied_intervals
    return _SHARED_OCCUPIED


@dataclass(frozen=True)
class CandidateTrialPayload:
    """Inputs for one isolated single-template MP trial (seed path).

    ``residual`` may be ``None`` when the worker was initialized with
    :func:`init_trial_worker_shared` (ProcessPool path).
    """

    segment_id: str
    template_id: int
    template_data: np.ndarray
    residual: dict[str, np.ndarray] | None
    eps_step_fraction: float
    normalize: bool
    disallow_overlap: bool
    occupied_intervals: dict[str, list[tuple[int, int]]] | None
    mp_mode: str
    n_workers: int  # window-level workers inside this trial (usually 1)
    max_repeats: int | None = None
    keep_residual: bool = False
    max_relative_distance: float | None = None


@dataclass
class CandidateTrialResult:
    segment_id: str
    n_matchings: int
    energy_after: float
    matchings: list[Matching]
    residual_stack: dict[str, np.ndarray]
    energies: list[float]
    template_data: np.ndarray
    template_id: int


def run_candidate_trial(payload: CandidateTrialPayload) -> CandidateTrialResult:
    """Module-level worker: copy residual + run single-template MP.

    By default omits ``residual_stack`` from the result so parallel maps do not
    retain one full stack per candidate. Rematerialize the winner with
    ``keep_residual=True`` (see ``rematerialize_candidate_trial``).
    """
    residual = _resolve_residual(payload.residual)
    occupied = _resolve_occupied(
        payload.occupied_intervals,
        residual_on_payload=payload.residual is not None,
    )
    tmpl = Template(
        template_id=payload.template_id,
        data=np.asarray(payload.template_data, dtype=np.float64),
        source_segment_id=payload.segment_id,
    )
    trial = run_mp_loop(
        residual,
        single_template_dictionary(tmpl),
        eps_step_fraction=payload.eps_step_fraction,
        max_repeats=payload.max_repeats,
        normalize=payload.normalize,
        disallow_overlap=payload.disallow_overlap,
        occupied_intervals=occupied,
        show_progress=False,
        progress_desc="seed mp",
        mp_mode=payload.mp_mode,
        n_workers=payload.n_workers,
        max_relative_distance=payload.max_relative_distance,
    )
    resid = trial.residual_stack if payload.keep_residual else {}
    return CandidateTrialResult(
        segment_id=payload.segment_id,
        n_matchings=len(trial.matchings),
        energy_after=(
            float(trial.energies[-1])
            if trial.energies
            else residual_energy(trial.residual_stack)
        ),
        matchings=list(trial.matchings),
        residual_stack=resid,
        energies=list(trial.energies),
        template_data=np.asarray(payload.template_data, dtype=np.float64),
        template_id=payload.template_id,
    )


def rematerialize_candidate_trial(
    payload: CandidateTrialPayload,
) -> CandidateTrialResult:
    """Re-run one trial and return the residual stack (winner path)."""
    if payload.residual is None:
        raise ValueError(
            "rematerialize_candidate_trial requires payload.residual to be set"
        )
    return run_candidate_trial(replace(payload, keep_residual=True))


@dataclass(frozen=True)
class SeedAvgTrialPayload:
    """Seed MP + average + re-MP for one GPL candidate (learn path).

    ``residual`` may be ``None`` when using :func:`init_trial_worker_shared`.
    """

    segment_id: str
    template_id: int
    template_data: np.ndarray
    residual: dict[str, np.ndarray] | None
    eps_step_fraction: float
    normalize: bool
    disallow_overlap: bool
    occupied_intervals: dict[str, list[tuple[int, int]]] | None
    min_repeats: int
    mp_mode: str
    n_workers: int
    template_method: str = "pca"
    max_repeats: int | None = None
    keep_residual: bool = False
    max_relative_distance: float | None = None


@dataclass
class SeedAvgTrialResult:
    segment_id: str
    seed_n_matchings: int
    seed_energy_after: float
    seed_matchings: list[Matching]
    seed_energies: list[float]
    seed_template_data: np.ndarray
    refined_n_matchings: int
    refined_energy_after: float
    refined_matchings: list[Matching]
    refined_energies: list[float]
    refined_template_data: np.ndarray
    refined_residual: dict[str, np.ndarray]
    qualifies: bool


def run_seed_avg_trial(payload: SeedAvgTrialPayload) -> SeedAvgTrialResult:
    """Module-level worker for learn_averaged_templates_from_gpl candidate trials.

    Omits ``refined_residual`` unless ``keep_residual=True`` so a parallel map
    over hundreds of candidates does not retain hundreds of full stacks.
    """
    from ..learn.learn_dictionary import refine_seed_by_averaging
    from ..learn.seed_from_gpl import Seed

    residual = _resolve_residual(payload.residual)
    occupied = _resolve_occupied(
        payload.occupied_intervals,
        residual_on_payload=payload.residual is not None,
    )
    tmpl = Template(
        template_id=payload.template_id,
        data=np.asarray(payload.template_data, dtype=np.float64),
        source_segment_id=payload.segment_id,
    )
    trial = run_mp_loop(
        residual,
        single_template_dictionary(tmpl),
        eps_step_fraction=payload.eps_step_fraction,
        max_repeats=payload.max_repeats,
        normalize=payload.normalize,
        disallow_overlap=payload.disallow_overlap,
        occupied_intervals=occupied,
        show_progress=False,
        progress_desc="seed mp",
        mp_mode=payload.mp_mode,
        n_workers=payload.n_workers,
        max_relative_distance=payload.max_relative_distance,
    )
    seed_n = len(trial.matchings)
    seed_e = (
        float(trial.energies[-1])
        if trial.energies
        else residual_energy(trial.residual_stack)
    )
    if seed_n < payload.min_repeats:
        return SeedAvgTrialResult(
            segment_id=payload.segment_id,
            seed_n_matchings=seed_n,
            seed_energy_after=seed_e,
            seed_matchings=list(trial.matchings),
            seed_energies=list(trial.energies),
            seed_template_data=np.asarray(payload.template_data, dtype=np.float64),
            refined_n_matchings=0,
            refined_energy_after=seed_e,
            refined_matchings=[],
            refined_energies=[],
            refined_template_data=np.asarray(payload.template_data, dtype=np.float64),
            refined_residual={},
            qualifies=False,
        )

    seed = Seed(
        template=tmpl,
        matchings=trial.matchings,
        # Seed residual is only needed inside refine; never returned from the
        # batch worker unless keep_residual is set on the refined result.
        residual_after=trial.residual_stack,
        energy_before=(
            float(trial.energies[0])
            if trial.energies
            else residual_energy(residual)
        ),
        energy_after=seed_e,
        n_candidates_tried=1,
        selected_segment_id=payload.segment_id,
        mp_energies=list(trial.energies),
    )
    refined = refine_seed_by_averaging(
        residual,
        seed,
        eps_step_fraction=payload.eps_step_fraction,
        max_repeats=payload.max_repeats,
        normalize=payload.normalize,
        disallow_overlap=payload.disallow_overlap,
        occupied_intervals=occupied,
        template_method=payload.template_method,
        show_progress=False,
        mp_mode=payload.mp_mode,
        n_workers=payload.n_workers,
        max_relative_distance=payload.max_relative_distance,
    )
    refined_tmpl = refined.get_template()
    refined_matchings = refined.get_matchings()
    refined_resid = (
        (refined.get_residual_after() or {}) if payload.keep_residual else {}
    )
    qualifies = len(refined_matchings) >= payload.min_repeats and refined_tmpl is not None
    return SeedAvgTrialResult(
        segment_id=payload.segment_id,
        seed_n_matchings=seed_n,
        seed_energy_after=seed_e,
        seed_matchings=list(trial.matchings),
        seed_energies=list(trial.energies),
        seed_template_data=np.asarray(payload.template_data, dtype=np.float64),
        refined_n_matchings=len(refined_matchings),
        refined_energy_after=refined.get_energy_after(),
        refined_matchings=list(refined_matchings),
        refined_energies=list(refined.get_mp_energies()),
        refined_template_data=(
            np.asarray(refined_tmpl.data, dtype=np.float64)
            if refined_tmpl is not None
            else np.asarray(payload.template_data, dtype=np.float64)
        ),
        refined_residual=refined_resid,
        qualifies=qualifies,
    )


def rematerialize_seed_avg_trial(payload: SeedAvgTrialPayload) -> SeedAvgTrialResult:
    """Re-run one seed+avg trial and return the refined residual (winner path)."""
    if payload.residual is None:
        raise ValueError(
            "rematerialize_seed_avg_trial requires payload.residual to be set"
        )
    return run_seed_avg_trial(replace(payload, keep_residual=True))


def trial_sort_key_energy_then_segment(result: Any) -> tuple:
    """Deterministic min-energy key with segment_id tie-break."""
    if isinstance(result, CandidateTrialResult):
        return (result.energy_after, result.segment_id)
    if isinstance(result, SeedAvgTrialResult):
        return (result.refined_energy_after, result.segment_id)
    raise TypeError(f"unsupported trial result type: {type(result)!r}")


def payload_by_segment_id(
    payloads: list[CandidateTrialPayload] | list[SeedAvgTrialPayload],
    segment_id: str,
) -> CandidateTrialPayload | SeedAvgTrialPayload:
    """Look up the trial payload for a winning ``segment_id``."""
    for payload in payloads:
        if payload.segment_id == segment_id:
            return payload
    raise KeyError(f"no trial payload for segment_id={segment_id!r}")


def slim_payloads_for_process_pool(
    payloads: list[CandidateTrialPayload] | list[SeedAvgTrialPayload],
) -> list[CandidateTrialPayload] | list[SeedAvgTrialPayload]:
    """Drop residual stacks from payloads (shared via worker initializer)."""
    return [replace(p, residual=None) for p in payloads]  # type: ignore[return-value]
