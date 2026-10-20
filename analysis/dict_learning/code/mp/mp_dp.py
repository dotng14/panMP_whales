"""Exact matching pursuit by dynamic programming (non-overlapping placements).

Greedy Mallat picks the single best (atom, shift) per step, so an early pick can
block a pair of later placements that together explain more. When placements may
not overlap, that trade-off is avoidable: disjoint spans make the energy drops
**additive**, because deflating one placement never touches another's patch. The
total explained energy of a placement set is then

    sum_i delta(a_i, t_i)

and minimizing residual energy becomes maximum-weight selection of
non-overlapping intervals -- weighted interval scheduling, solvable exactly by a
DP over time bins in ``O(T * K)`` per window.

Sparsity replaces ``eps_step_fraction``: each placement pays ``lambda``, set to
``lambda_fraction`` times the best single drop in that window. Without a penalty
the optimum tiles the window wherever a placement helps at all.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from ..core.objects import Dictionary, Matching, Window
from .mp_parallel import resolve_mp_workers
from .mp_step import (
    IntervalMap,
    MpLoopResult,
    _close_enough_mask,
    _deflate_matching,
    _dictionary_arrays,
    _sliding_patch_energy,
    _valid_shift_mask,
    copy_residual_stack,
    residual_energy,
)
from .score_shifts import score_shifts

# One candidate atom on one window: (template_id, basis, width, delta, amp, coeffs, feasible)
_Candidate = tuple[int, np.ndarray, int, np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray]


def _atom_candidate(
    template_id: int,
    basis: np.ndarray,
    resid: np.ndarray,
    *,
    normalize: bool,
    occupied_intervals: list[tuple[int, int]] | None,
    max_relative_distance: float | None,
    gpu_id: int,
) -> _Candidate | None:
    """Energy drop and feasibility for one atom at every shift on one window.

    ``delta`` is the drop deflation would actually realize with the amplitude /
    coefficients matching pursuit would use, so DP optimizes the same quantity
    greedy does -- not a proxy.
    """
    k = int(basis.shape[0])
    width = int(basis.shape[2])
    n_shifts = int(resid.shape[1]) - width + 1
    if n_shifts <= 0:
        return None

    feasible = _valid_shift_mask(n_shifts, width, occupied_intervals)
    if not np.any(feasible):
        return None

    amp: np.ndarray | None = None
    coeffs: np.ndarray | None = None

    if k == 1:
        tdata = basis[0]
        tmpl_energy = float(np.sum(tdata * tdata))
        if tmpl_energy <= 0.0:
            return None
        scores = np.asarray(
            score_shifts(tdata, resid, normalize=normalize, gpu_id=gpu_id),
            dtype=np.float64,
        )
        if scores.size == 0:
            return None
        amp = np.array(scores, copy=True)
        # ``scores`` is <x,T>/||T||^2 when normalized, else the raw inner product.
        inner = scores * tmpl_energy if normalize else scores
        # Drop from x -> x - amp*T, exactly what _deflate_matching applies.
        delta = 2.0 * amp * inner - amp * amp * tmpl_energy
        explained = (inner * inner) / tmpl_energy
    else:
        maps = []
        for mode in range(k):
            scores = np.asarray(
                score_shifts(basis[mode], resid, normalize=False, gpu_id=gpu_id),
                dtype=np.float64,
            )
            if scores.size == 0:
                return None
            maps.append(scores)
        coeffs = np.stack(maps, axis=0)  # (k, n_shifts)
        explained = np.sum(coeffs * coeffs, axis=0)
        # Gram term is identity for an orthonormal basis; keep it general so the
        # drop stays exact for bases that drifted from orthonormality.
        flat = basis.reshape(k, -1)
        gram = flat @ flat.T
        quad = np.einsum("mn,mt,nt->t", gram, coeffs, coeffs, optimize=True)
        delta = 2.0 * explained - quad

    if max_relative_distance is not None and float(np.sum(basis * basis)) > 0.0:
        patch_energy = _sliding_patch_energy(resid, width, n_shifts)
        feasible = feasible & _close_enough_mask(
            explained, patch_energy, float(max_relative_distance)
        )

    feasible = feasible & (delta > 0.0)
    if not np.any(feasible):
        return None
    return template_id, basis, width, delta, amp, coeffs, feasible


def _solve_interval_dp(
    n_bins: int,
    candidates: Sequence[_Candidate],
    weights: Sequence[np.ndarray],
    feasible: Sequence[np.ndarray],
) -> list[tuple[int, int]]:
    """Maximum-weight non-overlapping placements as ``(candidate_index, t_start)``.

    ``f[t]`` is the best total weight using only bins ``[0, t)``; a placement of
    width ``w`` starting at ``s`` contributes ``f[s] + W[s]`` to ``f[s + w]``.
    """
    f = np.zeros(n_bins + 1, dtype=np.float64)
    choice: list[tuple[int, int] | None] = [None] * (n_bins + 1)

    for t in range(1, n_bins + 1):
        best = f[t - 1]
        picked: tuple[int, int] | None = None
        for idx, cand in enumerate(candidates):
            width = cand[2]
            s = t - width
            if s < 0 or s >= feasible[idx].size or not feasible[idx][s]:
                continue
            value = f[s] + weights[idx][s]
            if value > best:
                best = value
                picked = (idx, s)
        f[t] = best
        choice[t] = picked

    out: list[tuple[int, int]] = []
    t = n_bins
    while t > 0:
        picked = choice[t]
        if picked is None:
            t -= 1
            continue
        idx, s = picked
        out.append((idx, s))
        t = s
    out.reverse()
    return out


def _solve_interval_dp_capped(
    n_bins: int,
    candidates: Sequence[_Candidate],
    weights: Sequence[np.ndarray],
    feasible: Sequence[np.ndarray],
    max_repeats: int,
) -> list[tuple[int, int]]:
    """Same DP with an exact cap on the number of placements.

    ``f[t, r]`` is the best weight using bins ``[0, t)`` with at most ``r``
    placements, so it is monotone in ``r`` by construction.
    """
    cap = int(max_repeats)
    f = np.zeros((n_bins + 1, cap + 1), dtype=np.float64)
    choice: dict[tuple[int, int], tuple[int, int]] = {}

    for t in range(1, n_bins + 1):
        f[t] = f[t - 1]
        for idx, cand in enumerate(candidates):
            width = cand[2]
            s = t - width
            if s < 0 or s >= feasible[idx].size or not feasible[idx][s]:
                continue
            gain = weights[idx][s]
            for r in range(1, cap + 1):
                value = f[s, r - 1] + gain
                if value > f[t, r]:
                    f[t, r] = value
                    choice[(t, r)] = (idx, s)

    out: list[tuple[int, int]] = []
    t, r = n_bins, cap
    while t > 0 and r > 0:
        picked = choice.get((t, r))
        if picked is None or f[t, r] <= f[t - 1, r] + 1e-15:
            t -= 1
            continue
        idx, s = picked
        out.append((idx, s))
        t, r = s, r - 1
    out.reverse()
    return out


def _dp_one_window(
    window_id: str,
    residual: np.ndarray,
    atoms: list[tuple[int, np.ndarray, int]],
    *,
    lambda_fraction: float,
    max_repeats: int | None,
    normalize: bool,
    occupied_intervals: list[tuple[int, int]] | None,
    max_relative_distance: float | None,
    gpu_id: int = 0,
    copy_residual: bool = True,
) -> tuple[str, np.ndarray, list[Matching], bool]:
    """Optimal placements on one window; returns a picklable tuple for pools."""
    if copy_residual:
        resid = np.array(residual, dtype=np.float64, copy=True)
    else:
        resid = np.asarray(residual, dtype=np.float64)

    candidates: list[_Candidate] = []
    for template_id, basis, _width in atoms:
        cand = _atom_candidate(
            int(template_id),
            basis,
            resid,
            normalize=normalize,
            occupied_intervals=occupied_intervals,
            max_relative_distance=max_relative_distance,
            gpu_id=gpu_id,
        )
        if cand is not None:
            candidates.append(cand)
    if not candidates:
        return window_id, resid, [], False

    best_delta = max(
        float(np.max(cand[3][cand[6]])) for cand in candidates if np.any(cand[6])
    )
    if best_delta <= 0.0:
        return window_id, resid, [], False
    penalty = float(lambda_fraction) * best_delta

    weights: list[np.ndarray] = []
    feasible: list[np.ndarray] = []
    for cand in candidates:
        w = cand[3] - penalty
        weights.append(w)
        feasible.append(cand[6] & (w > 0.0))
    if not any(np.any(mask) for mask in feasible):
        return window_id, resid, [], False

    n_bins = int(resid.shape[1])
    if max_repeats is None:
        picks = _solve_interval_dp(n_bins, candidates, weights, feasible)
    else:
        picks = _solve_interval_dp_capped(
            n_bins, candidates, weights, feasible, int(max_repeats)
        )

    matchings: list[Matching] = []
    for idx, t_start in picks:
        template_id, basis, _width, _delta, amp, coeffs, _mask = candidates[idx]
        if coeffs is None:
            matching = Matching(
                template_id=int(template_id),
                window_id=window_id,
                t_start=int(t_start),
                amplitude=float(amp[t_start]),
            )
        else:
            matching = Matching(
                template_id=int(template_id),
                window_id=window_id,
                t_start=int(t_start),
                coeffs=np.array(coeffs[:, t_start], dtype=np.float64, copy=True),
            )
        if not _deflate_matching(resid, matching, basis):
            continue
        matchings.append(matching)

    capped = max_repeats is not None and len(matchings) >= int(max_repeats)
    return window_id, resid, matchings, capped


def _dp_window_job(payload: tuple) -> tuple[str, np.ndarray, list[Matching], bool]:
    (
        window_id,
        residual,
        atoms,
        lambda_fraction,
        max_repeats,
        normalize,
        occupied_intervals,
        max_relative_distance,
        gpu_id,
        copy_residual,
    ) = payload
    return _dp_one_window(
        window_id,
        residual,
        atoms,
        lambda_fraction=lambda_fraction,
        max_repeats=max_repeats,
        normalize=normalize,
        occupied_intervals=occupied_intervals,
        max_relative_distance=max_relative_distance,
        gpu_id=int(gpu_id),
        copy_residual=bool(copy_residual),
    )


def run_mp_dp_loop(
    residual_stack: dict[str, np.ndarray] | list[Window],
    dictionary: Dictionary,
    *,
    lambda_fraction: float = 0.2,
    max_repeats: int | None = None,
    normalize: bool = True,
    occupied_intervals: IntervalMap | None = None,
    max_relative_distance: float | None = None,
    show_progress: bool | None = None,
    progress_desc: str = "mp-dp",
    n_workers: int | None = None,
) -> MpLoopResult:
    """Exact non-overlapping matching pursuit by DP, one window at a time.

    Placements never overlap, which is what makes the energy drops additive and
    the DP exact; there is no ``disallow_overlap`` switch. Spans in
    ``occupied_intervals`` stay off limits, as in :func:`run_mp_loop`.

    ``lambda_fraction`` on ``[0, 1)`` sets the per-placement penalty as a
    fraction of the best single energy drop in that window: ``0`` keeps every
    placement that helps at all, larger values keep only strong ones.
    ``max_repeats`` caps placements per window and stays exact under the cap.
    """
    if not (0.0 <= float(lambda_fraction) < 1.0):
        raise ValueError(
            f"lambda_fraction must be in [0, 1), got {lambda_fraction}"
        )
    if max_relative_distance is not None and not (
        0.0 <= float(max_relative_distance) <= 1.0
    ):
        raise ValueError(
            "max_relative_distance must be in [0, 1] when set, got "
            f"{max_relative_distance}"
        )
    if max_repeats is not None and int(max_repeats) < 1:
        raise ValueError(f"max_repeats must be >= 1 when set, got {max_repeats}")
    if len(dictionary) == 0:
        raise ValueError("dictionary must contain at least one atom")

    from .progress import progress_enabled, tqdm
    from .score_shifts import cupy_available, gpu_device_count, numba_available

    resid = copy_residual_stack(residual_stack)
    occupied: IntervalMap = {
        window_id: list(intervals)
        for window_id, intervals in (occupied_intervals or {}).items()
    }
    atoms = _dictionary_arrays(dictionary)
    window_ids = sorted(resid.keys())
    e0 = residual_energy(resid)

    n_gpus = gpu_device_count() if cupy_available() else 0
    jobs = [
        (
            wid,
            resid[wid],
            atoms,
            float(lambda_fraction),
            max_repeats,
            normalize,
            list(occupied.get(wid, [])),
            max_relative_distance,
            (i % n_gpus) if n_gpus else 0,
            False,  # the stack was already copied above
        )
        for i, wid in enumerate(window_ids)
    ]

    workers = resolve_mp_workers(n_workers)
    prefer_threads = numba_available() or cupy_available()
    results: list[tuple[str, np.ndarray, list[Matching], bool]] = []
    pbar = tqdm(
        total=len(jobs),
        desc=progress_desc,
        unit="win",
        leave=False,
        disable=not progress_enabled(show_progress),
        dynamic_ncols=True,
    )
    try:
        if workers <= 1 or len(jobs) <= 1:
            for job in jobs:
                results.append(_dp_window_job(job))
                pbar.update(1)
        else:
            from .mp_parallel import map_parallel

            results = map_parallel(
                _dp_window_job,
                jobs,
                n_workers=workers,
                prefer_threads=prefer_threads,
            )
            pbar.update(len(jobs))
    finally:
        pbar.close()

    out_resid: dict[str, np.ndarray] = {}
    all_matchings: list[Matching] = []
    any_capped = False
    for window_id, arr, matchings, capped in results:
        out_resid[window_id] = arr
        all_matchings.extend(matchings)
        any_capped = any_capped or capped

    all_matchings.sort(key=lambda m: (m.window_id, m.t_start, m.template_id))
    e1 = residual_energy(out_resid)

    return MpLoopResult(
        residual_stack=out_resid,
        matchings=all_matchings,
        energies=[e0, e1] if all_matchings else [e0],
        stopped_reason="max_repeats" if any_capped else "dp_optimal",
    )


__all__ = ["run_mp_dp_loop"]
