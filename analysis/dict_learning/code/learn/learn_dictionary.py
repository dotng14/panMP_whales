"""Multi-template dictionary learning: residual-deflation loop over seed_from_GPL rounds."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..io import Detection, load_gpl_inputs
from ..mp import residual_energy, single_template_dictionary
from ..core import TEMPLATE_METHOD
from ..core import Dictionary, Matching, Segment, Template, Window
from .seed_from_gpl import Seed, detections_to_segments, mean_center
from ..core import (
    coverage_from_matchings as _coverage_from_matchings_util,
    crop_candidates_from_residual,
    filter_candidates_by_windows,
    select_low_coverage_seed_windows,
    union_interval_length as _union_interval_length,
)


class DictionaryLearningError(Exception):
    """Base class for recoverable dictionary-learning failures.

    Raised when learning/refinement cannot proceed for algorithmic reasons
    (e.g. an atom lost all its matches), as opposed to programming errors.
    Callers can catch this to keep partial results instead of crashing.
    """


class NoMatchingPatchesError(DictionaryLearningError, ValueError):
    """An atom has no valid matched patches to average.

    Typically raised mid-refinement when a joint MP pass assigns zero matches
    to one template, so the next averaging step has nothing to average.
    """

    def __init__(
        self,
        template_id: int,
        *,
        iteration: int | None = None,
        detail: str | None = None,
    ) -> None:
        self.template_id = template_id
        self.iteration = iteration
        where = f"iteration {iteration}: " if iteration is not None else ""
        message = f"{where}no valid matching patches for template_id={template_id}"
        if detail:
            message = f"{message} ({detail})"
        super().__init__(message)


def _matching_key(matching: Matching) -> tuple[str, int, int]:
    return (matching.window_id, int(matching.template_id), int(matching.t_start))


@dataclass
class LearningRoundLog:
    """Structured telemetry for one completed outer seed→average→MP round."""

    round_index: int
    template_id: int
    source_segment_id: str | None
    n_candidates: int
    seed_n_matchings: int
    seed_energy_before: float
    seed_energy_after: float
    seed_delta_energy: float
    averaged_n_matchings: int
    energy_before: float
    energy_after: float
    delta_energy: float
    cumulative_delta_energy: float
    n_windows_total: int
    n_windows_matched_round: int
    n_windows_matched_cumulative: int
    window_coverage_fraction: float
    n_time_bins_total: int
    n_time_bins_covered_cumulative: int
    time_coverage_fraction: float
    template_width: int | None = None
    seed_window_policy: str | None = None
    seed_window_id: str | None = None
    n_candidates_global: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary_line(self) -> str:
        seed_note = ""
        if self.seed_window_policy and self.seed_window_policy != "global":
            seed_note = (
                f" seed={self.seed_window_policy}"
                f"@{self.seed_window_id or '?'}"
                f"({self.n_candidates}/{self.n_candidates_global or self.n_candidates} cand)"
            )
        return (
            f"round {self.round_index}: "
            f"seed_matches={self.seed_n_matchings} "
            f"avg_matches={self.averaged_n_matchings} "
            f"dE={self.delta_energy:.6g} "
            f"cum_dE={self.cumulative_delta_energy:.6g} "
            f"windows={self.n_windows_matched_cumulative}/{self.n_windows_total} "
            f"({100.0 * self.window_coverage_fraction:.1f}%) "
            f"time_bins={self.n_time_bins_covered_cumulative}/{self.n_time_bins_total} "
            f"({100.0 * self.time_coverage_fraction:.1f}%)"
            f"{seed_note}"
        )


def _coverage_from_matchings(
    residual: dict[str, np.ndarray],
    matchings: Sequence[Matching],
    *,
    template_width: int,
) -> tuple[int, int, int, int]:
    """Return (n_windows_matched, n_windows_total, n_bins_covered, n_bins_total)."""
    return _coverage_from_matchings_util(
        residual, matchings, template_width=template_width
    )


def _merge_coverage_intervals(
    covered: dict[str, list[tuple[int, int]]],
    matchings: Sequence[Matching],
    *,
    residual: dict[str, np.ndarray],
    template_width: int,
) -> None:
    for matching in matchings:
        wid = matching.window_id
        if wid not in residual:
            continue
        width = residual[wid].shape[1]
        start = max(0, int(matching.t_start))
        end = min(width, start + int(template_width))
        if end > start:
            covered.setdefault(wid, []).append((start, end))


class LearnedDictionary:
    """Templates selected by successive :class:`Seed` rounds on a deflating residual."""

    def __init__(
        self,
        *,
        seeds: list[Seed] | None = None,
        energy_before: float = 0.0,
        energy_after: float = 0.0,
        residual_after: dict[str, np.ndarray] | None = None,
        round_logs: list[LearningRoundLog] | None = None,
        stop_reason: str | None = None,
        tool_run_id: int | None = None,
        output_dictionary_id: int | None = None,
        stage_logs: list[Any] | None = None,
    ) -> None:
        self._seeds = list(seeds or [])
        self._energy_before = energy_before
        self._energy_after = energy_after
        self._residual_after = residual_after
        self._round_logs = list(round_logs or [])
        self._stop_reason = stop_reason
        self._tool_run_id = tool_run_id
        self._output_dictionary_id = output_dictionary_id
        self._stage_logs = list(stage_logs or [])

    def __len__(self) -> int:
        return len(self._seeds)

    def get_seeds(self) -> list[Seed]:
        return list(self._seeds)

    def get_templates(self) -> list[Template]:
        return [
            template
            for seed in self._seeds
            if (template := seed.get_template()) is not None
        ]

    def get_matchings(self) -> list[Matching]:
        out: list[Matching] = []
        for seed in self._seeds:
            out.extend(seed.get_matchings())
        return out

    def get_matchings_for(self, template_id: int) -> list[Matching]:
        return [m for m in self.get_matchings() if m.template_id == template_id]

    def get_energy_before(self) -> float:
        return self._energy_before

    def get_energy_after(self) -> float:
        return self._energy_after

    def get_delta_energy_total(self) -> float:
        return self._energy_before - self._energy_after

    def get_residual_after(self) -> dict[str, np.ndarray] | None:
        if self._residual_after is None:
            return None
        return {k: np.asarray(v) for k, v in self._residual_after.items()}

    def get_round_logs(self) -> list[LearningRoundLog]:
        return list(self._round_logs)

    def get_stop_reason(self) -> str | None:
        return self._stop_reason

    def get_tool_run_id(self) -> int | None:
        return self._tool_run_id

    def get_output_dictionary_id(self) -> int | None:
        return self._output_dictionary_id

    def get_stage_logs(self) -> list[Any]:
        return list(self._stage_logs)


@dataclass
class DictionaryRefinementResult:
    """Result of repeated dictionary averaging followed by multi-atom MP."""

    dictionary: Dictionary
    matchings: list[Matching] = field(default_factory=list)
    residual_after: dict[str, np.ndarray] = field(default_factory=dict)
    energies: list[float] = field(default_factory=list)
    iterations_completed: int = 0
    n_pinned_matchings: int = 0


def refine_dictionary_by_averaging(
    stack: Sequence[Window] | dict[str, np.ndarray],
    dictionary: Dictionary,
    matchings: Sequence[Matching],
    *,
    iter: int = 1,
    eps_step_fraction: float = 0.5,
    max_repeats: int | None = None,
    normalize: bool = True,
    disallow_overlap: bool = False,
    template_method: str = TEMPLATE_METHOD,
    n_components: int = 1,
    max_relative_distance: float | None = None,
    pinned_matchings: Sequence[Matching] | None = None,
    show_progress: bool | None = None,
    mp_mode: str | None = None,
    n_workers: int | None = None,
) -> DictionaryRefinementResult:
    """Repeatedly rebuild every atom's matched patches, then run MP.

    Each iteration composes elementary tools **B** then **A**:

    1. :func:`~.tools.average_templates` on the original stack + current matchings
    2. :func:`~.tools.matching_pursuit` on that same original stack

    ``n_components > 1`` rebuilds each atom as a top-``k`` PCA subspace and
    forces global MP. ``max_relative_distance`` (if set) gates both stages: MP
    only accepts placements close enough to the atom, and only close-enough
    patches contribute to the rebuild.
    Every dictionary atom must have at least one valid input matching in every
    iteration. ``iter=0`` returns a copy of the input dictionary and matchings
    without running averaging or MP.

    ``pinned_matchings`` (typically hand labels) survive every iteration: each
    one is re-fitted to its atom at its own ``t_start``, deflated before MP, and
    its span withheld from MP, so no other atom can claim that patch and a
    pinned atom can never be left without patches to average.
    """
    from ..mp import refit_and_deflate
    from ..mp import range_progress
    from ..tools import average_templates, matching_pursuit

    if iter < 0:
        raise ValueError(f"iter must be >= 0, got {iter}")
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    if len(dictionary) == 0:
        raise ValueError("dictionary must contain at least one atom")

    if isinstance(stack, dict):
        original = {k: np.asarray(v, dtype=np.float64) for k, v in stack.items()}
    else:
        original = {w.window_id: mean_center(w.data) for w in stack}
    if not original:
        raise ValueError("stack must contain at least one window")

    current_dictionary = Dictionary()
    for template in dictionary:
        current_dictionary.add(
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
    current_matchings = list(matchings)
    pinned = list(pinned_matchings or [])
    if pinned:
        seen = {_matching_key(m) for m in current_matchings}
        current_matchings.extend(m for m in pinned if _matching_key(m) not in seen)
    residual_after = {k: np.array(v, copy=True) for k, v in original.items()}
    energies = [residual_energy(residual_after)]
    eff_mode = "global" if n_components > 1 else mp_mode
    eff_normalize = False if n_components > 1 else normalize

    for iteration in range_progress(
        iter, desc="refine iters", show_progress=show_progress, unit="iter"
    ):
        from ..mp import loop_stats

        loop_stats.incr("refine.iter")
        try:
            avg = average_templates(
                original,
                current_dictionary,
                current_matchings,
                method=template_method,
                n_components=n_components,
                max_relative_distance=max_relative_distance,
            )
        except ValueError as exc:
            msg = str(exc)
            tid = None
            for template in current_dictionary:
                if f"template_id {template.template_id}" in msg:
                    tid = template.template_id
                    break
            if tid is None and current_dictionary.order:
                tid = current_dictionary.order[0]
            raise NoMatchingPatchesError(
                tid if tid is not None else -1,
                iteration=iteration,
                detail="MP left this atom without matches on the stack",
            ) from exc

        # Pinned patches are settled against the refreshed atoms first, so MP
        # sees them already explained and is barred from re-claiming their spans.
        mp_stack = original
        pinned_now: list[Matching] = []
        reserved: dict[str, list[tuple[int, int]]] = {}
        if pinned:
            mp_stack = {k: np.array(v, copy=True) for k, v in original.items()}
            pinned_now, reserved = refit_and_deflate(
                mp_stack, avg.dictionary, pinned, normalize=eff_normalize
            )

        mp = matching_pursuit(
            mp_stack,
            avg.dictionary,
            eps_step_fraction=eps_step_fraction,
            max_repeats=max_repeats,
            normalize=eff_normalize,
            disallow_overlap=disallow_overlap,
            occupied_intervals=reserved or None,
            show_progress=show_progress,
            progress_desc="refine mp",
            mp_mode=eff_mode,
            n_workers=n_workers,
            max_relative_distance=max_relative_distance,
        )
        current_dictionary = avg.dictionary
        residual_after = mp.loop.residual_stack
        if pinned_now:
            current_matchings = sorted(
                [*pinned_now, *mp.loop.matchings],
                key=lambda m: (m.window_id, m.t_start, m.template_id),
            )
            energies = [residual_energy(original), *mp.loop.energies]
        else:
            current_matchings = list(mp.loop.matchings)
            energies = list(mp.loop.energies)

    return DictionaryRefinementResult(
        dictionary=current_dictionary,
        matchings=current_matchings,
        residual_after=residual_after,
        energies=energies,
        iterations_completed=iter,
        n_pinned_matchings=len(pinned),
    )


def learn_from_gpl(
    stack: Sequence[Window] | dict[str, np.ndarray],
    detections: Sequence[Detection] | Sequence[Segment],
    *,
    windows_for_units: Sequence[Window] | None = None,
    max_templates: int | None = 5,
    min_seed_delta_fraction: float = 0.1,
    eps_step_fraction: float = 0.5,
    min_repeats: int = 1,
    normalize: bool = True,
    show_progress: bool | None = None,
    mp_mode: str | None = None,
    n_workers: int | None = None,
    seed_window_policy: str = "global",
    seed_window_neighborhood: int = 0,
) -> LearnedDictionary:
    """Learn templates by residual deflation (legacy seed-only path).

    Delegates to :func:`~.tools.build_dictionary_from_detections` (tool **C**).
    """
    from ..tools import build_dictionary_from_detections

    return build_dictionary_from_detections(
        stack,
        detections,
        windows_for_units=windows_for_units,
        max_templates=max_templates,
        min_seed_delta_fraction=min_seed_delta_fraction,
        eps_step_fraction=eps_step_fraction,
        min_repeats=min_repeats,
        normalize=normalize,
        show_progress=show_progress,
        mp_mode=mp_mode,
        n_workers=n_workers,
    )


def learn_from_gpl_paths(
    *,
    window_paths: Sequence[Path | str] | None = None,
    npy_dir: Path | str | None = None,
    csv_paths: Sequence[Path | str] | None = None,
    gpl_output_dir: Path | str | None = None,
    max_templates: int | None = 5,
    min_seed_delta_fraction: float = 0.1,
    eps_step_fraction: float = 0.5,
    min_repeats: int = 1,
    show_progress: bool | None = None,
) -> LearnedDictionary:
    """Run :func:`learn_from_gpl` on windows/detections loaded from disk."""
    windows, detections = load_gpl_inputs(
        window_paths=window_paths,
        npy_dir=npy_dir,
        csv_paths=csv_paths,
        gpl_output_dir=gpl_output_dir,
    )
    return learn_from_gpl(
        windows,
        detections,
        max_templates=max_templates,
        min_seed_delta_fraction=min_seed_delta_fraction,
        eps_step_fraction=eps_step_fraction,
        min_repeats=min_repeats,
        show_progress=show_progress,
    )


def refine_seed_by_averaging(
    stack: Sequence[Window] | dict[str, np.ndarray],
    seed: Seed,
    *,
    eps_step_fraction: float = 0.5,
    max_repeats: int | None = None,
    normalize: bool = True,
    disallow_overlap: bool = False,
    occupied_intervals: dict[str, list[tuple[int, int]]] | None = None,
    template_method: str = TEMPLATE_METHOD,
    show_progress: bool | None = None,
    mp_mode: str | None = None,
    n_workers: int | None = None,
    max_relative_distance: float | None = None,
) -> Seed:
    """Average MP-matched patches from the original signal, then re-run MP on it.

    Composes tools **B** then **A** for a single-atom dictionary.
    ``template_method`` selects the atom-rebuild rule: ``"pca"`` (leading
    eigenvector, default) or ``"average"`` (mean patch).
    ``max_relative_distance`` gates both the rebuild and the re-MP.
    """
    from ..tools import average_templates, matching_pursuit

    template = seed.get_template()
    matchings = seed.get_matchings()
    if template is None or not matchings:
        raise ValueError("seed must be selected and have at least one matching")

    if isinstance(stack, dict):
        original = {k: np.asarray(v, dtype=np.float64) for k, v in stack.items()}
    else:
        original = {w.window_id: mean_center(w.data) for w in stack}

    single = single_template_dictionary(template)
    avg = average_templates(
        original,
        single,
        matchings,
        method=template_method,
        max_relative_distance=max_relative_distance,
    )
    avg_template = next(iter(avg.dictionary))
    e0 = residual_energy(original)
    mp = matching_pursuit(
        original,
        single_template_dictionary(avg_template),
        eps_step_fraction=eps_step_fraction,
        max_repeats=max_repeats,
        normalize=normalize,
        disallow_overlap=disallow_overlap,
        occupied_intervals=occupied_intervals,
        show_progress=show_progress,
        progress_desc="avg mp",
        mp_mode=mp_mode,
        n_workers=n_workers,
        max_relative_distance=max_relative_distance,
    )
    return Seed(
        template=avg_template,
        matchings=mp.loop.matchings,
        residual_after=mp.loop.residual_stack,
        energy_before=e0,
        energy_after=residual_energy(mp.loop.residual_stack),
        n_candidates_tried=1,
        selected_segment_id=template.source_segment_id,
        candidates=seed.get_candidates(),
        mp_energies=list(mp.loop.energies),
    )



def save_round_template(
    out_dir: Path | str,
    seed: Seed,
    *,
    round_index: int,
) -> Path:
    """Write one outer-loop averaged template (+ matchings) after a round finishes.

    Layout::

        out_dir/
          round_000/
            template.npy
            matchings.json
          manifest.json   # updated after each round
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmpl = seed.get_template()
    if tmpl is None:
        raise ValueError("seed has no template to save")

    round_dir = out_dir / f"round_{round_index:03d}"
    round_dir.mkdir(parents=True, exist_ok=True)
    np.save(round_dir / "template.npy", tmpl.data)

    matchings = seed.get_matchings()
    payload = {
        "round_index": round_index,
        "template_id": tmpl.template_id,
        "source_segment_id": tmpl.source_segment_id,
        "shape": list(tmpl.data.shape),
        "energy_before": seed.get_energy_before(),
        "energy_after": seed.get_energy_after(),
        "delta_energy": seed.get_delta_energy_total(),
        "n_matchings": len(matchings),
        "matchings": [
            {
                "template_id": m.template_id,
                "window_id": m.window_id,
                "t_start": m.t_start,
                "amplitude": float(m.amplitude),
                **(
                    {"coeffs": np.asarray(m.coeffs, dtype=np.float64).tolist()}
                    if np.asarray(m.coeffs).size > 1
                    else {}
                ),
            }
            for m in matchings
        ],
    }
    (round_dir / "matchings.json").write_text(json.dumps(payload, indent=2))

    manifest_path = out_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
    else:
        manifest = {"rounds": []}
    # Replace entry for this round_index if re-run, else append.
    rounds = [r for r in manifest.get("rounds", []) if r.get("round_index") != round_index]
    rounds.append(
        {
            "round_index": round_index,
            "dir": round_dir.name,
            "template_id": tmpl.template_id,
            "shape": list(tmpl.data.shape),
            "source_segment_id": tmpl.source_segment_id,
            "delta_energy": seed.get_delta_energy_total(),
            "n_matchings": len(matchings),
            "template_file": f"{round_dir.name}/template.npy",
            "matchings_file": f"{round_dir.name}/matchings.json",
        }
    )
    rounds.sort(key=lambda r: r["round_index"])
    manifest["rounds"] = rounds
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return round_dir


def learn_averaged_templates_from_gpl(
    stack: Sequence[Window] | dict[str, np.ndarray],
    detections: Sequence[Detection] | Sequence[Segment],
    *,
    windows_for_units: Sequence[Window] | None = None,
    n_windows: int | None = None,
    window_offset: int = 0,
    n_rounds: int | None = None,
    eps_step_fraction: float = 0.5,
    min_repeats: int = 3,
    max_repeats: int | None = None,
    normalize: bool = True,
    disallow_overlap: bool = False,
    occupied_intervals: dict[str, list[tuple[int, int]]] | None = None,
    template_id_start: int = 0,
    min_template_width: int | None = None,
    max_template_width: int | None = None,
    template_method: str = TEMPLATE_METHOD,
    n_components: int = 1,
    template_out_dir: Path | str | None = None,
    show_progress: bool | None = None,
    on_round_complete: Callable[[LearningRoundLog], None] | None = None,
    mp_mode: str | None = None,
    n_workers: int | None = None,
    seed_window_policy: str = "global",
    seed_window_neighborhood: int = 0,
    max_relative_distance: float | None = None,
    max_candidates: int | None = None,
) -> LearnedDictionary:
    """Run repeated seed->average->MP rounds on a deflating residual stack.

    Orchestrates elementary tools **A** (:func:`~.tools.matching_pursuit`) and
    **B** (:func:`~.tools.average_templates` via :func:`refine_seed_by_averaging`).
    Averaging uses patches from the **residual** that seeded the trial (current
    peel stack), not a separate original copy.

    Each round trials GPL candidates (optionally restricted by
    ``seed_window_policy``):

    1) seed Mallat MP (**A**) on the current residual,
    2) if the seed has ``>= min_repeats`` matchings, average those patches (**B**),
    3) re-run Mallat (**A**) with the averaged template,
    4) keep candidates whose averaged MP also has ``>= min_repeats`` matchings,
    5) accept the qualifying averaged candidate with the lowest residual energy.

    ``seed_window_policy``:
      - ``"global"`` (default): trial every width-eligible candidate (legacy).
      - ``"low_coverage"``: seed only from the least-covered window that still
        has residual GPL candidates (plus ``seed_window_neighborhood`` adjacent
        windows). Matching still runs on the **full** residual stack. If no
        candidate in that pool qualifies, the round falls back to the global
        candidate set once before stopping.

    When ``n_components > 1``, step (5) rebuilds the winner as a top-``k`` PCA
    subspace from the seed matchings and re-places with subspace MP (global).
    ``max_relative_distance`` (if set) gates every stage of the round: MP only
    accepts placements leaving at most that fraction of the patch energy
    unexplained, and only such patches feed the rebuilds (``None`` = no gate).

    Rounds stop when no candidate qualifies (``below_min_repeats``), when no
    candidates remain, or when the optional ``n_rounds`` safety cap is hit.
    Mallat peeling itself is controlled by ``eps_step_fraction`` and optional
    ``max_repeats`` (cap on accepted matchings per Mallat run; per window in
    ``per_window`` mode).

    If ``disallow_overlap`` is true, no accepted matching may intersect an
    earlier matching in the same window, either within or across outer rounds.
    ``occupied_intervals`` seeds that coverage map (e.g. detections already
    claimed by a prior dictionary in an incremental stage).

    ``template_method`` selects how each atom is rebuilt from its matched
    patches in step (2): ``"pca"`` (leading principal component / largest
    eigenvector, default) or ``"average"`` (mean patch).
    Ignored when ``n_components > 1`` (PCA subspace rebuild).

    ``template_id_start`` sets the first new atom id (default ``0``); later
    rounds use ``template_id_start + 1``, …

    ``max_candidates`` (if set) trials only that many candidates per round,
    taking them in descending residual energy. Every trial is a full seed MP
    plus an averaging re-MP over the whole stack, so the round cost is linear in
    the pool size; the winner comes from the high-energy end of the ordering in
    practice (rank 2 of 222 on the measured SOCAL round, where every prefix down
    to the top 10 chose the same atom). ``None`` trials the whole pool.

    ``on_round_complete`` is called after each successful round with a
    :class:`LearningRoundLog` (seed matches, energy drop, window/time coverage).
    """
    from ..mp import (
        allocate_nested_workers,
        map_parallel,
        prefer_candidate_threads,
        resolve_mp_mode,
        resolve_mp_workers,
    )
    from ..mp import run_mp_loop, single_template_dictionary
    from ..mp import (
        SeedAvgTrialPayload,
        init_trial_worker_shared,
        payload_by_segment_id,
        rematerialize_seed_avg_trial,
        run_seed_avg_trial,
        slim_payloads_for_process_pool,
        trial_sort_key_energy_then_segment,
    )
    from ..mp import iter_progress, range_progress
    from ..tools import rebuild_template_from_matchings

    if n_rounds is not None and n_rounds < 1:
        raise ValueError(f"n_rounds must be >= 1 when set, got {n_rounds}")
    if n_windows is not None and n_windows < 1:
        raise ValueError(f"n_windows must be >= 1, got {n_windows}")
    if window_offset < 0:
        raise ValueError(f"window_offset must be >= 0, got {window_offset}")
    if eps_step_fraction <= 0.0:
        raise ValueError(f"eps_step_fraction must be > 0, got {eps_step_fraction}")
    if min_repeats < 1:
        raise ValueError(f"min_repeats must be >= 1, got {min_repeats}")
    if max_repeats is not None and int(max_repeats) < 1:
        raise ValueError(f"max_repeats must be >= 1 when set, got {max_repeats}")
    if max_repeats is not None and int(max_repeats) < int(min_repeats):
        raise ValueError(
            f"max_repeats ({max_repeats}) must be >= min_repeats ({min_repeats})"
        )
    if template_id_start < 0:
        raise ValueError(f"template_id_start must be >= 0, got {template_id_start}")
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    if max_relative_distance is not None and not (
        0.0 <= float(max_relative_distance) <= 1.0
    ):
        raise ValueError(
            f"max_relative_distance must be in [0, 1] or None, "
            f"got {max_relative_distance}"
        )
    if seed_window_policy not in ("global", "low_coverage"):
        raise ValueError(
            f"seed_window_policy must be 'global' or 'low_coverage', "
            f"got {seed_window_policy!r}"
        )
    if seed_window_neighborhood < 0:
        raise ValueError(
            f"seed_window_neighborhood must be >= 0, got {seed_window_neighborhood}"
        )
    if max_candidates is not None and int(max_candidates) < 1:
        raise ValueError(
            f"max_candidates must be >= 1 when set, got {max_candidates}"
        )

    mode = resolve_mp_mode(mp_mode)
    cand_workers = resolve_mp_workers(n_workers)

    # Mean-center raw Windows once. Residual dicts are used as-is (not re-centered).
    if isinstance(stack, dict):
        if n_windows is not None or window_offset:
            raise ValueError("n_windows/window_offset only supported when stack is a sequence of Window")
        residual = {k: np.asarray(v, dtype=np.float64) for k, v in stack.items()}
        unit_windows = list(windows_for_units or [])
    else:
        all_windows = list(stack)
        end = None if n_windows is None else window_offset + n_windows
        selected_windows = all_windows[window_offset:end]
        residual = {w.window_id: mean_center(w.data) for w in selected_windows}
        if windows_for_units is None:
            unit_windows = selected_windows
        else:
            unit_windows = list(windows_for_units)[window_offset:end]

    energy_before = residual_energy(residual)
    n_windows_total = len(residual)
    n_time_bins_total = sum(int(arr.shape[1]) for arr in residual.values())

    if detections and isinstance(detections[0], Segment):
        base_segments = list(detections)  # type: ignore[arg-type]
    elif detections:
        if not unit_windows:
            raise ValueError("windows_for_units (or Window stack) required with Detection inputs")
        base_segments = detections_to_segments(unit_windows, detections)  # type: ignore[arg-type]
    else:
        base_segments = []

    if min_template_width is not None or max_template_width is not None:
        lo = min_template_width if min_template_width is not None else 0
        hi = max_template_width if max_template_width is not None else 10**9
        base_segments = [
            s for s in base_segments if lo <= int(s.data.shape[1]) <= hi
        ]

    out_dir = Path(template_out_dir) if template_out_dir is not None else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    rounds: list[Seed] = []
    round_logs: list[LearningRoundLog] = []
    covered_intervals: dict[str, list[tuple[int, int]]] = {
        window_id: list(intervals)
        for window_id, intervals in (occupied_intervals or {}).items()
    }
    stop_reason = "completed"

    for round_offset in range_progress(
        n_rounds, desc="avg rounds", show_progress=show_progress, unit="round"
    ):
        from ..mp import loop_stats

        loop_stats.incr("learn.outer_round")
        template_id = template_id_start + round_offset
        candidates = _crop_candidates_from_residual(base_segments, residual)
        if not candidates:
            stop_reason = "no_candidates"
            break

        round_energy_before = residual_energy(residual)
        ordered_global = sorted(candidates, key=lambda s: s.energy, reverse=True)
        if max_candidates is not None:
            ordered_global = ordered_global[: int(max_candidates)]
        n_candidates_global = len(ordered_global)
        seed_anchor: str | None = None
        used_policy = "global"
        ordered = ordered_global
        if seed_window_policy == "low_coverage":
            seed_anchor, seed_windows = select_low_coverage_seed_windows(
                residual,
                ordered_global,
                covered_intervals,
                window_order=list(residual.keys()),
                neighborhood=seed_window_neighborhood,
            )
            filtered = filter_candidates_by_windows(ordered_global, seed_windows)
            if filtered:
                ordered = filtered
                used_policy = "low_coverage"

        def _trial_pool(pool_segments):
            loop_stats.record("learn.candidates_per_round", len(pool_segments))
            outer_workers, inner_workers = allocate_nested_workers(
                len(pool_segments), total_workers=cand_workers
            )
            pool_payloads = [
                SeedAvgTrialPayload(
                    segment_id=seg.segment_id,
                    template_id=template_id,
                    template_data=np.array(seg.data, copy=True),
                    residual=residual,
                    eps_step_fraction=eps_step_fraction,
                    normalize=normalize,
                    disallow_overlap=disallow_overlap,
                    occupied_intervals=covered_intervals,
                    min_repeats=min_repeats,
                    max_repeats=max_repeats,
                    mp_mode=mode,
                    n_workers=inner_workers,
                    template_method=template_method,
                    max_relative_distance=max_relative_distance,
                )
                for seg in pool_segments
            ]
            if outer_workers <= 1 or len(pool_payloads) <= 1:
                results = []
                for payload in iter_progress(
                    pool_payloads,
                    desc=f"seed+avg (t{template_id})",
                    total=len(pool_payloads),
                    leave=False,
                    show_progress=show_progress,
                    unit="cand",
                ):
                    loop_stats.incr("learn.candidate_trial")
                    results.append(run_seed_avg_trial(payload))
            else:
                loop_stats.incr("learn.candidate_trial", len(pool_payloads))
                # Threads: share residual by reference (MP copies internally); no
                # per-round spawn/pickle. Processes: slim payloads + initializer.
                use_threads = prefer_candidate_threads()
                if use_threads:
                    results = map_parallel(
                        run_seed_avg_trial,
                        pool_payloads,
                        n_workers=outer_workers,
                        prefer_threads=True,
                    )
                else:
                    results = map_parallel(
                        run_seed_avg_trial,
                        slim_payloads_for_process_pool(pool_payloads),
                        n_workers=outer_workers,
                        prefer_threads=False,
                        initializer=init_trial_worker_shared,
                        initargs=(residual, covered_intervals),
                    )
            winner_tr = None
            winner_key = None
            for tr in results:
                if not tr.qualifies:
                    continue
                key = trial_sort_key_energy_then_segment(tr)
                if winner_tr is None or key < winner_key:
                    winner_tr = tr
                    winner_key = key
            return winner_tr, pool_payloads

        best_tr, payloads = _trial_pool(ordered)
        # If the low-coverage pool finds no qualifier, fall back once to global.
        if (
            best_tr is None
            and used_policy == "low_coverage"
            and len(ordered) < n_candidates_global
        ):
            used_policy = "global_fallback"
            seed_anchor = None
            best_tr, payloads = _trial_pool(ordered_global)

        n_tried = len(payloads)
        if best_tr is None:
            stop_reason = "below_min_repeats"
            break

        # Batch trials omit refined residuals; rematerialize only the winner.
        winner = rematerialize_seed_avg_trial(
            payload_by_segment_id(payloads, best_tr.segment_id)  # type: ignore[arg-type]
        )
        seed_tmpl = Template(
            template_id=template_id,
            data=np.array(winner.seed_template_data, copy=True),
            source_segment_id=winner.segment_id,
        )
        seed = Seed(
            template=seed_tmpl,
            matchings=winner.seed_matchings,
            residual_after=None,
            energy_before=round_energy_before,
            energy_after=winner.seed_energy_after,
            n_candidates_tried=1,
            selected_segment_id=winner.segment_id,
            candidates=candidates,
            mp_energies=list(winner.seed_energies),
        )

        if n_components > 1:
            # Rank-1 seed select, then subspace rebuild + place on the round residual.
            subspace_tmpl = rebuild_template_from_matchings(
                residual,
                seed_tmpl,
                winner.seed_matchings,
                n_components=n_components,
                method="pca",
                max_relative_distance=max_relative_distance,
            )
            subspace_mp = run_mp_loop(
                residual,
                single_template_dictionary(subspace_tmpl),
                eps_step_fraction=eps_step_fraction,
                max_repeats=max_repeats,
                normalize=False,
                disallow_overlap=disallow_overlap,
                occupied_intervals=covered_intervals if disallow_overlap else None,
                show_progress=show_progress,
                progress_desc=f"subspace mp (t{template_id})",
                mp_mode="global",
                n_workers=1,
                max_relative_distance=max_relative_distance,
            )
            if len(subspace_mp.matchings) < min_repeats:
                stop_reason = "below_min_repeats"
                break
            refined_tmpl = subspace_tmpl
            refined = Seed(
                template=refined_tmpl,
                matchings=list(subspace_mp.matchings),
                residual_after=subspace_mp.residual_stack,
                energy_before=round_energy_before,
                energy_after=(
                    subspace_mp.energies[-1]
                    if subspace_mp.energies
                    else residual_energy(subspace_mp.residual_stack)
                ),
                n_candidates_tried=1,
                selected_segment_id=winner.segment_id,
                candidates=candidates,
                mp_energies=list(subspace_mp.energies),
            )
        else:
            refined_tmpl = Template(
                template_id=template_id,
                data=np.array(winner.refined_template_data, copy=True),
                source_segment_id=winner.segment_id,
            )
            refined = Seed(
                template=refined_tmpl,
                matchings=winner.refined_matchings,
                residual_after=winner.refined_residual,
                energy_before=round_energy_before,
                energy_after=winner.refined_energy_after,
                n_candidates_tried=1,
                selected_segment_id=winner.segment_id,
                candidates=candidates,
                mp_energies=list(winner.refined_energies),
            )

        tmpl = refined.get_template()
        assert tmpl is not None
        refined_matchings = refined.get_matchings()
        next_residual = refined.get_residual_after()
        if next_residual is None:
            stop_reason = "missing_residual"
            break

        round_energy_after = residual_energy(next_residual)
        n_windows_matched_round, _, _, _ = _coverage_from_matchings(
            residual,
            refined_matchings,
            template_width=tmpl.width,
        )
        _merge_coverage_intervals(
            covered_intervals,
            refined_matchings,
            residual=residual,
            template_width=tmpl.width,
        )
        n_windows_matched_cumulative = len(covered_intervals)
        n_time_bins_covered = sum(
            _union_interval_length(ivs) for ivs in covered_intervals.values()
        )
        window_frac = (
            n_windows_matched_cumulative / n_windows_total if n_windows_total else 0.0
        )
        time_frac = (
            n_time_bins_covered / n_time_bins_total if n_time_bins_total else 0.0
        )

        log = LearningRoundLog(
            round_index=template_id,
            template_id=tmpl.template_id,
            source_segment_id=tmpl.source_segment_id,
            n_candidates=n_tried,
            seed_n_matchings=len(seed.get_matchings()),
            seed_energy_before=seed.get_energy_before(),
            seed_energy_after=seed.get_energy_after(),
            seed_delta_energy=seed.get_delta_energy_total(),
            averaged_n_matchings=len(refined_matchings),
            energy_before=round_energy_before,
            energy_after=round_energy_after,
            delta_energy=round_energy_before - round_energy_after,
            cumulative_delta_energy=energy_before - round_energy_after,
            n_windows_total=n_windows_total,
            n_windows_matched_round=n_windows_matched_round,
            n_windows_matched_cumulative=n_windows_matched_cumulative,
            window_coverage_fraction=window_frac,
            n_time_bins_total=n_time_bins_total,
            n_time_bins_covered_cumulative=n_time_bins_covered,
            time_coverage_fraction=time_frac,
            template_width=tmpl.width,
            seed_window_policy=used_policy,
            seed_window_id=seed_anchor,
            n_candidates_global=n_candidates_global,
        )
        round_logs.append(log)
        if on_round_complete is not None:
            on_round_complete(log)

        rounds.append(refined)
        if out_dir is not None:
            save_round_template(out_dir, refined, round_index=template_id)
        residual = next_residual
    else:
        # Exhausted the optional n_rounds safety cap.
        if n_rounds is not None:
            stop_reason = "max_rounds"

    return LearnedDictionary(
        seeds=rounds,
        energy_before=energy_before,
        energy_after=residual_energy(residual),
        residual_after=residual,
        round_logs=round_logs,
        stop_reason=stop_reason,
    )


def learn_averaged_templates_from_gpl_paths(
    *,
    window_paths: Sequence[Path | str] | None = None,
    npy_dir: Path | str | None = None,
    csv_paths: Sequence[Path | str] | None = None,
    gpl_output_dir: Path | str | None = None,
    n_windows: int | None = None,
    window_offset: int = 0,
    n_rounds: int | None = None,
    eps_step_fraction: float = 0.5,
    min_repeats: int = 3,
    max_repeats: int | None = None,
    normalize: bool = True,
    disallow_overlap: bool = False,
    min_template_width: int | None = None,
    max_template_width: int | None = None,
    template_method: str = TEMPLATE_METHOD,
    n_components: int = 1,
    template_out_dir: Path | str | None = None,
    show_progress: bool | None = None,
    on_round_complete: Callable[[LearningRoundLog], None] | None = None,
    mp_mode: str | None = None,
    n_workers: int | None = None,
    seed_window_policy: str = "global",
    seed_window_neighborhood: int = 0,
    max_relative_distance: float | None = None,
) -> LearnedDictionary:
    """Run :func:`learn_averaged_templates_from_gpl` on inputs loaded from disk."""
    windows, detections = load_gpl_inputs(
        window_paths=window_paths,
        npy_dir=npy_dir,
        csv_paths=csv_paths,
        gpl_output_dir=gpl_output_dir,
    )
    return learn_averaged_templates_from_gpl(
        windows,
        detections,
        n_windows=n_windows,
        window_offset=window_offset,
        n_rounds=n_rounds,
        eps_step_fraction=eps_step_fraction,
        min_repeats=min_repeats,
        max_repeats=max_repeats,
        normalize=normalize,
        disallow_overlap=disallow_overlap,
        min_template_width=min_template_width,
        max_template_width=max_template_width,
        template_method=template_method,
        n_components=n_components,
        template_out_dir=template_out_dir,
        show_progress=show_progress,
        on_round_complete=on_round_complete,
        mp_mode=mp_mode,
        n_workers=n_workers,
        seed_window_policy=seed_window_policy,
        seed_window_neighborhood=seed_window_neighborhood,
        max_relative_distance=max_relative_distance,
    )


def _crop_candidates_from_residual(
    base_segments: Sequence[Segment],
    residual: dict[str, np.ndarray],
) -> list[Segment]:
    """Re-crop each candidate patch from the current residual at its fixed location."""
    return crop_candidates_from_residual(base_segments, residual)


def _build_arg_parser():
    import argparse

    from ..core import defaults

    p = argparse.ArgumentParser(
        prog="python -m analysis.dict_learning.code.learn.learn_dictionary",
        description="Learn averaged templates from GPL windows/detections (seed→average→MP).",
    )
    p.add_argument("--window", action="append", default=[], help="Path to a window .npy")
    p.add_argument("--npy-dir", type=str, default=None, help="Directory of window .npy files")
    p.add_argument("--csv", action="append", default=[], help="Path to a GPL CSV_*.csv")
    p.add_argument("--gpl-output-dir", type=str, default=None, help="GPL output dir (finds CSV_*.csv)")
    p.add_argument("--n-windows", type=int, default=defaults.N_WINDOWS)
    p.add_argument("--window-offset", type=int, default=defaults.WINDOW_OFFSET)
    p.add_argument(
        "--n-rounds",
        type=int,
        default=defaults.N_ROUNDS,
        help="Optional max outer rounds (omit / None = grow until below min-repeats)",
    )
    p.add_argument(
        "--eps-step-fraction",
        type=float,
        default=defaults.EPS_STEP_FRACTION,
        help="MP stop: relative energy improvement threshold",
    )
    p.add_argument(
        "--min-repeats",
        type=int,
        default=defaults.MIN_REPEATS,
        help="Require at least this many matches in both seed and averaged MP",
    )
    p.add_argument("--min-template-width", type=int, default=defaults.MIN_TEMPLATE_WIDTH)
    p.add_argument("--max-template-width", type=int, default=defaults.MAX_TEMPLATE_WIDTH)
    p.add_argument(
        "--seed-window-policy",
        choices=("global", "low_coverage"),
        default=defaults.SEED_WINDOW_POLICY,
    )
    p.add_argument(
        "--seed-window-neighborhood",
        type=int,
        default=defaults.SEED_WINDOW_NEIGHBORHOOD,
    )
    p.add_argument(
        "--max-relative-distance",
        type=float,
        default=defaults.MAX_RELATIVE_DISTANCE,
        help=(
            "Reject patches leaving more than this fraction of their energy "
            "unexplained by the atom, in MP and in template rebuilds "
            "(0-1; omit / None = no gate)"
        ),
    )
    p.add_argument("--template-out-dir", type=str, default=None, help="Write round_XXX/ templates")
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars (default: on when stderr is a TTY)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    result = learn_averaged_templates_from_gpl_paths(
        window_paths=args.window or None,
        npy_dir=args.npy_dir,
        csv_paths=args.csv or None,
        gpl_output_dir=args.gpl_output_dir,
        n_windows=args.n_windows,
        window_offset=args.window_offset,
        n_rounds=args.n_rounds,
        eps_step_fraction=args.eps_step_fraction,
        min_repeats=args.min_repeats,
        normalize=not args.no_normalize,
        min_template_width=args.min_template_width,
        max_template_width=args.max_template_width,
        seed_window_policy=args.seed_window_policy,
        seed_window_neighborhood=args.seed_window_neighborhood,
        max_relative_distance=args.max_relative_distance,
        template_out_dir=args.template_out_dir,
        show_progress=False if args.no_progress else True,
    )
    templates = result.get_templates()
    print(
        f"learned {len(templates)} templates "
        f"E0={result.get_energy_before():.6g} E1={result.get_energy_after():.6g} "
        f"matchings={len(result.get_matchings())} stop={result.get_stop_reason()}"
    )
    if args.template_out_dir:
        print(f"wrote rounds under {args.template_out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
