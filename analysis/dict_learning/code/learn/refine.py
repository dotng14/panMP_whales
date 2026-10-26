"""Refine a whole dictionary: average every atom's patches, then re-run MP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from ..core import TEMPLATE_METHOD
from ..core import Dictionary, Matching, Template, Window, mean_center
from ..mp import residual_energy


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
