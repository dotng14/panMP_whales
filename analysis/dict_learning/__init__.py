"""Offline dictionary-learning tools: Mallat MP and the randomized dictionary build."""

from __future__ import annotations

from typing import Any

from .code.core import Dictionary, Matching, Segment, Template, Window, mean_center
from .code.io import detections_to_segments
from .code.mp import MpStepResult, mp_step, residual_energy, run_mp_loop
from .code.mp import score_shifts
from .code.learn.refine import (
    DictionaryLearningError,
    DictionaryRefinementResult,
    NoMatchingPatchesError,
    refine_dictionary_by_averaging,
)
from .code.learn.randomized_dictionary import (
    RandomizedCreateResult,
    RandomizedOuterPassLog,
    RandomizedOuterResult,
    RandomizedRefineResult,
    build_randomized_dictionary,
    create_dictionary,
    dictionary_explained_energy,
    learn_randomized_outer,
    refine_dictionary,
)
from .code.tools import (
    average_patches,
    average_templates,
    combine_patches,
    matching_pursuit,
    pca_basis,
    pca_patches,
    rebuild_template_from_matchings,
)

# ``learn_from_motifs`` is runnable with ``python -m``; load it lazily so ``-m``
# does not warn that it was already imported via this package ``__init__``.
_LAZY_EXPORTS = {
    "MotifSeeds": (".code.learn.learn_from_motifs", "MotifSeeds"),
    "seeds_from_motif_pairs": (".code.learn.learn_from_motifs", "seeds_from_motif_pairs"),
    "window_from_band_profile": (".code.learn.learn_from_motifs", "window_from_band_profile"),
    "learn_dictionary_from_motifs": (
        ".code.learn.learn_from_motifs",
        "learn_dictionary_from_motifs",
    ),
    "MotifLearnResult": (".code.learn.learn_from_motifs", "MotifLearnResult"),
}


def __getattr__(name: str) -> Any:  # no-pipeline-cover: lazy learn_from_motifs exports
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = spec
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:  # no-pipeline-cover: interactive helper
    return sorted(set(__all__) | set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "Dictionary",
    "Matching",
    "Segment",
    "Template",
    "Window",
    "MpStepResult",
    "detections_to_segments",
    "mean_center",
    "mp_step",
    "residual_energy",
    "run_mp_loop",
    "score_shifts",
    "DictionaryLearningError",
    "DictionaryRefinementResult",
    "NoMatchingPatchesError",
    "refine_dictionary_by_averaging",
    "RandomizedCreateResult",
    "RandomizedOuterPassLog",
    "RandomizedOuterResult",
    "RandomizedRefineResult",
    "build_randomized_dictionary",
    "create_dictionary",
    "dictionary_explained_energy",
    "learn_randomized_outer",
    "refine_dictionary",
    "MotifSeeds",
    "seeds_from_motif_pairs",
    "window_from_band_profile",
    "learn_dictionary_from_motifs",
    "MotifLearnResult",
    "matching_pursuit",
    "average_templates",
    "average_patches",
    "pca_patches",
    "pca_basis",
    "combine_patches",
    "rebuild_template_from_matchings",
]
