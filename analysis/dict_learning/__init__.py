"""Offline dictionary-learning tools (seed_from_GPL, Mallat MP, …)."""

from __future__ import annotations

from typing import Any

from .code.core import Dictionary, Matching, Segment, Template, Window
from .code.mp import run_mp_dp_loop
from .code.mp import MpStepResult, mp_step, residual_energy, run_mp_loop
from .code.mp import score_shifts
from .code.learn.seed_from_gpl import Seed, detections_to_segments, mean_center, seed_from_gpl
from .code.learn.learn_from_motifs import (
    MotifSeeds,
    seeds_from_motif_pairs,
    window_from_band_profile,
)
from .code.learn.learn_dictionary import (
    DictionaryLearningError,
    DictionaryRefinementResult,
    NoMatchingPatchesError,
    LearnedDictionary,
    LearningRoundLog,
    learn_averaged_templates_from_gpl,
    learn_averaged_templates_from_gpl_paths,
    learn_from_gpl,
    learn_from_gpl_paths,
    refine_dictionary_by_averaging,
    refine_seed_by_averaging,
    save_round_template,
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
    build_dictionary_from_detections,
    combine_patches,
    matching_pursuit,
    merge_templates,
    pca_basis,
    pca_patches,
    pca_templates,
    rebuild_template_from_matchings,
    remove_template,
)

# Runnable ``python -m analysis.dict_learning.code.learn.{build,extend,match_to}_dictionary``
# modules are loaded lazily so ``-m`` does not warn that they were already imported via
# this package ``__init__``.
_LAZY_EXPORTS = {
    "build_dictionary_for_wav": (".code.learn.build_dictionary", "build_dictionary_for_wav"),
    "ExtendDictionaryResult": (".code.learn.extend_dictionary", "ExtendDictionaryResult"),
    "extend_dictionary_on_windows": (
        ".code.learn.extend_dictionary",
        "extend_dictionary_on_windows",
    ),
    "extend_dictionary_for_wav": (
        ".code.learn.extend_dictionary",
        "extend_dictionary_for_wav",
    ),
    "match_to_dictionary": (".code.learn.match_to_dictionary", "match_to_dictionary"),
    "write_match_to_dictionary_result": (
        ".code.learn.match_to_dictionary",
        "write_match_to_dictionary_result",
    ),
    "SplitResult": (".code.learn.split_template", "SplitResult"),
    "SplitOneResult": (".code.learn.split_template", "SplitOneResult"),
    "split_patches_rank1": (".code.learn.split_template", "split_patches_rank1"),
    "split_one_template": (".code.learn.split_template", "split_one_template"),
    "split_template_matchings": (
        ".code.learn.split_template",
        "split_template_matchings",
    ),
    "DivergenceResult": (".code.learn.similarity", "DivergenceResult"),
    "calc_divergences": (".code.learn.similarity", "calc_divergences"),
    "TemplateSimilarityResult": (
        ".code.learn.similarity",
        "TemplateSimilarityResult",
    ),
    "calc_template_similarity": (
        ".code.learn.similarity",
        "calc_template_similarity",
    ),
    "TemplateGroupSimilarityResult": (
        ".code.learn.similarity",
        "TemplateGroupSimilarityResult",
    ),
    "calc_template_group_similarity": (
        ".code.learn.similarity",
        "calc_template_group_similarity",
    ),
    "learn_dictionary_from_motifs": (
        ".code.learn.learn_from_motifs",
        "learn_dictionary_from_motifs",
    ),
    "MotifLearnResult": (
        ".code.learn.learn_from_motifs",
        "MotifLearnResult",
    ),
}


def __getattr__(name: str) -> Any:
    spec = _LAZY_EXPORTS.get(name)
    if spec is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr = spec
    from importlib import import_module

    value = getattr(import_module(module_name, __name__), attr)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()) | set(_LAZY_EXPORTS))


__all__ = [
    "Dictionary",
    "Matching",
    "Segment",
    "Seed",
    "Template",
    "Window",
    "MpStepResult",
    "detections_to_segments",
    "mean_center",
    "mp_step",
    "residual_energy",
    "run_mp_dp_loop",
    "run_mp_loop",
    "score_shifts",
    "seed_from_gpl",
    "MotifSeeds",
    "seeds_from_motif_pairs",
    "window_from_band_profile",
    "DictionaryLearningError",
    "DictionaryRefinementResult",
    "NoMatchingPatchesError",
    "LearnedDictionary",
    "LearningRoundLog",
    "learn_from_gpl",
    "learn_from_gpl_paths",
    "refine_dictionary_by_averaging",
    "refine_seed_by_averaging",
    "RandomizedCreateResult",
    "RandomizedOuterPassLog",
    "RandomizedOuterResult",
    "RandomizedRefineResult",
    "build_randomized_dictionary",
    "create_dictionary",
    "dictionary_explained_energy",
    "learn_randomized_outer",
    "refine_dictionary",
    "learn_averaged_templates_from_gpl",
    "learn_averaged_templates_from_gpl_paths",
    "build_dictionary_for_wav",
    "ExtendDictionaryResult",
    "extend_dictionary_on_windows",
    "extend_dictionary_for_wav",
    "match_to_dictionary",
    "write_match_to_dictionary_result",
    "SplitResult",
    "SplitOneResult",
    "split_patches_rank1",
    "split_one_template",
    "split_template_matchings",
    "DivergenceResult",
    "calc_divergences",
    "TemplateSimilarityResult",
    "calc_template_similarity",
    "TemplateGroupSimilarityResult",
    "calc_template_group_similarity",
    "learn_dictionary_from_motifs",
    "MotifLearnResult",
    "matching_pursuit",
    "average_templates",
    "average_patches",
    "pca_templates",
    "pca_patches",
    "pca_basis",
    "combine_patches",
    "rebuild_template_from_matchings",
    "build_dictionary_from_detections",
    "remove_template",
    "merge_templates",
    "save_round_template",
]
