"""Tests for per-chunk distanceNorm used by the template CDF plot."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.dict_learning.code.catalog.repeats_from_catalog import (  # noqa: E402
    _crop_patch_from_window,
    chunk_distance_norm,
    sort_chunks_by_fit,
)


def test_identical_patch_has_near_zero_distance():
    rng = np.random.default_rng(0)
    template = rng.normal(size=(8, 5))
    dist = chunk_distance_norm(template.copy(), template)
    assert dist is not None
    assert dist < 1e-9


def test_scaled_patch_has_near_zero_distance():
    rng = np.random.default_rng(1)
    template = rng.normal(size=(8, 5))
    dist = chunk_distance_norm(3.5 * template, template)
    assert dist is not None
    assert dist < 1e-9


def test_orthogonal_patch_has_large_distance():
    template = np.zeros((4, 3), dtype=np.float64)
    template[0, :] = 1.0
    patch = np.zeros((4, 3), dtype=np.float64)
    patch[1, :] = 1.0
    dist = chunk_distance_norm(patch, template)
    assert dist is not None
    assert dist > 0.99


def test_zero_energy_patch_returns_none():
    template = np.ones((4, 3), dtype=np.float64)
    patch = np.zeros((4, 3), dtype=np.float64)
    assert chunk_distance_norm(patch, template) is None


def test_sort_chunks_by_fit_orders_best_match_first():
    chunks = [
        {"id": "c-worst", "distanceNorm": 0.8},
        {"id": "c-best", "distanceNorm": 0.1},
        {"id": "c-mid", "distanceNorm": 0.4},
    ]
    ordered = sort_chunks_by_fit(chunks)
    assert [c["id"] for c in ordered] == ["c-best", "c-mid", "c-worst"]


def test_sort_chunks_by_fit_puts_undefined_distance_last():
    chunks = [
        {"id": "c-undefined"},
        {"id": "c-good", "distanceNorm": 0.05},
        {"id": "c-bad", "distanceNorm": 0.9},
    ]
    ordered = sort_chunks_by_fit(chunks)
    assert [c["id"] for c in ordered] == ["c-good", "c-bad", "c-undefined"]


def test_sort_chunks_by_fit_is_monotonic_non_decreasing_per_template():
    """Randomized check: within each independently-sorted template's chunk
    list, distanceNorm never decreases from one patch to the next (undefined
    distances are treated as +inf and so always sort after every defined one).
    """
    rng = random.Random(1234)
    for template_index in range(50):
        chunk_count = rng.randint(0, 15)
        chunks = []
        for chunk_index in range(chunk_count):
            distance = None if rng.random() < 0.15 else rng.uniform(0.0, 1.0)
            chunks.append({"id": f"t{template_index}-c{chunk_index}", "distanceNorm": distance})
        rng.shuffle(chunks)

        ordered = sort_chunks_by_fit(chunks)

        keys = [
            c["distanceNorm"] if c.get("distanceNorm") is not None else math.inf
            for c in ordered
        ]
        assert all(a <= b for a, b in zip(keys, keys[1:])), (
            f"template {template_index} distances not non-decreasing: {keys}"
        )
        # Same chunks in, same chunks out — only the order changes.
        assert {c["id"] for c in ordered} == {c["id"] for c in chunks}


def test_sort_chunks_by_fit_does_not_mutate_input():
    chunks = [
        {"id": "c-worst", "distanceNorm": 0.8},
        {"id": "c-best", "distanceNorm": 0.1},
    ]
    original_order = [c["id"] for c in chunks]
    sort_chunks_by_fit(chunks)
    assert [c["id"] for c in chunks] == original_order


def test_sort_chunks_by_fit_is_stable_for_equal_distances():
    chunks = [
        {"id": "a", "distanceNorm": 0.5},
        {"id": "b", "distanceNorm": 0.5},
        {"id": "c", "distanceNorm": 0.5},
        {"id": "d"},
        {"id": "e"},
    ]
    ordered = sort_chunks_by_fit(chunks)
    assert [c["id"] for c in ordered] == ["a", "b", "c", "d", "e"]


def test_sort_chunks_by_fit_orders_real_distances_end_to_end():
    """Build actual patches at varying similarity to a template via
    chunk_distance_norm (not fabricated numbers), then confirm sorting
    them with sort_chunks_by_fit reproduces the independently-computed
    best-to-worst order and a non-decreasing distance sequence.
    """
    rng = np.random.default_rng(7)
    template = rng.normal(size=(8, 6))
    noise = rng.normal(size=(8, 6))

    chunks = []
    for chunk_index, noise_fraction in enumerate([0.0, 0.2, 0.5, 0.8, 1.0]):
        patch = (1.0 - noise_fraction) * template + noise_fraction * noise
        chunk = {
            "id": f"c{chunk_index}",
            "distanceNorm": chunk_distance_norm(patch, template),
        }
        chunks.append(chunk)
    # Distances aren't guaranteed monotonic in noise_fraction (a noisy blend
    # can land closer to the template by chance) — derive the expected order
    # independently rather than assuming construction order.
    expected_order = [c["id"] for c in sorted(chunks, key=lambda c: c["distanceNorm"])]

    ordered = sort_chunks_by_fit(list(reversed(chunks)))

    assert [c["id"] for c in ordered] == expected_order
    distances = [c["distanceNorm"] for c in ordered]
    assert all(a <= b for a, b in zip(distances, distances[1:]))


def test_crop_pads_short_tail():
    window = np.arange(20, dtype=np.float64).reshape(4, 5)
    patch = _crop_patch_from_window(window, t_start=3, width=4, n_freq=4)
    assert patch is not None
    assert patch.shape == (4, 4)
    # Two columns copied, two zero-padded.
    np.testing.assert_allclose(patch[:, :2], window[:, 3:5])
    np.testing.assert_allclose(patch[:, 2:], 0.0)
