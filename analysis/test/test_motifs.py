"""Pair selection on a fake correlation profile. No audio and no matrix profile."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.matrix_profile import (  # noqa: E402
    format_clock,
    rms_envelope,
    top_pairs,
    top_pairs_by_zscore,
)


def test_rms_envelope_is_one_sample_per_frame():
    samples = np.array([3.0, 4.0, 0.0, 0.0], dtype=np.float64)
    envelope = rms_envelope(samples, sample_rate=4, envelope_rate=2)
    np.testing.assert_allclose(envelope, [np.sqrt(12.5), 0.0])


def test_top_pairs_drop_a_copy_that_overlaps():
    window = 5
    correlation = np.full(40, np.nan)
    neighbor = np.zeros(40, dtype=int)
    correlation[0] = 0.9
    neighbor[0] = 10
    correlation[2] = 0.85
    neighbor[2] = 20
    correlation[25] = 0.8
    neighbor[25] = 12
    correlation[20] = 0.7
    neighbor[20] = 30

    pairs = top_pairs(correlation, neighbor, window, k=3)

    assert pairs[0] == (0, 10, 0.9)
    assert pairs[1] == (20, 30, 0.7)
    assert len(pairs) == 2


def test_zscore_ranking_blocks_the_same_span_at_every_length():
    short = np.array([0.0, 5.0, 0.0])
    long = np.array([0.0, 3.0])
    profiles = {
        1: (short, np.array([2, 0, 1]), 1),
        2: (long, np.array([1, 0]), 2),
    }
    # Index 1 of the short profile is the only extreme. Its span blocks
    # index 0 of the long profile (window 2 overlaps sample 1).
    chosen = top_pairs_by_zscore(profiles, k=2)
    assert chosen[0][:3] == (1, 0, 1)
    assert all(row[0] != 2 for row in chosen)


def test_format_clock():
    assert format_clock(674.9) == "11:14.9"
