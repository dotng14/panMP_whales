"""``distance_profile`` against brute force and against the matrix profile it must agree with."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.matrix_profile import distance_profile, mpx_channels, mpx_fast, pan_minlag


def _brute(rows: np.ndarray, start: int, width: int) -> np.ndarray:
    def unit(x):
        x = x - x.mean()
        norm = np.linalg.norm(x)
        return None if norm == 0 else x / norm

    out = np.zeros(rows.shape[1] - width + 1)
    for j in range(len(out)):
        for row in rows:
            a, b = unit(row[start : start + width]), unit(row[j : j + width])
            if a is None and b is None:
                corr = 1.0
            elif a is None or b is None:
                corr = 0.0
            else:
                corr = float(np.clip(a @ b, -1.0, 1.0))
            out[j] += np.sqrt(2.0 * width * (1.0 - corr))
    return out


def test_matches_brute_force_on_several_rows():
    rows = np.random.default_rng(1).standard_normal((3, 200))
    np.testing.assert_allclose(distance_profile(rows, 40, 16), _brute(rows, 40, 16), atol=1e-6)


def test_flat_windows_follow_the_matrix_profile_rule():
    rows = np.random.default_rng(2).standard_normal((2, 120))
    rows[1, 10:40] = 3.0  # flat stretch on one row only
    np.testing.assert_allclose(distance_profile(rows, 12, 8), _brute(rows, 12, 8), atol=1e-6)
    np.testing.assert_allclose(distance_profile(rows, 60, 8), _brute(rows, 60, 8), atol=1e-6)


@pytest.mark.parametrize("start", [0, 37, 120])
def test_nearest_neighbour_equals_mpx_channels(start):
    rows = np.random.default_rng(3).standard_normal((4, 300))
    width = 16
    minlag = pan_minlag(width)
    distance, neighbor = mpx_channels(rows, minlag, width)
    profile = distance_profile(rows, start, width)
    profile[max(0, start - minlag + 1) : start + minlag] = np.inf
    assert profile.min() == pytest.approx(distance[start], rel=1e-9)
    assert int(np.argmin(profile)) == int(neighbor[start])


def test_one_row_equals_mpx_fast():
    series = np.random.default_rng(4).standard_normal(250)
    width = 20
    minlag = pan_minlag(width)
    distance, neighbor = mpx_fast(series, minlag, width)
    profile = distance_profile(series, 70, width)
    profile[70 - minlag + 1 : 70 + minlag] = np.inf
    assert profile.min() == pytest.approx(distance[70], rel=1e-9)
    assert int(np.argmin(profile)) == int(neighbor[70])


@pytest.mark.parametrize(
    "start, length, message",
    [(0, 3, "length must be between 4"), (0, 101, "length must be between 4"),
     (-1, 8, "start must be between"), (93, 8, "start must be between")],
)
def test_rejects_bad_arguments(start, length, message):
    with pytest.raises(ValueError, match=message):
        distance_profile(np.zeros((1, 100)), start, length)
