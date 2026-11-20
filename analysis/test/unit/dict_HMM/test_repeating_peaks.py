"""Synthetic checks for fixed-period peak finding."""

from __future__ import annotations

import numpy as np

from analysis.dict_HMM.find_songs.repeating_peaks import find_repeating_peaks


def test_finds_planted_period():
    dt = 0.1
    t = np.arange(0.0, 200.0, dt)
    period = 11.5
    y = np.zeros_like(t)
    # Tall pulses every `period` seconds, plus smaller mid-phrase pulses.
    for k, start in enumerate(np.arange(5.0, 190.0, period)):
        y += 3.0 * np.exp(-0.5 * ((t - start) / 0.25) ** 2)
        y += 1.2 * np.exp(-0.5 * ((t - (start + 0.35 * period)) / 0.25) ** 2)
    y += 0.05 * np.random.default_rng(0).standard_normal(t.size)

    result = find_repeating_peaks(
        t,
        y,
        t0=10.0,
        t1=180.0,
        smooth_s=0.2,
        min_separation_s=1.0,
        height_percentile=70.0,
        prominence_percentile=60.0,
        min_period_s=3.0,
        max_period_s=40.0,
        top_k=6,
    )
    assert result.n_peaks >= 10
    assert result.best_period_s is not None
    # Best period should be near the planted phrase length (or 2×).
    nearest = min(abs(result.best_period_s / period - m) for m in (1.0, 2.0))
    assert nearest < 0.12, (result.best_period_s, period, result.periods[:3])
    assert any(abs(h.period_s - period) / period < 0.1 for h in result.periods)
