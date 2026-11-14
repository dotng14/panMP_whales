"""The four-theme cyclic model of `03_themes_and_songs.md`, with its parameters held fixed.

The motif probabilities and visit lengths of the four themes come from the
fit on wav 2 and wav 3 (27 April 2023, 01:11:15 to 02:26:15 UTC), made from
the 10 kHz recording. A new stretch is decoded with those parameters; nothing
is re-estimated from the new stretch.
"""

from __future__ import annotations

import numpy as np

N_THEMES = 4
BIN_SEC = 30.0
ALPHA = 0.3
MIN_BINS = 2
MAX_BINS = 60
SIGMA_FLOOR = 0.25


class ThemeModel:
    def __init__(self, motif_ids, log_p, mu, sigma):
        self.motif_ids = list(motif_ids)
        self.column = {motif: i for i, motif in enumerate(self.motif_ids)}
        self.log_p = log_p  # (N_THEMES, n_motifs)
        self.mu = mu
        self.sigma = sigma
        lengths = np.arange(1, MAX_BINS + 1)
        self.log_duration = np.full((N_THEMES, MAX_BINS + 1), -np.inf)
        for theme in range(N_THEMES):
            raw = -0.5 * ((np.log(lengths) - mu[theme]) / sigma[theme]) ** 2 - np.log(lengths * sigma[theme])
            raw[lengths < MIN_BINS] = -np.inf
            self.log_duration[theme, 1:] = raw - np.log(np.sum(np.exp(raw)))

    @classmethod
    def fit(cls, motif_ids, placement_motifs, placement_starts, visits):
        """visits: rows (theme, start_sec, end_sec) of the reference segmentation."""
        motif_ids = list(motif_ids)
        column = {motif: i for i, motif in enumerate(motif_ids)}
        counts = np.zeros((N_THEMES, len(motif_ids)))
        log_lengths = [[] for _ in range(N_THEMES)]
        for theme, begin, end in visits:
            inside = (placement_starts >= begin) & (placement_starts < end)
            for motif in placement_motifs[inside]:
                counts[theme, column[int(motif)]] += 1
            log_lengths[theme].append(np.log((end - begin) / BIN_SEC))
        log_p = np.log((ALPHA + counts) / (ALPHA * len(motif_ids) + counts.sum(axis=1, keepdims=True)))
        mu = np.array([np.mean(v) for v in log_lengths])
        sigma = np.array([max(np.std(v), SIGMA_FLOOR) for v in log_lengths])
        return cls(motif_ids, log_p, mu, sigma)

    def bin_counts(self, motifs, starts, t0, n_bins):
        counts = np.zeros((n_bins, len(self.motif_ids)))
        index = np.floor((np.asarray(starts) - t0) / BIN_SEC).astype(int)
        for b, motif in zip(index, motifs):
            if 0 <= b < n_bins:
                counts[b, self.column[int(motif)]] += 1
        return counts

    def decode(self, counts):
        """Best cyclic path. Returns (score, visits) with visits as (theme, first bin, end bin)."""
        n_bins = counts.shape[0]
        emission = counts @ self.log_p.T  # (bin, theme)
        cumulative = np.vstack([np.zeros((1, N_THEMES)), np.cumsum(emission, axis=0)])
        best = np.full((n_bins + 1, N_THEMES), -np.inf)
        back = np.zeros((n_bins + 1, N_THEMES), dtype=int)
        for t in range(1, n_bins + 1):
            for theme in range(N_THEMES):
                previous = (theme - 1) % N_THEMES
                for d in range(MIN_BINS, min(MAX_BINS, t) + 1):
                    start = t - d
                    lead = -np.log(N_THEMES) if start == 0 else best[start, previous]
                    if not np.isfinite(lead):
                        continue
                    score = lead + self.log_duration[theme, d] + cumulative[t, theme] - cumulative[start, theme]
                    if score > best[t, theme]:
                        best[t, theme] = score
                        back[t, theme] = d
        theme = int(np.argmax(best[n_bins]))
        score = float(best[n_bins, theme])
        visits = []
        t = n_bins
        while t > 0:
            d = back[t, theme]
            visits.append((theme, t - d, t))
            t -= d
            theme = (theme - 1) % N_THEMES
        return score, visits[::-1]

    def visit_scores(self, counts, visit):
        """Log likelihood of one visit's placements under each of the four themes."""
        _theme, first, end = visit
        return counts[first:end].sum(axis=0) @ self.log_p.T
