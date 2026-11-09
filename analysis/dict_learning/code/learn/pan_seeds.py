"""Seed atoms from the pan matrix profile, for the randomized build's outer loop.

``PanSeeder`` replaces the random choice of seed patches in
:func:`~.randomized_dictionary.learn_randomized_outer`. Each outer pass:

1. Builds a search signal from the residual (``energy``, ``bands``, ``pca`` or
   ``full``; see ``analysis.matrix_profile.signals.gpl_rows``), optionally
   averaging every ``decimate`` columns, and runs the pan matrix profile on it
   once.
2. Takes pairs best first, by the z-score of their distance within their
   length (as ``top_pairs_by_zscore``), with the lengths taking turns, longest
   first: each try goes to the next length that has a pair left. A pair is
   skipped when either copy is masked, spans two windows, or sits on a flat
   stretch of the signal.
3. Finds the pair's other copies with one distance profile from its first
   copy. The threshold sits ``copy_threshold`` of the way from the pair's own
   distance ``d`` to the profile's median: ``d + copy_threshold * (median - d)``.
   Positions under it are kept greedily, best first, so no two copies are
   closer than the profile's exclusion zone. A pair with fewer than
   ``min_copies`` copies is dropped. (A multiple of ``d`` does not work on
   several rows: per-row z-normalization gives rows the call does not reach
   full weight, so even a perfect pair has a large ``d``, and twice that
   takes in the whole background.)
4. Cuts every copy from the full residual (re-aligned within ``decimate``
   columns of the first copy), then shifts all copies together, by up to half
   the width, to where their mean has the most energy. Z-normalized distance
   ranks the onset or the tail of a loud call (part noise, part call) as a
   strong motif; only structure the copies share adds up in the mean, so the
   shift moves such a cut onto the call. The mean, scaled to unit norm, is the
   seed atom.
5. Drops the seed when its best shifted cosine with a seed already taken, or
   an atom already in the dictionary, reaches ``dedupe_corr``: the same call
   at another length or offset.
6. Masks every copy, before and after the shift, at every length, so no
   other pair of that call is taken.

Steps 2–6 repeat until ``m`` seeds are taken or no pair is left. The outer
loop then refines the seeds with matching pursuit and subtracts them, and the
next pass searches what is left.

The matrix profile runs once per pass, on the residual. Masking uses the
nearest neighbours found before anything was masked, so a stretch whose
neighbour got masked loses its pair within the pass; the next pass's profile,
on the new residual, can find it again.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from analysis.matrix_profile import distance_profile, pan_minlag, pan_profile
from analysis.matrix_profile.signals import DEFAULT_PCA_ROWS, GPL_KINDS, decimate_columns, gpl_rows

from ..core import Dictionary, Template
from ..io import ColumnMap
from .randomized_dictionary import RandomizedCreateResult
from .shape_match import shape_similarity

DEFAULT_SIGNAL = "pca"
DEFAULT_COPY_THRESHOLD = 0.5
DEFAULT_MIN_COPIES = 3
DEFAULT_DEDUPE_CORR = 0.9
# Pairs tried per seed asked for, before a pass gives up on finding more.
TRIES_PER_SEED = 25


@dataclass(frozen=True)
class PanSeedConfig:
    """How :class:`PanSeeder` searches.

    ``lengths`` are motif lengths in native STFT columns. With ``decimate``
    above 1 each is rounded to a whole number of search samples (at least 4),
    and the atom is that many search samples wide.
    """

    lengths: tuple[int, ...]
    start_freq: float
    end_freq: float
    signal: str = DEFAULT_SIGNAL
    pca_rows: int = DEFAULT_PCA_ROWS
    band_edges: tuple[float, ...] | None = None
    decimate: int = 1
    copy_threshold: float = DEFAULT_COPY_THRESHOLD
    min_copies: int = DEFAULT_MIN_COPIES
    dedupe_corr: float = DEFAULT_DEDUPE_CORR
    nthreads: int = 1

    def __post_init__(self) -> None:
        if self.signal not in GPL_KINDS:
            raise ValueError(f"signal must be one of {GPL_KINDS}, got {self.signal!r}")  # no-pipeline-cover: argparse limits --signal
        if not self.lengths:
            raise ValueError("need at least one motif length")  # no-pipeline-cover: the CLI rejects an empty --lengths-s first
        if self.decimate < 1:
            raise ValueError(f"decimate must be >= 1, got {self.decimate}")
        if min(self.search_lengths) < 4:
            raise ValueError(
                f"motif lengths must be at least 4 search samples (4 × decimate = "
                f"{4 * self.decimate} columns), got {min(self.lengths)} columns"
            )
        if not 0.0 <= self.copy_threshold < 1.0:
            raise ValueError(f"copy threshold must be in [0, 1), got {self.copy_threshold}")
        if self.min_copies < 2:
            raise ValueError(f"min copies must be >= 2, got {self.min_copies}")
        if not 0.0 < self.dedupe_corr <= 1.0:
            raise ValueError(f"dedupe correlation must be in (0, 1], got {self.dedupe_corr}")

    @property
    def search_lengths(self) -> list[int]:
        """Distinct lengths in search samples, shortest first."""
        return sorted({int(round(length / self.decimate)) for length in self.lengths})


@dataclass
class PanPick:
    """One accepted seed: the pair it came from and the copies that built it."""

    outer: int
    template_id: int
    width: int
    distance: float
    zscore: float
    pair: tuple[tuple[str, int], tuple[str, int]]
    copies: list[tuple[str, int]]


@dataclass
class PanPass:
    """What one outer pass of :class:`PanSeeder` looked at and took."""

    outer: int
    signal_rows: int
    signal_columns: int
    n_pairs_tried: int = 0
    n_masked: int = 0
    n_too_few_copies: int = 0
    n_duplicates: int = 0
    picks: list[PanPick] = field(default_factory=list)


def _flat_starts(search: np.ndarray, length: int) -> np.ndarray:
    """True where some row of the search signal is constant over ``length`` samples."""
    n_starts = search.shape[1] - length + 1
    cumulative = np.cumsum(np.pad(search, ((0, 0), (1, 0))), axis=1)
    cumulative_sq = np.cumsum(np.pad(search * search, ((0, 0), (1, 0))), axis=1)
    total = cumulative[:, length:] - cumulative[:, :n_starts]
    total_sq = cumulative_sq[:, length:] - cumulative_sq[:, :n_starts]
    variance = total_sq - total * total / length
    scale = np.maximum(total_sq, 1e-300)
    return np.any(variance <= 1e-12 * scale, axis=0)


def _block(masks: dict[int, np.ndarray], start: int, width: int) -> None:
    """Mask every start, at every length, whose window overlaps ``[start, start + width)``."""
    for length, mask in masks.items():
        mask[max(0, start - length + 1) : min(len(mask), start + width)] = True


def _greedy_copies(order: np.ndarray, separation: int, n_starts: int) -> list[int]:
    """Starts in ``order`` (best first), skipping any closer than ``separation`` to one kept."""
    taken = np.zeros(n_starts, dtype=bool)
    kept: list[int] = []
    for start in order:
        start = int(start)
        if taken[start]:
            continue
        kept.append(start)
        taken[max(0, start - separation + 1) : min(n_starts, start + separation)] = True
    return kept


def _recentre(
    joined: np.ndarray, columns: ColumnMap, starts: list[int], width: int
) -> list[tuple[int, tuple[str, int]]]:
    """Shift every copy by the same amount to where their mean has the most energy.

    Shifts go up to ``width // 2`` columns either way, limited so every copy
    stays inside its window. Returns each copy's native start and
    ``(window_id, t_start)``.
    """
    lo, hi = -(width // 2), width // 2
    for start in starts:
        k = columns.window_index(start)
        lo = max(lo, columns.starts[k] - start)
        hi = min(hi, columns.starts[k + 1] - width - start)
    best_shift, best_energy = 0, -1.0
    for shift in range(lo, hi + 1):
        mean = np.mean([joined[:, c + shift : c + shift + width] for c in starts], axis=0)
        energy = float(np.sum(mean * mean))
        if energy > best_energy:
            best_shift, best_energy = shift, energy
    out = []
    for start in starts:
        location = columns.locate(start + best_shift, width)
        assert location is not None, "shifts keep every copy inside its window"
        out.append((start + best_shift, location))
    return out


class PanSeeder:
    """A :class:`~.randomized_dictionary.Seeder` that takes seeds from the pan matrix profile.

    ``passes`` keeps one :class:`PanPass` per call, for the run's
    ``pan_seeds.jsonl``.
    """

    def __init__(self, config: PanSeedConfig) -> None:
        self.config = config
        self.passes: list[PanPass] = []

    def __call__(
        self, residual: dict[str, np.ndarray], m: int, accumulated: Dictionary
    ) -> RandomizedCreateResult:
        cfg = self.config
        stack = {k: np.asarray(v, dtype=np.float64) for k, v in residual.items()}
        columns = ColumnMap.from_stack(stack)
        joined = columns.join(stack)
        blocks, _info = gpl_rows(
            [stack[w] for w in columns.window_ids],
            cfg.signal,
            start_freq=cfg.start_freq,
            end_freq=cfg.end_freq,
            band_edges=list(cfg.band_edges) if cfg.band_edges else None,
            pca_rows=cfg.pca_rows,
        )
        search = decimate_columns(np.concatenate(blocks, axis=1), cfg.decimate)
        log = PanPass(outer=len(self.passes) + 1, signal_rows=search.shape[0], signal_columns=search.shape[1])
        self.passes.append(log)

        lengths = cfg.search_lengths
        profile = pan_profile(search, lengths, nthreads=cfg.nthreads)
        d = cfg.decimate
        masks: dict[int, np.ndarray] = {}
        scores: dict[int, np.ndarray] = {}
        for length, distance in zip(lengths, profile.distances, strict=True):
            mask = _flat_starts(search, length)
            # A search start covers native columns [s * d, (s + length) * d).
            crosses = columns.crosses_boundary(length * d)
            starts = np.arange(len(mask)) * d
            mask |= crosses[np.minimum(starts, len(crosses) - 1)]
            masks[length] = mask
            finite = np.isfinite(distance)
            unusual = np.full(len(distance), -np.inf)
            spread = float(np.std(distance[finite], ddof=1)) if np.count_nonzero(finite) > 1 else 0.0
            if spread > 0.0:  # no-pipeline-branch: a real signal's distances are never all equal
                unusual[finite] = (float(np.mean(distance[finite])) - distance[finite]) / spread
            scores[length] = unusual
        neighbors = dict(zip(lengths, profile.neighbors, strict=True))
        distances = dict(zip(lengths, profile.distances, strict=True))

        existing = [t.basis[0] for t in accumulated]
        dictionary = Dictionary()
        next_id = max((int(t.template_id) for t in accumulated), default=-1) + 1
        max_tries = TRIES_PER_SEED * int(m)

        # Lengths take turns, longest first: z-scores of different lengths do
        # not compare (on real data the shortest length won every pick), and a
        # long seed masks the shorter cuts of its call.
        turns = lengths[::-1]
        turn = 0
        while len(dictionary) < m and log.n_pairs_tried < max_tries:
            best = None
            for step in range(len(turns)):
                length = turns[(turn + step) % len(turns)]
                score = np.where(masks[length], -np.inf, scores[length])
                i = int(np.argmax(score))
                if np.isfinite(score[i]):
                    best = (float(score[i]), length, i)
                    turn = (turn + step + 1) % len(turns)
                    break
            if best is None:
                break
            zscore, length, i = best
            scores[length][i] = -np.inf
            j = int(neighbors[length][i])
            log.n_pairs_tried += 1
            if masks[length][j]:
                log.n_masked += 1
                continue

            pair_distance = float(distances[length][i])
            dp = distance_profile(search, i, length)
            radius = pair_distance + cfg.copy_threshold * max(0.0, float(np.median(dp)) - pair_distance)
            close = np.flatnonzero((dp <= radius + 1e-9 * (1.0 + radius)) & ~masks[length])
            separation = pan_minlag(length)
            copies = _greedy_copies(close[np.argsort(dp[close], kind="stable")], separation, len(dp))
            if len(copies) < cfg.min_copies:
                log.n_too_few_copies += 1
                scores[length][j] = -np.inf
                continue

            width = length * d
            aligned = self._align(joined, columns, [c * d for c in copies], width)
            placed = _recentre(joined, columns, aligned, width)
            seed = np.mean([joined[:, c : c + width] for c, _ in placed], axis=0)
            norm = float(np.linalg.norm(seed))
            for c in copies:
                _block(masks, c, length)
            for c, _ in placed:
                _block(masks, c // d, (c % d + width + d - 1) // d)
            if norm <= 0.0:
                continue  # no-pipeline-cover: a flat mean needs flat copies, and flat starts are masked
            seed = seed / norm
            if any(shape_similarity(seed, other) >= cfg.dedupe_corr for other in existing):
                log.n_duplicates += 1
                continue
            first, second = columns.locate(i * d, width), columns.locate(j * d, width)
            assert first is not None and second is not None, "masked starts never span two windows"
            source = f"pan:{first[0]}:{first[1]}:{second[0]}:{second[1]}:w{width}"
            dictionary.add(Template(template_id=next_id, data=seed, source_segment_id=source))
            existing.append(seed)
            log.picks.append(
                PanPick(
                    outer=log.outer,
                    template_id=next_id,
                    width=width,
                    distance=pair_distance,
                    zscore=zscore,
                    pair=(first, second),
                    copies=[loc for _, loc in placed],
                )
            )
            next_id += 1

        return RandomizedCreateResult(
            dictionary=dictionary,
            pool_size=log.n_pairs_tried,
            n_sampled=len(dictionary),
            residual=stack,
        )

    def _align(self, joined: np.ndarray, columns: ColumnMap, starts: list[int], width: int) -> list[int]:
        """Native start of each copy.

        With ``decimate`` above 1, every copy after the first moves by up to
        ``decimate - 1`` columns, within its window, to the shift that best
        matches the first copy on the full residual.
        """
        d = self.config.decimate
        reference = joined[:, starts[0] : starts[0] + width]
        out: list[int] = []
        for index, start in enumerate(starts):
            best = start
            if d > 1 and index > 0:
                k = columns.window_index(start)
                lo = max(columns.starts[k], start - d + 1)
                hi = min(columns.starts[k + 1] - width, start + d - 1)
                scores = [float(np.sum(reference * joined[:, c : c + width])) for c in range(lo, hi + 1)]
                best = lo + int(np.argmax(scores))
            out.append(best)
        return out
