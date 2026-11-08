"""Limited-overlap dictionary learning.

Placements never subtract from the spectrogram. A time column may be covered
by at most ``k`` kept placements.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.fft import irfft, rfft
from scipy.fft import next_fast_len

from analysis.dict_learning.code.core import Window
from analysis.dict_learning.code.io import Detection, detections_to_segments

DEFAULT_K = 1
DEFAULT_EPS = 0.01
DEFAULT_W_MIN = 10
DEFAULT_W_MAX = 40
DEFAULT_M = 10
DEFAULT_N_MIN = 10
DEFAULT_REFINE_ITERS = 1
DEFAULT_TAU = 0.9
DEFAULT_MAX_OUTER = 50
DEFAULT_MIN_COVERAGE = 0.5
DEFAULT_N_COMPONENTS = 1


@dataclass
class Motif:
    motif_id: int
    data: np.ndarray  # (F, W) or (n_components, F, W), each component unit norm
    window_id: str
    t_start: int
    mean: np.ndarray | None = None

    @property
    def width(self) -> int:
        return int(self.data.shape[-1])

    @property
    def n_freq(self) -> int:
        return int(self.data.shape[-2])

    @property
    def n_components(self) -> int:
        return 1 if self.data.ndim == 2 else int(self.data.shape[0])

    @property
    def basis(self) -> np.ndarray:
        if self.data.ndim == 2:
            return self.data[None, ...]
        return self.data


@dataclass
class Placement:
    motif_id: int
    window_id: str
    t_start: int
    amplitude: float
    unexplained: float
    coeffs: np.ndarray | None = None


@dataclass
class LimitedOverlapResult:
    motifs: list[Motif]
    placements: list[Placement]
    counts: dict[str, np.ndarray]
    coverage: float
    n_bins_covered: int
    n_bins_total: int
    stop_reason: str
    passes: list[dict[str, object]] = field(default_factory=list)


def _check_knobs(
    *,
    k: int,
    eps: float,
    w_min: int,
    w_max: int,
    m: int,
    n_min: int,
    refine_iters: int,
    tau: float,
    max_outer: int,
    min_coverage: float,
    n_components: int,
) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if eps < 0.0:
        raise ValueError(f"eps must be >= 0, got {eps}")
    if w_min < 1 or w_max < w_min:
        raise ValueError(f"need 1 <= w_min <= w_max, got {w_min}, {w_max}")
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    if n_min < 1:
        raise ValueError(f"n_min must be >= 1, got {n_min}")
    if refine_iters < 1:
        raise ValueError(f"refine_iters must be >= 1, got {refine_iters}")
    if max_outer < 1:
        raise ValueError(f"max_outer must be >= 1, got {max_outer}")
    if not 0.0 < float(min_coverage) <= 1.0:
        raise ValueError(f"min_coverage must be in (0, 1], got {min_coverage}")
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")


def _dot_products(spectrogram: np.ndarray, motif: np.ndarray) -> np.ndarray:
    """Dot product of a unit-norm motif with each in-range start column."""
    _frequency, n_time = spectrogram.shape
    width = int(motif.shape[1])
    n_valid = n_time - width + 1
    if n_valid <= 0:
        return np.zeros(0, dtype=np.float64)
    nfft = next_fast_len(n_time + width - 1)
    full = irfft(
        rfft(spectrogram, n=nfft, axis=1)
        * rfft(np.ascontiguousarray(motif[:, ::-1]), n=nfft, axis=1),
        n=nfft,
        axis=1,
    )
    return full[:, width - 1 : width - 1 + n_valid].sum(axis=0)


def _patch_energy(spectrogram: np.ndarray, width: int) -> np.ndarray:
    column = np.sum(spectrogram * spectrogram, axis=0)
    cumulative = np.cumsum(column)
    n_valid = column.size - width + 1
    energy = np.empty(n_valid, dtype=np.float64)
    energy[0] = cumulative[width - 1]
    if n_valid > 1:
        energy[1:] = cumulative[width:] - cumulative[: n_valid - 1]
    return energy


def _admissible(counts: np.ndarray, width: int, k: int) -> np.ndarray:
    n_valid = counts.size - width + 1
    if n_valid <= 0:
        return np.zeros(0, dtype=bool)
    if width == 1:
        return counts <= (k - 1)
    window_max = np.lib.stride_tricks.sliding_window_view(counts, width).max(axis=1)
    return window_max <= (k - 1)


def _explain(spectrogram: np.ndarray, motif: Motif) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Explained fraction and energy at each start.

    Without a stored mean, the fraction is the squared projection onto the
    motif basis. With a mean, it is the energy kept by the principal-component
    reconstruction ``mean + sum coeff_i component_i``.
    """
    basis = motif.basis
    energy = _patch_energy(spectrogram, motif.width)
    first = _dot_products(spectrogram, basis[0])
    if motif.mean is None:
        explained = np.zeros(energy.shape, dtype=np.float64)
        for index, component in enumerate(basis):
            dots = first if index == 0 else _dot_products(spectrogram, component)
            explained += dots * dots
    else:
        mean_energy = float(np.sum(motif.mean * motif.mean))
        mean_dots = _dot_products(spectrogram, motif.mean)
        centered_energy = energy + mean_energy - 2.0 * mean_dots
        projected = np.zeros(energy.shape, dtype=np.float64)
        for component in basis:
            dots = _dot_products(spectrogram, component)
            mean_coeff = float(np.sum(motif.mean * component))
            projected += (dots - mean_coeff) ** 2
        residual = np.maximum(centered_energy - projected, 0.0)
        explained = energy - residual
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = np.divide(explained, energy, out=np.full(explained.shape, -np.inf), where=energy > 0.0)
    return fraction, explained, first


def _pursuit(
    motifs: Sequence[Motif],
    windows: dict[str, np.ndarray],
    counts: dict[str, np.ndarray],
    *,
    k: int,
    eps: float,
) -> list[Placement]:
    """Greedy admissible placements. Increments ``counts`` as it accepts them."""
    prepared: list[tuple[Motif, str, np.ndarray, np.ndarray, np.ndarray]] = []
    for motif in motifs:
        for window_id, spectrogram in windows.items():
            if spectrogram.shape[0] != motif.n_freq:
                raise ValueError(
                    f"motif {motif.motif_id} has {motif.n_freq} frequency rows, "
                    f"window {window_id} has {spectrogram.shape[0]}"
                )
            if spectrogram.shape[1] < motif.width:
                continue
            fraction, explained, dots = _explain(spectrogram, motif)
            prepared.append((motif, window_id, fraction, explained, dots))
    if not prepared:
        return []

    accepted: list[Placement] = []
    first_energy: float | None = None
    while True:
        best: tuple[float, Motif, str, int, float, float] | None = None
        for motif, window_id, fraction, explained, dots in prepared:
            allowed = _admissible(counts[window_id], motif.width, k)
            if allowed.size == 0 or not np.any(allowed):
                continue
            masked = np.where(allowed, fraction, -np.inf)
            index = int(np.argmax(masked))
            score = float(masked[index])
            if not np.isfinite(score):
                continue
            if best is None or score > best[0]:
                best = (score, motif, window_id, index, float(explained[index]), float(dots[index]))
        if best is None:
            return accepted
        _score, motif, window_id, start, energy, alpha = best
        if first_energy is None:
            first_energy = energy
        elif energy < eps * first_energy:
            return accepted
        counts[window_id][start : start + motif.width] += 1
        unexplained = 1.0 - _score
        accepted.append(
            Placement(
                motif_id=motif.motif_id,
                window_id=window_id,
                t_start=start,
                amplitude=alpha,
                unexplained=unexplained,
            )
        )


def _group(placements: Sequence[Placement]) -> dict[int, list[Placement]]:
    grouped: dict[int, list[Placement]] = {}
    for placement in placements:
        grouped.setdefault(placement.motif_id, []).append(placement)
    return grouped


def _drop_small(motifs: list[Motif], placements: list[Placement], n_min: int) -> tuple[list[Motif], list[Placement]]:
    counts = {motif_id: len(group) for motif_id, group in _group(placements).items()}
    kept_ids = {motif.motif_id for motif in motifs if counts.get(motif.motif_id, 0) >= n_min}
    return (
        [motif for motif in motifs if motif.motif_id in kept_ids],
        [placement for placement in placements if placement.motif_id in kept_ids],
    )


def _principal_components(patches: Sequence[np.ndarray], n_components: int) -> np.ndarray:
    """Top orthonormal principal components, shape ``(k, F, W)``.

    ``k`` is at most ``n_components`` and at most the number of patches. Each
    component is unit norm and oriented to correlate positively with the mean
    patch. One component is the normalized mean when the patches do not vary.
    """
    stacked = np.stack([np.asarray(patch, dtype=np.float64) for patch in patches], axis=0)
    mean = np.mean(stacked, axis=0)
    k = min(int(n_components), len(patches))
    basis = np.zeros((k, *mean.shape), dtype=np.float64)
    centered = stacked - mean
    if len(patches) == 1 or float(np.linalg.norm(centered)) == 0.0:
        norm = float(np.linalg.norm(mean))
        if norm == 0.0:
            raise ValueError("principal component has zero norm")
        basis[0] = mean / norm
        return basis
    flat = centered.reshape(len(patches), -1)
    eigvals, eigvecs = np.linalg.eigh(flat @ flat.T)
    order = np.argsort(eigvals)[::-1]
    filled = 0
    for idx in order:
        lam = float(eigvals[idx])
        if lam <= 0.0 or filled == k:
            break
        component = (flat.T @ eigvecs[:, idx]) / np.sqrt(lam)
        if float(np.dot(component, mean.ravel())) < 0.0:
            component = -component
        basis[filled] = component.reshape(mean.shape)
        filled += 1
    if filled == 0:
        norm = float(np.linalg.norm(mean))
        if norm == 0.0:
            raise ValueError("principal component has zero norm")
        basis[0] = mean / norm
        return basis[:1]
    return basis[:filled]


def _crop(spectrogram: np.ndarray, start: int, width: int) -> np.ndarray:
    return np.array(spectrogram[:, start : start + width], copy=True)


def _rebuild(
    motifs: list[Motif],
    placements: list[Placement],
    windows: dict[str, np.ndarray],
    n_components: int,
) -> None:
    grouped = _group(placements)
    for motif in motifs:
        patches = [
            _crop(windows[placement.window_id], placement.t_start, motif.width)
            for placement in grouped.get(motif.motif_id, [])
        ]
        basis = _principal_components(patches, n_components)
        if n_components == 1:
            motif.data = basis[0]
            motif.mean = None
        else:
            motif.data = basis
            motif.mean = np.mean(np.stack(patches, axis=0), axis=0)


def _fit(patch: np.ndarray, motif: Motif) -> tuple[np.ndarray, float]:
    energy = float(np.sum(patch * patch))
    if energy <= 0.0:
        return np.zeros(motif.n_components, dtype=np.float64), 1.0
    if motif.mean is None:
        coeffs = np.array(
            [float(np.sum(patch * component)) for component in motif.basis], dtype=np.float64
        )
        unexplained = 1.0 - float(np.dot(coeffs, coeffs) / energy)
        return coeffs, unexplained
    delta = patch - motif.mean
    coeffs = np.array(
        [float(np.sum(delta * component)) for component in motif.basis], dtype=np.float64
    )
    reconstruction = motif.mean + sum(
        float(coeff) * component for coeff, component in zip(coeffs, motif.basis)
    )
    residual = float(np.sum((patch - reconstruction) ** 2))
    return coeffs, residual / energy


def _rescore(motifs: list[Motif], placements: list[Placement], windows: dict[str, np.ndarray]) -> None:
    by_id = {motif.motif_id: motif for motif in motifs}
    for placement in placements:
        motif = by_id[placement.motif_id]
        patch = windows[placement.window_id][:, placement.t_start : placement.t_start + motif.width]
        coeffs, unexplained = _fit(patch, motif)
        placement.coeffs = coeffs
        placement.amplitude = float(coeffs[0] if coeffs.size == 1 else np.linalg.norm(coeffs))
        placement.unexplained = unexplained


def _apply_tau(
    motifs: list[Motif], placements: list[Placement], tau: float, n_min: int
) -> tuple[list[Motif], list[Placement]]:
    kept = [placement for placement in placements if placement.unexplained <= tau]
    return _drop_small(motifs, kept, n_min)


def _recompute_counts(
    windows: dict[str, np.ndarray],
    placements: Sequence[Placement],
    widths: dict[int, int],
) -> dict[str, np.ndarray]:
    counts = {window_id: np.zeros(spectrogram.shape[1], dtype=np.int32) for window_id, spectrogram in windows.items()}
    for placement in placements:
        width = widths[placement.motif_id]
        counts[placement.window_id][placement.t_start : placement.t_start + width] += 1
    return counts


def _coverage(counts: dict[str, np.ndarray]) -> tuple[float, int, int]:
    covered = 0
    total = 0
    for column in counts.values():
        total += int(column.size)
        covered += int(np.count_nonzero(column))
    if total == 0:
        return 0.0, 0, 0
    return covered / total, covered, total


def _eligible(segments, counts: dict[str, np.ndarray], w_min: int, w_max: int, k: int):
    chosen = []
    for segment in segments:
        width = int(segment.t_end - segment.t_start)
        if width < w_min or width > w_max:
            continue
        cover = counts[segment.window_id][segment.t_start : segment.t_end]
        if cover.size != width or np.any(cover > (k - 1)):
            continue
        chosen.append(segment)
    return chosen


def learn_limited_overlap(
    windows: Sequence[Window],
    detections: Sequence[Detection],
    *,
    k: int = DEFAULT_K,
    eps: float = DEFAULT_EPS,
    w_min: int = DEFAULT_W_MIN,
    w_max: int = DEFAULT_W_MAX,
    m: int = DEFAULT_M,
    n_min: int = DEFAULT_N_MIN,
    refine_iters: int = DEFAULT_REFINE_ITERS,
    tau: float = DEFAULT_TAU,
    max_outer: int = DEFAULT_MAX_OUTER,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    n_components: int = DEFAULT_N_COMPONENTS,
    rng: np.random.Generator | int | None = None,
) -> LimitedOverlapResult:
    """Learn motifs by the limited-overlap create, pursuit, and refine loop."""
    _check_knobs(
        k=k,
        eps=eps,
        w_min=w_min,
        w_max=w_max,
        m=m,
        n_min=n_min,
        refine_iters=refine_iters,
        tau=tau,
        max_outer=max_outer,
        min_coverage=min_coverage,
        n_components=n_components,
    )
    ordered = sorted(windows, key=lambda window: (window.start_sample, window.window_id))
    spectrograms = {window.window_id: np.asarray(window.data, dtype=np.float64) for window in ordered}
    counts = {window_id: np.zeros(data.shape[1], dtype=np.int32) for window_id, data in spectrograms.items()}
    segments = detections_to_segments(ordered, detections)
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)

    dictionary: list[Motif] = []
    kept: list[Placement] = []
    logs: list[dict[str, object]] = []
    next_id = 0
    stop_reason = "max_outer"

    for outer in range(1, max_outer + 1):
        pool = _eligible(segments, counts, w_min, w_max, k)
        if not pool:
            stop_reason = "no_seeds"
            break
        take = min(m, len(pool))
        picked = [pool[int(i)] for i in generator.choice(len(pool), size=take, replace=False)]
        seeds: list[Motif] = []
        for segment in picked:
            patch = np.asarray(segment.data, dtype=np.float64)
            norm = float(np.linalg.norm(patch))
            seeds.append(
                Motif(
                    motif_id=next_id,
                    data=patch / norm,
                    window_id=segment.window_id,
                    t_start=int(segment.t_start),
                )
            )
            next_id += 1

        pass_placements: list[Placement] = []
        for _round in range(refine_iters):
            working = {window_id: column.copy() for window_id, column in counts.items()}
            fresh = _pursuit(seeds, spectrograms, working, k=k, eps=eps)
            pass_placements.extend(fresh)
            seeds, pass_placements = _drop_small(seeds, pass_placements, n_min)
            if seeds:
                _rebuild(seeds, pass_placements, spectrograms, n_components)
                _rescore(seeds, pass_placements, spectrograms)
                seeds, pass_placements = _apply_tau(seeds, pass_placements, tau, n_min)
            widths = {motif.motif_id: motif.width for motif in dictionary}
            widths.update({motif.motif_id: motif.width for motif in seeds})
            counts = _recompute_counts(spectrograms, [*kept, *pass_placements], widths)

        if not seeds:
            stop_reason = "no_survivors"
            logs.append({"outer": outer, "n_sampled": len(picked), "n_motifs": 0, "n_placements": 0})
            break

        dictionary.extend(seeds)
        kept.extend(pass_placements)
        coverage, n_covered, n_total = _coverage(counts)
        logs.append(
            {
                "outer": outer,
                "n_sampled": len(picked),
                "n_motifs": len(seeds),
                "n_placements": len(pass_placements),
                "coverage": coverage,
            }
        )
        if coverage > float(min_coverage):
            stop_reason = "coverage"
            break

    coverage, n_covered, n_total = _coverage(counts)
    return LimitedOverlapResult(
        motifs=dictionary,
        placements=kept,
        counts=counts,
        coverage=coverage,
        n_bins_covered=n_covered,
        n_bins_total=n_total,
        stop_reason=stop_reason,
        passes=logs,
    )
