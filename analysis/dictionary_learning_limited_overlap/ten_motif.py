"""Search for a 20-motif limited-overlap dictionary with a residual gap.

See ``ten_motif_plan.md``. A dictionary is accepted only when it has exactly
20 motifs of 5 components, placement counts within a factor of 4, coverage
above 0.3, and every motif's mean residual gap is at least 0.1.
"""

from __future__ import annotations

import copy
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from analysis.dict_learning.code.cli import load_detections, load_windows
from analysis.dict_learning.code.io import detections_to_segments
from analysis.dictionary_learning_limited_overlap.learn import (
    Motif,
    Placement,
    _apply_tau,
    _coverage,
    _drop_small,
    _eligible,
    _principal_components,
    _pursuit,
    _rebuild,
    _recompute_counts,
    _rescore,
    learn_limited_overlap,
)
from analysis.evaluation.core import _dot_products, _peak_ratios, _specificity_p90

GPL = Path("/home/yfreund/whale_processing/result/LeiData/SOCAL_H_76_230427_014845.x.gpl/output")
OUT = Path(
    "/home/yfreund/whale_processing/result/LeiData/"
    "SOCAL_H_76_230427_014845.x.limited_overlap_twenty_motif_nomerge"
)

K_TRIALS = 20
S_REFINE = 3
SPLIT_EPSILON = 0.1
MAX_EDITS = 24
TARGET_MOTIFS = 20
TARGET_COMPONENTS = 5
TARGET_RATIO = 4.0
TARGET_COVERAGE = 0.3
TARGET_GAP = 0.1


@dataclass
class MotifGap:
    motif_id: int
    n_placements: int
    n_components: int
    mean_own: float
    mean_runner: float
    mean_gap: float
    runner_id: int | None


@dataclass
class Report:
    motifs: list[MotifGap] = field(default_factory=list)
    coverage: float = 0.0
    n_bins_covered: int = 0
    n_bins_total: int = 0
    runner_of: dict[int, int | None] = field(default_factory=dict)

    @property
    def n_motifs(self) -> int:
        return len(self.motifs)

    @property
    def min_components(self) -> int:
        if not self.motifs:
            return 0
        return min(row.n_components for row in self.motifs)

    @property
    def ratio(self) -> float:
        counts = [row.n_placements for row in self.motifs if row.n_placements > 0]
        if not counts:
            return float("inf")
        return max(counts) / min(counts)

    @property
    def min_gap(self) -> float:
        if not self.motifs:
            return float("-inf")
        return min(row.mean_gap for row in self.motifs)

    def accepted(self) -> bool:
        return (
            self.n_motifs == TARGET_MOTIFS
            and self.min_components >= TARGET_COMPONENTS
            and self.ratio <= TARGET_RATIO
            and self.coverage > TARGET_COVERAGE
            and self.min_gap >= TARGET_GAP
        )

    def score(self) -> tuple[int, float]:
        fails = (
            int(self.n_motifs != TARGET_MOTIFS)
            + int(self.min_components < TARGET_COMPONENTS)
            + int(self.ratio > TARGET_RATIO)
            + int(self.coverage <= TARGET_COVERAGE)
            + int(self.min_gap < TARGET_GAP)
        )
        penalty = (
            abs(self.n_motifs - TARGET_MOTIFS) / TARGET_MOTIFS
            + max(0, TARGET_COMPONENTS - self.min_components) / TARGET_COMPONENTS
            + max(0.0, self.ratio - TARGET_RATIO) / TARGET_RATIO
            + max(0.0, TARGET_COVERAGE - self.coverage) / TARGET_COVERAGE
            + max(0.0, TARGET_GAP - self.min_gap) / TARGET_GAP
        )
        return fails, float(penalty)

    def checks(self) -> dict[str, dict[str, object]]:
        return {
            "n_motifs": {"value": self.n_motifs, "ok": self.n_motifs == TARGET_MOTIFS},
            "n_components": {"value": self.min_components, "ok": self.min_components >= TARGET_COMPONENTS},
            "count_ratio": {"value": None if self.ratio == float("inf") else self.ratio, "ok": self.ratio <= TARGET_RATIO},
            "coverage": {"value": self.coverage, "ok": self.coverage > TARGET_COVERAGE},
            "min_gap": {"value": self.min_gap, "ok": self.min_gap >= TARGET_GAP},
        }


def unexplained(patch: np.ndarray, motif: Motif) -> float:
    """Fraction of patch energy left after the motif reconstruction."""
    energy = float(np.sum(patch * patch))
    if energy <= 0.0:
        return 1.0
    basis = motif.basis
    if motif.mean is None:
        coeffs = np.tensordot(basis, patch, axes=([1, 2], [0, 1]))
        return max(0.0, 1.0 - float(np.dot(coeffs, coeffs) / energy))
    delta = patch - motif.mean
    coeffs = np.tensordot(basis, delta, axes=([1, 2], [0, 1]))
    numer = float(np.sum(delta * delta) - np.dot(coeffs, coeffs))
    return max(0.0, numer) / energy


def _explained_batch(patches: np.ndarray, motif: Motif) -> np.ndarray:
    """Explained energy of each patch. ``patches`` has shape ``(N, F, W)``."""
    energy = np.sum(patches * patches, axis=(1, 2))
    basis = motif.basis
    if motif.mean is None:
        coeffs = np.tensordot(patches, basis, axes=([1, 2], [1, 2]))
        return np.sum(coeffs * coeffs, axis=1)
    delta = patches - motif.mean
    coeffs = np.tensordot(delta, basis, axes=([1, 2], [1, 2]))
    delta_energy = np.sum(delta * delta, axis=(1, 2))
    return energy - delta_energy + np.sum(coeffs * coeffs, axis=1)


def score_dictionary(
    motifs: list[Motif],
    placements: list[Placement],
    spectrograms: dict[str, np.ndarray],
) -> Report:
    """Mean advantage of each motif over the best other motif, on its own placements."""
    by_id = {motif.motif_id: motif for motif in motifs}
    own_gaps: dict[int, list[float]] = {motif.motif_id: [] for motif in motifs}
    own_values: dict[int, list[float]] = {motif.motif_id: [] for motif in motifs}
    runner_values: dict[int, list[float]] = {motif.motif_id: [] for motif in motifs}
    runners: dict[int, list[int]] = {motif.motif_id: [] for motif in motifs}
    for placement in placements:
        motif = by_id.get(placement.motif_id)
        if motif is None:
            continue
        spec = spectrograms[placement.window_id]
        scored: list[tuple[float, int]] = []
        for candidate in motifs:
            if placement.t_start + candidate.width > spec.shape[1]:
                continue
            patch = spec[:, placement.t_start : placement.t_start + candidate.width]
            scored.append((unexplained(patch, candidate), candidate.motif_id))
        if len(scored) < 2:
            continue
        own = next(value for value, motif_id in scored if motif_id == placement.motif_id)
        others = [(value, motif_id) for value, motif_id in scored if motif_id != placement.motif_id]
        if not others:
            continue
        runner_value, runner_id = min(others)
        own_values[placement.motif_id].append(own)
        runner_values[placement.motif_id].append(runner_value)
        own_gaps[placement.motif_id].append(runner_value - own)
        runners[placement.motif_id].append(runner_id)

    rows: list[MotifGap] = []
    runner_of: dict[int, int | None] = {}
    for motif in motifs:
        gaps = own_gaps[motif.motif_id]
        votes = Counter(runners[motif.motif_id])
        runner_id = votes.most_common(1)[0][0] if votes else None
        runner_of[motif.motif_id] = runner_id
        rows.append(
            MotifGap(
                motif_id=motif.motif_id,
                n_placements=sum(placement.motif_id == motif.motif_id for placement in placements),
                n_components=motif.n_components,
                mean_own=float(np.mean(own_values[motif.motif_id])) if gaps else 1.0,
                mean_runner=float(np.mean(runner_values[motif.motif_id])) if gaps else 1.0,
                mean_gap=float(np.mean(gaps)) if gaps else float("-inf"),
                runner_id=runner_id,
            )
        )
    widths = {motif.motif_id: motif.width for motif in motifs}
    counts = _recompute_counts(spectrograms, placements, widths)
    coverage, n_covered, n_total = _coverage(counts)
    return Report(
        motifs=rows,
        coverage=coverage,
        n_bins_covered=n_covered,
        n_bins_total=n_total,
        runner_of=runner_of,
    )


def refine(
    motifs: list[Motif],
    spectrograms: dict[str, np.ndarray],
    *,
    rounds: int = 2,
    k: int = 1,
    eps: float = 0.01,
    n_min: int = 10,
    tau: float = 0.4,
    n_components: int = 5,
) -> tuple[list[Motif], list[Placement]]:
    """Two pursuit, rebuild, and tau rounds. Counts start empty."""
    seeds = list(motifs)
    counts = {window_id: np.zeros(data.shape[1], dtype=np.int32) for window_id, data in spectrograms.items()}
    kept: list[Placement] = []
    for _round in range(rounds):
        working = {window_id: column.copy() for window_id, column in counts.items()}
        fresh = _pursuit(seeds, spectrograms, working, k=k, eps=eps)
        kept.extend(fresh)
        seeds, kept = _drop_small(seeds, kept, n_min)
        if seeds:
            _rebuild(seeds, kept, spectrograms, n_components)
            _rescore(seeds, kept, spectrograms)
            seeds, kept = _apply_tau(seeds, kept, tau, n_min)
        widths = {motif.motif_id: motif.width for motif in seeds}
        counts = _recompute_counts(spectrograms, kept, widths)
        if not seeds:
            break
    return seeds, kept


def _patches_of(
    motif: Motif,
    placements: list[Placement],
    spectrograms: dict[str, np.ndarray],
) -> tuple[list[np.ndarray], list[Placement]]:
    chosen = [placement for placement in placements if placement.motif_id == motif.motif_id]
    patches = [
        np.array(
            spectrograms[placement.window_id][:, placement.t_start : placement.t_start + motif.width],
            dtype=np.float64,
            copy=True,
        )
        for placement in chosen
    ]
    return patches, chosen


def _fit_child(motif_id: int, patches: list[np.ndarray], parent: Motif, n_components: int) -> Motif:
    basis = _principal_components(patches, n_components)
    mean = np.mean(np.stack(patches, axis=0), axis=0)
    return Motif(
        motif_id=motif_id,
        data=basis if n_components > 1 else basis[0],
        window_id=parent.window_id,
        t_start=parent.t_start,
        mean=None if n_components == 1 else mean,
    )


def binary_split(
    parent: Motif,
    placements: list[Placement],
    spectrograms: dict[str, np.ndarray],
    *,
    rng: np.random.Generator,
    n_components: int,
    n_min: int,
    next_id: int,
) -> tuple[list[Motif], list[Placement]] | None:
    """K trials of a ±ε perturbation. Keep the split with the smallest residual SSE."""
    patches, chosen = _patches_of(parent, placements, spectrograms)
    if len(patches) < 2 * n_min:
        return None
    stacked = np.stack(patches, axis=0)
    center = np.mean(stacked, axis=0)
    norm = float(np.linalg.norm(center))
    if norm == 0.0:
        return None
    word = center / norm
    best: tuple[float, list[Motif]] | None = None
    for _trial in range(K_TRIALS):
        direction = rng.standard_normal(word.shape)
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm == 0.0:
            continue
        step = direction / direction_norm * (SPLIT_EPSILON * float(np.linalg.norm(word)))
        children = [
            Motif(next_id, _unit(word + step), parent.window_id, parent.t_start),
            Motif(next_id + 1, _unit(word - step), parent.window_id, parent.t_start),
        ]
        labels = np.zeros(len(patches), dtype=int)
        feasible = True
        for _step in range(S_REFINE):
            left = _explained_batch(stacked, children[0])
            right = _explained_batch(stacked, children[1])
            labels = (right > left).astype(int)
            if int(labels.sum()) < n_min or int(len(labels) - labels.sum()) < n_min:
                feasible = False
                break
            for label in (0, 1):
                subset = [patches[index] for index in range(len(patches)) if int(labels[index]) == label]
                children[label] = _fit_child(next_id + label, subset, parent, n_components)
        if not feasible:
            continue
        explained = np.where(labels == 0, _explained_batch(stacked, children[0]), _explained_batch(stacked, children[1]))
        energy = np.sum(stacked * stacked, axis=(1, 2))
        sse = float(np.sum(energy - explained))
        if best is None or sse < best[0]:
            best = (sse, children)
    if best is None:
        return None
    return best[1], [placement for placement in placements if placement.motif_id != parent.motif_id]


def _unit(array: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(array))
    if norm == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return array / norm


def median_split(
    parent: Motif,
    placements: list[Placement],
    spectrograms: dict[str, np.ndarray],
    *,
    n_components: int,
    n_min: int,
    next_id: int,
) -> tuple[list[Motif], list[Placement]] | None:
    """One median cut on the component that most reduces within-half energy."""
    patches, _chosen = _patches_of(parent, placements, spectrograms)
    if len(patches) < 2 * n_min or parent.n_components < 1:
        return None
    mean = parent.mean if parent.mean is not None else np.mean(np.stack(patches, axis=0), axis=0)
    coeffs = np.stack(
        [
            np.tensordot(parent.basis, patch - mean, axes=([1, 2], [0, 1]))
            for patch in patches
        ]
    )
    grand = np.mean(np.stack(patches, axis=0), axis=0)
    total = float(sum(np.sum((patch - grand) ** 2) for patch in patches))
    best_index = 0
    best_reduction = -1.0
    for component in range(coeffs.shape[1]):
        order = np.argsort(coeffs[:, component], kind="mergesort")
        mid = len(order) // 2
        reduction = total
        for group in (order[:mid], order[mid:]):
            group_mean = np.mean(np.stack([patches[int(index)] for index in group], axis=0), axis=0)
            reduction -= float(sum(np.sum((patches[int(index)] - group_mean) ** 2) for index in group))
        if reduction > best_reduction:
            best_reduction = reduction
            best_index = component
    order = np.argsort(coeffs[:, best_index], kind="mergesort")
    mid = len(order) // 2
    groups = [order[:mid], order[mid:]]
    if len(groups[0]) < n_min or len(groups[1]) < n_min:
        return None
    children = [
        _fit_child(next_id + label, [patches[int(index)] for index in group], parent, n_components)
        for label, group in enumerate(groups)
    ]
    return children, [placement for placement in placements if placement.motif_id != parent.motif_id]


def _replace(
    motifs: list[Motif],
    parent_id: int,
    children: list[Motif],
) -> list[Motif]:
    return [motif for motif in motifs if motif.motif_id != parent_id] + children


def _drop_motif(motifs: list[Motif], motif_id: int) -> list[Motif]:
    return [motif for motif in motifs if motif.motif_id != motif_id]


def _format_report(report: Report) -> str:
    ratio = "inf" if report.ratio == float("inf") else f"{report.ratio:.2f}"
    return (
        f"motifs {report.n_motifs}  components {report.min_components}  "
        f"ratio {ratio}  coverage {report.coverage:.3f}  min_gap {report.min_gap:.3f}  "
        f"score {report.score()}"
    )


def _specificity(
    motifs: list[Motif],
    windows: list,
    spectrograms: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    """90th percentile of peak ratios after the one-minute dot-product shift and scale."""
    ordered = sorted(windows, key=lambda window: (window.start_sample, window.window_id))
    spectrogram = np.concatenate([spectrograms[window.window_id] for window in ordered], axis=1)
    sec_per_bin = float(ordered[0].overlap) / float(ordered[0].sample_freq)
    shift_sigma = 60.0 / sec_per_bin
    rows: list[dict[str, object]] = []
    for motif in motifs:
        kernel = motif.mean if motif.mean is not None else motif.basis[0]
        ratios, percentile = _specificity_p90(
            _peak_ratios(_dot_products(spectrogram, kernel), float(motif.width // 2), shift_sigma)
        )
        rows.append(
            {
                "motif_id": motif.motif_id,
                "k": motif.width // 2,
                "p90": None if not np.isfinite(percentile) else float(percentile),
                "n": int(ratios.size),
            }
        )
    return rows


def _children_clear(report: Report, child_ids: set[int]) -> bool:
    rows = {row.motif_id: row for row in report.motifs}
    return all(motif_id in rows and rows[motif_id].mean_gap >= TARGET_GAP for motif_id in child_ids)


def _add_coverage_pass(
    motifs: list[Motif],
    placements: list[Placement],
    windows: list,
    detections: list,
    rng: np.random.Generator,
) -> tuple[list[Motif], list[Placement]] | None:
    """One limited-overlap pass on columns the current dictionary does not cover."""
    ordered = sorted(windows, key=lambda window: (window.start_sample, window.window_id))
    spectrograms = {window.window_id: np.asarray(window.data, dtype=np.float64) for window in ordered}
    counts = _recompute_counts(
        spectrograms, placements, {motif.motif_id: motif.width for motif in motifs}
    )
    segments = detections_to_segments(ordered, detections)
    pool = _eligible(segments, counts, 10, 40, 1)
    if not pool:
        return None
    take = min(10, len(pool))
    picked = [pool[int(index)] for index in rng.choice(len(pool), size=take, replace=False)]
    next_id = max(motif.motif_id for motif in motifs) + 1
    seeds: list[Motif] = []
    for segment in picked:
        patch = np.asarray(segment.data, dtype=np.float64)
        norm = float(np.linalg.norm(patch))
        if norm == 0.0:
            continue
        seeds.append(Motif(next_id, patch / norm, segment.window_id, int(segment.t_start)))
        next_id += 1
    if not seeds:
        return None
    fresh: list[Placement] = []
    for _round in range(1):
        working = {window_id: column.copy() for window_id, column in counts.items()}
        fresh = _pursuit(seeds, spectrograms, working, k=1, eps=0.01)
        seeds, fresh = _drop_small(seeds, fresh, 10)
        if seeds:
            _rebuild(seeds, fresh, spectrograms, TARGET_COMPONENTS)
            _rescore(seeds, fresh, spectrograms)
            seeds, fresh = _apply_tau(seeds, fresh, 0.4, 10)
        if not seeds:
            return None
    return motifs + seeds, placements + fresh


def search() -> dict[str, object]:
    windows = load_windows(GPL, None, 0)
    detections = load_detections(GPL)
    print("learning initial dictionary", flush=True)
    learned = learn_limited_overlap(
        windows,
        detections,
        k=1,
        w_min=10,
        w_max=40,
        m=10,
        n_min=10,
        max_outer=50,
        min_coverage=TARGET_COVERAGE,
        n_components=TARGET_COMPONENTS,
        tau=0.4,
        eps=0.01,
        rng=0,
    )
    spectrograms = {window.window_id: np.asarray(window.data, dtype=np.float64) for window in windows}
    motifs = learned.motifs
    placements = learned.placements
    rng = np.random.default_rng(0)
    edits: list[dict[str, object]] = []
    report = score_dictionary(motifs, placements, spectrograms)
    print(f"initial  {_format_report(report)}", flush=True)
    stop_reason = "max_edits"
    tried: set[tuple[str, int]] = set()

    for _step in range(MAX_EDITS):
        if report.accepted():
            stop_reason = "accepted"
            break
        snapshot = (copy.deepcopy(motifs), copy.deepcopy(placements), report)
        improved = False

        if report.coverage <= TARGET_COVERAGE:
            added = _add_coverage_pass(motifs, placements, windows, detections, rng)
            if added is not None:
                candidate, candidate_placements = added
                candidate_report = score_dictionary(candidate, candidate_placements, spectrograms)
                print(f"add pass  {_format_report(candidate_report)}", flush=True)
                edits.append({"action": "add", "kept": candidate_report.coverage > report.coverage})
                if candidate_report.coverage > report.coverage:
                    motifs, placements, report = candidate, candidate_placements, candidate_report
                    improved = True
                else:
                    motifs, placements, report = snapshot

        if not improved and (report.n_motifs < TARGET_MOTIFS or report.ratio > TARGET_RATIO):
            largest = max(report.motifs, key=lambda row: row.n_placements)
            parent = next(motif for motif in motifs if motif.motif_id == largest.motif_id)
            next_id = max(motif.motif_id for motif in motifs) + 1
            for action, splitter in (
                ("binary", lambda: binary_split(parent, placements, spectrograms, rng=rng, n_components=5, n_min=10, next_id=next_id)),
                ("median", lambda: median_split(parent, placements, spectrograms, n_components=5, n_min=10, next_id=next_id)),
            ):
                key = (action, parent.motif_id)
                if key in tried:
                    continue
                tried.add(key)
                produced = splitter()
                if produced is None:
                    print(f"{action} split of {parent.motif_id} produced no children", flush=True)
                    edits.append({"action": action, "motif_id": parent.motif_id, "kept": False})
                    continue
                children, _remaining = produced
                candidate = _replace(motifs, parent.motif_id, children)
                candidate, candidate_placements = refine(candidate, spectrograms)
                candidate_report = score_dictionary(candidate, candidate_placements, spectrograms)
                child_ids = {child.motif_id for child in children}
                clear = _children_clear(candidate_report, child_ids)
                better = candidate_report.score() < report.score()
                print(
                    f"{action} split {parent.motif_id}  clear {clear}  {_format_report(candidate_report)}",
                    flush=True,
                )
                edits.append({"action": action, "motif_id": parent.motif_id, "kept": bool(clear and better)})
                if clear and better:
                    motifs, placements, report = candidate, candidate_placements, candidate_report
                    improved = True
                    break
                motifs, placements, report = snapshot
            if improved:
                pass

        if not improved:
            stop_reason = "no_improvement"
            break
    else:
        if report.accepted():
            stop_reason = "accepted"

    if report.accepted():
        stop_reason = "accepted"
    return {
        "motifs": motifs,
        "placements": placements,
        "spectrograms": spectrograms,
        "report": report,
        "edits": edits,
        "stop_reason": stop_reason,
        "learned_stop": learned.stop_reason,
        "passes": learned.passes,
        "specificity": _specificity(motifs, windows, spectrograms),
    }


def save_result(result: dict[str, object]) -> None:
    motifs: list[Motif] = result["motifs"]
    placements: list[Placement] = result["placements"]
    spectrograms: dict[str, np.ndarray] = result["spectrograms"]
    report: Report = result["report"]
    OUT.mkdir(parents=True, exist_ok=True)
    atoms = OUT / "atoms"
    atoms.mkdir(exist_ok=True)
    widths = {motif.motif_id: motif.width for motif in motifs}
    motif_rows = []
    by_gap = {row.motif_id: row for row in report.motifs}
    for motif in motifs:
        name = f"template_{motif.motif_id:03d}.npy"
        np.save(atoms / name, motif.data)
        if motif.mean is not None:
            np.save(atoms / f"mean_{motif.motif_id:03d}.npy", motif.mean)
        gap = by_gap[motif.motif_id]
        motif_rows.append(
            {
                "motif_id": motif.motif_id,
                "width": motif.width,
                "n_freq": motif.n_freq,
                "n_components": motif.n_components,
                "n_placements": gap.n_placements,
                "mean_own": gap.mean_own,
                "mean_runner": gap.mean_runner,
                "mean_gap": gap.mean_gap,
                "runner_id": gap.runner_id,
                "seed_window_id": motif.window_id,
                "seed_t_start": motif.t_start,
                "path": f"atoms/{name}",
            }
        )
    matchings = []
    for placement in placements:
        row = {
            "motif_id": placement.motif_id,
            "window_id": placement.window_id,
            "t_start": placement.t_start,
            "width": widths[placement.motif_id],
            "amplitude": placement.amplitude,
            "unexplained": placement.unexplained,
        }
        if placement.coeffs is not None and placement.coeffs.size > 1:
            row["coeffs"] = [float(value) for value in placement.coeffs]
        matchings.append(row)
    (OUT / "matchings.json").write_text(json.dumps(matchings) + "\n")
    counts = _recompute_counts(spectrograms, placements, widths)
    np.savez_compressed(OUT / "counts.npz", **counts)
    summary = {
        "command": "ten-motif",
        "gpl_dir": str(GPL.resolve()),
        "stop_reason": result["stop_reason"],
        "accepted": report.accepted(),
        "learned_stop": result["learned_stop"],
        "coverage": report.coverage,
        "n_bins_covered": report.n_bins_covered,
        "n_bins_total": report.n_bins_total,
        "n_motifs": report.n_motifs,
        "n_placements": len(placements),
        "max_count": max((int(column.max()) for column in counts.values() if column.size), default=0),
        "checks": report.checks(),
        "knobs": {
            "k": 1,
            "eps": 0.01,
            "w_min": 10,
            "w_max": 40,
            "m": 10,
            "n_min": 10,
            "refine_iters": 2,
            "tau": 0.4,
            "max_outer": 50,
            "min_coverage": TARGET_COVERAGE,
            "n_components": TARGET_COMPONENTS,
            "seed": 0,
            "dot_shift_minutes": 1.0,
        },
        "passes": result["passes"],
        "specificity": result["specificity"],
        "edits": result["edits"],
        "motifs": motif_rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {OUT}  accepted {report.accepted()}  stop {result['stop_reason']}", flush=True)


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} already exists")
    save_result(search())


if __name__ == "__main__":
    main()
