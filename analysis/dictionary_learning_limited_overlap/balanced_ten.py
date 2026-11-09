"""Build 10 motifs with balanced column coverage and a residual gap of 0.1.

Starts from the tau=0.4 dictionary. Each source motif keeps the placements on
which it already leads the other motifs by 0.1, or all of its placements when
that mean lead is already 0.1. Further motifs are drawn from detection crops
the dictionary does not explain, and from leftover placements, and are kept
only when every motif's mean lead stays at least 0.1 and the covered-column
counts stay within a factor of four.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from analysis.dict_learning.code.cli import load_detections, load_windows
from analysis.dict_learning.code.io import detections_to_segments
from analysis.dictionary_learning_limited_overlap.learn import (
    Motif,
    Placement,
    _coverage,
    _pursuit,
    _rebuild,
    _recompute_counts,
    _rescore,
)
from analysis.dictionary_learning_limited_overlap.ten_motif import score_dictionary, unexplained

GPL = Path("/home/yfreund/whale_processing/result/LeiData/SOCAL_H_76_230427_014845.x.gpl/output")
SOURCE = Path(
    "/home/yfreund/whale_processing/result/LeiData/"
    "SOCAL_H_76_230427_014845.x.limited_overlap_twenty_motif_gap01"
)
OUT = Path(
    "/home/yfreund/whale_processing/result/LeiData/"
    "SOCAL_H_76_230427_014845.x.limited_overlap_ten_balanced"
)

TARGET_MOTIFS = 10
TARGET_GAP = 0.1
TARGET_RATIO = 4.0
TARGET_COMPONENTS = 5
MIN_PATCHES = 5
MAX_SEED_TRIALS = 80
WIDTH_MIN = 10
WIDTH_MAX = 40


def _load_source() -> tuple[list[Motif], list[Placement], dict[int, int]]:
    summary = json.loads((SOURCE / "summary.json").read_text())
    matchings = json.loads((SOURCE / "matchings.json").read_text())
    motifs = []
    for row in summary["motifs"]:
        data = np.load(SOURCE / row["path"])
        mean = np.load(SOURCE / "atoms" / f"mean_{row['motif_id']:03d}.npy")
        motifs.append(Motif(row["motif_id"], data, row["seed_window_id"], int(row["seed_t_start"]), mean))
    placements = [
        Placement(
            motif_id=int(row["motif_id"]),
            window_id=row["window_id"],
            t_start=int(row["t_start"]),
            amplitude=float(row["amplitude"]),
            unexplained=float(row["unexplained"]),
            coeffs=np.asarray(row["coeffs"], dtype=np.float64) if "coeffs" in row else None,
        )
        for row in matchings
    ]
    widths = {int(row["motif_id"]): int(row["width"]) for row in summary["motifs"]}
    return motifs, placements, widths


def _columns(motif: Motif, placements: list[Placement]) -> int:
    covered: dict[str, set[int]] = {}
    for placement in placements:
        if placement.motif_id != motif.motif_id:
            continue
        covered.setdefault(placement.window_id, set()).update(
            range(placement.t_start, placement.t_start + motif.width)
        )
    return sum(len(columns) for columns in covered.values())


def _band(motifs: list[Motif], placements: list[Placement]) -> tuple[int, int]:
    counts = [_columns(motif, placements) for motif in motifs]
    return math.ceil(max(counts) / TARGET_RATIO), math.floor(min(counts) * TARGET_RATIO)


def _ratio(motifs: list[Motif], placements: list[Placement]) -> float:
    counts = [_columns(motif, placements) for motif in motifs]
    return max(counts) / min(counts)


def _refit(motif: Motif, placements: list[Placement], spectrograms: dict[str, np.ndarray]) -> None:
    own = [placement for placement in placements if placement.motif_id == motif.motif_id]
    _rebuild([motif], own, spectrograms, TARGET_COMPONENTS)


def _placement_gaps(
    motif: Motif,
    own: list[Placement],
    motifs: list[Motif],
    spectrograms: dict[str, np.ndarray],
) -> list[tuple[float, Placement]]:
    scored: list[tuple[float, Placement]] = []
    for placement in own:
        spec = spectrograms[placement.window_id]
        own_value = None
        others: list[float] = []
        for candidate in motifs:
            if placement.t_start + candidate.width > spec.shape[1]:
                continue
            patch = spec[:, placement.t_start : placement.t_start + candidate.width]
            value = unexplained(patch, candidate)
            if candidate.motif_id == motif.motif_id:
                own_value = value
            else:
                others.append(value)
        if own_value is None or not others:
            continue
        scored.append((min(others) - own_value, placement))
    return scored


def _prefix_for_band(
    motif: Motif,
    ordered: list[Placement],
    motifs: list[Motif],
    placements: list[Placement],
) -> list[Placement]:
    """Leading placements whose covered columns land inside the current band."""
    low, high = _band(motifs, placements)
    if low > high:
        return []
    chosen: list[Placement] = []
    covered: dict[str, set[int]] = {}
    for placement in ordered:
        columns = covered.setdefault(placement.window_id, set())
        incoming = set(range(placement.t_start, placement.t_start + motif.width))
        used = sum(len(values) for values in covered.values())
        if used + len(incoming - columns) > high:
            break
        columns.update(incoming)
        chosen.append(placement)
    used = sum(len(values) for values in covered.values()) if chosen else 0
    if used < low or len(chosen) < MIN_PATCHES:
        return []
    return chosen


def _try_add(
    motif: Motif,
    ordered: list[Placement],
    motifs: list[Motif],
    placements: list[Placement],
    spectrograms: dict[str, np.ndarray],
) -> tuple[list[Motif], list[Placement]] | None:
    chosen = _prefix_for_band(motif, ordered, motifs, placements)
    if not chosen:
        return None
    trial_motifs = motifs + [motif]
    trial_placements = placements + chosen
    low, _high = _band(motifs, placements)
    for _ in range(10):
        _refit(motif, trial_placements, spectrograms)
        if motif.n_components != TARGET_COMPONENTS:
            return None
        own = [placement for placement in trial_placements if placement.motif_id == motif.motif_id]
        _rescore([motif], own, spectrograms)
        report = score_dictionary(trial_motifs, trial_placements, spectrograms)
        gaps_ok = all(row.mean_gap >= TARGET_GAP and row.n_placements >= MIN_PATCHES for row in report.motifs)
        if gaps_ok and _ratio(trial_motifs, trial_placements) <= TARGET_RATIO:
            return trial_motifs, trial_placements
        scored = _placement_gaps(motif, own, trial_motifs, spectrograms)
        if len(scored) <= MIN_PATCHES:
            return None
        scored.sort(key=lambda item: item[0])
        dropped = scored[0][1]
        trial_placements = [placement for placement in trial_placements if placement is not dropped]
        if _columns(motif, trial_placements) < low:
            return None
    return None


def _print_state(
    label: str, motifs: list[Motif], placements: list[Placement], spectrograms: dict[str, np.ndarray]
) -> None:
    report = score_dictionary(motifs, placements, spectrograms)
    by_id = {row.motif_id: row for row in report.motifs}
    print(label, flush=True)
    for motif in motifs:
        row = by_id[motif.motif_id]
        flag = "ok" if row.mean_gap >= TARGET_GAP else "no"
        print(
            f"  {flag} {motif.motif_id:4d}  n {row.n_placements:4d}  "
            f"cols {_columns(motif, placements):5d}  gap {row.mean_gap:.3f}",
            flush=True,
        )
    print(
        f"  motifs {len(motifs)}  ratio {_ratio(motifs, placements):.3f}  "
        f"min_gap {min(row.mean_gap for row in report.motifs):.3f}",
        flush=True,
    )


def _repair(
    motifs: list[Motif], placements: list[Placement], spectrograms: dict[str, np.ndarray]
) -> tuple[list[Motif], list[Placement]]:
    for _ in range(6):
        report = score_dictionary(motifs, placements, spectrograms)
        weak = {row.motif_id for row in report.motifs if row.mean_gap < TARGET_GAP}
        if not weak:
            return motifs, placements
        kept: list[Placement] = []
        for motif in motifs:
            own = [placement for placement in placements if placement.motif_id == motif.motif_id]
            if motif.motif_id not in weak:
                kept.extend(own)
                continue
            kept.extend(
                placement
                for gap, placement in _placement_gaps(motif, own, motifs, spectrograms)
                if gap >= TARGET_GAP
            )
        placements = kept
        survive: list[Motif] = []
        for motif in motifs:
            own = [placement for placement in placements if placement.motif_id == motif.motif_id]
            if len(own) < MIN_PATCHES:
                continue
            _refit(motif, placements, spectrograms)
            if motif.n_components == TARGET_COMPONENTS:
                survive.append(motif)
            else:
                placements = [placement for placement in placements if placement.motif_id != motif.motif_id]
        motifs = survive
        if not motifs:
            break
    return motifs, placements


def _balance(
    motifs: list[Motif], placements: list[Placement], spectrograms: dict[str, np.ndarray]
) -> tuple[list[Motif], list[Placement]]:
    """Shrink the motif that covers the most columns until the coverage ratio is at most 4."""
    for _ in range(40):
        if not motifs or _ratio(motifs, placements) <= TARGET_RATIO:
            return motifs, placements
        motif = max(motifs, key=lambda item: _columns(item, placements))
        own = [placement for placement in placements if placement.motif_id == motif.motif_id]
        scored = _placement_gaps(motif, own, motifs, spectrograms)
        if len(scored) <= MIN_PATCHES:
            placements = [placement for placement in placements if placement.motif_id != motif.motif_id]
            motifs = [item for item in motifs if item.motif_id != motif.motif_id]
            continue
        scored.sort(key=lambda item: item[0])
        dropped = scored[0][1]
        placements = [placement for placement in placements if placement is not dropped]
        _refit(motif, placements, spectrograms)
    return motifs, placements


def _core(
    source_motifs: list[Motif],
    source_placements: list[Placement],
    spectrograms: dict[str, np.ndarray],
) -> tuple[list[Motif], list[Placement]]:
    """Choose placements against the original models, then refit."""
    report = score_dictionary(source_motifs, source_placements, spectrograms)
    by_id = {row.motif_id: row for row in report.motifs}
    chosen_by_id: dict[int, list[Placement]] = {}
    for motif in source_motifs:
        own = [placement for placement in source_placements if placement.motif_id == motif.motif_id]
        if by_id[motif.motif_id].mean_gap >= TARGET_GAP:
            chosen = list(own)
        else:
            chosen = [
                placement
                for gap, placement in _placement_gaps(motif, own, source_motifs, spectrograms)
                if gap >= TARGET_GAP
            ]
        print(
            f"select motif {motif.motif_id}  {len(chosen)}/{len(own)}  "
            f"source gap {by_id[motif.motif_id].mean_gap:.3f}",
            flush=True,
        )
        if len(chosen) >= MIN_PATCHES:
            chosen_by_id[motif.motif_id] = chosen
    motifs = [motif for motif in source_motifs if motif.motif_id in chosen_by_id]
    placements = [placement for motif in motifs for placement in chosen_by_id[motif.motif_id]]
    for motif in motifs:
        _refit(motif, placements, spectrograms)
    motifs = [motif for motif in motifs if motif.n_components == TARGET_COMPONENTS]
    kept_ids = {motif.motif_id for motif in motifs}
    placements = [placement for placement in placements if placement.motif_id in kept_ids]
    motifs, placements = _repair(motifs, placements, spectrograms)
    return _balance(motifs, placements, spectrograms)


def _blocked_counts(
    motifs: list[Motif], placements: list[Placement], spectrograms: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    return _recompute_counts(spectrograms, placements, {motif.motif_id: motif.width for motif in motifs})


def _from_seeds(
    motifs: list[Motif],
    placements: list[Placement],
    spectrograms: dict[str, np.ndarray],
) -> tuple[list[Motif], list[Placement]]:
    windows = sorted(load_windows(GPL, None, 0), key=lambda window: (window.start_sample, window.window_id))
    segments = [
        segment
        for segment in detections_to_segments(windows, load_detections(GPL))
        if WIDTH_MIN <= (segment.t_end - segment.t_start) <= WIDTH_MAX
    ]
    hardness: list[tuple[float, object]] = []
    for segment in segments:
        spec = spectrograms[segment.window_id]
        start = int(segment.t_start)
        best = min(
            (
                unexplained(spec[:, start : start + motif.width], motif)
                for motif in motifs
                if start + motif.width <= spec.shape[1]
            ),
            default=1.0,
        )
        if best >= 0.4:
            hardness.append((best, segment))
    hardness.sort(key=lambda item: -item[0])
    print(f"seed candidates {len(hardness)}", flush=True)
    next_id = 200
    for trial, (_best, segment) in enumerate(hardness[:MAX_SEED_TRIALS]):
        if len(motifs) >= TARGET_MOTIFS:
            break
        patch = np.asarray(segment.data, dtype=np.float64)
        norm = float(np.linalg.norm(patch))
        if norm == 0.0:
            continue
        motif = Motif(next_id, patch / norm, segment.window_id, int(segment.t_start))
        next_id += 1
        found = _pursuit([motif], spectrograms, _blocked_counts(motifs, placements, spectrograms), k=1, eps=0.01)
        added = _try_add(motif, found, motifs, placements, spectrograms)
        if added is None:
            if trial % 10 == 9:
                print(f"  tried {trial + 1} seeds, motifs {len(motifs)}", flush=True)
            continue
        motifs, placements = added
        print(
            f"added seed {motif.motif_id}  {segment.window_id} t {int(segment.t_start)}  "
            f"width {motif.width}  n {sum(p.motif_id == motif.motif_id for p in placements)}",
            flush=True,
        )
    return motifs, placements


def _from_leftovers(
    motifs: list[Motif],
    placements: list[Placement],
    leftovers: dict[int, list[Placement]],
    widths: dict[int, int],
    spectrograms: dict[str, np.ndarray],
    rng: np.random.Generator,
) -> tuple[list[Motif], list[Placement]]:
    next_id = 100
    for origin, pool in sorted(leftovers.items(), key=lambda item: len(item[1])):
        if len(motifs) >= TARGET_MOTIFS or len(pool) < MIN_PATCHES:
            continue
        width = widths[origin]
        print(f"try leftover {origin}  n {len(pool)}  width {width}", flush=True)
        for trial in range(16):
            if len(motifs) >= TARGET_MOTIFS:
                break
            low, high = _band(motifs, placements)
            n_min = max(MIN_PATCHES, math.ceil(low / width))
            n_max = high // width
            if n_min > n_max or n_min > len(pool):
                break
            take = min(len(pool), n_max)
            picked = [pool[int(index)] for index in rng.choice(len(pool), size=take, replace=False)]
            prototype = picked[0]
            patch = spectrograms[prototype.window_id][:, prototype.t_start : prototype.t_start + width]
            norm = float(np.linalg.norm(patch))
            if norm == 0.0:
                continue
            motif = Motif(next_id, patch / norm, prototype.window_id, prototype.t_start)
            next_id += 1
            copied = [
                Placement(motif.motif_id, item.window_id, item.t_start, item.amplitude, item.unexplained)
                for item in picked
            ]
            added = _try_add(motif, copied, motifs, placements, spectrograms)
            if added is None:
                continue
            motifs, placements = added
            print(f"added leftover {motif.motif_id} from {origin} trial {trial}", flush=True)
            break
    return motifs, placements


def _checks(
    motifs: list[Motif], placements: list[Placement], spectrograms: dict[str, np.ndarray]
) -> dict[str, dict[str, object]]:
    report = score_dictionary(motifs, placements, spectrograms)
    components = sorted({motif.n_components for motif in motifs})
    ratio = _ratio(motifs, placements) if motifs else float("inf")
    min_gap = min((row.mean_gap for row in report.motifs), default=float("-inf"))
    return {
        "n_motifs": {"value": len(motifs), "ok": len(motifs) == TARGET_MOTIFS},
        "n_components": {
            "value": components[0] if len(components) == 1 else components,
            "ok": components == [TARGET_COMPONENTS],
        },
        "coverage_ratio": {"value": ratio, "ok": bool(motifs) and ratio <= TARGET_RATIO},
        "min_gap": {"value": min_gap, "ok": min_gap >= TARGET_GAP},
    }


def _save(
    motifs: list[Motif],
    placements: list[Placement],
    spectrograms: dict[str, np.ndarray],
    notes: list[str],
) -> None:
    _rescore(motifs, placements, spectrograms)
    report = score_dictionary(motifs, placements, spectrograms)
    OUT.mkdir(parents=True, exist_ok=True)
    atoms = OUT / "atoms"
    atoms.mkdir(exist_ok=True)
    by_gap = {row.motif_id: row for row in report.motifs}
    motif_rows = []
    for motif in motifs:
        name = f"template_{motif.motif_id:03d}.npy"
        np.save(atoms / name, motif.data)
        if motif.mean is not None:
            np.save(atoms / f"mean_{motif.motif_id:03d}.npy", motif.mean)
        row = by_gap[motif.motif_id]
        motif_rows.append(
            {
                "motif_id": motif.motif_id,
                "width": motif.width,
                "n_freq": motif.n_freq,
                "n_components": motif.n_components,
                "n_placements": row.n_placements,
                "n_columns": _columns(motif, placements),
                "mean_own": row.mean_own,
                "mean_runner": row.mean_runner,
                "mean_gap": row.mean_gap,
                "runner_id": row.runner_id,
                "seed_window_id": motif.window_id,
                "seed_t_start": motif.t_start,
                "path": f"atoms/{name}",
            }
        )
    widths = {motif.motif_id: motif.width for motif in motifs}
    matchings = []
    for placement in placements:
        matching = {
            "motif_id": placement.motif_id,
            "window_id": placement.window_id,
            "t_start": placement.t_start,
            "width": widths[placement.motif_id],
            "amplitude": placement.amplitude,
            "unexplained": placement.unexplained,
        }
        if placement.coeffs is not None and placement.coeffs.size > 1:
            matching["coeffs"] = [float(value) for value in placement.coeffs]
        matchings.append(matching)
    (OUT / "matchings.json").write_text(json.dumps(matchings) + "\n")
    counts = _recompute_counts(spectrograms, placements, widths)
    np.savez_compressed(OUT / "counts.npz", **counts)
    coverage, n_covered, n_total = _coverage(counts)
    checks = _checks(motifs, placements, spectrograms)
    summary = {
        "command": "balanced-ten",
        "gpl_dir": str(GPL.resolve()),
        "source": str(SOURCE),
        "stop_reason": "accepted" if all(item["ok"] for item in checks.values()) else "incomplete",
        "accepted": all(item["ok"] for item in checks.values()),
        "coverage": coverage,
        "n_bins_covered": n_covered,
        "n_bins_total": n_total,
        "n_motifs": len(motifs),
        "n_placements": len(placements),
        "max_count": max((int(column.max()) for column in counts.values() if column.size), default=0),
        "checks": checks,
        "notes": notes,
        "knobs": {
            "target_motifs": TARGET_MOTIFS,
            "target_gap": TARGET_GAP,
            "target_coverage_ratio": TARGET_RATIO,
            "n_components": TARGET_COMPONENTS,
            "margin": TARGET_GAP,
            "k": 1,
            "eps": 0.01,
            "w_min": WIDTH_MIN,
            "w_max": WIDTH_MAX,
            "max_seed_trials": MAX_SEED_TRIALS,
        },
        "motifs": motif_rows,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {OUT}  accepted {summary['accepted']}", flush=True)


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} already exists")
    windows = sorted(load_windows(GPL, None, 0), key=lambda window: (window.start_sample, window.window_id))
    spectrograms = {window.window_id: np.asarray(window.data, dtype=np.float64) for window in windows}
    source_motifs, source_placements, widths = _load_source()
    motifs, placements = _core(source_motifs, source_placements, spectrograms)
    _print_state("core", motifs, placements, spectrograms)
    kept = {(placement.window_id, placement.t_start, placement.motif_id) for placement in placements}
    leftovers: dict[int, list[Placement]] = {}
    for placement in source_placements:
        key = (placement.window_id, placement.t_start, placement.motif_id)
        if key in kept:
            continue
        leftovers.setdefault(placement.motif_id, []).append(placement)
    notes = [f"core {[motif.motif_id for motif in motifs]}"]
    if len(motifs) < TARGET_MOTIFS:
        motifs, placements = _from_leftovers(
            motifs, placements, leftovers, widths, spectrograms, np.random.default_rng(0)
        )
        notes.append(f"after leftovers {[motif.motif_id for motif in motifs]}")
        _print_state("after leftovers", motifs, placements, spectrograms)
    if len(motifs) < TARGET_MOTIFS:
        motifs, placements = _from_seeds(motifs, placements, spectrograms)
        notes.append(f"after seeds {[motif.motif_id for motif in motifs]}")
        _print_state("after seeds", motifs, placements, spectrograms)
    checks = _checks(motifs, placements, spectrograms)
    for name, check in checks.items():
        print(f"{'pass' if check['ok'] else 'fail'}  {name} = {check['value']}", flush=True)
    if all(check["ok"] for check in checks.values()):
        _save(motifs, placements, spectrograms, notes)
    else:
        print("not saved", flush=True)


if __name__ == "__main__":
    main()
