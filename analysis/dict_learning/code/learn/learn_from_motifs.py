"""Grow a dictionary from matrix-profile motifs on a GPL whitened spectrogram.

Each round searches the residual for the strongest motif of length 5 s or
10 s, adds that rectangle to the dictionary, then refines every atom and
replaces the residual. The loop stops when matchings cover half the frames.
The outputs are the detection map and the dictionary.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from analysis.matrix_profile import (
    band_motif_profile,
    closest_pairs,
    motif_rectangle,
)

from ..core import EPS_STEP_FRACTION
from ..core import Dictionary, Matching, Template, Window
from analysis.bands import FREQ_MAX_HZ, N_BANDS
from ..io import stitch_whitened_spectrogram
from ..core import coverage_from_matchings, patch_template_relative_distance
from ..mp import refit_and_deflate
from .refine import NoMatchingPatchesError, refine_dictionary_by_averaging

WINDOW_ID = "band_profile"
LENGTHS_S = (5, 10)
# Fraction of patch energy that may stay unexplained. On the 014845 whitened
# spectrogram the motif copies sit near 0.09–0.13, and 0.859 is the loosest
# gate that still keeps the first motif under 20% of the frames.
MAX_RELATIVE_DISTANCE = 0.859


@dataclass
class MotifRound:
    """One motif added from the residual."""

    length_s: float
    width: int
    start_a: int
    start_b: int
    distance: float
    coverage: float


@dataclass
class MotifLearnResult:
    """Detection map and dictionary after the coverage loop."""

    dictionary: Dictionary
    matchings: list[Matching]
    detection_map: np.ndarray
    residual: np.ndarray
    coverage: float
    rounds: list[MotifRound] = field(default_factory=list)
    stop_reason: str = ""
    max_relative_distance: float | None = None
    sec_per_bin: float | None = None
    start_freq: float | None = None
    end_freq: float | None = None


def detection_map_from_matchings(
    n_time: int,
    matchings: list[Matching],
    widths: dict[int, int],
) -> np.ndarray:
    """Frame label for each time step. ``-1`` means no motif covers that frame."""
    labels = np.full(int(n_time), -1, dtype=np.int32)
    ordered = sorted(matchings, key=lambda m: (m.t_start, m.template_id))
    for matching in ordered:
        width = int(widths[matching.template_id])
        start = max(0, int(matching.t_start))
        end = min(int(n_time), start + width)
        unlabeled = labels[start:end] < 0
        labels[start:end][unlabeled] = int(matching.template_id)
    return labels


def _widths(dictionary: Dictionary) -> dict[int, int]:
    return {template.template_id: template.width for template in dictionary}


def _coverage(original: np.ndarray, matchings: list[Matching], dictionary: Dictionary) -> float:
    if not matchings:
        return 0.0
    _, _, covered, total = coverage_from_matchings(
        {WINDOW_ID: original},
        matchings,
        template_widths=_widths(dictionary),
    )
    if total <= 0:
        return 0.0
    return covered / total


def _window_energy(profile: np.ndarray, width: int) -> np.ndarray:
    power = np.sum(np.asarray(profile, dtype=np.float64) ** 2, axis=0)
    cumulative = np.cumsum(power)
    energy = np.empty(profile.shape[1] - width + 1, dtype=np.float64)
    energy[0] = cumulative[width - 1]
    if energy.size > 1:
        energy[1:] = cumulative[width:] - cumulative[:-width]
    return energy


def _covered_starts(occupied: np.ndarray, width: int, n_starts: int) -> np.ndarray:
    """True where a window of ``width`` overlaps a frame already labeled."""
    cumulative = np.cumsum(occupied.astype(np.int32, copy=False))
    hits = np.empty(n_starts, dtype=bool)
    hits[0] = cumulative[width - 1] > 0
    if n_starts > 1:
        hits[1:] = (cumulative[width:] - cumulative[:-width]) > 0
    return hits


def _strongest_motif(
    residual: np.ndarray,
    widths: list[tuple[float, int]],
    nthreads: int,
    occupied: np.ndarray | None = None,
):
    """Best pair across lengths. Score is distance divided by ``sqrt(width)``.

    ``occupied`` marks frames already labeled. A window that overlaps one of
    those frames is skipped, so a leftover of a motif already placed cannot
    be chosen again.
    """
    best = None
    for length_s, width in widths:
        if residual.shape[1] < width + width or width < 4:
            continue
        distance, neighbor = band_motif_profile(residual, width, nthreads=nthreads)
        energy = _window_energy(residual, width)
        floor = 1e-8 * float(np.max(energy)) if energy.size else 0.0
        usable = np.array(distance, dtype=np.float64, copy=True)
        usable[energy <= floor] = np.nan
        if occupied is not None:
            usable[_covered_starts(occupied, width, usable.shape[0])] = np.nan
        if not np.any(np.isfinite(usable)):
            continue
        while True:
            pairs = closest_pairs(usable, neighbor, width, k=1)
            if not pairs:
                break
            start_a, start_b, dist = pairs[0]

            def blocked(start: int) -> bool:
                if energy[start] <= floor:
                    return True
                return occupied is not None and bool(np.any(occupied[start : start + width]))

            if blocked(start_a) or blocked(start_b):
                usable[start_a] = np.nan
                usable[start_b] = np.nan
                continue
            score = float(dist) / np.sqrt(width)
            if best is None or score < best[0]:
                best = (score, float(length_s), width, int(start_a), int(start_b), float(dist))
            break
    return best


def _drop_unmatched(dictionary: Dictionary, matchings: list[Matching]) -> Dictionary:
    used = {matching.template_id for matching in matchings}
    kept = Dictionary()
    for template in dictionary:
        if template.template_id in used:
            kept.add(template)
    return kept


def _close_to_motif(
    original: np.ndarray,
    matchings: list[Matching],
    motif_seeds: dict[int, np.ndarray],
    max_relative_distance: float | None,
) -> list[Matching]:
    """Drop placements that are far from the motif the atom was seeded with."""
    if max_relative_distance is None:
        return list(matchings)
    kept: list[Matching] = []
    for matching in matchings:
        seed = motif_seeds.get(matching.template_id)
        if seed is None:
            continue
        width = seed.shape[1]
        start = int(matching.t_start)
        patch = original[:, start : start + width]
        if patch.shape[1] != width:
            continue
        distance = patch_template_relative_distance(patch, seed)
        if distance <= max_relative_distance:
            kept.append(matching)
    return kept


def _residual_from_matchings(
    original: np.ndarray,
    dictionary: Dictionary,
    matchings: list[Matching],
) -> np.ndarray:
    stack = {WINDOW_ID: np.array(original, copy=True)}
    refit_and_deflate(stack, dictionary, matchings, normalize=True)
    return stack[WINDOW_ID]


def learn_dictionary_from_motifs(
    profile: np.ndarray,
    *,
    frame_rate: float,
    lengths_s: tuple[float, ...] = LENGTHS_S,
    min_coverage: float = 0.5,
    refine_iters: int = 1,
    eps_step_fraction: float = EPS_STEP_FRACTION,
    max_relative_distance: float | None = MAX_RELATIVE_DISTANCE,
    nthreads: int = 1,
    max_rounds: int = 100,
    show_progress: bool = False,
) -> MotifLearnResult:
    """Find motifs, refine, and repeat on the residual until coverage is met.

    ``profile`` is a ``(frequency, time)`` whitened spectrogram. A round
    searches that residual at each length in ``lengths_s`` (seconds at
    ``frame_rate``). The stronger pair, after dividing distance by
    ``sqrt(width)``, is added. Its atom is the mean of the two rectangles
    on the original spectrogram. ``refine_dictionary_by_averaging`` then
    refits every atom and subtracts the matchings.
    ``max_relative_distance`` rejects a placement that leaves more than that
    fraction of its energy unexplained by the motif it was seeded with
    (``None`` accepts every placement matching pursuit would take). Coverage
    is the fraction of frames labeled in the detection map.
    """
    original = np.asarray(profile, dtype=np.float64)
    if original.ndim != 2 or original.shape[0] < 1 or original.shape[1] < 1:
        raise ValueError(f"profile must have shape (frequency, time), got {original.shape}")
    if frame_rate <= 0:
        raise ValueError(f"frame_rate must be positive, got {frame_rate}")
    if not 0.0 < min_coverage <= 1.0:
        raise ValueError(f"min_coverage must be in (0, 1], got {min_coverage}")
    widths: list[tuple[float, int]] = []
    for length_s in lengths_s:
        width = int(round(float(length_s) * float(frame_rate)))
        if width < 4:
            raise ValueError(f"length {length_s} s is {width} frames; need at least 4")
        widths.append((float(length_s), width))

    residual = np.array(original, copy=True)
    dictionary = Dictionary()
    matchings: list[Matching] = []
    motif_seeds: dict[int, np.ndarray] = {}
    rounds: list[MotifRound] = []
    coverage = 0.0
    stop_reason = "max_rounds"
    occupied = np.zeros(original.shape[1], dtype=bool)

    for _ in range(max_rounds):
        found = _strongest_motif(residual, widths, nthreads, occupied)
        if found is None:
            stop_reason = "no_motif"
            break
        _score, length_s, width, start_a, start_b, distance = found
        template_id = 0 if not dictionary.order else max(dictionary.order) + 1
        rectangle_a = motif_rectangle(original, start_a, width)
        rectangle_b = motif_rectangle(original, start_b, width)
        motif_seeds[template_id] = 0.5 * (rectangle_a + rectangle_b)
        dictionary.add(
            Template(
                template_id=template_id,
                data=np.array(motif_seeds[template_id], copy=True),
                source_segment_id=f"motif:{start_a}:{start_b}",
            )
        )
        seeded = list(matchings)
        for start in (start_a, start_b):
            seeded.append(
                Matching(
                    template_id=template_id,
                    window_id=WINDOW_ID,
                    t_start=start,
                    amplitude=1.0,
                )
            )
        try:
            refined = refine_dictionary_by_averaging(
                {WINDOW_ID: np.array(original, copy=True)},
                dictionary,
                seeded,
                iter=refine_iters,
                eps_step_fraction=eps_step_fraction,
                max_relative_distance=max_relative_distance,
                normalize=True,
                disallow_overlap=True,
                template_method="average",
                pinned_matchings=matchings,
                show_progress=False,
                n_workers=1,
            )
        except NoMatchingPatchesError:
            stop_reason = "refine_failed"
            dictionary = _drop_unmatched(dictionary, matchings)
            break
        dictionary = _drop_unmatched(refined.dictionary, refined.matchings)
        matchings = [m for m in refined.matchings if m.template_id in _widths(dictionary)]
        matchings = _close_to_motif(original, matchings, motif_seeds, max_relative_distance)
        dictionary = _drop_unmatched(dictionary, matchings)
        residual = _residual_from_matchings(original, dictionary, matchings)
        occupied = np.zeros(original.shape[1], dtype=bool)
        widths_now = _widths(dictionary)
        for matching in matchings:
            width_m = widths_now.get(matching.template_id)
            if width_m is None:
                continue
            start = max(0, int(matching.t_start))
            occupied[start : start + int(width_m)] = True
        new_coverage = _coverage(original, matchings, dictionary)
        rounds.append(
            MotifRound(
                length_s=length_s,
                width=width,
                start_a=start_a,
                start_b=start_b,
                distance=distance,
                coverage=new_coverage,
            )
        )
        if show_progress:
            print(
                f"motif {len(rounds)}  {length_s:g} s  "
                f"{start_a} and {start_b}  coverage {new_coverage:.3f}",
                flush=True,
            )
        previous = coverage
        coverage = new_coverage
        if coverage >= min_coverage:
            stop_reason = "coverage"
            break
        if coverage <= previous + 1e-12:
            stop_reason = "stalled"
            break
    else:
        stop_reason = "max_rounds"

    labels = detection_map_from_matchings(original.shape[1], matchings, _widths(dictionary))
    return MotifLearnResult(
        dictionary=dictionary,
        matchings=matchings,
        detection_map=labels,
        residual=residual,
        coverage=coverage,
        rounds=rounds,
        stop_reason=stop_reason,
        max_relative_distance=max_relative_distance,
        sec_per_bin=1.0 / float(frame_rate),
    )


DEFAULT_GPL_DIR = (
    Path.home()
    / "whale_processing/result/LeiData/SOCAL_H_76_230427_014845.x.gpl/output"
)


def _write_result(result: MotifLearnResult, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "detection_map.npy", result.detection_map)
    np.save(out_dir / "residual.npy", result.residual)
    for template in result.dictionary:
        np.save(out_dir / f"template_{template.template_id:03d}.npy", template.data)
    matchings = [
        {
            "template_id": matching.template_id,
            "window_id": matching.window_id,
            "t_start": matching.t_start,
            "amplitude": matching.amplitude,
        }
        for matching in result.matchings
    ]
    (out_dir / "matchings.json").write_text(json.dumps(matchings, indent=2))
    summary = {
        "stop_reason": result.stop_reason,
        "coverage": result.coverage,
        "max_relative_distance": result.max_relative_distance,
        "sec_per_bin": result.sec_per_bin,
        "start_freq": result.start_freq,
        "end_freq": result.end_freq,
        "n_atoms": len(result.dictionary),
        "n_matchings": len(result.matchings),
        "rounds": [
            {
                "length_s": round_.length_s,
                "width": round_.width,
                "start_a": round_.start_a,
                "start_b": round_.start_b,
                "distance": round_.distance,
                "coverage": round_.coverage,
            }
            for round_ in result.rounds
        ],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.dict_learning.code.learn.learn_from_motifs",
        description=(
            "Find 5 s and 10 s motifs on the GPL whitened spectrogram, "
            "refine the dictionary, and repeat on the residual until the "
            "detection map covers half the recording."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--gpl-dir",
        type=Path,
        default=DEFAULT_GPL_DIR,
        help="GPL output directory, or its spectrogram_npy folder",
    )
    parser.add_argument("--out", type=Path, required=True, help="Directory for the map and dictionary")
    parser.add_argument("--lengths", type=str, default="5,10", help="Motif lengths in seconds")
    parser.add_argument("--min-coverage", type=float, default=0.5)
    parser.add_argument("--refine-iters", type=int, default=1)
    parser.add_argument("--eps-step-fraction", type=float, default=EPS_STEP_FRACTION)
    parser.add_argument(
        "--max-relative-distance",
        type=float,
        default=MAX_RELATIVE_DISTANCE,
        help=(
            "Reject a placement that leaves more than this fraction of its "
            "energy unexplained (0 is an exact match, 1 accepts every placement)"
        ),
    )
    parser.add_argument("--nthreads", type=int, default=8, help="Matrix-profile threads")
    parser.add_argument("--max-rounds", type=int, default=100)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    if not args.gpl_dir.is_dir():
        print(f"GPL directory not found: {args.gpl_dir}", file=sys.stderr)
        return 1
    lengths = tuple(float(part) for part in args.lengths.split(",") if part.strip())
    spectrogram = stitch_whitened_spectrogram(args.gpl_dir)
    result = learn_dictionary_from_motifs(
        spectrogram.data,
        frame_rate=1.0 / spectrogram.sec_per_bin,
        lengths_s=lengths,
        min_coverage=args.min_coverage,
        refine_iters=args.refine_iters,
        eps_step_fraction=args.eps_step_fraction,
        max_relative_distance=args.max_relative_distance,
        nthreads=args.nthreads,
        max_rounds=args.max_rounds,
        show_progress=not args.no_progress,
    )
    result.start_freq = spectrogram.start_freq
    result.end_freq = spectrogram.end_freq
    _write_result(result, args.out)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "stop_reason": result.stop_reason,
                "coverage": result.coverage,
                "n_atoms": len(result.dictionary),
                "n_matchings": len(result.matchings),
            },
            indent=2,
        )
    )
    return 0

@dataclass(frozen=True)
class MotifSeeds:
    """One band-profile window, the seed atoms, and both copies of each pair."""

    window: Window
    dictionary: Dictionary
    matchings: list[Matching]


def window_from_band_profile(
    profile: np.ndarray,
    sample_rate: int,
    frame_rate: int,
    window_id: str = "band_profile",
) -> Window:
    """Wrap a ``(5, time)`` band-energy profile as one dictionary window."""
    data = np.asarray(profile, dtype=np.float64)
    if data.ndim != 2 or data.shape[0] != N_BANDS:
        raise ValueError(f"profile must have shape ({N_BANDS}, time), got {data.shape}")
    if frame_rate < 1 or sample_rate % frame_rate != 0:
        raise ValueError("frame_rate must divide sample_rate")
    hop = sample_rate // frame_rate
    n_time = data.shape[1]
    return Window(
        window_id=window_id,
        data=data,
        start_sample=1,
        end_sample=n_time * hop,
        overlap=hop,
        fftl=hop,
        sample_freq=float(sample_rate),
        start_freq=0.0,
        end_freq=float(FREQ_MAX_HZ),
    )


def seeds_from_motif_pairs(
    profile: np.ndarray,
    pairs,
    width: int,
    sample_rate: int,
    frame_rate: int,
    window_id: str = "band_profile",
) -> MotifSeeds:
    """Build seed atoms from motif pairs.

    ``pairs`` items are ``(start_a, start_b)`` or ``(start_a, start_b, score)``.
    ``width`` is the motif length in frames. The atom is the mean of the two
    rectangles. Matchings use amplitude 1; a later matching-pursuit pass
    replaces that weight.
    """
    width = int(width)
    if width < 1:
        raise ValueError(f"width must be >= 1, got {width}")
    window = window_from_band_profile(
        profile, sample_rate, frame_rate, window_id=window_id
    )
    dictionary = Dictionary()
    matchings: list[Matching] = []
    for template_id, pair in enumerate(pairs):
        start_a = int(pair[0])
        start_b = int(pair[1])
        rectangle_a = motif_rectangle(window.data, start_a, width)
        rectangle_b = motif_rectangle(window.data, start_b, width)
        dictionary.add(
            Template(
                template_id=template_id,
                data=0.5 * (rectangle_a + rectangle_b),
                source_segment_id=f"motif:{start_a}:{start_b}",
            )
        )
        for start in (start_a, start_b):
            matchings.append(
                Matching(
                    template_id=template_id,
                    window_id=window.window_id,
                    t_start=start,
                    amplitude=1.0,
                )
            )
    return MotifSeeds(window=window, dictionary=dictionary, matchings=matchings)

if __name__ == "__main__":
    raise SystemExit(main())
