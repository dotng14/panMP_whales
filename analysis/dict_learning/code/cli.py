"""Command line for the randomized dictionary build and Mallat matching pursuit.

Both commands read a GPL output folder directly: ``spectrogram_npy/`` (whitened
windows plus JSON sidecars) and ``CSV_*.csv`` (detections)::

    python -m analysis.dict_learning randomized-build --gpl-dir GPL_OUT --out DICT_DIR
    python -m analysis.dict_learning matching-pursuit --gpl-dir GPL_OUT \\
        --dictionary DICT_DIR --out MP_DIR

``randomized-build`` writes ``atoms/template_*.npy``, ``summary.json``,
``matchings.json`` and ``outer_logs.jsonl``. ``matching-pursuit`` places the
atoms listed in a dictionary's ``summary.json`` and writes ``summary.json`` and
``matchings.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from . import mp as mp_module
from .core import Dictionary, Matching, Template, Window, as_stack, coverage_from_matchings
from .io import Detection, discover_windows, find_detection_csvs, load_detections_from_paths
from .learn import randomized_dictionary as rd
from .tools import matching_pursuit


class CliError(Exception):
    """Bad input the user can fix. Printed without a traceback; exit code 2."""


def _npy_dir(gpl_dir: Path) -> Path:
    nested = gpl_dir / "spectrogram_npy"
    return nested if nested.is_dir() else gpl_dir


def load_windows(gpl_dir: Path, n_windows: int | None, window_offset: int) -> list[Window]:
    """Windows in time order, sliced to ``[window_offset, window_offset + n_windows)``."""
    if not gpl_dir.is_dir():
        raise CliError(f"GPL directory not found: {gpl_dir}")
    if window_offset < 0:
        raise CliError(f"--window-offset must be >= 0, got {window_offset}")
    if n_windows is not None and n_windows < 1:
        raise CliError(f"--n-windows must be >= 1, got {n_windows}")
    windows = discover_windows(_npy_dir(gpl_dir))
    if not windows:
        raise CliError(f"no whitened spectrogram windows in {_npy_dir(gpl_dir)}")
    windows.sort(key=lambda window: (window.start_sample, window.window_id))
    end = None if n_windows is None else window_offset + n_windows
    chosen = windows[window_offset:end]
    if not chosen:
        raise CliError(
            f"--window-offset {window_offset} is past the last of {len(windows)} windows"
        )
    return chosen


def load_detections(gpl_dir: Path) -> list[Detection]:
    paths = find_detection_csvs(gpl_dir)
    if not paths:
        raise CliError(f"no CSV_*.csv detection file in {gpl_dir}")
    detections = load_detections_from_paths(paths)
    if not detections:
        raise CliError(f"detection CSV in {gpl_dir} has no rows")
    return detections


def _sec_per_bin(window: Window) -> float | None:
    if window.sample_freq > 0:
        return window.overlap / window.sample_freq
    return None


def _matching_rows(matchings: Sequence[Matching]) -> list[dict[str, Any]]:
    rows = []
    for matching in matchings:
        row: dict[str, Any] = {
            "template_id": int(matching.template_id),
            "window_id": matching.window_id,
            "t_start": int(matching.t_start),
            "amplitude": float(matching.amplitude),
            "energy": matching.energy,
        }
        if matching.n_components > 1:
            row["coeffs"] = [float(c) for c in matching.coeffs]
        rows.append(row)
    return rows


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _coverage(stack: dict[str, np.ndarray], dictionary: Dictionary, matchings) -> dict[str, Any]:
    widths = {int(t.template_id): int(t.width) for t in dictionary}
    n_win_m, n_win_t, n_bins_c, n_bins_t = coverage_from_matchings(
        stack, matchings, template_widths=widths
    )
    return {
        "time_coverage_fraction": n_bins_c / n_bins_t if n_bins_t else 0.0,
        "window_coverage_fraction": n_win_m / n_win_t if n_win_t else 0.0,
        "n_bins_covered": n_bins_c,
        "n_bins_total": n_bins_t,
        "n_windows_matched": n_win_m,
        "n_windows_total": n_win_t,
    }


def _apply_compute(compute: str | None) -> None:
    """``--compute`` sets the MP backend for this process and any MP worker it spawns."""
    if compute is not None:
        os.environ["DICT_LEARNING_MP_COMPUTE"] = compute


def _run_with_stats(enabled: bool, out_dir: Path, fn):
    if not enabled:
        return fn()
    with mp_module.collecting():
        result = fn()
        _write_json(out_dir / "loop_stats.json", mp_module.snapshot())
    return result


def _probability(name: str, value: float | None) -> float | None:
    if value is not None and not 0.0 <= value <= 1.0:
        raise CliError(f"{name} must be in [0, 1], got {value}")
    return value


# ── randomized-build ─────────────────────────────────────────────────────────


def run_randomized_build(args: argparse.Namespace) -> int:
    windows = load_windows(args.gpl_dir, args.n_windows, args.window_offset)
    detections = load_detections(args.gpl_dir)
    _probability("--max-relative-distance", args.max_relative_distance)
    _probability("--post-refine-max-relative-distance", args.post_refine_max_relative_distance)
    _apply_compute(args.compute)
    args.out.mkdir(parents=True, exist_ok=True)

    def on_log(log: rd.RandomizedOuterPassLog) -> None:
        if args.verbose:
            print(
                f"outer {log.outer}: +{log.n_atoms_this_pass} atoms, "
                f"{log.n_atoms} total, time coverage {log.time_coverage_fraction:.3f}",
                file=sys.stderr,
            )

    def build() -> rd.RandomizedOuterResult:
        try:
            return rd.learn_randomized_outer(
                windows,
                detections,
                m_per_outer=args.m_per_outer,
                min_coverage=args.min_coverage,
                max_outer=args.max_outer,
                refine_iters=args.refine_iters,
                n_components=args.n_components,
                n_min=args.n_min,
                epsilon=args.epsilon,
                min_template_width=args.min_template_width,
                max_template_width=args.max_template_width,
                eps_step_fraction=args.eps_step_fraction,
                max_relative_distance=args.max_relative_distance,
                post_refine_max_relative_distance=args.post_refine_max_relative_distance,
                rng=args.seed,
                mp_mode=args.mp_mode,
                n_workers=args.workers,
                show_progress=args.progress,
                progress_callback=on_log,
            )
        except ValueError as exc:
            raise CliError(str(exc)) from exc

    result = _run_with_stats(args.loop_stats, args.out, build)

    atoms_dir = args.out / "atoms"
    atoms_dir.mkdir(exist_ok=True)
    counts: dict[int, int] = {}
    for matching in result.matchings:
        counts[int(matching.template_id)] = counts.get(int(matching.template_id), 0) + 1
    atoms = []
    for template in result.dictionary:
        rel = f"atoms/template_{template.template_id:03d}.npy"
        np.save(args.out / rel, template.data)
        atoms.append(
            {
                "template_id": int(template.template_id),
                "width": template.width,
                "n_freq": template.n_freq,
                "n_components": template.n_components,
                "n_patches": counts.get(int(template.template_id), 0),
                "source_segment_id": template.source_segment_id,
                "path": rel,
            }
        )
    _write_json(args.out / "matchings.json", _matching_rows(result.matchings))
    (args.out / "outer_logs.jsonl").write_text(
        "".join(json.dumps(asdict(log)) + "\n" for log in result.outer_logs)
    )
    summary = {
        "command": "randomized-build",
        "gpl_dir": str(args.gpl_dir),
        "wav": str(args.gpl_dir),
        "n_windows": len(windows),
        "window_offset": args.window_offset,
        "n_detections": len(detections),
        "sec_per_bin": _sec_per_bin(windows[0]),
        "m_per_outer": args.m_per_outer,
        "min_coverage": args.min_coverage,
        "max_outer": args.max_outer,
        "refine_iters": args.refine_iters,
        "n_components": args.n_components,
        "n_min": args.n_min,
        "epsilon": args.epsilon,
        "min_template_width": args.min_template_width,
        "max_template_width": args.max_template_width,
        "eps_step_fraction": args.eps_step_fraction,
        "max_relative_distance": args.max_relative_distance,
        "post_refine_max_relative_distance": args.post_refine_max_relative_distance,
        "seed": args.seed,
        "stop_reason": result.stop_reason,
        "n_outer_completed": result.n_outer_completed,
        "n_atoms": result.n_atoms,
        "n_matchings": result.n_matchings,
        "time_coverage_fraction": result.time_coverage_fraction,
        "window_coverage_fraction": result.window_coverage_fraction,
        "energy_before": result.energy_before,
        "energy_after": result.energy_after,
        "atoms": atoms,
    }
    _write_json(args.out / "summary.json", summary)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "stop_reason": result.stop_reason,
                "n_atoms": result.n_atoms,
                "n_matchings": result.n_matchings,
                "time_coverage_fraction": result.time_coverage_fraction,
            }
        )
    )
    if result.n_atoms == 0:
        print(f"no atoms survived (stop reason: {result.stop_reason})", file=sys.stderr)
        return 1
    return 0


# ── matching-pursuit ─────────────────────────────────────────────────────────


def load_dictionary(dictionary_dir: Path) -> Dictionary:
    """Atoms listed in a ``randomized-build`` ``summary.json``."""
    summary_path = dictionary_dir / "summary.json"
    if not summary_path.is_file():
        raise CliError(f"no summary.json in dictionary directory {dictionary_dir}")
    atoms = json.loads(summary_path.read_text()).get("atoms") or []
    if not atoms:
        raise CliError(f"{summary_path} lists no atoms")
    dictionary = Dictionary()
    for atom in atoms:
        path = dictionary_dir / atom["path"]
        if not path.is_file():
            raise CliError(f"atom file missing: {path}")
        dictionary.add(
            Template(
                template_id=int(atom["template_id"]),
                data=np.load(path),
                source_segment_id=atom.get("source_segment_id"),
            )
        )
    return dictionary


def run_matching_pursuit(args: argparse.Namespace) -> int:
    windows = load_windows(args.gpl_dir, args.n_windows, args.window_offset)
    dictionary = load_dictionary(args.dictionary)
    n_freq = windows[0].n_freq
    for template in dictionary:
        if template.n_freq != n_freq:
            raise CliError(
                f"atom {template.template_id} has {template.n_freq} frequency rows; "
                f"the windows have {n_freq}"
            )
    _probability("--max-relative-distance", args.max_relative_distance)
    _apply_compute(args.compute)
    args.out.mkdir(parents=True, exist_ok=True)
    stack = as_stack(windows)
    normalize = dictionary.max_n_components == 1

    def run():
        try:
            return matching_pursuit(
                stack,
                dictionary,
                eps_step_fraction=args.eps_step_fraction,
                max_repeats=args.max_repeats,
                normalize=normalize,
                disallow_overlap=args.disallow_overlap,
                show_progress=args.progress,
                mp_mode=args.mp_mode,
                n_workers=args.workers,
                max_relative_distance=args.max_relative_distance,
            )
        except ValueError as exc:
            raise CliError(str(exc)) from exc

    result = _run_with_stats(args.loop_stats, args.out, run)
    loop = result.loop
    _write_json(args.out / "matchings.json", _matching_rows(loop.matchings))
    summary = {
        "command": "matching-pursuit",
        "gpl_dir": str(args.gpl_dir),
        "dictionary": str(args.dictionary),
        "n_windows": len(windows),
        "window_offset": args.window_offset,
        "n_atoms": len(dictionary),
        "n_matchings": len(loop.matchings),
        "stop_reason": loop.stopped_reason,
        "energy_before": loop.energies[0],
        "energy_after": loop.energies[-1],
        **_coverage(stack, dictionary, loop.matchings),
    }
    _write_json(args.out / "summary.json", summary)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "stop_reason": loop.stopped_reason,
                "n_matchings": len(loop.matchings),
            }
        )
    )
    return 0


# ── parser ───────────────────────────────────────────────────────────────────


def _optional_int(text: str) -> int | None:
    return None if text.lower() == "none" else int(text)


def _optional_float(text: str) -> float | None:
    return None if text.lower() == "none" else float(text)


def _add_input_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--gpl-dir", type=Path, required=True, help="GPL output folder")
    parser.add_argument("--out", type=Path, required=True, help="Output folder")
    parser.add_argument(
        "--n-windows", type=int, default=None, help="Number of windows to use (default: all)"
    )
    parser.add_argument("--window-offset", type=int, default=0, help="First window to use")


def _add_mp_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--eps-step-fraction", type=float, default=rd.defaults.EPS_STEP_FRACTION)
    parser.add_argument(
        "--max-relative-distance",
        type=_optional_float,
        default=rd.defaults.MAX_RELATIVE_DISTANCE,
        help="Reject placements leaving more than this energy fraction unexplained ('none' = off)",
    )
    parser.add_argument("--mp-mode", choices=("global", "per_window"), default=None)
    parser.add_argument("--workers", type=int, default=None, help="MP worker processes")
    parser.add_argument("--compute", choices=("cpu", "cuda", "auto"), default=None)
    parser.add_argument("--progress", action="store_true", help="Show progress bars")
    parser.add_argument(
        "--loop-stats", action="store_true", help="Write MP loop counters to loop_stats.json"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.dict_learning",
        description="Randomized dictionary build and Mallat matching pursuit on GPL output.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser(
        "randomized-build",
        help="Learn a dictionary: random seeds, MP + PCA refine, coverage-gated outer loop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_input_args(build)
    build.add_argument("--m-per-outer", type=int, default=rd.DEFAULT_OUTER_M)
    build.add_argument("--min-coverage", type=float, default=rd.DEFAULT_OUTER_MIN_COVERAGE)
    build.add_argument("--max-outer", type=int, default=rd.DEFAULT_OUTER_MAX_OUTER)
    build.add_argument("--refine-iters", type=int, default=rd.DEFAULT_REFINE_ITERS)
    build.add_argument("--n-components", type=int, default=rd.defaults.N_COMPONENTS)
    build.add_argument("--n-min", type=int, default=rd.DEFAULT_OUTER_N_MIN)
    build.add_argument("--epsilon", type=float, default=rd.DEFAULT_CREATE_EPSILON)
    build.add_argument(
        "--min-template-width", type=_optional_int, default=rd.DEFAULT_MIN_TEMPLATE_WIDTH
    )
    build.add_argument(
        "--max-template-width", type=_optional_int, default=rd.DEFAULT_MAX_TEMPLATE_WIDTH
    )
    build.add_argument(
        "--post-refine-max-relative-distance",
        type=_optional_float,
        default=rd.DEFAULT_OUTER_POST_REFINE_TAU,
    )
    build.add_argument("--seed", type=int, default=None, help="Random seed")
    build.add_argument("--verbose", action="store_true", help="Print one line per outer pass")
    _add_mp_args(build)
    build.set_defaults(func=run_randomized_build)

    pursue = sub.add_parser(
        "matching-pursuit",
        help="Place a saved dictionary on GPL windows with Mallat MP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_input_args(pursue)
    pursue.add_argument(
        "--dictionary", type=Path, required=True, help="Output folder of randomized-build"
    )
    pursue.add_argument("--max-repeats", type=int, default=None)
    pursue.add_argument("--disallow-overlap", action="store_true")
    _add_mp_args(pursue)
    pursue.set_defaults(func=run_matching_pursuit)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

