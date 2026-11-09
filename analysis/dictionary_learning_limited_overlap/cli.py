"""Command line for limited-overlap dictionary learning.

Reads a GPL output folder (``spectrogram_npy/`` and ``CSV_*.csv``)::

    python -m analysis.dictionary_learning_limited_overlap \\
        --gpl-dir GPL_OUT --out DICT_DIR
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from analysis.dict_learning.code.cli import CliError, load_detections, load_windows

from .learn import (
    DEFAULT_EPS,
    DEFAULT_K,
    DEFAULT_M,
    DEFAULT_MAX_OUTER,
    DEFAULT_MIN_COVERAGE,
    DEFAULT_N_MIN,
    DEFAULT_REFINE_ITERS,
    DEFAULT_N_COMPONENTS,
    DEFAULT_TAU,
    DEFAULT_W_MAX,
    DEFAULT_W_MIN,
    learn_limited_overlap,
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def run(args: argparse.Namespace) -> int:
    windows = load_windows(args.gpl_dir, args.n_windows, args.window_offset)
    detections = load_detections(args.gpl_dir)
    try:
        result = learn_limited_overlap(
            windows,
            detections,
            k=args.k,
            eps=args.eps,
            w_min=args.w_min,
            w_max=args.w_max,
            m=args.m,
            n_min=args.n_min,
            refine_iters=args.refine_iters,
            tau=args.tau,
            max_outer=args.max_outer,
            min_coverage=args.min_coverage,
            n_components=args.n_components,
            rng=args.seed,
        )
    except ValueError as exc:
        raise CliError(str(exc)) from exc

    out = Path(args.out)
    atoms = out / "atoms"
    atoms.mkdir(parents=True, exist_ok=True)
    widths = {motif.motif_id: motif.width for motif in result.motifs}
    motif_rows = []
    for motif in result.motifs:
        name = f"template_{motif.motif_id:03d}.npy"
        np.save(atoms / name, motif.data)
        n_placements = sum(1 for placement in result.placements if placement.motif_id == motif.motif_id)
        motif_rows.append(
            {
                "motif_id": motif.motif_id,
                "width": motif.width,
                "n_freq": motif.n_freq,
                "n_components": motif.n_components,
                "n_placements": n_placements,
                "seed_window_id": motif.window_id,
                "seed_t_start": motif.t_start,
                "path": f"atoms/{name}",
            }
        )
    _write_json(
        out / "matchings.json",
        [
            {
                "motif_id": placement.motif_id,
                "window_id": placement.window_id,
                "t_start": placement.t_start,
                "width": widths[placement.motif_id],
                "amplitude": placement.amplitude,
                "unexplained": placement.unexplained,
                **(
                    {"coeffs": [float(value) for value in placement.coeffs]}
                    if placement.coeffs is not None and placement.coeffs.size > 1
                    else {}
                ),
            }
            for placement in result.placements
        ],
    )
    np.savez_compressed(out / "counts.npz", **result.counts)
    max_count = max((int(column.max()) for column in result.counts.values() if column.size), default=0)
    summary = {
        "command": "limited-overlap",
        "gpl_dir": str(Path(args.gpl_dir).expanduser().resolve()),
        "stop_reason": result.stop_reason,
        "coverage": result.coverage,
        "n_bins_covered": result.n_bins_covered,
        "n_bins_total": result.n_bins_total,
        "n_motifs": len(result.motifs),
        "n_placements": len(result.placements),
        "max_count": max_count,
        "knobs": {
            "k": args.k,
            "eps": args.eps,
            "w_min": args.w_min,
            "w_max": args.w_max,
            "m": args.m,
            "n_min": args.n_min,
            "refine_iters": args.refine_iters,
            "tau": args.tau,
            "max_outer": args.max_outer,
            "min_coverage": args.min_coverage,
            "n_components": args.n_components,
            "seed": args.seed,
            "n_windows": args.n_windows,
            "window_offset": args.window_offset,
        },
        "passes": result.passes,
        "motifs": motif_rows,
    }
    _write_json(out / "summary.json", summary)
    if args.verbose:
        print(
            f"stop {result.stop_reason}  motifs {len(result.motifs)}  "
            f"placements {len(result.placements)}  coverage {result.coverage:.4f}  "
            f"max count {max_count}"
        )
        for row in result.passes:
            print(
                f"pass {row['outer']}  sampled {row['n_sampled']}  "
                f"kept {row['n_motifs']}  placements {row['n_placements']}"
            )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.dictionary_learning_limited_overlap",
        description="Learn a dictionary whose placements cover each time column at most k times.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--gpl-dir", type=Path, required=True, help="GPL output directory")
    parser.add_argument("--out", type=Path, required=True, help="Directory for motifs, placements, and counts")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Maximum placements on one time column")
    parser.add_argument("--eps", type=float, default=DEFAULT_EPS, help="Pursuit energy stop, as a fraction of the first placement")
    parser.add_argument("--w-min", type=int, default=DEFAULT_W_MIN, help="Minimum detection-crop width")
    parser.add_argument("--w-max", type=int, default=DEFAULT_W_MAX, help="Maximum detection-crop width")
    parser.add_argument("--m", type=int, default=DEFAULT_M, help="Seed motifs sampled per pass")
    parser.add_argument("--n-min", type=int, default=DEFAULT_N_MIN, help="Minimum placements kept per motif")
    parser.add_argument("--refine-iters", type=int, default=DEFAULT_REFINE_ITERS, help="Rebuild rounds per pass")
    parser.add_argument("--tau", type=float, default=DEFAULT_TAU, help="Drop a placement whose unexplained fraction exceeds this")
    parser.add_argument("--max-outer", type=int, default=DEFAULT_MAX_OUTER, help="Maximum passes")
    parser.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE, help="Stop when coverage exceeds this")
    parser.add_argument(
        "--n-components",
        type=int,
        default=DEFAULT_N_COMPONENTS,
        help="Principal components kept when a motif is rebuilt from its patches",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for drawing detection crops")
    parser.add_argument("--n-windows", type=int, default=None, help="Use only this many windows")
    parser.add_argument("--window-offset", type=int, default=0, help="Skip this many windows before --n-windows")
    parser.add_argument("--verbose", action="store_true", help="Print one line per pass")
    parser.set_defaults(func=run)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
