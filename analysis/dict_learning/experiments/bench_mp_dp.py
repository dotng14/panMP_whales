#!/usr/bin/env python3
"""Greedy vs exact DP matching pursuit on a catalog dictionary.

Reports residual energy, matching counts, and wall time for both algorithms,
then re-runs :func:`calc_divergences` under each, with and without
``respect_occupancy``. Negative divergences mean a single-atom re-peel beat the
joint residual, so their count reads out how much energy the joint assignment
left on the table.

Two conditions have to hold before that count is a clean exactness check:
``lambda_fraction=0`` (otherwise the joint and single-atom peels optimize
different penalized objectives) and ``respect_occupancy`` (otherwise ``B`` may
stack on spans other atoms still hold, which lowers energy for reasons that have
nothing to do with substitutability). Under both, DP leaves zero negatives and
``D(A,A) == 0`` exactly, while greedy does not::

    python -m analysis.dict_learning.experiments.bench_mp_dp \\
        --dictionary-id 122 --n-windows 4 \\
        --lambda-fraction 0.0 --eps-step-fraction 0.02

Example::

    python -m analysis.dict_learning.experiments.bench_mp_dp \\
        --db /path/to/dict_learning_catalog.sqlite \\
        --dictionary-id 122 --n-windows 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import closing
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.dict_learning.code.core import as_stack  # noqa: E402
from analysis.dict_learning.code.io import load_windows_from_paths  # noqa: E402
from analysis.dict_learning.code.learn.similarity import (  # noqa: E402
    calc_divergences,
)
from analysis.dict_learning.code.learn.extend_dictionary import (  # noqa: E402
    load_dictionary_from_catalog,
)
from analysis.dict_learning.code.mp import run_mp_dp_loop  # noqa: E402
from analysis.dict_learning.code.mp import (  # noqa: E402
    residual_energy,
    run_mp_loop,
)
from database import catalog  # noqa: E402

DEFAULT_DB = Path("/home/yfreund/whale_processing/result/dict_learning_catalog.sqlite")
DEFAULT_OUT = Path(
    "/home/yfreund/whale_processing/result/LeiData/dict_learning_templates"
    "/mp_dp_bench"
)


def _negative_stats(divergences: np.ndarray) -> dict[str, float | int]:
    finite = divergences[np.isfinite(divergences)]
    if finite.size == 0:
        return {"n_finite": 0, "n_negative": 0, "fraction_negative": 0.0, "min": 0.0}
    negatives = finite[finite < 0.0]
    return {
        "n_finite": int(finite.size),
        "n_negative": int(negatives.size),
        "fraction_negative": float(negatives.size / finite.size),
        "min": float(finite.min()),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path, default=DEFAULT_DB)
    p.add_argument("--dictionary-id", type=int, default=122)
    p.add_argument("--n-windows", type=int, default=20)
    p.add_argument("--window-offset", type=int, default=0)
    p.add_argument("--eps-step-fraction", type=float, default=0.2)
    p.add_argument("--lambda-fraction", type=float, default=0.2)
    p.add_argument("--max-relative-distance", type=float, default=None)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument(
        "--skip-divergences",
        action="store_true",
        help="only benchmark the peel (divergences cost O(n^2) MPs)",
    )
    args = p.parse_args(argv)

    with closing(catalog.connect(args.db, read_only=True)) as conn:
        drow = catalog.get_dictionary(conn, int(args.dictionary_id))
        if drow is None:
            raise SystemExit(f"dictionary_id={args.dictionary_id} not found")
        dictionary = load_dictionary_from_catalog(conn, int(args.dictionary_id))
        seg_id = int(drow["segmentation_id"])
        window_rows = catalog.list_windows_for_segmentation(
            conn, seg_id, limit=int(args.n_windows), offset=int(args.window_offset)
        )
        if not window_rows:
            raise SystemExit(f"no windows for segmentation {seg_id}")
        paths = [row["npy_path"] for row in window_rows if row["npy_path"]]

    windows = load_windows_from_paths(paths)
    stack = as_stack(windows)
    e_stack = residual_energy(stack)
    print(
        f"dictionary={args.dictionary_id} atoms={len(dictionary)} "
        f"segmentation={seg_id} windows={len(windows)} E(S)={e_stack:.6g}",
        flush=True,
    )

    runs: dict[str, dict[str, float | int]] = {}

    t0 = time.perf_counter()
    greedy = run_mp_loop(
        stack,
        dictionary,
        eps_step_fraction=float(args.eps_step_fraction),
        disallow_overlap=True,
        max_relative_distance=args.max_relative_distance,
        show_progress=False,
    )
    greedy_wall = time.perf_counter() - t0
    runs["greedy"] = {
        "energy_residual": residual_energy(greedy.residual_stack),
        "n_matchings": len(greedy.matchings),
        "wall_s": greedy_wall,
        "stopped_reason": greedy.stopped_reason,
    }

    t0 = time.perf_counter()
    dp = run_mp_dp_loop(
        stack,
        dictionary,
        lambda_fraction=float(args.lambda_fraction),
        max_relative_distance=args.max_relative_distance,
        show_progress=False,
    )
    dp_wall = time.perf_counter() - t0
    runs["dp"] = {
        "energy_residual": residual_energy(dp.residual_stack),
        "n_matchings": len(dp.matchings),
        "wall_s": dp_wall,
        "stopped_reason": dp.stopped_reason,
    }

    print(f"\n{'algorithm':<10} {'E(R)':>14} {'E(R)/E(S)':>10} {'matches':>8} {'wall_s':>8}")
    for name in ("greedy", "dp"):
        row = runs[name]
        print(
            f"{name:<10} {row['energy_residual']:>14.6g} "
            f"{row['energy_residual'] / e_stack:>10.4f} "
            f"{row['n_matchings']:>8d} {row['wall_s']:>8.2f}"
        )
    drop = runs["greedy"]["energy_residual"] - runs["dp"]["energy_residual"]
    print(f"\nDP explains {drop:.6g} more energy ({drop / e_stack:.4%} of the stack)")

    divergence_stats: dict[str, dict[str, float | int]] = {}
    if not args.skip_divergences:
        for algorithm in ("greedy", "dp"):
            for respect in (False, True):
                name = f"{algorithm}{'_occ' if respect else ''}"
                print(
                    f"\ncalc_divergences under {algorithm}"
                    f"{' (re-peel respects occupancy)' if respect else ''} ...",
                    flush=True,
                )
                result = calc_divergences(
                    stack,
                    dictionary,
                    algorithm=algorithm,
                    eps_step_fraction=float(args.eps_step_fraction),
                    lambda_fraction=float(args.lambda_fraction),
                    max_relative_distance=args.max_relative_distance,
                    respect_occupancy=respect,
                    show_progress=False,
                )
                stats = _negative_stats(result.divergences)
                stats["n_significant"] = int(result.significant.sum())
                diag = np.diagonal(result.divergences)
                finite_diag = diag[np.isfinite(diag)]
                stats["max_abs_self_divergence"] = (
                    float(np.max(np.abs(finite_diag))) if finite_diag.size else 0.0
                )
                divergence_stats[name] = stats
                print(
                    f"  significant atoms={stats['n_significant']}  "
                    f"negatives={stats['n_negative']}/{stats['n_finite']} "
                    f"({stats['fraction_negative']:.1%})  min={stats['min']:.3f}  "
                    f"max|D(A,A)|={stats['max_abs_self_divergence']:.3g}"
                )
        print(
            "\nD(A,A) is 0 exactly when the joint peel is optimal and the re-peel "
            "may not take spans other atoms hold; residual negatives in the "
            "unconstrained columns come from B stacking on occupied spans."
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "dictionary_id": int(args.dictionary_id),
        "segmentation_id": seg_id,
        "n_windows": len(windows),
        "n_atoms": len(dictionary),
        "energy_stack": e_stack,
        "eps_step_fraction": float(args.eps_step_fraction),
        "lambda_fraction": float(args.lambda_fraction),
        "max_relative_distance": args.max_relative_distance,
        "runs": runs,
        "divergences": divergence_stats,
    }
    (out_dir / "bench_summary.json").write_text(json.dumps(summary, indent=2))

    with open(out_dir / "bench_summary.csv", "w") as fh:
        fh.write("algorithm,energy_residual,energy_ratio,n_matchings,wall_s\n")
        for name in ("greedy", "dp"):
            row = runs[name]
            fh.write(
                f"{name},{row['energy_residual']:.6g},"
                f"{row['energy_residual'] / e_stack:.6f},"
                f"{row['n_matchings']},{row['wall_s']:.3f}\n"
            )

    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
