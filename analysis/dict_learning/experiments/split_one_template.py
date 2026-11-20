#!/usr/bin/env python3
"""Split one template into two unit-L2 rank-1 atoms (best of K trials, S refines).

Example (synthetic demo — the original planted two-mode experiment)::

    python -m analysis.dict_learning.experiments.split_one_template --demo

Example (dictionary folder + GPL windows)::

    python -m analysis.dict_learning.experiments.split_one_template \\
        --gpl-dir GPL_OUT --dictionary DICT --template-id 1 \\
        --n-trials 20 --n-refine 3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.dict_learning.code.cli import load_dictionary, load_windows  # noqa: E402
from analysis.dict_learning.code.core import Matching  # noqa: E402
from analysis.dict_learning.code.learn.split_template import (  # noqa: E402
    SplitResult,
    split_patches_rank1,
    split_template_matchings,
)
def _demo_patches(rng: np.random.Generator) -> tuple[list[np.ndarray], np.ndarray]:
    """Two planted modes + noise; base = mean patch."""
    f, w = 4, 5
    a = np.zeros((f, w), dtype=np.float64)
    a[0, :] = 3.0
    a[1, :] = -3.0
    b = np.zeros((f, w), dtype=np.float64)
    b[2, :] = 2.0
    b[3, :] = -2.0
    patches = [a + 0.05 * rng.standard_normal(a.shape) for _ in range(6)]
    patches += [b + 0.05 * rng.standard_normal(b.shape) for _ in range(5)]
    base = np.mean(patches, axis=0)
    return patches, base


def run_split(
    patches: list[np.ndarray],
    base_template: np.ndarray | None,
    *,
    n_trials: int,
    n_refine: int,
    epsilon: float,
    min_size: int,
    min_relative_reduction: float,
    seed: int | None,
) -> SplitResult:
    result = split_patches_rank1(
        patches,
        base_template=base_template,
        n_trials=n_trials,
        max_refine=n_refine,
        epsilon=epsilon,
        min_size=min_size,
        min_relative_reduction=min_relative_reduction,
        rng=seed,
    )
    assert abs(float(np.linalg.norm(result.template_a)) - 1.0) <= 1e-6
    assert abs(float(np.linalg.norm(result.template_b)) - 1.0) <= 1e-6
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--demo", action="store_true", help="run synthetic two-mode demo")
    p.add_argument("--gpl-dir", type=Path, help="GPL output folder")
    p.add_argument("--dictionary", type=Path, help="dictionary folder (build layout)")
    p.add_argument("--template-id", type=int, help="template_id to split")
    p.add_argument("-K", "--n-trials", type=int, default=20, help="number of ±ε trials (K)")
    p.add_argument("-S", "--n-refine", type=int, default=3, help="refine iterations per trial (S)")
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--min-size", type=int, default=2)
    p.add_argument("--min-relative-reduction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.demo:
        patches, base = _demo_patches(np.random.default_rng(args.seed))
        result = run_split(
            patches,
            base,
            n_trials=args.n_trials,
            n_refine=args.n_refine,
            epsilon=args.epsilon,
            min_size=args.min_size,
            min_relative_reduction=args.min_relative_reduction,
            seed=args.seed,
        )
    else:
        if args.gpl_dir is None or args.dictionary is None or args.template_id is None:
            p.error("requires --demo, or --gpl-dir --dictionary --template-id")
        windows = load_windows(args.gpl_dir, None, 0)
        dictionary = load_dictionary(args.dictionary)
        if args.template_id not in dictionary.templates:
            raise SystemExit(f"template_id={args.template_id} not in {args.dictionary}")
        template = dictionary.templates[args.template_id]
        rows = json.loads((args.dictionary / "matchings.json").read_text())
        matchings = [
            Matching(
                template_id=int(row["template_id"]),
                window_id=str(row["window_id"]),
                t_start=int(row["t_start"]),
                amplitude=float(row.get("amplitude", 1.0)),
            )
            for row in rows
            if int(row["template_id"]) == int(args.template_id)
        ]
        if not matchings:
            raise SystemExit(f"no matchings for template {args.template_id}")
        result, _, _ = split_template_matchings(
            windows,
            template,
            matchings,
            epsilon=args.epsilon,
            n_trials=args.n_trials,
            max_refine=args.n_refine,
            min_size=args.min_size,
            min_relative_reduction=args.min_relative_reduction,
            rng=args.seed,
        )

    print(
        f"split {'accepted' if result.accepted else 'rejected'}  "
        f"K={result.n_trials} S={args.n_refine}  "
        f"best_trial={result.best_trial}  "
        f"n_a={result.n_a} n_b={result.n_b}  "
        f"rel_reduction={result.relative_reduction:.4f}  "
        f"sse_one={result.sse_one:.4g} sse_two={result.sse_two:.4g}  "
        f"||A||={np.linalg.norm(result.template_a):.6f}  "
        f"||B||={np.linalg.norm(result.template_b):.6f}"
    )
    return 0 if result.accepted or args.demo else 1


if __name__ == "__main__":
    raise SystemExit(main())
