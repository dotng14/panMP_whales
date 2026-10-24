#!/usr/bin/env python3
"""Split one template into two unit-L2 rank-1 atoms (best of K trials, S refines).

Example (synthetic demo)::

    python -m analysis.dict_learning.experiments.split_one_template --demo

Example (catalog)::

    python -m analysis.dict_learning.experiments.split_one_template \\
        --db /path/to/dict_learning_catalog.sqlite \\
        --dictionary-id 115 --template-id 9 \\
        --n-trials 20 --n-refine 3
"""

from __future__ import annotations

import argparse
import sys
from contextlib import closing
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.dict_learning.code.core import Matching  # noqa: E402
from analysis.dict_learning.code.io import load_windows_from_paths  # noqa: E402
from analysis.dict_learning.code.learn.extend_dictionary import (  # noqa: E402
    load_dictionary_from_catalog,
)
from analysis.dict_learning.code.learn.seed_from_gpl import mean_center  # noqa: E402
from analysis.dict_learning.code.learn.split_template import (  # noqa: E402
    SplitOneResult,
    split_one_template,
)
from analysis.dict_learning.code.tools import _extract_patches  # noqa: E402
from database import catalog  # noqa: E402


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
    seed: int | None,
) -> SplitOneResult:
    result = split_one_template(
        patches,
        base_template=base_template,
        n_trials=n_trials,
        n_refine=n_refine,
        epsilon=epsilon,
        min_size=min_size,
        rng=seed,
    )
    # Contract: both atoms are unit L2 (also asserted inside split_one_template).
    assert abs(float(np.linalg.norm(result.template_a)) - 1.0) <= 1e-6
    assert abs(float(np.linalg.norm(result.template_b)) - 1.0) <= 1e-6
    assert len(result.patches_a) + len(result.patches_b) == len(patches)
    return result


def _load_catalog_patches(
    db_path: Path,
    dictionary_id: int,
    template_id: int,
) -> tuple[list[np.ndarray], np.ndarray]:
    with closing(catalog.connect(db_path, read_only=True)) as conn:
        dictionary = load_dictionary_from_catalog(conn, dictionary_id)
        if template_id not in dictionary.templates:
            raise SystemExit(
                f"template_id={template_id} not in dictionary {dictionary_id}"
            )
        template = dictionary.templates[template_id]
        rows = conn.execute(
            """
            SELECT m.window_id, m.t_start, m.amplitude, t.template_id
            FROM matchings m
            JOIN templates t ON t.template_pk = m.template_pk
            WHERE m.dictionary_id = ? AND t.template_id = ?
            ORDER BY m.window_id, m.t_start
            """,
            (dictionary_id, template_id),
        ).fetchall()
        if not rows:
            raise SystemExit(
                f"no matchings for dictionary={dictionary_id} template={template_id}"
            )
        matchings = [
            Matching(
                template_id=int(r["template_id"]),
                window_id=str(r["window_id"]),
                t_start=int(r["t_start"]),
                amplitude=float(r["amplitude"] if r["amplitude"] is not None else 1.0),
            )
            for r in rows
        ]
        window_rows = catalog.list_dictionary_windows(conn, dictionary_id)

    paths = [r["npy_path"] for r in window_rows if r.get("npy_path")]
    windows = load_windows_from_paths(paths)
    stack = {w.window_id: mean_center(w.data) for w in windows}
    patches = _extract_patches(
        stack, matchings, width=int(template.width), n_freq=int(template.n_freq)
    )
    if not patches:
        raise SystemExit("no patches could be extracted")
    base = template.as_rank1 if template.n_components == 1 else template.basis[0]
    return patches, np.asarray(base, dtype=np.float64)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--demo", action="store_true", help="run synthetic two-mode demo")
    p.add_argument("--db", type=Path, help="catalog sqlite path")
    p.add_argument("--dictionary-id", type=int)
    p.add_argument("--template-id", type=int)
    p.add_argument("-K", "--n-trials", type=int, default=20, help="number of ±ε trials (K)")
    p.add_argument("-S", "--n-refine", type=int, default=3, help="refine iterations per trial (S)")
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--min-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    if args.demo:
        patches, base = _demo_patches(np.random.default_rng(args.seed))
    else:
        if args.db is None or args.dictionary_id is None or args.template_id is None:
            p.error("catalog mode requires --db --dictionary-id --template-id (or use --demo)")
        patches, base = _load_catalog_patches(
            args.db, args.dictionary_id, args.template_id
        )

    result = run_split(
        patches,
        base,
        n_trials=args.n_trials,
        n_refine=args.n_refine,
        epsilon=args.epsilon,
        min_size=args.min_size,
        seed=args.seed,
    )
    print(
        f"split OK  K={result.n_trials} S={result.n_refine} "
        f"best_trial={result.best_trial}  "
        f"n_a={len(result.patches_a)} n_b={len(result.patches_b)}  "
        f"rel_reduction={result.relative_reduction:.4f}  "
        f"||A||={np.linalg.norm(result.template_a):.6f}  "
        f"||B||={np.linalg.norm(result.template_b):.6f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
