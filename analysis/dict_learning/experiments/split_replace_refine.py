#!/usr/bin/env python3
"""Replace one dictionary template with its two split children, then refine.

1. Load a parent dictionary from the catalog.
2. Split ``--template-id`` with :func:`split_one_template` (K trials, S refines).
3. Remap that template's matchings onto the two children; keep all other atoms.
4. Rebuild + run ``--refine-iters`` average→MP iterations (labels pinned).
5. Register the result as a new catalog dictionary (``source="edit"``).

Example::

    python -m analysis.dict_learning.experiments.split_replace_refine \\
        --db /home/yfreund/whale_processing/result/dict_learning_catalog.sqlite \\
        --dictionary-id 115 --template-id 9 \\
        --n-trials 20 --n-refine 3 --refine-iters 2
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

from analysis.dict_learning.code.core import EPS_STEP_FRACTION  # noqa: E402
from analysis.dict_learning.code.core import Matching  # noqa: E402
from analysis.dict_learning.code.io import load_windows_from_paths  # noqa: E402
from analysis.dict_learning.code.learn.extend_dictionary import (  # noqa: E402
    load_dictionary_from_catalog,
)
from analysis.dict_learning.code.learn.match_to_dictionary import (  # noqa: E402
    match_to_dictionary,
    write_match_to_dictionary_result,
)
from analysis.dict_learning.code.learn.seed_from_gpl import mean_center  # noqa: E402
from analysis.dict_learning.code.tools import _extract_patches  # noqa: E402
from analysis.dict_learning.experiments.split_one_template import run_split  # noqa: E402
from database import catalog  # noqa: E402


def _load_all_matchings(conn, dictionary_id: int) -> list[Matching]:
    rows = conn.execute(
        """
        SELECT m.window_id, m.t_start, m.amplitude, t.template_id
        FROM matchings m
        JOIN templates t ON t.template_pk = m.template_pk
        WHERE m.dictionary_id = ?
        ORDER BY t.template_id, m.window_id, m.t_start
        """,
        (dictionary_id,),
    ).fetchall()
    return [
        Matching(
            template_id=int(r["template_id"]),
            window_id=str(r["window_id"]),
            t_start=int(r["t_start"]),
            amplitude=float(r["amplitude"] if r["amplitude"] is not None else 1.0),
        )
        for r in rows
    ]


def _remap_after_split(
    all_matchings: list[Matching],
    *,
    split_tid: int,
    split_matchings: list[Matching],
    labels: np.ndarray,
    tid_a: int,
    tid_b: int,
) -> list[Matching]:
    """Keep non-split matchings; assign split_tid matchings to tid_a / tid_b."""
    if len(split_matchings) != len(labels):
        raise ValueError(
            f"labels length {len(labels)} != split matchings {len(split_matchings)}"
        )
    kept = [m for m in all_matchings if int(m.template_id) != int(split_tid)]
    remapped: list[Matching] = list(kept)
    for matching, lab in zip(split_matchings, labels):
        new_tid = tid_a if int(lab) == 0 else tid_b
        remapped.append(
            Matching(
                template_id=int(new_tid),
                window_id=matching.window_id,
                t_start=int(matching.t_start),
                amplitude=float(matching.amplitude),
            )
        )
    return remapped


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--db",
        type=Path,
        default=Path("/home/yfreund/whale_processing/result/dict_learning_catalog.sqlite"),
    )
    p.add_argument("--dictionary-id", type=int, default=115)
    p.add_argument("--template-id", type=int, default=9)
    p.add_argument("-K", "--n-trials", type=int, default=20)
    p.add_argument("-S", "--n-refine", type=int, default=3, help="split refine steps")
    p.add_argument("--epsilon", type=float, default=0.1)
    p.add_argument("--min-size", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--refine-iters",
        type=int,
        default=2,
        help="average→MP refine iterations after replacing the template",
    )
    p.add_argument("--eps-step-fraction", type=float, default=EPS_STEP_FRACTION)
    p.add_argument(
        "--out-root",
        type=Path,
        default=Path(
            "/home/yfreund/whale_processing/result/LeiData/dict_learning_templates"
            "/split_replace_refine"
        ),
    )
    p.add_argument("--name", type=str, default=None)
    args = p.parse_args(argv)

    parent_id = int(args.dictionary_id)
    split_tid = int(args.template_id)

    with closing(catalog.connect(args.db, read_only=True)) as conn:
        parent = catalog.get_dictionary(conn, parent_id)
        if parent is None:
            raise SystemExit(f"dictionary_id={parent_id} not found")
        dictionary = load_dictionary_from_catalog(conn, parent_id)
        if split_tid not in dictionary.templates:
            raise SystemExit(
                f"template_id={split_tid} not in dictionary {parent_id}; "
                f"have {sorted(dictionary.templates)}"
            )
        all_matchings = _load_all_matchings(conn, parent_id)
        window_rows = catalog.list_dictionary_windows(conn, parent_id)
        segmentation_id = parent.get("segmentation_id")

    paths = [r["npy_path"] for r in window_rows if r.get("npy_path")]
    windows = load_windows_from_paths(paths)
    stack = {w.window_id: mean_center(w.data) for w in windows}

    template = dictionary.templates[split_tid]
    split_matchings = [m for m in all_matchings if int(m.template_id) == split_tid]
    patches = _extract_patches(
        stack,
        split_matchings,
        width=int(template.width),
        n_freq=int(template.n_freq),
    )
    if len(patches) != len(split_matchings):
        kept_m: list[Matching] = []
        kept_p: list[np.ndarray] = []
        for matching in split_matchings:
            one = _extract_patches(
                stack,
                [matching],
                width=int(template.width),
                n_freq=int(template.n_freq),
            )
            if one:
                kept_m.append(matching)
                kept_p.append(one[0])
        split_matchings, patches = kept_m, kept_p
    if len(patches) < 2 * int(args.min_size):
        raise SystemExit(
            f"template {split_tid}: only {len(patches)} extractable patches "
            f"(need >= {2 * int(args.min_size)})"
        )

    base = template.as_rank1 if template.n_components == 1 else template.basis[0]
    split = run_split(
        patches,
        np.asarray(base, dtype=np.float64),
        n_trials=args.n_trials,
        n_refine=args.n_refine,
        epsilon=args.epsilon,
        min_size=args.min_size,
        seed=args.seed,
    )
    print(
        f"split tid={split_tid}: n={len(patches)} → "
        f"|A|={len(split.patches_a)} |B|={len(split.patches_b)}  "
        f"rel_reduction={split.relative_reduction:.4f}  "
        f"best_trial={split.best_trial}/{split.n_trials}"
    )

    existing_ids = sorted(dictionary.templates)
    tid_a = max(existing_ids) + 1
    tid_b = tid_a + 1
    remapped = _remap_after_split(
        all_matchings,
        split_tid=split_tid,
        split_matchings=split_matchings,
        labels=split.labels,
        tid_a=tid_a,
        tid_b=tid_b,
    )

    widths = {
        int(t.template_id): int(t.width)
        for t in dictionary
        if int(t.template_id) != split_tid
    }
    widths[tid_a] = int(template.width)
    widths[tid_b] = int(template.width)

    print(
        f"rebuild+refine: drop tid={split_tid}, add tid={tid_a},{tid_b}; "
        f"refine_iters={args.refine_iters}; n_matchings={len(remapped)}"
    )
    result = match_to_dictionary(
        windows,
        remapped,
        widths=widths,
        refine_iters=int(args.refine_iters),
        eps_step_fraction=float(args.eps_step_fraction),
        normalize=True,
        disallow_overlap=True,
        pin_labels=True,
        show_progress=True,
    )

    parent_name = str(parent.get("name") or f"dict{parent_id}")
    run_name = args.name or f"split_t{split_tid}_refine{args.refine_iters}_{parent_name}"
    # Keep name reasonable length for catalog uniqueness / UI.
    if len(run_name) > 180:
        run_name = f"split_t{split_tid}_r{args.refine_iters}_from{parent_id}"

    out_dir = write_match_to_dictionary_result(
        result,
        args.out_root,
        name=run_name,
        params={
            "parent_dictionary_id": parent_id,
            "split_template_id": split_tid,
            "tid_a": tid_a,
            "tid_b": tid_b,
            "n_trials": args.n_trials,
            "n_refine_split": args.n_refine,
            "epsilon": args.epsilon,
            "refine_iters": args.refine_iters,
            "eps_step_fraction": args.eps_step_fraction,
            "pin_labels": True,
            "relative_sse_reduction": split.relative_reduction,
            "best_trial": split.best_trial,
            "n_a": len(split.patches_a),
            "n_b": len(split.patches_b),
            "learner": "split_replace_refine",
        },
    )

    templates_payload = [
        {
            "template_id": int(t.template_id),
            "npy_path": str(out_dir / f"template_{int(t.template_id):03d}.npy"),
            "n_freq": int(t.n_freq),
            "width": int(t.width),
            "source_segment_id": t.source_segment_id,
            "n_matchings": sum(
                1 for m in result.matchings if int(m.template_id) == int(t.template_id)
            ),
        }
        for t in result.dictionary
    ]
    matchings_payload = [
        {
            "template_id": int(m.template_id),
            "window_id": m.window_id,
            "t_start": int(m.t_start),
            "amplitude": float(m.amplitude),
        }
        for m in result.matchings
    ]
    window_ids = [w.window_id for w in windows]

    with closing(catalog.connect(args.db)) as conn:
        run_id, new_dict_id = catalog.register_dictionary_run(
            conn,
            label=run_name,
            out_dir=out_dir,
            templates=templates_payload,
            matchings=matchings_payload,
            window_ids=window_ids,
            name=run_name,
            source="edit",
            parent_dictionary_id=parent_id,
            segmentation_id=int(segmentation_id) if segmentation_id is not None else None,
            params={
                "learner": "split_replace_refine",
                "from_dictionary_id": parent_id,
                "split_template_id": split_tid,
                "tid_a": tid_a,
                "tid_b": tid_b,
                "n_trials": args.n_trials,
                "n_refine_split": args.n_refine,
                "epsilon": args.epsilon,
                "refine_iters": args.refine_iters,
                "eps_step_fraction": args.eps_step_fraction,
                "pin_labels": True,
                "relative_sse_reduction": split.relative_reduction,
                "best_trial": split.best_trial,
            },
            energy_before=result.energies[0] if result.energies else None,
            energy_after=result.energies[-1] if result.energies else None,
            stop_reason="split_replace_refine",
        )
        conn.commit()

    counts: dict[int, int] = {}
    for m in result.matchings:
        counts[int(m.template_id)] = counts.get(int(m.template_id), 0) + 1

    e0 = result.energies[0] if result.energies else None
    e1 = result.energies[-1] if result.energies else None
    print(f"registered dictionary_id={new_dict_id}  run_id={run_id}")
    print(f"name: {run_name}")
    print(f"out_dir: {out_dir}")
    print(f"n_templates={len(list(result.dictionary))}  n_matchings={len(result.matchings)}")
    print(f"children: tid_a={tid_a} (n={counts.get(tid_a, 0)})  tid_b={tid_b} (n={counts.get(tid_b, 0)})")
    if e0 is not None and e1 is not None:
        drop = float(e0) - float(e1)
        pct = 100.0 * drop / float(e0) if float(e0) else 0.0
        print(f"energy: {e0:.6g} → {e1:.6g}  (drop {drop:.6g}, {pct:.3f}%)")
        print(f"energies trail: {[float(x) for x in result.energies]}")
    print("matchings per template:", dict(sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
