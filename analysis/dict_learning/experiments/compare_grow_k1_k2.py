"""Grow dictionaries for a pair of ``n_components`` values and summarize energy.

Produces stamped run dirs under ``--out-dir`` plus ``compare_summary.json``.
Used by ``scripts/run_compare_grow_k1_k2_leidata_wav1.sh``,
``scripts/run_compare_grow_k1_k10_leidata_wav1.sh``, and the side-by-side notebook.

Default ``--ks`` is ``1,2`` (backward compatible with the K=1 vs K=2 runner).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from ..code.core import defaults
from ..code.learn.build_dictionary import build_dictionary_for_wav


def _parse_ks(value: str | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, str):
        parts = [p.strip() for p in value.split(",") if p.strip()]
        if len(parts) != 2:
            raise ValueError(f"--ks must be exactly two ints, got {value!r}")
        ks = tuple(int(p) for p in parts)
    else:
        ks = tuple(int(k) for k in value)
        if len(ks) != 2:
            raise ValueError(f"ks must be exactly two ints, got {ks!r}")
    if any(k < 1 for k in ks):
        raise ValueError(f"each k must be >= 1, got {ks}")
    return ks


def _energy_reduction(e0: float, e1: float) -> dict[str, float]:
    removed = float(e0) - float(e1)
    frac = removed / float(e0) if e0 else 0.0
    return {
        "energy_before": float(e0),
        "energy_after": float(e1),
        "energy_removed": removed,
        "fraction_removed": frac,
        "percent_removed": 100.0 * frac,
    }


def run_grow_compare(
    *,
    db: Path | str,
    wav: str,
    out_dir: Path | str,
    n_windows: int = 4,
    n_templates: int = 5,
    refine_iters: int = 2,
    min_templates: int = 5,
    eps_step_fraction: float = 0.3,
    min_repeats: int = 3,
    min_template_width: int | None = 12,
    max_template_width: int | None = 24,
    template_method: str = "pca",
    ks: str | Sequence[int] = (1, 2),
    seed_window_policy: str = defaults.SEED_WINDOW_POLICY,
    seed_window_neighborhood: int = defaults.SEED_WINDOW_NEIGHBORHOOD,
    max_relative_distance: float | None = defaults.MAX_RELATIVE_DISTANCE,
    show_progress: bool | None = True,
) -> dict[str, Any]:
    """Build dictionaries for each ``k`` in ``ks``; require ``>= min_templates`` each."""
    ks_tuple = _parse_ks(ks)
    if n_windows < 4:
        raise ValueError(f"n_windows must be >= 4 for this compare, got {n_windows}")
    if n_templates < min_templates:
        raise ValueError(
            f"n_templates (n_rounds) must be >= min_templates={min_templates}, "
            f"got {n_templates}"
        )
    if refine_iters < 2:
        raise ValueError(f"refine_iters must be >= 2 for this compare, got {refine_iters}")

    out_root = Path(out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {
        "wav": wav,
        "n_windows": n_windows,
        "n_templates_requested": n_templates,
        "refine_iters": refine_iters,
        "min_templates": min_templates,
        "eps_step_fraction": eps_step_fraction,
        "min_repeats": min_repeats,
        "template_method": template_method,
        "seed_window_policy": seed_window_policy,
        "seed_window_neighborhood": seed_window_neighborhood,
        "max_relative_distance": max_relative_distance,
        "ks": list(ks_tuple),
        "runs": {},
    }

    run_labels: list[str] = []
    for k in ks_tuple:
        label = f"k{k}"
        run_labels.append(label)
        k_out = out_root / f"wav1_grow_{label}"
        print(f"\n=== Growing dictionary n_components={k} → {k_out} ===", flush=True)
        summary = build_dictionary_for_wav(
            db,
            wav,
            n_windows=n_windows,
            window_offset=0,
            n_rounds=n_templates,
            out_dir=k_out,
            name=f"leidata_wav1_grow_k{k}",
            eps_step_fraction=eps_step_fraction,
            min_repeats=min_repeats,
            normalize=True,
            min_template_width=min_template_width,
            max_template_width=max_template_width,
            template_method=template_method,
            n_components=k,
            refine_iters=refine_iters,
            seed_window_policy=seed_window_policy,
            seed_window_neighborhood=seed_window_neighborhood,
            max_relative_distance=max_relative_distance,
            show_progress=show_progress,
        )
        n_got = int(summary["n_templates"])
        energy = _energy_reduction(summary["energy_before"], summary["energy_after"])
        results["runs"][label] = {
            "n_components": k,
            "out_dir": summary["out_dir"],
            "run_stamp": summary["run_stamp"],
            "dictionary_id": summary["dictionary_id"],
            "n_templates": n_got,
            "n_matchings": summary["n_matchings"],
            "n_windows": summary["n_windows"],
            "stop_reason": summary["stop_reason"],
            **energy,
        }
        print(
            f"K={k}: templates={n_got} matchings={summary['n_matchings']} "
            f"E {energy['energy_before']:.6g} → {energy['energy_after']:.6g} "
            f"({energy['percent_removed']:.2f}% removed) "
            f"dir={summary['out_dir']}",
            flush=True,
        )

    low_label, high_label = run_labels
    low = results["runs"][low_label]
    high = results["runs"][high_label]
    results["delta_percent_removed"] = (
        float(high["percent_removed"]) - float(low["percent_removed"])
    )
    summary_path = out_root / "compare_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    results["summary_path"] = str(summary_path)
    print(f"\nWrote {summary_path}", flush=True)

    # Enforce template floor after persisting the summary so notebooks can still load.
    shortfalls = []
    for label, run in results["runs"].items():
        if int(run["n_templates"]) < min_templates:
            shortfalls.append(
                f"{label} produced {run['n_templates']} templates "
                f"(need >= {min_templates}); out_dir={run['out_dir']}"
            )
    if shortfalls:
        raise RuntimeError(
            "Insufficient templates:\n  - "
            + "\n  - ".join(shortfalls)
            + "\nTry more windows, lower --min-repeats, or larger --n-templates. "
            f"Partial summary kept at {summary_path}"
        )

    print(
        f"Energy removed: K={ks_tuple[0]} {low['percent_removed']:.2f}%  "
        f"K={ks_tuple[1]} {high['percent_removed']:.2f}%  "
        f"(Δ = {results['delta_percent_removed']:+.2f} pp)",
        flush=True,
    )
    return results


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m analysis.dict_learning.experiments.compare_grow_k1_k2",
        description=(
            "Grow dictionaries for two n_components values on the same windows "
            "(≥4), ≥5 templates each, ≥2 refine iters; write compare_summary.json."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db", required=True)
    p.add_argument("--wav", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--ks",
        default="1,2",
        help="Comma-separated pair of n_components values (low,high), e.g. 1,10",
    )
    p.add_argument("--n-windows", type=int, default=4)
    p.add_argument("--n-templates", type=int, default=5, help="Outer rounds / template cap")
    p.add_argument("--min-templates", type=int, default=5)
    p.add_argument("--refine-iters", type=int, default=2)
    p.add_argument("--eps-step-fraction", type=float, default=0.3)
    p.add_argument("--min-repeats", type=int, default=defaults.MIN_REPEATS)
    p.add_argument("--min-template-width", type=int, default=12)
    p.add_argument("--max-template-width", type=int, default=24)
    p.add_argument("--template-method", choices=("average", "pca"), default="pca")
    p.add_argument(
        "--seed-window-policy",
        choices=("global", "low_coverage"),
        default=defaults.SEED_WINDOW_POLICY,
    )
    p.add_argument(
        "--seed-window-neighborhood",
        type=int,
        default=defaults.SEED_WINDOW_NEIGHBORHOOD,
    )
    p.add_argument(
        "--max-relative-distance",
        type=float,
        default=defaults.MAX_RELATIVE_DISTANCE,
        help=(
            "Reject patches leaving more than this fraction of their energy "
            "unexplained by the atom (0-1)"
        ),
    )
    p.add_argument("--no-progress", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    try:
        run_grow_compare(
            db=args.db,
            wav=args.wav,
            out_dir=args.out_dir,
            n_windows=args.n_windows,
            n_templates=args.n_templates,
            refine_iters=args.refine_iters,
            min_templates=args.min_templates,
            eps_step_fraction=args.eps_step_fraction,
            min_repeats=args.min_repeats,
            min_template_width=args.min_template_width,
            max_template_width=args.max_template_width,
            template_method=args.template_method,
            ks=args.ks,
            seed_window_policy=args.seed_window_policy,
            seed_window_neighborhood=args.seed_window_neighborhood,
            max_relative_distance=args.max_relative_distance,
            show_progress=False if args.no_progress else True,
        )
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
