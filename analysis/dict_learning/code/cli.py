"""CLI for elementary dictionary-learning tools and legacy helpers."""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import closing
from pathlib import Path

import numpy as np

from database import catalog

from .io import discover_windows, load_gpl_inputs, load_window
from .mp import residual_energy, single_template_dictionary
from .core import Dictionary, Matching, Template
from .learn.seed_from_gpl import Seed, seed_from_gpl_paths, write_seed_result
from .learn.match_to_dictionary import match_to_dictionary, write_match_to_dictionary_result
from .tools import (
    average_templates,
    build_dictionary_from_detections,
    matching_pursuit,
    merge_templates,
    remove_template,
)


def _load_windows(args: argparse.Namespace) -> list:
    if args.npy_dir:
        return discover_windows(Path(args.npy_dir))
    if getattr(args, "window", None):
        return [load_window(Path(p)) for p in args.window]
    return []


def _load_dictionary_from_dir(templates_dir: Path) -> Dictionary:
    dictionary = Dictionary()
    for i, path in enumerate(sorted(templates_dir.glob("*.npy"))):
        dictionary.add(Template(template_id=i, data=np.load(path)))
    return dictionary


def _cmd_seed_from_gpl(args: argparse.Namespace) -> int:
    if getattr(args, "mp_compute", None):
        import os

        os.environ["DICT_LEARNING_MP_COMPUTE"] = str(args.mp_compute)
    seed = seed_from_gpl_paths(
        window_paths=args.window or None,
        npy_dir=args.npy_dir,
        csv_paths=args.csv or None,
        gpl_output_dir=args.gpl_output_dir,
        eps_step_fraction=args.eps_step_fraction,
        min_repeats=args.min_repeats,
        normalize=not getattr(args, "no_normalize", False),
        mp_mode=getattr(args, "mp_mode", None),
        n_workers=getattr(args, "mp_workers", None),
    )
    if not seed.is_selected():
        print(
            f"no template selected "
            f"(tried={seed.get_n_candidates_tried()}/{len(seed.get_candidates())} "
            f"energy0={seed.get_energy_before():.6g})"
        )
    else:
        template = seed.get_template()
        assert template is not None
        print(
            f"selected segment={seed.get_selected_segment_id()} "
            f"shape={template.data.shape} "
            f"ΔE={seed.get_delta_energy_total():.6g} "
            f"E_after={seed.get_energy_after():.6g} "
            f"n_det={len(seed.get_matchings())} "
            f"tried={seed.get_n_candidates_tried()}"
        )
    if args.out:
        path = write_seed_result(seed, args.out)
        print(f"wrote {path}")
    return 0 if seed.is_selected() else 1


def _cmd_matching_pursuit(args: argparse.Namespace) -> int:
    windows = _load_windows(args)
    if not windows:
        print("No windows loaded", file=sys.stderr)
        return 1
    residual = {w.window_id: np.array(w.data, copy=True) for w in windows}
    dictionary = Dictionary()
    if args.templates_dir:
        dictionary = _load_dictionary_from_dir(Path(args.templates_dir))
    elif args.seed_manifest:
        manifest = json.loads(Path(args.seed_manifest).read_text())
        base = Path(args.seed_manifest).parent
        tinfo = manifest.get("template")
        if not tinfo:
            print("seed_manifest has no selected template", file=sys.stderr)
            return 1
        dictionary.add(
            Template(
                template_id=int(tinfo.get("template_id", 0)),
                data=np.load(base / tinfo["file"]),
            )
        )
    else:
        print("Provide --templates-dir or --seed-manifest", file=sys.stderr)
        return 1

    e0 = residual_energy(residual)
    conn = None
    if args.db and args.input_dictionary_id is not None:
        conn = catalog.connect(args.db)
    if getattr(args, "mp_compute", None):
        import os

        os.environ["DICT_LEARNING_MP_COMPUTE"] = str(args.mp_compute)
    try:
        result = matching_pursuit(
            residual,
            dictionary,
            energy_floor=args.energy_floor,
            eps_step_fraction=args.eps_step_fraction,
            mp_mode=getattr(args, "mp_mode", None),
            n_workers=getattr(args, "mp_workers", None),
            conn=conn,
            input_dictionary_id=args.input_dictionary_id,
            out_dir=args.out,
            name=args.name,
        )
    finally:
        if conn is not None:
            conn.commit()
            conn.close()

    loop = result.loop
    print(
        f"atoms={len(loop.matchings)} energy {e0:.6g} -> {loop.energies[-1]:.6g} "
        f"stopped={loop.stopped_reason}"
        + (
            f" tool_run_id={result.tool_run_id} dict={result.output_dictionary_id}"
            if result.tool_run_id
            else ""
        )
    )
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        summary = {
            "n_matchings": len(loop.matchings),
            "energies": loop.energies,
            "stopped_reason": loop.stopped_reason,
            "tool_run_id": result.tool_run_id,
            "output_dictionary_id": result.output_dictionary_id,
            "matchings": [
                {
                    "template_id": m.template_id,
                    "window_id": m.window_id,
                    "t_start": m.t_start,
                    "amplitude": m.amplitude,
                }
                for m in loop.matchings
            ],
        }
        (out / "mp_summary.json").write_text(json.dumps(summary, indent=2))
        for wid, arr in loop.residual_stack.items():
            safe = wid.replace("/", "_")
            np.save(out / f"residual_{safe}.npy", arr)
        print(f"wrote {out / 'mp_summary.json'}")
    return 0


def _cmd_average_templates(args: argparse.Namespace) -> int:
    windows = _load_windows(args)
    if not windows:
        print("No windows loaded", file=sys.stderr)
        return 1
    dictionary = _load_dictionary_from_dir(Path(args.templates_dir))
    matchings_payload = json.loads(Path(args.matchings).read_text())
    raw = matchings_payload.get("matchings", matchings_payload)
    matchings = [
        Matching(
            template_id=int(m["template_id"]),
            window_id=str(m["window_id"]),
            t_start=int(m["t_start"]),
            amplitude=float(m.get("amplitude", 1.0)),
        )
        for m in raw
    ]
    conn = None
    if args.db and args.input_dictionary_id is not None:
        conn = catalog.connect(args.db)
    try:
        result = average_templates(
            windows,
            dictionary,
            matchings,
            conn=conn,
            input_dictionary_id=args.input_dictionary_id,
            out_dir=args.out,
            name=args.name,
        )
    finally:
        if conn is not None:
            conn.commit()
            conn.close()
    print(
        f"averaged {len(result.dictionary)} templates"
        + (
            f" tool_run_id={result.tool_run_id} dict={result.output_dictionary_id}"
            if result.tool_run_id
            else ""
        )
    )
    if args.out and result.tool_run_id is None:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for template in result.dictionary:
            np.save(out / f"template_{template.template_id:03d}.npy", template.data)
    return 0


def _cmd_match_to_dictionary(args: argparse.Namespace) -> int:
    windows = _load_windows(args)
    if not windows:
        print("No windows loaded", file=sys.stderr)
        return 1
    matchings_payload = json.loads(Path(args.matchings).read_text())
    raw = matchings_payload.get("matchings", matchings_payload)
    matchings = []
    for m in raw:
        coeffs = m.get("coeffs")
        if coeffs is not None:
            matchings.append(
                Matching(
                    template_id=int(m["template_id"]),
                    window_id=str(m["window_id"]),
                    t_start=int(m["t_start"]),
                    coeffs=np.asarray(coeffs, dtype=np.float64),
                )
            )
        else:
            matchings.append(
                Matching(
                    template_id=int(m["template_id"]),
                    window_id=str(m["window_id"]),
                    t_start=int(m["t_start"]),
                    amplitude=float(m.get("amplitude", 1.0)),
                )
            )
    if args.widths is not None and args.width is not None:
        print("pass only one of --width or --widths", file=sys.stderr)
        return 1
    if args.widths is not None:
        raw_w = json.loads(args.widths)
        widths: int | dict[int, int] = {int(k): int(v) for k, v in raw_w.items()}
    elif args.width is not None:
        widths = int(args.width)
    else:
        print("one of --width or --widths is required", file=sys.stderr)
        return 1
    if getattr(args, "mp_compute", None):
        import os

        os.environ["DICT_LEARNING_MP_COMPUTE"] = str(args.mp_compute)
    result = match_to_dictionary(
        windows,
        matchings,
        widths=widths,
        refine_iters=args.refine_iters,
        template_method=args.template_method,
        n_components=args.n_components,
        max_relative_distance=args.max_relative_distance,
        eps_step_fraction=args.eps_step_fraction,
        normalize=not args.no_normalize,
        disallow_overlap=not args.allow_overlap,
        max_repeats=args.max_repeats,
        pin_labels=not getattr(args, "no_pin_labels", False),
        show_progress=not args.no_progress,
        mp_mode=getattr(args, "mp_mode", None),
        n_workers=getattr(args, "mp_workers", None),
    )
    out_dir = None
    if args.out:
        out_dir = write_match_to_dictionary_result(
            result,
            args.out,
            name=args.name,
            params={
                "refine_iters": args.refine_iters,
                "template_method": args.template_method,
                "n_components": args.n_components,
                "pin_labels": not getattr(args, "no_pin_labels", False),
            },
        )
    print(
        json.dumps(
            {
                "n_atoms": len(result.dictionary),
                "n_matchings": len(result.matchings),
                "iterations_completed": result.iterations_completed,
                "energy_before": result.energies[0] if result.energies else None,
                "energy_after": result.energies[-1] if result.energies else None,
                "out_dir": str(out_dir) if out_dir is not None else None,
            },
            indent=2,
            default=str,
        )
    )
    return 0


def _cmd_build_from_detections(args: argparse.Namespace) -> int:
    windows, detections = load_gpl_inputs(
        window_paths=args.window or None,
        npy_dir=args.npy_dir,
        csv_paths=args.csv or None,
        gpl_output_dir=args.gpl_output_dir,
    )
    conn = catalog.connect(args.db) if args.db else None
    if getattr(args, "mp_compute", None):
        import os

        os.environ["DICT_LEARNING_MP_COMPUTE"] = str(args.mp_compute)
    try:
        result = build_dictionary_from_detections(
            windows,
            detections,
            n_rounds=args.n_rounds,
            eps_step_fraction=args.eps_step_fraction,
            min_repeats=args.min_repeats,
            normalize=not args.no_normalize,
            mp_mode=getattr(args, "mp_mode", None),
            n_workers=getattr(args, "mp_workers", None),
            conn=conn,
            out_dir=args.out,
            name=args.name,
        )
    finally:
        if conn is not None:
            conn.commit()
            conn.close()
    print(
        f"templates={len(result)} energy "
        f"{result.get_energy_before():.6g} -> {result.get_energy_after():.6g} "
        f"stop={result.get_stop_reason()}"
    )
    return 0 if len(result) else 1


def _cmd_remove_template(args: argparse.Namespace) -> int:
    if not args.db or args.input_dictionary_id is None:
        print("--db and --input-dictionary-id required", file=sys.stderr)
        return 1
    with closing(catalog.connect(args.db)) as conn:
        rows = catalog.list_dictionary_templates(conn, args.input_dictionary_id)
        dictionary = Dictionary()
        for row in rows:
            dictionary.add(
                Template(
                    template_id=int(row["template_id"]),
                    data=np.load(row["npy_path"]),
                )
            )
        result = remove_template(
            dictionary,
            template_id=args.template_id,
            template_pk=args.template_pk,
            conn=conn,
            input_dictionary_id=args.input_dictionary_id,
            name=args.name,
        )
        conn.commit()
    print(
        f"removed -> dict={result.output_dictionary_id} "
        f"tool_run_id={result.tool_run_id} n={len(result.dictionary)}"
    )
    return 0


def _cmd_merge_templates(args: argparse.Namespace) -> int:
    windows = _load_windows(args)
    if not windows:
        print("No windows loaded", file=sys.stderr)
        return 1
    if not args.db or args.input_dictionary_id is None:
        print("--db and --input-dictionary-id required", file=sys.stderr)
        return 1
    matchings_payload = json.loads(Path(args.matchings).read_text())
    raw = matchings_payload.get("matchings", matchings_payload)
    matchings = [
        Matching(
            template_id=int(m["template_id"]),
            window_id=str(m["window_id"]),
            t_start=int(m["t_start"]),
            amplitude=float(m.get("amplitude", 1.0)),
        )
        for m in raw
    ]
    with closing(catalog.connect(args.db)) as conn:
        rows = catalog.list_dictionary_templates(conn, args.input_dictionary_id)
        dictionary = Dictionary()
        for row in rows:
            dictionary.add(
                Template(
                    template_id=int(row["template_id"]),
                    data=np.load(row["npy_path"]),
                )
            )
        result = merge_templates(
            windows,
            dictionary,
            matchings,
            args.template_a,
            args.template_b,
            conn=conn,
            input_dictionary_id=args.input_dictionary_id,
            out_dir=args.out,
            name=args.name,
        )
        conn.commit()
    print(
        f"merged -> dict={result.output_dictionary_id} "
        f"tool_run_id={result.tool_run_id} merged_id={result.merged_template_id}"
    )
    return 0


def _add_mp_parallel_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mp-mode",
        choices=("per_window", "global"),
        default=None,
        help="MP mode: per_window (default) or classic global Mallat",
    )
    parser.add_argument(
        "--mp-workers",
        type=int,
        default=None,
        help="Parallel workers (default: all CPUs available to this process)",
    )
    parser.add_argument(
        "--mp-compute",
        choices=("auto", "cpu", "cuda"),
        default=None,
        help="score_shifts backend: auto (GPU when beneficial), cpu, or cuda",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m analysis.dict_learning.code.cli",
        description="Elementary dictionary-learning tools (+ legacy seed/MP helpers)",
    )
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("seed-from-gpl", help="Select one template via energy-ranked MP trials")
    s.add_argument("--window", action="append", default=[], help="Path to analysis .npy")
    s.add_argument("--npy-dir", type=str, default=None)
    s.add_argument("--csv", action="append", default=[], help="Path to CSV_*.csv")
    s.add_argument("--gpl-output-dir", type=str, default=None)
    s.add_argument("--eps-step-fraction", type=float, default=0.5)
    s.add_argument("--min-repeats", type=int, default=1)
    s.add_argument("--no-normalize", action="store_true")
    s.add_argument("--out", type=str, default=None)
    _add_mp_parallel_args(s)
    s.set_defaults(func=_cmd_seed_from_gpl)

    m = sub.add_parser("matching-pursuit", aliases=["mp-run"], help="A: Mallat matching pursuit")
    m.add_argument("--window", action="append", default=[])
    m.add_argument("--npy-dir", type=str, default=None)
    m.add_argument("--templates-dir", type=str, default=None)
    m.add_argument("--seed-manifest", type=str, default=None)
    m.add_argument("--energy-floor", type=float, default=None)
    m.add_argument("--eps-step-fraction", type=float, default=0.5)
    m.add_argument("--out", type=str, default=None)
    m.add_argument("--db", type=str, default=None, help="Catalog SQLite path (optional persist)")
    m.add_argument("--input-dictionary-id", type=int, default=None)
    m.add_argument("--name", type=str, default=None)
    _add_mp_parallel_args(m)
    m.set_defaults(func=_cmd_matching_pursuit)

    a = sub.add_parser("average-templates", help="B: Pure 1–1 average of each atom's patches")
    a.add_argument("--window", action="append", default=[])
    a.add_argument("--npy-dir", type=str, default=None)
    a.add_argument("--templates-dir", type=str, required=True)
    a.add_argument("--matchings", type=str, required=True, help="JSON with matchings list")
    a.add_argument("--out", type=str, default=None)
    a.add_argument("--db", type=str, default=None)
    a.add_argument("--input-dictionary-id", type=int, default=None)
    a.add_argument("--name", type=str, default=None)
    a.set_defaults(func=_cmd_average_templates)

    mtd = sub.add_parser(
        "match-to-dictionary",
        help="Build dictionary from labeled matchings (+ widths), then refine",
    )
    mtd.add_argument("--window", action="append", default=[])
    mtd.add_argument("--npy-dir", type=str, default=None)
    mtd.add_argument("--matchings", type=str, required=True, help="JSON with matchings list")
    mtd.add_argument("--width", type=int, default=None, help="Shared template width")
    mtd.add_argument(
        "--widths",
        type=str,
        default=None,
        help='Per-template widths JSON, e.g. \'{"0":21,"1":20}\'',
    )
    mtd.add_argument("--refine-iters", type=int, default=1)
    mtd.add_argument("--template-method", default="pca", choices=("pca", "average"))
    mtd.add_argument("--n-components", type=int, default=1)
    mtd.add_argument(
        "--max-relative-distance",
        type=float,
        default=None,
        help=(
            "Reject patches leaving more than this fraction of their energy "
            "unexplained by the atom, in MP and in template rebuilds (0-1)"
        ),
    )
    mtd.add_argument("--eps-step-fraction", type=float, default=0.2)
    mtd.add_argument("--max-repeats", type=int, default=None)
    mtd.add_argument("--no-normalize", action="store_true")
    mtd.add_argument("--allow-overlap", action="store_true")
    mtd.add_argument(
        "--no-pin-labels",
        action="store_true",
        help="Let refine MP reassign the input labels instead of keeping them",
    )
    mtd.add_argument("--no-progress", action="store_true")
    mtd.add_argument("--out", type=str, default=None)
    mtd.add_argument("--name", type=str, default=None)
    _add_mp_parallel_args(mtd)
    mtd.set_defaults(func=_cmd_match_to_dictionary)

    c = sub.add_parser(
        "build-from-detections",
        help="C: Greedy GPL single-atom dictionary growth",
    )
    c.add_argument("--window", action="append", default=[])
    c.add_argument("--npy-dir", type=str, default=None)
    c.add_argument("--csv", action="append", default=[])
    c.add_argument("--gpl-output-dir", type=str, default=None)
    c.add_argument("--n-rounds", type=int, default=5)
    c.add_argument("--eps-step-fraction", type=float, default=0.5)
    c.add_argument("--min-repeats", type=int, default=1)
    c.add_argument("--no-normalize", action="store_true")
    c.add_argument("--out", type=str, default=None)
    c.add_argument("--db", type=str, default=None)
    c.add_argument("--name", type=str, default=None)
    _add_mp_parallel_args(c)
    c.set_defaults(func=_cmd_build_from_detections)

    d = sub.add_parser("remove-template", help="D: New dictionary without one atom")
    d.add_argument("--db", type=str, required=True)
    d.add_argument("--input-dictionary-id", type=int, required=True)
    d.add_argument("--template-id", type=int, default=None)
    d.add_argument("--template-pk", type=int, default=None)
    d.add_argument("--name", type=str, default=None)
    d.set_defaults(func=_cmd_remove_template)

    e = sub.add_parser("merge-templates", help="E: Merge two atoms by averaging detections")
    e.add_argument("--window", action="append", default=[])
    e.add_argument("--npy-dir", type=str, default=None)
    e.add_argument("--db", type=str, required=True)
    e.add_argument("--input-dictionary-id", type=int, required=True)
    e.add_argument("--template-a", type=int, required=True)
    e.add_argument("--template-b", type=int, required=True)
    e.add_argument("--matchings", type=str, required=True)
    e.add_argument("--out", type=str, required=True)
    e.add_argument("--name", type=str, default=None)
    e.set_defaults(func=_cmd_merge_templates)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
