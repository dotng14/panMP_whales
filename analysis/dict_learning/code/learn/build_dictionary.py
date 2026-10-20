"""Build and store a dictionary for one WAV from its GPL segmentation.

Uses :func:`learn_averaged_templates_from_gpl` (seed → average → MP each round),
optionally :func:`refine_dictionary_by_averaging`, writes template ``.npy`` files,
and registers one dictionary + matchings in the catalog (the analysis↔backend bus).

Usage::

    python -m analysis.dict_learning.code.learn.build_dictionary \\
        --db catalog.sqlite \\
        --wav /data/recordings/sub/clip.wav \\
        [--n-windows 10] [--n-rounds 20] \\
        [--eps-step-fraction 0.2] [--min-repeats 3] \\
        [--max-template-width 24] [--refine-iters 0] \\
        [--out-dir DIR] [--name NAME]
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np

from database import catalog
from database.migrate_flat_dict_dirs import utc_run_stamp

from ..core import defaults
from ..io.io_gpl import load_detections_csv, load_windows_from_paths
from .learn_dictionary import (
    DictionaryLearningError,
    LearnedDictionary,
    LearningRoundLog,
    learn_averaged_templates_from_gpl,
    refine_dictionary_by_averaging,
)
from ..core.objects import Dictionary
from .seed_from_gpl import Seed


def register_averaged_result(
    conn: Any,
    *,
    result: LearnedDictionary,
    out_dir: Path | str,
    label: str,
    segmentation_id: int | None = None,
    window_ids: list[str] | None = None,
    params: dict[str, Any] | None = None,
    name: str | None = None,
    run_id: int | None = None,
    register_catalog: bool = True,
    source: str = "seed",
    parent_dictionary_id: int | None = None,
) -> tuple[int | None, int | None]:
    """Persist an averaged-templates result to disk, and optionally the catalog.

    When any atom has ``n_components > 1``, writes a subspace manifest and
    skips catalog registration (``register_catalog`` is forced off). Returns
    ``(run_id, dictionary_id)``; both may be ``None`` when catalog is skipped.

    ``source`` / ``parent_dictionary_id`` follow catalog rules: ``seed`` requires
    no parent; ``refinement`` (and other non-seed sources) require a parent.
    """
    out = Path(out_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    templates = result.get_templates()
    matchings = result.get_matchings()
    if window_ids is None:
        window_ids = sorted({m.window_id for m in matchings if m.window_id})

    n_components = max((t.n_components for t in templates), default=1)
    kind = "subspace" if n_components > 1 else "rank1"
    if n_components > 1:
        register_catalog = False

    template_rows: list[dict[str, Any]] = []
    for template in templates:
        prefix = "subspace" if template.n_components > 1 else "template"
        npy_path = out / f"{prefix}_{template.template_id:03d}.npy"
        np.save(npy_path, template.data)
        row: dict[str, Any] = {
            "template_id": template.template_id,
            "npy_path": npy_path,
            "n_freq": template.n_freq,
            "width": template.width,
            "source_segment_id": template.source_segment_id,
            "n_components": template.n_components,
            "file": npy_path.name,
            "shape": list(template.data.shape),
        }
        if template.variances is not None:
            row["variances"] = np.asarray(template.variances, dtype=np.float64).tolist()
        template_rows.append(row)

    matching_rows: list[dict[str, Any]] = []
    for m in matchings:
        row = {
            "template_id": m.template_id,
            "window_id": m.window_id,
            "t_start": m.t_start,
            "amplitude": float(m.amplitude),
        }
        coeffs = np.asarray(m.coeffs, dtype=np.float64).reshape(-1)
        if coeffs.size > 1 or n_components > 1:
            row["coeffs"] = coeffs.tolist()
        matching_rows.append(row)

    energy_before = result.get_energy_before()
    energy_after = result.get_energy_after()
    round_logs = [log.to_dict() for log in result.get_round_logs()]
    manifest_params = dict(params or {})
    manifest_params.setdefault("n_components", n_components)
    (out / "manifest.json").write_text(
        json.dumps(
            {
                "name": name or label,
                "kind": kind,
                "n_components": n_components,
                "params": manifest_params,
                "energy_before": energy_before,
                "energy_after": energy_after,
                "stop_reason": result.get_stop_reason(),
                "window_ids": window_ids,
                "round_logs": round_logs,
                "templates": [
                    {
                        "template_id": t["template_id"],
                        "file": t["file"],
                        "shape": t["shape"],
                        "n_components": t["n_components"],
                        "n_freq": t["n_freq"],
                        "width": t["width"],
                        "source_segment_id": t["source_segment_id"],
                        **(
                            {"variances": t["variances"]}
                            if "variances" in t
                            else {}
                        ),
                    }
                    for t in template_rows
                ],
                "matchings": matching_rows,
            },
            separators=(",", ":"),
        )
    )
    if not register_catalog:
        return run_id, None

    catalog_template_rows = [
        {
            "template_id": t["template_id"],
            "npy_path": t["npy_path"],
            "n_freq": t["n_freq"],
            "width": t["width"],
            "source_segment_id": t["source_segment_id"],
        }
        for t in template_rows
    ]
    catalog_matching_rows = [
        {
            "template_id": m["template_id"],
            "window_id": m["window_id"],
            "t_start": m["t_start"],
            "amplitude": m["amplitude"],
        }
        for m in matching_rows
    ]
    return catalog.register_dictionary_run(
        conn,
        label=label,
        out_dir=out,
        templates=catalog_template_rows,
        matchings=catalog_matching_rows,
        window_ids=window_ids,
        name=name or label,
        source=source,
        parent_dictionary_id=parent_dictionary_id,
        segmentation_id=segmentation_id,
        params=params,
        energy_before=energy_before,
        energy_after=energy_after,
        run_id=run_id,
        stop_reason=result.get_stop_reason(),
    )


def apply_refine_iters(
    windows: list,
    result: LearnedDictionary,
    *,
    refine_iters: int,
    eps_step_fraction: float,
    normalize: bool,
    template_method: str = "pca",
    n_components: int = 1,
    max_relative_distance: float | None = None,
    show_progress: bool | None = None,
) -> LearnedDictionary:
    """Optional multi-atom average→MP pass after seed→average rounds.

    If refinement hits a recoverable :class:`DictionaryLearningError` (e.g. an
    atom loses all its matches during joint MP), the unrefined result is kept
    and its ``stop_reason`` records why refinement stopped.
    """
    if refine_iters <= 0:
        return result
    dictionary = Dictionary()
    for template in result.get_templates():
        dictionary.add(template)
    try:
        refined = refine_dictionary_by_averaging(
            windows,
            dictionary,
            result.get_matchings(),
            iter=refine_iters,
            eps_step_fraction=eps_step_fraction,
            normalize=normalize,
            disallow_overlap=True,
            template_method=template_method,
            n_components=n_components,
            max_relative_distance=max_relative_distance,
            show_progress=show_progress,
        )
    except DictionaryLearningError as exc:
        print(
            f"warning: refinement aborted, keeping unrefined dictionary: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return LearnedDictionary(
            seeds=result.get_seeds(),
            energy_before=result.get_energy_before(),
            energy_after=result.get_energy_after(),
            residual_after=result.get_residual_after(),
            round_logs=result.get_round_logs(),
            stop_reason=f"refine_aborted: {exc}",
        )
    seeds = [
        Seed(
            template=template,
            matchings=[m for m in refined.matchings if m.template_id == template.template_id],
            residual_after=refined.residual_after,
            energy_before=refined.energies[0] if refined.energies else result.get_energy_before(),
            energy_after=refined.energies[-1] if refined.energies else result.get_energy_after(),
        )
        for template in refined.dictionary
    ]
    return LearnedDictionary(
        seeds=seeds,
        energy_before=result.get_energy_before(),
        energy_after=refined.energies[-1] if refined.energies else result.get_energy_after(),
        residual_after=refined.residual_after,
        round_logs=result.get_round_logs(),
        stop_reason=result.get_stop_reason(),
    )


def _assert_non_overlapping_matchings(result: LearnedDictionary) -> None:
    """Fail before persistence if any two final detections overlap."""
    widths = {template.template_id: template.width for template in result.get_templates()}
    by_window: dict[str, list[tuple[int, int, int]]] = {}
    for matching in result.get_matchings():
        width = widths[matching.template_id]
        by_window.setdefault(matching.window_id, []).append(
            (matching.t_start, matching.t_start + width, matching.template_id)
        )

    for window_id, intervals in by_window.items():
        intervals.sort()
        previous = intervals[0] if intervals else None
        for current in intervals[1:]:
            assert previous is not None
            if current[0] < previous[1]:
                raise RuntimeError(
                    "overlapping dictionary detections: "
                    f"window_id={window_id!r}, "
                    f"template_id={previous[2]} interval=[{previous[0]}, {previous[1]}), "
                    f"template_id={current[2]} interval=[{current[0]}, {current[1]})"
                )
            previous = current


def build_dictionary_for_wav(
    db_path: Path | str,
    wav: str,
    *,
    n_windows: int = defaults.N_WINDOWS,
    window_offset: int = defaults.WINDOW_OFFSET,
    n_rounds: int | None = defaults.N_ROUNDS,
    out_dir: Path | str | None = None,
    name: str | None = None,
    segmentation_id: int | None = None,
    eps_step_fraction: float = defaults.EPS_STEP_FRACTION,
    min_repeats: int = defaults.MIN_REPEATS,
    normalize: bool = defaults.NORMALIZE,
    min_template_width: int | None = defaults.MIN_TEMPLATE_WIDTH,
    max_template_width: int | None = defaults.MAX_TEMPLATE_WIDTH,
    template_method: str = defaults.TEMPLATE_METHOD,
    n_components: int = defaults.N_COMPONENTS,
    refine_iters: int = defaults.REFINE_ITERS,
    seed_window_policy: str = defaults.SEED_WINDOW_POLICY,
    seed_window_neighborhood: int = defaults.SEED_WINDOW_NEIGHBORHOOD,
    max_relative_distance: float | None = defaults.MAX_RELATIVE_DISTANCE,
    template_out_dir: Path | str | None = None,
    show_progress: bool | None = None,
) -> dict[str, Any]:
    """Learn averaged templates for one catalog WAV and register the dictionary.

    Mallat peeling is controlled by ``eps_step_fraction``. Outer rounds grow
    until no GPL candidate yields ``>= min_repeats`` matches in both the seed
    trial and the averaged re-MP; ``n_rounds`` is an optional safety cap.
    ``min_template_width`` / ``max_template_width`` filter GPL candidates by
    time-bin width. ``refine_iters`` runs an optional multi-atom
    average→MP pass after the outer seed→average rounds.

    ``n_components > 1`` builds subspace atoms and writes a manifest only (no
    catalog schema change).
    """
    if n_windows < 1:
        raise ValueError(f"n_windows must be >= 1, got {n_windows}")
    if n_rounds is not None and n_rounds < 1:
        raise ValueError(f"n_rounds must be >= 1 when set, got {n_rounds}")
    if window_offset < 0:
        raise ValueError(f"window_offset must be >= 0, got {window_offset}")
    if refine_iters < 0:
        raise ValueError(f"refine_iters must be >= 0, got {refine_iters}")
    if min_repeats < 1:
        raise ValueError(f"min_repeats must be >= 1, got {min_repeats}")
    if eps_step_fraction <= 0.0:
        raise ValueError(f"eps_step_fraction must be > 0, got {eps_step_fraction}")
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")

    with closing(catalog.connect(db_path)) as conn:
        wav_row = catalog.find_wav_file(conn, wav)
        if wav_row is None:
            raise ValueError(f"WAV not found in catalog: {wav!r} (index it first)")

        segmentations = catalog.list_gpl_segmentations_for_wav(conn, wav_row["wav_id"])
        if not segmentations:
            raise ValueError(f"no GPL segmentation in catalog for {wav_row['stem']}")
        if segmentation_id is not None:
            seg = next(
                (s for s in segmentations if s["segmentation_id"] == segmentation_id),
                None,
            )
            if seg is None:
                raise ValueError(
                    f"segmentation_id={segmentation_id} not found for {wav_row['stem']}"
                )
        else:
            seg = segmentations[0]

        window_rows = catalog.list_windows_for_segmentation(
            conn,
            seg["segmentation_id"],
            limit=n_windows,
            offset=window_offset,
        )
        if not window_rows:
            raise ValueError(
                f"segmentation {seg['segmentation_id']} has no windows "
                f"at offset={window_offset} limit={n_windows}"
            )
        if seg["csv_path"] is None or not Path(seg["csv_path"]).is_file():
            raise ValueError(f"segmentation {seg['segmentation_id']} has no detection CSV")

        windows = load_windows_from_paths([r["npy_path"] for r in window_rows])
        detections = load_detections_csv(Path(seg["csv_path"]))

        run_name = name or f"{wav_row['stem']}_d0_w{len(windows)}"
        base_out = Path(out_dir).resolve() if out_dir else Path.cwd() / "dictionaries" / run_name
        stamp = utc_run_stamp()
        out = base_out / stamp
        out.mkdir(parents=True, exist_ok=False)
        round_out: Path | None = None
        if template_out_dir is not None:
            round_out = Path(template_out_dir).resolve() / stamp
            round_out.mkdir(parents=True, exist_ok=True)
        params = {
            "n_windows": len(windows),
            "window_offset": window_offset,
            "n_rounds": n_rounds,
            "eps_step_fraction": eps_step_fraction,
            "min_repeats": min_repeats,
            "normalize": normalize,
            "disallow_overlap": True,
            "min_template_width": min_template_width,
            "max_template_width": max_template_width,
            "template_method": template_method,
            "n_components": n_components,
            "refine_iters": refine_iters,
            "seed_window_policy": seed_window_policy,
            "seed_window_neighborhood": seed_window_neighborhood,
            "max_relative_distance": max_relative_distance,
            "wav": wav_row["abs_path"],
            "segmentation_id": seg["segmentation_id"],
            "learner": "learn_averaged_templates_from_gpl",
            "base_out_dir": str(base_out),
            "run_stamp": stamp,
        }
        run_id = catalog.start_learning_run(
            conn,
            label=run_name,
            out_dir=out,
            params=params,
            n_windows=len(windows),
            n_rounds_requested=n_rounds,
            n_units=None,
            eps_step_fraction=eps_step_fraction,
            max_atoms=None,
        )
        orch_tool_run_id = catalog.insert_tool_run(
            conn,
            tool="orchestrated_build",
            segmentation_id=seg["segmentation_id"],
            learning_run_id=run_id,
            params=params,
            out_dir=out,
        )
        conn.commit()
        pending_round_logs: list[LearningRoundLog] = []

        def _on_round(log: LearningRoundLog) -> None:
            # Defer SQLite writes to end of learning (same rows; less fsync).
            pending_round_logs.append(log)
            print(log.summary_line(), flush=True)

        try:
            result = learn_averaged_templates_from_gpl(
                windows,
                detections,
                n_windows=n_windows,
                window_offset=0,  # already sliced via catalog offset
                n_rounds=n_rounds,
                eps_step_fraction=eps_step_fraction,
                min_repeats=min_repeats,
                normalize=normalize,
                disallow_overlap=True,
                min_template_width=min_template_width,
                max_template_width=max_template_width,
                template_method=template_method,
                n_components=n_components,
                seed_window_policy=seed_window_policy,
                seed_window_neighborhood=seed_window_neighborhood,
                max_relative_distance=max_relative_distance,
                template_out_dir=round_out,
                show_progress=show_progress,
                on_round_complete=_on_round,
            )
            result = apply_refine_iters(
                windows,
                result,
                refine_iters=refine_iters,
                eps_step_fraction=eps_step_fraction,
                normalize=normalize,
                template_method=template_method,
                n_components=n_components,
                max_relative_distance=max_relative_distance,
                show_progress=show_progress,
            )
            _assert_non_overlapping_matchings(result)
            for log in pending_round_logs:
                catalog.insert_learning_round_log(conn, run_id, log.to_dict())
            templates = result.get_templates()
            if not templates:
                catalog.finish_tool_run(
                    conn,
                    orch_tool_run_id,
                    stop_reason=result.get_stop_reason() or "no_templates",
                    energy_before=result.get_energy_before(),
                    energy_after=result.get_energy_after(),
                )
                catalog.finish_learning_run(
                    conn,
                    run_id,
                    status="failed",
                    stop_reason=result.get_stop_reason() or "no_templates",
                    error_message="learning selected no templates (no usable detections?)",
                    energy_before=result.get_energy_before(),
                    energy_after=result.get_energy_after(),
                    n_rounds_completed=len(result.get_round_logs()),
                )
                conn.commit()
                raise RuntimeError("learning selected no templates (no usable detections?)")

            run_name = name or f"{wav_row['stem']}_d{len(templates)}_w{len(windows)}"
            params["n_windows"] = len(windows)
            run_id, dictionary_id = register_averaged_result(
                conn,
                result=result,
                out_dir=out,
                label=run_name,
                segmentation_id=seg["segmentation_id"],
                window_ids=[w.window_id for w in windows],
                params=params,
                name=run_name,
                run_id=run_id,
            )
            if dictionary_id is None:
                # Subspace / manifest-only: close the learning_run without a catalog dict.
                catalog.finish_learning_run(
                    conn,
                    run_id,
                    status="completed",
                    stop_reason=result.get_stop_reason() or "manifest_only",
                    energy_before=result.get_energy_before(),
                    energy_after=result.get_energy_after(),
                    n_rounds_completed=len(result.get_round_logs()),
                )
                print(
                    f"note: n_components={n_components} → manifest-only at {out} "
                    "(catalog dictionary registration skipped)",
                    flush=True,
                )
            catalog.finish_tool_run(
                conn,
                orch_tool_run_id,
                output_dictionary_id=dictionary_id,
                energy_before=result.get_energy_before(),
                energy_after=result.get_energy_after(),
                stop_reason=result.get_stop_reason(),
                out_dir=out,
                params=params,
            )
            conn.commit()
        except Exception as exc:
            try:
                for log in pending_round_logs:
                    catalog.insert_learning_round_log(conn, run_id, log.to_dict())
            except Exception:
                pass
            row = conn.execute(
                "SELECT status FROM learning_runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is not None and row["status"] == "running":
                catalog.finish_tool_run(
                    conn,
                    orch_tool_run_id,
                    stop_reason="exception",
                )
                catalog.finish_learning_run(
                    conn,
                    run_id,
                    status="failed",
                    stop_reason="exception",
                    error_message=str(exc),
                )
                conn.commit()
            raise

    try:
        from ..mp import loop_stats

        if loop_stats.enabled():
            loop_stats.dump_json(out / "loop_stats.json")
    except Exception as exc:
        print(f"warning: could not write loop_stats.json: {exc}", file=__import__("sys").stderr)

    return {
        "run_id": run_id,
        "dictionary_id": dictionary_id,
        "name": run_name,
        "wav_id": wav_row["wav_id"],
        "segmentation_id": seg["segmentation_id"],
        "out_dir": str(out),
        "base_out_dir": str(base_out),
        "run_stamp": stamp,
        "n_windows": len(windows),
        "window_offset": window_offset,
        "n_templates": len(templates),
        "n_matchings": len(result.get_matchings()),
        "n_rounds_logged": len(result.get_round_logs()),
        "stop_reason": result.get_stop_reason(),
        "energy_before": result.get_energy_before(),
        "energy_after": result.get_energy_after(),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m analysis.dict_learning.code.learn.build_dictionary",
        description=(
            "Learn averaged templates for one WAV (seed→average→MP) and store them "
            "in the catalog. Mallat peeling is controlled by --eps-step-fraction; "
            "outer rounds stop when no candidate yields --min-repeats matches."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db", required=True, help="Path to the catalog SQLite file")
    p.add_argument("--wav", required=True, help="WAV path, stem, or filename (must be indexed)")
    p.add_argument(
        "--out-dir",
        default=None,
        help="Base directory for template .npy files (each run writes under a UTC-stamped subdirectory)",
    )
    p.add_argument("--name", default=None, help="Dictionary/run name")
    p.add_argument(
        "--segmentation-id",
        type=int,
        default=None,
        help="Use this GPL segmentation (default: first for the WAV)",
    )

    win = p.add_argument_group("windows / rounds")
    win.add_argument("--n-windows", type=int, default=defaults.N_WINDOWS)
    win.add_argument("--window-offset", type=int, default=defaults.WINDOW_OFFSET,
                     help="Skip this many windows before taking --n-windows")
    win.add_argument(
        "--n-rounds",
        type=int,
        default=defaults.N_ROUNDS,
        help="Optional max outer rounds (default: grow until below min-repeats)",
    )

    stop = p.add_argument_group("stopping")
    stop.add_argument("--eps-step-fraction", type=float, default=defaults.EPS_STEP_FRACTION,
                      help="MP stop: relative energy improvement threshold")
    stop.add_argument("--min-repeats", type=int, default=defaults.MIN_REPEATS,
                      help="Require at least this many matches in both seed and averaged MP")

    avg = p.add_argument_group("optional refine")
    avg.add_argument("--refine-iters", type=int, default=defaults.REFINE_ITERS,
                     help="Extra multi-atom average→MP iterations after outer rounds")

    cand = p.add_argument_group("candidate filters")
    cand.add_argument("--min-template-width", type=int, default=defaults.MIN_TEMPLATE_WIDTH,
                      help="Drop GPL patches narrower than this (time bins)")
    cand.add_argument("--max-template-width", type=int, default=defaults.MAX_TEMPLATE_WIDTH,
                      help="Drop GPL patches wider than this (time bins)")
    cand.add_argument("--template-method", choices=("average", "pca"),
                      default=defaults.TEMPLATE_METHOD,
                      help="How to rebuild each atom from its patches: mean or leading PC")
    cand.add_argument(
        "--n-components",
        type=int,
        default=defaults.N_COMPONENTS,
        help="PCA modes per atom (1=rank-1 catalog path; >1=subspace manifest-only)",
    )
    cand.add_argument(
        "--seed-window-policy",
        choices=("global", "low_coverage"),
        default=defaults.SEED_WINDOW_POLICY,
        help="Where to pick seed patches: all windows, or least-covered window (+ neighborhood)",
    )
    cand.add_argument(
        "--seed-window-neighborhood",
        type=int,
        default=defaults.SEED_WINDOW_NEIGHBORHOOD,
        help="With low_coverage: also seed from ±N adjacent windows",
    )
    cand.add_argument(
        "--max-relative-distance",
        type=float,
        default=defaults.MAX_RELATIVE_DISTANCE,
        help=(
            "Reject patches leaving more than this fraction of their energy "
            "unexplained by the atom, in MP and in template rebuilds "
            "(0-1; default None = no gate)"
        ),
    )
    cand.add_argument("--no-normalize", action="store_true",
                      help="Disable template L2 normalization in MP")
    cand.add_argument("--template-out-dir", default=None,
                      help="Also write per-round round_XXX/ dumps during learning")
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm progress bars (default: on for this CLI)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    summary = build_dictionary_for_wav(
        args.db,
        args.wav,
        n_windows=args.n_windows,
        window_offset=args.window_offset,
        n_rounds=args.n_rounds,
        out_dir=args.out_dir,
        name=args.name,
        segmentation_id=args.segmentation_id,
        eps_step_fraction=args.eps_step_fraction,
        min_repeats=args.min_repeats,
        normalize=not args.no_normalize,
        min_template_width=args.min_template_width,
        max_template_width=args.max_template_width,
        template_method=args.template_method,
        n_components=args.n_components,
        refine_iters=args.refine_iters,
        seed_window_policy=args.seed_window_policy,
        seed_window_neighborhood=args.seed_window_neighborhood,
        max_relative_distance=args.max_relative_distance,
        template_out_dir=args.template_out_dir,
        show_progress=False if args.no_progress else True,
    )
    dict_note = (
        f"id={summary['dictionary_id']}"
        if summary["dictionary_id"] is not None
        else "manifest-only"
    )
    print(
        f"dictionary '{summary['name']}' ({dict_note}) registered: "
        f"run_id={summary['run_id']} templates={summary['n_templates']} "
        f"matchings={summary['n_matchings']} windows={summary['n_windows']} "
        f"offset={summary['window_offset']} rounds_logged={summary['n_rounds_logged']} "
        f"stop={summary['stop_reason']} "
        f"E0={summary['energy_before']:.6g} E1={summary['energy_after']:.6g}"
    )
    print(f"templates written to {summary['out_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
