"""Command line for the dictionary builds and Mallat matching pursuit.

The commands read a GPL output folder directly: ``spectrogram_npy/`` (whitened
windows plus JSON sidecars) and, for ``randomized-build``, ``CSV_*.csv``
(detections)::

    python -m analysis.dict_learning randomized-build --gpl-dir GPL_OUT --out DICT_DIR
    python -m analysis.dict_learning pan-build --gpl-dir GPL_OUT --out DICT_DIR
    python -m analysis.dict_learning matching-pursuit --gpl-dir GPL_OUT \\
        --dictionary DICT_DIR --out MP_DIR
    python -m analysis.dict_learning split --gpl-dir GPL_OUT \\
        --dictionary DICT_DIR --template 3 --out SPLIT_DIR

The two builds share the outer loop and differ in where seeds come from:
random detection patches, or the pan matrix profile
(``learn/pan_seeds.py``). Both write ``atoms/template_*.npy``,
``summary.json``, ``matchings.json`` and ``outer_logs.jsonl``; ``pan-build``
adds ``pan_seeds.jsonl``. ``matching-pursuit`` places the
atoms listed in a dictionary's ``summary.json`` and writes ``summary.json`` and
``matchings.json``. ``split`` replaces one atom of a dictionary by two
(``learn/split_template.py``) and writes a new dictionary folder in the build
layout. With ``--catalog DB`` every command also registers its output folder
in the SQLite catalog (``database/``).
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
from .learn.pan_seeds import (
    DEFAULT_COPY_THRESHOLD,
    DEFAULT_DEDUPE_CORR,
    DEFAULT_MIN_COPIES,
    DEFAULT_SIGNAL,
    PanSeedConfig,
    PanSeeder,
)
from analysis.matrix_profile.signals import DEFAULT_PCA_ROWS, GPL_KINDS
from .learn.split_template import split_template_matchings
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


def _register(catalog_db: Path | None, out_dir: Path) -> None:
    """Add the output folder to the catalog when ``--catalog`` was given."""
    if catalog_db is None:
        return
    from contextlib import closing

    from database import catalog

    with closing(catalog.connect(catalog_db)) as conn:
        try:
            with conn:
                result = catalog.register_run_dir(conn, out_dir)
        except catalog.CatalogError as exc:
            raise CliError(f"could not register {out_dir} in {catalog_db}: {exc}") from exc
    print(json.dumps({"catalog": str(catalog_db), **result}))


def _probability(name: str, value: float | None) -> float | None:
    if value is not None and not 0.0 <= value <= 1.0:
        raise CliError(f"{name} must be in [0, 1], got {value}")
    return value


# ── randomized-build and pan-build ───────────────────────────────────────────


def _overlap(args: argparse.Namespace) -> mp_module.OverlapRule | None:
    try:
        return mp_module.overlap_rule(args.overlap, args.max_partial_overlap, args.max_nested_ratio)
    except ValueError as exc:
        raise CliError(str(exc)) from exc


def _overlap_params(args: argparse.Namespace) -> dict[str, Any]:
    """``overlap`` and, when limited, its two limits, for ``summary.json``."""
    if args.overlap != "limited":
        return {"overlap": args.overlap}
    return {
        "overlap": args.overlap,
        "max_partial_overlap": args.max_partial_overlap,
        "max_nested_ratio": args.max_nested_ratio,
    }


def _outer_loop_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    """Options both builds pass to ``learn_randomized_outer``."""
    return {
        "m_per_outer": args.m_per_outer,
        "min_coverage": args.min_coverage,
        "max_outer": args.max_outer,
        "refine_iters": args.refine_iters,
        "n_components": args.n_components,
        "n_min": args.n_min,
        "eps_step_fraction": args.eps_step_fraction,
        "max_relative_distance": args.max_relative_distance,
        "post_refine_max_relative_distance": args.post_refine_max_relative_distance,
        "mp_mode": args.mp_mode,
        "n_workers": args.workers,
        "show_progress": args.progress,
        "overlap": _overlap(args),
    }


def _run_build(args: argparse.Namespace, windows: list[Window], partition, **extra) -> rd.RandomizedOuterResult:
    """Run the outer loop with ``--verbose`` lines, ``--loop-stats`` and ``--compute``."""
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
                windows, partition, **_outer_loop_kwargs(args), progress_callback=on_log, **extra
            )
        except ValueError as exc:
            raise CliError(str(exc)) from exc

    return _run_with_stats(args.loop_stats, args.out, build)


def _write_build(
    args: argparse.Namespace,
    command: str,
    windows: list[Window],
    result: rd.RandomizedOuterResult,
    params: dict[str, Any],
) -> int:
    """Write a build folder (atoms, matchings, outer logs, summary); exit 1 when no atom survived."""
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
        "command": command,
        "gpl_dir": str(args.gpl_dir.resolve()),
        "wav": str(args.gpl_dir.resolve()),
        "n_windows": len(windows),
        "window_ids": [window.window_id for window in windows],
        "window_offset": args.window_offset,
        "sec_per_bin": _sec_per_bin(windows[0]),
        "m_per_outer": args.m_per_outer,
        "min_coverage": args.min_coverage,
        "max_outer": args.max_outer,
        "refine_iters": args.refine_iters,
        "n_components": args.n_components,
        "n_min": args.n_min,
        **params,
        "eps_step_fraction": args.eps_step_fraction,
        "max_relative_distance": args.max_relative_distance,
        "post_refine_max_relative_distance": args.post_refine_max_relative_distance,
        **_overlap_params(args),
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
    _register(args.catalog, args.out)
    return 0


def run_randomized_build(args: argparse.Namespace) -> int:
    windows = load_windows(args.gpl_dir, args.n_windows, args.window_offset)
    detections = load_detections(args.gpl_dir)
    result = _run_build(
        args,
        windows,
        detections,
        epsilon=args.epsilon,
        min_template_width=args.min_template_width,
        max_template_width=args.max_template_width,
        rng=args.seed,
    )
    params = {
        "n_detections": len(detections),
        "epsilon": args.epsilon,
        "min_template_width": args.min_template_width,
        "max_template_width": args.max_template_width,
        "seed": args.seed,
    }
    return _write_build(args, "randomized-build", windows, result, params)


def _lengths_s(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def run_pan_build(args: argparse.Namespace) -> int:
    windows = load_windows(args.gpl_dir, args.n_windows, args.window_offset)
    sec_per_bin = _sec_per_bin(windows[0])
    if sec_per_bin is None:
        raise CliError("--lengths-s needs sample_freq in the window sidecars")
    if not args.lengths_s or min(args.lengths_s) <= 0:
        raise CliError(f"--lengths-s must be positive seconds, got {args.lengths_s}")
    columns = sorted({int(round(seconds / sec_per_bin)) for seconds in args.lengths_s})
    try:
        config = PanSeedConfig(
            lengths=tuple(columns),
            start_freq=windows[0].start_freq,
            end_freq=windows[0].end_freq,
            signal=args.signal,
            pca_rows=args.pca_rows,
            band_edges=tuple(args.band_edges) if args.band_edges else None,
            decimate=args.decimate,
            copy_threshold=args.copy_threshold,
            min_copies=args.min_copies,
            dedupe_corr=args.dedupe_corr,
            nthreads=args.pan_threads,
        )
    except ValueError as exc:
        raise CliError(str(exc)) from exc
    seeder = PanSeeder(config)
    result = _run_build(args, windows, None, seeder=seeder, merge_corr=args.dedupe_corr)
    (args.out / "pan_seeds.jsonl").write_text(
        "".join(json.dumps(asdict(log)) + "\n" for log in seeder.passes)
    )
    params = {
        "seeder": {
            "signal": args.signal,
            "pca_rows": args.pca_rows if args.signal == "pca" else None,
            "band_edges": args.band_edges if args.signal == "bands" else None,
            "lengths_s": args.lengths_s,
            "lengths_columns": columns,
            "decimate": args.decimate,
            "copy_threshold": args.copy_threshold,
            "min_copies": args.min_copies,
            "dedupe_corr": args.dedupe_corr,
        },
    }
    return _write_build(args, "pan-build", windows, result, params)


# ── matching-pursuit ─────────────────────────────────────────────────────────


def load_dictionary(dictionary_dir: Path) -> Dictionary:
    """Atoms listed in a ``randomized-build`` or ``pan-build`` ``summary.json``."""
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
                overlap=_overlap(args),
            )
        except ValueError as exc:
            raise CliError(str(exc)) from exc

    result = _run_with_stats(args.loop_stats, args.out, run)
    loop = result.loop
    _write_json(args.out / "matchings.json", _matching_rows(loop.matchings))
    summary = {
        "command": "matching-pursuit",
        "gpl_dir": str(args.gpl_dir.resolve()),
        "dictionary": str(args.dictionary.resolve()),
        "n_windows": len(windows),
        "window_ids": [window.window_id for window in windows],
        "window_offset": args.window_offset,
        "n_atoms": len(dictionary),
        "n_matchings": len(loop.matchings),
        **_overlap_params(args),
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
    _register(args.catalog, args.out)
    return 0


# ── parser ───────────────────────────────────────────────────────────────────


# ── split ────────────────────────────────────────────────────────────────────


def _read_matchings(folder: Path) -> list[Matching]:
    path = folder / "matchings.json"
    if not path.is_file():
        raise CliError(f"no matchings.json in {folder}")
    return [
        Matching(
            template_id=int(row["template_id"]),
            window_id=str(row["window_id"]),
            t_start=int(row["t_start"]),
            amplitude=float(row["amplitude"]),
        )
        for row in json.loads(path.read_text())
    ]


def run_split(args: argparse.Namespace) -> int:
    """Replace one atom by the two halves of its patches; write a new dictionary folder."""
    dictionary = load_dictionary(args.dictionary)
    source = json.loads((args.dictionary / "summary.json").read_text())
    placements_dir = args.matchings or args.dictionary
    if args.matchings is not None:
        mp_summary = args.matchings / "summary.json"
        recorded = json.loads(mp_summary.read_text()).get("dictionary") if mp_summary.is_file() else None
        if recorded is None or Path(recorded).resolve() != args.dictionary.resolve():
            raise CliError(f"{args.matchings} is not a matching-pursuit run of {args.dictionary}")
    matchings = _read_matchings(placements_dir)
    by_id = {int(t.template_id): t for t in dictionary}
    target = by_id.get(int(args.template))
    if target is None:
        raise CliError(f"template {args.template} not in {args.dictionary}; it has {sorted(by_id)}")

    all_windows = {w.window_id: w for w in load_windows(args.gpl_dir, None, 0)}
    window_ids = list(source.get("window_ids") or sorted({m.window_id for m in matchings}))
    missing = sorted({m.window_id for m in matchings} - set(all_windows))
    if missing:
        raise CliError(f"{len(missing)} placement windows are not in {args.gpl_dir}, e.g. {missing[0]}")
    windows = [all_windows[wid] for wid in window_ids if wid in all_windows]
    if any(t.n_freq != windows[0].n_freq for t in dictionary):
        raise CliError(f"the atoms and the windows of {args.gpl_dir} have different frequency rows")
    own = [m for m in matchings if int(m.template_id) == int(target.template_id)]
    if not own:
        raise CliError(f"template {args.template} has no placements in {placements_dir}")

    try:
        result, side_a, side_b = split_template_matchings(
            windows, target, own,
            epsilon=args.epsilon, n_trials=args.n_trials, min_relative_reduction=args.min_relative_reduction,
            min_size=args.min_size, max_refine=args.max_refine, rng=args.seed,
        )
    except ValueError as exc:
        raise CliError(str(exc)) from exc

    args.out.mkdir(parents=True, exist_ok=True)
    id_a = int(target.template_id)
    id_b = max(by_id) + 1
    report = {
        "command": "split",
        "gpl_dir": str(args.gpl_dir.resolve()),
        "dictionary": str(args.dictionary.resolve()),
        "matchings_from": str(placements_dir.resolve()),
        "split_template": id_a,
        "accepted": bool(result.accepted),
        "relative_reduction": result.relative_reduction,
        "sse_one": result.sse_one,
        "sse_two": result.sse_two,
        "n_a": result.n_a,
        "n_b": result.n_b,
        "best_trial": result.best_trial,
        "epsilon": args.epsilon,
        "n_trials": args.n_trials,
        "max_refine": args.max_refine,
        "min_relative_reduction": args.min_relative_reduction,
        "min_size": args.min_size,
        "seed": args.seed,
        "n_windows": len(windows),
        "window_ids": [w.window_id for w in windows],
        "window_offset": source.get("window_offset", 0),
        "sec_per_bin": _sec_per_bin(windows[0]),
    }
    if not result.accepted:
        _write_json(args.out / "summary.json", {**report, "new_template_ids": [], "atoms": []})
        print(
            f"split rejected: relative reduction {result.relative_reduction:.3f} is below "
            f"--min-relative-reduction {args.min_relative_reduction}",
            file=sys.stderr,
        )
        return 1

    new = Dictionary()
    for template in dictionary:
        if int(template.template_id) != id_a:
            new.add(template)
    parent = target.source_segment_id or f"template_{id_a}"
    new.add(Template(template_id=id_a, data=result.template_a, source_segment_id=f"split:{parent}:a"))
    new.add(Template(template_id=id_b, data=result.template_b, source_segment_id=f"split:{parent}:b"))
    placed = [m for m in matchings if int(m.template_id) != id_a]
    placed += [Matching(template_id=id_a, window_id=m.window_id, t_start=m.t_start, amplitude=m.amplitude) for m in side_a]
    placed += [Matching(template_id=id_b, window_id=m.window_id, t_start=m.t_start, amplitude=m.amplitude) for m in side_b]
    placed.sort(key=lambda m: (m.window_id, m.t_start, m.template_id))

    atoms_dir = args.out / "atoms"
    atoms_dir.mkdir(exist_ok=True)
    counts: dict[int, int] = {}
    for matching in placed:
        counts[int(matching.template_id)] = counts.get(int(matching.template_id), 0) + 1
    atoms = []
    for template in sorted(new, key=lambda t: int(t.template_id)):
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
    _write_json(args.out / "matchings.json", _matching_rows(placed))
    stack = as_stack(windows)
    _write_json(
        args.out / "summary.json",
        {
            **report,
            "new_template_ids": [id_a, id_b],
            "n_atoms": len(atoms),
            "n_matchings": len(placed),
            **_coverage(stack, new, placed),
            "atoms": atoms,
        },
    )
    print(json.dumps({"out": str(args.out), "split_template": id_a, "new_template_ids": [id_a, id_b],
                      "n_a": result.n_a, "n_b": result.n_b, "relative_reduction": result.relative_reduction}))
    _register(args.catalog, args.out)
    return 0


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
    parser.add_argument(
        "--catalog", type=Path, default=None,
        help="Also register the output folder in this catalog SQLite file",
    )


def _add_outer_loop_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--m-per-outer", type=int, default=rd.DEFAULT_OUTER_M)
    parser.add_argument("--min-coverage", type=float, default=rd.DEFAULT_OUTER_MIN_COVERAGE)
    parser.add_argument("--max-outer", type=int, default=rd.DEFAULT_OUTER_MAX_OUTER)
    parser.add_argument("--refine-iters", type=int, default=rd.DEFAULT_OUTER_REFINE_ITERS)
    parser.add_argument("--n-components", type=int, default=rd.defaults.N_COMPONENTS)
    parser.add_argument("--n-min", type=int, default=rd.DEFAULT_OUTER_N_MIN)
    parser.add_argument(
        "--post-refine-max-relative-distance",
        type=_optional_float,
        default=rd.DEFAULT_OUTER_POST_REFINE_TAU,
    )
    parser.add_argument("--verbose", action="store_true", help="Print one line per outer pass")


def _add_mp_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--eps-step-fraction", type=float, default=rd.defaults.EPS_STEP_FRACTION)
    parser.add_argument(
        "--max-relative-distance",
        type=_optional_float,
        default=rd.defaults.MAX_RELATIVE_DISTANCE,
        help="Reject placements leaving more than this energy fraction unexplained ('none' = off)",
    )
    parser.add_argument("--mp-mode", choices=("global", "per_window"), default=None)
    parser.add_argument(
        "--overlap", choices=mp_module.OVERLAP_MODES, default="free",
        help="free: placements may overlap freely. limited: see --max-partial-overlap and --max-nested-ratio",
    )
    parser.add_argument(
        "--max-partial-overlap", type=float, default=mp_module.DEFAULT_MAX_PARTIAL_OVERLAP,
        help="With --overlap limited: two placements may share at most this fraction of the narrower width",
    )
    parser.add_argument(
        "--max-nested-ratio", type=float, default=mp_module.DEFAULT_MAX_NESTED_RATIO,
        help="With --overlap limited: a placement may sit wholly inside another at most this fraction of its "
        "width (a unit in a phrase); 0 allows no nesting",
    )
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
    _add_outer_loop_args(build)
    build.add_argument("--epsilon", type=float, default=rd.DEFAULT_CREATE_EPSILON)
    build.add_argument(
        "--min-template-width", type=_optional_int, default=rd.DEFAULT_MIN_TEMPLATE_WIDTH
    )
    build.add_argument(
        "--max-template-width", type=_optional_int, default=rd.DEFAULT_MAX_TEMPLATE_WIDTH
    )
    build.add_argument("--seed", type=int, default=None, help="Random seed")
    _add_mp_args(build)
    build.set_defaults(func=run_randomized_build)

    pan = sub.add_parser(
        "pan-build",
        help="Learn a dictionary: pan matrix profile seeds, MP + PCA refine, coverage-gated outer loop",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_input_args(pan)
    _add_outer_loop_args(pan)
    pan.add_argument("--signal", choices=GPL_KINDS, default=DEFAULT_SIGNAL, help="Search signal")
    pan.add_argument(
        "--pca-rows", type=int, default=DEFAULT_PCA_ROWS, help="--signal pca: frequency patterns kept"
    )
    pan.add_argument(
        "--band-edges", type=_lengths_s, default=None,
        help="--signal bands: edges in Hz, e.g. 10,100,200 (default: 100 Hz bands)",
    )
    pan.add_argument(
        "--lengths-s", type=_lengths_s, default=[1.0, 2.0, 3.0, 4.0],
        help="Motif lengths in seconds, comma-separated",
    )
    pan.add_argument(
        "--decimate", type=int, default=1, help="Average every N STFT columns for the search signal"
    )
    pan.add_argument(
        "--copy-threshold", type=float, default=DEFAULT_COPY_THRESHOLD,
        help="A copy is closer than this fraction of the way from the pair's distance to the median",
    )
    pan.add_argument(
        "--min-copies", type=int, default=DEFAULT_MIN_COPIES, help="Drop a pair with fewer copies"
    )
    pan.add_argument(
        "--dedupe-corr", type=float, default=DEFAULT_DEDUPE_CORR,
        help="Drop a seed this similar (best shifted cosine) to one already taken, and fold "
        "together atoms of one pass that refine to this similarity",
    )
    pan.add_argument("--pan-threads", type=int, default=1, help="Matrix-profile threads")
    _add_mp_args(pan)
    pan.set_defaults(func=run_pan_build)

    pursue = sub.add_parser(
        "matching-pursuit",
        help="Place a saved dictionary on GPL windows with Mallat MP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_input_args(pursue)
    pursue.add_argument(
        "--dictionary", type=Path, required=True, help="Output folder of randomized-build or pan-build"
    )
    pursue.add_argument("--max-repeats", type=int, default=None)
    pursue.add_argument("--disallow-overlap", action="store_true")
    _add_mp_args(pursue)
    pursue.set_defaults(func=run_matching_pursuit)

    split = sub.add_parser(
        "split",
        help="Replace one atom by two: split its patches into two rank-1 clusters",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    split.add_argument("--gpl-dir", type=Path, required=True, help="GPL output folder the dictionary was built on")
    split.add_argument("--dictionary", type=Path, required=True, help="Dictionary folder (a build or an earlier split)")
    split.add_argument("--template", type=int, required=True, help="template_id of the atom to split")
    split.add_argument("--out", type=Path, required=True, help="Output folder for the new dictionary")
    split.add_argument(
        "--matchings", type=Path, default=None,
        help="Take the atom's patches from this matching-pursuit run of the dictionary "
        "(default: the dictionary's own matchings.json)",
    )
    split.add_argument("--epsilon", type=float, default=0.1, help="Size of the ± perturbation, relative to the atom")
    split.add_argument("--n-trials", type=int, default=10, help="Random ± starts; the best split is kept")
    split.add_argument("--max-refine", type=int, default=10, help="Assign-and-refit rounds per start")
    split.add_argument(
        "--min-relative-reduction", type=float, default=0.1,
        help="Accept only when two atoms cut the residual error by this fraction",
    )
    split.add_argument("--min-size", type=int, default=2, help="Fewest patches on each side")
    split.add_argument("--seed", type=int, default=None, help="Random seed")
    split.add_argument(
        "--catalog", type=Path, default=None,
        help="Also register the output folder in this catalog SQLite file",
    )
    split.set_defaults(func=run_split)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except CliError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

