"""Build a dictionary from labeled matchings, then optionally refine.

Given matchings grouped by ``template_id`` and explicit template widths,
bootstrap placeholder atoms, rebuild them from the matched patches
(:func:`~analysis.dict_learning.code.tools.average_templates`), and optionally
run :func:`~.learn_dictionary.refine_dictionary_by_averaging` (average→MP)
on the same windows.

Negative / non-matching patches are not used yet; see
``documentation/Docs/negative-patches-in-template-estimation.md``.

CLI::

    python -m analysis.dict_learning.code.learn.match_to_dictionary \\
      --npy-dir ... --matchings matchings.json --width 21 --refine-iters 2 --out OUT
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from database.migrate_flat_dict_dirs import utc_run_stamp

from ..core import EPS_STEP_FRACTION, TEMPLATE_METHOD
from ..core import Dictionary, Matching, Template, Window
from ..io import discover_windows, load_window
from ..mp import residual_energy
from ..tools import average_templates
from .learn_dictionary import DictionaryRefinementResult, refine_dictionary_by_averaging
from .seed_from_gpl import mean_center


def _as_stack(
    windows: Sequence[Window] | dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if isinstance(windows, dict):
        stack = {k: np.asarray(v, dtype=np.float64) for k, v in windows.items()}
    else:
        stack = {w.window_id: mean_center(w.data) for w in windows}
    if not stack:
        raise ValueError("windows must contain at least one window")
    return stack


def _n_freq_from_stack(stack: dict[str, np.ndarray]) -> int:
    first = next(iter(stack.values()))
    if first.ndim != 2:
        raise ValueError(f"window spectrograms must be 2-D (F, T), got shape {first.shape}")
    return int(first.shape[0])


def _resolve_widths(
    template_ids: Sequence[int],
    widths: int | Mapping[int, int],
) -> dict[int, int]:
    if isinstance(widths, int):
        if widths < 1:
            raise ValueError(f"widths must be >= 1, got {widths}")
        return {int(tid): int(widths) for tid in template_ids}
    resolved: dict[int, int] = {}
    for tid in template_ids:
        if tid not in widths:
            raise ValueError(
                f"missing width for template_id={tid}; "
                f"provide it in widths or pass a shared int"
            )
        w = int(widths[tid])
        if w < 1:
            raise ValueError(f"width for template_id={tid} must be >= 1, got {w}")
        resolved[int(tid)] = w
    return resolved


def _placeholder_dictionary(
    template_ids: Sequence[int],
    *,
    widths: Mapping[int, int],
    n_freq: int,
) -> Dictionary:
    dictionary = Dictionary()
    for tid in template_ids:
        dictionary.add(
            Template(
                template_id=int(tid),
                data=np.zeros((n_freq, int(widths[tid])), dtype=np.float64),
            )
        )
    return dictionary


def match_to_dictionary(
    windows: Sequence[Window] | dict[str, np.ndarray],
    matchings: Sequence[Matching],
    *,
    widths: int | Mapping[int, int],
    refine_iters: int = 1,
    template_method: str = TEMPLATE_METHOD,
    n_components: int = 1,
    max_relative_distance: float | None = None,
    eps_step_fraction: float = EPS_STEP_FRACTION,
    normalize: bool = True,
    disallow_overlap: bool = True,
    max_repeats: int | None = None,
    pin_labels: bool = True,
    show_progress: bool | None = None,
    mp_mode: str | None = None,
    n_workers: int | None = None,
) -> DictionaryRefinementResult:
    """Build a dictionary from labeled matchings, then optionally refine.

    Steps:

    1. Resolve per-``template_id`` widths (shared ``int`` or mapping).
    2. Build placeholder atoms (zeros) with ``n_freq`` from the windows.
    3. Rebuild each atom from its matched patches via ``average_templates``
       (``max_relative_distance`` is **not** applied here — placeholders are
       zeros, so every patch would look infinitely far away).
    4. If ``refine_iters > 0``, run ``refine_dictionary_by_averaging`` on the
       same windows (``max_relative_distance`` applies during refine).

    Future work may incorporate known-negative patches into the template
    rebuild; they are ignored in this version.

    Parameters
    ----------
    windows
        Target spectrogram windows (list of :class:`Window` or ``{id: array}``).
    matchings
        Labeled activations; grouped by ``template_id`` into atoms.
    widths
        Shared template width, or ``{template_id: width}``.
    refine_iters
        Number of average→MP refine iterations. ``0`` skips refine and returns
        the initial rebuild with the input matchings.
    pin_labels
        Carry ``matchings`` through every refine iteration instead of letting
        each unsupervised MP pass reassign them. Keeps hand labels on the atom
        they were given to and stops a labeled atom from losing all its patches.
        Only meaningful when ``refine_iters > 0``.
    """
    if refine_iters < 0:
        raise ValueError(f"refine_iters must be >= 0, got {refine_iters}")
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")
    items = list(matchings)
    if not items:
        raise ValueError("matchings must be non-empty")

    by_tid: dict[int, list[Matching]] = {}
    for m in items:
        by_tid.setdefault(int(m.template_id), []).append(m)
    template_ids = sorted(by_tid)
    for tid in template_ids:
        if not by_tid[tid]:
            raise ValueError(f"template_id={tid} has no matchings")

    stack = _as_stack(windows)
    n_freq = _n_freq_from_stack(stack)
    width_map = _resolve_widths(template_ids, widths)
    placeholders = _placeholder_dictionary(
        template_ids, widths=width_map, n_freq=n_freq
    )

    # First rebuild: no distance gate (reference atoms are zeros).
    initial = average_templates(
        stack,
        placeholders,
        items,
        method=template_method,
        n_components=n_components,
        max_relative_distance=None,
    )
    dictionary = initial.dictionary
    energy0 = residual_energy(stack)

    if refine_iters == 0:
        residual = {k: np.array(v, copy=True) for k, v in stack.items()}
        return DictionaryRefinementResult(
            dictionary=dictionary,
            matchings=list(items),
            residual_after=residual,
            energies=[energy0],
            iterations_completed=0,
        )

    return refine_dictionary_by_averaging(
        stack,
        dictionary,
        items,
        iter=refine_iters,
        eps_step_fraction=eps_step_fraction,
        max_repeats=max_repeats,
        normalize=normalize,
        disallow_overlap=disallow_overlap,
        template_method=template_method,
        n_components=n_components,
        max_relative_distance=max_relative_distance,
        pinned_matchings=items if pin_labels else None,
        show_progress=show_progress,
        mp_mode=mp_mode,
        n_workers=n_workers,
    )


def write_match_to_dictionary_result(
    result: DictionaryRefinementResult,
    out_dir: Path | str,
    *,
    name: str | None = None,
    params: dict[str, Any] | None = None,
) -> Path:
    """Write templates, matchings, and manifest under a stamped subdirectory."""
    base = Path(out_dir)
    base.mkdir(parents=True, exist_ok=True)
    stamp = utc_run_stamp()
    out = base / stamp
    out.mkdir(parents=True, exist_ok=True)

    n_components = max((t.n_components for t in result.dictionary), default=1)
    kind = "subspace" if n_components > 1 else "rank1"
    template_rows: list[dict[str, Any]] = []
    for template in result.dictionary:
        prefix = "subspace" if template.n_components > 1 else "template"
        npy_path = out / f"{prefix}_{template.template_id:03d}.npy"
        np.save(npy_path, template.data)
        row: dict[str, Any] = {
            "template_id": template.template_id,
            "file": npy_path.name,
            "shape": list(template.data.shape),
            "n_components": template.n_components,
            "n_freq": template.n_freq,
            "width": template.width,
            "source_segment_id": template.source_segment_id,
        }
        if template.variances is not None:
            row["variances"] = np.asarray(template.variances, dtype=np.float64).tolist()
        template_rows.append(row)

    matching_rows: list[dict[str, Any]] = []
    for m in result.matchings:
        row: dict[str, Any] = {
            "template_id": m.template_id,
            "window_id": m.window_id,
            "t_start": m.t_start,
            "amplitude": float(m.amplitude),
        }
        coeffs = np.asarray(m.coeffs, dtype=np.float64).reshape(-1)
        if coeffs.size > 1 or n_components > 1:
            row["coeffs"] = coeffs.tolist()
        matching_rows.append(row)

    energy_before = result.energies[0] if result.energies else None
    energy_after = result.energies[-1] if result.energies else None
    manifest = {
        "name": name,
        "kind": kind,
        "n_components": n_components,
        "learner": "match_to_dictionary",
        "params": dict(params or {}),
        "energy_before": energy_before,
        "energy_after": energy_after,
        "iterations_completed": result.iterations_completed,
        "templates": template_rows,
        "matchings": matching_rows,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (out / "matchings.json").write_text(
        json.dumps({"matchings": matching_rows}, indent=2)
    )
    return out


def load_matchings_json(path: Path) -> list[Matching]:
    """Read a ``matchings.json`` (or bare list) into :class:`Matching` objects."""
    payload = json.loads(path.read_text())
    raw = payload.get("matchings", payload)
    if not isinstance(raw, list):
        raise ValueError(f"matchings JSON must be a list or {{'matchings': [...]}}, got {path}")
    out: list[Matching] = []
    for m in raw:
        coeffs = m.get("coeffs")
        if coeffs is not None:
            out.append(
                Matching(
                    template_id=int(m["template_id"]),
                    window_id=str(m["window_id"]),
                    t_start=int(m["t_start"]),
                    coeffs=np.asarray(coeffs, dtype=np.float64),
                )
            )
        else:
            out.append(
                Matching(
                    template_id=int(m["template_id"]),
                    window_id=str(m["window_id"]),
                    t_start=int(m["t_start"]),
                    amplitude=float(m.get("amplitude", 1.0)),
                )
            )
    return out


def _parse_widths(args: argparse.Namespace) -> int | dict[int, int]:
    if args.widths is not None and args.width is not None:
        raise SystemExit("pass only one of --width or --widths")
    if args.widths is not None:
        raw = json.loads(args.widths) if isinstance(args.widths, str) else args.widths
        if not isinstance(raw, dict):
            raise SystemExit("--widths must be a JSON object {template_id: width}")
        return {int(k): int(v) for k, v in raw.items()}
    if args.width is not None:
        return int(args.width)
    raise SystemExit("one of --width or --widths is required")


def _load_windows_from_args(args: argparse.Namespace) -> list[Window]:
    if args.npy_dir:
        return discover_windows(Path(args.npy_dir))
    if args.window:
        return [load_window(Path(p)) for p in args.window]
    return []


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m analysis.dict_learning.code.learn.match_to_dictionary",
        description=(
            "Build a dictionary from labeled matchings (+ explicit widths), "
            "then optionally refine with average→MP on the given windows."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--window", action="append", default=[], help="Path to analysis .npy")
    p.add_argument("--npy-dir", type=str, default=None, help="Directory of window .npy files")
    p.add_argument(
        "--matchings",
        type=str,
        required=True,
        help="JSON with matchings list (same shape as average-templates)",
    )
    p.add_argument("--width", type=int, default=None, help="Shared template width")
    p.add_argument(
        "--widths",
        type=str,
        default=None,
        help='Per-template widths JSON, e.g. \'{"0":21,"1":20}\'',
    )
    p.add_argument("--refine-iters", type=int, default=1)
    p.add_argument("--template-method", default=TEMPLATE_METHOD, choices=("pca", "average"))
    p.add_argument("--n-components", type=int, default=1)
    p.add_argument(
        "--max-relative-distance",
        type=float,
        default=None,
        help=(
            "Reject patches leaving more than this fraction of their energy "
            "unexplained by the atom, in MP and in template rebuilds (0-1)"
        ),
    )
    p.add_argument("--eps-step-fraction", type=float, default=EPS_STEP_FRACTION)
    p.add_argument("--max-repeats", type=int, default=None)
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--allow-overlap", action="store_true", help="Disable disallow_overlap")
    p.add_argument(
        "--no-pin-labels",
        action="store_true",
        help="Let refine MP reassign the input labels instead of keeping them",
    )
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--mp-mode", default=None, choices=("per_window", "global"))
    p.add_argument("--n-workers", type=int, default=None)
    p.add_argument("--out", type=str, default=None, help="Base output directory")
    p.add_argument("--name", type=str, default=None)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    windows = _load_windows_from_args(args)
    if not windows:
        print("No windows loaded (--window or --npy-dir required)", file=sys.stderr)
        return 1
    matchings = load_matchings_json(Path(args.matchings))
    widths = _parse_widths(args)
    params = {
        "refine_iters": args.refine_iters,
        "template_method": args.template_method,
        "n_components": args.n_components,
        "max_relative_distance": args.max_relative_distance,
        "eps_step_fraction": args.eps_step_fraction,
        "max_repeats": args.max_repeats,
        "normalize": not args.no_normalize,
        "disallow_overlap": not args.allow_overlap,
        "pin_labels": not args.no_pin_labels,
        "widths": widths if isinstance(widths, int) else {str(k): v for k, v in widths.items()},
        "n_windows": len(windows),
        "n_input_matchings": len(matchings),
    }
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
        pin_labels=not args.no_pin_labels,
        show_progress=not args.no_progress,
        mp_mode=args.mp_mode,
        n_workers=args.n_workers,
    )
    out_dir = None
    if args.out:
        out_dir = write_match_to_dictionary_result(
            result,
            args.out,
            name=args.name,
            params=params,
        )
    energy_before = result.energies[0] if result.energies else None
    energy_after = result.energies[-1] if result.energies else None
    summary = {
        "n_atoms": len(result.dictionary),
        "n_matchings": len(result.matchings),
        "iterations_completed": result.iterations_completed,
        "energy_before": energy_before,
        "energy_after": energy_after,
        "out_dir": str(out_dir) if out_dir is not None else None,
        "name": args.name,
        "params": params,
    }
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
