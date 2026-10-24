"""Compare standard rank-1 MP against subspace (top-k PCA) MP on real data.

Loads an already-built dictionary (e.g. LeiData wav1) and its GPL windows from
the catalog, runs a standard :func:`.mp_step.run_mp_loop` baseline to obtain both
the reference matchings and the per-template matched patches, builds a top-k PCA
:class:`.subspace_mp.SubspaceAtom` per template, then runs
:func:`.subspace_mp.run_subspace_mp_loop` for several ``k`` and reports residual
energy removed, coverage, placements, ``score_shifts`` call counts (the ~k x
scoring cost), and wall time.

Example::

    python -m analysis.dict_learning.experiments.subspace_mp_compare \\
        --db   /home/yfreund/whale_processing/result/dict_learning_catalog.sqlite \\
        --wav  /home/yfreund/whale_processing/LeiData/SOCAL_H_76_230427_003345.x.wav \\
        --dict-dir /home/yfreund/whale_processing/result/LeiData/dict_learning_templates/wav1_averaged \\
        --n-windows 4 --ks 1 2 3
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import closing
from pathlib import Path
from typing import Any

import numpy as np

from ..code.mp import loop_stats
from ..code.io import load_windows_from_paths
from ..code.mp import residual_energy, run_mp_loop
from ..code.core import Dictionary, Matching, Template
from ..code.learn.seed_from_gpl import mean_center
from ..code.mp import SubspaceAtom, build_subspace_atom, run_subspace_mp_loop


def _load_templates_from_dir(dict_dir: Path) -> list[Template]:
    """Load a built dictionary's templates from its ``manifest.json`` or npy files."""
    dict_dir = Path(dict_dir)
    manifest = dict_dir / "manifest.json"
    templates: list[Template] = []
    if manifest.is_file():
        payload = json.loads(manifest.read_text())
        rows = payload.get("templates", [])
        for row in rows:
            fname = row.get("file") or row.get("npy_path")
            if fname is None:
                continue
            path = Path(fname)
            if not path.is_absolute():
                path = dict_dir / path.name
            if not path.is_file():
                continue
            templates.append(
                Template(
                    template_id=int(row.get("template_id", len(templates))),
                    data=np.load(path),
                    source_segment_id=row.get("source_segment_id"),
                )
            )
    if not templates:
        # Fallback: glob template/pca/avg npy files in a stable order.
        paths = sorted(
            p
            for pat in ("template_*.npy", "pca_*.npy", "avg_*.npy")
            for p in dict_dir.glob(pat)
        )
        for i, path in enumerate(paths):
            templates.append(Template(template_id=i, data=np.load(path)))
    if not templates:
        raise ValueError(f"no templates found under {dict_dir}")
    return templates


def _load_windows(
    db_path: str, wav: str, *, n_windows: int, window_offset: int, segmentation_id: int | None
) -> dict[str, np.ndarray]:
    """Mean-centered residual stack for the first ``n_windows`` GPL windows."""
    from database import catalog

    with closing(catalog.connect(db_path)) as conn:
        wav_row = catalog.find_wav_file(conn, wav)
        if wav_row is None:
            raise ValueError(f"WAV not found in catalog: {wav!r}")
        segmentations = catalog.list_gpl_segmentations_for_wav(conn, wav_row["wav_id"])
        if not segmentations:
            raise ValueError(f"no GPL segmentation for {wav_row['stem']}")
        if segmentation_id is not None:
            seg = next(
                (s for s in segmentations if s["segmentation_id"] == segmentation_id), None
            )
            if seg is None:
                raise ValueError(f"segmentation_id={segmentation_id} not found")
        else:
            seg = segmentations[0]
        window_rows = catalog.list_windows_for_segmentation(
            conn, seg["segmentation_id"], limit=n_windows, offset=window_offset
        )
        if not window_rows:
            raise ValueError("segmentation has no windows at the requested offset/limit")
        windows = load_windows_from_paths([r["npy_path"] for r in window_rows])
    return {w.window_id: mean_center(w.data) for w in windows}


def _coverage_fraction(
    residual: dict[str, np.ndarray],
    matchings,
    widths: dict[int, int],
) -> tuple[float, float]:
    """Time-bin and window coverage fractions using per-template widths."""
    from .utils import coverage_from_matchings

    n_matched, n_win, n_bins, n_bins_total = coverage_from_matchings(
        residual, matchings, template_widths=widths
    )
    time_frac = n_bins / n_bins_total if n_bins_total else 0.0
    win_frac = n_matched / n_win if n_win else 0.0
    return time_frac, win_frac


def _score_shift_calls() -> int:
    return int(loop_stats.snapshot()["counts"].get("score_shifts.calls", 0))


def _k_effective(calls: int, n_atoms: int, n_windows: int, n_placed: int) -> float:
    """Per-scan cost multiplier: score_shifts calls / (atoms x windows x steps).

    Each MP step scans every atom against every window; the loop runs
    ``n_placed + 1`` steps (one extra non-accepted scan). Standard MP does one
    correlation per atom/window (~1); subspace-k does ``k`` (clamped to the
    atom's available patches), so this isolates the ``~k x`` scoring cost from
    differing atom counts and step counts.
    """
    steps = max(1, n_placed + 1)
    denom = n_atoms * n_windows * steps
    return (calls / denom) if denom else 0.0


def run_comparison(
    *,
    db_path: str,
    wav: str,
    dict_dir: str,
    n_windows: int = 4,
    window_offset: int = 0,
    segmentation_id: int | None = None,
    ks: tuple[int, ...] = (1, 2, 3),
    eps_step_fraction: float = 0.3,
    pc1_screen: bool = False,
) -> dict[str, Any]:
    """Run baseline MP + subspace MP for each ``k`` and return a report dict."""
    residual = _load_windows(
        db_path,
        wav,
        n_windows=n_windows,
        window_offset=window_offset,
        segmentation_id=segmentation_id,
    )
    e0 = residual_energy(residual)
    templates = _load_templates_from_dir(Path(dict_dir))
    widths = {t.template_id: t.width for t in templates}
    dictionary = Dictionary()
    for t in templates:
        dictionary.add(t)

    loop_stats.enable(True)

    # --- Baseline: standard rank-1 MP (global). ---
    loop_stats.reset()
    base = run_mp_loop(
        residual,
        dictionary,
        eps_step_fraction=eps_step_fraction,
        normalize=True,
        disallow_overlap=True,
        mp_mode="global",
        show_progress=False,
    )
    base_calls = _score_shift_calls()
    base_report = {
        "k": 1,
        "method": "standard_mp",
        "n_atoms": len(templates),
        "n_placements": len(base.matchings),
        "energy_frac_left": residual_energy(base.residual_stack) / e0 if e0 else 0.0,
        "score_shifts_calls": base_calls,
        # calls / (atoms x windows x steps): the per-scan cost multiplier (~1).
        "k_effective": _k_effective(base_calls, len(templates), len(residual), len(base.matchings)),
        "wall_s": loop_stats.timing_sum("mp.loop_wall_s"),
    }
    tf, wf = _coverage_fraction(residual, base.matchings, widths)
    base_report["time_coverage_fraction"] = tf
    base_report["window_coverage_fraction"] = wf

    # Group baseline matchings by template for the PCA subspace fit.
    base_by_tid: dict[int, list[Matching]] = {}
    for m in base.matchings:
        base_by_tid.setdefault(m.template_id, []).append(m)

    runs: list[dict[str, Any]] = [base_report]
    for k in ks:
        atoms: list[SubspaceAtom] = []
        for t in templates:
            tid_matchings = base_by_tid.get(t.template_id, [])
            if not tid_matchings:
                continue  # template unused by baseline MP; no patches to fit
            atoms.append(
                build_subspace_atom(
                    residual,
                    t.template_id,
                    tid_matchings,
                    width=t.width,
                    n_freq=t.n_freq,
                    n_components=k,
                )
            )
        if not atoms:
            raise ValueError("no subspace atoms could be built (baseline had no matchings)")

        loop_stats.reset()
        res = run_subspace_mp_loop(
            residual,
            atoms,
            eps_step_fraction=eps_step_fraction,
            disallow_overlap=True,
            pc1_screen=pc1_screen,
            show_progress=False,
        )
        tf, wf = _coverage_fraction(residual, res.matchings, widths)
        calls = _score_shift_calls()
        # Mean retained components per atom (clamped to available patches).
        mean_k = float(np.mean([a.n_components for a in atoms])) if atoms else 0.0
        runs.append(
            {
                "k": k,
                "method": "subspace_mp" + ("_pc1screen" if pc1_screen else ""),
                "n_atoms": len(atoms),
                "mean_components": mean_k,
                "n_placements": len(res.matchings),
                "energy_frac_left": residual_energy(res.residual_stack) / e0 if e0 else 0.0,
                "score_shifts_calls": calls,
                "k_effective": _k_effective(calls, len(atoms), len(residual), len(res.matchings)),
                "wall_s": loop_stats.timing_sum("subspace_mp.loop_wall_s"),
                "time_coverage_fraction": tf,
                "window_coverage_fraction": wf,
            }
        )

    return {
        "wav": wav,
        "dict_dir": str(dict_dir),
        "n_windows": len(residual),
        "n_templates": len(templates),
        "energy_before": e0,
        "eps_step_fraction": eps_step_fraction,
        "pc1_screen": pc1_screen,
        "runs": runs,
    }


def _format_table(report: dict[str, Any]) -> str:
    header = (
        f"{'method':<22}{'k':>3}{'atoms':>7}{'placed':>8}"
        f"{'E_left/E0':>11}{'time_cov':>10}{'win_cov':>9}"
        f"{'ss_calls':>10}{'k_eff':>7}{'wall_s':>9}"
    )
    lines = [header, "-" * len(header)]
    for r in report["runs"]:
        lines.append(
            f"{r['method']:<22}{r['k']:>3}{r.get('n_atoms', 0):>7}"
            f"{r['n_placements']:>8}{r['energy_frac_left']:>11.4f}"
            f"{r['time_coverage_fraction']:>10.4f}{r['window_coverage_fraction']:>9.4f}"
            f"{r.get('score_shifts_calls', 0):>10}{r.get('k_effective', 0.0):>7.2f}"
            f"{r['wall_s']:>9.3f}"
        )
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", required=True, help="Catalog SQLite path")
    p.add_argument("--wav", required=True, help="WAV path/stem/filename (indexed)")
    p.add_argument(
        "--dict-dir",
        required=True,
        help="Built dictionary out_dir (manifest.json + template npy files)",
    )
    p.add_argument("--n-windows", type=int, default=4)
    p.add_argument("--window-offset", type=int, default=0)
    p.add_argument("--segmentation-id", type=int, default=None)
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--eps-step-fraction", type=float, default=0.3)
    p.add_argument(
        "--pc1-screen",
        action="store_true",
        help="Locate with PC1 only, evaluate full-k at top candidates (~1x scoring)",
    )
    p.add_argument("--out", default=None, help="Write the report JSON to this path")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    report = run_comparison(
        db_path=args.db,
        wav=args.wav,
        dict_dir=args.dict_dir,
        n_windows=args.n_windows,
        window_offset=args.window_offset,
        segmentation_id=args.segmentation_id,
        ks=tuple(args.ks),
        eps_step_fraction=args.eps_step_fraction,
        pc1_screen=args.pc1_screen,
    )
    print(_format_table(report))
    out = args.out or "subspace_mp_compare.json"
    Path(out).write_text(json.dumps(report, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
