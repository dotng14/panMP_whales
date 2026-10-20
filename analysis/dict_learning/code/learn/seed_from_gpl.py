"""seed_from_GPL: energy-ranked single-template Mallat MP selection.

Mean-center raw window images

Pipeline (one Seed):

  1. Build candidate patches from GPL detections (or use pre-built Segments).
  2. Trial every candidate with an isolated single-template Mallat MP run
     (peeling until ``eps_step_fraction``); discard trials with fewer than
     ``min_repeats`` matchings.
  3. Keep the qualifying candidate whose trial leaves the **lowest** residual
     energy.

This module does **not** average matched patches. For that step see
``learn_dictionary.refine_seed_by_averaging``.

CLI::

    python -m analysis.dict_learning.code.learn.seed_from_gpl \\
        --npy-dir PATH --gpl-output-dir PATH \\
        --eps-step-fraction 0.2 --min-repeats 1 --out OUT_DIR
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from ..io.io_gpl import (
    Detection,
    detection_overlaps_window,
    discover_windows,
    find_detection_csvs,
    global_samples_to_time_bins,
    load_detections_from_paths,
    load_windows_from_paths,
)
from ..mp.mp_step import residual_energy
from ..core.objects import Matching, Segment, Template, Window


# ── helpers ──────────────────────────────────────────────────────────────────

def mean_center(arr: np.ndarray) -> np.ndarray:
    """Subtract the global mean so ``arr`` has zero mean (float64 copy)."""
    a = np.asarray(arr, dtype=np.float64)
    return a - float(np.mean(a))


def _clone_window(window: Window, data: np.ndarray) -> Window:
    """Copy ``window`` metadata with a replacement spectrogram array."""
    return Window(
        window_id=window.window_id,
        data=np.asarray(data, dtype=np.float64),
        start_sample=window.start_sample,
        end_sample=window.end_sample,
        overlap=window.overlap,
        fftl=window.fftl,
        sample_freq=window.sample_freq,
        start_freq=window.start_freq,
        end_freq=window.end_freq,
        npy_path=window.npy_path,
        meta=window.meta,
    )


def _prepare_stack(
    stack: Sequence[Window] | dict[str, np.ndarray],
    windows_for_units: Sequence[Window] | None,
) -> tuple[dict[str, np.ndarray], list[Window]]:
    """Build residual stack (+ unit windows).

    - ``Sequence[Window]`` (raw images): mean-center each window **once**.
    - ``dict`` (already a residual): copy as float64; do **not** re-center.

    Returns ``(residual0, unit_windows)`` where ``unit_windows`` carry the same
    arrays as ``residual0`` for detection cropping.
    """
    if isinstance(stack, dict):
        residual0 = {k: np.asarray(v, dtype=np.float64) for k, v in stack.items()}
        bases = list(windows_for_units or [])
    else:
        residual0 = {w.window_id: mean_center(w.data) for w in stack}
        bases = list(windows_for_units) if windows_for_units is not None else list(stack)

    unit_windows = []
    for w in bases:
        if w.window_id in residual0:
            data = residual0[w.window_id]
        elif isinstance(stack, dict):
            data = np.asarray(w.data, dtype=np.float64)
        else:
            data = mean_center(w.data)
        unit_windows.append(_clone_window(w, data))
    return residual0, unit_windows


def _recrop_segments(
    residual0: dict[str, np.ndarray],
    segments: Sequence[Segment],
) -> list[Segment]:
    """Re-crop Segment patches from the residual stack (no mean-centering)."""
    out: list[Segment] = []
    for seg in segments:
        arr = residual0.get(seg.window_id)
        if arr is None:
            out.append(seg)
            continue
        patch = np.array(arr[:, seg.t_start : seg.t_end], copy=True)
        if patch.size == 0 or patch.shape[1] == 0 or float(np.sum(patch ** 2)) <= 0.0:
            continue
        out.append(
            Segment(
                segment_id=seg.segment_id,
                data=patch,
                window_id=seg.window_id,
                t_start=seg.t_start,
                t_end=seg.t_end,
                global_start_sample=seg.global_start_sample,
                global_end_sample=seg.global_end_sample,
                detection_index=seg.detection_index,
                boundary_cross_flag=seg.boundary_cross_flag,
            )
        )
    return out


# ── Seed result ──────────────────────────────────────────────────────────────

class Seed:
    """Result of one energy-ranked single-template Mallat selection.

    Construct via :meth:`from_gpl` / :meth:`from_paths` (or the module-level
    aliases). Access state through the ``get_*`` methods.
    """

    def __init__(
        self,
        *,
        template: Template | None = None,
        matchings: list[Matching] | None = None,
        residual_after: dict[str, np.ndarray] | None = None,
        energy_before: float = 0.0,
        energy_after: float = 0.0,
        n_candidates_tried: int = 0,
        selected_segment_id: str | None = None,
        candidates: list[Segment] | None = None,
        mp_energies: list[float] | None = None,
    ) -> None:
        self._template = template
        self._matchings = list(matchings or [])
        self._residual_after = residual_after
        self._energy_before = energy_before
        self._energy_after = energy_after
        self._n_candidates_tried = n_candidates_tried
        self._selected_segment_id = selected_segment_id
        self._candidates = list(candidates or [])
        # Stack energy after each MP step of the winning trial (index 0 = before atoms).
        self._mp_energies = list(mp_energies or [])

    # -- constructors ---------------------------------------------------------

    @classmethod
    def from_gpl(
        cls,
        stack: Sequence[Window] | dict[str, np.ndarray],
        detections: Sequence[Detection] | Sequence[Segment],
        *,
        windows_for_units: Sequence[Window] | None = None,
        eps_step_fraction: float = 0.5,
        min_repeats: int = 1,
        template_id: int = 0,
        normalize: bool = True,
        disallow_overlap: bool = False,
        occupied_intervals: dict[str, list[tuple[int, int]]] | None = None,
        show_progress: bool | None = None,
        mp_mode: str | None = None,
        n_workers: int | None = None,
        max_relative_distance: float | None = None,
    ) -> Seed:
        """Select one template by trialing every GPL candidate with Mallat MP.

        Parameters
        ----------
        stack
            Raw ``Window`` sequence (mean-centered once) or a residual
            ``{window_id: array}`` (used as-is; not re-centered).
        detections
            GPL ``Detection`` rows, or pre-built ``Segment`` patches.
        windows_for_units
            Windows used to crop detection patches. Defaults to ``stack`` when
            ``stack`` is a sequence of ``Window``. Required when ``stack`` is a
            bare dict and ``detections`` are ``Detection``s.
        eps_step_fraction
            Mallat stops when a step's ΔE falls below this × the first step's ΔE.
        min_repeats
            Discard trials with fewer than this many matchings.
        template_id
            Id stamped on the winning ``Template``.
        normalize
            Passed through to ``run_mp_loop`` (template / residual normalization).
        disallow_overlap
            Exclude placements that overlap an accepted matching in the same
            trial or an interval supplied through ``occupied_intervals``.
        occupied_intervals
            Existing half-open ``[start, end)`` detections to avoid per window.
        show_progress
            Optional tqdm bars for candidate trials and nested MP loops.
        mp_mode
            ``per_window`` (default) or ``global`` Mallat; see ``run_mp_loop``.
        n_workers
            Parallel candidate trials (``DICT_LEARNING_MP_WORKERS`` / CPU count).
        max_relative_distance
            Reject placements leaving more than this fraction of the patch
            energy unexplained by the candidate (``None`` = no gate).
        """
        from ..mp.mp_parallel import (
            allocate_nested_workers,
            map_parallel,
            resolve_mp_mode,
            resolve_mp_workers,
        )
        from ..mp.mp_workers import (
            CandidateTrialPayload,
            init_trial_worker_shared,
            payload_by_segment_id,
            rematerialize_candidate_trial,
            run_candidate_trial,
            slim_payloads_for_process_pool,
            trial_sort_key_energy_then_segment,
        )
        from ..mp.progress import iter_progress

        if eps_step_fraction <= 0.0:
            raise ValueError(f"eps_step_fraction must be > 0, got {eps_step_fraction}")
        if min_repeats < 1:
            raise ValueError(f"min_repeats must be >= 1, got {min_repeats}")

        # Mean-center raw Windows once; residual dicts are left unchanged.
        residual0, unit_windows = _prepare_stack(stack, windows_for_units)
        e0 = residual_energy(residual0)

        # Build / re-crop candidate patches from the prepared residual.
        if detections and isinstance(detections[0], Segment):
            candidates = _recrop_segments(residual0, detections)  # type: ignore[arg-type]
        else:
            if not unit_windows:
                raise ValueError(
                    "windows_for_units (or Window stack) required with Detection inputs"
                )
            candidates = detections_to_segments(unit_windows, detections)  # type: ignore[arg-type]

        # Trial every candidate (energy-ranked only for progress ordering).
        ordered = sorted(candidates, key=lambda s: s.energy, reverse=True)
        mode = resolve_mp_mode(mp_mode)
        cand_workers = resolve_mp_workers(n_workers)
        outer_workers, inner_workers = allocate_nested_workers(
            len(ordered), total_workers=cand_workers
        )

        payloads = [
            CandidateTrialPayload(
                segment_id=seg.segment_id,
                template_id=template_id,
                template_data=np.array(seg.data, copy=True),
                residual=residual0,
                eps_step_fraction=eps_step_fraction,
                normalize=normalize,
                disallow_overlap=disallow_overlap,
                occupied_intervals=occupied_intervals,
                mp_mode=mode,
                n_workers=inner_workers,
                max_relative_distance=max_relative_distance,
            )
            for seg in ordered
        ]

        # Isolated Mallat trial per unit; keep min residual among those with
        # at least ``min_repeats`` matchings.
        n_tried = len(payloads)
        if outer_workers <= 1 or len(payloads) <= 1:
            trial_results = []
            for payload in iter_progress(
                payloads,
                desc=f"seed trials (t{template_id})",
                total=len(payloads),
                leave=False,
                show_progress=show_progress,
                unit="cand",
            ):
                trial_results.append(run_candidate_trial(payload))
        else:
            # ProcessPool: real multi-core (Numba nogil + Python outside kernels).
            # Residual is shared once per worker via initializer — not pickled per task.
            trial_results = map_parallel(
                run_candidate_trial,
                slim_payloads_for_process_pool(payloads),
                n_workers=outer_workers,
                prefer_threads=False,
                initializer=init_trial_worker_shared,
                initargs=(residual0, occupied_intervals),
            )

        best_tr = None
        best_key = None
        for tr in trial_results:
            if tr.n_matchings < min_repeats:
                continue
            key = trial_sort_key_energy_then_segment(tr)
            if best_tr is None or key < best_key:
                best_tr = tr
                best_key = key

        if best_tr is None:
            return cls(
                energy_before=e0,
                n_candidates_tried=n_tried,
                candidates=candidates,
            )

        # Batch trials omit residual stacks; rematerialize only the winner.
        winner = rematerialize_candidate_trial(
            payload_by_segment_id(payloads, best_tr.segment_id)  # type: ignore[arg-type]
        )
        tmpl = Template(
            template_id=winner.template_id,
            data=np.array(winner.template_data, copy=True),
            source_segment_id=winner.segment_id,
        )
        return cls(
            template=tmpl,
            matchings=winner.matchings,
            residual_after=winner.residual_stack,
            energy_before=e0,
            energy_after=winner.energy_after,
            n_candidates_tried=n_tried,
            selected_segment_id=winner.segment_id,
            candidates=candidates,
            mp_energies=list(winner.energies),
        )

    @classmethod
    def from_paths(
        cls,
        *,
        window_paths: Sequence[Path | str] | None = None,
        npy_dir: Path | str | None = None,
        csv_paths: Sequence[Path | str] | None = None,
        gpl_output_dir: Path | str | None = None,
        eps_step_fraction: float = 0.5,
        min_repeats: int = 1,
        normalize: bool = True,
        show_progress: bool | None = None,
        mp_mode: str | None = None,
        n_workers: int | None = None,
    ) -> Seed:
        """Load windows + detections from disk, then call :meth:`from_gpl`.

        Provide windows via ``window_paths`` or ``npy_dir``, and detections via
        ``csv_paths`` or ``gpl_output_dir`` (looks for ``CSV_*.csv``).
        Raw windows are mean-centered once inside :meth:`from_gpl`.
        """
        if window_paths:
            windows = load_windows_from_paths(window_paths)
        elif npy_dir is not None:
            windows = discover_windows(Path(npy_dir))
        else:
            raise ValueError("Provide window_paths or npy_dir")

        if csv_paths:
            detections = load_detections_from_paths(csv_paths)
        elif gpl_output_dir is not None:
            out = Path(gpl_output_dir)
            csvs = find_detection_csvs(out) or (
                find_detection_csvs(out / "output") if (out / "output").is_dir() else []
            )
            detections = load_detections_from_paths(csvs)
        else:
            raise ValueError("Provide csv_paths or gpl_output_dir")

        return cls.from_gpl(
            windows,
            detections,
            eps_step_fraction=eps_step_fraction,
            min_repeats=min_repeats,
            normalize=normalize,
            show_progress=show_progress,
            mp_mode=mp_mode,
            n_workers=n_workers,
        )

    # -- getters --------------------------------------------------------------

    def is_selected(self) -> bool:
        """True if a winning template was chosen."""
        return self._template is not None

    def get_template(self) -> Template | None:
        return self._template

    def get_matchings(self) -> list[Matching]:
        return list(self._matchings)

    def get_residual_after(self) -> dict[str, np.ndarray] | None:
        if self._residual_after is None:
            return None
        return {k: np.asarray(v) for k, v in self._residual_after.items()}

    def get_energy_before(self) -> float:
        return self._energy_before

    def get_energy_after(self) -> float:
        return self._energy_after

    def get_delta_energy_total(self) -> float:
        """E_before − E_after for the winning trial."""
        return self._energy_before - self._energy_after

    def get_mp_energies(self) -> list[float]:
        """Stack energy after each MP step of the winning trial (0 = before atoms)."""
        return list(self._mp_energies)

    def get_mp_delta_energies(self) -> list[float]:
        """Per-iteration ΔE for the winning MP trial: E_{k-1} − E_k."""
        e = self._mp_energies
        return [e[i] - e[i + 1] for i in range(len(e) - 1)] if len(e) >= 2 else []

    def get_n_candidates_tried(self) -> int:
        return self._n_candidates_tried

    def get_selected_segment_id(self) -> str | None:
        return self._selected_segment_id

    def get_candidates(self) -> list[Segment]:
        """All ranked candidates (not only the shortlist that was trialed)."""
        return list(self._candidates)

    def write(self, out_dir: Path | str) -> Path:
        """Write template + ``seed_manifest.json`` under ``out_dir``."""
        return write_seed_result(self, out_dir)


# ── detection → segment ──────────────────────────────────────────────────────

def detections_to_segments(
    windows: Sequence[Window],
    detections: Sequence[Detection],
) -> list[Segment]:
    """One spectrogram patch per overlapping (window, detection) pair.

    Maps each detection's global sample interval into window-local time bins,
    crops ``window.data[:, t0:t1]``, and skips empty / zero-energy patches.
    """
    segments: list[Segment] = []
    idx = 0
    for window in windows:
        for det in detections:
            if not detection_overlaps_window(det, window):
                continue
            bins = global_samples_to_time_bins(
                det.start_time,
                det.end_time,
                win_start_sample=window.start_sample,
                overlap=window.overlap,
                n_cols=window.n_time,
            )
            if bins is None:
                continue
            t0, t1 = bins
            patch = np.array(window.data[:, t0:t1], copy=True)
            if patch.size == 0 or patch.shape[1] == 0 or float(np.sum(patch ** 2)) <= 0.0:
                continue
            segments.append(
                Segment(
                    segment_id=f"{window.window_id}::det{det.index}::{idx}",
                    data=patch,
                    window_id=window.window_id,
                    t_start=t0,
                    t_end=t1,
                    global_start_sample=det.start_time,
                    global_end_sample=det.end_time,
                    detection_index=det.index,
                    boundary_cross_flag=det.boundary_cross_flag,
                )
            )
            idx += 1
    return segments


# ── public aliases ───────────────────────────────────────────────────────────

def seed_from_gpl(
    stack: Sequence[Window] | dict[str, np.ndarray],
    detections: Sequence[Detection] | Sequence[Segment],
    *,
    windows_for_units: Sequence[Window] | None = None,
    eps_step_fraction: float = 0.5,
    min_repeats: int = 1,
    template_id: int = 0,
    normalize: bool = True,
    show_progress: bool | None = None,
    mp_mode: str | None = None,
    n_workers: int | None = None,
    disallow_overlap: bool = False,
    occupied_intervals: dict[str, list[tuple[int, int]]] | None = None,
) -> Seed:
    """Alias for :meth:`Seed.from_gpl` (same parameters)."""
    return Seed.from_gpl(
        stack,
        detections,
        windows_for_units=windows_for_units,
        eps_step_fraction=eps_step_fraction,
        min_repeats=min_repeats,
        template_id=template_id,
        normalize=normalize,
        show_progress=show_progress,
        mp_mode=mp_mode,
        n_workers=n_workers,
        disallow_overlap=disallow_overlap,
        occupied_intervals=occupied_intervals,
    )


def seed_from_gpl_paths(
    *,
    window_paths: Sequence[Path | str] | None = None,
    npy_dir: Path | str | None = None,
    csv_paths: Sequence[Path | str] | None = None,
    gpl_output_dir: Path | str | None = None,
    eps_step_fraction: float = 0.5,
    min_repeats: int = 1,
    normalize: bool = True,
    show_progress: bool | None = None,
    mp_mode: str | None = None,
    n_workers: int | None = None,
) -> Seed:
    """Alias for :meth:`Seed.from_paths` (same parameters)."""
    return Seed.from_paths(
        window_paths=window_paths,
        npy_dir=npy_dir,
        csv_paths=csv_paths,
        gpl_output_dir=gpl_output_dir,
        eps_step_fraction=eps_step_fraction,
        min_repeats=min_repeats,
        normalize=normalize,
        show_progress=show_progress,
        mp_mode=mp_mode,
        n_workers=n_workers,
    )


# ── I/O ──────────────────────────────────────────────────────────────────────

def write_seed_result(seed: Seed, out_dir: Path | str) -> Path:
    """Write ``template.npy``, optional residuals, and ``seed_manifest.json``."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict = {
        "selected": seed.is_selected(),
        "selected_segment_id": seed.get_selected_segment_id(),
        "delta_energy_total": seed.get_delta_energy_total(),
        "energy_before": seed.get_energy_before(),
        "energy_after": seed.get_energy_after(),
        "n_candidates_tried": seed.get_n_candidates_tried(),
        "n_candidates": len(seed.get_candidates()),
        "n_matchings": len(seed.get_matchings()),
        "matchings": [
            {
                "template_id": m.template_id,
                "window_id": m.window_id,
                "t_start": m.t_start,
                "amplitude": m.amplitude,
            }
            for m in seed.get_matchings()
        ],
    }

    template = seed.get_template()
    if template is not None:
        np.save(out_dir / "template.npy", template.data)
        manifest["template"] = {
            "file": "template.npy",
            "template_id": template.template_id,
            "shape": list(template.data.shape),
            "source_segment_id": template.source_segment_id,
        }
        residual_after = seed.get_residual_after()
        if residual_after is not None:
            resid_dir = out_dir / "residual"
            resid_dir.mkdir(exist_ok=True)
            for wid, arr in residual_after.items():
                np.save(resid_dir / f"{wid.replace('/', '_')}.npy", arr)

    path = out_dir / "seed_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    return path


# ── CLI ──────────────────────────────────────────────────────────────────────

def _build_arg_parser() -> argparse.ArgumentParser:
    """Argparse for ``python -m analysis.dict_learning.code.learn.seed_from_gpl``."""
    p = argparse.ArgumentParser(
        prog="python -m analysis.dict_learning.code.learn.seed_from_gpl",
        description="Select one template via energy-ranked single-template Mallat MP.",
    )
    # Inputs
    p.add_argument("--window", action="append", default=[], help="Path to a window .npy")
    p.add_argument("--npy-dir", type=str, default=None, help="Directory of window .npy files")
    p.add_argument("--csv", action="append", default=[], help="Path to a GPL CSV_*.csv")
    p.add_argument("--gpl-output-dir", type=str, default=None, help="GPL output dir (finds CSV_*.csv)")
    # Algorithm knobs (same names as Seed.from_gpl / seed_from_gpl)
    p.add_argument(
        "--eps-step-fraction",
        type=float,
        default=0.5,
        help="Stop MP when ΔE_step < this × first (largest) ΔE_step",
    )
    p.add_argument(
        "--min-repeats",
        type=int,
        default=1,
        help="Require at least this many matchings for a seed to be selected",
    )
    p.add_argument("--no-normalize", action="store_true", help="Disable MP normalization")
    p.add_argument("--out", type=str, default=None, help="Write template + seed_manifest.json here")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars")
    p.add_argument(
        "--mp-mode",
        choices=("per_window", "global"),
        default=None,
        help="MP mode (default: per_window)",
    )
    p.add_argument(
        "--mp-workers",
        type=int,
        default=None,
        help="Parallel candidate-trial workers (default: CPU count)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    """CLI entry: load paths, run seeding, optionally write results."""
    args = _build_arg_parser().parse_args(argv)
    seed = seed_from_gpl_paths(
        window_paths=args.window or None,
        npy_dir=args.npy_dir,
        csv_paths=args.csv or None,
        gpl_output_dir=args.gpl_output_dir,
        eps_step_fraction=args.eps_step_fraction,
        min_repeats=args.min_repeats,
        normalize=not args.no_normalize,
        show_progress=False if args.no_progress else True,
        mp_mode=args.mp_mode,
        n_workers=args.mp_workers,
    )

    if not seed.is_selected():
        print(
            f"no template selected "
            f"(tried={seed.get_n_candidates_tried()}/{len(seed.get_candidates())} "
            f"energy0={seed.get_energy_before():.6g})"
        )
        return 1

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
        print(f"wrote {write_seed_result(seed, args.out)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
