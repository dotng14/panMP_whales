"""Single-stage dictionary extend: peel → add atoms until coverage target.

Primary API: :func:`extend_dictionary_on_windows` (in-memory) and
:func:`extend_dictionary_for_wav` (catalog I/O). The target WAV / windows may
differ from the WAV used to build the input dictionary.
"""

from __future__ import annotations

import argparse
import json
from contextlib import closing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from database import catalog
from database.migrate_flat_dict_dirs import utc_run_stamp

from ..core.defaults import (
    EPS_STEP_FRACTION,
    INCREMENTAL_MAX_TEMPLATE_WIDTH,
    INCREMENTAL_MIN_TEMPLATE_WIDTH,
    MAX_RELATIVE_DISTANCE,
    MAX_REPEATS,
    MIN_COVERAGE,
    MIN_REPEATS,
    N_COMPONENTS,
    NORMALIZE,
    SEED_WINDOW_NEIGHBORHOOD,
    SEED_WINDOW_POLICY,
    TEMPLATE_METHOD,
)
from ..core.objects import Dictionary, Matching, Segment, Template, Window
from ..core.utils import coverage_from_matchings, mean_center
from ..io.io_gpl import Detection, load_detections_csv, load_windows_from_paths
from ..mp.mp_step import refit_and_deflate, residual_energy
from .build_dictionary import register_averaged_result
from .learn_dictionary import (
    LearnedDictionary,
    NoMatchingPatchesError,
    learn_averaged_templates_from_gpl,
    refine_dictionary_by_averaging,
)
from .seed_from_gpl import Seed


def _emit_progress(
    callback: Callable[[dict[str, Any]], None] | None, event: dict[str, Any]
) -> None:
    """Best-effort progress notification; a broken callback never fails a run."""
    if callback is None:
        return
    try:
        callback(event)
    except Exception:  # noqa: BLE001 - progress is advisory only
        pass


@dataclass
class ExtendDictionaryResult:
    """In-memory result of one extend stage."""

    dictionary: Dictionary
    matchings: list[Matching]
    residual: dict[str, np.ndarray]
    energies: list[float]
    coverage: float
    n_atoms_added: int
    stop_reason: str
    stage_logs: list[dict[str, Any]] = field(default_factory=list)

    def as_learned_dictionary(self) -> LearnedDictionary:
        energy_before = float(self.energies[0]) if self.energies else 0.0
        energy_after = float(self.energies[-1]) if self.energies else energy_before
        seeds = [
            Seed(
                template=Template(
                    template_id=t.template_id,
                    data=np.array(t.data, dtype=np.float64, copy=True),
                    source_segment_id=t.source_segment_id,
                    variances=(
                        None
                        if t.variances is None
                        else np.array(t.variances, dtype=np.float64, copy=True)
                    ),
                ),
                matchings=[m for m in self.matchings if m.template_id == t.template_id],
                residual_after=self.residual,
                energy_before=energy_before,
                energy_after=energy_after,
            )
            for t in self.dictionary
        ]
        return LearnedDictionary(
            seeds=seeds,
            energy_before=energy_before,
            energy_after=energy_after,
            residual_after=self.residual,
            stop_reason=self.stop_reason,
            stage_logs=list(self.stage_logs),
        )


def _occupied_from_matchings(
    dictionary: Dictionary,
    matchings: Sequence[Matching],
) -> dict[str, list[tuple[int, int]]]:
    widths = {t.template_id: t.width for t in dictionary}
    occupied: dict[str, list[tuple[int, int]]] = {}
    for matching in matchings:
        width = widths.get(matching.template_id)
        if width is None:
            continue
        occupied.setdefault(matching.window_id, []).append(
            (int(matching.t_start), int(matching.t_start) + int(width))
        )
    for wid in occupied:
        occupied[wid].sort()
    return occupied


def _time_coverage(
    dictionary: Dictionary,
    matchings: Sequence[Matching],
    residual: dict[str, np.ndarray],
) -> float:
    widths = {t.template_id: int(t.width) for t in dictionary}
    _, _, n_bins_covered, n_bins_total = coverage_from_matchings(
        residual,
        matchings,
        template_widths=widths,
    )
    if n_bins_total <= 0:
        return 0.0
    return float(n_bins_covered) / float(n_bins_total)


def load_dictionary_from_catalog(conn: Any, dictionary_id: int) -> Dictionary:
    """Load rank-1 templates for ``dictionary_id`` into a :class:`Dictionary`."""
    rows = catalog.list_dictionary_templates(conn, int(dictionary_id))
    if not rows:
        raise ValueError(f"dictionary_id={dictionary_id} has no templates")
    dictionary = Dictionary()
    for row in rows:
        path = Path(str(row["npy_path"]))
        if not path.is_file():
            raise FileNotFoundError(f"Missing template file: {path}")
        data = np.load(path)
        dictionary.add(
            Template(
                template_id=int(row["template_id"]),
                data=np.asarray(data, dtype=np.float64),
                source_segment_id=row.get("source_segment_id"),
            )
        )
    return dictionary


def load_dictionary_from_dir(out_dir: Path | str) -> Dictionary:
    """Load templates from a stamped extend/build run directory (``manifest.json``).

    Supports rank-1 (``template_*.npy``) and subspace (``subspace_*.npy``) atoms.
    Used to chain ``n_components > 1`` stages that skip catalog registration.
    """
    root = Path(out_dir).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"No manifest.json under {root}")
    manifest = json.loads(manifest_path.read_text())
    templates = manifest.get("templates") or []
    if not templates:
        raise ValueError(f"manifest at {manifest_path} has no templates")
    dictionary = Dictionary()
    for row in templates:
        file_name = row.get("file")
        if not file_name:
            raise ValueError(f"template row missing file: {row!r}")
        path = root / str(file_name)
        if not path.is_file():
            # Older layouts may store absolute npy_path in the row.
            alt = row.get("npy_path")
            path = Path(str(alt)) if alt else path
        if not path.is_file():
            raise FileNotFoundError(f"Missing template file: {path}")
        data = np.load(path)
        variances = row.get("variances")
        dictionary.add(
            Template(
                template_id=int(row["template_id"]),
                data=np.asarray(data, dtype=np.float64),
                source_segment_id=row.get("source_segment_id"),
                variances=(
                    None
                    if variances is None
                    else np.asarray(variances, dtype=np.float64)
                ),
            )
        )
    return dictionary


def _drop_unmatched_atoms(
    dictionary: Dictionary, matchings: Sequence[Matching]
) -> tuple[Dictionary, list[int]]:
    """Drop atoms that own no matching; return the kept dictionary and dropped ids.

    An atom can end a stage with nothing assigned to it — the peel or the final
    MP gives its spans to a better atom under ``disallow_overlap``. Such an atom
    is dead weight: it shows up in the reviewer with no patches, and it also
    *breaks* the next refine, because ``average_templates`` has nothing to
    average for it and raises :class:`NoMatchingPatchesError` for the whole
    dictionary. Left in place they accumulate, since every later extend inherits
    them and skips its own refine.
    """
    matched = {int(m.template_id) for m in matchings}
    kept = Dictionary()
    dropped: list[int] = []
    for template in dictionary:
        if int(template.template_id) in matched:
            kept.add(template)
        else:
            dropped.append(int(template.template_id))
    return (dictionary, []) if not dropped else (kept, dropped)


def extend_dictionary_on_windows(
    stack: Sequence[Window],
    detections: Sequence[Detection] | Sequence[Segment],
    *,
    dictionary: Dictionary | None = None,
    min_coverage: float = MIN_COVERAGE,
    max_new_atoms: int | None = None,
    eps_step_fraction: float = EPS_STEP_FRACTION,
    normalize: bool = NORMALIZE,
    min_repeats: int = MIN_REPEATS,
    max_repeats: int | None = MAX_REPEATS,
    min_template_width: int | None = INCREMENTAL_MIN_TEMPLATE_WIDTH,
    max_template_width: int | None = INCREMENTAL_MAX_TEMPLATE_WIDTH,
    n_components: int = N_COMPONENTS,
    max_relative_distance: float | None = MAX_RELATIVE_DISTANCE,
    refine_iters: int = 1,
    pinned_matchings: Sequence[Matching] | None = None,
    seed_window_policy: str = SEED_WINDOW_POLICY,
    seed_window_neighborhood: int = SEED_WINDOW_NEIGHBORHOOD,
    disallow_overlap: bool = True,
    show_progress: bool | None = None,
    mp_mode: str | None = None,
    n_workers: int | None = None,
    max_candidates: int | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> ExtendDictionaryResult:
    """Peel existing dictionary, then grow atoms on a deflating residual.

    Growth matches :func:`~.learn_dictionary.learn_averaged_templates_from_gpl`
    (one outer round per new atom). Optional ``refine_iters`` runs a single
    multi-atom average→MP polish on the original stack **after** growth, same
    schedule as :func:`~.build_dictionary.build_dictionary_for_wav`.

    ``dictionary`` may be ``None`` / empty (cold start). Target windows need not
    come from the same WAV as the parent dictionary.

    ``pinned_matchings`` (typically hand labels on the parent dictionary) keep
    their template and position through the peel, the growth rounds, and the
    end-of-stage refine, so relabeled patches are neither reassigned nor lost.

    ``max_candidates`` caps how many GPL candidates each growth round trials
    (highest residual energy first); it is the main knob on round wall time.
    """
    if not stack:
        raise ValueError("stack must contain at least one window")
    if not (0.0 < float(min_coverage) <= 1.0):
        raise ValueError(f"min_coverage must be in (0, 1], got {min_coverage}")
    if max_new_atoms is not None and int(max_new_atoms) < 0:
        raise ValueError(f"max_new_atoms must be >= 0, got {max_new_atoms}")
    if refine_iters < 0:
        raise ValueError(f"refine_iters must be >= 0, got {refine_iters}")
    if max_repeats is not None and int(max_repeats) < 1:
        raise ValueError(f"max_repeats must be >= 1 when set, got {max_repeats}")
    if max_repeats is not None and int(max_repeats) < int(min_repeats):
        raise ValueError(
            f"max_repeats ({max_repeats}) must be >= min_repeats ({min_repeats})"
        )

    from ..tools import matching_pursuit

    original = {w.window_id: mean_center(w.data) for w in stack}
    current = Dictionary()
    if dictionary is not None:
        for t in dictionary:
            current.add(
                Template(
                    template_id=t.template_id,
                    data=np.array(t.data, dtype=np.float64, copy=True),
                    source_segment_id=t.source_segment_id,
                    variances=(
                        None
                        if t.variances is None
                        else np.array(t.variances, dtype=np.float64, copy=True)
                    ),
                )
            )

    energies: list[float] = [residual_energy(original)]
    stage_logs: list[dict[str, Any]] = []
    n_atoms_added = 0
    next_template_id = max((t.template_id for t in current), default=-1) + 1

    pinned = list(pinned_matchings or [])
    if len(current) > 0:
        peel_stack = original
        pinned_now: list[Matching] = []
        reserved: dict[str, list[tuple[int, int]]] = {}
        if pinned:
            peel_stack = {k: np.array(v, copy=True) for k, v in original.items()}
            pinned_now, reserved = refit_and_deflate(
                peel_stack, current, pinned, normalize=normalize
            )
        peel = matching_pursuit(
            peel_stack,
            current,
            eps_step_fraction=eps_step_fraction,
            max_repeats=max_repeats,
            normalize=normalize,
            disallow_overlap=disallow_overlap,
            occupied_intervals=reserved or None,
            show_progress=show_progress,
            mp_mode=mp_mode,
            n_workers=n_workers,
            max_relative_distance=max_relative_distance,
        )
        residual = {
            k: np.array(v, copy=True) for k, v in peel.loop.residual_stack.items()
        }
        matchings = [*pinned_now, *peel.loop.matchings]
        energies.append(residual_energy(residual))
    else:
        residual = {k: np.array(v, copy=True) for k, v in original.items()}
        matchings = []

    coverage = _time_coverage(current, matchings, residual)
    stop_reason = "min_coverage"

    max_add = None if max_new_atoms is None else int(max_new_atoms)
    _emit_progress(
        progress_callback,
        {
            "event": "extend_start",
            "coverage": coverage,
            "target": max_add,
            "min_coverage": float(min_coverage),
            "n_atoms": len(current),
        },
    )
    # An unbounded run (no --max-atoms) has to be able to stop on its own: each
    # round costs a full seed+average trial per GPL candidate, so a coverage
    # target the recording can never reach would grow atoms forever. Give up
    # after this many consecutive atoms that buy no coverage.
    stall_limit = 2
    stalled = 0
    while coverage < float(min_coverage):
        if max_add is not None and n_atoms_added >= max_add:
            stop_reason = "max_new_atoms"
            break

        _emit_progress(
            progress_callback,
            {
                "event": "grow_start",
                "n_atoms_added": n_atoms_added,
                "coverage": coverage,
                "target": max_add,
                "min_coverage": float(min_coverage),
            },
        )
        coverage_before = coverage
        occupied = _occupied_from_matchings(current, matchings)
        grown = learn_averaged_templates_from_gpl(
            residual,
            detections,
            windows_for_units=stack,
            n_rounds=1,
            eps_step_fraction=eps_step_fraction,
            normalize=normalize,
            min_repeats=min_repeats,
            max_repeats=max_repeats,
            disallow_overlap=disallow_overlap,
            min_template_width=min_template_width,
            max_template_width=max_template_width,
            template_method=TEMPLATE_METHOD,
            n_components=n_components,
            max_relative_distance=max_relative_distance,
            occupied_intervals=occupied,
            template_id_start=next_template_id,
            seed_window_policy=seed_window_policy,
            seed_window_neighborhood=seed_window_neighborhood,
            show_progress=show_progress,
            mp_mode=mp_mode,
            n_workers=n_workers,
            max_candidates=max_candidates,
        )
        grown_templates = grown.get_templates()
        if not grown_templates:
            stop_reason = grown.get_stop_reason() or "below_min_repeats"
            stage_logs.append(
                {
                    "event": "grow_empty",
                    "coverage": coverage,
                    "n_atoms": len(current),
                    "stop_reason": stop_reason,
                }
            )
            break

        new_atom = grown_templates[0]
        current.add(
            Template(
                template_id=new_atom.template_id,
                data=np.array(new_atom.data, dtype=np.float64, copy=True),
                source_segment_id=new_atom.source_segment_id,
                variances=(
                    None
                    if new_atom.variances is None
                    else np.array(new_atom.variances, dtype=np.float64, copy=True)
                ),
            )
        )
        next_template_id = max(next_template_id, new_atom.template_id + 1)
        n_atoms_added += 1

        # Same residual-deflation schedule as build_dictionary / learn_averaged:
        # keep grow matchings and peel residual; do not re-MP on the original yet.
        grown_matchings = list(grown.get_matchings())
        matchings = list(matchings) + grown_matchings
        grown_residual = grown.get_residual_after()
        if grown_residual is None:
            stop_reason = "missing_residual"
            stage_logs.append(
                {
                    "event": "grow_empty",
                    "coverage": coverage,
                    "n_atoms": len(current),
                    "stop_reason": stop_reason,
                }
            )
            break
        residual = {k: np.array(v, copy=True) for k, v in grown_residual.items()}
        energies.append(residual_energy(residual))

        coverage = _time_coverage(current, matchings, residual)
        if max_add is None:
            stalled = 0 if coverage > coverage_before + 1e-9 else stalled + 1
        stage_logs.append(
            {
                "event": "atom_added",
                "template_id": int(new_atom.template_id),
                "coverage": coverage,
                "n_atoms": len(current),
                "n_atoms_added": n_atoms_added,
            }
        )
        _emit_progress(
            progress_callback,
            {
                "event": "atom_added",
                "n_atoms_added": n_atoms_added,
                "coverage": coverage,
                "n_atoms": len(current),
                "target": max_add,
                "min_coverage": float(min_coverage),
            },
        )

        if stalled >= stall_limit:
            stop_reason = "coverage_stalled"
            stage_logs.append(
                {
                    "event": "coverage_stalled",
                    "coverage": coverage,
                    "n_atoms": len(current),
                    "n_atoms_added": n_atoms_added,
                }
            )
            break

    if coverage >= float(min_coverage):
        stop_reason = "min_coverage"

    # Drop atoms the peel/growth left with nothing before refining: an unmatched
    # atom makes refine_dictionary_by_averaging raise for the whole dictionary,
    # which is what used to skip the refine entirely and let dead atoms pile up
    # across successive extends.
    current, dropped = _drop_unmatched_atoms(current, matchings)
    if dropped:
        stage_logs.append(
            {
                "event": "dropped_unmatched_atoms",
                "template_ids": dropped,
                "n_dropped": len(dropped),
                "n_atoms": len(current),
                "stage": "before_refine",
            }
        )

    # Single end-of-stage refine (same schedule as build_dictionary).
    if refine_iters > 0 and len(current) > 0 and matchings:
        _emit_progress(
            progress_callback,
            {"event": "refine_start", "refine_iters": int(refine_iters)},
        )
        try:
            refined = refine_dictionary_by_averaging(
                original,
                current,
                matchings,
                iter=refine_iters,
                eps_step_fraction=eps_step_fraction,
                max_repeats=max_repeats,
                normalize=normalize,
                disallow_overlap=disallow_overlap,
                template_method=TEMPLATE_METHOD,
                n_components=n_components,
                max_relative_distance=max_relative_distance,
                pinned_matchings=pinned or None,
                show_progress=show_progress,
                mp_mode=mp_mode,
                n_workers=n_workers,
            )
            current = refined.dictionary
            matchings = list(refined.matchings)
            residual = {
                k: np.array(v, copy=True) for k, v in refined.residual_after.items()
            }
            if refined.energies:
                energies.extend(refined.energies[1:])
            else:
                energies.append(residual_energy(residual))
            coverage = _time_coverage(current, matchings, residual)
            stage_logs.append(
                {
                    "event": "refine_done",
                    "refine_iters": int(refine_iters),
                    "coverage": coverage,
                    "n_atoms": len(current),
                    "n_matchings": len(matchings),
                }
            )
            _emit_progress(
                progress_callback, {"event": "refine_done", "coverage": coverage}
            )
        except NoMatchingPatchesError as exc:
            stage_logs.append(
                {
                    "event": "refine_skipped",
                    "reason": str(exc),
                    "n_atoms": len(current),
                }
            )

    # The refine runs its own MP, which can starve an atom in turn.
    current, dropped = _drop_unmatched_atoms(current, matchings)
    if dropped:
        stage_logs.append(
            {
                "event": "dropped_unmatched_atoms",
                "template_ids": dropped,
                "n_dropped": len(dropped),
                "n_atoms": len(current),
                "stage": "after_refine",
            }
        )

    return ExtendDictionaryResult(
        dictionary=current,
        matchings=matchings,
        residual=residual,
        energies=energies,
        coverage=coverage,
        n_atoms_added=n_atoms_added,
        stop_reason=stop_reason,
        stage_logs=stage_logs,
    )


def extend_dictionary_for_wav(
    db_path: Path | str,
    wav: str,
    *,
    input_dictionary_id: int | None = None,
    input_dictionary_dir: Path | str | None = None,
    window_offset: int = 0,
    n_windows: int | None = None,
    window_ids: Sequence[str] | None = None,
    out_dir: Path | str | None = None,
    name: str | None = None,
    label: str | None = None,
    segmentation_id: int | None = None,
    min_coverage: float = MIN_COVERAGE,
    max_new_atoms: int | None = None,
    eps_step_fraction: float = EPS_STEP_FRACTION,
    normalize: bool = NORMALIZE,
    min_repeats: int = MIN_REPEATS,
    max_repeats: int | None = MAX_REPEATS,
    min_template_width: int | None = INCREMENTAL_MIN_TEMPLATE_WIDTH,
    max_template_width: int | None = INCREMENTAL_MAX_TEMPLATE_WIDTH,
    n_components: int = N_COMPONENTS,
    max_relative_distance: float | None = MAX_RELATIVE_DISTANCE,
    refine_iters: int = 1,
    pinned_matchings: Sequence[Matching] | None = None,
    seed_window_policy: str = SEED_WINDOW_POLICY,
    seed_window_neighborhood: int = SEED_WINDOW_NEIGHBORHOOD,
    show_progress: bool | None = None,
    mp_mode: str | None = None,
    n_workers: int | None = None,
    max_candidates: int | None = None,
    register_catalog: bool = True,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Load parent dict + target WAV windows, extend, register with lineage."""
    if window_offset < 0:
        raise ValueError(f"window_offset must be >= 0, got {window_offset}")
    if n_windows is not None and n_windows < 1:
        raise ValueError(f"n_windows must be >= 1 when set, got {n_windows}")
    if refine_iters < 0:
        raise ValueError(f"refine_iters must be >= 0, got {refine_iters}")
    if min_repeats < 1:
        raise ValueError(f"min_repeats must be >= 1, got {min_repeats}")
    if max_repeats is not None and int(max_repeats) < 1:
        raise ValueError(f"max_repeats must be >= 1 when set, got {max_repeats}")
    if max_repeats is not None and int(max_repeats) < int(min_repeats):
        raise ValueError(
            f"max_repeats ({max_repeats}) must be >= min_repeats ({min_repeats})"
        )
    if eps_step_fraction <= 0.0:
        raise ValueError(f"eps_step_fraction must be > 0, got {eps_step_fraction}")
    if n_components < 1:
        raise ValueError(f"n_components must be >= 1, got {n_components}")

    run_label = label or name

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

        # When filtering by explicit ids, load the full segmentation then subset.
        if window_ids is not None:
            window_rows = catalog.list_windows_for_segmentation(
                conn,
                seg["segmentation_id"],
                limit=None,
                offset=0,
            )
        else:
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
            raise ValueError(
                f"segmentation {seg['segmentation_id']} has no detection CSV"
            )

        windows = load_windows_from_paths([r["npy_path"] for r in window_rows])
        if window_ids is not None:
            wanted = {str(w) for w in window_ids}
            windows = [w for w in windows if w.window_id in wanted]
            if not windows:
                raise ValueError("no windows left after applying --window-ids filter")

        detections = load_detections_csv(Path(seg["csv_path"]))
        if input_dictionary_id is not None and input_dictionary_dir is not None:
            raise ValueError(
                "pass only one of input_dictionary_id or input_dictionary_dir"
            )
        parent: Dictionary | None = None
        if input_dictionary_dir is not None:
            parent = load_dictionary_from_dir(input_dictionary_dir)
        elif input_dictionary_id is not None:
            parent = load_dictionary_from_catalog(conn, int(input_dictionary_id))

        used_ids = [w.window_id for w in windows]
        run_name = run_label or (
            f"{wav_row['stem']}_extend"
            + (f"_from{input_dictionary_id}" if input_dictionary_id is not None else "")
            + (
                f"_fromdir"
                if input_dictionary_id is None and input_dictionary_dir is not None
                else ""
            )
            + f"_w{len(windows)}"
        )
        base_out = (
            Path(out_dir).resolve()
            if out_dir
            else Path.cwd() / "dictionaries" / run_name
        )
        stamp = utc_run_stamp()
        out = base_out / stamp
        out.mkdir(parents=True, exist_ok=False)

        params: dict[str, Any] = {
            "wav": wav_row["abs_path"],
            "input_dictionary_id": input_dictionary_id,
            "input_dictionary_dir": (
                str(Path(input_dictionary_dir).resolve())
                if input_dictionary_dir is not None
                else None
            ),
            "segmentation_id": seg["segmentation_id"],
            "window_ids": used_ids,
            "window_offset": int(window_offset),
            "n_windows": len(used_ids),
            "min_coverage": float(min_coverage),
            "max_new_atoms": max_new_atoms,
            "eps_step_fraction": float(eps_step_fraction),
            "normalize": bool(normalize),
            "min_repeats": int(min_repeats),
            "max_repeats": max_repeats,
            "min_template_width": min_template_width,
            "max_template_width": max_template_width,
            "n_components": int(n_components),
            "max_relative_distance": max_relative_distance,
            "refine_iters": int(refine_iters),
            "n_pinned_matchings": len(list(pinned_matchings or [])),
            "seed_window_policy": seed_window_policy,
            "seed_window_neighborhood": int(seed_window_neighborhood),
            "max_candidates": max_candidates,
            "learner": "extend_dictionary",
            "base_out_dir": str(base_out),
            "run_stamp": stamp,
        }

        learning_run_id = catalog.start_learning_run(
            conn,
            label=run_name,
            out_dir=out,
            params=params,
            n_windows=len(windows),
            n_rounds_requested=max_new_atoms,
            n_units=None,
            eps_step_fraction=eps_step_fraction,
            max_atoms=max_new_atoms,
        )
        tool_run_id = catalog.insert_tool_run(
            conn,
            tool="extend_dictionary",
            input_dictionary_id=input_dictionary_id,
            segmentation_id=seg["segmentation_id"],
            learning_run_id=learning_run_id,
            params=params,
            out_dir=out,
        )
        conn.commit()

        try:
            result = extend_dictionary_on_windows(
                windows,
                detections,
                dictionary=parent,
                min_coverage=min_coverage,
                max_new_atoms=max_new_atoms,
                eps_step_fraction=eps_step_fraction,
                normalize=normalize,
                min_repeats=min_repeats,
                max_repeats=max_repeats,
                min_template_width=min_template_width,
                max_template_width=max_template_width,
                n_components=n_components,
                max_relative_distance=max_relative_distance,
                refine_iters=refine_iters,
                pinned_matchings=pinned_matchings,
                seed_window_policy=seed_window_policy,
                seed_window_neighborhood=seed_window_neighborhood,
                show_progress=show_progress,
                mp_mode=mp_mode,
                n_workers=n_workers,
                max_candidates=max_candidates,
                progress_callback=progress_callback,
            )

            if len(result.dictionary) == 0:
                catalog.finish_tool_run(
                    conn,
                    tool_run_id,
                    stop_reason=result.stop_reason or "no_templates",
                    energy_before=result.energies[0] if result.energies else None,
                    energy_after=result.energies[-1] if result.energies else None,
                )
                catalog.finish_learning_run(
                    conn,
                    learning_run_id,
                    status="failed",
                    stop_reason=result.stop_reason or "no_templates",
                    error_message="extend selected no templates",
                    energy_before=result.energies[0] if result.energies else None,
                    energy_after=result.energies[-1] if result.energies else None,
                    n_rounds_completed=result.n_atoms_added,
                )
                conn.commit()
                raise RuntimeError(
                    "extend selected no templates (coverage loop could not grow?)"
                )

            source = (
                "refinement"
                if input_dictionary_id is not None
                else "seed"
            )
            # Catalog lineage only when parent is a catalog id (subspace dirs skip DB).
            parent_id = (
                int(input_dictionary_id) if input_dictionary_id is not None else None
            )
            params_out = {
                **params,
                "coverage": result.coverage,
                "n_atoms_added": result.n_atoms_added,
                "stage_logs": result.stage_logs,
            }
            out_run_id, dictionary_id = register_averaged_result(
                conn,
                result=result.as_learned_dictionary(),
                out_dir=out,
                label=run_name,
                segmentation_id=seg["segmentation_id"],
                window_ids=used_ids,
                params=params_out,
                name=run_name,
                run_id=learning_run_id,
                register_catalog=register_catalog and n_components <= 1,
                source=source,
                parent_dictionary_id=parent_id,
            )
            catalog.finish_tool_run(
                conn,
                tool_run_id,
                output_dictionary_id=dictionary_id,
                energy_before=result.energies[0] if result.energies else None,
                energy_after=result.energies[-1] if result.energies else None,
                stop_reason=result.stop_reason,
                out_dir=out,
                params=params_out,
            )
            conn.commit()
        except Exception as exc:
            row = conn.execute(
                "SELECT status FROM learning_runs WHERE run_id = ?",
                (learning_run_id,),
            ).fetchone()
            if row is not None and row["status"] == "running":
                catalog.finish_tool_run(
                    conn,
                    tool_run_id,
                    stop_reason="exception",
                )
                catalog.finish_learning_run(
                    conn,
                    learning_run_id,
                    status="failed",
                    stop_reason="exception",
                    error_message=str(exc),
                )
                conn.commit()
            raise

        return {
            "run_id": out_run_id if out_run_id is not None else learning_run_id,
            "dictionary_id": dictionary_id,
            "tool_run_id": tool_run_id,
            "coverage": result.coverage,
            "n_atoms": len(result.dictionary),
            "n_atoms_added": result.n_atoms_added,
            "stop_reason": result.stop_reason,
            "window_ids": used_ids,
            "segmentation_id": seg["segmentation_id"],
            "input_dictionary_id": input_dictionary_id,
            "parent_dictionary_id": parent_id,
            "source": source,
            "stage_logs": result.stage_logs,
            "out_dir": str(out),
            "params": params_out,
        }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Extend a catalog dictionary on arbitrary target-WAV windows until "
            "time-bin coverage reaches --min-coverage (default 0.9)."
        )
    )
    p.add_argument("--db", type=Path, required=True, help="Path to catalog.sqlite")
    p.add_argument(
        "--wav",
        required=True,
        help="Target WAV (windows/detections); may differ from input dictionary WAV",
    )
    p.add_argument(
        "--input-dictionary-id",
        type=int,
        default=None,
        help="Optional parent dictionary_id (any prior WAV)",
    )
    p.add_argument(
        "--input-dictionary-dir",
        type=Path,
        default=None,
        help=(
            "Optional parent stamped run dir (manifest.json + template/subspace "
            "*.npy). Use to chain n_components>1 stages that skip the catalog."
        ),
    )
    p.add_argument("--window-offset", type=int, default=0)
    p.add_argument("--n-windows", type=int, default=None)
    p.add_argument(
        "--window-ids",
        default=None,
        help="Comma-separated window ids (subset of target WAV)",
    )
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--name", default=None, help="Dictionary / run name")
    p.add_argument("--label", default=None, help="Alias for --name")
    p.add_argument("--segmentation-id", type=int, default=None)
    p.add_argument("--min-coverage", type=float, default=MIN_COVERAGE)
    p.add_argument("--max-new-atoms", type=int, default=None)
    p.add_argument("--eps-step-fraction", type=float, default=EPS_STEP_FRACTION)
    p.add_argument("--min-repeats", type=int, default=MIN_REPEATS)
    p.add_argument(
        "--max-repeats",
        type=int,
        default=MAX_REPEATS,
        help=(
            "Cap accepted Mallat matchings per MP run (per window in "
            "per_window mode). Default: no cap."
        ),
    )
    p.add_argument(
        "--min-template-width",
        type=int,
        default=INCREMENTAL_MIN_TEMPLATE_WIDTH,
    )
    p.add_argument(
        "--max-template-width",
        type=int,
        default=INCREMENTAL_MAX_TEMPLATE_WIDTH,
    )
    p.add_argument("--n-components", type=int, default=N_COMPONENTS)
    p.add_argument("--max-relative-distance", type=float, default=None)
    p.add_argument("--refine-iters", type=int, default=1)
    p.add_argument(
        "--pin-matchings",
        type=Path,
        default=None,
        help=(
            "matchings.json whose activations keep their template and position "
            "through peel, growth, and refine (e.g. hand labels from a review "
            "rebuild)"
        ),
    )
    p.add_argument("--seed-window-policy", default=SEED_WINDOW_POLICY)
    p.add_argument(
        "--seed-window-neighborhood",
        type=int,
        default=SEED_WINDOW_NEIGHBORHOOD,
    )
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--mp-mode", default=None)
    p.add_argument("--n-workers", type=int, default=None)
    p.add_argument("--no-register-catalog", action="store_true")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    window_ids = None
    if args.window_ids:
        window_ids = [s.strip() for s in str(args.window_ids).split(",") if s.strip()]
    max_rel = args.max_relative_distance
    if max_rel is None:
        max_rel = MAX_RELATIVE_DISTANCE

    pinned = None
    if args.pin_matchings is not None:
        from .match_to_dictionary import load_matchings_json

        pinned = load_matchings_json(Path(args.pin_matchings))

    summary = extend_dictionary_for_wav(
        args.db,
        args.wav,
        input_dictionary_id=args.input_dictionary_id,
        input_dictionary_dir=args.input_dictionary_dir,
        window_offset=args.window_offset,
        n_windows=args.n_windows,
        window_ids=window_ids,
        out_dir=args.out_dir,
        name=args.name,
        label=args.label,
        segmentation_id=args.segmentation_id,
        min_coverage=args.min_coverage,
        max_new_atoms=args.max_new_atoms,
        eps_step_fraction=args.eps_step_fraction,
        normalize=not args.no_normalize,
        min_repeats=args.min_repeats,
        max_repeats=args.max_repeats,
        min_template_width=args.min_template_width,
        max_template_width=args.max_template_width,
        n_components=args.n_components,
        max_relative_distance=max_rel,
        refine_iters=args.refine_iters,
        pinned_matchings=pinned,
        seed_window_policy=args.seed_window_policy,
        seed_window_neighborhood=args.seed_window_neighborhood,
        show_progress=not args.no_progress,
        mp_mode=args.mp_mode,
        n_workers=args.n_workers,
        register_catalog=not args.no_register_catalog,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
