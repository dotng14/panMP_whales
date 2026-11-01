"""Dictionary-action dispatch for the repeat-review "Actions" form.

Slim port of gplrev3's command service. Only two actions are exposed:

* ``randomized_builder`` -> ``randomized_dictionary.learn_randomized_outer`` (builds a
  fresh dictionary from the job's GPL segmentation)
* ``history`` -> read of the ``action_history`` table

The analysis runs in a child process (:mod:`app.services.terminal_runner`) so the
CPU-bound matching pursuit never starves the web server's event loop. A finished
build is written as a ``randomized-build``-shaped folder (``atoms/``,
``summary.json``, ``matchings.json``, ``outer_logs.jsonl``) and registered with
``catalog.register_run_dir``; the job then switches onto the new dictionary.

Tunable parameters are declared per action as :class:`Flag` specs, so the form
catalog, coercion and the analysis kwargs stay in sync.
"""

from __future__ import annotations

import importlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import OrderedDict
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from app.config import BACKEND_DIR, CATALOG_DB, CATALOG_RUNS_ROOTS

# command name -> (module path, function name) of the analysis entry point it runs.
FUNCTION_TARGETS: dict[str, tuple[str, str]] = {
    "randomized_builder": (
        "analysis.dict_learning.code.learn.randomized_dictionary",
        "learn_randomized_outer",
    ),
}

# Only these actions are exposed (catalog, run_action, history).
ALLOWED_ACTIONS = ("history", "randomized_builder")

@dataclass(frozen=True)
class Flag:
    """One named parameter that maps onto a handler/analysis kwarg."""

    name: str  # form field / catalog name
    kwarg: str  # handler keyword it fills
    kind: str  # "int" | "float" | "str"
    metavar: str  # placeholder shown in the catalog, e.g. "N" or "0-1"
    help: str
    # What the analysis uses when the field is omitted, shown in the form.
    # Human-facing (e.g. "0.9", "none") — mirrors the backing default.
    default: str = ""
    choices: tuple[str, ...] | None = None
    allow_none: bool = False  # "none" (any case) coerces to Python None
    # Pass ``default`` to the analysis call when the caller omits this flag,
    # instead of letting the library default stand. For knobs where the library
    # default is unbounded but the app needs a run that ends on its own.
    apply_default: bool = False
    # Long-form explanation behind the form's "Read more" link; ``help`` stays
    # the one-line version shown under the field. Blank lines separate paragraphs.
    details: str = ""

    def coerce(self, raw: Any) -> Any:
        if raw is None:
            if self.allow_none:
                return None
            raise CommandError(f"{self.name} is required")
        if isinstance(raw, bool):
            raise CommandError(f"invalid value for {self.name}: {raw!r}")
        if isinstance(raw, (int, float)) and self.kind in ("int", "float"):
            if self.kind == "int":
                if isinstance(raw, float) and not raw.is_integer():
                    raise CommandError(f"invalid value for {self.name}: {raw!r}")
                return int(raw)
            return float(raw)
        text = str(raw).strip()
        if self.allow_none and text.lower() == "none":
            return None
        try:
            if self.kind == "int":
                return int(text)
            if self.kind == "float":
                return float(text)
        except ValueError as exc:
            raise CommandError(f"invalid value for {self.name}: {raw!r}") from exc
        if self.choices is not None and text not in self.choices:
            allowed = "|".join(self.choices)
            raise CommandError(f"{self.name} must be one of {allowed}, got {raw!r}")
        return text


@dataclass(frozen=True)
class CommandDef:
    name: str
    summary: str
    positional: str  # usage fragment, e.g. "[template#]" or "[iterations]" or ""
    flags: tuple[Flag, ...]
    handler: "Callable[[CommandContext, dict[str, Any], list[str]], CommandResult]"
    example: str = ""
    analysis: bool = True  # backed by a FUNCTION_TARGETS entry (readiness-checked)
    runnable: bool = True  # False = listed in the form, pipeline not wired yet

    @property
    def usage(self) -> str:
        parts = [self.name]
        if self.positional:
            parts.append(self.positional)
        parts.extend(f"[--{flag.name} {flag.metavar}]" for flag in self.flags)
        return " ".join(parts)


@dataclass
class CommandContext:
    """Everything a handler needs to reconstruct a job's dictionary state."""

    stem: str
    segmentation_id: int | None
    dictionary_id: int | None
    # Template the reviewer is viewing when the terminal opens; used as the
    # default target for template-scoped commands (e.g. bare ``split``).
    current_template: int | None = None
    # Real WAV path behind the job — needed to place persisted output dirs.
    source_wav: str | None = None
    # Set by the subprocess runner to stream live progress out to the parent.
    # Receives normalized ``{"message", "fraction", "elapsed_s", "eta_s", ...}``
    # dicts. Never serialized across the process boundary — the child installs
    # its own emitter after rebuilding the context.
    progress_callback: "Callable[[dict[str, Any]], None] | None" = None


@dataclass
class CommandResult:
    ok: bool
    lines: list[str]
    # Set when a command persisted a new (child) dictionary; the caller switches
    # the job to it so the repeats view refreshes onto the change.
    new_dictionary_id: int | None = None

    @classmethod
    def error(cls, *lines: str) -> "CommandResult":
        return cls(ok=False, lines=list(lines))

    @classmethod
    def output(cls, *lines: str, new_dictionary_id: int | None = None) -> "CommandResult":
        return cls(ok=True, lines=list(lines), new_dictionary_id=new_dictionary_id)


class CommandError(Exception):
    """Raised by handlers/parsing for an expected, user-facing failure."""


def _resolve_function(command: str) -> Callable[..., Any]:
    """Import and return the analysis function a command runs.

    Raises :class:`CommandError` with a readable message if the module or
    function is missing — this is the runtime half of "check the function
    exists" so the terminal never dies on an ``ImportError``/``AttributeError``.
    """
    module_path, function_name = FUNCTION_TARGETS[command]
    try:
        module = importlib.import_module(module_path)
    except Exception as exc:  # pragma: no cover - defensive
        raise CommandError(f"could not import {module_path}: {exc}") from exc
    fn = getattr(module, function_name, None)
    if fn is None or not callable(fn):
        raise CommandError(f"function {function_name}() not found in {module_path}")
    return fn


def function_availability() -> dict[str, bool]:
    """command -> True if its backing analysis function is importable."""
    available: dict[str, bool] = {}
    for command in FUNCTION_TARGETS:
        try:
            _resolve_function(command)
            available[command] = True
        except CommandError:
            available[command] = False
    return available


# --------------------------------------------------------------------------- #
# Catalog / window helpers
# --------------------------------------------------------------------------- #


def _connect_read():
    from database import catalog

    return catalog.connect(CATALOG_DB, read_only=True)


def _require_segmentation(ctx: CommandContext) -> int:
    if ctx.segmentation_id is None:
        raise CommandError("no GPL segmentation is associated with this job yet")
    return int(ctx.segmentation_id)


def output_root_for(source_wav: Path | str | None, stem: str) -> Path:
    """Folder under which one recording's dictionary-action outputs are written.

    Prefers the first ``CATALOG_RUNS_ROOTS`` entry (so ``POST /api/catalog/reindex``
    finds the runs again), else ``backend/tmp/dict_actions``. ``source_wav`` is
    accepted for parity with gplrev3 but not needed here.
    """
    del source_wav
    root = CATALOG_RUNS_ROOTS[0] / "dict_actions" if CATALOG_RUNS_ROOTS else BACKEND_DIR / "tmp" / "dict_actions"
    return root / stem


def _resolve_window_scope(window_limit: int | None) -> tuple[int | None, int]:
    """Turn the optional ``--windows`` override into a (limit, offset).

    ``None`` -> the configured ``N_WINDOWS`` default (matches the CLI). ``0`` or
    negative -> every window. A positive value -> that many windows from offset 0.
    """
    from analysis.dict_learning.code.core import N_WINDOWS

    if window_limit is None:
        return int(N_WINDOWS), 0
    return (None if int(window_limit) <= 0 else int(window_limit)), 0


def _window_rows(conn, segmentation_id: int, limit: int | None, offset: int = 0) -> list[dict]:
    from database import catalog

    rows = [row for row in catalog.list_windows_for_segmentation(conn, segmentation_id) if row["npy_path"]]
    end = None if limit is None else offset + int(limit)
    return rows[offset:end]


def _load_windows(rows: list[dict]):
    from analysis.dict_learning.code.io import load_windows_from_paths

    if not rows:
        raise CommandError("no segmentation windows are available for this recording")
    return load_windows_from_paths([row["npy_path"] for row in rows])


def _stack_bytes(rows: list[dict]) -> int:
    """On-disk size of the window .npy files a run will hold in memory."""
    total = 0
    for row in rows:
        try:
            total += os.path.getsize(row["npy_path"])
        except OSError:
            continue
    return total

# One candidate trial keeps roughly this many copies of the window stack alive
# at once (the seed MP residual plus the averaging re-MP residual, each copied
# internally by run_mp_loop), measured at ~2.2x on a 30-window stack.
_TRIAL_STACK_COPIES = 3.0
# Fraction of currently-claimable memory the candidate pool may take.
_MEMORY_BUDGET_FRACTION = 0.5
# Past ~8 workers the pool is memory-bandwidth bound: 8 and 16 threads measured
# the same trials/second, and 16 doubles the footprint for nothing.
_MAX_MP_WORKERS = 8


def _available_memory_bytes() -> int | None:
    """Memory a new allocation can actually claim, or ``None`` if unknown.

    On Windows the binding limit is the system commit charge rather than free
    RAM — this dev box pairs 32 GB of RAM with a 2 GB pagefile — so take the
    smaller of the two.
    """
    if sys.platform == "win32":
        import ctypes

        class _MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = _MemoryStatusEx()
        status.dwLength = ctypes.sizeof(_MemoryStatusEx)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return int(min(status.ullAvailPhys, status.ullAvailPageFile))

    try:
        import psutil  # optional; absent in slim deployments
    except ImportError:
        return None
    try:
        return int(psutil.virtual_memory().available)
    except Exception:  # noqa: BLE001 - never let a probe break a run
        return None


def _stack_bytes_for_scope(conn, segmentation_id: int, limit: int | None, offset: int) -> int:
    """On-disk size of the window .npy files a run will hold in memory."""
    from database import catalog

    total = 0
    rows = catalog.list_windows_for_segmentation(
        conn, segmentation_id, limit=limit, offset=offset
    )
    for row in rows:
        path = row["npy_path"]
        if not path:
            continue
        try:
            total += os.path.getsize(path)
        except OSError:
            continue
    return total


def _mp_workers_for_stack(stack_bytes: int | None) -> int | None:
    """Parallel-worker count for matching pursuit, capped by free memory.

    Each candidate trial holds its own copies of the residual stack, so the
    pool's peak footprint scales with the worker count: 8 workers on a
    30-window stack measured 5.2 GB. Saturating all 32 cores would ask for
    several times that, and there is no gain past ~8 anyway (8 and 16 threads
    ran the same trials/second), so cap on both counts.

    ``DICT_LEARNING_MP_WORKERS`` still wins: returning ``None`` hands the choice
    back to the analysis library, which reads that env var.
    """
    if os.environ.get("DICT_LEARNING_MP_WORKERS", "").strip():
        return None

    workers = min(_MAX_MP_WORKERS, max(1, os.cpu_count() or 1))
    available = _available_memory_bytes()
    if available and stack_bytes and stack_bytes > 0:
        per_worker = stack_bytes * _TRIAL_STACK_COPIES
        budget = available * _MEMORY_BUDGET_FRACTION
        workers = min(workers, max(1, int(budget // per_worker)))
    return workers


def coerce_params(defn: CommandDef, params: dict[str, Any] | None) -> dict[str, Any]:
    """Coerce a form/JSON param dict onto the action's Flag kwargs."""
    flags_by_name = {flag.name: flag for flag in defn.flags}
    flags_by_kwarg = {flag.kwarg: flag for flag in defn.flags}
    kwargs: dict[str, Any] = {}
    for key, raw in (params or {}).items():
        flag = flags_by_name.get(key) or flags_by_kwarg.get(key)
        if flag is None:
            raise CommandError(f"unknown parameter: {key}")
        kwargs[flag.kwarg] = flag.coerce(raw)
    for flag in defn.flags:
        if flag.apply_default and flag.kwarg not in kwargs:
            kwargs[flag.kwarg] = flag.coerce(flag.default)
    return kwargs



def _fmt_duration(seconds: float | None) -> str:
    """Human, compact wall-clock string (e.g. "45s", "3m 12s")."""
    if seconds is None or seconds < 0:
        return "?"
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"



def _make_randomized_builder_progress(
    ctx: CommandContext,
    *,
    min_coverage: float,
    max_outer: int,
) -> "Callable[[Any], None] | None":
    """Forward outer-pass logs from learn_randomized_outer to the terminal UI."""
    sink = ctx.progress_callback
    if sink is None:
        return None

    start = time.monotonic()

    def on_log(log: Any) -> None:
        elapsed = time.monotonic() - start
        outer = int(getattr(log, "outer", 0) or 0)
        coverage = float(getattr(log, "time_coverage_fraction", 0.0) or 0.0)
        n_atoms = int(getattr(log, "n_atoms", 0) or 0)
        target = float(min_coverage) if min_coverage > 0 else 1.0
        fraction = min(1.0, coverage / target) if target > 0 else None
        # Prefer coverage progress; fall back to outer-count when coverage is flat.
        if (fraction is None or fraction <= 0) and max_outer > 0 and outer > 0:
            fraction = min(1.0, outer / int(max_outer))
        eta = None
        if fraction and fraction > 0:
            eta = elapsed * (1.0 - fraction) / fraction
        message = (
            f"outer {outer}/{max_outer} · atoms {n_atoms} · "
            f"coverage {coverage:.2f} / target {min_coverage:.2f}"
        )
        try:
            sink(
                {
                    "message": message,
                    "fraction": fraction,
                    "elapsed_s": round(elapsed, 1),
                    "eta_s": None if eta is None else round(eta, 1),
                    "elapsed": _fmt_duration(elapsed),
                    "eta": None if eta is None else _fmt_duration(eta),
                }
            )
        except Exception:  # noqa: BLE001 - progress must never break a run
            pass

    return on_log


# --------------------------------------------------------------------------- #
# Persistence (randomized-build-shaped folder + catalog registration)
# --------------------------------------------------------------------------- #


def _matching_rows(matchings) -> list[dict[str, Any]]:
    rows = []
    for m in matchings:
        row: dict[str, Any] = {
            "template_id": int(m.template_id),
            "window_id": m.window_id,
            "t_start": int(m.t_start),
            "amplitude": float(m.amplitude),
            "energy": m.energy,
        }
        if getattr(m, "n_components", 1) > 1:
            row["coeffs"] = [float(c) for c in m.coeffs]
        rows.append(row)
    return rows


def _persist_randomized_build(
    ctx: CommandContext,
    *,
    segmentation: dict[str, Any],
    windows,
    n_detections: int,
    window_offset: int,
    result,
    kwargs: dict[str, Any],
) -> tuple[int, Path]:
    """Write a ``randomized-build`` folder, register it, return (dictionary_id, out_dir).

    Mirrors ``analysis.dict_learning.code.cli.run_randomized_build`` so the
    catalog's ``register_run_dir`` accepts it unchanged. ``gpl_dir`` points at the
    segmentation's output folder so the run attaches to the same segmentation.
    """
    from dataclasses import asdict
    from datetime import datetime, timezone

    import numpy as np

    from database import catalog

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    out_dir = output_root_for(ctx.source_wav, ctx.stem) / f"randomized_builder_{stamp}"
    atoms_dir = out_dir / "atoms"
    atoms_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[int, int] = {}
    for m in result.matchings:
        counts[int(m.template_id)] = counts.get(int(m.template_id), 0) + 1
    atoms = []
    for template in result.dictionary:
        rel = f"atoms/template_{int(template.template_id):03d}.npy"
        np.save(out_dir / rel, template.data)
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

    (out_dir / "matchings.json").write_text(json.dumps(_matching_rows(result.matchings), indent=2))
    (out_dir / "outer_logs.jsonl").write_text(
        "".join(json.dumps(asdict(log)) + "\n" for log in result.outer_logs)
    )

    from analysis.dict_learning.code import core as defaults
    from analysis.dict_learning.code.learn import randomized_dictionary as rd

    def _kw(name: str, default: Any) -> Any:
        return kwargs[name] if name in kwargs else default

    first = windows[0] if windows else None
    sec_per_bin = (
        first.overlap / first.sample_freq if first is not None and first.sample_freq > 0 else None
    )
    gpl_dir = str(Path(segmentation["output_dir"]).resolve())
    summary = {
        "command": "randomized-build",
        "gpl_dir": gpl_dir,
        "wav": gpl_dir,
        "n_windows": len(windows),
        "window_ids": [w.window_id for w in windows],
        "window_offset": window_offset,
        "n_detections": n_detections,
        "sec_per_bin": sec_per_bin,
        "m_per_outer": _kw("m_per_outer", rd.DEFAULT_OUTER_M),
        "min_coverage": _kw("min_coverage", rd.DEFAULT_OUTER_MIN_COVERAGE),
        "max_outer": _kw("max_outer", rd.DEFAULT_OUTER_MAX_OUTER),
        "refine_iters": _kw("refine_iters", rd.DEFAULT_OUTER_REFINE_ITERS),
        "n_components": _kw("n_components", defaults.N_COMPONENTS),
        "n_min": _kw("n_min", rd.DEFAULT_OUTER_N_MIN),
        "epsilon": _kw("epsilon", rd.DEFAULT_CREATE_EPSILON),
        "min_template_width": _kw("min_template_width", rd.DEFAULT_MIN_TEMPLATE_WIDTH),
        "max_template_width": _kw("max_template_width", rd.DEFAULT_MAX_TEMPLATE_WIDTH),
        "eps_step_fraction": _kw("eps_step_fraction", defaults.EPS_STEP_FRACTION),
        "max_relative_distance": _kw("max_relative_distance", None),
        "post_refine_max_relative_distance": _kw(
            "post_refine_max_relative_distance", rd.DEFAULT_OUTER_POST_REFINE_TAU
        ),
        "seed": None,
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
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    with closing(catalog.connect(CATALOG_DB)) as conn:
        registered = catalog.register_run_dir(
            conn, out_dir, label=f"randomized_builder {stamp[:8]}-{stamp[8:14]}"
        )
        conn.commit()
    return int(registered["dictionary_id"]), out_dir


# --------------------------------------------------------------------------- #
# Command handlers
# --------------------------------------------------------------------------- #


def _cmd_randomized_builder(
    ctx: CommandContext, kwargs: dict[str, Any], positionals: list[str]
) -> CommandResult:
    if positionals:
        raise CommandError(f"unexpected argument: {positionals[0]}")
    window_limit = kwargs.pop("__window_limit__", None)

    learn_outer = _resolve_function("randomized_builder")
    seg_id = _require_segmentation(ctx)

    from analysis.dict_learning.code.io import load_detections_csv
    from database import catalog

    with closing(_connect_read()) as conn:
        seg = catalog.get_segmentation(conn, seg_id)
        if seg is None:
            raise CommandError(f"segmentation {seg_id} not found")
        csv_path = seg.get("csv_path")
        if not csv_path or not Path(csv_path).is_file():
            raise CommandError("this segmentation has no detection CSV to build a dictionary from")
        limit, offset = _resolve_window_scope(window_limit)
        rows = _window_rows(conn, seg_id, limit, offset)

    windows = _load_windows(rows)
    stack_bytes = _stack_bytes(rows)

    detections = load_detections_csv(Path(csv_path))
    if not detections:
        raise CommandError("detection CSV is empty - nothing to sample seeds from")

    if "n_workers" not in kwargs:
        workers = _mp_workers_for_stack(stack_bytes)
        if workers is not None:
            kwargs["n_workers"] = workers

    min_coverage = float(kwargs.get("min_coverage", 0.5))
    max_outer = int(kwargs.get("max_outer", 100))
    try:
        result = learn_outer(
            windows,
            detections,
            show_progress=False,
            progress_callback=_make_randomized_builder_progress(
                ctx, min_coverage=min_coverage, max_outer=max_outer
            ),
            **kwargs,
        )
    except ValueError as exc:
        raise CommandError(str(exc)) from exc
    if len(result.dictionary) == 0:
        raise CommandError(f"randomized builder produced no atoms (stop_reason={result.stop_reason})")

    new_id, out_dir = _persist_randomized_build(
        ctx,
        segmentation=seg,
        windows=windows,
        n_detections=len(detections),
        window_offset=offset,
        result=result,
        kwargs=kwargs,
    )
    lines = [
        "randomized_builder: learn_randomized_outer",
        f"  atoms: {result.n_atoms}   matchings: {result.n_matchings}",
        f"  outer passes: {result.n_outer_completed}   stop reason: {result.stop_reason}",
        f"  time coverage: {result.time_coverage_fraction:.3f}   "
        f"window coverage: {result.window_coverage_fraction:.3f}",
    ]
    if result.energy_before is not None and result.energy_after is not None:
        lines.append(
            f"  residual energy: {result.energy_before:.6g} -> {result.energy_after:.6g} "
            f"(delta {result.energy_after - result.energy_before:+.6g})"
        )
    lines.append(f"  output: {out_dir}")
    lines.append(f"  -> saved as new dictionary {new_id} (now active)")
    return CommandResult.output(*lines, new_dictionary_id=new_id)

def _cmd_history(
    ctx: CommandContext, kwargs: dict[str, Any], positionals: list[str]
) -> CommandResult:
    if positionals:
        raise CommandError(f"unexpected argument: {positionals[0]}")
    from database import catalog

    with closing(_connect_read()) as conn:
        history_rows = catalog.list_action_history(
            conn,
            segmentation_id=ctx.segmentation_id,
            stem=ctx.stem,
            limit=50,
        )

    if not history_rows:
        return CommandResult.output(
            f"Action history for '{ctx.stem}':",
            "  (no dictionary actions have been performed on this dataset yet)",
        )

    lines = [
        f"Action history for '{ctx.stem}' (segmentation {ctx.segmentation_id or 'none'}):",
        f"  Total actions recorded: {len(history_rows)}",
        "",
    ]
    for idx, row in enumerate(history_rows, start=1):
        status_tag = f"[{row['status'].upper()}]"
        duration_str = f" · took {row['duration_sec']:.2f}s" if row.get("duration_sec") is not None else ""
        created_str = row.get("created_at", "")[:19].replace("T", " ")
        lines.append(f"  #{idx}  {status_tag} {row['action']} · {created_str}{duration_str}")

        params_str = ""
        if row.get("params_json"):
            try:
                p = json.loads(row["params_json"])
                if p:
                    params_str = ", ".join(f"{k}={v}" for k, v in sorted(p.items()))
            except Exception:
                pass
        if params_str:
            lines.append(f"      Params: {params_str}")

        in_dict = row.get("input_dictionary_id")
        out_dict = row.get("output_dictionary_id")
        if in_dict is not None or out_dict is not None:
            dict_flow = f"dict {in_dict or 'none'} → dict {out_dict or 'unchanged'}"
            lines.append(f"      Dictionary: {dict_flow}")

        if row.get("summary"):
            first_summary = row["summary"].splitlines()[0]
            lines.append(f"      Summary: {first_summary}")
        lines.append("")

    return CommandResult.output(*lines)


# --------------------------------------------------------------------------- #
# Command registry
# --------------------------------------------------------------------------- #

# Loader flag (kwarg starts with "__" so it is popped before the analysis call
# and skipped by the analysis-kwarg validation). "none" = learning scope.
_WINDOWS_FLAG = Flag(
    "windows", "__window_limit__", "int", "N",
    "how many windows to load — default is the configured N_WINDOWS; 0 = all",
    default="none",
    allow_none=True,
)

# Long descriptions follow documentation/Docs/knobNerative.md (itself consistent
# with randomized-dictionary.tex/.pdf); keep them in step when a knob changes.
_RANDOMIZED_BUILDER_FLAGS = (
    _WINDOWS_FLAG,
    Flag(
        "m-per-outer", "m_per_outer", "int", "N",
        "random seeds sampled each outer pass",
        default="5",
        details=(
            "Symbol M. How many random seeds Create draws on each outer pass.\n\n"
            "Seed locations are exactly the intervals of the GPL partition, "
            "re-cropped from whatever residual the builder currently holds — it "
            "never invents new ones. Of the crops that pass the width bounds and "
            "the epsilon floor, M are drawn uniformly without replacement and each "
            "is normalized into a rank-1 atom. That is all Create does: no pursuit, "
            "no PCA. Refine then decides which of those M survive."
        ),
    ),
    Flag(
        "min-coverage", "min_coverage", "float", "0-1",
        "stop once this time fraction is explained",
        default="0.5",
        details=(
            "Stopping rule for Grow. After each Create → Refine → merge → peel "
            "pass the builder measures the fraction of time bins on the original "
            "residual that are claimed by a kept placement; once that fraction "
            "passes min-coverage the run stops.\n\n"
            "Lower values finish sooner with fewer atoms; raising it keeps the "
            "builder peeling for more marginal structure, bounded by max-outer. A "
            "pass that finds no pool (empty_pool) or loses every atom "
            "(no_survivors) also ends the run early."
        ),
    ),
    Flag(
        "max-outer", "max_outer", "int", "N",
        "cap on outer Create→Refine passes",
        default="100",
        details=(
            "Hard cap on the number of outer Create → Refine passes, so a run "
            "whose coverage never reaches min-coverage still ends.\n\n"
            "Each pass samples m-per-outer new seeds from the current residual, "
            "refines them for refine-iters rounds, merges the survivors into the "
            "dictionary and deflates their placements from the residual before "
            "the next pass samples again."
        ),
    ),
    Flag(
        "refine-iters", "refine_iters", "int", "N",
        "refine iterations per outer batch",
        default="3",
        details=(
            "Symbol I. Rounds of Refine applied to each outer batch.\n\n"
            "Every round runs one matching pursuit of every atom against the "
            "residual, assigns each accepted placement to exactly one atom, drops "
            "atoms holding fewer than max(n-min, n-components) patches, and "
            "rebuilds the survivors as the top-K principal subspace of their "
            "patches. More rounds let atoms drift toward the shapes they actually "
            "match. The post-refine-distance filter runs only once, after the "
            "last round."
        ),
    ),
    Flag(
        "n-min", "n_min", "int", "N",
        "minimum patches required to keep an atom",
        default="10",
        details=(
            "Symbol n_min. Minimum number of patches an atom must hold after a "
            "refine round to survive; atoms below max(n-min, n-components) are "
            "dropped, and the same rule is applied again after the "
            "post-refine-distance filter removes placements.\n\n"
            "Grow retunes this to 10 (refine on its own uses 3), stricter, so it "
            "prefers shapes that truly repeat over one-off events."
        ),
    ),
    Flag(
        "n-components", "n_components", "int", "K",
        "subspace rank per atom (1 = rank-1)",
        default="1",
        details=(
            "Symbol K. Rank of each atom's subspace. Surviving atoms are rebuilt "
            "as the top-K principal subspace of their assigned patches; K = 1 "
            "gives ordinary rank-1 atoms.\n\n"
            "K also sets a floor on survival: an atom needs at least "
            "max(n-min, K) patches to be rebuilt."
        ),
    ),
    Flag(
        "min-template-width",
        "min_template_width",
        "int",
        "N",
        "minimum seed patch width (time bins); none = no lower bound",
        default="10",
        allow_none=True,
        apply_default=True,
        details=(
            "Symbol W_min. Lower bound, in time bins, on the width of a partition "
            "crop for it to enter the Create pool; narrower crops are skipped "
            "before any sampling happens.\n\n"
            "Together with max-template-width this decides which GPL intervals "
            "can become seeds at all. Write none to remove the bound."
        ),
    ),
    Flag(
        "max-template-width",
        "max_template_width",
        "int",
        "N",
        "maximum seed patch width (time bins); none = no upper bound",
        default="20",
        allow_none=True,
        apply_default=True,
        details=(
            "Symbol W_max. Upper bound, in time bins, on the width of a partition "
            "crop for it to enter the Create pool; wider crops are skipped before "
            "any sampling happens.\n\n"
            "Together with min-template-width this decides which GPL intervals "
            "can become seeds at all. Write none to remove the bound."
        ),
    ),
    Flag(
        "eps", "eps_step_fraction", "float", "F",
        "matching-pursuit stop: next gain below eps × first gain",
        default="0.01",
        details=(
            "Symbol ε (varepsilon). Stop criterion inside each matching pursuit "
            "during Refine. Pursuit accepts the highest-gain placement, deflates "
            "it from the residual and repeats; it stops once the next gain falls "
            "below eps times the first accepted gain.\n\n"
            "Lowering eps makes each pursuit place more atoms; raising it keeps "
            "only the strongest placements. Not to be confused with epsilon, "
            "which filters the seed pool before any pursuit runs."
        ),
    ),
    Flag(
        "post-refine-distance",
        "post_refine_max_relative_distance",
        "float",
        "0-1",
        "reject patches over this unexplained-energy fraction after refine",
        default="0.9",
        details=(
            "Symbol τ. After the last refine round, each placement's unexplained "
            "fraction — the relative distance between the patch and its atom's "
            "reconstruction — is compared with τ. Placements above it are "
            "eliminated, atoms that then fall below max(n-min, n-components) go "
            "with them, and the residual is re-deflated from the kept matchings "
            "alone.\n\n"
            "Refine on its own uses 0.5; Grow loosens it to 0.9 to keep marginal "
            "placements so coverage can rise. This is the post-refine τ; the "
            "optional τ inside matching pursuit (max_relative_distance) is not "
            "exposed here."
        ),
    ),
    Flag(
        "epsilon", "epsilon", "float", "F",
        "patch-norm floor for the seed pool",
        default="0.1",
        details=(
            "Symbol ϵ (epsilon). Near-silence floor for the Create pool: a crop "
            "enters the pool only if its norm exceeds epsilon.\n\n"
            "Because the residual carries only what is still unexplained, this "
            "floor is what makes \"poorly explained\" concrete. Raising it pushes "
            "seeds toward louder leftover structure; lowering it lets faint crops "
            "be sampled. Distinct from eps, the matching-pursuit stop inside Refine."
        ),
    ),
)


COMMAND_DEFS: dict[str, CommandDef] = {
    "history": CommandDef(
        name="history",
        summary="Show past dictionary actions recorded for this recording.",
        positional="",
        flags=(),
        handler=_cmd_history,
        example="history",
        analysis=False,
        runnable=True,
    ),
    "randomized_builder": CommandDef(
        name="randomized_builder",
        summary="Build a new dictionary by random sampling + refine until coverage "
        "(learn_randomized_outer).",
        positional="",
        flags=_RANDOMIZED_BUILDER_FLAGS,
        handler=_cmd_randomized_builder,
        example="randomized_builder with m-per-outer 5 and min-coverage 0.5",
    ),
}


def action_catalog() -> list[dict[str, Any]]:
    """Form-facing catalog of actions and their parameters (history + randomized_builder only)."""
    available = function_availability()
    return [
        {
            "name": defn.name,
            "summary": defn.summary,
            "example": defn.example,
            "available": available.get(defn.name, True),
            "runnable": defn.runnable,
            "flags": [
                {
                    "name": flag.name,
                    "kwarg": flag.kwarg,
                    "kind": flag.kind,
                    "metavar": flag.metavar,
                    "help": flag.help,
                    "default": flag.default,
                    "choices": list(flag.choices) if flag.choices else None,
                    "allowNone": flag.allow_none,
                    "details": flag.details,
                }
                for flag in defn.flags
            ],
        }
        for name, defn in COMMAND_DEFS.items()
        if name in ALLOWED_ACTIONS
    ]


def run_action(
    ctx: CommandContext, action: str, params: dict[str, Any] | None = None
) -> CommandResult:
    """Coerce structured params and execute one dictionary action."""
    name = (action or "").strip().lower()
    defn = COMMAND_DEFS.get(name) if name in ALLOWED_ACTIONS else None
    if defn is None:
        return CommandResult.error(f"unknown action: {name!r}")
    if not defn.runnable:
        return CommandResult.error(f"{name}: not implemented in this form yet")
    try:
        kwargs = coerce_params(defn, params)
        return defn.handler(ctx, kwargs, [])
    except CommandError as exc:
        return CommandResult.error(f"{name}: {exc}")
    except Exception as exc:  # noqa: BLE001 - surface analysis failures to the form
        return CommandResult.error(f"{name}: {type(exc).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# Background runner
# --------------------------------------------------------------------------- #
#
# extend can run for minutes of CPU-bound Python that holds the GIL, so it runs
# in a *separate process* (app.services.terminal_runner). A thread wouldn't help
# — it would starve uvicorn's event loop and hang every request. The worker
# thread here only blocks on subprocess IO (GIL released), so the server stays
# responsive. The POST starts a run and returns its id; the client polls
# get_run() until it is done.

_RUNS: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_RUNS_LOCK = threading.Lock()
_MAX_RUNS = 200  # cap so abandoned runs (never polled) can't leak unbounded.

# Sentinel prefix the child prints one JSON progress dict behind, on its own
# line; the worker below tails stdout for these to update the run's progress.
PROGRESS_PREFIX = "@@PROGRESS@@"


# Give the child its own process group/session so terminate can kill the whole
# tree. ``Popen.kill()`` reaches only the direct child: anything it spawned (a
# ProcessPool candidate pool, say) survives and keeps burning cores after the
# user hits Terminate.
_SPAWN_KWARGS: dict[str, Any] = {} if os.name == "nt" else {"start_new_session": True}


def _kill_process_tree(proc: subprocess.Popen) -> None:
    """Kill the action child and every process it spawned."""
    if proc.poll() is not None:  # already exited; nothing to reap here
        return
    try:
        if os.name == "nt":
            # /T walks the child tree; plain kill() would orphan it.
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                check=False,
            )
        else:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        # Process group gone, or no permission - fall back to the direct child.
        try:
            proc.kill()
        except Exception:
            pass


def start_run(
    ctx: CommandContext, action: str, params: dict[str, Any] | None = None
) -> str:
    """Kick off an action in a subprocess; return its run id."""
    from database import catalog

    run_id = uuid.uuid4().hex
    start_time = time.monotonic()
    action_id = None
    if action != "history":
        try:
            with closing(catalog.connect(CATALOG_DB)) as conn:
                action_id = catalog.record_action_start(
                    conn,
                    segmentation_id=ctx.segmentation_id,
                    stem=ctx.stem,
                    action=action,
                    params=params,
                    input_dictionary_id=ctx.dictionary_id,
                )
                conn.commit()
        except Exception:
            pass

    with _RUNS_LOCK:
        _RUNS[run_id] = {
            "status": "running",
            "ok": None,
            "lines": [],
            "new_dictionary_id": None,
            "progress": None,
            "proc": None,
            "result_file": None,
            "action_id": action_id,
            "start_time": start_time,
            "terminated": False,
        }
        while len(_RUNS) > _MAX_RUNS:
            _RUNS.popitem(last=False)

    payload = json.dumps(
        {
            "stem": ctx.stem,
            "segmentation_id": ctx.segmentation_id,
            "dictionary_id": ctx.dictionary_id,
            "current_template": ctx.current_template,
            "source_wav": ctx.source_wav,
            "action": action,
            "params": params or {},
        }
    )

    def _finish(ok: bool, lines: list[str], new_dictionary_id: int | None = None) -> None:
        duration_sec = time.monotonic() - start_time
        with _RUNS_LOCK:
            run = _RUNS.get(run_id)
            if run is None or run.get("terminated"):
                return
            _RUNS[run_id] = {
                "status": "done",
                "ok": ok,
                "lines": lines,
                "new_dictionary_id": new_dictionary_id,
                "progress": None,
                "proc": None,
                "result_file": None,
                "action_id": action_id,
                "start_time": start_time,
                "terminated": False,
            }

        if action_id is not None:
            try:
                summary_text = None
                if lines:
                    summary_text = lines[0] if len(lines) == 1 else f"{lines[0]} ({lines[1].strip() if len(lines) > 1 else ''})"
                with closing(catalog.connect(CATALOG_DB)) as conn:
                    catalog.record_action_finish(
                        conn,
                        action_id,
                        status="completed" if ok else "failed",
                        output_dictionary_id=new_dictionary_id,
                        summary=summary_text,
                        duration_sec=duration_sec,
                    )
                    conn.commit()
            except Exception:
                pass

    def _set_progress(progress: dict[str, Any]) -> None:
        with _RUNS_LOCK:
            run = _RUNS.get(run_id)
            if run is not None and run["status"] == "running":
                run["progress"] = progress

    def _worker() -> None:
        result_file = Path(tempfile.gettempdir()) / f"terminal_run_{run_id}.json"
        diagnostics: list[str] = []
        try:
            # Popen (not run) so we can tail stdout live: the child prints one
            # PROGRESS_PREFIX line per event, which we surface without waiting for
            # the command to finish. stderr is merged in so failures still report.
            proc = subprocess.Popen(
                [sys.executable, "-m", "app.services.terminal_runner", str(result_file)],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                cwd=str(BACKEND_DIR),  # so `import app...` resolves in the child
                **_SPAWN_KWARGS,
            )
            with _RUNS_LOCK:
                run = _RUNS.get(run_id)
                if run is None or run.get("terminated"):
                    try:
                        _kill_process_tree(proc)
                    except Exception:
                        pass
                    return
                run["proc"] = proc
                run["result_file"] = result_file

            assert proc.stdin is not None and proc.stdout is not None
            proc.stdin.write(payload)
            proc.stdin.close()
            for raw_line in proc.stdout:
                line = raw_line.rstrip("\n")
                if line.startswith(PROGRESS_PREFIX):
                    try:
                        _set_progress(json.loads(line[len(PROGRESS_PREFIX):]))
                    except ValueError:
                        pass
                elif line:
                    diagnostics.append(line)
            returncode = proc.wait()

            with _RUNS_LOCK:
                run = _RUNS.get(run_id)
                if run is not None and run.get("terminated"):
                    return

            if result_file.is_file():
                data = json.loads(result_file.read_text(encoding="utf-8"))
                _finish(
                    bool(data.get("ok")),
                    list(data.get("lines", [])),
                    data.get("new_dictionary_id"),
                )
            else:
                detail = ("\n".join(diagnostics) or "no output").strip()
                _finish(False, [f"command runner failed (exit {returncode}): {detail[-500:]}"])
        except Exception as exc:  # noqa: BLE001
            with _RUNS_LOCK:
                run = _RUNS.get(run_id)
                if run is not None and run.get("terminated"):
                    return
            _finish(False, [f"command runner error: {type(exc).__name__}: {exc}"])
        finally:
            result_file.unlink(missing_ok=True)

    threading.Thread(target=_worker, name=f"terminal-{run_id[:8]}", daemon=True).start()
    return run_id


def terminate_run(run_id: str) -> bool:
    """Terminate an active background action, kill its subprocess, and clean up."""
    from database import catalog

    proc = None
    result_file = None
    action_id = None
    start_time = None

    with _RUNS_LOCK:
        run = _RUNS.get(run_id)
        if run is None or run.get("status") != "running":
            return False
        run["terminated"] = True
        run["status"] = "done"
        run["ok"] = False
        run["lines"] = ["[Action terminated by user]"]
        run["new_dictionary_id"] = None
        run["progress"] = None
        proc = run.get("proc")
        result_file = run.get("result_file")
        action_id = run.get("action_id")
        start_time = run.get("start_time")

    if proc is not None:
        _kill_process_tree(proc)

    if result_file is not None:
        try:
            Path(result_file).unlink(missing_ok=True)
        except Exception:
            pass

    if action_id is not None:
        duration_sec = time.monotonic() - (start_time or time.monotonic())
        try:
            with closing(catalog.connect(CATALOG_DB)) as conn:
                catalog.record_action_finish(
                    conn,
                    action_id,
                    status="terminated",
                    output_dictionary_id=None,
                    summary="[Action terminated by user]",
                    duration_sec=duration_sec,
                )
                conn.commit()
        except Exception:
            pass

    return True


def get_run(run_id: str) -> dict[str, Any] | None:
    """Snapshot a run's state. A finished run is dropped once fetched (one-shot)."""
    with _RUNS_LOCK:
        run = _RUNS.get(run_id)
        if run is None:
            return None
        progress = run.get("progress")
        if progress is not None and run["status"] == "running":
            # A round emits nothing for minutes, so refresh elapsed (and the ETA
            # derived from it) at poll time — otherwise the readout looks frozen
            # exactly when the run is slowest.
            progress = dict(progress)
            elapsed = time.monotonic() - float(run.get("start_time") or time.monotonic())
            progress["elapsed_s"] = round(elapsed, 1)
            progress["elapsed"] = _fmt_duration(elapsed)
            fraction = progress.get("fraction")
            if fraction:
                eta = elapsed * (1.0 - float(fraction)) / float(fraction)
                progress["eta_s"] = round(eta, 1)
                progress["eta"] = _fmt_duration(eta)
        snapshot = {
            "status": run["status"],
            "ok": run.get("ok"),
            "lines": list(run.get("lines") or []),
            "new_dictionary_id": run.get("new_dictionary_id"),
            "progress": progress,
        }
        if run["status"] == "done":
            _RUNS.pop(run_id, None)
        return snapshot
