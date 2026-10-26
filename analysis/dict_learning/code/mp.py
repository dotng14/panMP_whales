"""Place dictionary atoms on a spectrogram and subtract them."""

from __future__ import annotations

import json
import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator


_LOCK = threading.Lock()
_ENABLED = False
# name -> count
_COUNTS: dict[str, int] = defaultdict(int)
# name -> list of per-invocation sizes (e.g. atoms per MP loop)
_SERIES: dict[str, list[int]] = defaultdict(list)
# name -> list of durations in seconds
_TIMINGS: dict[str, list[float]] = defaultdict(list)


def enabled() -> bool:
    if _ENABLED:
        return True
    return os.environ.get("DICT_LEARNING_LOOP_STATS", "").strip() not in ("", "0", "false", "False")


def enable(on: bool = True) -> None:
    global _ENABLED
    _ENABLED = bool(on)


def reset() -> None:
    with _LOCK:
        _COUNTS.clear()
        _SERIES.clear()
        _TIMINGS.clear()


def incr(name: str, n: int = 1) -> None:
    if not enabled():
        return
    with _LOCK:
        _COUNTS[name] += int(n)


def record(name: str, value: int) -> None:
    """Append one observation (e.g. atoms placed in one MP loop)."""
    if not enabled():
        return
    with _LOCK:
        _SERIES[name].append(int(value))
        _COUNTS[name + ".calls"] += 1
        _COUNTS[name + ".sum"] += int(value)


def record_seconds(name: str, seconds: float) -> None:
    """Append one wall-time observation in seconds."""
    if not enabled():
        return
    with _LOCK:
        _TIMINGS[name].append(float(seconds))
        _COUNTS[name + ".calls"] += 1


def snapshot() -> dict[str, Any]:
    with _LOCK:
        series_summary: dict[str, Any] = {}
        for name, values in _SERIES.items():
            if not values:  # no-pipeline-cover: record() always appends a value
                continue
            series_summary[name] = {
                "n": len(values),
                "sum": int(sum(values)),
                "min": int(min(values)),
                "max": int(max(values)),
                "mean": float(sum(values) / len(values)),
                "values": list(values),
            }
        timing_summary: dict[str, Any] = {}
        for name, values in _TIMINGS.items():
            if not values:  # no-pipeline-cover: record_seconds() always appends a value
                continue
            timing_summary[name] = {
                "n": len(values),
                "sum_s": float(sum(values)),
                "min_s": float(min(values)),
                "max_s": float(max(values)),
                "mean_s": float(sum(values) / len(values)),
                "values_s": list(values),
            }
        return {
            "counts": dict(sorted(_COUNTS.items())),
            "series": series_summary,
            "timings_s": timing_summary,
        }


@contextmanager
def collecting(*, reset_first: bool = True) -> Iterator[None]:
    """Context manager: enable counters, optionally reset, disable on exit."""
    prev = _ENABLED
    enable(True)
    if reset_first:  # no-pipeline-branch: the CLIs always start from zero
        reset()
    try:
        yield
    finally:
        enable(prev)


import os
import sys
from collections.abc import Iterable, Iterator
from typing import Any, TypeVar

try:
    from tqdm.auto import tqdm as _tqdm_impl
except ImportError:  # pragma: no cover
    _tqdm_impl = None  # type: ignore[assignment]


class _NullPbar:  # no-pipeline-cover: stand-in when tqdm is not installed
    """No-op stand-in when tqdm is unavailable."""

    def update(self, n: int = 1) -> None:
        return None

    def close(self) -> None:
        return None

    def __enter__(self) -> _NullPbar:
        return self

    def __exit__(self, *args: object) -> None:
        return None


def tqdm(*args: Any, **kwargs: Any):
    """tqdm wrapper that returns a no-op bar when tqdm is not installed."""
    if _tqdm_impl is None:  # no-pipeline-cover: tqdm is a dependency
        return _NullPbar()
    return _tqdm_impl(*args, **kwargs)


T = TypeVar("T")


def progress_enabled(show_progress: bool | None = None) -> bool:
    if show_progress is None:
        return _progress_by_default()  # no-pipeline-cover: the CLIs always pass a flag
    return bool(show_progress) and _tqdm_impl is not None


def _progress_by_default() -> bool:  # no-pipeline-cover: the CLIs always pass a flag
    """Bars on when tqdm is installed, not disabled, and stderr is a terminal."""
    if _tqdm_impl is None:
        return False
    if os.environ.get("TQDM_DISABLE", "").strip():
        return False
    try:
        return bool(sys.stderr.isatty())
    except Exception:  # noqa: BLE001
        return False


def iter_progress(  # no-pipeline-cover: used by refine_dictionary_by_averaging (learn_from_motifs)
    iterable: Iterable[T],
    *,
    desc: str = "",
    total: int | None = None,
    leave: bool = True,
    show_progress: bool | None = None,
    unit: str = "it",
) -> Iterator[T]:
    """Wrap ``iterable`` in tqdm when progress is enabled."""
    enabled = progress_enabled(show_progress)
    if not enabled:
        yield from iterable
        return
    yield from tqdm(
        iterable,
        desc=desc,
        total=total,
        leave=leave,
        unit=unit,
        dynamic_ncols=True,
    )


def range_progress(  # no-pipeline-cover: used by refine_dictionary_by_averaging (learn_from_motifs)
    n: int | None,
    *,
    desc: str = "",
    leave: bool = True,
    show_progress: bool | None = None,
    unit: str = "it",
) -> Iterator[int]:
    """Like ``range(n)`` with an optional progress bar.

    When ``n`` is ``None``, yields ``0, 1, 2, …`` indefinitely (caller breaks).
    """
    if n is not None:
        yield from iter_progress(
            range(n),
            desc=desc,
            total=n,
            leave=leave,
            show_progress=show_progress,
            unit=unit,
        )
        return

    enabled = progress_enabled(show_progress)
    if not enabled:
        i = 0
        while True:
            yield i
            i += 1
        return

    pbar = tqdm(
        desc=desc,
        total=None,
        leave=leave,
        unit=unit,
        dynamic_ncols=True,
    )
    try:
        i = 0
        while True:
            yield i
            i += 1
            pbar.update(1)
    finally:
        pbar.close()

import os
from typing import Literal

import numpy as np

try:
    from numba import njit

    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover - optional acceleration
    _NUMBA_AVAILABLE = False

    def njit(*_args, **_kwargs):  # type: ignore[misc]
        def decorator(fn):
            return fn

        return decorator


try:
    import cupy as cp  # type: ignore[import]

    _CUPY_IMPORTABLE = True
except Exception:  # pragma: no cover
    cp = None  # type: ignore[assignment]
    _CUPY_IMPORTABLE = False

# Cached probe: None = not probed, True/False = usable CUDA kernels.
_CUPY_USABLE: bool | None = None
_CUPY_DEVICE_COUNT = 0

# Min fused multiply-adds before GPU transfer overhead is usually worth it.
_GPU_MIN_OPS = int(os.environ.get("DICT_LEARNING_MP_GPU_MIN_OPS", str(2_000_000)))

ComputeBackend = Literal["cpu", "cuda", "auto"]


@njit(cache=True, nogil=True)  # no-pipeline-cover: numba-compiled, coverage cannot trace it; unit tests compare it with the NumPy version
def _score_shifts_numba(
    tmpl: np.ndarray,
    resid: np.ndarray,
    normalize: bool,
) -> np.ndarray:
    f, w = tmpl.shape
    t = resid.shape[1]
    n = t - w + 1
    scores = np.zeros(n, dtype=np.float64)
    for fi in range(f):
        for s in range(n):
            acc = 0.0
            for k in range(w):
                acc += resid[fi, s + k] * tmpl[fi, k]
            scores[s] += acc
    if normalize:
        denom = 0.0
        for fi in range(f):
            for k in range(w):
                denom += tmpl[fi, k] * tmpl[fi, k]
        if denom <= 0.0:
            return np.zeros(n, dtype=np.float64)
        for s in range(n):
            scores[s] /= denom
    return scores


def _score_shifts_numpy(  # no-pipeline-cover: fallback when numba is not installed
    tmpl: np.ndarray,
    resid: np.ndarray,
    *,
    normalize: bool,
) -> np.ndarray:
    f, w = tmpl.shape
    _fr, t = resid.shape
    if t < w:
        return np.zeros(0, dtype=np.float64)
    scores = np.zeros(t - w + 1, dtype=np.float64)
    for fi in range(f):
        scores += np.correlate(resid[fi], tmpl[fi], mode="valid")
    if normalize:
        denom = float(np.sum(tmpl * tmpl))
        if denom <= 0.0:
            return np.zeros_like(scores)
        scores = scores / denom
    return scores


def _probe_cupy() -> bool:
    """Return True only if CuPy can actually run a kernel on a device."""
    global _CUPY_USABLE, _CUPY_DEVICE_COUNT
    if _CUPY_USABLE is not None:
        return _CUPY_USABLE
    if not _CUPY_IMPORTABLE or cp is None:  # no-pipeline-cover: CuPy not installed
        _CUPY_USABLE = False
        _CUPY_DEVICE_COUNT = 0
        return False
    try:  # gpu-only: needs a CUDA device
        n = int(cp.cuda.runtime.getDeviceCount())
        if n <= 0:  # no-pipeline-cover: CuPy installed but no CUDA device
            _CUPY_USABLE = False
            _CUPY_DEVICE_COUNT = 0
            return False
        with cp.cuda.Device(0):
            a = cp.arange(8, dtype=cp.float64)
            _ = float(cp.asnumpy((a * a).sum()))
        _CUPY_DEVICE_COUNT = n
        _CUPY_USABLE = True
    except Exception:  # no-pipeline-cover: broken CUDA toolkit
        _CUPY_USABLE = False
        _CUPY_DEVICE_COUNT = 0
    return _CUPY_USABLE


def cupy_available() -> bool:
    return _probe_cupy()


def gpu_device_count() -> int:  # gpu-only: needs a CUDA device
    _probe_cupy()
    return int(_CUPY_DEVICE_COUNT)


def resolve_mp_compute(compute: str | None = None) -> ComputeBackend:
    """Resolve ``cpu`` / ``cuda`` / ``auto`` from arg → env → defaults."""
    from .core import defaults

    raw = compute
    if raw is None or str(raw).strip() == "":  # no-pipeline-branch: the CLIs set the backend through DICT_LEARNING_MP_COMPUTE
        raw = os.environ.get("DICT_LEARNING_MP_COMPUTE")
    if raw is None or str(raw).strip() == "":
        # Honor the shared GPL compute switch. It is documented as cpu|cuda and
        # GPL_Python's backend reads it that way, so "cpu" has to force CPU here
        # too: recognising only "cuda" meant GPL_COMPUTE=cpu fell through to
        # MP_COMPUTE ("auto"), silently re-enabling CUDA on any host where CuPy
        # happened to be installed - while the same setting did pin the GPL
        # pipeline to CPU. Unrecognised values still defer to defaults.
        gpl = os.environ.get("GPL_COMPUTE", "").strip().lower()
        raw = gpl if gpl in ("cpu", "cuda", "auto") else defaults.MP_COMPUTE
    m = str(raw).strip().lower()
    if m not in ("cpu", "cuda", "auto"):
        raise ValueError(f"mp_compute must be cpu|cuda|auto, got {compute!r}")
    return m  # type: ignore[return-value]


def gpu_beneficial(f: int, w: int, t: int) -> bool:  # gpu-only: needs a CUDA device
    """Heuristic: GPU when correlation work exceeds transfer overhead."""
    if t < w:
        return False  # no-pipeline-cover: windows are wider than atoms
    ops = int(f) * int(w) * int(t - w + 1)
    return ops >= _GPU_MIN_OPS


def _score_shifts_cupy(  # gpu-only: needs a CUDA device
    tmpl: np.ndarray,
    resid: np.ndarray,
    *,
    normalize: bool,
    gpu_id: int = 0,
) -> np.ndarray:
    """CuPy valid correlation (float64). Requires a working CUDA toolchain."""
    assert cp is not None
    device_id = int(gpu_id) % max(1, gpu_device_count())
    with cp.cuda.Device(device_id):
        g_tmpl = cp.asarray(tmpl, dtype=cp.float64)
        g_resid = cp.asarray(resid, dtype=cp.float64)
        f, w = int(g_tmpl.shape[0]), int(g_tmpl.shape[1])
        t = int(g_resid.shape[1])
        n = t - w + 1
        scores = cp.zeros(n, dtype=cp.float64)
        # Prefer cupyx.scipy when importable; else FFT/convolve via cupy.
        try:
            import cupyx.scipy.signal as cs  # type: ignore[import]

            for fi in range(f):
                scores += cs.correlate(g_resid[fi], g_tmpl[fi], mode="valid")
        except Exception:  # no-pipeline-cover: fallback when cupyx.scipy is missing
            for fi in range(f):
                # correlate(a, v) valid == convolve(a, v[::-1]) valid
                scores += cp.convolve(g_resid[fi], g_tmpl[fi, ::-1], mode="valid")
        if normalize:
            denom = float(cp.asnumpy(cp.sum(g_tmpl * g_tmpl)))
            if denom <= 0.0:  # no-pipeline-cover: atoms always have energy
                return np.zeros(n, dtype=np.float64)
            scores = scores / denom
        return np.asarray(cp.asnumpy(scores), dtype=np.float64)


def score_shifts(
    template: np.ndarray,
    residual: np.ndarray,
    *,
    normalize: bool = True,
    use_numba: bool | None = None,
    compute: str | None = None,
    gpu_id: int = 0,
) -> np.ndarray:
    """Score ``template`` (F×W) against ``residual`` (F×T) at each admissible shift.

    Returns length ``T - W + 1`` of amplitudes α. If ``normalize``:
    ``α = ⟨R[:, t:t+W], T⟩ / ‖T‖_F²``.

    Backend selection (``compute`` / ``DICT_LEARNING_MP_COMPUTE`` / ``GPL_COMPUTE``):

    - ``auto`` (default): CUDA when CuPy kernels work **and** the problem is large
      enough (``DICT_LEARNING_MP_GPU_MIN_OPS``); else Numba/NumPy on CPU.
    - ``cuda``: force GPU when usable, else CPU fallback.
    - ``cpu``: never use GPU.
    """
    tmpl = np.ascontiguousarray(template, dtype=np.float64)
    resid = np.ascontiguousarray(residual, dtype=np.float64)
    if tmpl.ndim != 2 or resid.ndim != 2:
        raise ValueError("template and residual must be 2-D (F×T)")  # no-pipeline-cover: library input check
    if tmpl.shape[0] != resid.shape[0]:  # no-pipeline-cover: the CLI checks atom rows against the windows
        raise ValueError(
            f"Frequency mismatch: template {tmpl.shape[0]} vs residual {resid.shape[0]}"
        )
    f, w = tmpl.shape
    t = resid.shape[1]
    if t < w:  # no-pipeline-cover: windows are wider than atoms
        return np.zeros(0, dtype=np.float64)

    backend = resolve_mp_compute(compute)
    want_gpu = False
    if backend == "cuda":
        want_gpu = cupy_available()  # gpu-only: needs a CUDA device
    elif backend == "auto":
        want_gpu = cupy_available() and gpu_beneficial(f, w, t)

    out: np.ndarray | None = None
    if want_gpu:  # gpu-only: needs a CUDA device
        try:
            out = _score_shifts_cupy(
                tmpl, resid, normalize=normalize, gpu_id=gpu_id
            )
        except Exception:  # no-pipeline-cover: broken CUDA toolkit or out of GPU memory
            # Broken toolkit / OOM → fall through to CPU.
            out = None

    if out is None:  # gpu-only-branch: only CuPy fills out first
        want_numba = _NUMBA_AVAILABLE if use_numba is None else bool(use_numba)
        if want_numba and _NUMBA_AVAILABLE:
            out = _score_shifts_numba(tmpl, resid, bool(normalize))
        else:
            out = _score_shifts_numpy(tmpl, resid, normalize=normalize)  # no-pipeline-cover: numba not installed

    try:

        loop_stats.incr("score_shifts.calls")
    except Exception:  # no-pipeline-cover: counting must never break scoring
        pass
    return out


def numba_available() -> bool:
    return _NUMBA_AVAILABLE

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable, Sequence, TypeVar

from .core import defaults

T = TypeVar("T")
R = TypeVar("R")


def available_cpu_count() -> int:
    """Max CPUs available to this process (affinity-aware when possible)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):  # no-pipeline-cover: platforms without sched_getaffinity
        return max(1, int(os.cpu_count() or 1))


def resolve_mp_workers(n_workers: int | None = None) -> int:
    """Resolve worker count: explicit arg → env → all available CPUs."""
    if n_workers is not None:
        return max(1, int(n_workers))
    env = os.environ.get("DICT_LEARNING_MP_WORKERS")
    if env is not None and str(env).strip() != "":
        return max(1, int(env))
    # defaults.MP_WORKERS is already max CPUs; re-query affinity in case of cgroup change.
    return max(1, int(defaults.MP_WORKERS), available_cpu_count())


def resolve_mp_mode(mode: str | None = None) -> str:
    """Resolve MP mode: ``global`` or ``per_window``."""
    if mode is not None and str(mode).strip() != "":
        m = str(mode).strip().lower()
    else:
        env = os.environ.get("DICT_LEARNING_MP_MODE")
        if env is not None and str(env).strip() != "":
            m = str(env).strip().lower()
        else:
            m = str(defaults.MP_MODE).strip().lower()
    if m not in ("global", "per_window"):
        raise ValueError(f"mp_mode must be 'global' or 'per_window', got {mode!r}")
    return m


from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Literal
import math
import time

import numpy as np

from .core import defaults
from .core import Dictionary, Matching, Template, Window

IntervalMap = dict[str, list[tuple[int, int]]]
MpMode = Literal["global", "per_window"]


@dataclass
class MpStepResult:
    matching: Matching | None
    energy_before: float
    energy_after: float
    delta_energy: float


def residual_energy(stack: dict[str, np.ndarray] | list[Window] | list[np.ndarray]) -> float:
    total = 0.0
    if isinstance(stack, dict):
        arrays = stack.values()
    else:  # no-pipeline-cover: library input: a list of windows or arrays
        arrays = []
        for item in stack:
            arrays.append(item.data if isinstance(item, Window) else item)
    for arr in arrays:
        total += float(np.sum(np.asarray(arr, dtype=np.float64) ** 2))
    return total


def _as_residual_map(
    residual_stack: dict[str, np.ndarray] | list[Window],
) -> dict[str, np.ndarray]:
    if isinstance(residual_stack, dict):
        return {k: np.asarray(v, dtype=np.float64) for k, v in residual_stack.items()}
    out: dict[str, np.ndarray] = {}  # no-pipeline-cover: library input: a list of windows
    for w in residual_stack:  # no-pipeline-cover: library input: a list of windows
        if not isinstance(w, Window):
            raise TypeError("list residual_stack must contain Window objects")
        out[w.window_id] = np.asarray(w.data, dtype=np.float64)
    return out  # no-pipeline-cover: library input: a list of windows


def copy_residual_stack(
    residual_stack: dict[str, np.ndarray] | list[Window],
) -> dict[str, np.ndarray]:
    base = _as_residual_map(residual_stack)
    return {k: np.array(v, copy=True) for k, v in base.items()}


def _dictionary_arrays(
    dictionary: Dictionary,
) -> list[tuple[int, np.ndarray, int]]:
    """``(template_id, basis (k,F,W), width)`` for pickling / scoring."""
    out: list[tuple[int, np.ndarray, int]] = []
    for tmpl in dictionary:
        basis = np.ascontiguousarray(tmpl.basis, dtype=np.float64)
        out.append((int(tmpl.template_id), basis, int(tmpl.width)))
    return out


def _valid_shift_mask(
    n_shifts: int, width: int, occupied: list[tuple[int, int]] | None
) -> np.ndarray:
    valid = np.ones(n_shifts, dtype=bool)
    for start, end in occupied or []:
        lo = max(0, int(start) - width + 1)
        hi = min(n_shifts, int(end))
        if lo < hi:  # no-pipeline-branch: reserved spans always overlap the shift range
            valid[lo:hi] = False
    return valid


def _sliding_patch_energy(resid: np.ndarray, width: int, n_shifts: int) -> np.ndarray:
    """``||x_s||_F²`` for every shift ``s``, via a cumsum over column energies."""
    col = np.sum(resid * resid, axis=0, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(col)))
    return cumulative[width : width + n_shifts] - cumulative[:n_shifts]


def _close_enough_mask(
    explained: np.ndarray,
    patch_energy: np.ndarray,
    max_relative_distance: float,
) -> np.ndarray:
    """Shifts whose unexplained energy fraction is within the threshold.

    ``r = 1 − explained / ||x||²``; empty patches never qualify.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(
            explained,
            patch_energy,
            out=np.zeros_like(patch_energy, dtype=np.float64),
            where=patch_energy > 0.0,
        )
    relative = 1.0 - ratio
    return (patch_energy > 0.0) & (relative <= float(max_relative_distance))


PlacementScore = tuple[float, float]
"""``(relative_fit, absolute_energy)``, compared via :func:`_placement_beats`.

``relative_fit`` (``explained / ||patch||_F²``, on ``[0, 1]``) is the primary
key so atoms of different norm — or patches of different loudness — compete
on how well they're explained, not on raw amplitude. ``absolute_energy`` (the
raw energy explained) only breaks ties, so two equally-perfect fits at very
different scales resolve deterministically (preferring the one that reduces
more residual energy) instead of on floating-point noise.
"""

_SCORE_TOLERANCE = 1e-9


def _placement_beats(candidate: PlacementScore, current: PlacementScore) -> bool:
    """True when ``candidate`` should replace ``current`` as the running best.

    Two independently-computed "identical" atoms (e.g. two PCA rebuilds of
    the same patches) can differ from a perfect fit by ~1e-15 in opposite
    directions purely from floating-point rounding — comparing
    ``PlacementScore`` tuples with plain ``>`` would let that noise (rather
    than genuine fit quality) decide the winner. Both components are compared
    with a relative tolerance first; only a difference that exceeds it counts
    as one candidate genuinely fitting better. When both are within
    tolerance, ``candidate`` does *not* beat ``current`` — the earlier-seen
    placement wins, keeping tie-breaks deterministic (stable by encounter
    order) instead of noise-dependent.
    """
    c_fit, c_energy = candidate
    b_fit, b_energy = current
    if not math.isclose(c_fit, b_fit, rel_tol=_SCORE_TOLERANCE, abs_tol=_SCORE_TOLERANCE):
        return c_fit > b_fit
    if not math.isclose(c_energy, b_energy, rel_tol=_SCORE_TOLERANCE, abs_tol=_SCORE_TOLERANCE):  # no-pipeline-branch: an exact tie in both fit and energy
        return c_energy > b_energy
    return False  # no-pipeline-cover: an exact tie in both fit and energy


def _best_placement_on_window(
    basis: np.ndarray,
    resid: np.ndarray,
    *,
    template_id: int,
    window_id: str,
    normalize: bool,
    occupied_intervals: list[tuple[int, int]] | None,
    gpu_id: int = 0,
    pc1_screen: bool = False,
    screen_top: int = 8,
    max_relative_distance: float | None = None,
) -> tuple[PlacementScore, Matching, np.ndarray] | None:
    """Best placement of one atom on one window.

    Returns ``(score, matching, basis)`` or ``None`` — see :data:`PlacementScore`.
    Its relative-fit component is the same scale-free normalization as
    :func:`~..core.patch_template_relative_distance` (that function's
    relative distance is ``1 - relative_fit``), which makes it comparable
    *across atoms of different norm*: callers that pick the best of several
    templates for a patch (or, in global MP, the best of many candidate
    windows) rank by relative fit quality rather than raw amplitude or
    absolute explained energy.

    Shift selection *within* one atom still ranks candidates by raw
    amplitude/energy — that comparison is already scale-free since the atom's
    own norm is constant across shifts of the same atom, so it can't bias the
    choice of shift the way it would bias a choice between different atoms.

    ``max_relative_distance`` gates acceptance on closeness to the atom: a shift
    qualifies only when the fraction of patch energy left unexplained,
    ``1 − ||Π_V x||² / ||x||²``, is within the threshold. The best *qualifying*
    shift wins, so a rejected peak falls through to the next-best candidate
    instead of stopping the search. Rejected shifts are never deflated.
    """
    k = int(basis.shape[0])
    width = int(basis.shape[2])
    n_shifts = int(resid.shape[1]) - width + 1
    if n_shifts <= 0:  # no-pipeline-cover: windows are wider than atoms
        return None
    valid = _valid_shift_mask(n_shifts, width, occupied_intervals)
    if not np.any(valid):  # no-pipeline-cover: a window is never fully reserved
        return None

    # Needed both to gate acceptance (when gated) and to normalize the
    # returned cross-atom score (always).
    patch_energy = _sliding_patch_energy(resid, width, n_shifts)
    # Zero atoms explain nothing, so gating them would reject every shift.
    gated = max_relative_distance is not None and float(np.sum(basis * basis)) > 0.0

    if k == 1:
        tdata = basis[0]
        scores = score_shifts(tdata, resid, normalize=normalize, gpu_id=gpu_id)
        if scores.size == 0:  # no-pipeline-cover: follows from n_shifts > 0
            return None
        scores = np.array(scores, copy=True)
        tmpl_energy = float(np.sum(tdata * tdata))
        if tmpl_energy <= 0.0:  # no-pipeline-cover: atoms always have energy
            return None
        # explained = ⟨x, T⟩² / ||T||²; `scores` is ⟨x,T⟩/||T||² when
        # normalized, else the raw inner product.
        inner = scores * tmpl_energy if normalize else scores
        explained = (inner * inner) / tmpl_energy
        if gated:
            valid = valid & _close_enough_mask(
                explained, patch_energy, float(max_relative_distance)
            )
            if not np.any(valid):
                return None
        scores[~valid] = 0.0
        # Preserve argmax-|amp| among valid (zeros on invalid).
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size == 0:  # no-pipeline-cover: valid is non-empty here
            return None
        score_index = int(np.argmax(np.abs(scores[valid_indices])))
        t_start = int(valid_indices[score_index])
        amp = float(scores[t_start])
        if amp == 0.0:  # no-pipeline-cover: needs a residual exactly orthogonal to the atom
            return None
        patch_energy_at_t = float(patch_energy[t_start])
        if patch_energy_at_t <= 0.0:  # no-pipeline-cover: placed patches always have energy
            return None
        matching = Matching(
            template_id=template_id,
            window_id=window_id,
            t_start=t_start,
            amplitude=amp,
        )
        explained_at_t = float(explained[t_start])
        # explained <= patch_energy always (Bessel's inequality); clip so
        # floating-point overshoot past a perfect fit can't beat an exact 1.0
        # and starve the absolute-energy tiebreak.
        relative_fit = min(1.0, explained_at_t / patch_energy_at_t)
        score = (relative_fit, explained_at_t)
        return score, matching, basis

    # Subspace (k > 1): orthonormal modes → raw inner products.
    def _full_scan() -> tuple[float, int, np.ndarray] | None:
        maps: list[np.ndarray] = []
        for m in range(k):
            scores = score_shifts(basis[m], resid, normalize=False, gpu_id=gpu_id)
            if scores.size == 0:  # no-pipeline-cover: follows from n_shifts > 0
                return None
            maps.append(scores)
        cmaps = np.stack(maps, axis=0)
        energy_map = np.sum(cmaps * cmaps, axis=0)
        energy_map = np.array(energy_map, copy=True)
        allowed = valid
        if gated:
            allowed = allowed & _close_enough_mask(
                energy_map, patch_energy, float(max_relative_distance)
            )
            if not np.any(allowed):
                return None
        energy_map[~allowed] = -np.inf
        t_best = int(np.argmax(energy_map))
        best_energy = float(energy_map[t_best])
        if best_energy <= 0.0 or not allowed[t_best]:  # no-pipeline-cover: an allowed shift always explains some energy
            return None
        return best_energy, t_best, cmaps[:, t_best].astype(np.float64, copy=True)

    scanned: tuple[float, int, np.ndarray] | None = None
    if pc1_screen:  # no-pipeline-cover: pc1 screening is not exposed by the CLIs
        pc1 = score_shifts(basis[0], resid, normalize=False, gpu_id=gpu_id)
        if pc1.size == 0:
            return None
        energy0 = pc1 * pc1
        energy0 = np.array(energy0, copy=True)
        energy0[~valid] = -np.inf
        n_cand = min(int(screen_top), int(np.count_nonzero(valid)))
        if n_cand <= 0:
            return None
        cand = np.argpartition(energy0, -n_cand)[-n_cand:]
        local_best: tuple[float, int, np.ndarray] | None = None
        for t in cand:
            t = int(t)
            patch = resid[:, t : t + width]
            coeffs = np.array(
                [float(np.sum(patch * basis[m])) for m in range(k)],
                dtype=np.float64,
            )
            energy = float(np.dot(coeffs, coeffs))
            if gated and not bool(
                _close_enough_mask(
                    np.array([energy]),
                    patch_energy[t : t + 1],
                    float(max_relative_distance),
                )[0]
            ):
                continue
            if local_best is None or energy > local_best[0]:
                local_best = (energy, t, coeffs)
        if local_best is None or local_best[0] <= 0.0:
            # Screening looked only at pc1 energy, so every candidate can fail
            # the gate while a qualifying shift exists elsewhere.
            scanned = _full_scan() if gated else None
        else:
            scanned = local_best
    else:
        scanned = _full_scan()

    if scanned is None:
        return None
    energy, t_star, coeffs = scanned
    patch_energy_at_t = float(patch_energy[t_star])
    if patch_energy_at_t <= 0.0:  # no-pipeline-cover: placed patches always have energy
        return None

    matching = Matching(
        template_id=template_id,
        window_id=window_id,
        t_start=t_star,
        coeffs=coeffs,
    )
    relative_fit = min(1.0, energy / patch_energy_at_t)
    score = (relative_fit, energy)
    return score, matching, basis


def _deflate_matching(
    resid: np.ndarray,
    matching: Matching,
    basis: np.ndarray,
) -> bool:
    """Subtract the atom projection at ``matching``; return False if crop invalid."""
    width = int(basis.shape[2])
    t0 = matching.t_start
    patch = resid[:, t0 : t0 + width]
    if patch.shape[1] != width:  # no-pipeline-cover: MP places whole atoms
        return False
    k = int(basis.shape[0])
    if k == 1:
        resid[:, t0 : t0 + width] = patch - float(matching.amplitude) * basis[0]
    else:
        coeffs = np.asarray(matching.coeffs, dtype=np.float64)
        projection = np.tensordot(coeffs, basis, axes=(0, 0))
        resid[:, t0 : t0 + width] = patch - projection
    return True


def refit_and_deflate(
    resid: dict[str, np.ndarray],
    dictionary: Dictionary,
    matchings: Sequence[Matching],
    *,
    normalize: bool = True,
) -> tuple[list[Matching], IntervalMap]:
    """Re-fit fixed-position matchings to the current atoms and deflate them.

    Each matching keeps its ``(window_id, template_id, t_start)`` — only its
    amplitude / subspace coefficients are recomputed against the atom as it
    stands now, using the same projection MP would use. The fitted projection is
    subtracted from ``resid`` in place.

    Returns the re-fitted matchings and the time spans they occupy, so a
    following MP pass can be handed those spans as ``occupied_intervals`` and
    leave them alone. Matchings for unknown templates or windows are dropped;
    ``t_start`` is clamped when the atom is too wide to fit at its old offset.
    """
    by_template = {int(t.template_id): t for t in dictionary}
    refitted: list[Matching] = []
    occupied: IntervalMap = {}
    for matching in matchings:
        template = by_template.get(int(matching.template_id))
        window = resid.get(matching.window_id)
        if template is None or window is None:  # no-pipeline-cover: refits only matchings of the current dictionary and stack
            continue
        basis = np.ascontiguousarray(template.basis, dtype=np.float64)
        width = int(basis.shape[2])
        n_shifts = int(window.shape[1]) - width + 1
        if n_shifts <= 0:  # no-pipeline-cover: windows are wider than atoms
            continue
        t0 = min(max(int(matching.t_start), 0), n_shifts - 1)
        patch = window[:, t0 : t0 + width]
        if int(basis.shape[0]) == 1:
            tdata = basis[0]
            raw = float(np.sum(patch * tdata))
            denom = float(np.sum(tdata * tdata)) if normalize else 1.0
            if denom <= 0.0:  # no-pipeline-cover: atoms always have energy
                continue
            refit = Matching(
                template_id=int(matching.template_id),
                window_id=matching.window_id,
                t_start=t0,
                amplitude=raw / denom,
            )
        else:
            coeffs = np.array(
                [
                    float(np.sum(patch * basis[mode]))
                    for mode in range(int(basis.shape[0]))
                ],
                dtype=np.float64,
            )
            refit = Matching(
                template_id=int(matching.template_id),
                window_id=matching.window_id,
                t_start=t0,
                coeffs=coeffs,
            )
        if not _deflate_matching(window, refit, basis):  # no-pipeline-cover: t0 is clamped so the crop always fits
            continue
        refitted.append(refit)
        occupied.setdefault(refit.window_id, []).append((t0, t0 + width))
    return refitted, occupied


def _mp_step_one_window_inplace(
    resid: np.ndarray,
    atoms: list[tuple[int, np.ndarray, int]],
    *,
    window_id: str,
    normalize: bool = True,
    occupied_intervals: list[tuple[int, int]] | None = None,
    gpu_id: int = 0,
    pc1_screen: bool = False,
    screen_top: int = 8,
    max_relative_distance: float | None = None,
) -> MpStepResult:
    """Core one-window Mallat step; mutates ``resid`` in place.

    Caller must own ``resid`` (a private, already-copied array) — this never
    copies. Shared by :func:`mp_step_one_window` (copy-then-step, safe for any
    caller) and :func:`_run_mp_loop_one_window`'s hot loop (which owns one
    private array for the whole trial and would otherwise re-copy the full
    ``(F, T)`` residual on every step just to change a ``(F, W)`` slice).
    """
    e_before = float(np.sum(resid**2))
    if not atoms or resid.size == 0:  # no-pipeline-cover: MP never runs with an empty dictionary or window
        return MpStepResult(
            matching=None,
            energy_before=e_before,
            energy_after=e_before,
            delta_energy=0.0,
        )

    best: tuple[PlacementScore, Matching, np.ndarray] | None = None
    for template_id, basis, _width in atoms:
        placed = _best_placement_on_window(
            basis,
            resid,
            template_id=template_id,
            window_id=window_id,
            normalize=normalize,
            occupied_intervals=occupied_intervals,
            gpu_id=gpu_id,
            pc1_screen=pc1_screen,
            screen_top=screen_top,
            max_relative_distance=max_relative_distance,
        )
        if placed is None:
            continue
        if best is None or _placement_beats(placed[0], best[0]):
            best = placed

    if best is None or best[0][0] <= 0.0:
        return MpStepResult(
            matching=None,
            energy_before=e_before,
            energy_after=e_before,
            delta_energy=0.0,
        )

    _score, matching, basis = best
    if not _deflate_matching(resid, matching, basis):  # no-pipeline-cover: placements always fit
        return MpStepResult(
            matching=None,
            energy_before=e_before,
            energy_after=e_before,
            delta_energy=0.0,
        )
    e_after = float(np.sum(resid**2))
    return MpStepResult(
        matching=matching,
        energy_before=e_before,
        energy_after=e_after,
        delta_energy=e_before - e_after,
    )


def mp_step_one_window(  # no-pipeline-cover: single-window API used by unit tests
    residual: np.ndarray,
    dictionary: Dictionary,
    *,
    window_id: str,
    normalize: bool = True,
    occupied_intervals: list[tuple[int, int]] | None = None,
    gpu_id: int = 0,
    pc1_screen: bool = False,
    screen_top: int = 8,
    max_relative_distance: float | None = None,
) -> tuple[np.ndarray, MpStepResult]:
    """One Mallat step on a single window residual (F×T).

    Copy-safe public API: copies ``residual``, then mutates the private copy
    via :func:`_mp_step_one_window_inplace`.
    """
    resid = np.array(residual, dtype=np.float64, copy=True)
    step = _mp_step_one_window_inplace(
        resid,
        _dictionary_arrays(dictionary),
        window_id=window_id,
        normalize=normalize,
        occupied_intervals=occupied_intervals,
        gpu_id=gpu_id,
        pc1_screen=pc1_screen,
        screen_top=screen_top,
        max_relative_distance=max_relative_distance,
    )
    return resid, step


def mp_step(
    residual_stack: dict[str, np.ndarray] | list[Window],
    dictionary: Dictionary,
    *,
    normalize: bool = True,
    occupied_intervals: IntervalMap | None = None,
    pc1_screen: bool = False,
    screen_top: int = 8,
    max_relative_distance: float | None = None,
) -> tuple[dict[str, np.ndarray], MpStepResult]:
    """One Mallat MP iteration over the residual stack (global best atom).

    Candidate placements intersecting a half-open interval in
    ``occupied_intervals[window_id]`` are excluded. The winning ``(template,
    window, shift)`` triple is the one with the highest explained-energy
    *fraction* of its own candidate patch (scale-free across atoms of
    different norm and across patches of different energy — see
    :data:`PlacementScore` / :func:`_best_placement_on_window`), not the one
    with the largest absolute amplitude or explained energy; ties in that
    fraction fall back to absolute explained energy. Placements whose
    unexplained energy fraction exceeds ``max_relative_distance`` are rejected.
    """
    resid = _as_residual_map(residual_stack)
    e_before = residual_energy(resid)

    if len(dictionary) == 0 or not resid:  # no-pipeline-cover: MP never runs with an empty dictionary or stack
        return resid, MpStepResult(
            matching=None,
            energy_before=e_before,
            energy_after=e_before,
            delta_energy=0.0,
        )

    best: tuple[PlacementScore, Matching, np.ndarray] | None = None
    for tmpl in dictionary:
        basis = tmpl.basis
        for wid, r in resid.items():
            placed = _best_placement_on_window(
                basis,
                r,
                template_id=tmpl.template_id,
                window_id=wid,
                normalize=normalize,
                occupied_intervals=(occupied_intervals or {}).get(wid, []),
                pc1_screen=pc1_screen,
                screen_top=screen_top,
                max_relative_distance=max_relative_distance,
            )
            if placed is None:
                continue
            if best is None or _placement_beats(placed[0], best[0]):
                best = placed

    if best is None or best[0][0] <= 0.0:
        return resid, MpStepResult(
            matching=None,
            energy_before=e_before,
            energy_after=e_before,
            delta_energy=0.0,
        )

    _score, matching, basis = best
    r = resid[matching.window_id]
    if not _deflate_matching(r, matching, basis):  # no-pipeline-cover: placements always fit
        return resid, MpStepResult(
            matching=None,
            energy_before=e_before,
            energy_after=e_before,
            delta_energy=0.0,
        )
    e_after = residual_energy(resid)
    return resid, MpStepResult(
        matching=matching,
        energy_before=e_before,
        energy_after=e_after,
        delta_energy=e_before - e_after,
    )


@dataclass
class MpLoopResult:
    residual_stack: dict[str, np.ndarray]
    matchings: list[Matching] = field(default_factory=list)
    energies: list[float] = field(default_factory=list)
    stopped_reason: str = ""


def _run_mp_loop_global(
    resid: dict[str, np.ndarray],
    dictionary: Dictionary,
    *,
    energy_floor: float | None,
    eps_step_fraction: float,
    max_repeats: int | None,
    normalize: bool,
    disallow_overlap: bool,
    occupied: IntervalMap,
    show_progress: bool | None,
    progress_desc: str,
    pc1_screen: bool = False,
    screen_top: int = 8,
    max_relative_distance: float | None = None,
) -> MpLoopResult:

    template_widths = {template.template_id: template.width for template in dictionary}
    matchings: list[Matching] = []
    energies = [residual_energy(resid)]
    reason = "no_improvement"
    first_delta: float | None = None

    pbar = tqdm(
        total=None,
        desc=progress_desc,
        unit="atom",
        leave=False,
        disable=not progress_enabled(show_progress),
        dynamic_ncols=True,
    )
    try:
        while True:
            resid, step = mp_step(
                resid,
                dictionary,
                normalize=normalize,
                occupied_intervals=occupied,
                pc1_screen=pc1_screen,
                screen_top=screen_top,
                max_relative_distance=max_relative_distance,
            )
            if step.matching is None or step.delta_energy <= 0.0:
                reason = "no_improvement"
                break
            if first_delta is None:
                first_delta = step.delta_energy
            elif step.delta_energy < eps_step_fraction * first_delta:
                reason = "eps_step_fraction"
                break
            matchings.append(step.matching)
            energies.append(step.energy_after)
            if disallow_overlap:
                width = template_widths[step.matching.template_id]
                occupied.setdefault(step.matching.window_id, []).append(
                    (step.matching.t_start, step.matching.t_start + width)
                )
            pbar.update(1)

            loop_stats.incr("mp.step.accepted")
            if energy_floor is not None and step.energy_after <= energy_floor:  # no-pipeline-cover: --energy-floor is not exposed by the CLIs
                reason = "energy_floor"
                break
            if max_repeats is not None and len(matchings) >= int(max_repeats):
                reason = "max_repeats"
                break
    finally:
        pbar.close()

    return MpLoopResult(
        residual_stack=resid,
        matchings=matchings,
        energies=energies,
        stopped_reason=reason,
    )


def _run_mp_loop_one_window(
    window_id: str,
    residual: np.ndarray,
    dictionary: Dictionary,
    *,
    energy_floor: float | None,
    eps_step_fraction: float,
    max_repeats: int | None,
    normalize: bool,
    disallow_overlap: bool,
    occupied_intervals: list[tuple[int, int]] | None,
    gpu_id: int = 0,
    copy_residual: bool = True,
    max_relative_distance: float | None = None,
) -> tuple[str, np.ndarray, list[Matching], list[float], str]:
    """Mallat loop on one window; returns picklable tuple for process pools.

    Mutates a private ``resid`` via :func:`_mp_step_one_window_inplace` and
    converts the dictionary once up front.

    When ``copy_residual`` is True (default), copies ``residual`` on entry so
    any caller is safe. :func:`run_mp_loop` already deep-copies the stack, so
    :func:`_run_mp_loop_per_window` passes ``copy_residual=False`` to avoid a
    second full ``(F, T)`` memcpy per window per trial (same math).
    """
    if copy_residual:
        resid = np.array(residual, dtype=np.float64, copy=True)  # no-pipeline-cover: run_mp_loop passes copy_residual=False
    else:
        resid = np.asarray(residual, dtype=np.float64)
    atoms = _dictionary_arrays(dictionary)
    occupied = list(occupied_intervals or [])
    template_widths = {t.template_id: t.width for t in dictionary}
    matchings: list[Matching] = []
    energies = [float(np.sum(resid**2))]
    reason = "no_improvement"
    first_delta: float | None = None

    while True:
        step = _mp_step_one_window_inplace(
            resid,
            atoms,
            window_id=window_id,
            normalize=normalize,
            occupied_intervals=occupied,
            gpu_id=gpu_id,
            max_relative_distance=max_relative_distance,
        )
        if step.matching is None or step.delta_energy <= 0.0:
            reason = "no_improvement"
            break
        if first_delta is None:
            first_delta = step.delta_energy
        elif step.delta_energy < eps_step_fraction * first_delta:
            reason = "eps_step_fraction"
            break
        matchings.append(step.matching)
        energies.append(step.energy_after)
        if disallow_overlap:
            width = template_widths[step.matching.template_id]
            occupied.append(
                (step.matching.t_start, step.matching.t_start + width)
            )

        loop_stats.incr("mp.step.accepted")
        if energy_floor is not None and step.energy_after <= energy_floor:  # no-pipeline-cover: --energy-floor is not exposed by the CLIs
            reason = "energy_floor"
            break
        if max_repeats is not None and len(matchings) >= int(max_repeats):
            reason = "max_repeats"
            break

    return window_id, resid, matchings, energies, reason


def _per_window_job(
    payload: tuple,
) -> tuple[str, np.ndarray, list[Matching], list[float], str]:
    (
        window_id,
        residual,
        atoms,
        energy_floor,
        eps_step_fraction,
        max_repeats,
        normalize,
        disallow_overlap,
        occupied_intervals,
        gpu_id,
        copy_residual,
        max_relative_distance,
    ) = payload
    dictionary = Dictionary()
    for template_id, basis, _width in atoms:
        # basis is (k,F,W); Template accepts 2D or 3D.
        data = basis[0] if basis.shape[0] == 1 else basis
        dictionary.add(Template(template_id=template_id, data=data))
    return _run_mp_loop_one_window(
        window_id,
        residual,
        dictionary,
        energy_floor=energy_floor,
        eps_step_fraction=eps_step_fraction,
        max_repeats=max_repeats,
        normalize=normalize,
        disallow_overlap=disallow_overlap,
        occupied_intervals=occupied_intervals,
        gpu_id=int(gpu_id),
        copy_residual=bool(copy_residual),
        max_relative_distance=max_relative_distance,
    )


def _run_mp_loop_per_window(
    resid: dict[str, np.ndarray],
    dictionary: Dictionary,
    *,
    energy_floor: float | None,
    eps_step_fraction: float,
    max_repeats: int | None,
    normalize: bool,
    disallow_overlap: bool,
    occupied: IntervalMap,
    n_workers: int,
    show_progress: bool | None,
    progress_desc: str,
    max_relative_distance: float | None = None,
) -> MpLoopResult:

    atoms = _dictionary_arrays(dictionary)
    window_ids = sorted(resid.keys())
    # Workers deflate ``resid`` in place (copy_residual=False), so read the
    # starting energy before they run.
    e0 = residual_energy(resid)

    n_gpus = gpu_device_count() if cupy_available() else 0
    jobs = [
        (
            wid,
            resid[wid],
            atoms,
            energy_floor,
            eps_step_fraction,
            max_repeats,
            normalize,
            disallow_overlap,
            list(occupied.get(wid, [])),
            (i % n_gpus) if n_gpus else 0,
            False,  # copy_residual: run_mp_loop already copied the stack
            max_relative_distance,
        )
        for i, wid in enumerate(window_ids)
    ]

    results: dict[str, tuple[np.ndarray, list[Matching], list[float], str]] = {}
    workers = max(1, int(n_workers))
    # Threads when Numba (nogil) or CuPy owns the hot loop; ProcessPool when the
    # score path is pure Python / NumPy so workers are not GIL-serialized.
    prefer_threads = numba_available() or cupy_available()
    pool_cls = ThreadPoolExecutor if prefer_threads else ProcessPoolExecutor

    if workers <= 1 or len(jobs) <= 1:
        iterator = jobs
        pbar = tqdm(
            total=len(jobs),
            desc=progress_desc,
            unit="win",
            leave=False,
            disable=not progress_enabled(show_progress),
            dynamic_ncols=True,
        )
        try:
            for job in iterator:
                wid, arr, matchings, energies, reason = _per_window_job(job)
                results[wid] = (arr, matchings, energies, reason)
                pbar.update(1)
        finally:
            pbar.close()
    else:
        pbar = tqdm(
            total=len(jobs),
            desc=progress_desc,
            unit="win",
            leave=False,
            disable=not progress_enabled(show_progress),
            dynamic_ncols=True,
        )
        try:
            with pool_cls(max_workers=min(workers, len(jobs))) as pool:
                futures = {pool.submit(_per_window_job, job): job[0] for job in jobs}
                for fut in as_completed(futures):
                    wid, arr, matchings, energies, reason = fut.result()
                    results[wid] = (arr, matchings, energies, reason)
                    pbar.update(1)
        finally:
            pbar.close()

    out_resid: dict[str, np.ndarray] = {}
    all_matchings: list[Matching] = []
    reasons: list[str] = []
    for wid in window_ids:
        arr, matchings, _energies, reason = results[wid]
        out_resid[wid] = arr
        # Stable order within window: placement order from that window's Mallat path.
        all_matchings.extend(matchings)
        reasons.append(reason)

    # Re-sort globally by (window_id, t_start) for reproducible catalogs.
    all_matchings.sort(key=lambda m: (m.window_id, m.t_start, m.template_id))

    e1 = residual_energy(out_resid)
    if all(r == "no_improvement" for r in reasons) and not all_matchings:
        stopped = "no_improvement"
    elif any(r == "energy_floor" for r in reasons):
        stopped = "energy_floor"  # no-pipeline-cover: --energy-floor is not exposed by the CLIs
    elif any(r == "max_repeats" for r in reasons):
        stopped = "max_repeats"
    elif any(r == "eps_step_fraction" for r in reasons):
        stopped = "eps_step_fraction"
    else:
        stopped = "no_improvement" if not all_matchings else "completed"

    return MpLoopResult(
        residual_stack=out_resid,
        matchings=all_matchings,
        energies=[e0, e1] if all_matchings else [e0],
        stopped_reason=stopped,
    )


def run_mp_loop(
    residual_stack: dict[str, np.ndarray] | list[Window],
    dictionary: Dictionary,
    *,
    energy_floor: float | None = None,
    eps_step_fraction: float = defaults.EPS_STEP_FRACTION,
    max_repeats: int | None = None,
    normalize: bool = True,
    disallow_overlap: bool = False,
    occupied_intervals: IntervalMap | None = None,
    show_progress: bool | None = None,
    progress_desc: str = "mp",
    mp_mode: str | None = None,
    n_workers: int | None = None,
    pc1_screen: bool = False,
    screen_top: int = 8,
    max_relative_distance: float | None = None,
) -> MpLoopResult:
    """Repeat Mallat steps until a stopping rule fires.

    ``mp_mode``:
      - ``global``: classical stack-wide Mallat (one best atom per step).
      - ``per_window``: independent Mallat per window (parallelizable); each
        window uses its own first-step ΔE for ``eps_step_fraction``. Atoms
        never straddle window boundaries.

    When any atom has ``n_components > 1``, mode is forced to ``global``
    (per-window subspace MP is not implemented yet).

    Spans passed in ``occupied_intervals`` are always off limits, whatever
    ``disallow_overlap`` says; ``disallow_overlap`` controls only whether each
    accepted match reserves its own span against later steps.

    Stop when a step's ΔE falls below ``eps_step_fraction`` times the first
    step's ΔE (per window in ``per_window`` mode), when no improving placement
    remains, when ``energy_floor`` is reached, or when ``max_repeats`` accepted
    matchings have been collected (per window in ``per_window`` mode). The
    failing eps / no-improvement step is not applied; the ``max_repeats`` step
    is applied.

    ``max_relative_distance`` (``tau`` on ``[0, 1]``, ``None`` = off) rejects
    placements that leave more than ``tau`` of the patch energy unexplained;
    MP then falls through to the next-best qualifying shift.
    """
    if eps_step_fraction <= 0.0:
        raise ValueError(f"eps_step_fraction must be > 0, got {eps_step_fraction}")
    if max_relative_distance is not None and not (
        0.0 <= float(max_relative_distance) <= 1.0
    ):
        raise ValueError(  # no-pipeline-cover: the CLIs check the range first
            "max_relative_distance must be in [0, 1] when set, got "
            f"{max_relative_distance}"
        )
    if max_repeats is not None and int(max_repeats) < 1:
        raise ValueError(f"max_repeats must be >= 1 when set, got {max_repeats}")
    mode = resolve_mp_mode(mp_mode)
    if dictionary.max_n_components > 1 and mode != "global":
        mode = "global"

    resid = copy_residual_stack(residual_stack)
    occupied: IntervalMap = {
        window_id: list(intervals)
        for window_id, intervals in (occupied_intervals or {}).items()
    }


    t0 = time.perf_counter()
    if mode == "global":
        result = _run_mp_loop_global(
            resid,
            dictionary,
            energy_floor=energy_floor,
            eps_step_fraction=eps_step_fraction,
            max_repeats=max_repeats,
            normalize=normalize,
            disallow_overlap=disallow_overlap,
            occupied=occupied,
            show_progress=show_progress,
            progress_desc=progress_desc,
            pc1_screen=pc1_screen,
            screen_top=screen_top,
            max_relative_distance=max_relative_distance,
        )
    else:
        workers = resolve_mp_workers(n_workers)
        result = _run_mp_loop_per_window(
            resid,
            dictionary,
            energy_floor=energy_floor,
            eps_step_fraction=eps_step_fraction,
            max_repeats=max_repeats,
            normalize=normalize,
            disallow_overlap=disallow_overlap,
            occupied=occupied,
            n_workers=workers,
            show_progress=show_progress,
            progress_desc=progress_desc,
            max_relative_distance=max_relative_distance,
        )
    elapsed = time.perf_counter() - t0

    loop_stats.incr(f"mp.loop.{mode}")
    loop_stats.record("mp.loop_atoms", len(result.matchings))
    loop_stats.record("mp.loop_windows", len(result.residual_stack))
    loop_stats.record_seconds("mp.loop_wall_s", elapsed)
    # Tag by progress_desc when present (peel / seed mp / avg mp / refine mp).
    tag = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in (progress_desc or "mp"))
    loop_stats.record_seconds(f"mp.loop_wall_s.{tag}", elapsed)
    return result


def single_template_dictionary(template: Template | np.ndarray, template_id: int = 0) -> Dictionary:  # no-pipeline-cover: helper used by unit tests
    d = Dictionary()
    if isinstance(template, Template):
        d.add(template)
    else:
        d.add(Template(template_id=template_id, data=np.asarray(template, dtype=np.float64)))
    return d

loop_stats = __import__('sys').modules[__name__]
