"""Shared correlation kernel: score a template at every time shift in a window."""

from __future__ import annotations

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


@njit(cache=True, nogil=True)
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


def _score_shifts_numpy(
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
    if not _CUPY_IMPORTABLE or cp is None:
        _CUPY_USABLE = False
        _CUPY_DEVICE_COUNT = 0
        return False
    try:
        n = int(cp.cuda.runtime.getDeviceCount())
        if n <= 0:
            _CUPY_USABLE = False
            _CUPY_DEVICE_COUNT = 0
            return False
        with cp.cuda.Device(0):
            a = cp.arange(8, dtype=cp.float64)
            _ = float(cp.asnumpy((a * a).sum()))
        _CUPY_DEVICE_COUNT = n
        _CUPY_USABLE = True
    except Exception:
        _CUPY_USABLE = False
        _CUPY_DEVICE_COUNT = 0
    return _CUPY_USABLE


def cupy_available() -> bool:
    return _probe_cupy()


def gpu_device_count() -> int:
    _probe_cupy()
    return int(_CUPY_DEVICE_COUNT)


def resolve_mp_compute(compute: str | None = None) -> ComputeBackend:
    """Resolve ``cpu`` / ``cuda`` / ``auto`` from arg → env → defaults."""
    from ..core import defaults

    raw = compute
    if raw is None or str(raw).strip() == "":
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


def gpu_beneficial(f: int, w: int, t: int) -> bool:
    """Heuristic: GPU when correlation work exceeds transfer overhead."""
    if t < w:
        return False
    ops = int(f) * int(w) * int(t - w + 1)
    return ops >= _GPU_MIN_OPS


def _score_shifts_cupy(
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
        except Exception:
            for fi in range(f):
                # correlate(a, v) valid == convolve(a, v[::-1]) valid
                scores += cp.convolve(g_resid[fi], g_tmpl[fi, ::-1], mode="valid")
        if normalize:
            denom = float(cp.asnumpy(cp.sum(g_tmpl * g_tmpl)))
            if denom <= 0.0:
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
        raise ValueError("template and residual must be 2-D (F×T)")
    if tmpl.shape[0] != resid.shape[0]:
        raise ValueError(
            f"Frequency mismatch: template {tmpl.shape[0]} vs residual {resid.shape[0]}"
        )
    f, w = tmpl.shape
    t = resid.shape[1]
    if t < w:
        return np.zeros(0, dtype=np.float64)

    backend = resolve_mp_compute(compute)
    want_gpu = False
    if backend == "cuda":
        want_gpu = cupy_available()
    elif backend == "auto":
        want_gpu = cupy_available() and gpu_beneficial(f, w, t)

    out: np.ndarray | None = None
    if want_gpu:
        try:
            out = _score_shifts_cupy(
                tmpl, resid, normalize=normalize, gpu_id=gpu_id
            )
        except Exception:
            # Broken toolkit / OOM → fall through to CPU.
            out = None

    if out is None:
        want_numba = _NUMBA_AVAILABLE if use_numba is None else bool(use_numba)
        if want_numba and _NUMBA_AVAILABLE:
            out = _score_shifts_numba(tmpl, resid, bool(normalize))
        else:
            out = _score_shifts_numpy(tmpl, resid, normalize=normalize)

    try:
        from . import loop_stats

        loop_stats.incr("score_shifts.calls")
    except Exception:
        pass
    return out


def best_shift(
    template: np.ndarray,
    residual: np.ndarray,
    *,
    normalize: bool = True,
) -> tuple[int, float] | None:
    scores = score_shifts(template, residual, normalize=normalize)
    if scores.size == 0:
        return None
    idx = int(np.argmax(np.abs(scores)))
    return idx, float(scores[idx])


def numba_available() -> bool:
    return _NUMBA_AVAILABLE
