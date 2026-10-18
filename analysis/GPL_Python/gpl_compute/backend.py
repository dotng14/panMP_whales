"""
Array-module abstraction: returns cupy on a specific GPU, or numpy when CUDA
is unavailable or GPL_COMPUTE is not set to "cuda".

Usage
-----
from analysis.GPL_Python.gpl_compute import get_array_module, numpy_from

xp = get_array_module(gpu_id)      # cupy or numpy
arr = xp.array(data)               # works with both
back = numpy_from(xp, arr)         # always returns a numpy array
"""
from __future__ import annotations

import os
import numpy as np

try:
    import cupy as cp  # type: ignore[import]
    _CUPY_AVAILABLE = True
except ModuleNotFoundError:
    cp = None  # type: ignore[assignment]
    _CUPY_AVAILABLE = False

_BACKEND = os.environ.get("GPL_COMPUTE", "cpu").lower()

if _CUPY_AVAILABLE and _BACKEND == "cuda" and cp is not None:
    # Reuse freed VRAM blocks instead of calling cudaMalloc each time.
    cp.get_default_memory_pool().set_limit(size=2 * 1024**3)  # 2 GB per GPU


def get_array_module(gpu_id: int = 0):
    """Return cupy (with the given device active) or numpy."""
    if _BACKEND == "cuda" and _CUPY_AVAILABLE and cp is not None:
        cp.cuda.Device(gpu_id).use()
        return cp
    return np


def numpy_from(xp, arr) -> np.ndarray:
    """Convert a cupy or numpy array to a plain numpy array."""
    if xp is np:
        return np.asarray(arr)
    return cp.asnumpy(arr)
