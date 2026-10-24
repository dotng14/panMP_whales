"""Microbenchmarks for Matching Pursuit hot paths.

Run::

    python -m analysis.dict_learning.experiments.bench_mp
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from ..code.mp import run_mp_loop, single_template_dictionary
from ..code.core import Template
from ..code.mp import numba_available, score_shifts
from ..code.learn.seed_from_gpl import Seed
from ..code.core import Segment


def _synth_stack(
    *,
    n_windows: int = 4,
    f: int = 32,
    t: int = 400,
    w: int = 16,
    n_events: int = 3,
    seed: int = 0,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    rng = np.random.default_rng(seed)
    tmpl = rng.standard_normal((f, w))
    tmpl /= max(float(np.linalg.norm(tmpl)), 1e-12)
    stack: dict[str, np.ndarray] = {}
    for i in range(n_windows):
        resid = 0.05 * rng.standard_normal((f, t))
        for _ in range(n_events):
            t0 = int(rng.integers(0, max(1, t - w)))
            resid[:, t0 : t0 + w] += rng.uniform(1.5, 3.0) * tmpl
        stack[f"w{i}"] = resid
    return stack, tmpl


def _time(fn, *, repeats: int = 3) -> float:
    # Warmup
    fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(min(times))


def bench_score_shifts(*, repeats: int = 5) -> dict[str, Any]:
    stack, tmpl = _synth_stack(n_windows=1, f=48, t=800, w=20)
    resid = next(iter(stack.values()))

    def numpy_once():
        score_shifts(tmpl, resid, normalize=True, use_numba=False, compute="cpu")

    out: dict[str, Any] = {
        "numba_available": numba_available(),
        "shape": {"F": tmpl.shape[0], "W": tmpl.shape[1], "T": resid.shape[1]},
        "numpy_s": _time(numpy_once, repeats=repeats),
    }
    if numba_available():
        score_shifts(tmpl, resid, normalize=True, use_numba=True, compute="cpu")

        def numba_once():
            score_shifts(tmpl, resid, normalize=True, use_numba=True, compute="cpu")

        out["numba_s"] = _time(numba_once, repeats=repeats)
        out["speedup_vs_numpy"] = out["numpy_s"] / max(out["numba_s"], 1e-12)

    from .mp_parallel import available_cpu_count
    from .score_shifts import cupy_available, gpu_device_count

    out["cpus"] = available_cpu_count()
    out["cupy_available"] = cupy_available()
    out["gpus"] = gpu_device_count()
    if cupy_available():
        # Warm + time forced CUDA path on a large patch.
        big_t, big_w = _synth_stack(n_windows=1, f=96, t=4000, w=24)
        big_r = next(iter(big_t.values()))

        def cuda_once():
            score_shifts(big_w, big_r, normalize=True, compute="cuda")

        try:
            cuda_once()
            out["cuda_s"] = _time(cuda_once, repeats=max(2, repeats // 2))
        except Exception as exc:  # pragma: no cover
            out["cuda_error"] = str(exc)
    return out


def bench_mp_loop(*, repeats: int = 3) -> dict[str, Any]:
    stack, tmpl = _synth_stack(n_windows=8, f=32, t=300, w=12)
    dictionary = single_template_dictionary(Template(template_id=0, data=tmpl))

    def global_once():
        return run_mp_loop(
            stack,
            dictionary,
            eps_step_fraction=0.2,
            mp_mode="global",
            n_workers=1,
            show_progress=False,
        )

    def per_window_once():
        return run_mp_loop(
            stack,
            dictionary,
            eps_step_fraction=0.2,
            mp_mode="per_window",
            n_workers=4,
            show_progress=False,
        )

    g = global_once()
    p = per_window_once()
    return {
        "n_windows": len(stack),
        "global_s": _time(global_once, repeats=repeats),
        "per_window_s": _time(per_window_once, repeats=repeats),
        "global_atoms": len(g.matchings),
        "per_window_atoms": len(p.matchings),
    }


def bench_candidate_trials(*, repeats: int = 2) -> dict[str, Any]:
    stack, tmpl = _synth_stack(n_windows=2, f=24, t=200, w=10, n_events=2)
    # Build a few segment candidates from the template and a weaker variant.
    segs = []
    for i, scale in enumerate((1.0, 0.7, 1.1, 0.5)):
        segs.append(
            Segment(
                segment_id=f"c{i}",
                data=scale * tmpl,
                window_id="w0",
                t_start=0,
                t_end=tmpl.shape[1],
                global_start_sample=1,
                global_end_sample=2,
                detection_index=i,
            )
        )

    def serial():
        return Seed.from_gpl(
            stack,
            segs,
            eps_step_fraction=0.2,
            min_repeats=1,
            n_workers=1,
            mp_mode="per_window",
            show_progress=False,
        )

    def parallel():
        return Seed.from_gpl(
            stack,
            segs,
            eps_step_fraction=0.2,
            min_repeats=1,
            n_workers=4,
            mp_mode="per_window",
            show_progress=False,
        )

    s = serial()
    p = parallel()
    return {
        "n_candidates": len(segs),
        "serial_s": _time(serial, repeats=repeats),
        "parallel_s": _time(parallel, repeats=repeats),
        "serial_selected": s.get_selected_segment_id(),
        "parallel_selected": p.get_selected_segment_id(),
    }


def main() -> None:
    print("=== score_shifts ===")
    print(bench_score_shifts())
    print("=== run_mp_loop ===")
    print(bench_mp_loop())
    print("=== candidate trials ===")
    print(bench_candidate_trials())


if __name__ == "__main__":
    main()
