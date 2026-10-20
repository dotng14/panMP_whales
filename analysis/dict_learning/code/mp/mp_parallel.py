"""Parallel helpers for Matching Pursuit (candidate trials and window loops)."""

from __future__ import annotations

import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Iterable, Sequence, TypeVar

from ..core import defaults

T = TypeVar("T")
R = TypeVar("R")


def _process_pool_context() -> mp.context.BaseContext:
    """Use spawn to avoid fork-after-threads deadlocks (Numba/OpenMP)."""
    return mp.get_context("spawn")


def available_cpu_count() -> int:
    """Max CPUs available to this process (affinity-aware when possible)."""
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
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


def allocate_nested_workers(
    n_outer_tasks: int,
    *,
    total_workers: int | None = None,
) -> tuple[int, int]:
    """Split CPUs between outer tasks and nested per-task workers.

    Returns ``(outer_workers, inner_workers)`` such that
    ``outer_workers * inner_workers <= total`` and outer work saturates first.
    """
    total = resolve_mp_workers(total_workers)
    n_outer = max(1, int(n_outer_tasks))
    if n_outer >= total:
        return total, 1
    outer = n_outer
    inner = max(1, total // outer)
    return outer, inner


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


def prefer_candidate_threads(explicit: bool | None = None) -> bool:
    """Whether GPL candidate trials should use threads instead of processes.

    Threads share the residual by reference (``run_mp_loop`` copies internally),
    avoiding per-round ``spawn`` ProcessPool startup and residual pickling — a
    large win when ``n_windows`` grows. Numba ``score_shifts`` releases the GIL.

    Resolution: explicit arg → ``DICT_LEARNING_MP_CANDIDATE_POOL=threads|process``
    → default **threads**.
    """
    if explicit is not None:
        return bool(explicit)
    raw = os.environ.get("DICT_LEARNING_MP_CANDIDATE_POOL", "").strip().lower()
    if raw in ("threads", "thread", "1", "true", "yes"):
        return True
    if raw in ("process", "processes", "0", "false", "no"):
        return False
    return True


def map_parallel(
    fn: Callable[[T], R],
    items: Sequence[T],
    *,
    n_workers: int | None = None,
    prefer_threads: bool = False,
    initializer: Callable[..., Any] | None = None,
    initargs: tuple[Any, ...] = (),
) -> list[R]:
    """Map ``fn`` over ``items``, preserving order.

    With ``n_workers <= 1`` or a single item, runs sequentially (easier debugging
    and avoids process spawn overhead in unit tests). Sequential mode ignores
    ``initializer`` / ``initargs`` — callers should keep shared data on each item.

    Candidate trials default to threads via :func:`prefer_candidate_threads`
    (shared residual, no spawn). Pass ``prefer_threads=False`` with an
    ``initializer`` / ``initargs`` to use a ProcessPool and share the residual
    once per worker instead of pickling it into every task.
    """
    workers = resolve_mp_workers(n_workers)
    if workers <= 1 or len(items) <= 1:
        return [fn(item) for item in items]

    pool_cls = ThreadPoolExecutor if prefer_threads else ProcessPoolExecutor
    results: list[R | None] = [None] * len(items)
    pool_kwargs: dict[str, Any] = {"max_workers": min(workers, len(items))}
    if not prefer_threads:
        pool_kwargs["mp_context"] = _process_pool_context()
    if initializer is not None:
        pool_kwargs["initializer"] = initializer
        pool_kwargs["initargs"] = initargs
    with pool_cls(**pool_kwargs) as pool:
        futures = {pool.submit(fn, item): i for i, item in enumerate(items)}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    return [r for r in results]  # type: ignore[return-value]


def map_parallel_unordered_best(
    fn: Callable[[T], R | None],
    items: Iterable[T],
    *,
    n_workers: int | None = None,
    key: Callable[[R], Any],
    prefer_threads: bool = False,
    initializer: Callable[..., Any] | None = None,
    initargs: tuple[Any, ...] = (),
) -> tuple[list[R], R | None]:
    """Evaluate ``fn`` on each item; return (all non-None results, best by ``key``).

    ``key`` is minimized (lower is better). Ties keep the first-seen result in
    input order when running sequentially; with a pool, callers should make
    ``key`` include a stable secondary (e.g. segment_id) for determinism.
    """
    item_list = list(items)
    workers = resolve_mp_workers(n_workers)
    if workers <= 1 or len(item_list) <= 1:
        collected: list[R] = []
        best: R | None = None
        best_key = None
        for item in item_list:
            out = fn(item)
            if out is None:
                continue
            collected.append(out)
            k = key(out)
            if best is None or k < best_key:  # type: ignore[operator]
                best = out
                best_key = k
        return collected, best

    pool_cls = ThreadPoolExecutor if prefer_threads else ProcessPoolExecutor
    collected = []
    best = None
    best_key = None
    pool_kwargs: dict[str, Any] = {"max_workers": min(workers, len(item_list))}
    if not prefer_threads:
        pool_kwargs["mp_context"] = _process_pool_context()
    if initializer is not None:
        pool_kwargs["initializer"] = initializer
        pool_kwargs["initargs"] = initargs
    with pool_cls(**pool_kwargs) as pool:
        futures = [pool.submit(fn, item) for item in item_list]
        for fut in as_completed(futures):
            out = fut.result()
            if out is None:
                continue
            collected.append(out)
            k = key(out)
            if best is None or k < best_key:  # type: ignore[operator]
                best = out
                best_key = k
    return collected, best
