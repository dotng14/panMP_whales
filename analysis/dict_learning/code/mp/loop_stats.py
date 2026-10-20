"""Lightweight counters for nested MP / incremental-build loops.

Enable with ``DICT_LEARNING_LOOP_STATS=1`` or :func:`enable`. Thread-safe enough
for ThreadPool workers (per-thread counters merged on :func:`snapshot`).
"""

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


def timing_sum(name: str) -> float:
    """Sum of recorded seconds for ``name`` (0 if none)."""
    with _LOCK:
        return float(sum(_TIMINGS.get(name, ())))


@contextmanager
def timed(name: str) -> Iterator[None]:
    """Context manager that records wall time under ``name`` when enabled."""
    if not enabled():
        yield
        return
    t0 = time.perf_counter()
    try:
        yield
    finally:
        record_seconds(name, time.perf_counter() - t0)


def snapshot() -> dict[str, Any]:
    with _LOCK:
        series_summary: dict[str, Any] = {}
        for name, values in _SERIES.items():
            if not values:
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
            if not values:
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


def format_report(data: dict[str, Any] | None = None) -> str:
    data = data if data is not None else snapshot()
    lines = ["=== loop stats ===", "counts:"]
    for k, v in data.get("counts", {}).items():
        lines.append(f"  {k}: {v}")
    lines.append("series:")
    for k, s in data.get("series", {}).items():
        lines.append(
            f"  {k}: n={s['n']} sum={s['sum']} min={s['min']} "
            f"max={s['max']} mean={s['mean']:.2f} values={s['values']}"
        )
    lines.append("timings_s:")
    for k, s in data.get("timings_s", {}).items():
        lines.append(
            f"  {k}: n={s['n']} sum={s['sum_s']:.4f}s min={s['min_s']:.4f}s "
            f"max={s['max_s']:.4f}s mean={s['mean_s']:.4f}s"
        )
    return "\n".join(lines)


def dump_json(path: str | os.PathLike[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot(), f, indent=2)


@contextmanager
def collecting(*, reset_first: bool = True) -> Iterator[None]:
    """Context manager: enable counters, optionally reset, disable on exit."""
    prev = _ENABLED
    enable(True)
    if reset_first:
        reset()
    try:
        yield
    finally:
        enable(prev)


@dataclass
class LevelTotals:
    """Convenience rollup for the usual nesting."""

    stages: int = 0
    grow_rounds: int = 0
    candidate_trials: int = 0
    mp_loops: int = 0
    mp_atoms: int = 0
    refine_iters: int = 0
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_snapshot(cls, data: dict[str, Any] | None = None) -> LevelTotals:
        data = data if data is not None else snapshot()
        c = data.get("counts", {})
        s = data.get("series", {})
        return cls(
            stages=int(c.get("incremental.stage", 0)),
            grow_rounds=int(c.get("learn.outer_round", 0)),
            candidate_trials=int(c.get("learn.candidate_trial", 0)),
            mp_loops=int(c.get("mp.loop.calls", 0) or s.get("mp.loop_atoms", {}).get("n", 0)),
            mp_atoms=int(c.get("mp.loop_atoms.sum", 0) or s.get("mp.loop_atoms", {}).get("sum", 0)),
            refine_iters=int(c.get("refine.iter", 0)),
            extra=data,
        )
