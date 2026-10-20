"""Optional tqdm progress bars for dictionary-learning loops.

``show_progress=None`` (default) enables bars when stderr is a TTY and
``TQDM_DISABLE`` is unset. Pass ``True``/``False`` to force on/off.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Iterable, Iterator
from typing import Any, TypeVar

try:
    from tqdm.auto import tqdm as _tqdm_impl
except ImportError:  # pragma: no cover
    _tqdm_impl = None  # type: ignore[assignment]


class _NullPbar:
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
    if _tqdm_impl is None:
        return _NullPbar()
    return _tqdm_impl(*args, **kwargs)


__all__ = ["progress_enabled", "iter_progress", "range_progress", "tqdm"]

T = TypeVar("T")


def progress_enabled(show_progress: bool | None = None) -> bool:
    if show_progress is not None:
        return bool(show_progress) and _tqdm_impl is not None
    if _tqdm_impl is None:
        return False
    if os.environ.get("TQDM_DISABLE", "").strip():
        return False
    try:
        return bool(sys.stderr.isatty())
    except Exception:  # noqa: BLE001
        return False


def iter_progress(
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


def range_progress(
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
