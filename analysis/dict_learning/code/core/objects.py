"""Core objects for dictionary learning on GPL spectrogram windows.

Compatibility contract (subspace refactor)
------------------------------------------
* ``Template.data`` may be ``(F, W)`` (rank-1) or ``(k, F, W)`` (subspace).
  Use ``.basis`` for a view always shaped ``(k, F, W)`` and ``.as_rank1`` when
  ``k == 1`` is required (catalog / older call sites).
* ``Matching`` accepts either ``amplitude=`` (rank-1) or ``coeffs=`` (length
  ``k``). ``.amplitude`` stays a float field: signed coeff when ``k == 1``,
  else ``‖coeffs‖₂``. Energy removed is ``.energy`` (= ``‖coeffs‖₂²``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Window:
    """One spectrogram tile (GPL process window) with metadata."""

    window_id: str
    data: np.ndarray  # shape (F, T)
    start_sample: int  # file-global 1-based
    end_sample: int
    overlap: int  # STFT hop samples
    fftl: int = 0
    sample_freq: float = 0.0
    start_freq: float = 0.0
    end_freq: float = 0.0
    npy_path: Path | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_freq(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_time(self) -> int:
        return int(self.data.shape[1])


@dataclass
class Segment:
    """Spectrogram patch cropped from a window (one GPL detection overlap)."""

    segment_id: str
    data: np.ndarray  # (F, W)
    window_id: str
    t_start: int
    t_end: int  # exclusive
    global_start_sample: int
    global_end_sample: int
    detection_index: int = -1
    boundary_cross_flag: int = 0

    @property
    def energy(self) -> float:
        """Frobenius energy ‖U‖²_F of the patch."""
        return float(np.sum(np.asarray(self.data, dtype=np.float64) ** 2))


@dataclass
class Template:
    """Dictionary atom: rank-1 ``(F, W)`` or subspace ``(k, F, W)`` basis."""

    template_id: int
    data: np.ndarray  # (F, W) or (k, F, W)
    source_segment_id: str | None = None
    variances: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.data = np.asarray(self.data, dtype=np.float64)
        if self.data.ndim not in (2, 3):
            raise ValueError(
                f"Template.data must be (F, W) or (k, F, W), got shape {self.data.shape}"
            )
        if self.variances is not None:
            self.variances = np.asarray(self.variances, dtype=np.float64)

    @property
    def n_components(self) -> int:
        return 1 if self.data.ndim == 2 else int(self.data.shape[0])

    @property
    def basis(self) -> np.ndarray:
        """Orthonormal modes as ``(k, F, W)`` (view/copy-safe contiguous)."""
        if self.data.ndim == 2:
            return np.ascontiguousarray(self.data[np.newaxis, ...])
        return np.ascontiguousarray(self.data)

    @property
    def as_rank1(self) -> np.ndarray:
        """``(F, W)`` slice; requires ``n_components == 1``."""
        if self.n_components != 1:
            raise ValueError(
                f"as_rank1 requires n_components=1, got {self.n_components}"
            )
        return self.basis[0]

    @property
    def width(self) -> int:
        return int(self.data.shape[-1])

    @property
    def n_freq(self) -> int:
        return int(self.data.shape[-2])


@dataclass
class Matching:
    """One activation of a template in a window.

    Construct with ``amplitude=`` (rank-1) and/or ``coeffs=`` (length ``k``).
    """

    template_id: int
    window_id: str
    t_start: int
    amplitude: float | None = None
    coeffs: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.coeffs is not None:
            c = np.asarray(self.coeffs, dtype=np.float64).reshape(-1)
        elif self.amplitude is not None:
            c = np.asarray([float(self.amplitude)], dtype=np.float64)
        else:
            raise ValueError("Matching requires amplitude= or coeffs=")
        object.__setattr__(self, "coeffs", c)
        if c.size == 1:
            object.__setattr__(self, "amplitude", float(c[0]))
        else:
            object.__setattr__(self, "amplitude", float(np.linalg.norm(c)))

    @property
    def energy(self) -> float:
        c = np.asarray(self.coeffs, dtype=np.float64)
        return float(np.dot(c, c))

    @property
    def n_components(self) -> int:
        return int(np.asarray(self.coeffs).size)


@dataclass
class Dictionary:
    """Active set of templates keyed by template_id."""

    templates: dict[int, Template] = field(default_factory=dict)
    order: list[int] = field(default_factory=list)

    def add(self, template: Template) -> None:
        tid = template.template_id
        if tid not in self.templates:
            self.order.append(tid)
        self.templates[tid] = template

    def __iter__(self):
        for tid in self.order:
            yield self.templates[tid]

    def __len__(self) -> int:
        return len(self.order)

    @property
    def max_n_components(self) -> int:
        if not self.order:
            return 1
        return max(t.n_components for t in self)
