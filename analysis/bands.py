"""Shared time-frequency profile for motif search and dictionary learning.

The range 0–500 Hz is 5 bands of 100 Hz. A motif is one rectangle cut from
this profile: 5 rows by the motif's time span.
"""

from __future__ import annotations

import numpy as np

BAND_WIDTH_HZ = 100
N_BANDS = 5
FREQ_MAX_HZ = BAND_WIDTH_HZ * N_BANDS


def band_edges() -> list[tuple[int, int]]:  # no-pipeline-cover: used by tests and notebooks
    """Half-open edges ``[lo, hi)`` in Hz, low band first."""
    return [
        (i * BAND_WIDTH_HZ, (i + 1) * BAND_WIDTH_HZ) for i in range(N_BANDS)
    ]


def band_energy(
    samples: np.ndarray,
    sample_rate: int,
    frame_rate: int,
) -> np.ndarray:
    """Energy in each 100 Hz band. Shape ``(5, n_frames)``.

    Frames are non-overlapping and line up with ``rms_envelope`` at the same
    ``frame_rate``. A bin belongs to the band that contains its frequency.
    Energy at 500 Hz and above is left out.
    """
    if frame_rate < 1 or sample_rate % frame_rate != 0:
        raise ValueError("frame_rate must divide sample_rate")
    audio = np.asarray(samples)
    if audio.ndim == 2 and 1 in audio.shape:
        audio = audio.reshape(-1)  # no-pipeline-cover: WAV readers return 1-D mono
    if audio.ndim != 1:
        raise ValueError("expected a mono recording")
    frame = sample_rate // frame_rate
    usable = (len(audio) // frame) * frame
    blocks = audio[:usable].astype(np.float64, copy=False).reshape(-1, frame)
    spectrum = np.fft.rfft(blocks, axis=1)
    power = spectrum.real * spectrum.real + spectrum.imag * spectrum.imag
    freqs = np.fft.rfftfreq(frame, d=1.0 / sample_rate)
    edges = np.arange(N_BANDS + 1) * BAND_WIDTH_HZ
    band_index = np.searchsorted(edges, freqs, side="right") - 1
    band_index = np.where(freqs >= FREQ_MAX_HZ, -1, band_index)
    energy = np.zeros((N_BANDS, blocks.shape[0]), dtype=np.float64)
    for band in range(N_BANDS):
        mask = band_index == band
        if np.any(mask):
            energy[band] = power[:, mask].sum(axis=1)
    return energy
