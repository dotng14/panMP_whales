"""Shared fixtures: synthetic GPL folders, series and WAV files (see ``synthetic.py``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from analysis.test.synthetic import planted_series, write_gpl_folder, write_wav


@pytest.fixture(scope="session")
def gpl_folder(tmp_path_factory):
    """Three windows, three planted motifs, detections over every copy."""
    return write_gpl_folder(tmp_path_factory.mktemp("gpl") / "out")


@pytest.fixture(scope="session")
def gpl_dir(gpl_folder) -> Path:
    return gpl_folder.root


@pytest.fixture(scope="session")
def gpl_variant(tmp_path_factory):
    """``gpl_variant(detections=..., n_windows=...)`` writes a GPL folder once per variant."""
    made: dict[tuple, Path] = {}

    def make(**kwargs) -> Path:
        key = tuple(sorted(kwargs.items()))
        if key not in made:
            made[key] = write_gpl_folder(tmp_path_factory.mktemp("gplv") / "out", **kwargs).root
        return made[key]

    return make


@pytest.fixture(scope="session")
def series_files(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("series")
    files = {
        "1d": root / "series_1d.npy",
        "3row": root / "series_3row.npy",
        "short": root / "series_short.npy",
        "3d": root / "series_3d.npy",
    }
    np.save(files["1d"], planted_series())
    np.save(files["3row"], planted_series(rows=3))
    np.save(files["short"], planted_series(n=40, starts=(2,), length=8))
    np.save(files["3d"], np.zeros((2, 2, 2)))
    return files


@pytest.fixture(scope="session")
def wav_file(tmp_path_factory) -> Path:
    return write_wav(tmp_path_factory.mktemp("wav") / "clip.wav")


@pytest.fixture(scope="session")
def stereo_wav_file(tmp_path_factory) -> Path:
    return write_wav(tmp_path_factory.mktemp("wav") / "stereo.wav", channels=2)
