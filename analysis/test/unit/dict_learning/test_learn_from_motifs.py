"""Motif search, dictionary refine, and the coverage stop."""

from __future__ import annotations

from pathlib import Path

import json

import numpy as np


from analysis.bands import N_BANDS  # noqa: E402
from analysis.dict_learning.code.io import stitch_whitened_spectrogram  # noqa: E402
from analysis.dict_learning.code.learn.learn_from_motifs import (  # noqa: E402
    learn_dictionary_from_motifs,
    main,
)


def _write_window(directory: Path, name: str, data: np.ndarray, start_sample: int) -> None:
    path = directory / name
    np.save(path, np.asarray(data, dtype=np.float32))
    path.with_suffix(".json").write_text(
        json.dumps(
            {
                "start_sample": start_sample,
                "end_sample": start_sample + data.shape[1] - 1,
                "overlap": 1,
                "fftl": 8,
                "sample_freq": 1.0,
                "start_freq": 0.0,
                "end_freq": 100.0,
            }
        )
    )


def test_repeated_motif_covers_and_stops():
    profile = np.zeros((N_BANDS, 80))
    motif = np.arange(N_BANDS, dtype=np.float64)[:, None] + np.linspace(1.0, 2.0, 5)
    for start in (0, 20, 40, 60):
        profile[:, start : start + 5] = motif
    result = learn_dictionary_from_motifs(
        profile,
        frame_rate=1,
        lengths_s=(5, 10),
        min_coverage=0.05,
        nthreads=1,
    )
    assert result.stop_reason == "coverage"
    assert result.coverage >= 0.05
    assert len(result.dictionary) >= 1
    assert result.detection_map.shape == (80,)
    assert result.residual.shape == profile.shape
    assert set(np.unique(result.detection_map)) <= {-1, *result.dictionary.order}
    width = result.rounds[0].width
    assert width in (5, 10)
    atom = next(iter(result.dictionary))
    assert atom.data.shape == (N_BANDS, atom.width)


def test_silence_adds_no_motif():
    result = learn_dictionary_from_motifs(
        np.zeros((N_BANDS, 40)),
        frame_rate=1,
        lengths_s=(5, 10),
        min_coverage=0.5,
        nthreads=1,
    )
    assert result.stop_reason == "no_motif"
    assert result.coverage == 0.0
    assert len(result.dictionary) == 0
    assert np.all(result.detection_map == -1)


def test_covered_frames_are_not_chosen_again():
    from analysis.dict_learning.code.learn.learn_from_motifs import _strongest_motif

    profile = np.zeros((N_BANDS, 80))
    first = np.arange(N_BANDS, dtype=np.float64)[:, None] + np.linspace(1.0, 2.0, 5)
    second = first[::-1] + 3.0
    profile[:, 0:5] = first
    profile[:, 30:35] = first
    profile[:, 50:55] = second
    profile[:, 65:70] = second
    occupied = np.zeros(profile.shape[1], dtype=bool)
    occupied[0:5] = True
    occupied[30:35] = True
    found = _strongest_motif(profile, [(5.0, 5)], nthreads=1, occupied=occupied)
    assert found is not None
    width = found[2]
    for start in (found[3], found[4]):
        assert not np.any(occupied[start : start + width])
    assert {found[3], found[4]} != {0, 30}


def test_stitch_joins_windows_in_time_order(tmp_path):
    npy_dir = tmp_path / "spectrogram_npy"
    npy_dir.mkdir()
    _write_window(npy_dir, "spwhiten_b.npy", np.full((2, 3), 2.0), start_sample=4)
    _write_window(npy_dir, "spwhiten_a.npy", np.full((2, 4), 1.0), start_sample=1)
    stitched = stitch_whitened_spectrogram(tmp_path)
    assert stitched.data.shape == (2, 7)
    assert np.all(stitched.data[:, :4] == 1.0)
    assert np.all(stitched.data[:, 4:] == 2.0)
    assert stitched.sec_per_bin == 1.0
    assert stitched.start_freq == 0.0
    assert stitched.end_freq == 100.0


def test_cli_writes_detection_map_and_dictionary(tmp_path):
    npy_dir = tmp_path / "spectrogram_npy"
    npy_dir.mkdir()
    _write_window(npy_dir, "spwhiten_silence.npy", np.zeros((N_BANDS, 40)), start_sample=1)
    out = tmp_path / "out"
    code = main(
        [
            "--gpl-dir",
            str(tmp_path),
            "--out",
            str(out),
            "--lengths",
            "5,10",
            "--nthreads",
            "1",
            "--no-progress",
        ]
    )
    assert code == 0
    assert (out / "detection_map.npy").is_file()
    assert (out / "summary.json").is_file()
    assert (out / "matchings.json").is_file()
    labels = np.load(out / "detection_map.npy")
    assert labels.shape == (40,)
    assert np.all(labels == -1)
