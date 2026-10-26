"""``python -m analysis.matrix_profile pan`` end to end."""

from __future__ import annotations

import numpy as np
import pytest

from analysis.matrix_profile.cli import main
from analysis.test.synthetic import read_json, run_module

LENGTHS = ["--min-len", "16", "--max-len", "32", "--step", "8"]


def pan(out, *args):
    return main(["pan", *args, "--out", str(out)])


def test_series_finds_planted_repeats(series_files, tmp_path, capsys):
    out = tmp_path / "pan"
    assert pan(out, "--series", str(series_files["1d"]), *LENGTHS, "--top-k", "3", "--plot") == 0
    surface = np.load(out / "pan.npy")
    meta = read_json(out / "pan.json")
    assert meta["lengths"] == [16, 24, 32]
    assert meta["rows"] == 1 and meta["signal"]["source"] == "series"
    assert surface.shape == (3, 600)
    assert np.nanmax(surface) == pytest.approx(1.0)
    pairs = read_json(out / "pairs.json")
    assert len(pairs) == 3
    planted = (40, 220, 420)
    best = pairs[0]
    assert any(abs(best["start_a"] - p) <= 8 for p in planted)
    assert any(abs(best["start_b"] - p) <= 8 for p in planted)
    assert (out / "pan.png").stat().st_size > 0
    assert '"n_pairs": 3' in capsys.readouterr().out


def test_reference_engine_matches_fast(series_files, tmp_path):
    fast, ref = tmp_path / "fast", tmp_path / "ref"
    assert pan(fast, "--series", str(series_files["1d"]), *LENGTHS) == 0
    assert pan(ref, "--series", str(series_files["1d"]), *LENGTHS, "--engine", "reference") == 0
    np.testing.assert_allclose(np.load(fast / "pan.npy"), np.load(ref / "pan.npy"), atol=1e-8)


def test_multi_row_series(series_files, tmp_path):
    out = tmp_path / "pan"
    assert pan(out, "--series", str(series_files["3row"]), *LENGTHS, "--threads", "2") == 0
    assert read_json(out / "pan.json")["rows"] == 3


@pytest.mark.parametrize("kind, rows", [("rms", 1), ("bands", 5)])
def test_wav(wav_file, tmp_path, kind, rows):
    out = tmp_path / "pan"
    assert pan(out, "--wav", str(wav_file), "--rate", "50", "--signal", kind,
               "--min-len", "8", "--max-len", "24", "--step", "8") == 0
    meta = read_json(out / "pan.json")
    assert meta["rows"] == rows and meta["rate"] == 50.0
    assert meta["n_samples"] == 300


def test_wav_bands_finer_than_100_hz_bins(wav_file, tmp_path):
    # 200 frames per second leave 200 Hz FFT bins, so some 100 Hz bands get none.
    out = tmp_path / "pan"
    assert pan(out, "--wav", str(wav_file), "--rate", "200", "--signal", "bands",
               "--min-len", "16", "--max-len", "16") == 0
    assert read_json(out / "pan.json")["rows"] == 5


@pytest.mark.parametrize("kind", ["rms", "bands"])
def test_stereo_wav_exit_2(stereo_wav_file, tmp_path, capsys, kind):
    assert pan(tmp_path / "pan", "--wav", str(stereo_wav_file), "--rate", "50", "--signal", kind) == 2
    assert "expected a mono recording" in capsys.readouterr().err


def test_wav_default_signal_is_rms(wav_file, tmp_path):
    out = tmp_path / "pan"
    assert pan(out, "--wav", str(wav_file), "--rate", "50", "--min-len", "8", "--max-len", "8") == 0
    assert read_json(out / "pan.json")["signal"]["kind"] == "rms"


@pytest.mark.parametrize(
    "extra, rows",
    [
        ([], 1),
        (["--signal", "energy", "--rate", "10"], 1),
        (["--signal", "bands", "--rate", "10"], 2),
        (["--signal", "bands", "--band-edges", "10,40,80,130"], 3),
        (["--signal", "full"], 12),
    ],
)
def test_gpl_signals(gpl_dir, tmp_path, extra, rows):
    out = tmp_path / "pan"
    assert pan(out, "--gpl-dir", str(gpl_dir), *extra, "--min-len", "8", "--max-len", "16",
               "--step", "4") == 0
    meta = read_json(out / "pan.json")
    assert meta["rows"] == rows
    assert meta["n_samples"] == (72 if "--rate" in extra else 720)


def test_gpl_rate_fills_bins_without_frames(gpl_dir, tmp_path):
    # 200 bins per second is finer than the 100 frames per second, so some
    # bins get no frame and are filled by interpolation.
    out = tmp_path / "pan"
    assert pan(out, "--gpl-dir", str(gpl_dir), "--rate", "200", "--min-len", "16",
               "--max-len", "16") == 0
    assert np.isfinite(np.load(out / "pan.npy")).any()


def test_gpl_spectrogram_npy_folder_directly(gpl_folder, tmp_path):
    out = tmp_path / "pan"
    assert pan(out, "--gpl-dir", str(gpl_folder.npy_dir), "--min-len", "8", "--max-len", "8") == 0


@pytest.mark.parametrize(
    "args, message",
    [
        (["--series", "MISSING"], "series file not found"),
        (["--series", "3d"], "must hold a 1-D or (rows, T) array"),
        (["--series", "short", "--min-len", "16", "--max-len", "32", "--step", "16"], "too short"),
        (["--series", "1d", "--min-len", "2"], "MinMotifLen must be >= 4"),
        (["--series", "1d", "--min-len", "32", "--max-len", "16"], "MaxMotifLen must be >= MinMotifLen"),
        (["--series", "1d", "--step", "0"], "StepSize must be >= 1"),
        (["--series", "1d", "--top-k", "0"], "--top-k must be >= 1"),
        (["--series", "3row", "--engine", "reference", "--min-len", "16", "--max-len", "16"],
         "reference engine takes a single row"),
        (["--wav", "MISSING", "--rate", "50"], "WAV file not found"),
        (["--wav", "WAV"], "--rate is required with --wav"),
        (["--wav", "WAV", "--rate", "50", "--signal", "full"], "WAV signal must be one of"),
        (["--wav", "WAV", "--rate", "300"], "must divide sample_rate"),
        (["--wav", "WAV", "--rate", "300", "--signal", "bands"], "must divide sample_rate"),
        (["--gpl-dir", "MISSING"], "GPL directory not found"),
        (["--gpl-dir", "EMPTY"], "no whitened spectrogram windows"),
        (["--gpl-dir", "GPL", "--signal", "rms"], "GPL signal must be one of"),
        (["--gpl-dir", "GPL", "--rate", "0"], "rate must be positive"),
        (["--gpl-dir", "GPL", "--signal", "bands", "--band-edges", "100,50"], "band edges must increase"),
    ],
)
def test_bad_input_exit_2(series_files, wav_file, gpl_dir, tmp_path, capsys, args, message):
    (tmp_path / "EMPTY").mkdir()
    names = {
        "MISSING": str(tmp_path / "missing"),
        "EMPTY": str(tmp_path / "EMPTY"),
        "WAV": str(wav_file),
        "GPL": str(gpl_dir),
        **{key: str(path) for key, path in series_files.items()},
    }
    resolved = [names.get(arg, arg) for arg in args]
    assert pan(tmp_path / "pan", *resolved) == 2
    assert message in capsys.readouterr().err


def test_gpl_windows_with_different_rows_exit_2(gpl_variant, tmp_path, capsys):
    root = gpl_variant(n_windows=2)
    windows = sorted((root / "spectrogram_npy").glob("spwhiten_*_e*[0-9].npy"))
    np.save(windows[1], np.zeros((5, 240), dtype=np.float32))
    assert pan(tmp_path / "pan", "--gpl-dir", str(root), "--min-len", "8", "--max-len", "8") == 2
    assert "different frequency rows" in capsys.readouterr().err


def test_python_dash_m_entry_point(series_files, tmp_path):
    out = tmp_path / "pan"
    proc = run_module("analysis.matrix_profile", "pan", "--series", str(series_files["1d"]),
                      *LENGTHS, "--out", str(out))
    assert proc.returncode == 0, proc.stderr
    assert (out / "pairs.json").is_file()
