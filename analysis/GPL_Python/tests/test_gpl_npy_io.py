import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from analysis.GPL_Python.gpl_signal import DetectorParams
from analysis.GPL_Python.gpl_outputs import (
    analysis_band_slice,
    freq_axis_hz,
    imshow_extent,
    is_legacy_band_limited_export,
    load_display,
    load_raw,
    load_sidecar,
)
from analysis.GPL_Python.process_v4 import ProcessingOrchestrator


class TestBandLimitedExport(unittest.TestCase):
    def test_export_rows_match_freq_band(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wav_path = root / "demo_20240101_000000.wav"
            sr = 10000
            t = np.arange(sr * 8) / sr
            signal = (0.3 * np.sin(2 * np.pi * 450 * t)).astype(np.float32)
            wavfile.write(wav_path, sr, signal)

            out = root / "out"
            ProcessingOrchestrator(out, freq_lo=10.0, freq_hi=4_000.0).process(wav_path)

            json_files = list((out / "spectrogram_npy").glob("*.json"))
            self.assertTrue(json_files)
            meta = load_sidecar(json_files[0])
            parm = DetectorParams.for_audio(sr, len(signal), freq_lo=10.0, freq_hi=4_000.0)
            self.assertEqual(meta["rows"], parm.num_freq_bins)
            self.assertEqual(meta["start_freq"], 10.0)
            self.assertEqual(meta["end_freq"], 4_000.0)
            self.assertTrue(is_legacy_band_limited_export(meta))

            raw = load_raw(json_files[0].with_name(meta["display_npy"].replace("_display.npy", ".npy")))
            self.assertEqual(raw.shape[0], parm.num_freq_bins)

    def test_loader_band_crop(self) -> None:
        sp = np.arange(20 * 5, dtype=np.float32).reshape(20, 5)
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "test.npy"
            np.save(path, sp)
            full = load_raw(path)
            self.assertEqual(full.shape[0], 20)
            cropped = load_raw(path, freq_bin_min=2, freq_bin_max=5)
            self.assertEqual(cropped.shape, (4, 5))
            np.testing.assert_array_equal(cropped, sp[2:6, :])

    def test_freq_axis_and_extent(self) -> None:
        meta = {
            "rows": 100,
            "cols": 50,
            "start_freq": 10.0,
            "end_freq": 5000.0,
            "window_length_sec": 75.0,
        }
        axis = freq_axis_hz(meta)
        self.assertEqual(len(axis), 100)
        self.assertAlmostEqual(axis[0], 10.0)
        self.assertAlmostEqual(axis[-1], 5000.0)
        extent = imshow_extent(meta, (100, 50))
        self.assertEqual(extent, [0.0, 75.0, 10.0, 5000.0])
        self.assertIsNone(analysis_band_slice(meta))

    def test_legacy_sidecar_detection(self) -> None:
        meta = {"start_freq": 150.0, "end_freq": 2000.0, "rows": 185}
        self.assertTrue(is_legacy_band_limited_export(meta))
        self.assertIsNone(analysis_band_slice(meta))


if __name__ == "__main__":
    unittest.main()
