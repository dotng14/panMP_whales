import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from analysis.GPL_Python.process_v4 import ProcessingOrchestrator, normalize_input_path
from analysis.GPL_Python.gpl_models import DetectorParams


class TestSmoke(unittest.TestCase):
    def test_end_to_end_outputs_npy(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wav_path = root / "demo_20240101_000000.wav"
            sr = 10000
            t = np.arange(sr * 8) / sr
            signal = (0.3 * np.sin(2 * np.pi * 450 * t)).astype(np.float32)
            wavfile.write(wav_path, sr, signal)

            out = root / "out"
            ProcessingOrchestrator(out, freq_lo=10.0, freq_hi=4_000.0).process(wav_path)
            npy_files = list((out / "spectrogram_npy").glob("*.npy"))
            self.assertTrue(npy_files)

    def test_discover_x_x_wav(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            wav_path = root / "CHNMS_NO01_231106_050140.x.x.wav"
            wav_path.write_bytes(b"RIFF    WAVEfmt ")
            found = ProcessingOrchestrator.discover_wavs(root)
            self.assertEqual(found, [wav_path.resolve()])

    def test_normalize_shell_escaped_path(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            escaped = Path(str(root).replace(" ", "\\ "))
            self.assertEqual(normalize_input_path(escaped), root.resolve())

    def test_for_audio_default_range_at_leidata_sample_rate(self) -> None:
        parm = DetectorParams.for_audio(200_000, 1_000_000)
        self.assertEqual(parm.freq_lo, 10.0)
        self.assertEqual(parm.freq_hi, 10000.0)
        self.assertGreater(parm.num_freq_bins, 0)
        parm = DetectorParams.for_audio(200_000, 1_000_000, freq_lo=300.0, freq_hi=1500.0)
        self.assertEqual(parm.freq_lo, 300.0)
        self.assertEqual(parm.freq_hi, 1500.0)
        self.assertGreater(parm.num_freq_bins, 0)
        self.assertLess(parm.freq_bin_lo, parm.freq_bin_hi)

    def test_for_audio_rejects_freq_hi_above_nyquist(self) -> None:
        with self.assertRaises(ValueError):
            DetectorParams.for_audio(10_000, 80_000, freq_lo=100.0, freq_hi=6_000.0)


if __name__ == "__main__":
    unittest.main()

