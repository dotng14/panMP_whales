import tempfile
import unittest
from pathlib import Path

import numpy as np

from analysis.GPL_Python.gpl_signal import DetectorParams
from analysis.GPL_Python.gpl_outputs import (
    OutputWriter,
    clip_intensity_percentile,
    display_from_spwhiten,
)


class TestIntensityClip(unittest.TestCase):
    def test_default_percentile_is_unclipped(self) -> None:
        # Production default is 100 (no clipping) so panel_builder.py owns the
        # one global clip; see plans/10-integrate-gpl-python-matlab-parity.md.
        sp = np.array([[0.0, 1.0, 100.0], [2.0, 50.0, 200.0]], dtype=np.float32)
        clipped, lo, hi = clip_intensity_percentile(sp)
        self.assertEqual(lo, 0.0)
        self.assertEqual(hi, float(sp.max()))
        np.testing.assert_array_equal(clipped, sp)

    def test_clip_at_explicit_99_9_percentile(self) -> None:
        sp = np.array([[0.0, 1.0, 100.0], [2.0, 50.0, 200.0]], dtype=np.float32)
        clipped, lo, hi = clip_intensity_percentile(sp, hi_percentile=99.9)
        self.assertEqual(lo, 0.0)
        self.assertAlmostEqual(hi, float(np.percentile(sp.astype(np.float64), 99.9)))
        self.assertLessEqual(clipped.max(), hi + 1e-6)
        self.assertGreaterEqual(clipped.min(), 0.0)
        self.assertEqual(clipped[1, 2], hi)

    def test_display_inherits_clipped_raw(self) -> None:
        sp = np.array([[0.0, 8.0], [1.0, 27.0]], dtype=np.float32)
        clipped, _, _ = clip_intensity_percentile(sp)
        disp = display_from_spwhiten(clipped)
        expected = np.power(clipped.astype(np.float64), 1.0 / 3.0).astype(np.float32)
        np.testing.assert_allclose(disp, expected)

    def test_export_pipeline_writes_clip_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            writer = OutputWriter(out)
            sp = np.linspace(0, 10, 100, dtype=np.float32).reshape(10, 10)
            parm = DetectorParams.for_audio(10_000, 80_000, freq_lo=10.0, freq_hi=4_000.0)
            npy_path = writer.write_window_spectrogram_npy(
                file_stem="demo",
                window_index=1,
                sp_whiten=sp,
                parm=parm,
                start_sample=1,
                end_sample=1000,
            )
            raw = np.load(npy_path)
            disp = np.load(npy_path.with_name(npy_path.stem + "_display.npy"))
            # Default percentile is 100 (unclipped) — analysis .npy keeps the raw
            # whitened band; only panel_builder.py applies a global clip.
            self.assertAlmostEqual(float(raw.max()), float(sp.max()), places=5)
            np.testing.assert_allclose(disp, display_from_spwhiten(raw))
            meta = (out / "spectrogram_npy" / f"{npy_path.stem}.json").read_text()
            self.assertIn("intensity_clip_hi_percentile", meta)
            self.assertIn("intensity_clip_hi", meta)


if __name__ == "__main__":
    unittest.main()
