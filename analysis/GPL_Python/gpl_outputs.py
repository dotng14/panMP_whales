from __future__ import annotations

import csv
import json
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from .gpl_models import DetectorParams


def matlab_datenum_from_datetime(dt: datetime) -> float:
    # MATLAB datenum epoch is 0000-01-00; Python ordinal epoch is 0001-01-01.
    return dt.toordinal() + 366 + (
        dt.hour / 24.0 + dt.minute / 1440.0 + dt.second / 86400.0 + dt.microsecond / 86400.0 / 1e6
    )


DEFAULT_INTENSITY_CLIP_HI_PERCENTILE = 100.0


def clip_intensity_percentile(
    sp: np.ndarray,
    *,
    lo: float = 0.0,
    hi_percentile: float = DEFAULT_INTENSITY_CLIP_HI_PERCENTILE,
) -> tuple[np.ndarray, float, float]:
    """Clip spectrogram intensities to ``[lo, percentile(sp, hi_percentile)]``."""
    sp = np.asarray(sp, dtype=np.float64)
    hi = float(np.percentile(sp, hi_percentile))
    if hi < lo:
        hi = lo
    return np.clip(sp, lo, hi).astype(np.float32), lo, hi


def display_from_spwhiten(sp_whiten: np.ndarray) -> np.ndarray:
    """Cube-root display transform applied after intensity clipping."""
    return np.power(np.abs(sp_whiten).astype(np.float64), 1.0 / 3.0).astype(np.float32)


class OutputWriter:
    def __init__(
        self,
        output_dir: Path,
        *,
        intensity_clip_hi_percentile: float = DEFAULT_INTENSITY_CLIP_HI_PERCENTILE,
    ) -> None:
        self.output_dir = output_dir
        self.intensity_clip_hi_percentile = intensity_clip_hi_percentile
        self.npy_dir = output_dir / "spectrogram_npy"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.npy_dir.mkdir(parents=True, exist_ok=True)

    def write_window_spectrogram_npy(
        self,
        file_stem: str,
        window_index: int,
        sp_whiten: np.ndarray,
        parm: DetectorParams,
        start_sample: int,
        end_sample: int,
    ) -> Path:
        base = f"spwhiten_{file_stem}_w{window_index:06d}_s{start_sample:09d}_e{end_sample:09d}"
        npy_path = self.npy_dir / f"{base}.npy"
        # Keep the analysis array raw. Panel v3 applies one global clip after
        # stitching all windows; clipping each window here creates visible seams.
        sp_raw = np.asarray(sp_whiten, dtype=np.float32)
        np.save(npy_path, sp_raw)

        # Clipping is only for the offline display/BMP preview.
        sp_export, clip_lo, clip_hi = clip_intensity_percentile(
            sp_whiten, hi_percentile=self.intensity_clip_hi_percentile
        )

        bmp_path = self.npy_dir / f"{base}.bmp"
        disp_bins = display_from_spwhiten(sp_export)
        disp = disp_bins
        display_npy_path = self.npy_dir / f"{base}_display.npy"
        np.save(display_npy_path, disp)
        plt.imsave(bmp_path, disp, cmap="jet", origin="lower", format="bmp")

        meta = {
            "rows": int(sp_raw.shape[0]),
            "cols": int(sp_raw.shape[1]),
            "npy_is_clipped": False,
            "sample_freq": parm.sample_freq,
            "fftl": parm.fftl,
            "overlap": parm.fft_overlap,
            "start_freq": parm.freq_lo,
            "end_freq": parm.freq_hi,
            "window_length_sec": parm.window_length,
            "noise_ceiling": parm.noise_ceiling,
            "thresh": parm.thresh,
            "start_sample": int(start_sample),
            "end_sample": int(end_sample),
            "intensity_clip_lo": clip_lo,
            "intensity_clip_hi": clip_hi,
            "intensity_clip_hi_percentile": self.intensity_clip_hi_percentile,
            "preview_bmp": bmp_path.name,
            "display_npy": display_npy_path.name,
            "display_rows": int(disp.shape[0]),
            "analysis_rows": int(sp_raw.shape[0]),
            "display_y_axis": "analysis_bins",
            "display_transform": (
                f"clip[0, p{self.intensity_clip_hi_percentile:g}]; "
                "abs(sp_whiten)^(1/3); jet, origin=lower"
            ),
        }
        (self.npy_dir / f"{base}.json").write_text(json.dumps(meta, indent=2) + "\n")
        return npy_path

    def write_detection_outputs(
        self, clean_name: str, detections: list[dict[str, Any]], parm: DetectorParams
    ) -> tuple[Path, Path, Path]:
        mat_path = self.output_dir / f"Detections_{clean_name}.mat"
        tlab_path = self.output_dir / f"Labels_{clean_name}.tlab"
        csv_path = self.output_dir / f"CSV_{clean_name}.csv"

        if detections:
            times_struct = []
            labels = []
            for d in detections:
                times_struct.append(
                    {
                        "start_time": d["start_time"],
                        "end_time": d["end_time"],
                        "julian_start_time": d["julian_start_time"],
                        "julian_end_time": d["julian_end_time"],
                        "start_freq_hz": d.get("start_freq_hz", np.nan),
                        "end_freq_hz": d.get("end_freq_hz", np.nan),
                        "min_freq_hz": d.get("min_freq_hz", np.nan),
                        "max_freq_hz": d.get("max_freq_hz", np.nan),
                        "energy": d.get("energy", np.array([])),
                        "fname": d["fname"],
                    }
                )
                labels.append(int(d.get("boundary_cross_flag", 0)))
            savemat(
                mat_path,
                {"Times": np.array(times_struct, dtype=object), "Labels": np.array(labels, dtype=object), "parm": asdict(parm)},
                do_compression=True,
            )
            with tlab_path.open("w") as f:
                for d, lbl in zip(detections, labels):
                    # Keep simple: absolute MATLAB datenums in file.
                    f.write(f"{d['julian_start_time']:.10f}\t{d['julian_end_time']:.10f}\t{lbl}\n")

            with csv_path.open("w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "start_time",
                        "end_time",
                        "julian_start_time",
                        "julian_end_time",
                        "start_freq_hz",
                        "end_freq_hz",
                        "min_freq_hz",
                        "max_freq_hz",
                        "boundary_cross_flag",
                        "fname",
                    ],
                )
                writer.writeheader()
                for d in detections:
                    writer.writerow(
                        {
                            "start_time": d["start_time"],
                            "end_time": d["end_time"],
                            "julian_start_time": d["julian_start_time"],
                            "julian_end_time": d["julian_end_time"],
                            "start_freq_hz": d.get("start_freq_hz", np.nan),
                            "end_freq_hz": d.get("end_freq_hz", np.nan),
                            "min_freq_hz": d.get("min_freq_hz", np.nan),
                            "max_freq_hz": d.get("max_freq_hz", np.nan),
                            "boundary_cross_flag": d.get("boundary_cross_flag", 0),
                            "fname": d["fname"],
                        }
                    )
        else:
            mat_path.write_bytes(b"")
            tlab_path.write_text("")
            csv_path.write_text("")
        return mat_path, tlab_path, csv_path

