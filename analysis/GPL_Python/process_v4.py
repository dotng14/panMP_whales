#!/usr/bin/env python3
"""
Python translation baseline for GPL_v4/process_v4.m.

Core behavior:
- discover WAV files
- split into process windows
- run GPL-like detector per window
- aggregate detections
- export whitened spectrogram windows as .npy (+ sidecar JSON)
- export Detections_*.mat, Labels_*.tlab, CSV_*.csv summaries
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from .gpl_detector import GPLDetector
from .gpl_signal import DetectorParams, read_wav_mono
from .gpl_outputs import (
    DEFAULT_INTENSITY_CLIP_HI_PERCENTILE,
    OutputWriter,
    matlab_datenum_from_datetime,
    write_panel_bin,
)


def normalize_input_path(path: Path) -> Path:
    """Expand ~ and fix shell-escaped paths pasted inside quotes (e.g. ``Shared\\ drives``)."""
    candidate = Path(path.expanduser())
    if candidate.exists():
        return candidate.resolve()

    unescaped = re.sub(r"\\([ @&|*?<>()$!#'\"\\])", r"\1", str(candidate))
    if unescaped != str(candidate):
        candidate = Path(unescaped)
        if candidate.exists():
            return candidate.resolve()

    return candidate.resolve()


def is_wav_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() == ".wav"


class ProcessingOrchestrator:
    def __init__(
        self,
        output_dir: Path,
        *,
        freq_lo: float = 10.0,
        freq_hi: float = 10000.0,
        intensity_clip_hi_percentile: float = DEFAULT_INTENSITY_CLIP_HI_PERCENTILE,
    ) -> None:
        self.output_dir = output_dir
        self.freq_lo = freq_lo
        self.freq_hi = freq_hi
        self.detector = GPLDetector()
        self.writer = OutputWriter(
            output_dir,
            intensity_clip_hi_percentile=intensity_clip_hi_percentile,
        )

    @staticmethod
    def discover_wavs(input_path: Path) -> list[Path]:
        input_path = normalize_input_path(input_path)
        if is_wav_file(input_path):
            return [input_path]
        if not input_path.is_dir():
            raise FileNotFoundError(f"Input path not found: {input_path}")
        wavs = sorted(p for p in input_path.iterdir() if is_wav_file(p))
        return wavs

    @staticmethod
    def wav_start_datetime(path: Path) -> datetime:
        # Pattern from getwavheaders.m: YYYYMMDD_HHMMSS in filename
        m = re.search(r"(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})", path.stem)
        if m:
            y, mo, d, hh, mm, ss = map(int, m.groups())
            return datetime(y, mo, d, hh, mm, ss)
        return datetime(2000, 1, 1, 0, 0, 0)

    def process(self, input_path: Path) -> None:
        wav_files = self.discover_wavs(input_path)
        if not wav_files:
            resolved = normalize_input_path(input_path)
            raise FileNotFoundError(
                f"No WAV files found at {resolved}. "
                "Use an unescaped path (quote the directory only, not spaces: "
                '"/path/Shared drives/CHNMS/"*wav*).'
            )

        all_detections: list[dict] = []
        for wav_path in wav_files:
            print(f"Open file: {wav_path.name}")
            data, sample_rate = read_wav_mono(wav_path)
            nyquist = sample_rate / 2.0
            effective_freq_hi = min(self.freq_hi, nyquist)
            if effective_freq_hi != self.freq_hi:
                print(
                    f"  Note: freq_hi capped {self.freq_hi:g} -> "
                    f"{effective_freq_hi:g} Hz "
                    f"(Nyquist for {sample_rate} Hz sample rate)"
                )
            parm = DetectorParams.for_audio(
                sample_rate,
                len(data),
                nwav=1,
                freq_lo=self.freq_lo,
                freq_hi=effective_freq_hi,
            )
            start_dt = self.wav_start_datetime(wav_path)
            num_wav_loops = int(np.ceil(len(data) / parm.nrec))

            for q in range(1, num_wav_loops + 1):
                sample_start = (q - 1) * parm.nrec
                sample_end = min(sample_start + parm.nrec, len(data))
                sub_data = data[sample_start:sample_end]
                if len(sub_data) <= parm.fftl:
                    continue
                local_parm = parm.with_window_samples(len(sub_data))
                if local_parm.num_time_bins <= 0:
                    continue

                this_time = start_dt + timedelta(seconds=(q - 1) * (parm.nrec / parm.sample_freq))
                print(f"  Processing window {q}/{num_wav_loops}: {this_time.isoformat(sep=' ')}")
                result = self.detector.detect_window(sub_data, local_parm)

                self.writer.write_window_spectrogram_npy(
                    file_stem=wav_path.stem,
                    window_index=q,
                    sp_whiten=result.sp_whiten,
                    parm=local_parm,
                    start_sample=sample_start + 1,
                    end_sample=sample_end,
                )

                for det in result.detections:
                    start_global = int(det["start_time"] + sample_start)
                    end_global = int(det["end_time"] + sample_start)
                    det["start_time"] = start_global
                    det["end_time"] = end_global
                    det["julian_start_time"] = matlab_datenum_from_datetime(
                        start_dt + timedelta(seconds=start_global / parm.sample_freq)
                    )
                    det["julian_end_time"] = matlab_datenum_from_datetime(
                        start_dt + timedelta(seconds=end_global / parm.sample_freq)
                    )
                    det["fname"] = wav_path.name
                    all_detections.append(det)

        clean_name = wav_files[0].stem
        mat_path, tlab_path, csv_path = self.writer.write_detection_outputs(clean_name, all_detections, parm)
        print(f"Saved MAT:  {mat_path}")
        print(f"Saved TLAB: {tlab_path}")
        print(f"Saved CSV:  {csv_path}")
        print(f"Saved NPY spectrograms: {self.writer.npy_dir}")
        print(f"Detections: {len(all_detections)}")

        panel_path = write_panel_bin(self.output_dir, clean_name)
        print(f"Saved panel: {panel_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Translate GPL_v4 process_v4 core to Python.")
    p.add_argument("input_path", type=Path, help="WAV file or folder with WAV files")
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("GPL_Python_outputs"),
        help="Output directory for detections and spectrogram_npy",
    )
    p.add_argument(
        "--freq-lo",
        type=float,
        default=10.0,
        help="Lower frequency bound (Hz) for spectrograms and detection (default: 10).",
    )
    p.add_argument(
        "--freq-hi",
        type=float,
        default=10000.0,
        help="Upper frequency bound (Hz) for spectrograms and detection (default: 10000).",
    )
    p.add_argument(
        "--intensity-clip-hi-percentile",
        type=float,
        default=DEFAULT_INTENSITY_CLIP_HI_PERCENTILE,
        help=(
            "Upper intensity clip for exported spectrograms, as a percentile of "
            f"sp_whiten (default: {DEFAULT_INTENSITY_CLIP_HI_PERCENTILE:g}; "
            "100 = clip at max, i.e. no upper trim)."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    input_path = normalize_input_path(args.input_path)
    output_dir = normalize_input_path(args.output_dir)
    orchestrator = ProcessingOrchestrator(
        output_dir,
        freq_lo=args.freq_lo,
        freq_hi=args.freq_hi,
        intensity_clip_hi_percentile=args.intensity_clip_hi_percentile,
    )
    orchestrator.process(input_path)


if __name__ == "__main__":
    main()

