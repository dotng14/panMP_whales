"""Pure-function version of generate_manifest.py that returns a dict (no file I/O)."""
from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_SIDECAR_RE = re.compile(
    r"^spwhiten_(?P<stem>.+?)_w(?P<win>\d+)_s(?P<start>\d+)_e(?P<end>\d+)$"
)


def build_manifest_from_outputs(output_dir: Path, job_id: str) -> dict:
    """Scan output_dir for GPL outputs and return manifest dict (no file writes)."""
    npy_dir = output_dir / "spectrogram_npy"
    if not npy_dir.exists():
        raise FileNotFoundError(f"spectrogram_npy not found at {npy_dir}")

    sidecars = sorted(npy_dir.glob("spwhiten_*.json"))
    if not sidecars:
        raise FileNotFoundError(f"No sidecar JSON files found in {npy_dir}")

    windows: dict = {}
    for json_path in sidecars:
        meta = json.loads(json_path.read_text())
        match = _SIDECAR_RE.match(json_path.stem)
        if not match:
            logger.warning("Skipping unexpected filename: %s", json_path.name)
            continue

        file_stem = match.group("stem")
        window_num = int(match.group("win"))
        sample_freq = meta["sample_freq"]
        start_sample = meta["start_sample"]
        end_sample = meta["end_sample"]

        boundaries: list[dict] = []
        csv_files = list(output_dir.glob(f"CSV_{file_stem}.csv"))
        if csv_files:
            with open(csv_files[0]) as f:
                for row in csv.DictReader(f):
                    det_start = int(row["start_time"])
                    if start_sample <= det_start <= end_sample:
                        local_start = (det_start - start_sample + 1) / sample_freq
                        local_end = (int(row["end_time"]) - start_sample + 1) / sample_freq
                        boundaries.append({
                            "start": max(0.0, local_start),
                            "end": local_end,
                            "flag": int(row.get("boundary_cross_flag", 0)),
                        })

        window_key = json_path.stem
        windows[window_key] = {
            "windowName": f"{file_stem} (Window {window_num})",
            "sampleRate": sample_freq,
            "windowStartSample": start_sample,
            "windowEndSample": end_sample,
            "freqBins": meta["rows"],
            "timeCols": meta["cols"],
            "detections": boundaries,
        }

    return {"jobId": job_id, "windows": windows}
