"""Export the two 30-minute test-half clips that get hand-labelled in Audacity.

    python -m analysis.song_hmm.exp3.reference.export_clips

Writes ``clip1.wav``, ``clip2.wav`` (2 kHz int16, DC offset removed) and
``clips.json`` (recording offset of each clip) into
``analysis/song_hmm/exp3/work/reference/``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.io import wavfile

from analysis.song_hmm.exp1_slot_hmm.labels import load_labels

REPO = Path(__file__).resolve().parents[4]
WAV_PATH = Path(r"C:\Users\hoang\Downloads\SOCAL_H_76_230426_193500_df100.x.x.wav")
CSV_PATH = REPO / "backend/data/song_structure/SOCAL_H_76_230426_193500_df100.x.x.motif_segments.csv"
OUT_DIR = REPO / "analysis/song_hmm/exp3/work/reference"
N_SECONDS = 36000
TEST_START = 18000

# name -> (start_s, end_s) in recording time
CLIPS: dict[str, tuple[int, int]] = {
    "clip1": (25200, 27000),  # contains CSV song starts at ~25505 s and ~26271 s
    "clip2": (30600, 32400),  # mid-session; CSV song starts at ~31113 s and ~31807 s
}


def dc_free_int16(x: np.ndarray) -> np.ndarray:
    """Subtract the mean and return int16 (clipped)."""
    y = x.astype(np.float64)
    y -= y.mean()
    return np.clip(np.round(y), -32768, 32767).astype(np.int16)


def export(wav_path: Path = WAV_PATH, out_dir: Path = OUT_DIR) -> dict:
    sr, data = wavfile.read(wav_path, mmap=True)
    if data.ndim > 1:
        data = data[:, 0]
    song_starts = load_labels(CSV_PATH, N_SECONDS).song_bounds[:, 0]
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"source_wav": str(wav_path), "sample_rate": int(sr), "clips": {}}
    for name, (start, end) in CLIPS.items():
        if not (TEST_START <= start < end <= N_SECONDS):
            raise ValueError(f"{name} must lie in the test half")
        inside = song_starts[(song_starts >= start) & (song_starts < end)]
        if name == "clip1" and inside.size == 0:
            raise ValueError("clip1 must contain a CSV song start")
        clip = dc_free_int16(np.asarray(data[start * sr : end * sr]))
        wavfile.write(out_dir / f"{name}.wav", sr, clip)
        meta["clips"][name] = {
            "file": f"{name}.wav",
            "offset_s": start,
            "duration_s": end - start,
            "csv_song_starts_s": [round(float(s), 1) for s in inside],
        }
        print(f"{name}: {start}-{end} s, CSV song starts {meta['clips'][name]['csv_song_starts_s']}")
    (out_dir / "clips.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--wav", type=Path, default=WAV_PATH)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    export(args.wav, args.out)


if __name__ == "__main__":
    main()
