"""Fast GPU hybrid song-detector pipeline (2023 dictionary + HMM theme model).

Runs ``analysis/dict_HMM/find_songs/c/song_detector`` (~1-2 s per 22h 10 kHz WAV)
with ``--save-spec`` to produce:
  - ``panel.bin`` (whitened spectrogram in the viewer's native binary format)
  - ``song_structure.json`` (motif placements + decoded themes + conforming songs)
  - ``decoded_songs.csv`` and ``placements_<stem>.csv``
"""
from __future__ import annotations

import bisect
import csv
import functools
import json
import re
import struct
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import argrelmin

from analysis.GPL_Python.gpl_outputs import (
    PANEL_VERSION,
    _MAGNITUDE_PERCENTILE,
    _header_padding,
    clip_intensity_percentile,
    display_from_spwhiten,
)
from app.config import BACKEND_DIR

HYBRID_DIR = BACKEND_DIR.parent / "analysis" / "dict_HMM" / "find_songs" / "c"
HYBRID_DETECTOR_BIN = HYBRID_DIR / "song_detector"
HYBRID_MODEL_BIN = HYBRID_DIR / "build" / "song_model.bin"

SOURCE_RATE = 10000
_STAMP_RE = re.compile(r"_(\d{6})_(\d{6})")


def _read_wav_fmt(path: Path) -> tuple[int, int, int, int] | None:
    try:
        with path.open("rb") as f:
            head = f.read(12)
            if len(head) < 12 or head[:4] != b"RIFF" or head[8:12] != b"WAVE":
                return None
            while True:
                chunk = f.read(8)
                if len(chunk) < 8:
                    return None
                name, size = struct.unpack("<4sI", chunk)
                if name == b"fmt ":
                    body = f.read(size)
                    if len(body) < 16:
                        return None
                    audio, channels, rate = struct.unpack("<HHI", body[:8])
                    bits = struct.unpack("<H", body[14:16])[0]
                    return rate, channels, audio, bits
                f.seek(size + (size & 1), 1)
    except OSError:
        return None


def can_run_hybrid(wav_path: Path) -> bool:
    """True when the C/CUDA song_detector + 2023 model are built and the WAV is 10 kHz mono PCM16."""
    if not HYBRID_DETECTOR_BIN.is_file() or not HYBRID_MODEL_BIN.is_file():
        return False
    fmt = _read_wav_fmt(wav_path)
    if fmt is None:
        return False
    rate, channels, audio, bits = fmt
    return rate == SOURCE_RATE and channels == 1 and audio == 1 and bits == 16


def _wav_start_unix(name: str) -> float:
    m = _STAMP_RE.search(name)
    if not m:
        return 0.0
    yy, mm, dd = int(m.group(1)[:2]), int(m.group(1)[2:4]), int(m.group(1)[4:6])
    hh, mi, ss = int(m.group(2)[:2]), int(m.group(2)[2:4]), int(m.group(2)[4:6])
    try:
        return datetime(2000 + yy, mm, dd, hh, mi, ss, tzinfo=timezone.utc).timestamp()
    except ValueError:
        return 0.0


def _write_panel_from_spec_bin(spec_path: Path, output_dir: Path, title: str) -> Path:
    with spec_path.open("rb") as f:
        magic = f.read(8)
        if magic != b"SPEC0001":
            raise ValueError(f"Unexpected spec header: {magic!r}")
        n_rows, n_cols, _stride, _n_windows, source_rate = struct.unpack("<5i", f.read(20))
        (total_samples,) = struct.unpack("<q", f.read(8))
        freq_lo, freq_hi = struct.unpack("<2d", f.read(16))
        raw = np.fromfile(f, dtype=np.float32, count=n_rows * n_cols).reshape((n_rows, n_cols))

    clipped, _, _ = clip_intensity_percentile(raw, hi_percentile=_MAGNITUDE_PERCENTILE)
    display = display_from_spwhiten(clipped)

    energy_raw = display.sum(axis=0).astype(np.float32)
    energy_smooth = gaussian_filter1d(energy_raw, sigma=3).astype(np.float32)
    cutoffs = argrelmin(energy_smooth, order=5)[0].tolist()

    # Gentle floor (0.35) + adaptive p99.9 ceiling (floor 1.85) keeps both
    # loud and quieter songs smooth-blue with bright red calls.
    vmin = 0.35
    norm_max = max(float(np.percentile(display, _MAGNITUDE_PERCENTILE)), 1.85)
    magnitude_u8 = np.clip((display - vmin) / (norm_max - vmin) * 255.0, 0, 255).astype(np.uint8)

    audio_duration_sec = float(total_samples) / float(source_rate)
    header = {
        "panelVersion": PANEL_VERSION,
        "freqBins": int(n_rows),
        "timeCols": int(n_cols),
        "cutoffs": cutoffs,
        "title": title,
        "colormap": "jet",
        "brightness": 1.0,
        "normMax": norm_max,
        "audioDurationSec": audio_duration_sec,
        "freqLo": float(freq_lo),
        "freqHi": float(freq_hi),
        "arrays": [
            {"name": "magnitude", "dtype": "uint8", "length": int(magnitude_u8.size)},
            {"name": "energyRaw", "dtype": "float32", "length": int(energy_raw.size)},
            {"name": "energySmoothed", "dtype": "float32", "length": int(energy_smooth.size)},
        ],
    }

    header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
    pad = _header_padding(len(header_json))
    panel_path = output_dir / "panel.bin"
    with panel_path.open("wb") as f:
        f.write(struct.pack("<I", len(header_json)))
        f.write(header_json)
        f.write(b"\x00" * pad)
        f.write(magnitude_u8.tobytes())
        f.write(energy_raw.tobytes())
        f.write(energy_smooth.tobytes())
    return panel_path


# Bump whenever the song_structure.json builder changes so cached files rebuild on read.
SONG_STRUCTURE_VERSION = 2


def _build_song_structure_json(
    output_dir: Path,
    wav_name: str,
) -> dict[str, Any]:
    songs: list[dict[str, Any]] = []
    themes: list[dict[str, Any]] = []

    decoded_csv = output_dir / "decoded_songs.csv"
    if decoded_csv.is_file():
        with decoded_csv.open(newline="") as f:
            for row in csv.DictReader(f):
                try:
                    min_margin = float(row["min_margin"])
                    shuffles = int(float(row["shuffles_at_or_above"]))
                    start_sec = float(row["start_sec"])
                    end_sec = float(row["end_sec"])
                except (KeyError, TypeError, ValueError):
                    continue
                # Only conforming windows count as songs, but every decoded window
                # keeps its themes (themes are not dropped with the song filter).
                if min_margin > 0 and shuffles == 0:
                    songs.append({
                        "song": len(songs) + 1,
                        "startSec": start_sec,
                        "endSec": end_sec,
                    })
                try:
                    t_starts = [float(x) for x in row["theme_starts_sec"].split("/")]
                    t_mins = [float(x) for x in row["theme_minutes"].split("/")]
                    for t_idx, (ts, tm) in enumerate(zip(t_starts, t_mins)):
                        if tm > 0:
                            themes.append({
                                "theme": t_idx,
                                "startSec": ts,
                                "endSec": ts + tm * 60.0,
                            })
                except (KeyError, ValueError):
                    pass

    # The span lookup bisects on start times, so both lists must be time-ordered.
    themes.sort(key=lambda t: t["startSec"])
    songs.sort(key=lambda s: s["startSec"])
    for number, song in enumerate(songs, start=1):
        song["song"] = number

    song_starts = [s["startSec"] for s in songs]
    theme_starts = [t["startSec"] for t in themes]

    def _find_span(starts: list[float], spans: list[dict[str, Any]], t: float, key: str) -> int | None:
        if not starts:
            return None
        idx = bisect.bisect_right(starts, t) - 1
        if idx >= 0 and spans[idx]["startSec"] <= t <= spans[idx]["endSec"]:
            return int(spans[idx][key])
        return None

    stem = wav_name.split(".")[0]
    placements_csv = output_dir / f"placements_{stem}.csv"
    units: list[dict[str, Any]] = []
    if placements_csv.is_file():
        with placements_csv.open(newline="") as f:
            for row in csv.DictReader(f):
                try:
                    s_sec = float(row["start_sec"])
                    e_sec = float(row["end_sec"])
                    motif = str(row["motif"])
                except (KeyError, TypeError, ValueError):
                    continue
                mid = 0.5 * (s_sec + e_sec)
                units.append({
                    "motif": motif,
                    "song": _find_span(song_starts, songs, mid, "song"),
                    "theme": _find_span(theme_starts, themes, mid, "theme"),
                    "phraseId": None,
                    "phraseCount": None,
                    "startSec": s_sec,
                    "endSec": e_sec,
                })

    payload = {
        "version": SONG_STRUCTURE_VERSION,
        "wavStartUnix": _wav_start_unix(wav_name),
        "units": units,
        "songs": songs,
        "themes": themes if themes else None,
    }
    try:
        (output_dir / "song_structure.json").write_text(json.dumps(payload))
    except OSError:
        pass
    return payload


def load_or_rebuild_song_structure(output_dir: Path, wav_name: str) -> dict[str, Any] | None:
    """Return cached ``song_structure.json``, rebuilding from ``decoded_songs.csv`` if stale."""
    hybrid_json = output_dir / "song_structure.json"
    cached: dict[str, Any] | None = None
    if hybrid_json.is_file():
        try:
            cached = json.loads(hybrid_json.read_text())
        except (OSError, ValueError):
            cached = None
        if cached is not None and cached.get("version") == SONG_STRUCTURE_VERSION:
            return cached
    if (output_dir / "decoded_songs.csv").is_file():
        return _build_song_structure_json(output_dir, wav_name)
    return cached


def run_hybrid_pipeline(wav_path: Path, output_dir: Path, title: str) -> dict[str, Any]:
    """Run ``song_detector`` (2023 dictionary + HMM) and write ``panel.bin`` + ``song_structure.json``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    spec_path = output_dir / "spec.bin"
    try:
        proc = subprocess.run(
            [
                str(HYBRID_DETECTOR_BIN),
                "--save-spec",
                str(spec_path),
                str(HYBRID_MODEL_BIN),
                str(output_dir),
                str(wav_path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or f"exit {proc.returncode}").strip()
            raise RuntimeError(f"song_detector failed: {err}")

        _write_panel_from_spec_bin(spec_path, output_dir, title)
        _build_song_structure_json(output_dir, wav_path.name)
    finally:
        spec_path.unlink(missing_ok=True)

    return {"jobId": title, "windows": {}, "source": "hybrid_2023"}


# ── On-demand full-resolution spectrogram slices (0.123 s/col) ───────────────

_SLICE_MAX_SPAN_SEC = 360.0
_SLICE_N_ROWS = 736


@functools.lru_cache(maxsize=48)
def _compute_window_u8_float(wav_path_str: str, win_idx: int, n_rows: int) -> np.ndarray | None:
    """Compute one 75-second full-res whitened spectrogram window (float32, shape n_rows × 609)."""
    from analysis.dict_HMM.find_songs import spectro  # noqa: PLC0415

    wav_path = Path(wav_path_str)
    start_sec = win_idx * float(spectro.WINDOW_SEC)
    audio = spectro.read_seconds(wav_path, start_sec, float(spectro.WINDOW_SEC))
    expected = spectro.WINDOW_SEC * spectro.SOURCE_RATE
    if audio.size < expected:
        if audio.size < spectro.SOURCE_RATE:
            return None
        padded = np.zeros(expected, dtype=audio.dtype)
        padded[: audio.size] = audio
        audio = padded
    spec = spectro.window_spectrogram(audio, n_rows)
    return spec.astype(np.float32, copy=False)


def render_highres_slice(
    wav_path: Path,
    start_sec: float,
    end_sec: float,
    *,
    n_rows: int = _SLICE_N_ROWS,
) -> tuple[bytes, int, int, float, float] | None:
    """Return full-res (0.123 s/col) uint8 spectrogram slice for [start_sec, end_sec].

    Returns ``(raw_u8_bytes, freq_bins, time_cols, actual_start_sec, actual_end_sec)``
    or ``None`` if the WAV is not a readable 10 kHz mono PCM16 file.
    """
    fmt = _read_wav_fmt(wav_path)
    if fmt is None:
        return None
    rate, channels, audio_fmt, bits = fmt
    if rate != SOURCE_RATE or channels != 1 or audio_fmt != 1 or bits != 16:
        return None

    from analysis.dict_HMM.find_songs import spectro  # noqa: PLC0415

    s0 = max(0.0, float(start_sec))
    s1 = max(s0 + 1.0, min(float(end_sec), s0 + _SLICE_MAX_SPAN_SEC))
    win_sec = float(spectro.WINDOW_SEC)
    hop = float(spectro.HOP_SEC)
    w0 = int(s0 // win_sec)
    w1 = int(s1 // win_sec)

    windows: list[np.ndarray] = []
    wav_str = str(wav_path)
    for w in range(w0, w1 + 1):
        arr = _compute_window_u8_float(wav_str, w, n_rows)
        if arr is None:
            break
        windows.append(arr)
    if not windows:
        return None

    full = np.concatenate(windows, axis=1)
    base_sec = w0 * win_sec
    c0 = max(0, int(round((s0 - base_sec) / hop)))
    c1 = min(full.shape[1], max(c0 + 1, int(round((s1 - base_sec) / hop))))
    sl = full[:, c0:c1]

    clipped, _, _ = clip_intensity_percentile(sl, hi_percentile=_MAGNITUDE_PERCENTILE)
    display = display_from_spwhiten(clipped)
    # Gentle floor (0.35) + adaptive p99.9 ceiling (floor 1.85) keeps both
    # quieter songs (Song 1) and loud songs (Song 14) smooth-blue with bright red calls.
    vmin = 0.35
    norm_max = max(float(np.percentile(display, _MAGNITUDE_PERCENTILE)), 1.85)
    u8 = np.clip((display - vmin) / (norm_max - vmin) * 255.0, 0, 255).astype(np.uint8)
    u8 = np.ascontiguousarray(u8)

    actual_start = base_sec + c0 * hop
    actual_end = base_sec + c1 * hop
    return u8.tobytes(), int(u8.shape[0]), int(u8.shape[1]), actual_start, actual_end

