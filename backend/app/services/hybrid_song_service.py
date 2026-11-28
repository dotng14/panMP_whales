"""Fast GPU hybrid song-detector pipeline (2023 dictionary + HMM theme model).

Runs ``analysis/dict_HMM/find_songs/c/song_detector`` (~1-2 s per 22h 10 kHz WAV)
with ``--save-spec`` to produce:
  - ``panel.bin`` (whitened spectrogram in the viewer's native binary format)
  - ``song_structure.json`` (motif placements + decoded themes + conforming songs)
  - ``decoded_songs.csv`` and ``placements_<stem>.csv``
"""
from __future__ import annotations

import base64
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
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.signal import argrelmin

from analysis.GPL_Python.gpl_outputs import (
    PANEL_VERSION,
    _MAGNITUDE_PERCENTILE,
    _header_padding,
    clip_intensity_percentile,
    display_from_spwhiten,
    read_panel_header,
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
#
# Tiles lie on the 12.5 kHz frame grid: tile k covers frames [k*TILE_FRAMES,
# (k+1)*TILE_FRAMES), so neighbouring tiles share an exact frame boundary.
# Each column depends only on global data, so tiles agree at their edges:
#   * the noise floor (mu) of each row is estimated once per tile-sized block
#     and linearly interpolated across block centres (no step at tile edges);
#   * the colour scale is fixed (no per-tile percentile);
#   * resampling, whitening and smoothing run on a context of CONTEXT_FRAMES
#     on each side, which is then cropped away.

_SLICE_MAX_SPAN_SEC = 600.0
_SLICE_N_ROWS = 736
TILE_FRAMES = 610  # 74.9568 s; a multiple of 5 so 10 kHz tile starts are integers
CONTEXT_FRAMES = 20  # multiple of 5 for the same reason
DISPLAY_VMIN = 0.35
DISPLAY_VMAX = 1.85


@functools.lru_cache(maxsize=64)
def _read_wav_fmt_cached(wav_path_str: str) -> tuple[int, int, int, int] | None:
    return _read_wav_fmt(Path(wav_path_str))


def _frame_to_sample_10k(frame: int) -> int:
    """10 kHz sample index of the start of a 12.5 kHz frame (exact for frame % 5 == 0)."""
    return frame * 1536 * 4 // 5


def _read_10k_samples(wav_path: Path, start: int, count: int) -> np.ndarray:
    """``count`` samples from ``start`` in [-1, 1); samples outside the file read as silence."""
    from analysis.dict_HMM.find_songs import spectro  # noqa: PLC0415

    _rate, offset, n_total = spectro.wav_layout(wav_path)
    out = np.zeros(count, dtype=np.float64)
    lo, hi = max(0, start), min(n_total, start + count)
    if hi > lo:
        data = np.fromfile(wav_path, dtype="<i2", count=hi - lo, offset=offset + 2 * lo)
        out[lo - start : hi - start] = data / 32768.0
    return out


def _padded_magnitudes(wav_path: Path, first_frame: int, n_frames: int) -> np.ndarray:
    """Magnitudes (rows 0..N, frames) for global frames [first_frame, first_frame + n_frames)."""
    from scipy.signal import resample_poly  # noqa: PLC0415

    from analysis.dict_HMM.find_songs import spectro  # noqa: PLC0415

    need_12k = (n_frames - 1) * spectro.STEP + spectro.FFTL
    count_10k = -(-need_12k * 4 // 5) + 16
    x = _read_10k_samples(wav_path, _frame_to_sample_10k(first_frame), count_10k)
    audio = resample_poly(x, 5, 4)
    frames = np.lib.stride_tricks.sliding_window_view(audio, spectro.FFTL)[:: spectro.STEP][:n_frames]
    spectrum = np.abs(np.fft.rfft(frames * np.hamming(spectro.FFTL), axis=1))
    # Rows 0..N: one extra row above the kept band so the smoothing sees the same neighbour.
    return spectrum[:, spectro.FIRST_BIN : spectro.FIRST_BIN + _SLICE_N_ROWS + 1].T


def _n_blocks(wav_path: Path) -> int:
    from analysis.dict_HMM.find_songs import spectro  # noqa: PLC0415

    _rate, _offset, n_total = spectro.wav_layout(wav_path)
    total_frames = (n_total * 5) // 4 // spectro.STEP
    return max(1, total_frames // TILE_FRAMES)


@functools.lru_cache(maxsize=256)
def _block_floor(wav_str: str, block: int) -> np.ndarray:
    """Per-row noise floor (N+1,) from the frames of one tile-sized block only."""
    from analysis.dict_HMM.find_songs import spectro  # noqa: PLC0415

    wav_path = Path(wav_str)
    block = min(max(block, 0), _n_blocks(wav_path) - 1)
    mag = _padded_magnitudes(
        wav_path,
        block * TILE_FRAMES - CONTEXT_FRAMES,
        TILE_FRAMES + 2 * CONTEXT_FRAMES,
    )
    mu = spectro.shortest_half_mean(mag[:, CONTEXT_FRAMES : CONTEXT_FRAMES + TILE_FRAMES])
    return np.maximum(mu, 1e-12)


def _column_floors(wav_str: str, tile: int, first_frame: int, n_frames: int) -> np.ndarray:
    """Noise floor for every column (N+1, n_frames): linear between block centres."""
    frames = first_frame + np.arange(n_frames)
    block = frames // TILE_FRAMES
    offset = frames - block * TILE_FRAMES
    frac = (offset + 0.5 - TILE_FRAMES / 2) / TILE_FRAMES  # in [-0.5, 0.5)
    partner = np.where(frac < 0, block - 1, block + 1)
    weight = np.abs(frac)

    # Blocks needed: tile-1, tile, tile+1 (the padded frames never reach further).
    stack = np.stack([_block_floor(wav_str, tile + d) for d in (-1, 0, 1)], axis=0)  # (3, N+1)
    idx_block = block - (tile - 1)
    idx_partner = partner - (tile - 1)
    assert idx_block.min() >= 0 and idx_block.max() <= 2
    assert idx_partner.min() >= 0 and idx_partner.max() <= 2
    mu_block = stack[idx_block].T  # (N+1, n_frames)
    mu_partner = stack[idx_partner].T
    return (1.0 - weight) * mu_block + weight * mu_partner


def _whiten_with_floor(mag: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """Same neighbourhood smoothing as ``spectro.whiten``, with a per-column noise floor."""
    white = np.abs((mag - mu) / mu)
    out = 4.0 * white
    out[1:, :] += white[:-1, :]
    out[:-1, :] += white[1:, :]
    out[:, 1:] += white[:, :-1]
    out[:, :-1] += white[:, 1:]
    return (out / 8.0)[:_SLICE_N_ROWS]


@functools.lru_cache(maxsize=160)
def _tile_u8(wav_str: str, tile: int) -> bytes:
    """Colour-mapped uint8 bytes (N, TILE_FRAMES) of one tile, row 0 = lowest frequency."""
    from scipy.ndimage import gaussian_filter  # noqa: PLC0415

    wav_path = Path(wav_str)
    first = tile * TILE_FRAMES - CONTEXT_FRAMES
    n = TILE_FRAMES + 2 * CONTEXT_FRAMES
    mag = _padded_magnitudes(wav_path, first, n)
    mu = _column_floors(wav_str, tile, first, n)
    white = _whiten_with_floor(mag, mu)
    display = display_from_spwhiten(white)
    display = gaussian_filter(display, sigma=(1.2, 0.7))
    u8 = np.clip((display - DISPLAY_VMIN) / (DISPLAY_VMAX - DISPLAY_VMIN) * 255.0, 0, 255)
    cropped = u8[:, CONTEXT_FRAMES : CONTEXT_FRAMES + TILE_FRAMES].astype(np.uint8)
    return np.ascontiguousarray(cropped).tobytes()


def render_highres_slice(
    wav_path: Path,
    start_sec: float,
    end_sec: float,
) -> tuple[bytes, int, int, float, float] | None:
    """Return full-res (0.123 s/col) uint8 spectrogram slice for [start_sec, end_sec].

    Returns ``(raw_u8_bytes, freq_bins, time_cols, actual_start_sec, actual_end_sec)``
    or ``None`` if the WAV is not a readable 10 kHz mono PCM16 file.
    """
    from analysis.dict_HMM.find_songs import spectro  # noqa: PLC0415

    wav_str = str(wav_path)
    fmt = _read_wav_fmt_cached(wav_str)
    if fmt is None:
        return None
    rate, channels, audio_fmt, bits = fmt
    if rate != SOURCE_RATE or channels != 1 or audio_fmt != 1 or bits != 16:
        return None

    hop = float(spectro.HOP_SEC)
    g0 = max(0, int(round(float(start_sec) / hop)))
    g1 = max(g0 + 1, int(round(float(end_sec) / hop)))
    g1 = min(g1, g0 + int(_SLICE_MAX_SPAN_SEC / hop))

    k0 = g0 // TILE_FRAMES
    k1 = (g1 - 1) // TILE_FRAMES
    tiles = [
        np.frombuffer(_tile_u8(wav_str, k), dtype=np.uint8).reshape(_SLICE_N_ROWS, TILE_FRAMES)
        for k in range(k0, k1 + 1)
    ]
    full = np.concatenate(tiles, axis=1)
    off = g0 - k0 * TILE_FRAMES
    sl = np.ascontiguousarray(full[:, off : off + (g1 - g0)])
    return sl.tobytes(), _SLICE_N_ROWS, int(sl.shape[1]), g0 * hop, g1 * hop


# ── Repeated Matches from 2023 motif dictionary (matching-pursuit placements) ─

_MODEL_HOP_SEC = 1536.0 / 12500.0
_HYBRID_REPEATS_MAX_PER_MOTIF = 200
_HYBRID_REPEATS_VERSION = 7


@functools.lru_cache(maxsize=4)
def _load_model_averages(mtime: float) -> dict[int, tuple[int, dict[str, Any]]]:
    """Read 2023 motif averages from ``song_model.bin`` -> ``{motif_id: (width_bins, average_dict)}``."""
    _ = mtime
    out: dict[int, tuple[int, dict[str, Any]]] = {}
    with HYBRID_MODEL_BIN.open("rb") as f:
        if f.read(8) != b"SONGMDL1":
            return out
        n_motifs, n_comp, n_rows, _n_themes = struct.unpack("<4i", f.read(16))
        ids = np.frombuffer(f.read(n_motifs * 4), dtype="<i4").tolist()
        widths = np.frombuffer(f.read(n_motifs * 4), dtype="<i4").tolist()
        for mid, w in zip(ids, widths):
            mean_arr = np.frombuffer(f.read(n_rows * w * 8), dtype="<f8").reshape((n_rows, w))
            f.seek(n_comp * n_rows * w * 8, 1)
            clipped, _, _ = clip_intensity_percentile(mean_arr, hi_percentile=_MAGNITUDE_PERCENTILE)
            display = display_from_spwhiten(clipped)
            norm_max = float(np.percentile(display, _MAGNITUDE_PERCENTILE))
            if norm_max <= 0:
                norm_max = 1.0
            u8 = np.clip(display / norm_max * 255.0, 0, 255).astype(np.uint8)
            out[int(mid)] = (
                int(w),
                {
                    "rows": int(n_rows),
                    "cols": int(w),
                    "durationSec": round(w * _MODEL_HOP_SEC, 4),
                    "normMax": norm_max,
                    "magnitude": base64.b64encode(np.ascontiguousarray(u8).tobytes()).decode("ascii"),
                },
            )
    return out


def build_hybrid_repeats(
    output_dir: Path,
    wav_name: str,
    *,
    max_per_motif: int = _HYBRID_REPEATS_MAX_PER_MOTIF,
) -> dict[str, Any] | None:
    """Build a ``RepeatMatches`` payload from ``placements_*.csv`` + 2023 ``song_model.bin``."""
    stem = Path(wav_name).stem
    placements_csv = output_dir / f"placements_{stem}.csv"
    if not placements_csv.is_file():
        candidates = sorted(output_dir.glob("placements_*.csv"))
        if not candidates:
            return None
        placements_csv = candidates[0]

    panel_path = output_dir / "panel.bin"
    if not panel_path.is_file():
        return None
    header = read_panel_header(panel_path)
    time_cols = int(header.get("timeCols", 0))
    audio_duration_sec = float(header.get("audioDurationSec", 0.0))
    if time_cols <= 0 or audio_duration_sec <= 0.0:
        return None

    averages: dict[int, tuple[int, dict[str, Any]]] = {}
    if HYBRID_MODEL_BIN.is_file():
        averages = _load_model_averages(HYBRID_MODEL_BIN.stat().st_mtime)

    by_motif: dict[int, list[tuple[float, float, float]]] = {}
    with placements_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                mid = int(row["motif"])
                s0 = float(row["start_sec"])
                s1 = float(row["end_sec"])
                unexp = float(row["unexplained"])
            except (KeyError, ValueError):
                continue
            by_motif.setdefault(mid, []).append((unexp, s0, s1))

    motif_ids = sorted(set(averages.keys()) | set(by_motif.keys()))
    templates: list[dict[str, Any]] = []
    for mid in motif_ids:
        rows = by_motif.get(mid, [])
        # Best matches first (lowest unexplained energy fraction).
        rows.sort(key=lambda r: r[0])
        if max_per_motif > 0 and len(rows) > max_per_motif:
            rows = rows[:max_per_motif]

        chunks: list[dict[str, Any]] = []
        for idx, (unexp, s0, s1) in enumerate(rows):
            c0 = max(0, min(time_cols - 1, int(round(s0 / audio_duration_sec * time_cols))))
            c1 = max(c0 + 1, min(time_cols, int(round(s1 / audio_duration_sec * time_cols))))
            samp0 = max(0, int(round(s0 * SOURCE_RATE)))
            samp1 = max(samp0 + 1, int(round(s1 * SOURCE_RATE)))
            chunks.append({
                "id": f"template-{mid}:{idx}",
                "sourceStartCol": c0,
                "sourceEndCol": c1,
                "sourceStartSample": samp0,
                "sourceEndSample": samp1,
                "windowId": f"w{int(s0 // 75):06d}",
                "windowStartCol": 0,
                "amplitude": 1.0,
                "originalTemplateNumber": mid,
                "stagedTemplateNumber": mid,
                "distanceNorm": max(0.0, min(1.0, unexp)),
            })

        width_bins, avg = averages.get(mid, (None, None))
        templates.append({
            "id": f"template-{mid}",
            "templateNumber": mid,
            "name": f"M{mid}",
            "sourceSegmentId": None,
            "widthBins": width_bins,
            "chunks": chunks,
            "average": avg,
        })

    return {"version": _HYBRID_REPEATS_VERSION, "templates": templates}


def render_hybrid_repeat_strip(
    wav_path: Path,
    output_dir: Path,
    wav_name: str,
    motif_id: int,
    *,
    max_chunks: int = _HYBRID_REPEATS_MAX_PER_MOTIF,
) -> tuple[bytes, int, int, list[int]] | None:
    """Stitch full-resolution (0.123 s/col) patches for one 2023 motif from ``_tile_u8``."""
    from analysis.dict_HMM.find_songs import spectro  # noqa: PLC0415

    wav_str = str(wav_path)
    fmt = _read_wav_fmt_cached(wav_str)
    if fmt is None:
        return None
    rate, channels, audio_fmt, bits = fmt
    if rate != SOURCE_RATE or channels != 1 or audio_fmt != 1 or bits != 16:
        return None

    stem = Path(wav_name).stem
    placements_csv = output_dir / f"placements_{stem}.csv"
    if not placements_csv.is_file():
        candidates = sorted(output_dir.glob("placements_*.csv"))
        if not candidates:
            return None
        placements_csv = candidates[0]

    rows: list[tuple[float, float, float]] = []
    with placements_csv.open(newline="") as f:
        for row in csv.DictReader(f):
            try:
                if int(row["motif"]) != motif_id:
                    continue
                rows.append((float(row["unexplained"]), float(row["start_sec"]), float(row["end_sec"])))
            except (KeyError, ValueError):
                continue

    if not rows:
        return None
    rows.sort(key=lambda r: r[0])
    if max_chunks > 0 and len(rows) > max_chunks:
        rows = rows[:max_chunks]

    hop = float(spectro.HOP_SEC)
    patches: list[np.ndarray] = []
    widths: list[int] = []
    tile_arrays: dict[int, np.ndarray] = {}

    for _unexp, s0, s1 in rows:
        g0 = max(0, int(round(s0 / hop)))
        g1 = max(g0 + 1, int(round(s1 / hop)))
        k0 = g0 // TILE_FRAMES
        k1 = (g1 - 1) // TILE_FRAMES
        parts: list[np.ndarray] = []
        for k in range(k0, k1 + 1):
            arr = tile_arrays.get(k)
            if arr is None:
                arr = np.frombuffer(_tile_u8(wav_str, k), dtype=np.uint8).reshape(_SLICE_N_ROWS, TILE_FRAMES)
                tile_arrays[k] = arr
            lo = max(0, g0 - k * TILE_FRAMES)
            hi = min(TILE_FRAMES, g1 - k * TILE_FRAMES)
            parts.append(arr[:, lo:hi])
        patch = parts[0] if len(parts) == 1 else np.concatenate(parts, axis=1)
        patches.append(patch)
        widths.append(int(patch.shape[1]))

    stitched = np.ascontiguousarray(np.concatenate(patches, axis=1))
    return stitched.tobytes(), _SLICE_N_ROWS, int(stitched.shape[1]), widths


