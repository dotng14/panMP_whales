"""Co-located .gpl cache helpers (shared by preprocess shim and the web pipeline)."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from app.config import GPL_RESULT_DIRNAME, WAV_LIBRARY_ROOTS
from app.services.manifest_builder import build_manifest_from_outputs
from analysis.GPL_Python.gpl_outputs import PANEL_VERSION, read_panel_header, write_panel_bin


@dataclass(frozen=True)
class CacheStatus:
    cached: bool
    repeats_rebuilt: bool = False


def containing_root(wav_path: Path) -> Path | None:
    wav = wav_path.resolve()
    for root in WAV_LIBRARY_ROOTS:
        r = root.resolve()
        if wav.is_relative_to(r):
            return r
    return None


def cache_dir_for(wav_path: Path) -> Path:
    root = containing_root(wav_path)
    if root is None:
        return wav_path.with_suffix(".gpl")
    rel = wav_path.resolve().relative_to(root)
    return root / GPL_RESULT_DIRNAME / rel.with_suffix(".gpl")


def _sidecar_freq_stale(output_dir: Path) -> bool:
    """True when the cache predates the current frequency range.

    Compares end_freq against min(5000, nyquist) rather than a flat 5000 Hz:
    a decimated recording's own Nyquist can legitimately sit below 5000 Hz,
    and that is not the same thing as a cache built by an older pipeline
    version that capped end_freq below the recording's real range.
    """
    try:
        # sorted(): glob() order isn't stable across calls, and an arbitrary
        # pick here can flip this check non-deterministically between calls.
        sc = sorted((output_dir / "spectrogram_npy").glob("spwhiten_*.json"))[0]
        meta = json.loads(sc.read_text())
        end_freq = float(meta.get("end_freq", 0.0))
        nyquist = float(meta.get("sample_freq", 0.0)) / 2
        expected = min(5000.0, nyquist) if nyquist > 0 else 5000.0
        return end_freq < expected
    except Exception:
        return False


def _sidecar_npy_legacy(output_dir: Path) -> bool:
    try:
        # sorted(): see _sidecar_freq_stale — same non-determinism risk.
        sc = sorted((output_dir / "spectrogram_npy").glob("spwhiten_*.json"))[0]
        return json.loads(sc.read_text()).get("npy_is_clipped", True)
    except Exception:
        return False


def _ensure_panel_current(output_dir: Path, stem: str) -> bool:
    """Ensure panel.bin exists (prefer analysis process_v4; backfill if missing)."""
    panel_path = output_dir / "panel.bin"
    hdr = read_panel_header(panel_path) if panel_path.exists() else None
    if hdr is not None and hdr.get("panelVersion", 0) >= PANEL_VERSION:
        if "freqLo" in hdr and "freqHi" in hdr:
            return True
    try:
        write_panel_bin(output_dir, stem)
        return True
    except Exception:
        panel_path.unlink(missing_ok=True)
        return False


def check_gpl_cache(wav_path: Path) -> CacheStatus:
    """Cache is valid when manifest + panel.bin exist (dictionary lives in the catalog)."""
    cache_dir = cache_dir_for(wav_path)
    manifest_path = cache_dir / "manifest.json"
    output_dir = cache_dir / "output"
    panel_path = output_dir / "panel.bin"

    if not manifest_path.exists() or not panel_path.exists():
        return CacheStatus(cached=False)

    if _sidecar_freq_stale(output_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)
        return CacheStatus(cached=False)

    if _sidecar_npy_legacy(output_dir):
        shutil.rmtree(cache_dir, ignore_errors=True)
        return CacheStatus(cached=False)

    if not _ensure_panel_current(output_dir, wav_path.stem):
        return CacheStatus(cached=False)

    cached = panel_path.exists() and manifest_path.exists()
    return CacheStatus(cached=cached, repeats_rebuilt=False)


def is_gpl_cached(wav_path: Path) -> bool:
    return check_gpl_cache(wav_path).cached


def ensure_cache_output_dir(wav_path: Path) -> Path:
    output_dir = cache_dir_for(wav_path) / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def finalize_gpl_cache(wav_path: Path, output_dir: Path, *, title: str | None = None) -> Path:
    """Ensure panel.bin + manifest.json exist (panel usually already written by process_v4)."""
    cache_dir = cache_dir_for(wav_path)
    cache_dir.mkdir(parents=True, exist_ok=True)

    panel_path = output_dir / "panel.bin"
    if not panel_path.is_file():
        write_panel_bin(output_dir, title or wav_path.stem)

    manifest = build_manifest_from_outputs(output_dir, job_id=wav_path.stem)
    manifest_path = cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return cache_dir
