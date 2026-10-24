"""Compare the Python detector with the MATLAB GPL export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .gpl_detector import GPLDetector
from .gpl_signal import (
    DetectorParams,
    QuietEstimator,
    STFTProcessor,
    WhiteningProcessor,
    read_wav_mono,
    slice_gpl_freq_band,
    whiten_matrix,
)


def compute_python_parity(
    wav_path: Path,
    *,
    freq_lo: float = 10.0,
    freq_hi: float = 10_000.0,
    window_sec: float = 75.0,
    start_sample: int = 1,
    fifo_removal: int = 0,
) -> dict[str, Any]:
    """Compute checkpoint arrays on one GPL window (Python GPL_Python path)."""
    wav_path = wav_path.expanduser().resolve()
    audio, sample_freq = read_wav_mono(wav_path)

    nrec = int(round(window_sec * sample_freq))
    start_idx = max(0, int(start_sample) - 1)
    finish_idx = start_idx + nrec
    if finish_idx > len(audio):
        raise ValueError(
            f"Window exceeds audio length: need {finish_idx} samples, have {len(audio)}"
        )
    sub_data = audio[start_idx:finish_idx]

    parm = DetectorParams.for_audio(
        sample_freq,
        len(audio),
        nwav=1,
        freq_lo=freq_lo,
        freq_hi=freq_hi,
    )
    local = parm.with_window_samples(len(sub_data))
    if fifo_removal:
        raise NotImplementedError("Python parity export does not implement fifo_removal yet")

    sp_fft_full = STFTProcessor().compute(sub_data, local)
    sp_band = slice_gpl_freq_band(sp_fft_full, local.freq_bin_lo, local.freq_bin_hi)

    whitener = WhiteningProcessor()
    sp_whiten = whitener.apply(sp_band, local)
    spc, _, mu = whiten_matrix(sp_band, local.white_x)
    sp_whiten_norm = np.abs(spc / (mu[:, None]))

    quiet_est = QuietEstimator()
    _, _, blocked, baseline0 = quiet_est.estimate(sp_band, sp_whiten, local)
    base_save = baseline0

    sp_loop = sp_whiten.copy()
    sum_t = np.maximum(np.sum(sp_loop**2, axis=0, keepdims=True) ** 0.5, 1e-12)
    sum_f = np.maximum(np.sum(sp_loop.T**2, axis=0, keepdims=True).T ** 0.5, 1e-12)
    norm_v = sp_loop / sum_t
    norm_h = sp_loop / sum_f
    sp_w_v, _, _ = whiten_matrix(norm_v.T, local.white_x)
    norm_v = sp_w_v.T
    sp_w_h, _, _ = whiten_matrix(norm_h, local.white_x)
    norm_h = sp_w_h
    bas = np.abs(norm_v) ** local.xp1 * np.abs(norm_h) ** local.xp2
    base_in_iter0 = np.sum(bas**2, axis=0)

    detector = GPLDetector()
    result = detector.detect_window(sub_data, local)
    det_start = [int(d["start_time"]) for d in result.detections]
    det_finish = [int(d["end_time"]) for d in result.detections]

    return {
        "engine": "GPL_Python",
        "wav_path": str(wav_path),
        "freq_lo": freq_lo,
        "freq_hi": freq_hi,
        "sample_freq": sample_freq,
        "start_sample": start_sample,
        "nrec": nrec,
        "fftl": local.fftl,
        "fft_overlap": local.fft_overlap,
        "num_time_bins": local.num_time_bins,
        "freq_bin_lo": local.freq_bin_lo,
        "freq_bin_hi": local.freq_bin_hi,
        "num_freq_bins": local.num_freq_bins,
        "noise_ceiling": local.noise_ceiling,
        "thresh": local.thresh,
        "sp_fft_pos": sp_fft_full[: local.fftl // 2 + 1, :],
        "sp_band": sp_band,
        "mu": mu,
        "sp_whiten_norm": sp_whiten_norm,
        "sp_whiten": sp_whiten,
        "baseline0": baseline0,
        "base_save": base_save,
        "base_in_iter0": base_in_iter0,
        "blocked": blocked,
        "n_detections": len(result.detections),
        "det_start_bin": np.asarray(det_start, dtype=np.int64),
        "det_finish_bin": np.asarray(det_finish, dtype=np.int64),
    }

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat, savemat

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GPL_V4_DIR = REPO_ROOT / "GPL_v4"
DEFAULT_PARITY_SCRIPT = "gpl_parity_export.m"
MATLAB_R2004_ENV = "MATLAB_R2004_EXECUTABLE"
MATLAB_EXECUTABLE_ENV = "MATLAB_EXECUTABLE"


@dataclass(frozen=True)
class ParityConfig:
    wav_path: Path
    out_mat_path: Path
    freq_lo: float = 10.0
    freq_hi: float = 10_000.0
    window_sec: float = 75.0
    start_sample: int = 1
    fifo_removal: int = 0

    def resolved(self) -> ParityConfig:
        return ParityConfig(
            wav_path=self.wav_path.expanduser().resolve(),
            out_mat_path=self.out_mat_path.expanduser().resolve(),
            freq_lo=self.freq_lo,
            freq_hi=self.freq_hi,
            window_sec=self.window_sec,
            start_sample=self.start_sample,
            fifo_removal=self.fifo_removal,
        )


@dataclass(frozen=True)
class MatlabConnectorOptions:
    matlab_executable: Path | None = None
    gpl_v4_dir: Path = DEFAULT_GPL_V4_DIR
    parity_script: str = DEFAULT_PARITY_SCRIPT
    use_mat_config: bool = True
    timeout_sec: float | None = 600.0


class Matlab2004Connector:
    """Python bridge to ``GPL_v4/gpl_parity_export.m``."""

    def __init__(self, options: MatlabConnectorOptions | None = None) -> None:
        self.options = options or MatlabConnectorOptions()

    @property
    def matlab_executable(self) -> Path:
        return resolve_matlab_executable(self.options.matlab_executable)

    @property
    def gpl_v4_dir(self) -> Path:
        path = self.options.gpl_v4_dir.expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"GPL_v4 directory not found: {path}")
        return path

    def write_config(self, config: ParityConfig, path: Path) -> Path:
        cfg = config.resolved()
        path = path.expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.options.use_mat_config or path.suffix.lower() == ".mat":
            write_parity_config_mat(cfg, path)
        else:
            write_parity_config_json(cfg, path)
        return path

    def run_parity_export(self, config: ParityConfig) -> Path:
        cfg = config.resolved()
        if not cfg.wav_path.is_file():
            raise FileNotFoundError(f"WAV not found: {cfg.wav_path}")

        script_path = self.gpl_v4_dir / self.options.parity_script
        if not script_path.is_file():
            raise FileNotFoundError(f"MATLAB parity script not found: {script_path}")

        cfg.out_mat_path.parent.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(prefix="gpl_parity_cfg_") as tmp:
            if self.options.use_mat_config:
                config_path = Path(tmp) / "parity_config.mat"
                self.write_config(cfg, config_path)
            else:
                config_path = Path(tmp) / "parity_config.json"
                self.write_config(cfg, config_path)

            matlab_cmd = (
                f"addpath('{_matlab_quote(self.gpl_v4_dir)}'); "
                f"gpl_parity_export('{_matlab_quote(config_path)}');"
            )
            cmd = [
                str(self.matlab_executable),
                "-batch",
                matlab_cmd,
            ]
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.options.timeout_sec,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    "MATLAB parity export failed "
                    f"(exit {proc.returncode}).\n"
                    f"stdout:\n{proc.stdout}\n"
                    f"stderr:\n{proc.stderr}"
                )

        if not cfg.out_mat_path.is_file():
            raise FileNotFoundError(
                f"MATLAB did not write parity output: {cfg.out_mat_path}\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}"
            )
        return cfg.out_mat_path

    def load_parity(self, mat_path: Path) -> dict[str, Any]:
        return load_parity_mat(mat_path)


def resolve_matlab_executable(explicit: Path | None = None) -> Path:
    """Resolve MATLAB binary for GPL_v4 parity runs."""
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    for env_name in (MATLAB_R2004_ENV, MATLAB_EXECUTABLE_ENV):
        env_val = os.environ.get(env_name)
        if env_val:
            candidates.append(Path(env_val).expanduser())

    which = shutil.which("matlab")
    if which:
        candidates.append(Path(which))

    # Common macOS bundle layouts (R14-era through current).
    candidates.extend(sorted(Path("/Applications").glob("MATLAB_R2004b.app/Contents/MacOS/matlab")))
    candidates.extend(sorted(Path("/Applications").glob("MATLAB_R2004b.app/bin/matlab")))
    candidates.extend(sorted(Path("/Applications").glob("MATLAB_R20*.app/bin/matlab")))

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved

    raise FileNotFoundError(
        "MATLAB executable not found. Set "
        f"{MATLAB_R2004_ENV} or {MATLAB_EXECUTABLE_ENV}, "
        "or pass --matlab /path/to/matlab."
    )


def write_parity_config_mat(config: ParityConfig, path: Path) -> None:
    cfg = config.resolved()
    savemat(
        str(path),
        {
            "wav_path": np.array(str(cfg.wav_path), dtype=np.str_),
            "out_mat_path": np.array(str(cfg.out_mat_path), dtype=np.str_),
            "freq_lo": float(cfg.freq_lo),
            "freq_hi": float(cfg.freq_hi),
            "window_sec": float(cfg.window_sec),
            "start_sample": int(cfg.start_sample),
            "fifo_removal": int(cfg.fifo_removal),
        },
        do_compression=False,
    )


def write_parity_config_json(config: ParityConfig, path: Path) -> None:
    cfg = config.resolved()
    payload = {
        "wav_path": str(cfg.wav_path),
        "out_mat_path": str(cfg.out_mat_path),
        "freq_lo": cfg.freq_lo,
        "freq_hi": cfg.freq_hi,
        "window_sec": cfg.window_sec,
        "start_sample": cfg.start_sample,
        "fifo_removal": cfg.fifo_removal,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_parity_mat(mat_path: Path) -> dict[str, Any]:
    """Load ``parity`` struct from ``gpl_parity_export`` output."""
    mat_path = mat_path.expanduser().resolve()
    raw = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    if "parity" not in raw:
        raise KeyError(f"No 'parity' variable in {mat_path}")
    return _mat_struct_to_dict(raw["parity"])


def compare_checkpoint_arrays(
    name: str,
    matlab: np.ndarray | None,
    python: np.ndarray | None,
) -> dict[str, Any]:
    if matlab is None or python is None:
        return {
            "name": name,
            "ok": False,
            "reason": "missing_array",
            "matlab_shape": None if matlab is None else list(matlab.shape),
            "python_shape": None if python is None else list(python.shape),
        }
    a = np.asarray(matlab, dtype=np.float64)
    b = np.asarray(python, dtype=np.float64)
    if a.shape != b.shape:
        return {
            "name": name,
            "ok": False,
            "reason": "shape_mismatch",
            "matlab_shape": list(a.shape),
            "python_shape": list(b.shape),
        }
    diff = a - b
    denom = max(float(np.linalg.norm(a)), 1e-12)
    return {
        "name": name,
        "ok": bool(np.max(np.abs(diff)) < 1e-5),
        "max_abs_diff": float(np.max(np.abs(diff))),
        "mean_abs_diff": float(np.mean(np.abs(diff))),
        "relative_l2": float(np.linalg.norm(diff) / denom),
        "corr": float(np.corrcoef(a.ravel(), b.ravel())[0, 1]),
    }


def _matlab_quote(path: Path) -> str:
    return str(path).replace("'", "''")


def _mat_struct_to_dict(value: Any) -> dict[str, Any]:
    if hasattr(value, "_fieldnames"):
        return {name: _mat_value_to_python(getattr(value, name)) for name in value._fieldnames}
    if isinstance(value, np.void) and value.dtype.names:
        return {name: _mat_value_to_python(value[name]) for name in value.dtype.names}
    raise TypeError(f"Expected MATLAB struct, got {type(value)!r}")


def _mat_value_to_python(value: Any) -> Any:
    if hasattr(value, "_fieldnames"):
        return _mat_struct_to_dict(value)
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            if value.size == 1:
                return _mat_value_to_python(value.item())
            return [_mat_value_to_python(v) for v in value.ravel()]
        if value.dtype.kind in {"U", "S"}:
            if value.ndim == 0:
                return str(value.item())
            return "".join(value.tolist()).strip()
        return np.asarray(value)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    if isinstance(value, np.generic):
        if np.issubdtype(value.dtype, np.floating):
            return float(value)
        if np.issubdtype(value.dtype, np.integer):
            return int(value)
        return value.item()
    return value
