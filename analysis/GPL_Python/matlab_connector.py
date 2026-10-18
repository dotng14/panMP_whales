"""Launch GPL_v4 MATLAB parity exports and load checkpoint .mat files.

Designed for legacy MATLAB (Release 14 / ~2004) constraints:

- config passed as ``.mat`` (no ``jsondecode``)
- audio via ``wavread`` fallback in ``gpl_parity_export.m``
- batch invocation via ``-nojvm -nodesktop -r ...; exit``

Modern MATLAB can also use JSON configs when ``jsondecode`` is available.
"""

from __future__ import annotations

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
