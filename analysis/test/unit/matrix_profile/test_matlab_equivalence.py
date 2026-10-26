"""Run panMP.m and mpxForPan.m in MATLAB and check the Python ports match.

Skipped when no ``matlab`` executable is on PATH. All MATLAB work happens in
one ``matlab -batch`` call because each startup costs several seconds.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


from analysis.matrix_profile import mpxForPan  # noqa: E402
import analysis.matrix_profile as matrix_profile_pkg
from analysis.matrix_profile import panMP  # noqa: E402

ROOT = Path(matrix_profile_pkg.__file__).resolve().parent

MATLAB = shutil.which("matlab")
pytestmark = pytest.mark.skipif(MATLAB is None, reason="MATLAB not on PATH")

# panMP.m hardcodes motif lengths 32:64:512, so its series must be well over
# 512 + 32 samples long.
PAN_LEN = 1500

# (name, minlag, w) for the mpxForPan cases.
MPX_CASES = [("noise", 4, 4), ("noise", 14, 16), ("sine", 30, 32), ("walk", 62, 64)]

# zscore needs the Statistics and Machine Learning Toolbox. Appended to the
# end of the path so a real zscore wins when it is installed.
ZSCORE_FALLBACK = "function z = zscore(x)\nz = (x - mean(x)) ./ std(x);\nend\n"


def _series():
    rng = np.random.default_rng(7)
    t = np.arange(PAN_LEN)
    return {
        "noise": rng.standard_normal(PAN_LEN),
        "sine": np.sin(t / 20.0) + 0.3 * rng.standard_normal(PAN_LEN),
        "walk": np.cumsum(rng.standard_normal(PAN_LEN)),
    }


def _matlab_str(path):
    return "'" + str(path).replace("'", "''") + "'"


@pytest.fixture(scope="module")
def matlab_out(tmp_path_factory):
    work = tmp_path_factory.mktemp("matlab_equivalence")
    fallback = work / "fallback"
    fallback.mkdir()
    (fallback / "zscore.m").write_text(ZSCORE_FALLBACK)

    series = _series()
    lines = [
        f"addpath({_matlab_str(ROOT)});",
        f"addpath({_matlab_str(fallback)}, '-end');",
        "set(0, 'DefaultFigureVisible', 'off');",
    ]
    for name, x in series.items():
        np.savetxt(work / f"{name}.txt", x, fmt="%.17g")
        lines += [
            f"x = load({_matlab_str(work / f'{name}.txt')});",
            f"writematrix(panMP(x), {_matlab_str(work / f'pan_{name}.csv')});",
            "close all;",
        ]
    for name, minlag, w in MPX_CASES:
        tag = f"{name}_{minlag}_{w}"
        lines += [
            f"x = load({_matlab_str(work / f'{name}.txt')});",
            f"[mp, mpi] = mpxForPan(x, {minlag}, {w});",
            f"writematrix([mp, mpi], {_matlab_str(work / f'mpx_{tag}.csv')});",
        ]
    script = work / "run_equivalence.m"
    script.write_text("\n".join(lines) + "\n")

    proc = subprocess.run(
        [MATLAB, "-batch", f"run({_matlab_str(script)})"],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if proc.returncode != 0:
        pytest.fail(f"MATLAB failed:\n{proc.stdout}\n{proc.stderr}")

    def read(stem):
        return np.genfromtxt(work / f"{stem}.csv", delimiter=",")

    return series, read


@pytest.mark.parametrize("name,minlag,w", MPX_CASES)
def test_mpxForPan_matches_matlab(matlab_out, name, minlag, w):
    series, read = matlab_out
    ml = read(f"mpx_{name}_{minlag}_{w}")
    mp, mpi = mpxForPan(series[name], minlag, w)
    np.testing.assert_allclose(mp, ml[:, 0], rtol=0.0, atol=1e-10)
    # MATLAB indices are 1-based.
    np.testing.assert_array_equal(mpi + 1, ml[:, 1])


@pytest.mark.parametrize("name", ["noise", "sine", "walk"])
def test_panMP_matches_matlab(matlab_out, name):
    series, read = matlab_out
    ml = read(f"pan_{name}")
    pan = panMP(series[name], plot=False)
    assert pan.shape == ml.shape
    np.testing.assert_array_equal(np.isnan(pan), np.isnan(ml))
    np.testing.assert_allclose(pan, ml, rtol=0.0, atol=1e-10, equal_nan=True)
