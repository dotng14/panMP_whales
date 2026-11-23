"""Call contours and their measurements: ``GPL_template`` / ``GPL_contour`` / ``GPL_measurements`` / ``GPL_filter``.

For each detection, ``GPL_v31`` cuts the call's columns out of the whitened
spectrogram and finds its contours:

- ``cm``: every island of high-energy pixels (islands smaller than
  ``smaller_island_cutoff`` of the largest are left out);
- ``cm_max``: the island with the most energy;
- ``cm_max2``: the island with the second most.

A pixel is high-energy when the call's whitened level over its own noise
(``sp / mu - 1``, ``mu`` from ``whiten_vec``) passes ``contour_cutoff``. The
mask is dilated by a 3x3 count (a pixel joins when 3 of its 9 neighbours pass)
and split into islands of directly adjacent pixels.

A contour is kept as MATLAB's ``GPL_sparse`` keeps it: the values scaled to
16-bit integers of their peak (``values``, ``scale``), their 1-based
column-major ``index`` and the contour's ``size``. Measurements are taken on
the contour rebuilt from that (``GPL_full``), as MATLAB does, so a pixel whose
whitened value is negative (it can enter through the dilation) rounds to 0 and
drops out.

Measurements (``parm.contour.measure`` picks them, per contour):

- ``duration_sec`` / ``duration_bin``: columns holding the contour;
- ``slope`` (Hz/s), ``slope_ci`` (width of its 95% interval, bins per bin),
  ``slope_y_int`` (frequency bin): a fit of frequency bin on time bin,
  weighted by the squared values, when the contour spans more than 2 columns;
- ``freq_bandwidth_hz`` / ``freq_bandwidth_bin``: rows holding the contour;
- ``start_freq_hz`` / ``end_freq_hz`` (mean row of the first / last column),
  ``min_freq_hz`` / ``max_freq_hz``, ``abs_bandwidth_hz``;
- per call: ``spec_rl`` (received level of ``cm`` over the noise around it, on
  the unwhitened spectrogram) and ``spec_noise`` (the window's quiet level).

The frequency formulas copy ``GPL_freq_measurements.m`` exactly, including its
offsets: start / end are ``(row + FreqBinLo) * df - 2`` and min / max are
``(row + FreqBinLo - 2) * df``, and ``abs_bandwidth_hz`` is ``max - min + 1``.

Differences from MATLAB:

- ``index`` is stored as uint32. MATLAB's uint16 saturates at 65,535, which a
  call reaches at 200 kHz (1,637 rows by 40 columns), and every later pixel then
  lands on that one index.
- ``GPL_filter.m`` tests each call's slope against the upper limit with the
  first call's slope (``GPL_struct(1)``); here each call uses its own.
- Only ``island_radius = 1`` is ported (MATLAB's ``radial_island`` is marked
  unfinished).
- Every contour is trimmed to the call's size; MATLAB leaves an empty
  ``cm_max2`` one bin larger on each side.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage, stats
from scipy.spatial import cKDTree

from .gpl_signal import CONTOURS, ContourLimits, DetectorParams, whiten_vec

_Q = 2**16 - 1
_SQUARE = np.ones((3, 3), dtype=int)
_CROSS = ndimage.generate_binary_structure(2, 1)


# ── contours (GPL_contour.m) ─────────────────────────────────────────────────


def find_contours(call_whiten: np.ndarray, parm: DetectorParams) -> dict[str, np.ndarray]:
    """``cm``, ``cm_max`` and ``cm_max2`` of one call's whitened spectrogram, each its shape (zeros when absent)."""
    cp = parm.contour
    if cp.island_radius != 1:
        raise NotImplementedError("only island_radius = 1 is ported from GPL_contour.m")
    call_whiten = np.asarray(call_whiten, dtype=float)
    bas = np.pad(call_whiten, 1)  # one-bin buffer around the call
    empty = np.zeros_like(bas)
    out = {"cm": empty, "cm_max": empty, "cm_max2": empty}

    _, mu = whiten_vec(bas)
    with np.errstate(divide="ignore", invalid="ignore"):
        qre = bas / mu - 1
    passed = qre > cp.contour_cutoff
    if passed.sum() > cp.min_island_size:
        # dm = conv2(msk0, ones(3,3)/9, 'same') > 2/9, i.e. at least 3 of 9; then OR the mask.
        counts = ndimage.convolve(passed.astype(int), _SQUARE, mode="constant")
        mask = (counts >= 3) | passed
        labels, order = _islands(mask)
        if order:
            area = ndimage.sum_labels(np.ones_like(bas), labels, order)
            energy = ndimage.sum_labels(bas, labels, order)
            if area.max() > cp.min_island_size:
                kept = [lab for lab, a in zip(order, area) if a > cp.smaller_island_cutoff * area.max()]
                by_energy = [order[i] for i in np.argsort(energy, kind="stable")]  # weakest first
                out["cm_max"] = np.where(labels == by_energy[-1], bas, 0.0)
                if len(by_energy) > 1:
                    out["cm_max2"] = np.where(labels == by_energy[-2], bas, 0.0)
                out["cm"] = np.where(np.isin(labels, kept), bas, 0.0)
    return {name: c[1:-1, 1:-1] for name, c in out.items()}


def _islands(mask: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Islands of ``mask`` as ``island1x.m`` finds them, and their labels in its order.

    ``island1x`` joins linear (column-major) indices ``k +- 1`` and
    ``k +- rows``: direct neighbours, plus the last row of one column with the
    first row of the next. Its islands come in order of their first index.
    """
    labels, n = ndimage.label(mask, structure=_CROSS)
    if n == 0:
        return labels, []
    parent = np.arange(n + 1)

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in zip(labels[-1, :-1], labels[0, 1:]):  # column wrap
        if a and b:
            ra, rb = root(int(a)), root(int(b))
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    roots = np.array([root(i) for i in range(n + 1)])
    labels = roots[labels]
    # Order by each island's first column-major index.
    flat = labels.ravel(order="F")
    nz = np.flatnonzero(flat)
    _, first = np.unique(flat[nz], return_index=True)
    order = [int(flat[nz[i]]) for i in sorted(first)]
    return labels, order


# ── sparse storage (GPL_sparse.m / GPL_full.m) ───────────────────────────────


def to_sparse(contour: np.ndarray) -> dict[str, Any]:
    """``values`` (uint16 of the peak), 1-based column-major ``index`` (uint32), ``scale`` and ``size``."""
    flat = contour.ravel(order="F")
    index = np.flatnonzero(flat)
    values = flat[index]
    peak = float(values.max()) if values.size else float("nan")
    if values.size:
        # uint16(round(v / peak * 65535)): round half away from zero, saturate to [0, 65535].
        q = values / peak * _Q
        q = np.sign(q) * np.floor(np.abs(q) + 0.5)
        ivalues = np.clip(q, 0, _Q).astype(np.uint16)
    else:
        ivalues = np.zeros(0, dtype=np.uint16)
    return {
        "values": ivalues,
        "index": (index + 1).astype(np.uint32),
        "scale": peak,
        "size": np.array(contour.shape, dtype=np.int64),
    }


def from_sparse(sparse: dict[str, Any]) -> np.ndarray:
    """The contour rebuilt from ``to_sparse``."""
    rows, cols = (int(v) for v in sparse["size"])
    flat = np.zeros(rows * cols)
    if len(sparse["values"]):
        flat[np.asarray(sparse["index"], dtype=np.int64) - 1] = (
            np.asarray(sparse["values"], dtype=float) * sparse["scale"] / _Q
        )
    return flat.reshape((rows, cols), order="F")


# ── measurements (GPL_measurements.m) ────────────────────────────────────────


def measure_contour(contour: np.ndarray, name: str, parm: DetectorParams) -> dict[str, float]:
    """The measurements ``parm.contour.measure`` turns on for contour ``name`` (``cm``, ``cm_max``, ``cm_max2``)."""
    m = parm.contour.measure
    df = parm.sample_freq / parm.fftl
    dt = parm.fft_overlap / parm.sample_freq
    out: dict[str, float] = {}
    cols = np.flatnonzero(contour.sum(axis=0))
    rows = np.flatnonzero(contour.sum(axis=1))
    if m.on(name, "slope") or m.on(name, "duration"):
        out["duration_sec"] = len(cols) * dt
        out["duration_bin"] = float(len(cols))
    if m.on(name, "slope"):
        ci, slope, y_int = _slope(contour, len(cols))
        out["slope_ci"] = ci
        out["slope"] = slope * df / dt
        out["slope_y_int"] = y_int + parm.freq_bin_lo - 1
    if m.on(name, "freq_bandwidth"):
        out["freq_bandwidth_hz"] = len(rows) * df
        out["freq_bandwidth_bin"] = float(len(rows))
    if m.on(name, "freq_limits"):
        out.update(_freq_limits(contour, cols, rows, parm))
    return out


def _slope(contour: np.ndarray, n_cols: int) -> tuple[float, float, float]:
    """``GPL_slope.m``: weighted poly1 fit of row on column (1-based), weights the squared values."""
    if n_cols <= 2:
        return float("nan"), float("nan"), float("nan")
    r, c = np.nonzero(contour)
    y, x, w = r + 1.0, c + 1.0, contour[r, c] ** 2
    X = np.column_stack((x, np.ones_like(x)))
    xtw = X.T * w
    a = xtw @ X
    beta = np.linalg.solve(a, xtw @ y)
    dfe = len(y) - 2
    resid = y - X @ beta
    mse = float(w @ resid**2) / dfe
    se = np.sqrt(mse * np.linalg.inv(a)[0, 0])
    ci = 2 * stats.t.ppf(0.975, dfe) * se  # |upper - lower| of confint(fit, 0.95) for p1
    return float(ci), float(beta[0]), float(beta[1])


def _freq_limits(contour: np.ndarray, cols: np.ndarray, rows: np.ndarray, parm: DetectorParams) -> dict[str, float]:
    df = parm.sample_freq / parm.fftl
    lo = parm.freq_bin_lo
    nan = float("nan")
    if cols.size == 0:
        return dict.fromkeys(("start_freq_hz", "end_freq_hz", "min_freq_hz", "max_freq_hz", "abs_bandwidth_hz"), nan)
    start_row = np.mean(np.flatnonzero(contour[:, cols[0]]) + 1)
    end_row = np.mean(np.flatnonzero(contour[:, cols[-1]]) + 1)
    max_freq = (rows[-1] + 1 + lo - 2) * df
    min_freq = (rows[0] + 1 + lo - 2) * df
    return {
        "start_freq_hz": float((start_row + lo) * df - 2),
        "end_freq_hz": float((end_row + lo) * df - 2),
        "min_freq_hz": float(min_freq),
        "max_freq_hz": float(max_freq),
        "abs_bandwidth_hz": float(max_freq - min_freq + 1),
    }


def spec_noise(quiet_fft: np.ndarray) -> float:
    """The window's quiet level: ``sqrt(mean(sum(2 * quiet_fft.^2, 1)))``."""
    q = np.asarray(quiet_fft, dtype=float)
    if q.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.sum(2 * q**2, axis=0))))


def estimate_rl(cm: np.ndarray, call_sp: np.ndarray) -> float:
    """``estimate_rl.m``: RMS over columns of the contour's level above the nearest noise pixel."""
    on = cm != 0
    new_slate = np.where(on, call_sp, 0.0)
    near = ndimage.convolve((new_slate != 0).astype(int), _SQUARE, mode="constant") > 0
    noise = np.where(near, 0.0, call_sp)
    nr, nc = np.nonzero(noise)
    if nr.size == 0:
        return float("nan")
    cr, cc = np.nonzero(on)
    _, nearest = cKDTree(np.column_stack((nr, nc))).query(np.column_stack((cr, cc)))
    new_slate[cr, cc] -= noise[nr[nearest], nc[nearest]]
    new_slate[new_slate < 0] = 0
    return float(np.sqrt(np.mean(np.sum(new_slate**2, axis=0))))


# ── filter flags (GPL_filter.m) ──────────────────────────────────────────────


def filter_flags(measures: dict[str, float], name: str, parm: DetectorParams) -> dict[str, int]:
    """1 where contour ``name`` passes ``parm.contour.limits[name]``, 0 where not; only measured values are tested.

    As in MATLAB, a NaN measurement passes every test but the slope test.
    """
    lim = _resolved(parm.contour.limits[name], parm)
    tests = {
        "duration_sec": lambda v: not v < lim.min_duration_s,
        "freq_bandwidth_hz": lambda v: not v < lim.min_freq_bandwidth_hz,
        "start_freq_hz": lambda v: not v < lim.start_freq_hz,
        "end_freq_hz": lambda v: not v > lim.end_freq_hz,
        "min_freq_hz": lambda v: not v > lim.min_freq_hz,
        "max_freq_hz": lambda v: not v < lim.max_freq_hz,
        "slope": lambda v: lim.slope_lower < v < lim.slope_upper,
    }
    return {key: int(test(measures[key])) for key, test in tests.items() if key in measures}


def _resolved(lim: ContourLimits, parm: DetectorParams) -> ContourLimits:
    def pick(value: float | None, default: float) -> float:
        return default if value is None else value

    return ContourLimits(
        min_duration_s=pick(lim.min_duration_s, parm.min_call_duration),
        min_freq_bandwidth_hz=lim.min_freq_bandwidth_hz,
        start_freq_hz=pick(lim.start_freq_hz, parm.freq_lo),
        end_freq_hz=pick(lim.end_freq_hz, parm.freq_hi),
        min_freq_hz=pick(lim.min_freq_hz, parm.freq_lo),
        max_freq_hz=pick(lim.max_freq_hz, parm.freq_hi),
        slope_lower=lim.slope_lower,
        slope_upper=lim.slope_upper,
    )


# ── one call (GPL_template + GPL_measurements + GPL_filter) ──────────────────


def describe_call(
    call_whiten: np.ndarray,
    call_sp: np.ndarray,
    window_noise: float,
    parm: DetectorParams,
) -> dict[str, Any]:
    """The contour fields of one detection, as ``GPL_v31`` adds them to ``GPL_struct(k)``.

    ``call_whiten`` / ``call_sp``: the call's columns of the whitened and the
    unwhitened (band-cropped) spectrogram; ``window_noise``: ``spec_noise`` of
    the window's quiet spectrogram. Returns ``cm`` / ``cm_max`` / ``cm_max2``
    (each the sparse contour plus its measurements) for the contours turned on,
    ``spec_noise`` / ``spec_rl`` when measured, and ``filter_v3``: per contour,
    its pass flags.
    """
    cp = parm.contour
    out: dict[str, Any] = {}
    rebuilt: dict[str, np.ndarray] = {}
    if cp.template_on:
        found = find_contours(call_whiten, parm)
        for name in CONTOURS:
            if cp.contour_on(name):
                out[name] = to_sparse(found[name])
                rebuilt[name] = from_sparse(out[name])
    if cp.measurements_on:
        for name, contour in rebuilt.items():
            out[name].update(measure_contour(contour, name, parm))
        if cp.measure.spec_noise:
            out["spec_noise"] = window_noise
        if cp.measure.spec_rl and "cm" in rebuilt:
            cm = rebuilt["cm"]
            enough = np.count_nonzero(cm) > cp.measure.rl_min_size
            out["spec_rl"] = estimate_rl(cm, np.asarray(call_sp, dtype=float)) if enough else float("nan")
    if cp.filter_on and rebuilt:
        out["filter_v3"] = {name: filter_flags(out[name], name, parm) for name in rebuilt}
    return out



def table_columns(parm: DetectorParams) -> list[str]:
    """The per-call measurement columns the CSV gets for ``parm`` (``<contour>_<measure>``, flags ``filter_<contour>_<measure>``)."""
    cp = parm.contour
    if not cp.measurements_on:
        return []
    m = cp.measure
    cols: list[str] = []
    flags: list[str] = []
    for name in CONTOURS:
        if not (cp.template_on and cp.contour_on(name)):
            continue
        keys: list[str] = []
        if m.on(name, "slope") or m.on(name, "duration"):
            keys += ["duration_sec", "duration_bin"]
        if m.on(name, "slope"):
            keys += ["slope", "slope_ci", "slope_y_int"]
        if m.on(name, "freq_bandwidth"):
            keys += ["freq_bandwidth_hz", "freq_bandwidth_bin"]
        if m.on(name, "freq_limits"):
            keys += ["start_freq_hz", "end_freq_hz", "min_freq_hz", "max_freq_hz", "abs_bandwidth_hz"]
        cols += [f"{name}_{k}" for k in keys]
        if cp.filter_on:
            testable = {"duration_sec", "freq_bandwidth_hz", "start_freq_hz", "end_freq_hz",
                        "min_freq_hz", "max_freq_hz", "slope"}
            flags += [f"filter_{name}_{k}" for k in keys if k in testable]
    if m.spec_noise:
        cols.append("spec_noise")
    if m.spec_rl and cp.template_on and cp.cm_on:
        cols.append("spec_rl")
    return cols + flags


def table_row(call: dict[str, Any], columns: list[str]) -> dict[str, Any]:
    """``call``'s values for ``table_columns`` (NaN where missing)."""
    row: dict[str, Any] = {}
    for col in columns:
        value: Any = float("nan")
        if col.startswith("filter_"):
            name, key = _split(col[len("filter_"):])
            value = call.get("filter_v3", {}).get(name, {}).get(key, value)
        elif col in ("spec_noise", "spec_rl"):
            value = call.get(col, value)
        else:
            name, key = _split(col)
            value = call.get(name, {}).get(key, value)
        row[col] = value
    return row


def _split(col: str) -> tuple[str, str]:
    for name in sorted(CONTOURS, key=len, reverse=True):  # cm_max2 before cm_max before cm
        if col.startswith(name + "_"):
            return name, col[len(name) + 1:]
    raise KeyError(col)
