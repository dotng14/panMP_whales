"""Check the wav -> matrix pipeline of Experiment 2: correct, and what it keeps or loses.

    python -m analysis.song_hmm.exp2_phrase_hmm.pipeline_check.check

Sections follow plan.md: A file, B frequency content, C STFT framing, D resolution,
E whitening, F windows / descriptors / PCA. Writes reports/exp2_pipeline_check.md, figures/ and
work/ (results.json, Griffin-Lim audio). Uses the real functions in features.py.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import signal  # noqa: E402
from scipy.io import wavfile  # noqa: E402

from analysis.song_hmm.report_paths import write_report  # noqa: E402
from analysis.song_hmm.exp1_slot_hmm.labels import load_labels, recording_start_unix  # noqa: E402
from analysis.song_hmm.exp1_slot_hmm.train import CSV_PATH, SPLIT_S, WAV_PATH  # noqa: E402

from .. import features as F  # noqa: E402

HERE = Path(__file__).resolve().parent
FIG = HERE / "figures"
WORK = HERE / "work"
SR = 2000
FPS = 20
N_S = 36000
SEMITONES_PER_BAND = 12 * np.log2((F.F_HI / F.F_LO) ** (1 / F.N_BANDS_EDGES))
SONG_PERIOD_MIN = 13.0  # from diagnostics/boundary_checks.txt


# ================================================================ helpers


def read_audio():
    sr, a = wavfile.read(WAV_PATH, mmap=True)
    return sr, a


def riff_chunks(path: Path) -> list[tuple[str, int, int]]:
    """(chunk id, size, offset) for every top-level RIFF chunk."""
    out = []
    with path.open("rb") as h:
        riff, _, wave = struct.unpack("<4sI4s", h.read(12))
        assert riff == b"RIFF" and wave == b"WAVE"
        while True:
            hdr = h.read(8)
            if len(hdr) < 8:
                break
            cid, size = struct.unpack("<4sI", hdr)
            out.append((cid.decode("latin-1"), size, h.tell() - 8))
            h.seek(size + (size & 1), 1)
    return out


def minute_levels(a: np.ndarray) -> np.ndarray:
    """30-1000 Hz power per minute (dB), from a Welch PSD of each minute."""
    lv = []
    for m in range(N_S // 60):
        f, p = signal.welch(np.asarray(a[m * 60 * SR : (m + 1) * 60 * SR], np.float64), SR, nperseg=2048)
        lv.append(10 * np.log10(p[(f >= 30) & (f <= 1000)].sum() + 1e-12))
    return np.array(lv)


def runs_of(mask: np.ndarray) -> np.ndarray:
    e = np.flatnonzero(np.diff(np.r_[0, mask.astype(np.int8), 0])).reshape(-1, 2)
    return e


# ================================================================ A. file


def check_file(a: np.ndarray) -> dict:
    chunks = riff_chunks(WAV_PATH)
    data = [c for c in chunks if c[0] == "data"][0]
    size = WAV_PATH.stat().st_size
    n = a.shape[0]
    clip = 0
    zero_runs = []
    means, rmss = [], []
    step = 600 * SR
    for i in range(0, n, step):
        x = np.asarray(a[i : i + step])
        clip += int(np.sum((x >= 32767) | (x <= -32768)))
        xf = x.astype(np.float64)
        means.append(xf.mean())
        rmss.append(np.sqrt(np.mean((xf - xf.mean()) ** 2)))
        flat = np.r_[False, np.diff(x) == 0]
        for s, e in runs_of(flat):
            if e - s >= 0.1 * SR:
                zero_runs.append({"t_s": (i + s) / SR, "length_s": (e - s) / SR, "value": int(x[s])})
    labels = load_labels(CSV_PATH, N_S)
    means, rmss = np.array(means), np.array(rmss)
    x = np.asarray(a[(SPLIT_S + 7200) * SR : (SPLIT_S + 7800) * SR], np.float64)
    with_dc = F.band_spectrogram(x, SR, 600 * FPS)
    no_dc = F.band_spectrogram(x - x.mean(), SR, 600 * FPS)
    dc_leak_db = float(np.max((with_dc - no_dc).mean(0)) * 10 / np.log(10))
    return {
        "chunks": [{"id": c, "size": s, "offset": o} for c, s, o in chunks],
        "file_size": size,
        "data_bytes": data[1],
        "data_bytes_match": data[1] == n * 2 and data[2] + 8 + data[1] <= size,
        "sample_rate": SR,
        "n_samples": int(n),
        "duration_s": n / SR,
        "dtype": str(a.dtype),
        "clipped_fraction": clip / n,
        "dc_over_rms_max": float(np.max(np.abs(means) / np.maximum(rmss, 1e-9))),
        "dc_mean_counts": float(np.mean(means)),
        "dc_mean_range": [float(means.min()), float(means.max())],
        "ac_rms_median": float(np.median(rmss)),
        "dc_leak_max_db_in_any_band": dc_leak_db,
        "constant_runs": zero_runs[:50],
        "n_constant_runs": len(zero_runs),
        "csv_first_unit_s": float(labels.units[:, 0].min()),
        "start_utc_from_name": recording_start_unix(WAV_PATH.name),
    }


# ================================================================ B. spectrum


def check_spectrum(a: np.ndarray, levels: np.ndarray) -> dict:
    psd = []
    for m in range(N_S // 60):
        f, p = signal.welch(np.asarray(a[m * 60 * SR : (m + 1) * 60 * SR], np.float64), SR, nperseg=2048)
        psd.append(p)
    psd = np.array(psd)
    loud = levels >= np.percentile(levels, 80)
    quiet = levels <= np.percentile(levels, 20)
    pl, pq = np.median(psd[loud], 0), np.median(psd[quiet], 0)
    excess = np.maximum(pl - pq, 0)
    inside = (f >= 30) & (f <= 1000)
    below = f < 30
    ref = np.median(10 * np.log10(pq[(f >= 500) & (f <= 800)]))
    q_db = 10 * np.log10(pq)
    roll = f[(f > 800) & (q_db < ref - 10)]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for p, name, c in ((psd[loud], "loud minutes (top 20%)", "tab:red"), (psd[quiet], "quiet minutes (bottom 20%)", "tab:blue")):
        lo, md, hi = np.percentile(10 * np.log10(p), [10, 50, 90], axis=0)
        ax.plot(f, md, color=c, label=name)
        ax.fill_between(f, lo, hi, color=c, alpha=0.15)
    ax.axvspan(0, 30, color="0.85", label="outside 30-1000 Hz")
    ax.set(xlabel="Hz", ylabel="dB (counts²/Hz)", title="Long-term spectrum (10th-50th-90th percentile over minutes)", xlim=(0, 1000))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "B_spectrum.png", dpi=130)
    plt.close(fig)
    return {
        "song_excess_outside_fraction": float(excess[below].sum() / max(excess[inside].sum(), 1e-12)),
        "loud_energy_below_30_fraction": float(pl[below].sum() / pl[inside | below].sum()),
        "quiet_energy_below_30_fraction": float(pq[below].sum() / pq[inside | below].sum()),
        "rolloff_10db_hz": float(roll.min()) if roll.size else None,
        "loud_minus_quiet_db_by_band": {f"{lo}-{hi}": float(10 * np.log10(pl[(f >= lo) & (f < hi)].sum() / pq[(f >= lo) & (f < hi)].sum())) for lo, hi in ((0, 30), (30, 100), (100, 300), (300, 600), (600, 1000))},
        "loud_minutes": np.flatnonzero(loud).tolist(),
        "quiet_minutes": np.flatnonzero(quiet).tolist(),
    }


# ================================================================ C. STFT


def istft_snr(x: np.ndarray) -> float:
    _, _, z = signal.stft(x, SR, window="hann", nperseg=F.NFFT, noverlap=F.NFFT - F.HOP, boundary="even", padded=True)
    _, y = signal.istft(z, SR, window="hann", nperseg=F.NFFT, noverlap=F.NFFT - F.HOP, boundary=True)
    y = y[: len(x)]
    return float(10 * np.log10(np.sum(x**2) / max(np.sum((x - y) ** 2), 1e-30)))


def tone_burst_frame(t0: float = 10.3, dur: float = 0.1, freq: float = 300.0, seconds: int = 30) -> tuple[int, int]:
    """(frame of the burst's peak, frame expected from the frame-centre convention)."""
    a = np.random.default_rng(0).normal(0, 1, seconds * SR)
    t = np.arange(int(dur * SR)) / SR
    i0 = int(t0 * SR)
    a[i0 : i0 + t.size] += 2000 * np.sin(2 * np.pi * freq * t)
    s = F.band_spectrogram(a, SR, seconds * FPS)
    _, edges = F.band_matrix(SR)
    b = int(np.searchsorted(edges[:, 0], freq, side="right") - 1)
    centre = t0 + dur / 2
    return int(np.argmax(s[:, b])), int(round((centre - F.NFFT / 2 / SR) / (F.HOP / SR)))


def check_stft(a: np.ndarray) -> dict:
    x = np.asarray(a[(SPLIT_S + 7200) * SR : (SPLIT_S + 7800) * SR], np.float64)
    snr = istft_snr(x)
    seg = np.asarray(a[1700 * SR : 1940 * SR])
    one = F.band_spectrogram(seg, SR, 240 * FPS, chunk=10**9)
    chunked = F.band_spectrogram(seg, SR, 240 * FPS, chunk=1237)
    got, want = tone_burst_frame()
    m, _ = F.band_matrix(SR)
    freqs = np.fft.rfftfreq(F.NFFT, 1 / SR)
    inside = (freqs >= F.F_LO) & (freqs < F.F_HI)
    p = np.random.default_rng(1).random((50, freqs.size))
    return {
        "istft_snr_db": snr,
        "chunked_max_abs_diff": float(np.max(np.abs(one - chunked))),
        "tone_frame": got,
        "tone_frame_expected": want,
        "bins_in_exactly_one_band": bool(np.all(m[inside].sum(1) == 1) and np.all(m[~inside].sum(1) == 0)),
        "band_power_identity_rel_err": float(np.max(np.abs((p @ m).sum(1) - p[:, inside].sum(1)) / p[:, inside].sum(1))),
        "n_bands": int(m.shape[1]),
        "first_band_edges_hz": [float(e) for e in F.band_matrix(SR)[1][0]],
        "frame_time_convention_offset_s": F.NFFT / 2 / SR,
    }


# ================================================================ D. resolution


def envelope_200hz(x: np.ndarray) -> np.ndarray:
    sos = signal.butter(4, [30, 990], btype="band", fs=SR, output="sos")
    env = np.abs(signal.sosfiltfilt(sos, x))
    return env[: len(env) // 10 * 10].reshape(-1, 10).mean(1)


def modulation_spectrum(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    env = envelope_200hz(x)
    f, p = signal.welch(env - env.mean(), 200, nperseg=2000)
    return f, p


def mtf(nfft: int, hop: int, mods=np.arange(0.5, 20.5, 0.5)) -> np.ndarray:
    """Modulation depth kept in the 400 Hz band series for a 100%-AM 400 Hz tone, per modulation rate."""
    out = []
    t = np.arange(20 * SR) / SR
    fps = SR // hop
    _, edges = F.band_matrix(SR, nfft)
    b = int(np.searchsorted(edges[:, 0], 400, side="right") - 1)
    for fm in mods:
        x = np.sin(2 * np.pi * 400 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * fm * t))
        p = np.exp(F.band_spectrogram(x, SR, 18 * fps, nfft=nfft, hop=hop)[2 * fps : 16 * fps, b])
        amp = np.sqrt(p)
        out.append((amp.max() - amp.min()) / (amp.max() + amp.min()))
    return np.array(out)


def chirp_concentration(nfft: int, hop: int) -> float:
    """Share of a 200->800 Hz, 0.5 s up-sweep's band energy in the single strongest band per frame."""
    t = np.arange(int(0.5 * SR)) / SR
    x = np.zeros(4 * SR)
    x[SR : SR + t.size] = signal.chirp(t, 200, 0.5, 800, method="logarithmic")
    fps = SR // hop
    s = np.exp(F.band_spectrogram(x, SR, 4 * fps, nfft=nfft, hop=hop))
    on = s.sum(1) > 0.1 * s.sum(1).max()
    return float(np.mean(s[on].max(1) / s[on].sum(1)))


def check_resolution(a: np.ndarray, loud_minutes: list[int], quiet_minutes: list[int]) -> dict:
    rng = np.random.default_rng(0)
    res = {}
    for name, mins in (("loud", loud_minutes), ("quiet", quiet_minutes)):
        ps = []
        for m in rng.choice(mins, size=min(40, len(mins)), replace=False):
            f, p = modulation_spectrum(np.asarray(a[m * 60 * SR : (m + 1) * 60 * SR], np.float64))
            ps.append(p)
        p = np.mean(ps, 0)
        keep = f >= 0.5
        res[f"mod_{name}"] = {"f": f[keep].tolist(), "p": p[keep].tolist()}
        tot = p[keep].sum()
        res[f"mod_share_{name}"] = {str(c): float(p[f >= c].sum() / tot) for c in (2, 5, 10, 20)}
    settings = {"a: 512 / 0.05 s (current)": (512, 100), "b: 256 / 0.025 s": (256, 50), "c: 1024 / 0.05 s": (1024, 100)}
    mods = np.arange(0.5, 20.5, 0.5)
    f = np.array(res["mod_loud"]["f"])
    p = np.array(res["mod_loud"]["p"])
    res["settings"] = {}
    for name, (nfft, hop) in settings.items():
        m = mtf(nfft, hop, mods)
        gain = np.interp(f, mods, m, right=0.0)
        res["settings"][name] = {
            "mtf": m.tolist(),
            "song_modulation_kept": float(np.sum(p * gain**2) / np.sum(p)),
            "depth_at_10hz": float(np.interp(10, mods, m)),
            "depth_at_5hz": float(np.interp(5, mods, m)),
            "chirp_concentration": chirp_concentration(nfft, hop),
            "freq_step_hz": SR / nfft,
        }
    res["mods"] = mods.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
    for name in ("loud", "quiet"):
        axes[0].semilogy(res[f"mod_{name}"]["f"], res[f"mod_{name}"]["p"], label=f"{name} minutes")
    axes[0].set(xlabel="modulation rate (Hz)", ylabel="power", title="Envelope modulation spectrum (30-1000 Hz)", xlim=(0, 40))
    axes[0].legend(fontsize=8)
    for name, v in res["settings"].items():
        axes[1].plot(mods, v["mtf"], label=name)
    axes[1].set(xlabel="modulation rate (Hz)", ylabel="depth kept (1 = all)", title="What each STFT setting keeps of amplitude modulation", ylim=(0, 1.05))
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "D_modulation.png", dpi=130)
    plt.close(fig)
    return res


def unit_and_sweep_stats(sw: np.ndarray, loud_minutes: list[int], threshold: float = 1.0) -> dict:
    """Unit lengths and ridge slopes in the whitened spectrogram during loud minutes.

    A frame is 'on' when its strongest band is more than ``threshold`` ln-units above the floor
    (1 = 4.3 dB, 2 = 8.7 dB, 3 = 13 dB)."""
    frames = np.concatenate([np.arange(m * 60 * FPS, (m + 1) * 60 * FPS) for m in loud_minutes if (m + 1) * 60 * FPS <= len(sw)])
    x = sw[frames]
    peak = x.max(1)
    on = peak > threshold
    contiguous = np.r_[True, np.diff(frames) == 1]
    on_runs = runs_of(on & contiguous)
    lengths = (on_runs[:, 1] - on_runs[:, 0]) / FPS
    ridge = x.argmax(1).astype(float)
    d = np.diff(ridge)
    ok = on[1:] & on[:-1] & (np.abs(d) <= 3) & contiguous[1:]
    slope = np.abs(d[ok]) * SEMITONES_PER_BAND * FPS  # semitones per second
    return {
        "unit_length_s_percentiles": {str(q): float(np.percentile(lengths, q)) for q in (10, 25, 50, 75, 90)},
        "share_units_shorter_than_window": float(np.mean(lengths < F.NFFT / SR)),
        "share_frames_on": float(on.mean()),
        "sweep_st_per_s_percentiles": {str(q): float(np.percentile(slope, q)) for q in (50, 75, 90)},
        "share_steps_with_band_change": float(np.mean(slope > 0)),
    }


# ================================================================ E. whitening

WHITEN_VARIANTS = {
    "a: median 5 min, clip (current)": dict(pct=50.0, n_blocks_window=5, clip=True),
    "b: 10th pct 5 min, clip": dict(pct=10.0, n_blocks_window=5, clip=True),
    "c: 10th pct 10 min, no clip": dict(pct=10.0, n_blocks_window=11, clip=False),
    "d: no whitening": None,
}


def whiten_variant(s_half: np.ndarray, spec) -> np.ndarray:
    if spec is None:
        return s_half - np.median(s_half, axis=0)  # only a global per-band offset
    return F.whiten(s_half, FPS, **spec)


def check_whitening(s: np.ndarray, loud_minutes, quiet_minutes, stretches: dict, q) -> tuple[dict, dict]:
    """Per variant: floor statistics, retrieval, and the example stretches (for the figure)."""
    t_q, t_c, q_idx, c_idx = q
    res = {}
    whitened = {}
    lm = np.zeros(N_S // 60, bool)
    lm[loud_minutes] = True
    qm = np.zeros(N_S // 60, bool)
    qm[quiet_minutes] = True
    for name, spec in WHITEN_VARIANTS.items():
        sw = np.vstack([whiten_variant(s[: SPLIT_S * FPS], spec), whiten_variant(s[SPLIT_S * FPS :], spec)])
        per_min = sw.reshape(N_S // 60, 60 * FPS, -1)
        above = per_min > 0
        mean_min = per_min.mean(axis=(1, 2))
        d = (mean_min[lm].mean() - mean_min[qm].mean()) / np.sqrt(0.5 * (mean_min[lm].var() + mean_min[qm].var()))
        band_above_loud = above[lm].mean(axis=(0, 1))
        # energy of the raw band power that the floor subtracts in loud minutes (clip variants lose it)
        removed = None
        if spec is not None and spec["clip"]:
            raw = np.exp(s.reshape(N_S // 60, 60 * FPS, -1)[lm])
            kept = np.exp(per_min[lm])  # ratio over floor, >= 1
            floor_lin = raw / np.maximum(kept, 1.0)  # floor where above; raw where clipped
            removed = float(np.sum(np.minimum(raw, floor_lin)) / np.sum(raw))
        res[name] = {
            "share_cells_above_floor_loud": float(above[lm].mean()),
            "share_cells_above_floor_quiet": float(above[qm].mean()),
            "weakest_band_above_share_loud": float(band_above_loud.min()),
            "weakest_band_hz": int(np.argmin(band_above_loud)),
            "loud_vs_quiet_separation_d": float(d),
            "quiet_mean_level": float(mean_min[qm].mean()),
            "loud_mean_level": float(mean_min[lm].mean()),
            "raw_power_under_floor_loud": removed,
        }
        r = representation(sw[: SPLIT_S * FPS], sw[SPLIT_S * FPS :], n_pc=(20,))
        u, uz = unit(r["pc20_test"]), unit(r["z_test"])
        res[name]["retrieval"] = {
            "PCA-20": retrieval(u[q_idx] @ u[c_idx].T, t_q, t_c),
            "295 descriptors": retrieval(uz[q_idx] @ uz[c_idx].T, t_q, t_c),
        }
        whitened[name] = {k: sw[t0 * FPS : t1 * FPS].copy() for k, (t0, t1) in stretches.items()}
        del sw, per_min, above
    return res, whitened


def griffin_lim(mag: np.ndarray, n_iter: int = 32, nfft: int = F.NFFT, hop: int = F.HOP) -> np.ndarray:
    rng = np.random.default_rng(0)
    phase = np.exp(2j * np.pi * rng.random(mag.shape))
    for _ in range(n_iter):
        _, x = signal.istft(mag * phase, SR, window="hann", nperseg=nfft, noverlap=nfft - hop)
        _, _, z = signal.stft(x, SR, window="hann", nperseg=nfft, noverlap=nfft - hop)
        z = z[:, : mag.shape[1]]
        if z.shape[1] < mag.shape[1]:
            z = np.pad(z, ((0, 0), (0, mag.shape[1] - z.shape[1])))
        phase = np.exp(1j * np.angle(z))
    _, x = signal.istft(mag * phase, SR, window="hann", nperseg=nfft, noverlap=nfft - hop)
    return x


def write_clip(x: np.ndarray, path: Path) -> None:
    y = signal.resample_poly(x, 4, 1)  # 8 kHz so any player opens it
    y = y / max(np.max(np.abs(y)), 1e-9) * 0.9
    wavfile.write(path, SR * 4, (y * 32767).astype(np.int16))


def listening_clips(a: np.ndarray, stretches: dict) -> list[str]:
    m, _ = F.band_matrix(SR)
    width = m.sum(0)
    out = []
    for name, (t0, t1) in stretches.items():
        x = np.asarray(a[t0 * SR : t1 * SR], np.float64)
        _, _, z = signal.stft(x, SR, window="hann", nperseg=F.NFFT, noverlap=F.NFFT - F.HOP)
        mag = np.abs(z)
        band = (mag.T**2) @ m  # (frames, B) band power
        spread = np.sqrt((band / width) @ m.T).T  # back to bins, flat inside each band
        sl = np.log(band + 1e-6)
        floor = np.median(sl, 0)
        white = np.sqrt((np.maximum(np.exp(np.maximum(sl - floor, 0)) - 1, 0) / width) @ m.T).T
        clips = {"0_original": x, "1_stft_magnitude": griffin_lim(mag), "2_59_bands": griffin_lim(spread), "3_whitened": griffin_lim(white)}
        for k, v in clips.items():
            p = WORK / "audio" / f"{name}_{k}.wav"
            p.parent.mkdir(parents=True, exist_ok=True)
            write_clip(v, p)
            out.append(str(p.relative_to(HERE)))
    return out


# ================================================================ F. windows, descriptors, PCA


def unit(z: np.ndarray) -> np.ndarray:
    z = z - z.mean(0)
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-9)


def repeat_targets(t_q: np.ndarray, t_c: np.ndarray) -> np.ndarray:
    """(Q, C) True where the candidate sits one or two song periods (+-2 min) from the query."""
    lag = np.abs(t_c[None, :] - t_q[:, None]) / 60
    return ((np.abs(lag - SONG_PERIOD_MIN) <= 2) | (np.abs(lag - 2 * SONG_PERIOD_MIN) <= 2))


def retrieval(sim: np.ndarray, t_q: np.ndarray, t_c: np.ndarray, min_gap_s: float = 300) -> dict:
    """Nearest neighbour among candidates >= min_gap_s away: share landing 1-2 song periods away."""
    far = np.abs(t_c[None, :] - t_q[:, None]) >= min_gap_s
    s = np.where(far, sim, -np.inf)
    nn = np.argmax(s, 1)
    tgt = repeat_targets(t_q, t_c)
    hit = tgt[np.arange(len(t_q)), nn]
    chance = (tgt & far).sum(1) / far.sum(1)
    return {"hit_rate": float(hit.mean()), "chance": float(chance.mean()), "lift": float(hit.mean() / chance.mean())}


def knn_overlap(sim_a: np.ndarray, sim_b: np.ndarray, t_q, t_c, k: int = 10, min_gap_s: float = 12) -> float:
    far = np.abs(t_c[None, :] - t_q[:, None]) >= min_gap_s
    ia = np.argsort(-np.where(far, sim_a, -np.inf), 1)[:, :k]
    ib = np.argsort(-np.where(far, sim_b, -np.inf), 1)[:, :k]
    return float(np.mean([len(set(x) & set(y)) / k for x, y in zip(ia, ib, strict=True)]))


def patch_similarity(sw: np.ndarray, starts_q: np.ndarray, starts_c: np.ndarray, w: int, max_shift_s: float = 3.0, ds: int = 8) -> np.ndarray:
    """Best cosine over time shifts (+-max_shift_s) between whitened patches, time-downsampled by ds."""
    def patches(starts):
        p = np.stack([sw[s : s + w] for s in starts]).astype(np.float32)
        p = p[:, : w // ds * ds].reshape(len(starts), w // ds, ds, -1).mean(2)
        return p - p.mean(axis=(1, 2), keepdims=True)

    q, c = patches(starts_q), patches(starts_c)
    rows = q.shape[1]
    max_k = int(round(max_shift_s * FPS / ds))
    best = np.full((len(q), len(c)), -np.inf, np.float32)
    for k in range(-max_k, max_k + 1):
        qa = q[:, max(0, k) : rows + min(0, k)]
        cb = c[:, max(0, -k) : rows + min(0, -k)]
        qf, cf = qa.reshape(len(q), -1), cb.reshape(len(c), -1)
        sim = (qf @ cf.T) / np.outer(np.linalg.norm(qf, axis=1) + 1e-9, np.linalg.norm(cf, axis=1) + 1e-9)
        best = np.maximum(best, sim)
    return best


def level_normalized(sw: np.ndarray) -> np.ndarray:
    """Divide each 12 s neighbourhood's spectrogram by its own mean (frame-wise rolling mean over 12 s)."""
    k = F.WINDOW_S * FPS
    c = np.cumsum(np.r_[0.0, sw.mean(1)])
    roll = np.empty(len(sw))
    for i in range(len(sw)):
        lo, hi = max(0, i - k // 2), min(len(sw), i + k // 2)
        roll[i] = (c[hi] - c[lo]) / (hi - lo)
    return (sw / (roll[:, None] + 0.1)).astype(np.float32)


def representation(sw_tr, sw_te, window_s=F.WINDOW_S, n_pc=(20, 40), fps=FPS) -> dict:
    st_tr, v_tr = F.descriptors(sw_tr, fps, window_s=window_s)
    st_te, v_te = F.descriptors(sw_te, fps, window_s=window_s)
    mu, sd = v_tr.mean(0), np.maximum(v_tr.std(0), 1e-6)
    z_tr, z_te = (v_tr - mu) / sd, (v_te - mu) / sd
    _, sv, vt = np.linalg.svd(z_tr - z_tr.mean(0), full_matrices=False)
    ev = np.cumsum(sv**2) / np.sum(sv**2)
    out = {"starts_test": st_te, "z_test": z_te, "explained_cum": ev}
    for n in n_pc:
        out[f"pc{n}_test"] = z_te @ vt[:n].T
    return out


def query_set(level_te: np.ndarray, starts_te: np.ndarray):
    """500 query windows from the louder half of the test windows; candidates every 4 s. Times = window centres."""
    rng = np.random.default_rng(0)
    t_all = SPLIT_S + starts_te / FPS + F.WINDOW_S / 2
    q_idx = np.sort(rng.choice(np.flatnonzero(level_te >= np.median(level_te)), 500, replace=False))
    c_idx = np.arange(0, len(starts_te), 2)
    return t_all[q_idx], t_all[c_idx], q_idx, c_idx


def check_windows(sw_tr: np.ndarray, sw_te: np.ndarray, level_te: np.ndarray) -> dict:
    res = {}
    base = representation(sw_tr, sw_te)
    st = base["starts_test"]
    t_q, t_c, q_idx, c_idx = query_set(level_te, st)
    w = F.WINDOW_S * FPS
    print("   patch similarity (shift-tolerant) ...", flush=True)
    sim_patch = patch_similarity(sw_te, st[q_idx], st[c_idx], w)
    sims = {"raw whitened patch (shift-tolerant)": sim_patch}
    zt = unit(base["z_test"])
    sims["295 descriptors"] = zt[q_idx] @ zt[c_idx].T
    for n in (20, 40):
        u = unit(base[f"pc{n}_test"])
        sims[f"PCA-{n}"] = u[q_idx] @ u[c_idx].T
    res["retrieval"] = {k: retrieval(v, t_q, t_c) for k, v in sims.items()}
    res["knn_overlap_vs_patch"] = {k: knn_overlap(v, sim_patch, t_q, t_c) for k, v in sims.items() if k != "raw whitened patch (shift-tolerant)"}
    ev = base["explained_cum"]
    res["pca_components_for"] = {str(p): int(np.searchsorted(ev, p / 100) + 1) for p in (64, 80, 90, 95)}
    res["pca_explained_20"] = float(ev[19])

    # window length
    res["window_length"] = {}
    for ws in (8, 12, 20):
        r = representation(sw_tr, sw_te, window_s=ws, n_pc=(20,))
        stw = r["starts_test"]
        tw = SPLIT_S + stw / FPS + ws / 2
        qi = np.searchsorted(tw, t_q)
        ci = np.searchsorted(tw, t_c)
        qi, ci = np.clip(qi, 0, len(tw) - 1), np.clip(ci, 0, len(tw) - 1)
        u = unit(r["pc20_test"])
        uz = unit(r["z_test"])
        res["window_length"][f"{ws} s"] = {
            "PCA-20": retrieval(u[qi] @ u[ci].T, tw[qi], tw[ci]),
            "295 descriptors": retrieval(uz[qi] @ uz[ci].T, tw[qi], tw[ci]),
        }
    # phrase period inside windows: dominant autocorrelation lag of the patch over 1.5-20 s
    periods = []
    for s in st[q_idx]:
        seg = sw_te[s : s + 30 * FPS]
        if len(seg) < 30 * FPS:
            continue
        e = seg - seg.mean(0)
        lags = np.arange(int(1.5 * FPS), 20 * FPS)
        ac = [np.sum(e[:-lg] * e[lg:]) / np.sqrt(np.sum(e[:-lg] ** 2) * np.sum(e[lg:] ** 2) + 1e-9) for lg in lags]
        periods.append(lags[int(np.argmax(ac))] / FPS)
    periods = np.array(periods)
    res["phrase_period_s_percentiles"] = {str(q): float(np.percentile(periods, q)) for q in (10, 25, 50, 75, 90)}
    res["share_phrase_period_under_12s"] = float(np.mean(periods < 12))
    res["_q"] = (t_q, t_c, q_idx, c_idx)
    return res


def check_level_norm(sw_tr, sw_te, t_q, t_c, q_idx, c_idx) -> dict:
    r = representation(level_normalized(sw_tr), level_normalized(sw_te), n_pc=(20,))
    nb = sw_tr.shape[1]
    out = {"retrieval": {}}
    u = unit(r["z_test"])
    out["retrieval"]["295 descriptors"] = retrieval(u[q_idx] @ u[c_idx].T, t_q, t_c)
    u = unit(r["pc20_test"])
    out["retrieval"]["PCA-20"] = retrieval(u[q_idx] @ u[c_idx].T, t_q, t_c)
    return out | {"n_bands": nb}


COMBOS = {
    "1 current: 512/0.05 s, median floor, 12 s": (512, 100, "median", 12),
    "2 short FFT: 256/0.025 s, median floor, 12 s": (256, 50, "median", 12),
    "3 no whitening: 512/0.05 s, 12 s": (512, 100, "none", 12),
    "4 10th-pct floor: 512/0.05 s, 12 s": (512, 100, "p10", 12),
    "5 short FFT + no whitening, 12 s": (256, 50, "none", 12),
    "6 short FFT + no whitening, 20 s": (256, 50, "none", 20),
    "7 short FFT + 10th-pct floor, 20 s": (256, 50, "p10", 20),
}


def _floor(s_half, fps, kind):
    if kind == "median":
        return F.whiten(s_half, fps)
    if kind == "p10":
        return F.whiten(s_half, fps, pct=10.0)
    return s_half - np.median(s_half, axis=0)


def check_combinations(a: np.ndarray, t_q: np.ndarray, t_c: np.ndarray) -> dict:
    """Song-repeat retrieval with the settings combined (same queries as section F)."""
    out = {}
    for name, (nfft, hop, floor, ws) in COMBOS.items():
        fps = SR // hop
        s = F.band_spectrogram(np.asarray(a), SR, N_S * fps, nfft=nfft, hop=hop)
        sw_tr, sw_te = _floor(s[: SPLIT_S * fps], fps, floor), _floor(s[SPLIT_S * fps :], fps, floor)
        del s
        r = representation(sw_tr, sw_te, window_s=ws, n_pc=(20, 40), fps=fps)
        tw = SPLIT_S + r["starts_test"] / fps + ws / 2
        qi = np.clip(np.searchsorted(tw, t_q), 0, len(tw) - 1)
        ci = np.clip(np.searchsorted(tw, t_c), 0, len(tw) - 1)
        res = {"n_bands": int(sw_tr.shape[1])}
        for key, z in (("295-type descriptors", r["z_test"]), ("PCA-20", r["pc20_test"]), ("PCA-40", r["pc40_test"])):
            u = unit(z)
            res[key] = retrieval(u[qi] @ u[ci].T, tw[qi], tw[ci])
        out[name] = res
        print(f"   {name}: PCA-20 hit {res['PCA-20']['hit_rate']:.3f}", flush=True)
        del sw_tr, sw_te
    return out


# ================================================================ figures per stage


def stage_figure(a: np.ndarray, s: np.ndarray, sw: np.ndarray, stretches: dict) -> None:
    fig, axes = plt.subplots(3, len(stretches), figsize=(5 * len(stretches), 8))
    for j, (name, (t0, t1)) in enumerate(stretches.items()):
        x = np.asarray(a[t0 * SR : t1 * SR], np.float64)
        f, t, z = signal.stft(x, SR, window="hann", nperseg=F.NFFT, noverlap=F.NFFT - F.HOP)
        axes[0, j].pcolormesh(t, f, 10 * np.log10(np.abs(z) ** 2 + 1e-9), shading="auto", cmap="magma", vmin=np.percentile(10 * np.log10(np.abs(z) ** 2 + 1e-9), 30))
        axes[0, j].set_title(f"{name}: STFT (3.9 Hz bins)", fontsize=9)
        axes[1, j].imshow(s[t0 * FPS : t1 * FPS].T, aspect="auto", origin="lower", cmap="magma", extent=(0, t1 - t0, 0, s.shape[1]))
        axes[1, j].set_title("I-1: 59 log bands", fontsize=9)
        axes[2, j].imshow(sw[t0 * FPS : t1 * FPS].T, aspect="auto", origin="lower", cmap="magma", vmin=0, vmax=np.percentile(sw[t0 * FPS : t1 * FPS], 99.5), extent=(0, t1 - t0, 0, s.shape[1]))
        axes[2, j].set_title("I-2: whitened (current)", fontsize=9)
        axes[2, j].set_xlabel("s")
    fig.tight_layout()
    fig.savefig(FIG / "stages.png", dpi=110)
    plt.close(fig)


def whitening_figure(whitened: dict, stretches: dict) -> None:
    fig, axes = plt.subplots(len(whitened), len(stretches), figsize=(5 * len(stretches), 2.3 * len(whitened)))
    for i, (vname, parts) in enumerate(whitened.items()):
        for j, (name, (t0, t1)) in enumerate(stretches.items()):
            blk = parts[name]
            axes[i, j].imshow(blk.T, aspect="auto", origin="lower", cmap="magma", vmin=np.percentile(blk, 5), vmax=np.percentile(blk, 99.5), extent=(0, t1 - t0, 0, blk.shape[1]))
            axes[i, j].set_title(f"{vname} | {name}", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "E_whitening_variants.png", dpi=100)
    plt.close(fig)


# ================================================================ main


def pick_stretches(levels: np.ndarray) -> dict:
    labels = load_labels(CSV_PATH, N_S)
    starts = labels.song_bounds[:, 0]
    b = int(starts[(starts > SPLIT_S + 3600)][0])
    th = labels.theme
    runs = [(s, e) for s, e in runs_of(th == 5) if s >= SPLIT_S]
    s6, e6 = max(runs, key=lambda r: r[1] - r[0])
    mid = (s6 + e6) // 2
    q = int(np.argmin(levels[:30])) * 60
    return {"song_boundary": (b - 60, b + 60), "theme6": (int(mid - 60), int(mid + 60)), "quiet": (q, q + 120)}


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    sr, a = read_audio()
    assert sr == SR
    R: dict = {}
    print("A file ...", flush=True)
    R["A"] = check_file(a)
    print("B spectrum ...", flush=True)
    levels = minute_levels(a)
    R["B"] = check_spectrum(a, levels)
    loud, quiet = R["B"]["loud_minutes"], R["B"]["quiet_minutes"]
    print("C STFT ...", flush=True)
    R["C"] = check_stft(a)
    print("D resolution ...", flush=True)
    R["D"] = check_resolution(a, loud, quiet)
    s = F.band_spectrogram(a, SR, N_S * FPS)
    sw_cur = np.vstack([F.whiten(s[: SPLIT_S * FPS], FPS), F.whiten(s[SPLIT_S * FPS :], FPS)])
    R["D"]["units_and_sweeps"] = {str(th): unit_and_sweep_stats(sw_cur, loud, th) for th in (1.0, 2.0, 3.0)}
    stretches = pick_stretches(levels)
    R["stretches"] = stretches
    stage_figure(a, s, sw_cur, stretches)
    print("F windows, descriptors, PCA ...", flush=True)
    feats = F.load_or_build(WAV_PATH, SPLIT_S, HERE.parent / "work" / "phrase_features.npz")
    R["F"] = check_windows(sw_cur[: SPLIT_S * FPS], sw_cur[SPLIT_S * FPS :], feats["level_test"])
    q = R["F"].pop("_q")
    R["F"]["level_normalized"] = check_level_norm(sw_cur[: SPLIT_S * FPS], sw_cur[SPLIT_S * FPS :], *q)
    del sw_cur
    print("E whitening ...", flush=True)
    R["E"], whitened = check_whitening(s, loud, quiet, stretches, q)
    whitening_figure(whitened, stretches)
    print("   listening clips ...", flush=True)
    R["E_audio"] = listening_clips(a, stretches)
    print("G combined settings ...", flush=True)
    R["G"] = check_combinations(a, q[0], q[1])
    (WORK / "results.json").write_text(json.dumps(R, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    from .render import render

    print(f"wrote {write_report(render(R), HERE, HERE / 'report.md')}")


if __name__ == "__main__":
    main()
