"""Run Experiment 4: per-wav train-half / eval-half theme-boundary experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly

from analysis.dict_learning.code.cli import load_windows
from analysis.GPL_Python.process_v4 import ProcessingOrchestrator

from . import paths
from .baselines import b1_boundaries, b2_boundaries
from .marks import propose_provisional_marks, save_marks
from .metrics_boundary import boundary_scores
from .option1 import (
    build_vocab,
    decode_full,
    decode_pair_v0,
    save_model,
    train_full,
    train_pair_v0,
)
from .option2 import decode_option2, save_option2, train_option2
from .paths import (
    GPL_014845,
    GPL_193500,
    WAV_014845_2KHZ,
    WAV_014845_10KHZ,
    WORK,
    FileSpec,
    all_hour_files,
    ensure_work_dirs,
    results_dir,
    split_014845,
    split_193500,
    train_files,
)
from .phrases import cut_phrases, phrases_to_jsonable
from .sequence import (
    Session,
    clean_events,
    energy_threshold,
    events_to_sessions,
    load_matchings,
    load_sequences,
    load_template_widths,
    placements_to_events,
    save_sequences,
)
from .spectrogram import plot_eval_slices


PY = sys.executable
REPO = paths.REPO


def _run(cmd: list[str]) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO))


def stage0_data() -> dict:
    ensure_work_dirs()
    if not GPL_193500.is_dir():
        raise FileNotFoundError(f"missing 193500 GPL: {GPL_193500}")
    w193 = load_windows(GPL_193500, None, 0)
    meta = {
        "gpl_193500": str(GPL_193500),
        "n_windows_193500": len(w193),
        "shape_193500": list(w193[0].data.shape),
        "sample_freq_193500": w193[0].sample_freq,
        "note": "Per-wav half split: 193500 h1–5/h6–10; 014845 first/second half of windows.",
    }
    if not WAV_014845_2KHZ.is_file():
        if not WAV_014845_10KHZ.is_file():
            raise FileNotFoundError(WAV_014845_10KHZ)
        print("decimating 014845 10 kHz → 2 kHz …", flush=True)
        sr, data = wavfile.read(WAV_014845_10KHZ)
        if data.ndim > 1:
            data = data[:, 0]
        data_f = (
            data.astype(np.float64) / 32768.0
            if np.issubdtype(data.dtype, np.integer)
            else data.astype(np.float64)
        )
        out = resample_poly(data_f, up=1, down=sr // 2000, axis=0)
        pcm = np.clip(np.rint(out * 32768.0), -32768, 32767).astype(np.int16)
        WAV_014845_2KHZ.parent.mkdir(parents=True, exist_ok=True)
        wavfile.write(WAV_014845_2KHZ, 2000, pcm)

    npy_dir = GPL_014845 / "spectrogram_npy"
    if not npy_dir.is_dir() or not any(npy_dir.glob("*.npy")):
        print("running GPL on 014845 2 kHz …", flush=True)
        GPL_014845.mkdir(parents=True, exist_ok=True)
        ProcessingOrchestrator(GPL_014845, freq_lo=10.0, freq_hi=1000.0).process(WAV_014845_2KHZ)
    w014 = load_windows(GPL_014845, None, 0)
    meta.update(
        {
            "gpl_014845": str(GPL_014845),
            "n_windows_014845": len(w014),
            "shape_014845": list(w014[0].data.shape),
        }
    )
    s193 = split_193500()
    s014 = split_014845(len(w014))
    files = []
    for sp in s193.train + s193.eval + s014.train + s014.eval:
        files.append({**sp.__dict__, "gpl_dir": str(sp.gpl_dir)})
    (WORK / "file_manifest.json").write_text(json.dumps({"meta": meta, "files": files}, indent=2))
    print(json.dumps(meta, indent=2))
    return meta


def stage1_dictionary(*, n_windows: int = 48, seed: int = 0) -> Path:
    ensure_work_dirs()
    out = WORK / "dict" / f"train_w{n_windows}_s{seed}"
    if (out / "summary.json").is_file():
        print(f"dictionary exists: {out}")
        return out
    t0 = time.time()
    _run(
        [
            PY, "-m", "analysis.dict_learning", "randomized-build",
            "--gpl-dir", str(GPL_193500), "--out", str(out),
            "--n-windows", str(n_windows), "--window-offset", "0",
            "--seed", str(seed), "--loop-stats", "--verbose", "--workers", "8",
            "--max-outer", "8", "--m-per-outer", "5", "--min-coverage", "0.25",
        ]
    )
    (WORK / "dict" / "timing.json").write_text(
        json.dumps({"build_s": time.time() - t0, "dict": str(out)}, indent=2)
    )
    return out


def _mp_specs() -> list[FileSpec]:
    specs = list(all_hour_files())
    w014 = load_windows(GPL_014845, None, 0)
    # full-file MP for 014845 (half split applied later when loading sessions)
    specs.append(
        FileSpec(
            file_id="014845",
            wav_id="014845",
            gpl_dir=GPL_014845,
            window_offset=0,
            n_windows=len(w014),
            role="train",  # placeholder; half split is by time
            time_offset_s=0.0,
        )
    )
    return specs


def stage1_mp(dictionary: Path, *, workers: int = 8) -> None:
    ensure_work_dirs()
    for spec in _mp_specs():
        out = WORK / "mp" / spec.file_id
        if (out / "matchings.json").is_file():
            print(f"MP exists: {out}")
            continue
        cmd = [
            PY, "-m", "analysis.dict_learning", "matching-pursuit",
            "--gpl-dir", str(spec.gpl_dir), "--dictionary", str(dictionary),
            "--out", str(out), "--workers", str(workers),
        ]
        if spec.n_windows:
            cmd += ["--n-windows", str(spec.n_windows), "--window-offset", str(spec.window_offset)]
        t0 = time.time()
        _run(cmd)
        (out / "timing.json").write_text(json.dumps({"mp_s": time.time() - t0}, indent=2))


def stage2_sequences(dictionary: Path) -> None:
    ensure_work_dirs()
    widths = load_template_widths(dictionary)
    train_events = []
    for spec in train_files():
        mp = WORK / "mp" / spec.file_id / "matchings.json"
        if not mp.is_file():
            continue
        windows = load_windows(spec.gpl_dir, spec.n_windows, spec.window_offset)
        train_events.extend(placements_to_events(load_matchings(mp), windows, widths))
    emin = energy_threshold(train_events)
    (WORK / "sequences" / "energy_min.json").write_text(json.dumps({"energy_min": emin}, indent=2))

    for spec in _mp_specs():
        mp = WORK / "mp" / spec.file_id / "matchings.json"
        if not mp.is_file():
            print(f"skip sequences, no MP: {spec.file_id}")
            continue
        windows = load_windows(spec.gpl_dir, spec.n_windows or None, spec.window_offset)
        events = clean_events(
            placements_to_events(load_matchings(mp), windows, widths), energy_min=emin
        )
        sessions = events_to_sessions(events)
        save_sequences(
            WORK / "sequences" / f"{spec.file_id}.json",
            sessions,
            meta={
                "file_id": spec.file_id,
                "wav_id": spec.wav_id,
                "role": spec.role,
                "n_events": len(events),
                "n_sessions": len(sessions),
                "energy_min": emin,
                "time_offset_s": spec.time_offset_s,
                "window_offset": spec.window_offset,
                "n_windows": spec.n_windows,
            },
        )
        phrases = []
        for s in sessions:
            phrases.extend(phrases_to_jsonable(cut_phrases(s)))
        (WORK / "sequences" / f"{spec.file_id}_phrases.json").write_text(
            json.dumps(phrases, indent=2)
        )
        print(f"{spec.file_id}: {len(events)} events, {len(sessions)} sessions, {len(phrases)} phrases")


def _sessions_in_time_range(sessions: list[Session], t0: float, t1: float) -> list[Session]:
    """Keep tokens that fall in [t0, t1) and rebuild sessions."""
    out: list[Session] = []
    for s in sessions:
        toks = [t for t in s.tokens if t0 <= t.t0 < t1]
        if not toks:
            continue
        # drop leading/trailing gap-only
        while toks and toks[0].kind != "motif":
            toks = toks[1:]
        while toks and toks[-1].kind != "motif":
            toks = toks[:-1]
        if toks:
            out.append(Session(tokens=toks, t0=toks[0].t0, t1=toks[-1].t1))
    return out


def _decode_sessions(sessions, m_pair, m_full, state_names, vocab, m2):
    pair_bounds, full_bounds, o2_bounds = [], [], []
    pair_themes, full_themes, o2_themes = [], [], []
    for s in sessions:
        d1 = decode_pair_v0(m_pair, s, vocab)
        dF = decode_full(m_full, s, vocab, state_names)
        d2 = decode_option2(m2, s)
        pair_bounds.extend(d1.boundaries)
        full_bounds.extend(dF.boundaries)
        o2_bounds.extend(d2.boundaries)
        pair_themes.extend(d1.themes)
        full_themes.extend(dF.themes)
        o2_themes.extend(d2.themes)
    return {
        "option1_pair": {"boundaries": pair_bounds, "themes": pair_themes},
        "option1_full": {"boundaries": full_bounds, "themes": full_themes},
        "option2": {"boundaries": o2_bounds, "themes": o2_themes},
        "B1": {"boundaries": b1_boundaries(sessions)},
        "B2": {"boundaries": b2_boundaries(sessions)},
    }


def _run_one_wav(wav_id: str, train_sessions: list[Session], eval_sessions: list[Session],
                 gpl_dir: Path, eval_window_offset: int, eval_n_windows: int) -> dict:
    """Train on train half, decode eval half, write results/{wav_id}/."""
    out_dir = results_dir(wav_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = WORK / "figures" / wav_id
    fig_dir.mkdir(parents=True, exist_ok=True)

    marks = propose_provisional_marks(train_sessions, eval_sessions)
    save_marks(marks, paths.REFERENCE / f"marks_{wav_id}.json")
    train_marks = marks.get("A_to_B") or marks.get("R0", []) + marks.get("R1", [])
    eval_marks = marks.get("R2") or []

    vocab = build_vocab(train_sessions)
    print(f"[{wav_id}] training option1 pair-v0 on train half …", flush=True)
    m_pair = train_pair_v0(train_sessions, vocab, train_marks)
    save_model(WORK / "models" / wav_id / "option1_pair_v0.npz", m_pair, {"vocab": vocab, "kind": "pair_v0", "wav_id": wav_id})

    print(f"[{wav_id}] training option1 full …", flush=True)
    m_full, state_names = train_full(train_sessions, vocab, n_themes=6, slots=3)
    save_model(
        WORK / "models" / wav_id / "option1_full.npz",
        m_full,
        {"vocab": vocab, "kind": "full", "state_names": state_names, "wav_id": wav_id},
    )

    print(f"[{wav_id}] training option2 …", flush=True)
    m2 = train_option2(train_sessions, train_marks, m=2)
    save_option2(WORK / "models" / wav_id / "option2.npz", m2)

    decoded = _decode_sessions(eval_sessions, m_pair, m_full, state_names, vocab, m2)
    (out_dir / "eval.json").write_text(json.dumps(decoded, indent=2))

    hours = 1.0
    if eval_sessions:
        hours = max((max(s.t1 for s in eval_sessions) - min(s.t0 for s in eval_sessions)) / 3600.0, 1e-6)

    scores = {}
    for name, block in decoded.items():
        scores[name] = boundary_scores(block["boundaries"], eval_marks, hours=hours)

    summary = {
        "wav_id": wav_id,
        "n_train_sessions": len(train_sessions),
        "n_eval_sessions": len(eval_sessions),
        "n_eval_marks_provisional": len(eval_marks),
        "scores": scores,
        "n_boundaries": {k: len(v["boundaries"]) for k, v in decoded.items()},
        "n_themes": {
            k: len(v.get("themes") or []) for k, v in decoded.items() if "themes" in v
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # spectrogram figures for option1_full and option2 on eval half
    for model_name in ("option1_full", "option2"):
        block = decoded[model_name]
        plot_eval_slices(
            gpl_dir,
            block.get("themes") or [],
            block["boundaries"],
            window_offset=eval_window_offset,
            n_windows=eval_n_windows,
            slice_s=600.0 if wav_id == "193500" else 300.0,
            title_prefix=f"{wav_id} eval / {model_name}",
            out_dir=fig_dir / model_name,
        )
        import matplotlib.pyplot as plt
        plt.close("all")

    print(json.dumps(summary, indent=2))
    return summary


def stage3_train_and_decode() -> dict:
    ensure_work_dirs()
    # --- 193500 ---
    train_193, eval_193 = [], []
    for path in sorted((WORK / "sequences").glob("193500_h*.json")):
        if path.name.endswith("_phrases.json"):
            continue
        sess, meta = load_sequences(path)
        h = int(meta["file_id"].split("_h")[1])
        if h <= 5:
            train_193.extend(sess)
        else:
            eval_193.extend(sess)

    summaries = {}
    summaries["193500"] = _run_one_wav(
        "193500",
        train_193,
        eval_193,
        GPL_193500,
        eval_window_offset=5 * 48,
        eval_n_windows=5 * 48,
    )

    # --- 014845 ---
    seq_path = WORK / "sequences" / "014845.json"
    if seq_path.is_file():
        all_014, _ = load_sequences(seq_path)
        w014 = load_windows(GPL_014845, None, 0)
        n = len(w014)
        n_train = n // 2
        t_mid = float(n_train * 75.0)
        train_014 = _sessions_in_time_range(all_014, 0.0, t_mid)
        eval_014 = _sessions_in_time_range(all_014, t_mid, 1e12)
        summaries["014845"] = _run_one_wav(
            "014845",
            train_014,
            eval_014,
            GPL_014845,
            eval_window_offset=n_train,
            eval_n_windows=n - n_train,
        )

    (WORK / "results" / "summary_all.json").write_text(json.dumps(summaries, indent=2))
    return summaries


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["0", "1", "2", "3", "all"])
    p.add_argument("--n-windows", type=int, default=48)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)

    if args.stage in ("0", "all"):
        stage0_data()
    dictionary = WORK / "dict" / f"train_w{args.n_windows}_s0"
    if args.stage in ("1", "all"):
        dictionary = stage1_dictionary(n_windows=args.n_windows)
        stage1_mp(dictionary, workers=args.workers)
    if args.stage in ("2", "all"):
        if not (dictionary / "summary.json").is_file():
            dicts = sorted((WORK / "dict").glob("train_w*_s0"))
            if not dicts:
                raise SystemExit("no dictionary; run stage 1")
            dictionary = dicts[-1]
        stage2_sequences(dictionary)
    if args.stage in ("3", "all"):
        stage3_train_and_decode()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
