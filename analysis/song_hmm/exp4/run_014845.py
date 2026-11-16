"""014845-only Experiment 4: train/eval half split, figures, summary."""

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
    REFERENCE,
    WAV_014845_2KHZ,
    WAV_014845_10KHZ,
    WORK,
    ensure_work_dirs,
    results_dir,
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
REPO = Path(__file__).resolve().parents[3]
WAV_ID = "014845"


def _run(cmd: list[str]) -> None:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO))


def ensure_gpl() -> int:
    ensure_work_dirs()
    if not WAV_014845_2KHZ.is_file():
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
    npy = GPL_014845 / "spectrogram_npy"
    if not npy.is_dir() or not any(npy.glob("*.npy")):
        from analysis.GPL_Python.process_v4 import ProcessingOrchestrator

        GPL_014845.mkdir(parents=True, exist_ok=True)
        ProcessingOrchestrator(GPL_014845, freq_lo=10.0, freq_hi=1000.0).process(WAV_014845_2KHZ)
    return len(load_windows(GPL_014845, None, 0))


def build_dict_and_mp(*, workers: int = 8) -> Path:
    n = ensure_gpl()
    n_train = n // 2
    dictionary = WORK / "dict" / "014845_train_half"
    if not (dictionary / "summary.json").is_file():
        t0 = time.time()
        _run(
            [
                PY, "-m", "analysis.dict_learning", "randomized-build",
                "--gpl-dir", str(GPL_014845), "--out", str(dictionary),
                "--n-windows", str(n_train), "--window-offset", "0",
                "--seed", "0", "--loop-stats", "--verbose", "--workers", str(workers),
                "--max-outer", "8", "--m-per-outer", "5", "--min-coverage", "0.25",
            ]
        )
        (dictionary / "timing.json").write_text(json.dumps({"build_s": time.time() - t0}, indent=2))
    else:
        print(f"dictionary exists: {dictionary}")

    mp_out = WORK / "mp" / "014845"
    need_mp = True
    if (mp_out / "summary.json").is_file():
        prev = json.loads((mp_out / "summary.json").read_text())
        if "014845_train_half" in str(prev.get("dictionary", "")):
            print(f"MP exists: {mp_out}")
            need_mp = False
    if need_mp:
        t0 = time.time()
        _run(
            [
                PY, "-m", "analysis.dict_learning", "matching-pursuit",
                "--gpl-dir", str(GPL_014845), "--dictionary", str(dictionary),
                "--out", str(mp_out), "--workers", str(workers),
            ]
        )
        (mp_out / "timing.json").write_text(json.dumps({"mp_s": time.time() - t0}, indent=2))
    return dictionary


def build_sequences(dictionary: Path) -> tuple[list[Session], list[Session], float, int, int]:
    n = len(load_windows(GPL_014845, None, 0))
    n_train = n // 2
    t_mid = float(n_train * 75.0)
    widths = load_template_widths(dictionary)
    windows = load_windows(GPL_014845, None, 0)
    matchings = load_matchings(WORK / "mp" / "014845" / "matchings.json")
    events = placements_to_events(matchings, windows, widths)
    # energy threshold from train-half events only
    train_ev = [e for e in events if e.t0 < t_mid]
    emin = energy_threshold(train_ev)
    events = clean_events(events, energy_min=emin)
    sessions = events_to_sessions(events)
    save_sequences(
        WORK / "sequences" / "014845.json",
        sessions,
        meta={
            "file_id": "014845",
            "wav_id": "014845",
            "n_events": len(events),
            "energy_min": emin,
            "t_mid_s": t_mid,
            "n_train_windows": n_train,
            "n_eval_windows": n - n_train,
            "dictionary": str(dictionary),
        },
    )
    phrases = []
    for s in sessions:
        phrases.extend(phrases_to_jsonable(cut_phrases(s)))
    (WORK / "sequences" / "014845_phrases.json").write_text(json.dumps(phrases, indent=2))

    def in_range(lo, hi):
        out = []
        for s in sessions:
            toks = [t for t in s.tokens if lo <= t.t0 < hi]
            while toks and toks[0].kind != "motif":
                toks = toks[1:]
            while toks and toks[-1].kind != "motif":
                toks = toks[:-1]
            if toks:
                out.append(Session(toks, toks[0].t0, toks[-1].t1))
        return out

    train = in_range(0.0, t_mid)
    eval_s = in_range(t_mid, 1e12)
    print(f"014845: {len(events)} events; train sessions={len(train)} eval={len(eval_s)}; t_mid={t_mid}s")
    return train, eval_s, t_mid, n_train, n - n_train


def train_decode(train: list[Session], eval_s: list[Session], t_mid: float, n_train: int, n_eval: int) -> dict:
    out_dir = results_dir(WAV_ID)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = WORK / "figures" / WAV_ID
    fig_dir.mkdir(parents=True, exist_ok=True)

    marks = propose_provisional_marks(train, eval_s)
    # shorter separation for this short file
    save_marks(marks, REFERENCE / "marks_014845.json")
    train_marks = marks.get("A_to_B") or (marks.get("R0") or []) + (marks.get("R1") or [])
    eval_marks = marks.get("R2") or []

    vocab = build_vocab(train)
    print("training option1 pair-v0 …", flush=True)
    m_pair = train_pair_v0(train, vocab, train_marks)
    save_model(WORK / "models" / WAV_ID / "option1_pair_v0.npz", m_pair, {"vocab": vocab, "kind": "pair_v0"})

    print("training option1 full …", flush=True)
    n_themes = 4  # short file: fewer themes than 10 h recording
    m_full, state_names = train_full(train, vocab, n_themes=n_themes, slots=3)
    save_model(
        WORK / "models" / WAV_ID / "option1_full.npz",
        m_full,
        {"vocab": vocab, "kind": "full", "state_names": state_names},
    )

    print("training option2 …", flush=True)
    m2 = train_option2(train, train_marks, m=2)
    save_option2(WORK / "models" / WAV_ID / "option2.npz", m2)

    pair_b, full_b, o2_b = [], [], []
    pair_t, full_t, o2_t = [], [], []
    for s in eval_s:
        d1 = decode_pair_v0(m_pair, s, vocab)
        dF = decode_full(m_full, s, vocab, state_names)
        d2 = decode_option2(m2, s)
        pair_b.extend(d1.boundaries); pair_t.extend(d1.themes)
        full_b.extend(dF.boundaries); full_t.extend(dF.themes)
        o2_b.extend(d2.boundaries); o2_t.extend(d2.themes)

    decoded = {
        "option1_pair": {"boundaries": pair_b, "themes": pair_t},
        "option1_full": {"boundaries": full_b, "themes": full_t},
        "option2": {"boundaries": o2_b, "themes": o2_t},
        "B1": {"boundaries": b1_boundaries(eval_s, min_sep_s=45.0)},
        "B2": {"boundaries": b2_boundaries(eval_s)},
        "t_mid_s": t_mid,
        "eval_marks_provisional": eval_marks,
    }
    (out_dir / "eval.json").write_text(json.dumps(decoded, indent=2))

    hours = max((max(s.t1 for s in eval_s) - min(s.t0 for s in eval_s)) / 3600.0, 1e-6) if eval_s else 1.0
    scores = {name: boundary_scores(block["boundaries"], eval_marks, hours=hours)
              for name, block in decoded.items() if isinstance(block, dict) and "boundaries" in block}

    # qualitative extras
    o2_durs = [th["t1"] - th["t0"] for th in o2_t] or [0.0]
    qual = {
        "option1_pair_n_themes": len(pair_t),
        "option1_full_n_themes": len(full_t),
        "option2_n_themes": len(o2_t),
        "option2_median_theme_s": float(np.median(o2_durs)),
        "option2_frac_themes_lt_30s": float(np.mean([d < 30 for d in o2_durs])),
        "train_n_tokens": sum(len(s.tokens) for s in train),
        "eval_n_tokens": sum(len(s.tokens) for s in eval_s),
    }

    summary = {
        "wav_id": WAV_ID,
        "t_mid_s": t_mid,
        "n_train_windows": n_train,
        "n_eval_windows": n_eval,
        "n_eval_marks_provisional": len(eval_marks),
        "scores": scores,
        "n_boundaries": {k: len(decoded[k]["boundaries"]) for k in scores},
        "n_themes": {k: len(decoded[k].get("themes") or []) for k in ("option1_pair", "option1_full", "option2")},
        "qualitative": qual,
        "note": "Scores use provisional B1 marks on eval half; replace with hand marks for final judgment.",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    import matplotlib.pyplot as plt

    for model_name in ("option1_full", "option1_pair", "option2", "B1"):
        block = decoded[model_name]
        themes = block.get("themes") or []
        # for B1 invent pseudo-themes between bounds for display
        if model_name == "B1" and not themes:
            bs = [t_mid] + sorted(block["boundaries"]) + ([eval_s[-1].t1] if eval_s else [t_mid + 1])
            themes = [
                {"theme": f"seg{i}", "t0": bs[i], "t1": bs[i + 1], "n_tokens": 0}
                for i in range(len(bs) - 1)
            ]
        plot_eval_slices(
            GPL_014845,
            themes,
            block["boundaries"],
            window_offset=n_train,
            n_windows=n_eval,
            slice_s=300.0,
            title_prefix=f"014845 eval / {model_name}",
            out_dir=fig_dir / model_name,
        )
        plt.close("all")

    print(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--skip-mp", action="store_true", help="reuse existing 014845 MP if present")
    args = p.parse_args(argv)
    ensure_work_dirs()
    dictionary = build_dict_and_mp(workers=args.workers)
    train, eval_s, t_mid, n_train, n_eval = build_sequences(dictionary)
    train_decode(train, eval_s, t_mid, n_train, n_eval)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
