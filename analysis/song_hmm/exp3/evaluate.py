"""Phase 3: score every model (Exp 2, Exp 2b, M1-M4) and the CSV with the same metrics.

    python -m analysis.song_exp3.evaluate

Groups (definitions in metrics.md):
  H  against the hand labels (if reference/labels/*.txt exist), test half
  P  pair precision (if the pair-judging export exists)
  B  agreement with the CSV
  C  label-free structure on the common phrase-descriptor yardstick (Experiment 2's)
  R  rules: session breaks <= 60 s, complete songs with all six themes
Writes work/eval.json.
"""

from __future__ import annotations

import json

import numpy as np

from analysis.song_hmm.train import SPLIT_S
from analysis.song_hmm_phrase.decode import Decoded, from_csv
from analysis.song_hmm_phrase.metrics import csv_agreement, kappa, label_free, match_events, novelty_curve, novelty_peaks, unit_rows

from .models.common import HALVES, HERE, N_S, N_THEMES, SESSION_GAP_S, WORK, labels

PHRASE = HERE.parent / "song_hmm_phrase"
MODELS = ("exp2", "exp2b", "M1", "M2", "M3", "M4")
NAMES = {"csv": "CSV", "exp2": "Exp 2", "exp2b": "Exp 2b", "M1": "M1 theme HMM", "M2": "M2 song-first", "M3": "M3 units", "M4": "M4 six theme HMMs"}


def load_decoded(model: str, half: str, offset: int) -> Decoded | None:
    base = (PHRASE / "work" / model / half) if model.startswith("exp") else (WORK / "models" / model / half)
    p = base / "decoded.npz"
    if not p.exists():
        return None
    d = np.load(p)
    return Decoded(model, d["theme"].astype(np.int32), d["session"].astype(bool), (d["boundaries"] - offset).astype(int))


def rules(dec: Decoded) -> dict:
    n = len(dec.session)
    e = np.flatnonzero(np.diff(np.r_[1, dec.session.astype(np.int8), 1])).reshape(-1, 2)
    inner = [(a, b) for a, b in e if a > 0 and b < n]
    complete = [s for s in dec.songs if s.complete]
    return {
        "background_gaps": len(inner),
        "gaps_le_60s": int(sum(b - a <= SESSION_GAP_S for a, b in inner)),
        "complete_songs": len(complete),
        "complete_with_all_six_themes": int(sum(all(v > 0 for v in s.theme_seconds) for s in complete)),
        "complete_7_to_30_min": int(sum(420 <= s.end_s - s.start_s <= 1800 for s in complete)),
        "median_complete_min": float(np.median([s.end_s - s.start_s for s in complete]) / 60) if complete else float("nan"),
    }


def hand_labels():
    """Per-second theme / song starts / background from the Audacity exports, if they exist."""
    try:
        from .reference import labels as ref
    except ImportError:
        return None
    try:
        clips = ref.load_all()
    except (FileNotFoundError, KeyError, ValueError):
        return None
    if not clips:
        return None
    theme, labelled = ref.recording_theme(clips, N_S)
    starts = np.sort(np.concatenate([c.song_starts for c in clips.values()])) if clips else np.zeros(0)
    return {"theme": theme, "labelled": labelled.astype(bool), "song_starts": starts, "clips": sorted(clips)}


def score_hand(dec: Decoded, ref: dict, offset: int) -> dict:
    th = ref["theme"][offset : offset + len(dec.theme)]
    mask = ref["labelled"][offset : offset + len(dec.theme)]
    if mask.sum() == 0:
        return {}
    y, p = th[mask], dec.theme[mask]
    conf = np.zeros((N_THEMES + 1, N_THEMES + 1), np.int64)
    np.add.at(conf, (np.where(y >= 0, y, N_THEMES), np.where(p >= 0, p, N_THEMES)), 1)
    rec = [conf[k, k] / conf[k].sum() if conf[k].sum() else float("nan") for k in range(N_THEMES)]
    starts = ref["song_starts"] - offset
    starts = starts[(starts >= 0) & (starts < len(dec.theme))]
    lo, hi = np.flatnonzero(mask)[[0, -1]]
    pred = dec.boundaries[(dec.boundaries >= lo) & (dec.boundaries <= hi)]
    return {
        "seconds_labelled": int(mask.sum()),
        "theme_accuracy": float(np.mean(y == p)),
        "theme_balanced_accuracy": float(np.nanmean(rec)),
        "theme_kappa": kappa(conf),
        "theme_recall": rec,
        "song_start": {f"tol_{t}s": match_events(pred, starts, t) for t in (15, 30)},
    }


def main() -> None:
    lab = labels()
    majority = int(np.bincount(lab.theme[:SPLIT_S][lab.theme[:SPLIT_S] >= 0]).argmax())
    f = np.load(PHRASE / "work" / "phrase_features.npz")
    ref = hand_labels()
    out: dict = {"B": {}, "C": {}, "R": {}, "H": {}, "hand_labels": ref is not None}
    decs: dict = {}
    for h, off, end in HALVES:
        n = end - off
        decs[h] = {m: load_decoded(m, h, off) for m in MODELS}
        decs[h]["csv"] = from_csv("csv", lab.theme[off:end], lab.song_bounds[:, 0], off)
    for h, off, end in HALVES:
        zn = unit_rows(f[f"z_{h}"])
        peaks = novelty_peaks(novelty_curve(zn))
        out["B"][h], out["C"][h], out["R"][h], out["H"][h] = {}, {}, {}, {}
        for m, dec in decs[h].items():
            if dec is None:
                continue
            if m != "csv":
                out["B"][h][m] = csv_agreement(dec, decs[h]["csv"].theme, lab.song_bounds[:, 0] - off, majority)
            out["C"][h][m] = label_free(dec, zn, f[f"t_{h}"], off, peaks, decs["train"][m])
            out["R"][h][m] = rules(dec)
            if ref is not None:
                out["H"][h][m] = score_hand(dec, ref, off)
    (WORK / "eval.json").write_text(json.dumps(out, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    t = out["B"]["test"]
    for m in t:
        c = out["C"]["test"][m]
        print(f"{NAMES[m]:20s} bal {t[m]['theme_balanced_accuracy']:.3f} kappa {t[m]['theme_kappa']:.3f} bF1@30 {t[m]['boundaries']['tol_30s']['f1']:.2f} "
              f"songs {t[m]['n_songs']}/{t[m]['n_complete_songs']} sess {t[m]['n_sessions']} | coh {c['coherence']['coherence']:.3f} id {c['identity']['identity']:.3f} "
              f"nov {c['novelty']['precision']:.2f} | gaps<=60 {out['R']['test'][m]['gaps_le_60s']}")



if __name__ == "__main__":
    main()
