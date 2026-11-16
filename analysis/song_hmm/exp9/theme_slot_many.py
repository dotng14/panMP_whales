"""Exp9 many-theme HMM: Exp8 topology + slot self-loops + larger T grid.

Topology per theme (``L_max`` slots):
  slot1 → slot2 → … → slot_Lmax
  each slot may self-loop;
  only slot_Lmax → slot1 (phrase repeat) or → next theme slot1.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from hmmlearn.hmm import CategoricalHMM
from numpy.typing import NDArray

from analysis.song_hmm.exp4.phrases import cut_phrases
from analysis.song_hmm.exp4.sequence import Session, encode_session
from analysis.song_hmm.exp5.theme_slot import (
    ThemeSlotSpec,
    _cluster_phrases,
    _init_categorical,
    _project_transmat,
    _seed_emissions,
    decode_theme_slot,
)

__all__ = [
    "L_MAX",
    "THEME_GRID",
    "build_spec_self_loop",
    "train_theme_slot_many",
    "select_theme_slot_many",
    "decode_theme_slot",
    "save_theme_slot_many",
    "load_theme_slot_many",
]

L_MAX = 10
THEME_GRID = (8, 10, 12, 14, 16)


def build_spec_self_loop(slot_counts: Sequence[int]) -> ThemeSlotSpec:
    """Left-to-right + self-loops; phrase-repeat / next-theme only from last slot."""
    slot_counts = tuple(int(x) for x in slot_counts)
    if any(c < 1 for c in slot_counts):
        raise ValueError(slot_counts)
    n_themes = len(slot_counts)
    state_names: list[str] = []
    theme_of_state: list[int] = []
    slot_of_state: list[int] = []
    first_state: list[int] = []
    last_state: list[int] = []
    idx = 0
    for t, L in enumerate(slot_counts):
        first_state.append(idx)
        for s in range(L):
            state_names.append(f"T{t + 1}.{s + 1}")
            theme_of_state.append(t)
            slot_of_state.append(s)
            idx += 1
        last_state.append(idx - 1)

    n = len(state_names)
    mask = np.zeros((n, n), dtype=float)
    for t, L in enumerate(slot_counts):
        base = first_state[t]
        for s in range(L):
            st = base + s
            mask[st, st] = 1.0  # self-loop
            if s < L - 1:
                mask[st, st + 1] = 1.0  # advance
        last = last_state[t]
        mask[last, first_state[t]] = 1.0  # phrase repeat
        if t + 1 < n_themes:
            mask[last, first_state[t + 1]] = 1.0  # next theme
    return ThemeSlotSpec(
        n_themes=n_themes,
        slot_counts=slot_counts,
        state_names=state_names,
        theme_of_state=theme_of_state,
        slot_of_state=slot_of_state,
        first_state=first_state,
        last_state=last_state,
        mask=mask,
    )


def _init_transmat_self_loop(
    spec: ThemeSlotSpec,
    *,
    self_prior: float = 0.35,
    repeat_prior: float = 0.85,
) -> NDArray[np.floating]:
    n = len(spec.state_names)
    trans = np.zeros((n, n))
    for t, L in enumerate(spec.slot_counts):
        base = spec.first_state[t]
        for s in range(L - 1):
            st = base + s
            trans[st, st] = self_prior
            trans[st, st + 1] = 1.0 - self_prior
        last = spec.last_state[t]
        # last: self + phrase-repeat (+ next)
        if t + 1 < spec.n_themes:
            # mass: self, repeat, next
            rem = 1.0 - self_prior
            trans[last, last] = self_prior
            trans[last, spec.first_state[t]] = rem * repeat_prior
            trans[last, spec.first_state[t + 1]] = rem * (1.0 - repeat_prior)
        else:
            trans[last, last] = self_prior
            trans[last, spec.first_state[t]] = 1.0 - self_prior
    return _project_transmat(trans, spec.mask)


def train_theme_slot_many(
    sessions: Sequence[Session],
    vocab: Sequence[str],
    *,
    n_themes: int = 8,
    l_max: int = L_MAX,
    n_iter: int = 40,
) -> tuple[CategoricalHMM, ThemeSlotSpec, dict]:
    phrases = []
    for s in sessions:
        phrases.extend(cut_phrases(s))
    if not phrases:
        raise ValueError("no phrases to train theme–slot HMM")
    medoids = _cluster_phrases(phrases, n_themes)
    # Shared L_max for all themes (Exp9)
    slot_counts = tuple(int(l_max) for _ in range(n_themes))
    spec = build_spec_self_loop(slot_counts)
    emission = _seed_emissions(spec, medoids, vocab)
    # Pad/truncate medoid seeding already handled inside _seed_emissions for L vs med len
    trans = _init_transmat_self_loop(spec)
    start = np.zeros(len(spec.state_names))
    start[spec.first_state[0]] = 1.0

    model = _init_categorical(
        len(spec.state_names),
        len(vocab),
        startprob=start,
        transmat=trans,
        emissionprob=emission,
    )
    model.n_iter = n_iter

    seqs = [encode_session(s, vocab) for s in sessions]
    # With self-loops, short sequences are OK; keep a mild floor
    min_len = min(4, int(l_max))
    seqs = [x for x in seqs if len(x) >= min_len]
    meta = {
        "kind": "theme_slot_many_self_loop",
        "n_themes": spec.n_themes,
        "slot_counts": list(spec.slot_counts),
        "l_max": int(l_max),
        "self_loops": True,
        "state_names": spec.state_names,
        "vocab": list(vocab),
        "n_train_seqs": len(seqs),
        "n_phrases": len(phrases),
        "medoids": [list(m) for m in medoids],
    }
    if not seqs:
        return model, spec, meta

    X = np.concatenate(seqs).reshape(-1, 1)
    lengths = [len(x) for x in seqs]
    model.fit(X, lengths)
    model.transmat_ = _project_transmat(model.transmat_, spec.mask)
    model.startprob_ = np.maximum(model.startprob_, 0.0)
    model.startprob_ = model.startprob_ / model.startprob_.sum()
    model.startprob_ = 0.7 * start + 0.3 * model.startprob_
    model.startprob_ = model.startprob_ / model.startprob_.sum()
    try:
        meta["train_score"] = float(model.score(X, lengths))
    except Exception:
        meta["train_score"] = None
    return model, spec, meta


def select_theme_slot_many(
    sessions: Sequence[Session],
    vocab: Sequence[str],
    *,
    theme_grid: Sequence[int] = THEME_GRID,
    l_max: int = L_MAX,
) -> tuple[CategoricalHMM, ThemeSlotSpec, dict]:
    """Choose T by BIC on train; fixed ``l_max`` slots + self-loops."""
    phrases = []
    for s in sessions:
        phrases.extend(cut_phrases(s))
    n_obs = sum(len(encode_session(s, vocab)) for s in sessions)
    best = None
    rows = []
    for T in theme_grid:
        if T > max(1, len(phrases)):
            continue
        model, spec, meta = train_theme_slot_many(
            sessions, vocab, n_themes=T, l_max=l_max
        )
        ll = meta.get("train_score")
        if ll is None:
            continue
        n_params = (
            spec.n_themes
            + len(spec.state_names) * (len(vocab) - 1)
            + len(spec.state_names)  # self-loop / exit params rough
        )
        total_ll = float(ll) * max(n_obs, 1)
        bic = -2.0 * total_ll + n_params * np.log(max(n_obs, 2))
        row = {
            "T": T,
            "slot_counts": list(spec.slot_counts),
            "avg_ll": ll,
            "bic": bic,
        }
        rows.append(row)
        if best is None or bic < best[0]:
            best = (bic, model, spec, meta)
    if best is None:
        return train_theme_slot_many(sessions, vocab, n_themes=8, l_max=l_max)
    _, model, spec, meta = best
    meta["selection"] = rows
    return model, spec, meta


def save_theme_slot_many(
    path: Path, model: CategoricalHMM, spec: ThemeSlotSpec, meta: dict
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        startprob=model.startprob_,
        transmat=model.transmat_,
        emissionprob=model.emissionprob_,
        mask=spec.mask,
        theme_of_state=np.asarray(spec.theme_of_state, dtype=np.int32),
        slot_of_state=np.asarray(spec.slot_of_state, dtype=np.int32),
        first_state=np.asarray(spec.first_state, dtype=np.int32),
        last_state=np.asarray(spec.last_state, dtype=np.int32),
        slot_counts=np.asarray(spec.slot_counts, dtype=np.int32),
        self_loops=np.asarray([1], dtype=np.int32),
    )
    Path(str(path) + ".meta.json").write_text(json.dumps(meta, indent=2))


def load_theme_slot_many(path: Path) -> tuple[CategoricalHMM, ThemeSlotSpec, dict]:
    path = Path(path)
    data = np.load(path)
    meta = json.loads(Path(str(path) + ".meta.json").read_text())
    slot_counts = tuple(int(x) for x in data["slot_counts"])
    spec = build_spec_self_loop(slot_counts)
    model = _init_categorical(
        len(data["startprob"]),
        data["emissionprob"].shape[1],
        startprob=data["startprob"],
        transmat=_project_transmat(data["transmat"], spec.mask),
        emissionprob=data["emissionprob"],
    )
    model.params = ""
    return model, spec, meta
