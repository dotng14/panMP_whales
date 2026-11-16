"""Option 1: shared-vocabulary categorical HMMs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from hmmlearn.hmm import CategoricalHMM
from numpy.typing import NDArray

from .phrases import cut_phrases, normalised_edit
from .sequence import Session, Token, encode_session, vocabulary


@dataclass
class DecodeResult:
    theme_path: list[str]  # theme label per token (or Other/Silence)
    boundaries: list[float]  # times where theme changes
    themes: list[dict]  # [{theme, t0, t1, n_tokens}, ...]


def _token_times(session: Session) -> list[float]:
    return [t.t0 for t in session.tokens]


def build_vocab(sessions: Sequence[Session]) -> list[str]:
    return vocabulary(sessions)


def _init_categorical(
    n_states: int,
    n_features: int,
    *,
    startprob: NDArray[np.floating] | None = None,
    transmat: NDArray[np.floating] | None = None,
    emissionprob: NDArray[np.floating] | None = None,
) -> CategoricalHMM:
    model = CategoricalHMM(
        n_components=n_states,
        n_features=n_features,
        n_iter=100,
        tol=1e-4,
        init_params="",
        params="ste",
        random_state=0,
    )
    if startprob is None:
        startprob = np.full(n_states, 1.0 / n_states)
    if transmat is None:
        transmat = np.full((n_states, n_states), 1.0 / n_states)
    if emissionprob is None:
        emissionprob = np.full((n_states, n_features), 1.0 / n_features)
    model.startprob_ = startprob / startprob.sum()
    model.transmat_ = transmat / transmat.sum(axis=1, keepdims=True)
    model.emissionprob_ = emissionprob / emissionprob.sum(axis=1, keepdims=True)
    return model


def _counts_to_emission(counts: NDArray[np.floating], n_features: int) -> NDArray[np.floating]:
    if counts.ndim != 1:
        raise ValueError("counts must be 1-D")
    e = np.ones(n_features)  # add-one
    n = min(len(counts), n_features)
    e[:n] += counts[:n]
    return e / e.sum()


def tokens_in_window(session: Session, t0: float, t1: float) -> list[Token]:
    return [t for t in session.tokens if t0 <= t.t0 < t1]


def train_pair_v0(
    sessions: Sequence[Session],
    vocab: Sequence[str],
    marks: Sequence[float],
    *,
    pre_s: float = 120.0,
    post_s: float = 120.0,
    clip_s: float = 900.0,
) -> CategoricalHMM:
    """Other → A → B → Other categorical HMM seeded from marks."""
    n_features = len(vocab)
    # states: 0 Other_in, 1 A, 2 B, 3 Other_out
    counts = {
        "A": np.zeros(n_features),
        "B": np.zeros(n_features),
        "Other": np.zeros(n_features),
    }
    clips: list[NDArray[np.int64]] = []
    for s in sessions:
        X = encode_session(s, vocab)
        if len(X):
            for i, tok in enumerate(s.tokens):
                if tok.kind == "motif":
                    key = f"M{tok.motif}"
                else:
                    key = tok.kind
                if key in vocab:
                    counts["Other"][list(vocab).index(key)] += 1
        for m in marks:
            if not (s.t0 - clip_s <= m <= s.t1 + clip_s):
                continue
            before = tokens_in_window(s, m - pre_s, m)
            after = tokens_in_window(s, m, m + post_s)
            for t in before:
                key = f"M{t.motif}" if t.kind == "motif" else t.kind
                if key in vocab:
                    counts["A"][list(vocab).index(key)] += 1
            for t in after:
                key = f"M{t.motif}" if t.kind == "motif" else t.kind
                if key in vocab:
                    counts["B"][list(vocab).index(key)] += 1
            # clip around mark
            clip_tokens = tokens_in_window(s, m - clip_s, m + clip_s)
            if clip_tokens:
                tmp = Session(tokens=clip_tokens, t0=clip_tokens[0].t0, t1=clip_tokens[-1].t1)
                x = encode_session(tmp, vocab)
                if len(x) >= 5:
                    clips.append(x)

    emission = np.vstack(
        [
            _counts_to_emission(counts["Other"], n_features),
            _counts_to_emission(counts["A"], n_features),
            _counts_to_emission(counts["B"], n_features),
            _counts_to_emission(counts["Other"], n_features),
        ]
    )
    trans = np.array(
        [
            [0.98, 0.02, 0.00, 0.00],
            [0.00, 0.97, 0.03, 0.00],
            [0.00, 0.00, 0.97, 0.03],
            [0.00, 0.00, 0.00, 1.00],
        ]
    )
    start = np.array([0.7, 0.2, 0.1, 0.0])
    model = _init_categorical(4, n_features, startprob=start, transmat=trans, emissionprob=emission)
    if clips:
        X = np.concatenate(clips).reshape(-1, 1)
        lengths = [len(c) for c in clips]
        model.fit(X, lengths)
    return model


def decode_pair_v0(model: CategoricalHMM, session: Session, vocab: Sequence[str]) -> DecodeResult:
    X = encode_session(session, vocab)
    if len(X) == 0:
        return DecodeResult([], [], [])
    states = model.predict(X.reshape(-1, 1))
    labels = []
    for st in states:
        labels.append({0: "Other", 1: "A", 2: "B", 3: "Other"}[int(st)])
    boundaries: list[float] = []
    themes: list[dict] = []
    times = _token_times(session)
    # keep tokens aligned: encode_session drops unknown; assume all known
    n = min(len(labels), len(times))
    labels, times = labels[:n], times[:n]
    cur = labels[0]
    t0 = times[0]
    for i in range(1, n):
        if labels[i] != cur:
            if cur in ("A", "B") and labels[i] in ("A", "B") and cur != labels[i]:
                boundaries.append(times[i])
            themes.append({"theme": cur, "t0": t0, "t1": times[i], "n_tokens": i})
            cur = labels[i]
            t0 = times[i]
    themes.append({"theme": cur, "t0": t0, "t1": times[-1], "n_tokens": n})
    # A→B boundary specifically for pair model
    for i in range(1, n):
        if labels[i - 1] == "A" and labels[i] == "B":
            if times[i] not in boundaries:
                boundaries.append(times[i])
    return DecodeResult(theme_path=labels, boundaries=sorted(set(boundaries)), themes=themes)


def train_full(
    sessions: Sequence[Session],
    vocab: Sequence[str],
    *,
    n_themes: int = 6,
    slots: int = 3,
) -> tuple[CategoricalHMM, list[str]]:
    """Cycle of theme slot-loops + silence. State names returned with the model."""
    # Build phrase strings and k-medoids-like seeding via greedy clustering
    phrases = []
    for s in sessions:
        phrases.extend(cut_phrases(s))
    if not phrases:
        raise ValueError("no phrases to train 1-full")
    # cluster by picking medoids greedily
    medoids = [phrases[0].motifs]
    for _ in range(1, n_themes):
        best, best_d = None, -1.0
        for ph in phrases:
            d = min(normalised_edit(ph.motifs, m) for m in medoids)
            if d > best_d:
                best, best_d = ph.motifs, d
        if best is not None:
            medoids.append(best)

    # states: theme t slot s, plus silence
    state_names: list[str] = []
    for t in range(n_themes):
        for s in range(slots):
            state_names.append(f"T{t + 1}_s{s + 1}")
    state_names.append("Silence")
    n_states = len(state_names)
    n_features = len(vocab)
    emission = np.ones((n_states, n_features))
    index = {v: i for i, v in enumerate(vocab)}
    for t, med in enumerate(medoids):
        for s in range(slots):
            st = t * slots + s
            if s < len(med):
                key = f"M{med[s]}"
                if key in index:
                    emission[st, index[key]] += 10.0
    # silence prefers gaps
    for g in ("GAP_S", "GAP_L", "BREAK"):
        if g in index:
            emission[-1, index[g]] += 20.0
    emission = emission / emission.sum(axis=1, keepdims=True)

    trans = np.zeros((n_states, n_states))
    for t in range(n_themes):
        base = t * slots
        for s in range(slots):
            st = base + s
            nxt = base + (s + 1) % slots
            trans[st, st] = 0.7
            trans[st, nxt] = 0.25
            # leave theme from last slot
            if s == slots - 1:
                nxt_theme = ((t + 1) % n_themes) * slots
                trans[st, nxt_theme] = 0.04
                trans[st, -1] = 0.01
        # from silence back
    trans[-1, -1] = 0.9
    for t in range(n_themes):
        trans[-1, t * slots] = 0.1 / n_themes
    # renormalise
    trans = np.maximum(trans, 1e-6)
    trans = trans / trans.sum(axis=1, keepdims=True)
    start = np.zeros(n_states)
    for t in range(n_themes):
        start[t * slots] = 1.0 / n_themes
    model = _init_categorical(n_states, n_features, startprob=start, transmat=trans, emissionprob=emission)

    seqs = [encode_session(s, vocab) for s in sessions]
    seqs = [x for x in seqs if len(x) >= 5]
    if seqs:
        X = np.concatenate(seqs).reshape(-1, 1)
        lengths = [len(x) for x in seqs]
        model.fit(X, lengths)
    return model, state_names


def decode_full(
    model: CategoricalHMM,
    session: Session,
    vocab: Sequence[str],
    state_names: Sequence[str],
) -> DecodeResult:
    X = encode_session(session, vocab)
    if len(X) == 0:
        return DecodeResult([], [], [])
    states = model.predict(X.reshape(-1, 1))
    labels = [state_names[int(s)].split("_")[0] for s in states]  # T1 / Silence
    times = _token_times(session)
    n = min(len(labels), len(times))
    labels, times = labels[:n], times[:n]
    boundaries: list[float] = []
    themes: list[dict] = []
    cur = labels[0]
    t0 = times[0]
    count = 1
    for i in range(1, n):
        if labels[i] != cur:
            if cur != "Silence" and labels[i] != "Silence" and cur != labels[i]:
                boundaries.append(times[i])
            if cur != "Silence":
                themes.append({"theme": cur, "t0": t0, "t1": times[i], "n_tokens": count})
            cur = labels[i]
            t0 = times[i]
            count = 1
        else:
            count += 1
    if cur != "Silence":
        themes.append({"theme": cur, "t0": t0, "t1": times[-1], "n_tokens": count})
    return DecodeResult(labels, sorted(set(boundaries)), themes)


def save_model(path: Path, model: CategoricalHMM, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        startprob=model.startprob_,
        transmat=model.transmat_,
        emissionprob=model.emissionprob_,
    )
    Path(str(path) + ".meta.json").write_text(json.dumps(meta, indent=2))


def load_model(path: Path) -> tuple[CategoricalHMM, dict]:
    path = Path(path)
    data = np.load(path)
    meta = json.loads(Path(str(path) + ".meta.json").read_text())
    model = _init_categorical(
        len(data["startprob"]),
        data["emissionprob"].shape[1],
        startprob=data["startprob"],
        transmat=data["transmat"],
        emissionprob=data["emissionprob"],
    )
    model.params = ""  # freeze
    return model, meta
