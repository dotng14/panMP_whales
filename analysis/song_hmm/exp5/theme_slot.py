"""Flexible theme–slot categorical HMM (Approach A).

Topology: each theme is a left-to-right slot chain (no skips). From the last
slot: repeat (→ first slot of same theme) or next (→ first slot of next theme).
``T`` and per-theme slot counts are chosen on train.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from hmmlearn.hmm import CategoricalHMM
from numpy.typing import NDArray

from analysis.song_hmm.exp4.option1 import DecodeResult
from analysis.song_hmm.exp4.phrases import cut_phrases, normalised_edit
from analysis.song_hmm.exp4.sequence import Session, encode_session, vocabulary


def _init_categorical(
    n_states: int,
    n_features: int,
    *,
    startprob: NDArray[np.floating],
    transmat: NDArray[np.floating],
    emissionprob: NDArray[np.floating],
) -> CategoricalHMM:
    model = CategoricalHMM(
        n_components=n_states,
        n_features=n_features,
        n_iter=80,
        tol=1e-4,
        init_params="",
        params="ste",
        random_state=0,
    )
    model.startprob_ = startprob / startprob.sum()
    model.transmat_ = transmat / np.maximum(transmat.sum(axis=1, keepdims=True), 1e-12)
    model.emissionprob_ = emissionprob / np.maximum(
        emissionprob.sum(axis=1, keepdims=True), 1e-12
    )
    return model


@dataclass
class ThemeSlotSpec:
    n_themes: int
    slot_counts: tuple[int, ...]  # length T
    state_names: list[str]
    theme_of_state: list[int]
    slot_of_state: list[int]
    first_state: list[int]
    last_state: list[int]
    mask: NDArray[np.floating]  # allowed transitions (0/1)


def build_spec(slot_counts: Sequence[int]) -> ThemeSlotSpec:
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
        for s in range(L - 1):
            mask[base + s, base + s + 1] = 1.0  # advance only
        # last slot: repeat or next (or stay on last theme via repeat)
        last = last_state[t]
        mask[last, first_state[t]] = 1.0  # repeat
        if t + 1 < n_themes:
            mask[last, first_state[t + 1]] = 1.0  # next
        else:
            mask[last, first_state[t]] = 1.0  # only repeat on final theme
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


def _project_transmat(trans: NDArray[np.floating], mask: NDArray[np.floating]) -> NDArray[np.floating]:
    out = np.maximum(trans, 0.0) * mask
    row = out.sum(axis=1, keepdims=True)
    # if a row was zeroed, fall back to uniform on allowed
    bad = row.ravel() < 1e-12
    if np.any(bad):
        for i in np.where(bad)[0]:
            allowed = mask[i] > 0
            out[i, allowed] = 1.0 / max(int(allowed.sum()), 1)
        row = out.sum(axis=1, keepdims=True)
    return out / np.maximum(row, 1e-12)


def _cluster_phrases(phrases, n_themes: int) -> list[tuple[int, ...]]:
    if not phrases:
        raise ValueError("no phrases")
    medoids = [phrases[0].motifs]
    for _ in range(1, n_themes):
        best, best_d = None, -1.0
        for ph in phrases:
            d = min(normalised_edit(ph.motifs, m) for m in medoids)
            if d > best_d:
                best, best_d = ph.motifs, d
        if best is None:
            break
        medoids.append(best)
    return medoids


def _slot_counts_from_medoids(medoids: Sequence[Sequence[int]], *, min_slots: int = 2, max_slots: int = 10) -> tuple[int, ...]:
    counts = []
    for m in medoids:
        L = int(np.clip(len(m), min_slots, max_slots))
        counts.append(max(L, min_slots))
    return tuple(counts)


def _seed_emissions(
    spec: ThemeSlotSpec,
    medoids: Sequence[Sequence[int]],
    vocab: Sequence[str],
) -> NDArray[np.floating]:
    n_states = len(spec.state_names)
    n_features = len(vocab)
    emission = np.ones((n_states, n_features))
    index = {v: i for i, v in enumerate(vocab)}
    for t, med in enumerate(medoids[: spec.n_themes]):
        L = spec.slot_counts[t]
        base = spec.first_state[t]
        for s in range(L):
            st = base + s
            if s < len(med):
                key = f"M{med[s]}"
                if key in index:
                    emission[st, index[key]] += 25.0
            # gaps lightly discouraged on motif slots
        for g in ("GAP_S", "GAP_L", "BREAK"):
            if g in index:
                emission[base : base + L, index[g]] *= 0.2
    return emission / emission.sum(axis=1, keepdims=True)


def _init_transmat(spec: ThemeSlotSpec, *, repeat_prior: float = 0.85) -> NDArray[np.floating]:
    n = len(spec.state_names)
    trans = np.zeros((n, n))
    for t, L in enumerate(spec.slot_counts):
        base = spec.first_state[t]
        for s in range(L - 1):
            trans[base + s, base + s + 1] = 1.0
        last = spec.last_state[t]
        if t + 1 < spec.n_themes:
            trans[last, spec.first_state[t]] = repeat_prior
            trans[last, spec.first_state[t + 1]] = 1.0 - repeat_prior
        else:
            trans[last, spec.first_state[t]] = 1.0
    return _project_transmat(trans, spec.mask)


def train_theme_slot(
    sessions: Sequence[Session],
    vocab: Sequence[str],
    *,
    n_themes: int = 6,
    slot_counts: Sequence[int] | None = None,
    n_iter: int = 40,
) -> tuple[CategoricalHMM, ThemeSlotSpec, dict]:
    phrases = []
    for s in sessions:
        phrases.extend(cut_phrases(s))
    if not phrases:
        raise ValueError("no phrases to train theme–slot HMM")
    medoids = _cluster_phrases(phrases, n_themes)
    if slot_counts is None:
        slot_counts = _slot_counts_from_medoids(medoids)
    else:
        slot_counts = tuple(slot_counts)
        if len(slot_counts) != n_themes:
            raise ValueError("slot_counts length must equal n_themes")

    spec = build_spec(slot_counts)
    emission = _seed_emissions(spec, medoids, vocab)
    trans = _init_transmat(spec)
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
    seqs = [x for x in seqs if len(x) >= max(spec.slot_counts)]
    meta = {
        "kind": "theme_slot_categorical",
        "n_themes": spec.n_themes,
        "slot_counts": list(spec.slot_counts),
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
    # Fit, then hard-project transitions onto topology (no skips).
    model.fit(X, lengths)
    model.transmat_ = _project_transmat(model.transmat_, spec.mask)
    model.startprob_ = np.maximum(model.startprob_, 0.0)
    model.startprob_ = model.startprob_ / model.startprob_.sum()
    # keep start mass on theme-1 first slot mostly
    model.startprob_ = 0.7 * start + 0.3 * model.startprob_
    model.startprob_ = model.startprob_ / model.startprob_.sum()
    try:
        meta["train_score"] = float(model.score(X, lengths))
    except Exception:
        meta["train_score"] = None
    return model, spec, meta


def select_theme_slot(
    sessions: Sequence[Session],
    vocab: Sequence[str],
    *,
    theme_grid: Sequence[int] = (3, 4, 5, 6, 7, 8),
) -> tuple[CategoricalHMM, ThemeSlotSpec, dict]:
    """Choose T by train log-likelihood BIC; slot counts from phrase medoids."""
    phrases = []
    for s in sessions:
        phrases.extend(cut_phrases(s))
    n_obs = sum(len(encode_session(s, vocab)) for s in sessions)
    best = None
    rows = []
    for T in theme_grid:
        if T > max(1, len(phrases)):
            continue
        model, spec, meta = train_theme_slot(sessions, vocab, n_themes=T)
        ll = meta.get("train_score")
        if ll is None:
            continue
        n_params = (
            spec.n_themes  # repeat probs roughly
            + len(spec.state_names) * (len(vocab) - 1)  # emissions
        )
        bic = -2.0 * float(ll) * max(n_obs, 1) / max(n_obs, 1) + n_params * np.log(max(n_obs, 2))
        # hmmlearn score is average loglik per sample; convert
        total_ll = float(ll) * max(n_obs, 1)
        bic = -2.0 * total_ll + n_params * np.log(max(n_obs, 2))
        row = {"T": T, "slot_counts": list(spec.slot_counts), "avg_ll": ll, "bic": bic}
        rows.append(row)
        if best is None or bic < best[0]:
            best = (bic, model, spec, meta)
    if best is None:
        return train_theme_slot(sessions, vocab, n_themes=6)
    _, model, spec, meta = best
    meta["selection"] = rows
    return model, spec, meta


def decode_theme_slot(
    model: CategoricalHMM,
    session: Session,
    vocab: Sequence[str],
    spec: ThemeSlotSpec,
) -> DecodeResult:
    X = encode_session(session, vocab)
    if len(X) == 0:
        return DecodeResult([], [], [])
    # ensure topology before decode
    model.transmat_ = _project_transmat(model.transmat_, spec.mask)
    states = model.predict(X.reshape(-1, 1))
    labels = [f"T{spec.theme_of_state[int(s)] + 1}" for s in states]
    times = [t.t0 for t in session.tokens]
    # encode_session may drop unknown tokens — align by re-encoding keys
    kept_times: list[float] = []
    index = {v: i for i, v in enumerate(vocab)}
    for t in session.tokens:
        key = f"M{t.motif}" if t.kind == "motif" else t.kind
        if key in index:
            kept_times.append(t.t0)
    n = min(len(labels), len(kept_times))
    labels, times = labels[:n], kept_times[:n]
    boundaries: list[float] = []
    themes: list[dict] = []
    cur = labels[0]
    t0 = times[0]
    count = 1
    for i in range(1, n):
        if labels[i] != cur:
            boundaries.append(times[i])
            themes.append({"theme": cur, "t0": t0, "t1": times[i], "n_tokens": count})
            cur = labels[i]
            t0 = times[i]
            count = 1
        else:
            count += 1
    themes.append({"theme": cur, "t0": t0, "t1": times[-1], "n_tokens": count})
    return DecodeResult(labels, boundaries, themes)


def save_theme_slot(path: Path, model: CategoricalHMM, spec: ThemeSlotSpec, meta: dict) -> None:
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
    )
    Path(str(path) + ".meta.json").write_text(json.dumps(meta, indent=2))


def load_theme_slot(path: Path) -> tuple[CategoricalHMM, ThemeSlotSpec, dict]:
    path = Path(path)
    data = np.load(path)
    meta = json.loads(Path(str(path) + ".meta.json").read_text())
    slot_counts = tuple(int(x) for x in data["slot_counts"])
    spec = build_spec(slot_counts)
    model = _init_categorical(
        len(data["startprob"]),
        data["emissionprob"].shape[1],
        startprob=data["startprob"],
        transmat=_project_transmat(data["transmat"], spec.mask),
        emissionprob=data["emissionprob"],
    )
    model.params = ""
    return model, spec, meta
