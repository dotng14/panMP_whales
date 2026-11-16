"""Theme–slot Gaussian HMM (Approach B) — same topology as Approach A."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np
from hmmlearn.hmm import GaussianHMM
from numpy.typing import NDArray

from analysis.song_hmm.exp4.option1 import DecodeResult

from .theme_slot import ThemeSlotSpec, _init_transmat, _project_transmat, build_spec


def _init_gaussian(
    n_states: int,
    n_features: int,
    *,
    startprob: NDArray[np.floating],
    transmat: NDArray[np.floating],
    means: NDArray[np.floating],
    covars: NDArray[np.floating],
) -> GaussianHMM:
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=60,
        tol=1e-3,
        init_params="",
        params="stmc",
        random_state=0,
    )
    model.startprob_ = startprob / startprob.sum()
    model.transmat_ = _project_transmat(transmat, (transmat > 0).astype(float))
    model.means_ = means
    model.n_features = int(means.shape[1])
    cov = np.asarray(covars, dtype=float)
    # hmmlearn may expose diag covars as (n, d), (n, 1, d), or full (n, d, d)
    if cov.ndim == 3 and cov.shape[1] == cov.shape[2]:
        cov = np.diagonal(cov, axis1=1, axis2=2)
    elif cov.ndim == 3:
        cov = cov[:, 0, :]
    cov = np.maximum(cov, 1e-4)
    model.covars_ = cov
    return model


def train_theme_slot_gauss(
    X: NDArray[np.floating],
    times: NDArray[np.floating],
    *,
    n_themes: int = 6,
    slot_counts: Sequence[int] | None = None,
    lengths: Sequence[int] | None = None,
) -> tuple[GaussianHMM, ThemeSlotSpec, dict]:
    if slot_counts is None:
        # Soft-HMM-like unequal lengths as default schematic when no phrase info
        base = [3, 3, 6, 3, 5, 8]
        if n_themes <= len(base):
            slot_counts = tuple(base[:n_themes])
        else:
            slot_counts = tuple(base + [3] * (n_themes - len(base)))
    spec = build_spec(slot_counts)
    n_states = len(spec.state_names)
    d = X.shape[1] if X.ndim == 2 else 0
    if len(X) < n_states:
        raise ValueError("not enough feature frames to train")

    # Seed means by slicing the train timeline into T equal blocks, then L slots
    means = np.zeros((n_states, d))
    covars = np.ones((n_states, d))
    edges = np.linspace(0, len(X), n_themes + 1).astype(int)
    for t in range(n_themes):
        block = X[edges[t] : edges[t + 1]]
        if len(block) == 0:
            continue
        L = spec.slot_counts[t]
        sub = np.linspace(0, len(block), L + 1).astype(int)
        for s in range(L):
            chunk = block[sub[s] : sub[s + 1]]
            st = spec.first_state[t] + s
            if len(chunk):
                means[st] = chunk.mean(axis=0)
                covars[st] = np.maximum(chunk.var(axis=0), 1e-3)

    trans = _init_transmat(spec, repeat_prior=0.9)
    start = np.zeros(n_states)
    start[spec.first_state[0]] = 1.0
    model = _init_gaussian(
        n_states,
        d,
        startprob=start,
        transmat=trans,
        means=means,
        covars=covars,
    )
    # override project with real mask
    model.transmat_ = _project_transmat(trans, spec.mask)

    if lengths is None:
        # one long sequence; split into ~hour chunks for numerical stability
        chunk = max(500, len(X) // 14)
        lengths = []
        i = 0
        while i < len(X):
            lengths.append(min(chunk, len(X) - i))
            i += lengths[-1]
    model.fit(X, lengths)
    model.transmat_ = _project_transmat(model.transmat_, spec.mask)
    meta = {
        "kind": "theme_slot_gaussian",
        "n_themes": spec.n_themes,
        "slot_counts": list(spec.slot_counts),
        "state_names": spec.state_names,
        "n_frames": int(len(X)),
        "feat_dim": int(d),
    }
    try:
        meta["train_score"] = float(model.score(X, lengths))
    except Exception:
        meta["train_score"] = None
    return model, spec, meta


def select_theme_slot_gauss(
    X: NDArray[np.floating],
    times: NDArray[np.floating],
    *,
    theme_grid: Sequence[int] = (3, 4, 5, 6),
) -> tuple[GaussianHMM, ThemeSlotSpec, dict]:
    n_obs = len(X)
    best = None
    rows = []
    for T in theme_grid:
        model, spec, meta = train_theme_slot_gauss(X, times, n_themes=T)
        ll = meta.get("train_score")
        if ll is None:
            continue
        total_ll = float(ll) * max(n_obs, 1)
        n_params = len(spec.state_names) * (X.shape[1] * 2) + spec.n_themes
        bic = -2.0 * total_ll + n_params * np.log(max(n_obs, 2))
        rows.append({"T": T, "slot_counts": list(spec.slot_counts), "avg_ll": ll, "bic": bic})
        if best is None or bic < best[0]:
            best = (bic, model, spec, meta)
    if best is None:
        return train_theme_slot_gauss(X, times, n_themes=6)
    _, model, spec, meta = best
    meta["selection"] = rows
    return model, spec, meta


def decode_theme_slot_gauss(
    model: GaussianHMM,
    X: NDArray[np.floating],
    times: NDArray[np.floating],
    spec: ThemeSlotSpec,
) -> DecodeResult:
    if len(X) == 0:
        return DecodeResult([], [], [])
    model.transmat_ = _project_transmat(model.transmat_, spec.mask)
    states = model.predict(X)
    labels = [f"T{spec.theme_of_state[int(s)] + 1}" for s in states]
    boundaries: list[float] = []
    themes: list[dict] = []
    cur = labels[0]
    t0 = float(times[0])
    count = 1
    for i in range(1, len(labels)):
        if labels[i] != cur:
            boundaries.append(float(times[i]))
            themes.append({"theme": cur, "t0": t0, "t1": float(times[i]), "n_tokens": count})
            cur = labels[i]
            t0 = float(times[i])
            count = 1
        else:
            count += 1
    themes.append({"theme": cur, "t0": t0, "t1": float(times[-1]), "n_tokens": count})
    return DecodeResult(labels, boundaries, themes)


def save_theme_slot_gauss(path: Path, model: GaussianHMM, spec: ThemeSlotSpec, meta: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cov = np.asarray(model.covars_, dtype=float)
    if cov.ndim == 3 and cov.shape[1] == cov.shape[2]:
        cov = np.diagonal(cov, axis1=1, axis2=2)
    elif cov.ndim == 3:
        cov = cov[:, 0, :]
    np.savez(
        path,
        startprob=model.startprob_,
        transmat=model.transmat_,
        means=model.means_,
        covars=cov,
        mask=spec.mask,
        slot_counts=np.asarray(spec.slot_counts, dtype=np.int32),
        theme_of_state=np.asarray(spec.theme_of_state, dtype=np.int32),
    )
    Path(str(path) + ".meta.json").write_text(json.dumps(meta, indent=2))


def load_theme_slot_gauss(path: Path) -> tuple[GaussianHMM, ThemeSlotSpec, dict]:
    path = Path(path)
    data = np.load(path)
    meta = json.loads(Path(str(path) + ".meta.json").read_text())
    slot_counts = tuple(int(x) for x in data["slot_counts"])
    spec = build_spec(slot_counts)
    model = _init_gaussian(
        len(data["startprob"]),
        data["means"].shape[1],
        startprob=data["startprob"],
        transmat=_project_transmat(data["transmat"], spec.mask),
        means=data["means"],
        covars=data["covars"],
    )
    model.params = ""
    return model, spec, meta
