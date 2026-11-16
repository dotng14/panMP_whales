"""Option 2: repetition HMM on phrase-distance features."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
from hmmlearn.hmm import GaussianHMM
from numpy.typing import NDArray

from .option1 import DecodeResult
from .phrases import (
    cut_phrases,
    features_matrix,
    phrase_feature_rows,
    standardise,
)
from .sequence import Session


@dataclass
class Option2Model:
    hmm: GaussianHMM
    mean: NDArray[np.floating]
    std: NDArray[np.floating]
    m: int
    state_names: list[str]


def _state_names(m: int) -> list[str]:
    return ["C"] + [f"R{i}" for i in range(1, m + 1)] + ["R"]


def _init_gaussian(n_states: int, n_features: int = 5) -> GaussianHMM:
    model = GaussianHMM(
        n_components=n_states,
        covariance_type="diag",
        n_iter=100,
        tol=1e-4,
        init_params="",
        params="stmc",
        random_state=0,
    )
    model.startprob_ = np.full(n_states, 1.0 / n_states)
    model.transmat_ = np.full((n_states, n_states), 1.0 / n_states)
    model.means_ = np.zeros((n_states, n_features))
    model.covars_ = np.ones((n_states, n_features))
    return model


def _label_phrases(phrases, marks: Sequence[float], m: int) -> list[int]:
    """Map phrases to state indices using hand marks (first phrase after mark = C)."""
    marks = sorted(marks)
    labels: list[int] = []
    since_c = 10**9
    mark_i = 0
    for i, ph in enumerate(phrases):
        hit = False
        while mark_i < len(marks) and marks[mark_i] <= ph.t0:
            hit = True
            mark_i += 1
        if i == 0 or hit:
            labels.append(0)
            since_c = 0
        elif since_c < m:
            since_c += 1
            labels.append(since_c)  # R1..Rm
        else:
            since_c += 1
            labels.append(m + 1)  # R
    return labels


def train_option2(
    sessions: Sequence[Session],
    marks: Sequence[float],
    *,
    m: int = 2,
) -> Option2Model:
    state_names = _state_names(m)
    n_states = len(state_names)
    all_X = []
    all_y = []
    lengths = []
    for s in sessions:
        phrases = cut_phrases(s)
        if len(phrases) < 3:
            continue
        feats = phrase_feature_rows(phrases)
        X = features_matrix(feats)
        y = _label_phrases(phrases, [mk for mk in marks if s.t0 - 60 <= mk <= s.t1 + 60], m)
        if len(y) != len(X):
            # fall back: unsupervised-ish — high d1 → C
            y = []
            for i, fe in enumerate(feats):
                if i == 0 or fe.d1 >= 0.5:
                    y.append(0)
                    since = 0
                else:
                    since = since + 1 if i else 0
                    if since <= m:
                        y.append(since)
                    else:
                        y.append(m + 1)
        all_X.append(X)
        all_y.append(np.asarray(y, dtype=int))
        lengths.append(len(X))
    if not all_X:
        raise ValueError("no phrase sequences for option 2")
    Xcat = np.vstack(all_X)
    Xz, mean, std = standardise(Xcat)
    # supervised means
    ycat = np.concatenate(all_y)
    means = np.zeros((n_states, 5))
    covars = np.ones((n_states, 5))
    for st in range(n_states):
        mask = ycat == st
        if mask.any():
            means[st] = Xz[mask].mean(axis=0)
            covars[st] = np.maximum(Xz[mask].var(axis=0), 1e-2)
        else:
            means[st] = Xz.mean(axis=0)
            covars[st] = np.maximum(Xz.var(axis=0), 1e-2)
    # transitions: C→R1→…→Rm→R↺, R→C, C→C
    trans = np.zeros((n_states, n_states))
    # C
    trans[0, 0] = 0.05
    trans[0, 1] = 0.95
    for i in range(1, m):
        trans[i, i + 1] = 1.0
    trans[m, m + 1] = 1.0  # Rm → R
    trans[m + 1, m + 1] = 0.9
    trans[m + 1, 0] = 0.1
    trans = np.maximum(trans, 1e-6)
    trans = trans / trans.sum(axis=1, keepdims=True)
    start = np.zeros(n_states)
    start[0] = 1.0

    model = _init_gaussian(n_states)
    model.startprob_ = start
    model.transmat_ = trans
    model.means_ = means
    model.covars_ = covars

    # refine with Baum–Welch
    # rebuild standardised sequences
    seqs = []
    for X in all_X:
        Xz_i, _, _ = standardise(X, mean, std)
        seqs.append(Xz_i)
    Xfit = np.vstack(seqs)
    model.fit(Xfit, lengths)
    return Option2Model(hmm=model, mean=mean, std=std, m=m, state_names=state_names)


def decode_option2(model: Option2Model, session: Session) -> DecodeResult:
    phrases = cut_phrases(session)
    if not phrases:
        return DecodeResult([], [], [])
    feats = phrase_feature_rows(phrases)
    X = features_matrix(feats)
    Xz, _, _ = standardise(X, model.mean, model.std)
    states = model.hmm.predict(Xz)
    labels = [model.state_names[int(s)] for s in states]
    boundaries = [phrases[i].t0 for i, lab in enumerate(labels) if lab == "C" and i > 0]
    # also first C starts a theme but is not a "change"
    themes: list[dict] = []
    cur_i = 0
    theme_id = 0
    for i, lab in enumerate(labels):
        if lab == "C" and i > 0:
            themes.append(
                {
                    "theme": f"seg{theme_id}",
                    "t0": phrases[cur_i].t0,
                    "t1": phrases[i - 1].t1,
                    "n_phrases": i - cur_i,
                }
            )
            theme_id += 1
            cur_i = i
    themes.append(
        {
            "theme": f"seg{theme_id}",
            "t0": phrases[cur_i].t0,
            "t1": phrases[-1].t1,
            "n_phrases": len(phrases) - cur_i,
        }
    )
    return DecodeResult(theme_path=labels, boundaries=boundaries, themes=themes)


def save_option2(path: Path, model: Option2Model) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        startprob=model.hmm.startprob_,
        transmat=model.hmm.transmat_,
        means=model.hmm.means_,
        covars=model.hmm.covars_,
        mean=model.mean,
        std=model.std,
        m=model.m,
    )
    Path(str(path) + ".meta.json").write_text(
        json.dumps({"state_names": model.state_names, "m": model.m}, indent=2)
    )


def load_option2(path: Path) -> Option2Model:
    path = Path(path)
    data = np.load(path)
    meta = json.loads(Path(str(path) + ".meta.json").read_text())
    m = int(data["m"])
    hmm = _init_gaussian(len(data["startprob"]))
    hmm.startprob_ = data["startprob"]
    hmm.transmat_ = data["transmat"]
    hmm.means_ = data["means"]
    hmm.covars_ = data["covars"]
    hmm.params = ""
    return Option2Model(
        hmm=hmm,
        mean=data["mean"],
        std=data["std"],
        m=m,
        state_names=meta["state_names"],
    )
