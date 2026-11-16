"""Split a wav23 Session into train (<1h) and test (rest)."""

from __future__ import annotations

from analysis.song_hmm.exp4.sequence import Session, Token

from .paths import TRAIN_T1_S


def split_session_at(
    session: Session,
    *,
    cut_s: float = TRAIN_T1_S,
) -> tuple[Session, Session]:
    """Split tokens by ``t0 < cut_s`` (train) vs ``t0 >= cut_s`` (test)."""
    train_toks: list[Token] = []
    test_toks: list[Token] = []
    for t in session.tokens:
        if float(t.t0) < float(cut_s):
            train_toks.append(t)
        else:
            test_toks.append(t)
    if not train_toks:
        raise ValueError(f"no train tokens before cut_s={cut_s}")
    if not test_toks:
        raise ValueError(f"no test tokens at/after cut_s={cut_s}")
    train = Session(
        tokens=train_toks,
        t0=float(train_toks[0].t0),
        t1=float(train_toks[-1].t1),
    )
    test = Session(
        tokens=test_toks,
        t0=float(test_toks[0].t0),
        t1=float(test_toks[-1].t1),
    )
    return train, test


def split_sessions(
    sessions: list[Session],
    *,
    cut_s: float = TRAIN_T1_S,
) -> tuple[list[Session], list[Session]]:
    train: list[Session] = []
    test: list[Session] = []
    for s in sessions:
        tr, te = split_session_at(s, cut_s=cut_s)
        train.append(tr)
        test.append(te)
    return train, test
