"""Decode a recording's placements with the fixed theme model and list the songs that fit it.

usage: python3 songs.py PLACEMENTS_CSV [PLACEMENTS_CSV ...]

A song is one pass through themes 0, 1, 2, 3. It conforms when
- each of its four theme visits is given a higher likelihood by its own theme
  than by any of the other three, and
- the order matters: the best one-cycle fit of the song scores higher than
  the best one-cycle fit of the same 30-second bins in shuffled order.
"""

from __future__ import annotations

import datetime as dt
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import themes

HERE = Path(__file__).parent
N_SHUFFLES = 500


def load_model() -> themes.ThemeModel:
    z = np.load(HERE / "results" / "theme_model.npz")
    return themes.ThemeModel(z["motif_ids"], z["log_p"], z["mu"], z["sigma"])


def file_start(name: str) -> dt.datetime:
    y, mo, d, h, mi, s = map(int, re.search(r"_(\d\d)(\d\d)(\d\d)_(\d\d)(\d\d)(\d\d)", name).groups())
    return dt.datetime(2000 + y, mo, d, h, mi, s)


def one_cycle_score(model: themes.ThemeModel, counts: np.ndarray) -> float:
    """Best score of the bins as exactly one visit of each theme, in order 0, 1, 2, 3."""
    n = counts.shape[0]
    cumulative = np.vstack([np.zeros((1, themes.N_THEMES)), np.cumsum(counts @ model.log_p.T, axis=0)])
    best = np.full(n + 1, -np.inf)
    best[0] = 0.0
    for theme in range(themes.N_THEMES):
        nxt = np.full(n + 1, -np.inf)
        for end in range(1, n + 1):
            for d in range(themes.MIN_BINS, min(themes.MAX_BINS, end) + 1):
                start = end - d
                if not np.isfinite(best[start]):
                    continue
                score = best[start] + model.log_duration[theme, d] + cumulative[end, theme] - cumulative[start, theme]
                if score > nxt[end]:
                    nxt[end] = score
        best = nxt
    return float(best[n])


def songs_of_file(model: themes.ThemeModel, csv: Path, rng: np.random.Generator) -> pd.DataFrame:
    df = pd.read_csv(csv)
    start = file_start(csv.name)
    n_bins = int(np.ceil(df.end_sec.max() / themes.BIN_SEC))
    counts = model.bin_counts(df.motif.values, df.start_sec.values, 0.0, n_bins)
    _score, visits = model.decode(counts)
    rows = []
    firsts = [i for i, v in enumerate(visits) if v[0] == 0]
    for a, b in zip(firsts[:-1], firsts[1:]):
        cycle = visits[a:b]
        if [v[0] for v in cycle] != [0, 1, 2, 3]:
            continue
        margins = []
        for visit in cycle:
            scores = model.visit_scores(counts, visit)
            margins.append(float(scores[visit[0]] - np.max(np.delete(scores, visit[0]))))
        first_bin, end_bin = cycle[0][1], cycle[-1][2]
        block = counts[first_bin:end_bin]
        actual = one_cycle_score(model, block)
        null = np.array([one_cycle_score(model, block[rng.permutation(len(block))]) for _ in range(N_SHUFFLES)]) if min(margins) > 0 else np.array([np.nan])
        rows.append(
            {
                "file": csv.name.replace("placements_", "").replace(".csv", ""),
                "start_utc": start + dt.timedelta(seconds=first_bin * themes.BIN_SEC),
                "end_utc": start + dt.timedelta(seconds=end_bin * themes.BIN_SEC),
                "start_sec": first_bin * themes.BIN_SEC,
                "end_sec": end_bin * themes.BIN_SEC,
                "minutes": (end_bin - first_bin) * themes.BIN_SEC / 60.0,
                "placements": int(block.sum()),
                "theme_minutes": "/".join(f"{(v[2] - v[1]) / 2:g}" for v in cycle),
                "theme_starts_sec": "/".join(f"{v[1] * themes.BIN_SEC:g}" for v in cycle),
                "margins": "/".join(f"{m:.1f}" for m in margins),
                "min_margin": min(margins),
                "order_z": float((actual - null.mean()) / null.std()) if null.size > 1 else np.nan,
                "shuffles_at_or_above": int(np.sum(null >= actual)) if null.size > 1 else -1,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    model = load_model()
    rng = np.random.default_rng(0)
    tables = [songs_of_file(model, Path(name), rng) for name in sys.argv[1:]]
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(HERE / "results" / "decoded_songs.csv", index=False)
    good = table[(table.min_margin > 0) & (table.shuffles_at_or_above == 0)]
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(f"cycles decoded {len(table)}, conforming {len(good)}")
    print(good[["file", "start_utc", "minutes", "placements", "theme_minutes", "margins", "min_margin", "order_z"]].to_string(index=False))


if __name__ == "__main__":
    main()
