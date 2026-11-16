"""Weaker filter: theme-2 visits that prefer theme 2 over the other themes.

usage: python3 theme2_filter.py

A visit of theme 2 from the fixed four-theme decode is kept when its motif
counts score higher under theme 2 than under any of the other three
(theme2_margin > 0). That uses only the theme-2 motif distribution against
the alternatives; it does not require themes 0, 1, and 3 to pass, and it
does not run the shuffle test. Visits that overlap a conforming song are
dropped. The ranked list is written to results/theme2_outside_songs.csv.
"""

from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import numpy as np
import pandas as pd

import themes

HERE = Path(__file__).parent
THEME = 2
TOP_MOTIFS = (43, 1, 44, 31, 27, 0, 10)


def load_model() -> themes.ThemeModel:
    z = np.load(HERE / "results" / "theme_model.npz")
    return themes.ThemeModel(z["motif_ids"], z["log_p"], z["mu"], z["sigma"])


def file_start(name: str) -> dt.datetime:
    y, mo, d, h, mi, s = map(int, re.search(r"_(\d\d)(\d\d)(\d\d)_(\d\d)(\d\d)(\d\d)", name).groups())
    return dt.datetime(2000 + y, mo, d, h, mi, s)


def conforming_song_spans(songs: pd.DataFrame) -> dict[str, list[tuple[float, float]]]:
    good = songs[(songs.min_margin > 0) & (songs.shuffles_at_or_above == 0)]
    spans: dict[str, list[tuple[float, float]]] = {}
    for row in good.itertuples():
        spans.setdefault(row.file, []).append((float(row.start_sec), float(row.end_sec)))
    return spans


def overlaps(begin: float, end: float, spans: list[tuple[float, float]]) -> bool:
    return any(begin < hi and end > lo for lo, hi in spans)


def theme2_visits(model: themes.ThemeModel, csv: Path, song_spans: dict[str, list[tuple[float, float]]]) -> pd.DataFrame:
    stem = csv.name.replace("placements_", "").replace(".csv", "")
    df = pd.read_csv(csv)
    start = file_start(stem)
    n_bins = int(np.ceil(df.end_sec.max() / themes.BIN_SEC))
    counts = model.bin_counts(df.motif.values, df.start_sec.values, 0.0, n_bins)
    _score, visits = model.decode(counts)
    rows = []
    for theme, first, end in visits:
        if theme != THEME:
            continue
        scores = model.visit_scores(counts, (theme, first, end))
        margin = float(scores[THEME] - np.max(np.delete(scores, THEME)))
        begin = first * themes.BIN_SEC
        finish = end * themes.BIN_SEC
        mix = {m: int(counts[first:end, model.column[m]].sum()) for m in TOP_MOTIFS if m in model.column}
        rows.append(
            {
                "file": stem,
                "start_utc": start + dt.timedelta(seconds=begin),
                "end_utc": start + dt.timedelta(seconds=finish),
                "start_sec": begin,
                "end_sec": finish,
                "minutes": (finish - begin) / 60.0,
                "placements": int(counts[first:end].sum()),
                "theme2_margin": margin,
                "theme2_score": float(scores[THEME]),
                "in_song": overlaps(begin, finish, song_spans.get(stem, [])),
                **{f"n_{m}": mix.get(m, 0) for m in TOP_MOTIFS},
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    model = load_model()
    songs = pd.read_csv(HERE / "results" / "decoded_songs.csv")
    song_spans = conforming_song_spans(songs)
    tables = [theme2_visits(model, csv, song_spans) for csv in sorted((HERE / "results").glob("placements_*.csv"))]
    table = pd.concat(tables, ignore_index=True)
    table.to_csv(HERE / "results" / "theme2_all_visits.csv", index=False)
    outside = table[(table.theme2_margin > 0) & ~table.in_song].sort_values("theme2_margin", ascending=False)
    outside.to_csv(HERE / "results" / "theme2_outside_songs.csv", index=False)
    print(f"theme-2 visits {len(table)}, margin > 0 {int((table.theme2_margin > 0).sum())}, outside songs {len(outside)}")
    print(outside.head(10)[["file", "start_utc", "minutes", "theme2_margin", "n_43", "n_1", "n_44"]].to_string(index=False))


if __name__ == "__main__":
    main()
