"""Fit the four themes on the known 75 minutes, from the 10 kHz recording.

Places the dictionary on 27 April 2023, 01:11:15 to 02:26:15 UTC (wav 2 and
wav 3), runs the C theme fit of `03_themes_and_songs.md` on those placements,
and saves each theme's motif probabilities and visit-length parameters.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

import place
import themes

HERE = Path(__file__).parent
WAV = Path("/mnt/sda/SiteH/SOCAL_H_76/SOCAL_H_76_disk01_df20/SOCAL_H_76_230426_054845_df20.x.wav")
WAV2_OFFSET_SEC = 69750.0  # 01:11:15 on 27 April, from the file start at 05:48:45 on 26 April
WAV2_UNIX = 1682557875.0
HMM_RUN = HERE.parent / "c" / "build" / "hmm_run"


def main() -> None:
    out = HERE / "results"
    out.mkdir(exist_ok=True)
    dictionary = place.load_dictionary()
    rows = place.place_span(WAV, WAV2_OFFSET_SEC, 60, dictionary)
    placements = out / "reference_placements.txt"
    with open(placements, "w") as handle:
        for motif_id, begin, end, _unexplained in rows:
            handle.write(f"{motif_id} {begin - WAV2_OFFSET_SEC:.6f} {end - WAV2_OFFSET_SEC:.6f}\n")
    segments = out / "reference_segments.csv"
    subprocess.run(
        [str(HMM_RUN), str(placements), str(segments), str(out / "reference_motif_segments.csv")], check=True
    )
    table = pd.read_csv(segments)
    visits = [
        (int(row["segment index"]), row.start_unix - WAV2_UNIX, row.end_unix - WAV2_UNIX)
        for _, row in table[table.type == "theme"].iterrows()
    ]
    motifs = np.array([row[0] for row in rows])
    starts = np.array([row[1] - WAV2_OFFSET_SEC for row in rows])
    model = themes.ThemeModel.fit(dictionary.ids, motifs, starts, visits)
    np.savez(out / "theme_model.npz", motif_ids=dictionary.ids, log_p=model.log_p, mu=model.mu, sigma=model.sigma)
    print(f"placements {len(rows)}, theme visits {len(visits)}")


if __name__ == "__main__":
    main()
