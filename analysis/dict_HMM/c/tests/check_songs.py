"""Check the theme HMM against the published song intervals and CSV columns."""

import sys
from pathlib import Path

EXPECTED = [
    (1, 0.0, 8.0),
    (2, 8.0, 20.5),
    (3, 20.5, 28.5),
    (4, 28.5, 39.5),
    (5, 39.5, 52.5),
    (6, 52.5, 65.5),
    (7, 65.5, 75.0),
]
UNIX0 = 1682557875.0


def main() -> None:
    seg_path, mot_path = Path(sys.argv[1]), Path(sys.argv[2])
    seg_lines = seg_path.read_text().strip().splitlines()
    header = seg_lines[0]
    if header != "type,segment index,start_unix,end_unix":
        raise SystemExit(f"segment header {header}")
    songs = []
    themes = []
    motifs = []
    for line in seg_lines[1:]:
        kind, index, start, end = line.split(",")
        if kind == "song":
            songs.append((int(index), (float(start) - UNIX0) / 60.0, (float(end) - UNIX0) / 60.0))
        elif kind == "theme":
            themes.append(int(index))
        elif kind == "motif":
            motifs.append(int(index))
    if len(songs) != 7:
        raise SystemExit(f"expected 7 songs, got {len(songs)}: {songs}")
    for got, exp in zip(songs, EXPECTED):
        if got[0] != exp[0] or abs(got[1] - exp[1]) > 1e-6 or abs(got[2] - exp[2]) > 1e-6:
            raise SystemExit(f"song mismatch {got} vs {exp}")
    if len(themes) != 27:
        raise SystemExit(f"expected 27 themes, got {len(themes)}")
    if themes[0] != 0 or themes[-1] != 2:
        raise SystemExit(f"theme cycle ends unexpectedly: {themes[:4]} ... {themes[-4:]}")
    for a, b in zip(themes, themes[1:]):
        if (a + 1) % 4 != b and not (a == 3 and b == 0):
            # cycle is strict except we already check +1 mod 4
            pass
        if (a + 1) % 4 != b:
            raise SystemExit(f"theme order breaks at {a} -> {b}")
    mot_header = mot_path.read_text().splitlines()[0]
    if mot_header != "motif,song,theme,phrase_id,phrase_count,start_unix,end_unix":
        raise SystemExit(f"motif header {mot_header}")
    if len(motifs) != 2367:
        raise SystemExit(f"expected 2367 motifs, got {len(motifs)}")
    print("song and theme checks passed")


if __name__ == "__main__":
    main()
