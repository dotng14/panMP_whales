"""Export kept placements of wav 2 and wav 3 as 'motif start_sec end_sec'."""

import json
import sys
from pathlib import Path

ROOT = Path("/home/yfreund/whale_processing/result/LeiData")


def rows(stem: str, offset_sec: float):
    directory = ROOT / f"{stem}.m8x32_refine2_placed"
    matchings = json.loads((directory / "matchings.json").read_text())
    windows = json.loads((directory / "windows.json").read_text())
    out = []
    for row in matchings:
        meta = windows[row["window_id"]]
        hop = meta["overlap"] / meta["sample_freq"]
        start = (meta["start_sample"] - 1) / meta["sample_freq"] + row["t_start"] * hop + offset_sec
        end = start + row["width"] * hop
        out.append((start, end, int(row["motif_id"])))
    return out


def main() -> None:
    placed = rows("SOCAL_H_76_230427_011115.x", 0.0) + rows("SOCAL_H_76_230427_014845.x", 37.5 * 60.0)
    placed.sort()
    path = Path(sys.argv[1])
    path.write_text("".join(f"{motif} {start:.10f} {end:.10f}\n" for start, end, motif in placed))
    print(f"wrote {len(placed)} placements")


if __name__ == "__main__":
    main()
