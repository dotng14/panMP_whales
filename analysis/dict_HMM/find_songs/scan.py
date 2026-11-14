"""Place the dictionary on whole Site H recordings and save the placements.

usage: python3 scan.py OUT_DIR WAV [WAV ...]
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import place
import spectro


def main() -> None:
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    dictionary = place.load_dictionary()
    for name in sys.argv[2:]:
        wav = Path(name)
        target = out_dir / f"placements_{wav.name.split('.')[0]}.csv"
        if target.exists():
            print(f"{target.name} exists", flush=True)
            continue
        began = time.time()
        _rate, _offset, n_samples = spectro.wav_layout(wav)
        n_windows = n_samples // (spectro.WINDOW_SEC * spectro.SOURCE_RATE)
        rows = place.place_span(wav, 0.0, n_windows, dictionary)
        with open(target, "w") as handle:
            handle.write("motif,start_sec,end_sec,unexplained\n")
            for motif_id, begin, end, unexplained in rows:
                handle.write(f"{motif_id},{begin:.5f},{end:.5f},{unexplained:.6f}\n")
        print(f"{wav.name}: {n_windows} windows, {len(rows)} placements, {time.time() - began:.0f} s", flush=True)


if __name__ == "__main__":
    main()
