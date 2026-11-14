"""Compare the C detector's output with the Python detector's.

The C detector matches templates in single precision on the GPUs, so a few
placements that tie to about six digits go to a neighboring motif or frame.
The comparison passes when at least MIN_SHARED of the Python placements are
reproduced and the decoded cycles and conforming songs are the same.

usage: python3 compare.py PYTHON_RESULTS_DIR C_OUT_DIR
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

MIN_SHARED = 0.999


def placement_keys(path: Path) -> set[tuple[int, int]]:
    table = pd.read_csv(path)
    return set(zip(table.motif.astype(int), (table.start_sec * 1000).round().astype("int64")))


def song_keys(path: Path) -> set[tuple[str, int, int, str]]:
    table = pd.read_csv(path)
    good = table[(table.min_margin > 0) & (table.shuffles_at_or_above == 0)]
    return set(zip(good.file, good.start_sec.astype(int), good.end_sec.astype(int), good.theme_starts_sec))


def main() -> None:
    reference, out = Path(sys.argv[1]), Path(sys.argv[2])
    ok = True
    total = shared = 0
    for theirs in sorted(reference.glob("placements_*.csv")):
        mine = out / theirs.name
        if not mine.exists():
            print(f"{theirs.name}: not produced")
            ok = False
            continue
        a, b = placement_keys(theirs), placement_keys(mine)
        print(f"{theirs.name}: python {len(a)}, c {len(b)}, same {len(a & b)}, python only {len(a - b)}, c only {len(b - a)}")
        total += len(a)
        shared += len(a & b)
    share = shared / total if total else 0.0
    print(f"placements: {shared} of {total} python placements reproduced ({100 * share:.3f}%)")
    ok = ok and share >= MIN_SHARED
    a, b = song_keys(reference / "decoded_songs.csv"), song_keys(out / "decoded_songs.csv")
    print(f"conforming songs: python {len(a)}, c {len(b)}, same {len(a & b)}")
    for key in sorted(a ^ b):
        print("  differs:", key, "python" if key in a else "c")
    ok = ok and a == b
    cycles_a = pd.read_csv(reference / "decoded_songs.csv")
    cycles_b = pd.read_csv(out / "decoded_songs.csv")
    same_cycles = len(cycles_a) == len(cycles_b) and (cycles_a.theme_starts_sec.values == cycles_b.theme_starts_sec.values).all()
    print(f"decoded cycles: python {len(cycles_a)}, c {len(cycles_b)}, same theme boundaries: {bool(same_cycles)}")
    ok = ok and bool(same_cycles)
    print("MATCH" if ok else "DIFFERENT")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
