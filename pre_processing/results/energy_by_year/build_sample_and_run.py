#!/usr/bin/env python3
"""Sample Site H df20 wavs (up to 4 per calendar month) and run energy_filter --cpu.

Writes segments_monthly.csv and PROGRESS.md. Safe to re-run; skips files already
present in the CSV.
"""

from __future__ import annotations

import csv
import re
import struct
import subprocess
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path("/mnt/sda/SiteH")
HERE = Path(__file__).resolve().parent
EF = Path(__file__).resolve().parents[2] / "c" / "energy_filter"
OUT = HERE / "segments_monthly.csv"
PROGRESS = HERE / "PROGRESS.md"
STAMP = re.compile(r"(?<!\d)(\d{6})_(\d{6})(?!\d)")
PER_MONTH = 4


def wav_rate(path: Path) -> int | None:
    with path.open("rb") as handle:
        if handle.read(12)[:4] != b"RIFF":
            return None
        while True:
            chunk = handle.read(8)
            if len(chunk) < 8:
                return None
            name, size = struct.unpack("<4sI", chunk)
            if name == b"fmt ":
                body = handle.read(size)
                return struct.unpack("<I", body[4:8])[0]
            handle.seek(size + (size & 1), 1)


def parse_time(name: str) -> datetime | None:
    m = STAMP.search(name)
    if not m:
        return None
    a, b = m.group(1), m.group(2)
    yy, mo, dd = int(a[0:2]), int(a[2:4]), int(a[4:6])
    hh, mi, ss = int(b[0:2]), int(b[2:4]), int(b[4:6])
    candidates = []
    if 1 <= mo <= 12 and 1 <= dd <= 31:
        candidates.append((2000 + yy, mo, dd))
    mo2, dd2, yy2 = int(a[0:2]), int(a[2:4]), int(a[4:6])
    if 1 <= mo2 <= 12 and 1 <= dd2 <= 31:
        candidates.append((2000 + yy2, mo2, dd2))
    for year, month, day in candidates:
        try:
            return datetime(year, month, day, hh, mi, ss)
        except ValueError:
            continue
    return None


def write_progress(done: int, total: int, current: str, began: float) -> None:
    elapsed = time.time() - began
    rate = done / elapsed if elapsed > 0 and done else 0.0
    eta = (total - done) / rate if rate > 0 else float("nan")
    pct = 100.0 * done / total if total else 0.0
    bar_n = int(pct // 2)
    bar = "█" * bar_n + "·" * (50 - bar_n)
    PROGRESS.write_text(
        f"""# Energy-by-year sample — progress

**Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Status:** {'done' if done >= total else 'running'}

```
[{bar}] {pct:5.1f}%
```

| | |
|---|---:|
| Files in sample | {total} |
| Completed | {done} |
| Elapsed | {elapsed / 3600:.2f} h |
| ETA | {eta / 3600:.1f} h |

**Current**  
`{current or '—'}`

Output: `{OUT.name}` (600 s GPL whitened 50–500 Hz / overall ratios).
"""
    )


def already_done() -> set[str]:
    if not OUT.exists():
        return set()
    names = set()
    with OUT.open(newline="") as handle:
        for row in csv.DictReader(handle):
            names.add(row["filename"])
    return names


def pick_sample() -> list[tuple[datetime, Path]]:
    by_ym: dict[tuple[int, int], list[tuple[datetime, Path]]] = defaultdict(list)
    for path in sorted(ROOT.rglob("*df20*.wav")):
        rate = wav_rate(path)
        if rate not in (10000, 16000):
            continue
        t = parse_time(path.name)
        if t is None:
            continue
        by_ym[(t.year, t.month)].append((t, path))
    chosen: list[tuple[datetime, Path]] = []
    for key in sorted(by_ym):
        rows = sorted(by_ym[key], key=lambda x: x[0])
        if len(rows) <= PER_MONTH:
            chosen.extend(rows)
        else:
            # evenly spaced indices through the month
            for i in range(PER_MONTH):
                idx = int(round(i * (len(rows) - 1) / (PER_MONTH - 1)))
                chosen.append(rows[idx])
    # unique by path, keep time order
    seen = set()
    out = []
    for t, p in sorted(chosen, key=lambda x: x[0]):
        if p in seen:
            continue
        seen.add(p)
        out.append((t, p))
    return out


def main() -> None:
    HERE.mkdir(parents=True, exist_ok=True)
    sample = pick_sample()
    done_names = already_done()
    todo = [(t, p) for t, p in sample if p.name not in done_names]
    print(f"sample {len(sample)} files, already {len(sample) - len(todo)}, todo {len(todo)}", flush=True)
    if not OUT.exists():
        with OUT.open("w", newline="") as handle:
            csv.writer(handle).writerow(
                ["site", "filename", "year", "segment_index", "start_sec", "band_energy", "overall_energy", "ratio"]
            )
    began = time.time()
    completed = len(sample) - len(todo)
    write_progress(completed, len(sample), "", began)
    for t, path in todo:
        write_progress(completed, len(sample), str(path), began)
        tmp = HERE / f"_tmp_{path.name}.csv"
        cmd = [str(EF), "--cpu", "--segment-sec", "600", "--out", str(tmp), str(path)]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"FAIL {path}: {(proc.stderr or '')[-300:]}", flush=True)
            tmp.unlink(missing_ok=True)
            completed += 1  # count as attempted so we do not stall forever
            with (HERE / "failures.log").open("a") as handle:
                handle.write(f"{path}\n{proc.stderr}\n")
            continue
        with tmp.open(newline="") as src, OUT.open("a", newline="") as dst:
            reader = csv.DictReader(src)
            writer = csv.DictWriter(
                dst,
                fieldnames=["site", "filename", "year", "segment_index", "start_sec", "band_energy", "overall_energy", "ratio"],
            )
            for row in reader:
                writer.writerow(row)
        tmp.unlink(missing_ok=True)
        completed += 1
        print(f"OK {path.name} ({t.isoformat()})", flush=True)
        write_progress(completed, len(sample), "", began)
    write_progress(len(sample), len(sample), "", began)
    print("done", flush=True)


if __name__ == "__main__":
    main()
