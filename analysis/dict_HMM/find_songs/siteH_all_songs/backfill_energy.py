#!/usr/bin/env python3
"""Compute energy segments for eligible wavs missing from energy_segments.csv."""

from __future__ import annotations

import csv
import subprocess
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ENERGY = HERE.parents[3] / "pre_processing" / "c" / "energy_filter"
CSV = HERE / "energy_segments.csv"
LIST = HERE / "missing_energy_files.txt"
PROGRESS = HERE / "ENERGY_BACKFILL_PROGRESS.md"
DONE = HERE / "energy_backfill_done.txt"
FAIL = HERE / "energy_backfill_failures.log"
WORK = HERE / "work_energy"


def log(msg: str) -> None:
    print(f"{datetime.now().isoformat(timespec='seconds')} {msg}", flush=True)


def write_progress(done: int, total: int, current: str, began: float, last: str) -> None:
    elapsed = time.time() - began
    rate = done / elapsed if elapsed > 0 and done else 0.0
    rem = total - done
    eta = rem / rate if rate > 0 else float("nan")
    pct = 100.0 * done / total if total else 0.0
    bar = "█" * int(pct // 2) + "·" * (50 - int(pct // 2))
    PROGRESS.write_text(
        f"""# Energy backfill — progress

**Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Status:** {'done' if done >= total else 'running'}

```
[{bar}] {pct:5.1f}%
```

| | |
|---|---:|
| Missing files | {total} |
| Completed | {done} |
| Elapsed | {elapsed / 3600:.2f} h |
| ETA | {eta / 3600:.1f} h |
| Mean s/file | {(elapsed / done if done else float('nan')):.1f} |

**Current**  
`{current or '—'}`

**Last**  
`{last or '—'}`

Appending to `{CSV.name}`.
"""
    )


def main() -> int:
    if not ENERGY.exists():
        raise SystemExit(f"missing {ENERGY}")
    WORK.mkdir(exist_ok=True)
    paths = [Path(line.strip()) for line in LIST.read_text().splitlines() if line.strip()]
    already = set()
    if DONE.exists():
        already = {line.strip() for line in DONE.read_text().splitlines() if line.strip()}
    # also skip if filename already in CSV (race-safe)
    have = set()
    with CSV.open(newline="") as handle:
        for row in csv.DictReader(handle):
            have.add(row["filename"])
    todo = [p for p in paths if p.name not in already and p.name not in have]
    log(f"backfill {len(todo)} files (list {len(paths)}, already {len(paths) - len(todo)})")
    began = time.time()
    completed = len(paths) - len(todo)
    last = ""
    write_progress(completed, len(paths), "", began, last)
    for path in todo:
        write_progress(completed, len(paths), str(path), began, last)
        tmp = WORK / f"e_{path.name}.csv"
        proc = subprocess.run(
            [str(ENERGY), "--cpu", "--segment-sec", "600", "--out", str(tmp), str(path)],
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip().splitlines()
            tail = err[-1] if err else f"exit {proc.returncode}"
            log(f"FAIL {path.name}: {tail}")
            with FAIL.open("a") as handle:
                handle.write(f"{path}\t{tail}\n")
            tmp.unlink(missing_ok=True)
            # do not mark done — allow retry
            completed += 1  # count toward progress so ETA moves; leave off DONE
            last = f"FAIL {path.name}"
            write_progress(completed, len(paths), "", began, last)
            continue
        with tmp.open(newline="") as src, CSV.open("a", newline="") as dst:
            reader = csv.DictReader(src)
            writer = csv.DictWriter(
                dst,
                fieldnames=["site", "filename", "year", "segment_index", "start_sec", "band_energy", "overall_energy", "ratio"],
            )
            for row in reader:
                writer.writerow(row)
        tmp.unlink(missing_ok=True)
        with DONE.open("a") as handle:
            handle.write(path.name + "\n")
        completed += 1
        last = path.name
        log(f"OK {path.name}")
        write_progress(completed, len(paths), "", began, last)
    write_progress(len(paths), len(paths), "", began, last)
    log("backfill finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
