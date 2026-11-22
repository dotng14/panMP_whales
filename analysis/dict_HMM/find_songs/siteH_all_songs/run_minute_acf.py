#!/usr/bin/env python3
"""Per-minute ACF peak of the whitened 50–500 Hz energy ratio, newest recording first.

Runs pre_processing/c/minute_acf on each eligible 10 kHz Site H wav. Resumes from
done.txt, failures.log, and CSVs already in the results directory.

Restarting this script is safe: completed files are skipped.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
BIN = HERE.parents[3] / "pre_processing" / "c" / "minute_acf"
MANIFEST = HERE / "manifest.csv"
OUT = Path("/home/yfreund/whale_processing/12TB_results/minute_acf")
PROGRESS = OUT / "PROGRESS.md"
DONE = OUT / "done.txt"
FAILED = OUT / "failures.log"
LOG = OUT / "minute_acf.log"
PID = OUT / "runner.pid"

STAMP = re.compile(r"(?<!\d)(\d{6})_(\d{6})(?!\d)")


@dataclass(frozen=True)
class Rec:
    path: Path
    stamp: datetime

    @property
    def csv_path(self) -> Path:
        return OUT / f"{self.path.name}.minute_acf.csv"


def log(msg: str) -> None:
    print(f"{datetime.now().isoformat(timespec='seconds')} {msg}", flush=True)


def parse_stamp(name: str) -> datetime | None:
    found = list(STAMP.finditer(name))
    if not found:
        return None
    date, clock = found[-1].group(1), found[-1].group(2)
    year = 2000 + int(date[:2])
    month, day = int(date[2:4]), int(date[4:6])
    hour, minute, second = int(clock[:2]), int(clock[2:4]), int(clock[4:6])
    try:
        return datetime(year, month, day, hour, minute, second)
    except ValueError:
        return None


def load_recordings() -> list[Rec]:
    found: list[Rec] = []
    with MANIFEST.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["status"] == "fail":
                continue
            path = Path(row["wav_path"])
            stamp = parse_stamp(path.name)
            if stamp is None:
                continue
            found.append(Rec(path, stamp))
    found.sort(key=lambda rec: (rec.stamp, rec.path.as_posix()), reverse=True)
    return found


def read_lines(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")}


def csv_ok(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 80:
        return False
    with path.open() as handle:
        header = handle.readline()
        second = handle.readline()
    return header.startswith("filename,year,") and "acf_peak" in header and bool(second.strip())


def append_line(path: Path, text: str) -> None:
    with path.open("a") as handle:
        handle.write(text + "\n")
        handle.flush()


def acquire_lock() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if PID.exists():
        try:
            old = int(PID.read_text().strip())
        except ValueError:
            old = None
        if old and Path(f"/proc/{old}").exists():
            raise SystemExit(f"minute_acf is already running as pid {old}")
    PID.write_text(str(os.getpid()))


def release_lock() -> None:
    if PID.exists() and PID.read_text().strip() == str(os.getpid()):
        PID.unlink()


def write_progress(
    recordings: list[Rec],
    done: set[str],
    failed: set[str],
    *,
    status: str,
    current: str,
    last: str,
    began: float,
    session_done: int,
) -> None:
    total = len(recordings)
    finished = sum(1 for rec in recordings if str(rec.path) in done or str(rec.path) in failed)
    elapsed = max(time.time() - began, 0.0)
    rate = session_done / elapsed if elapsed > 0 and session_done else 0.0
    remaining = total - finished
    eta = remaining / rate if rate else float("nan")
    pct = 100.0 * finished / total if total else 0.0
    filled = min(50, int(pct // 2))
    bar = "█" * filled + "·" * (50 - filled)
    current_month = ""
    if current:
        stamp = parse_stamp(Path(current).name)
        if stamp:
            current_month = f"{stamp.year:04d}-{stamp.month:02d}"
    text = f"""# Minute ACF — progress

**Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Status:** {status}

```
[{bar}] {pct:5.1f}%
```

| | |
|---|---:|
| Recordings | {total} |
| Completed | {sum(1 for rec in recordings if str(rec.path) in done)} |
| Failed | {sum(1 for rec in recordings if str(rec.path) in failed)} |
| Remaining | {remaining} |
| Elapsed this session | {elapsed / 3600:.2f} h |
| ETA | {eta / 3600:.1f} h |
| Mean s/file this session | {(elapsed / session_done if session_done else float('nan')):.1f} |

Newest recording first. Each CSV has per-minute ACF peak height and lag (1–20 s)
of the 0.2 s whitened 50–500 Hz energy ratio.

**Current month**
`{current_month or '—'}`

**Current file**
`{current or '—'}`

**Last completed**
`{last or '—'}`

Per-minute CSVs are in `{OUT}`.
"""
    tmp = PROGRESS.with_suffix(".md.tmp")
    tmp.write_text(text)
    tmp.replace(PROGRESS)


def run_file(rec: Rec, *, attempts: int = 3) -> tuple[bool, str]:
    if not rec.path.exists():
        return False, "file is missing"
    last_err = "unknown"
    for attempt in range(1, attempts + 1):
        with LOG.open("a") as handle:
            handle.write(f"\n# {datetime.now().isoformat(timespec='seconds')} {rec.path}")
            if attempt > 1:
                handle.write(f" (retry {attempt}/{attempts})")
            handle.write("\n")
            handle.flush()
            proc = subprocess.run(
                [str(BIN), "--out-dir", str(OUT), str(rec.path)],
                stdout=handle,
                stderr=subprocess.STDOUT,
            )
        if proc.returncode == 0 and csv_ok(rec.csv_path):
            return True, ""
        if proc.returncode != 0:
            last_err = f"exit {proc.returncode}"
        else:
            last_err = "csv was not written"
        if attempt < attempts:
            time.sleep(min(30 * attempt, 90))
    return False, last_err


def main() -> int:
    if not BIN.exists():
        raise SystemExit(f"missing {BIN}; build with: make -C {BIN.parent} minute_acf")
    acquire_lock()
    try:
        recordings = load_recordings()
        done = read_lines(DONE)
        failed_rows = []
        if FAILED.exists():
            failed_rows = [
                line.split("\t", 1)[0].strip()
                for line in FAILED.read_text().splitlines()
                if line.strip()
            ]
        failed = set(failed_rows)
        seeded = 0
        for rec in recordings:
            key = str(rec.path)
            if key in done or key in failed:
                continue
            if csv_ok(rec.csv_path):
                append_line(DONE, key)
                done.add(key)
                seeded += 1
        log(f"{len(recordings)} recordings, {len(done)} already done, {len(failed)} failed, seeded {seeded}")
        began = time.time()
        todo = [rec for rec in recordings if str(rec.path) not in done and str(rec.path) not in failed]
        write_progress(
            recordings,
            done,
            failed,
            status="running" if todo else "done",
            current="",
            last="",
            began=began,
            session_done=0,
        )
        session_done = 0
        last = ""
        for rec in todo:
            write_progress(
                recordings,
                done,
                failed,
                status="running",
                current=str(rec.path),
                last=last,
                began=began,
                session_done=session_done,
            )
            ok, err = run_file(rec)
            key = str(rec.path)
            if ok:
                append_line(DONE, key)
                done.add(key)
                session_done += 1
                last = rec.path.name
                log(f"OK {rec.path.name}")
            else:
                append_line(FAILED, f"{key}\t{err}")
                failed.add(key)
                last = f"FAIL {rec.path.name}"
                log(f"FAIL {rec.path.name}: {err}")
            write_progress(
                recordings,
                done,
                failed,
                status="running",
                current="",
                last=last,
                began=began,
                session_done=session_done,
            )
        write_progress(
            recordings,
            done,
            failed,
            status="done",
            current="",
            last=last,
            began=began,
            session_done=session_done,
        )
        log("finished")
        return 0
    finally:
        release_lock()


if __name__ == "__main__":
    sys.exit(main())
