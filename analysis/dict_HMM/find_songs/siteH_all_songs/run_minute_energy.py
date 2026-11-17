#!/usr/bin/env python3
"""Per-minute energy ratios for Site H, newest recording first.

Runs pre_processing/c/minute_energy on each eligible 10 kHz wav. A month is
finished when every recording that starts in that month has succeeded or
failed. The month then gets an hourly figure in monthly_hourly.md. The
following month is redrawn at the same time, so minutes that cross the
boundary are included.

Restarting this script resumes from done.txt, failures.log, and CSVs already
in the results directory.
"""

from __future__ import annotations

import csv
import os
import re
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
BIN = HERE.parents[3] / "pre_processing" / "c" / "minute_energy"
MANIFEST = HERE / "manifest.csv"
OUT = Path("/home/yfreund/whale_processing/12TB_results")
PROGRESS = OUT / "PROGRESS.md"
DONE = OUT / "done.txt"
FAILED = OUT / "failures.log"
LOG = OUT / "minute_energy.log"
PID = OUT / "minute_energy.pid"
MONTHLY = OUT / "monthly_hourly.md"
FIGDIR = OUT / "hourly"

STAMP = re.compile(r"(?<!\d)(\d{6})_(\d{6})(?!\d)")
LOOKBACK = timedelta(hours=48)
HEADER = """# Hourly summaries

A section is added when every recording that starts in a month has been
processed, newest month first. Each point is the mean of the per-minute
values in that hour. Hours with no audio are left blank. After the previous
month finishes, the figure is redrawn so minutes from a recording that
crosses the first of the month are included.

"""


@dataclass(frozen=True)
class Rec:
    path: Path
    stamp: datetime

    @property
    def csv_path(self) -> Path:
        return OUT / f"{self.path.name}.minute_energy.csv"

    @property
    def month(self) -> tuple[int, int]:
        return self.stamp.year, self.stamp.month


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


def month_start(month: tuple[int, int]) -> datetime:
    return datetime(month[0], month[1], 1)


def next_month(month: tuple[int, int]) -> tuple[int, int]:
    year, mon = month
    return (year + 1, 1) if mon == 12 else (year, mon + 1)


def month_end(month: tuple[int, int]) -> datetime:
    return month_start(next_month(month))


def month_label(month: tuple[int, int]) -> str:
    return f"{month[0]:04d}-{month[1]:02d}"


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
    return header.startswith("filename,year,") and bool(second.strip())


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
            raise SystemExit(f"minute energy is already running as pid {old}")
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
    months_done: int,
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
            current_month = month_label((stamp.year, stamp.month))
    text = f"""# Minute energy — progress

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
| Months summarized | {months_done} |
| Elapsed this session | {elapsed / 3600:.2f} h |
| ETA | {eta / 3600:.1f} h |
| Mean s/file this session | {(elapsed / session_done if session_done else float('nan')):.1f} |

Newest recording first. A month is summarized, at one hour resolution, when every recording that starts in that month has been attempted.

**Current month**
`{current_month or '—'}`

**Current file**
`{current or '—'}`

**Last completed**
`{last or '—'}`

Per-minute CSVs and the hourly report are in `{OUT}`.
"""
    tmp = PROGRESS.with_suffix(".md.tmp")
    tmp.write_text(text)
    tmp.replace(PROGRESS)


def run_file(rec: Rec) -> tuple[bool, str]:
    if not rec.path.exists():
        return False, "file is missing"
    with LOG.open("a") as handle:
        handle.write(f"\n# {datetime.now().isoformat(timespec='seconds')} {rec.path}\n")
        handle.flush()
        proc = subprocess.run(
            [str(BIN), "--out-dir", str(OUT), str(rec.path)],
            stdout=handle,
            stderr=subprocess.STDOUT,
        )
    if proc.returncode != 0:
        return False, f"exit {proc.returncode}"
    if not csv_ok(rec.csv_path):
        return False, "csv was not written"
    return True, ""


def minute_rows(rec: Rec) -> list[tuple[datetime, float, float]]:
    rows: list[tuple[datetime, float, float]] = []
    with rec.csv_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            when = rec.stamp + timedelta(seconds=float(row["start_sec"]))
            rows.append((when, float(row["average_ratio"]), float(row["fft_fraction_0p1_1hz"])))
    return rows


def hourly_means(month: tuple[int, int], recordings: list[Rec], done: set[str]) -> dict:
    start = month_start(month)
    end = month_end(month)
    buckets: dict[datetime, list[tuple[float, float]]] = defaultdict(list)
    minute_avg: list[float] = []
    minute_frac: list[float] = []
    used = 0
    for rec in recordings:
        if str(rec.path) not in done:
            continue
        if not (start - LOOKBACK <= rec.stamp < end):
            continue
        if not csv_ok(rec.csv_path):
            continue
        hit = False
        for when, avg, frac in minute_rows(rec):
            if when < start or when >= end:
                continue
            hour = when.replace(minute=0, second=0, microsecond=0)
            buckets[hour].append((avg, frac))
            minute_avg.append(avg)
            minute_frac.append(frac)
            hit = True
        if hit:
            used += 1
    hours = []
    cursor = start
    while cursor < end:
        hours.append(cursor)
        cursor += timedelta(hours=1)
    avg_y = []
    frac_y = []
    present = 0
    for hour in hours:
        values = buckets.get(hour)
        if not values:
            avg_y.append(float("nan"))
            frac_y.append(float("nan"))
            continue
        present += 1
        avg_y.append(sum(v[0] for v in values) / len(values))
        frac_y.append(sum(v[1] for v in values) / len(values))
    return {
        "hours": hours,
        "avg": avg_y,
        "frac": frac_y,
        "present": present,
        "files": used,
        "minutes": len(minute_avg),
        "mean_avg": sum(minute_avg) / len(minute_avg) if minute_avg else float("nan"),
        "mean_frac": sum(minute_frac) / len(minute_frac) if minute_frac else float("nan"),
    }


def write_figure(month: tuple[int, int], summary: dict) -> Path:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    label = month_label(month)
    dest = FIGDIR / f"{label}.png"
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True, constrained_layout=True)
    axes[0].plot(summary["hours"], summary["avg"], color="#1f4e79", lw=0.9)
    axes[0].set_ylabel("Average energy ratio")
    axes[0].set_ylim(0, 1)
    axes[0].set_title(f"{label}  ·  hourly means")
    axes[1].plot(summary["hours"], summary["frac"], color="#b85c38", lw=0.9)
    axes[1].set_ylabel("FFT fraction 0.1–1 Hz")
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("Recorder clock")
    axes[1].set_xlim(summary["hours"][0], summary["hours"][-1])
    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax.grid(True, axis="y", lw=0.4, color="#cccccc")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.savefig(dest, dpi=140)
    plt.close(fig)
    return dest


def upsert_month(month: tuple[int, int], body: str) -> None:
    label = month_label(month)
    tag = f"<!-- month:{label} -->"
    end = f"<!-- /month:{label} -->"
    block = f"{tag}\n{body.rstrip()}\n{end}\n"
    text = MONTHLY.read_text() if MONTHLY.exists() else HEADER
    if not text.startswith("# Hourly summaries"):
        text = HEADER + text
    if tag in text:
        pre, rest = text.split(tag, 1)
        _, post = rest.split(end, 1)
        text = pre + block + post.lstrip("\n")
    else:
        if not text.endswith("\n"):
            text += "\n"
        text += "\n" + block
    tmp = MONTHLY.with_suffix(".md.tmp")
    tmp.write_text(text)
    tmp.replace(MONTHLY)


def render_month(month: tuple[int, int], recordings: list[Rec], done: set[str]) -> None:
    summary = hourly_means(month, recordings, done)
    label = month_label(month)
    if summary["minutes"] == 0:
        body = f"## {label}\n\nNo per-minute values fall in this month.\n"
        upsert_month(month, body)
        log(f"month {label}: no minutes")
        return
    write_figure(month, summary)
    body = f"""## {label}

![{label} hourly means](hourly/{label}.png)

{summary['files']} recordings contribute. {summary['present']} hours contain audio ({summary['minutes']} minutes). Across those minutes the mean energy ratio is {summary['mean_avg']:.4f} and the mean 0.1–1 Hz FFT fraction is {summary['mean_frac']:.4f}.
"""
    upsert_month(month, body)
    log(
        f"month {label}: {summary['files']} files, {summary['present']} hours, "
        f"ratio {summary['mean_avg']:.4f}, fft {summary['mean_frac']:.4f}"
    )


def month_closed(month: tuple[int, int], recordings: list[Rec], done: set[str], failed: set[str]) -> bool:
    members = [rec for rec in recordings if rec.month == month]
    if not members:
        return False
    return all(str(rec.path) in done or str(rec.path) in failed for rec in members)


def rendered_months() -> set[str]:
    if not MONTHLY.exists():
        return set()
    return set(re.findall(r"<!-- month:(\d{4}-\d{2}) -->", MONTHLY.read_text()))


def close_month(month: tuple[int, int], recordings: list[Rec], done: set[str], failed: set[str]) -> int:
    """Render this month if it just finished, and redraw the next month."""
    if not month_closed(month, recordings, done, failed):
        return len(rendered_months())
    label = month_label(month)
    have = rendered_months()
    if label not in have:
        render_month(month, recordings, done)
    nxt = next_month(month)
    if month_label(nxt) in rendered_months():
        render_month(nxt, recordings, done)
    return len(rendered_months())


def catch_up(recordings: list[Rec], done: set[str], failed: set[str]) -> int:
    months = []
    seen = set()
    for rec in recordings:
        if rec.month not in seen:
            seen.add(rec.month)
            months.append(rec.month)
    for month in months:
        if month_label(month) in rendered_months():
            continue
        if month_closed(month, recordings, done, failed):
            render_month(month, recordings, done)
            nxt = next_month(month)
            if month_label(nxt) in rendered_months():
                render_month(nxt, recordings, done)
    return len(rendered_months())


def main() -> int:
    if not BIN.exists():
        raise SystemExit(f"missing {BIN}")
    acquire_lock()
    try:
        recordings = load_recordings()
        done = read_lines(DONE)
        failed_rows = []
        if FAILED.exists():
            failed_rows = [line.split("\t", 1)[0].strip() for line in FAILED.read_text().splitlines() if line.strip()]
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
        months_done = catch_up(recordings, done, failed)
        todo = [rec for rec in recordings if str(rec.path) not in done and str(rec.path) not in failed]
        write_progress(
            recordings, done, failed,
            status="running" if todo else "done",
            current="",
            last="",
            began=began,
            session_done=0,
            months_done=months_done,
        )
        session_done = 0
        last = ""
        for rec in todo:
            write_progress(
                recordings, done, failed,
                status="running",
                current=str(rec.path),
                last=last,
                began=began,
                session_done=session_done,
                months_done=months_done,
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
            months_done = close_month(rec.month, recordings, done, failed)
            write_progress(
                recordings, done, failed,
                status="running",
                current="",
                last=last,
                began=began,
                session_done=session_done,
                months_done=months_done,
            )
        write_progress(
            recordings, done, failed,
            status="done",
            current="",
            last=last,
            began=began,
            session_done=session_done,
            months_done=months_done,
        )
        log("finished")
        return 0
    finally:
        release_lock()


if __name__ == "__main__":
    sys.exit(main())
