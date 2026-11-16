#!/usr/bin/env python3
"""Run song_detector on every compatible wav under /mnt/sda.

Only 10 kHz 16-bit mono PCM files with a _YYMMDD_HHMMSS stamp can be processed
(the C detector requires that). Progress is written to PROGRESS.md for viewing
in an editor. Conforming songs are appended to conforming_songs.csv.
"""

from __future__ import annotations

import csv
import os
import re
import shutil
import struct
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent  # find_songs
DETECTOR = ROOT / "c" / "song_detector"
MODEL = ROOT / "c" / "build" / "song_model.bin"
WAV_ROOT = Path("/mnt/sda")
WORK = HERE / "work"
PLACEMENTS = HERE / "placements"
PROGRESS = HERE / "PROGRESS.md"
SONGS_CSV = HERE / "conforming_songs.csv"
MANIFEST = HERE / "manifest.csv"
LOG = HERE / "run.log"
DONE = HERE / "done_stems.txt"

SOURCE_RATE = 10000
STAMP = re.compile(r"_\d{6}_\d{6}")

SONG_FIELDS = [
    "wav_path",
    "file",
    "sample_rate",
    "start_utc",
    "end_utc",
    "start_sec",
    "end_sec",
    "duration_sec",
    "theme0_start_sec",
    "theme1_start_sec",
    "theme2_start_sec",
    "theme3_start_sec",
    "margin0",
    "margin1",
    "margin2",
    "margin3",
    "min_margin",
    "order_z",
    "shuffles_at_or_above",
    "placements",
    "theme_minutes",
]


def log(msg: str) -> None:
    # Stdout is redirected to run.log by the launcher; do not also append here.
    print(f"{datetime.now().isoformat(timespec='seconds')} {msg}", flush=True)


def wav_fmt(path: Path):
    """Return (rate, channels, audio, bits) or raise."""
    with path.open("rb") as handle:
        head = handle.read(12)
        if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
            raise ValueError("not wav")
        while True:
            chunk = handle.read(8)
            if len(chunk) < 8:
                raise ValueError("no fmt")
            name, size = struct.unpack("<4sI", chunk)
            if name == b"fmt ":
                body = handle.read(size)
                audio, channels, rate = struct.unpack("<HHI", body[:8])
                bits = struct.unpack("<H", body[14:16])[0]
                return rate, channels, audio, bits
            handle.seek(size + (size & 1), 1)


def eligible(path: Path) -> tuple[bool, str]:
    if not STAMP.search(path.name):
        return False, "no_timestamp_stamp"
    try:
        rate, channels, audio, bits = wav_fmt(path)
    except Exception as exc:
        return False, f"header:{exc}"
    if rate != SOURCE_RATE:
        return False, f"rate_{rate}"
    if audio != 1 or channels != 1 or bits != 16:
        return False, f"fmt_{audio}_{channels}_{bits}"
    return True, "ok"


def write_progress(
    *,
    status: str,
    total: int,
    done: int,
    skipped: int,
    failed: int,
    songs: int,
    current: str,
    last_ok: str,
    last_err: str,
    began: float,
    file_sec: float | None = None,
) -> None:
    elapsed = time.time() - began
    remaining = max(total - done, 0)
    rate = done / elapsed if elapsed > 0 and done > 0 else 0.0
    eta = remaining / rate if rate > 0 else float("nan")
    eta_text = (
        f"{eta / 3600:.1f} h"
        if eta == eta
        else "…"
    )
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pct = 100.0 * done / total if total else 0.0
    bar_n = int(pct // 2)
    bar = "█" * bar_n + "·" * (50 - bar_n)
    text = f"""# Site H song detection — progress

**Updated:** {now}  
**Status:** {status}

```
[{bar}] {pct:5.1f}%
```

| | |
|---|---:|
| Eligible 10 kHz wavs | {total} |
| Completed | {done} |
| Failed | {failed} |
| Skipped (ineligible) | {skipped} |
| Conforming songs so far | {songs} |
| Elapsed | {elapsed / 3600:.2f} h |
| ETA | {eta_text} |
| Last file seconds | {f'{file_sec:.1f}' if file_sec is not None else '—'} |

**Current file**  
`{current or '—'}`

**Last completed**  
`{last_ok or '—'}`

**Last error**  
`{last_err or '—'}`

## Outputs

- Songs CSV: `{SONGS_CSV.name}`
- Manifest: `{MANIFEST.name}`
- Placements (files with songs): `placements/`
- Log: `{LOG.name}`

Open this file in an editor and reload to watch progress. The run is safe to
interrupt; restarting `run_all.py` resumes from completed stems.
"""
    PROGRESS.write_text(text)


def ensure_songs_header() -> None:
    if not SONGS_CSV.exists():
        with SONGS_CSV.open("w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=SONG_FIELDS).writeheader()


def append_songs(wav: Path, decoded: Path) -> int:
    if not decoded.exists():
        return 0
    added = 0
    with decoded.open(newline="") as handle, SONGS_CSV.open("a", newline="") as out:
        reader = csv.DictReader(handle)
        writer = csv.DictWriter(out, fieldnames=SONG_FIELDS)
        for row in reader:
            try:
                min_margin = float(row["min_margin"])
                shuffles = int(float(row["shuffles_at_or_above"]))
            except (TypeError, ValueError):
                continue
            if not (min_margin > 0 and shuffles == 0):
                continue
            starts = [float(x) for x in row["theme_starts_sec"].split("/")]
            margins = [float(x) for x in row["margins"].split("/")]
            writer.writerow(
                {
                    "wav_path": str(wav),
                    "file": row["file"],
                    "sample_rate": SOURCE_RATE,
                    "start_utc": row["start_utc"],
                    "end_utc": row["end_utc"],
                    "start_sec": row["start_sec"],
                    "end_sec": row["end_sec"],
                    "duration_sec": float(row["end_sec"]) - float(row["start_sec"]),
                    "theme0_start_sec": starts[0],
                    "theme1_start_sec": starts[1],
                    "theme2_start_sec": starts[2],
                    "theme3_start_sec": starts[3],
                    "margin0": margins[0],
                    "margin1": margins[1],
                    "margin2": margins[2],
                    "margin3": margins[3],
                    "min_margin": min_margin,
                    "order_z": row["order_z"],
                    "shuffles_at_or_above": shuffles,
                    "placements": row["placements"],
                    "theme_minutes": row["theme_minutes"],
                }
            )
            added += 1
    return added


def load_done() -> set[str]:
    if not DONE.exists():
        return set()
    return {line.strip() for line in DONE.read_text().splitlines() if line.strip()}


def mark_done(stem: str) -> None:
    with DONE.open("a") as handle:
        handle.write(stem + "\n")


def process_one(wav: Path) -> tuple[str, int, float]:
    """Returns (status, n_songs, seconds). status is ok|fail."""
    stem = wav.name.split(".")[0]
    out = WORK / f"tmp_{os.getpid()}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    began = time.time()
    try:
        proc = subprocess.run(
            [str(DETECTOR), str(MODEL), str(out), str(wav)],
            capture_output=True,
            text=True,
            check=False,
        )
        sec = time.time() - began
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "").strip().splitlines()
            tail = err[-1] if err else f"exit {proc.returncode}"
            log(f"FAIL {wav}: {tail}")
            with (HERE / "failures.log").open("a") as handle:
                handle.write(f"{wav}\t{tail}\n")
                if proc.stderr:
                    handle.write(proc.stderr[-2000:] + "\n")
            return "fail", 0, sec
        decoded = out / "decoded_songs.csv"
        n = append_songs(wav, decoded)
        place = out / f"placements_{stem}.csv"
        if n > 0 and place.exists():
            dest = PLACEMENTS / f"placements_{stem}.csv"
            shutil.move(str(place), dest)
        return "ok", n, sec
    finally:
        if out.exists():
            shutil.rmtree(out, ignore_errors=True)


def main() -> int:
    if not DETECTOR.exists() or not MODEL.exists():
        print("missing song_detector or song_model.bin", file=sys.stderr)
        return 2
    HERE.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(exist_ok=True)
    PLACEMENTS.mkdir(exist_ok=True)
    ensure_songs_header()
    if not MANIFEST.exists():
        with MANIFEST.open("w", newline="") as handle:
            csv.writer(handle).writerow(["wav_path", "stem", "status", "detail", "seconds", "songs"])

    log("listing wavs under /mnt/sda …")
    wavs = sorted(WAV_ROOT.rglob("*.wav"))
    log(f"found {len(wavs)} wav paths")

    eligible_paths: list[Path] = []
    skipped = 0
    with MANIFEST.open("a", newline="") as handle:
        writer = csv.writer(handle)
        # Only write skip rows once per fresh listing when stem not done
        # Skips are recorded in a side file to avoid rewriting every resume.
        skip_file = HERE / "skipped.csv"
        if not skip_file.exists():
            with skip_file.open("w", newline="") as skip_handle:
                skip_writer = csv.writer(skip_handle)
                skip_writer.writerow(["wav_path", "reason"])
                for wav in wavs:
                    ok, reason = eligible(wav)
                    if ok:
                        eligible_paths.append(wav)
                    else:
                        skipped += 1
                        skip_writer.writerow([str(wav), reason])
            log(f"eligible {len(eligible_paths)}, skipped {skipped}")
        else:
            for wav in wavs:
                ok, _reason = eligible(wav)
                if ok:
                    eligible_paths.append(wav)
            skipped = sum(1 for _ in skip_file.open()) - 1
            log(f"eligible {len(eligible_paths)}, skipped (cached) {skipped}")

    done_stems = load_done()
    todo = [w for w in eligible_paths if w.name.split(".")[0] not in done_stems]
    already = len(eligible_paths) - len(todo)
    log(f"resume: {already} done, {len(todo)} remaining")

    # Count existing songs
    songs = 0
    if SONGS_CSV.exists():
        with SONGS_CSV.open() as handle:
            songs = max(sum(1 for _ in handle) - 1, 0)

    began = time.time()
    failed = 0
    completed = already
    last_ok = ""
    last_err = ""
    write_progress(
        status="running",
        total=len(eligible_paths),
        done=completed,
        skipped=skipped,
        failed=failed,
        songs=songs,
        current="",
        last_ok="",
        last_err="",
        began=began,
    )

    with MANIFEST.open("a", newline="") as handle:
        writer = csv.writer(handle)
        for wav in todo:
            stem = wav.name.split(".")[0]
            write_progress(
                status="running",
                total=len(eligible_paths),
                done=completed,
                skipped=skipped,
                failed=failed,
                songs=songs,
                current=str(wav),
                last_ok=last_ok,
                last_err=last_err,
                began=began,
            )
            try:
                status, n, sec = process_one(wav)
            except Exception as exc:
                status, n, sec = "fail", 0, 0.0
                last_err = f"{wav}: {exc}"
                log(last_err)
                traceback.print_exc()
            if status == "ok":
                mark_done(stem)
                completed += 1
                songs += n
                last_ok = f"{wav} ({sec:.1f}s, +{n} songs)"
                writer.writerow([str(wav), stem, "ok", "", f"{sec:.2f}", n])
                handle.flush()
                log(f"OK {stem}: {sec:.1f}s, songs +{n} (total {songs})")
            else:
                failed += 1
                last_err = str(wav)
                writer.writerow([str(wav), stem, "fail", "see failures.log", f"{sec:.2f}", 0])
                handle.flush()
                # Do not mark done — allow retry
            write_progress(
                status="running",
                total=len(eligible_paths),
                done=completed,
                skipped=skipped,
                failed=failed,
                songs=songs,
                current="",
                last_ok=last_ok,
                last_err=last_err,
                began=began,
                file_sec=sec,
            )

    write_progress(
        status="done",
        total=len(eligible_paths),
        done=completed,
        skipped=skipped,
        failed=failed,
        songs=songs,
        current="",
        last_ok=last_ok,
        last_err=last_err,
        began=began,
    )
    log(f"finished: completed {completed}, failed {failed}, songs {songs}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
