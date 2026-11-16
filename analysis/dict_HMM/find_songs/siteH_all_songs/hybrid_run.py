#!/usr/bin/env python3
"""Hybrid Site H pass: energy ratio first, song detector only on high-ratio files.

Skips stems already listed in done_stems.txt (from the earlier song-only run).

Per remaining 10 kHz wav:
  1. energy_filter --cpu --segment-sec 600  → append energy_segments.csv
  2. If any 600 s segment has ratio ≥ ENERGY_THR (≈ ≥10 min high band fraction),
     run song_detector on the whole file and append conforming songs.
  3. Otherwise skip song detection and mark the stem done.

Progress: PROGRESS.md (reload in an editor).
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
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DETECTOR = ROOT / "c" / "song_detector"
MODEL = ROOT / "c" / "build" / "song_model.bin"
ENERGY = HERE.parents[3] / "pre_processing" / "c" / "energy_filter"  # repo root / pre_processing/...
WAV_ROOT = Path("/mnt/sda")
WORK = HERE / "work"
PLACEMENTS = HERE / "placements"
PROGRESS = HERE / "PROGRESS.md"
SONGS_CSV = HERE / "conforming_songs.csv"
ENERGY_CSV = HERE / "energy_segments.csv"
MANIFEST = HERE / "manifest.csv"
DONE = HERE / "done_stems.txt"
SKIP_FILE = HERE / "skipped.csv"

SOURCE_RATE = 10000
ENERGY_THR = 0.15
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
    print(f"{datetime.now().isoformat(timespec='seconds')} {msg}", flush=True)


def wav_fmt(path: Path):
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
    parts = {p.lower() for p in path.parts}
    if "misc" in parts or "error" in parts or "recycle.bin" in parts or parts & {".fseventsd", ".spotlight-v100"}:
        return False, "excluded_dir"
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
    already: int,
    skipped: int,
    failed: int,
    songs: int,
    song_runs: int,
    energy_only: int,
    current: str,
    last_ok: str,
    last_err: str,
    began: float,
    file_sec: float | None = None,
) -> None:
    elapsed = time.time() - began
    remaining = max(total - done, 0)
    # Rate from this hybrid session only — not counting stems skipped on resume.
    hybrid_done = max(done - already, 0)
    rate = hybrid_done / elapsed if elapsed > 0 and hybrid_done > 0 else 0.0
    eta = remaining / rate if rate > 0 else float("nan")
    eta_text = f"{eta / 3600:.1f} h" if eta == eta else "…"
    per_file = f"{elapsed / hybrid_done:.1f} s" if hybrid_done > 0 else "—"
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    pct = 100.0 * done / total if total else 0.0
    bar_n = int(pct // 2)
    bar = "█" * bar_n + "·" * (50 - bar_n)
    PROGRESS.write_text(
        f"""# Site H hybrid detection — progress

**Updated:** {now}  
**Status:** {status}  
**Mode:** energy (CPU) → song only if any 600 s ratio ≥ {ENERGY_THR}

```
[{bar}] {pct:5.1f}%
```

| | |
|---|---:|
| Eligible 10 kHz wavs | {total} |
| Completed | {done} |
| Already done at resume | {already} |
| Done this session | {hybrid_done} |
| Failed | {failed} |
| Ineligible | {skipped} |
| Song detector runs | {song_runs} |
| Energy-only (song skipped) | {energy_only} |
| Conforming songs so far | {songs} |
| Elapsed this session | {elapsed / 3600:.2f} h |
| Mean s / file this session | {per_file} |
| ETA | {eta_text} |
| Last file seconds | {f'{file_sec:.1f}' if file_sec is not None else '—'} |

**Current file**  
`{current or '—'}`

**Last completed**  
`{last_ok or '—'}`

**Last error**  
`{last_err or '—'}`

## Outputs

- `{SONGS_CSV.name}` — conforming songs (extractable spans)
- `{ENERGY_CSV.name}` — 600 s whitened 50–500 Hz ratios (new files only)
- `{MANIFEST.name}` — per-file status (`ok_song` / `ok_energy_only` / `fail`)
- `placements/` — kept when a file yields conforming songs
- `{DONE.name}` — completed stems (song-only history + hybrid)

Reload this file in an editor to watch progress. Restarting `hybrid_run.py`
resumes from `{DONE.name}` and ignores those files.
"""
    )


def ensure_headers() -> None:
    if not SONGS_CSV.exists():
        with SONGS_CSV.open("w", newline="") as handle:
            csv.DictWriter(handle, fieldnames=SONG_FIELDS).writeheader()
    if not ENERGY_CSV.exists():
        with ENERGY_CSV.open("w", newline="") as handle:
            csv.writer(handle).writerow(
                ["site", "filename", "year", "segment_index", "start_sec", "band_energy", "overall_energy", "ratio"]
            )
    if not MANIFEST.exists():
        with MANIFEST.open("w", newline="") as handle:
            csv.writer(handle).writerow(
                ["wav_path", "stem", "status", "detail", "seconds", "songs", "max_ratio", "n_high_segments"]
            )


def load_done() -> set[str]:
    if not DONE.exists():
        return set()
    return {line.strip() for line in DONE.read_text().splitlines() if line.strip()}


def mark_done(stem: str) -> None:
    with DONE.open("a") as handle:
        handle.write(stem + "\n")


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


def run_energy(wav: Path, tmp: Path) -> tuple[list[dict], float]:
    out = tmp / "energy.csv"
    proc = subprocess.run(
        [str(ENERGY), "--cpu", "--segment-sec", "600", "--out", str(out), str(wav)],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or proc.stdout or f"exit {proc.returncode}").strip().splitlines()[-1])
    rows = list(csv.DictReader(out.open()))
    with ENERGY_CSV.open("a", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["site", "filename", "year", "segment_index", "start_sec", "band_energy", "overall_energy", "ratio"],
        )
        for row in rows:
            writer.writerow(row)
    ratios = [float(r["ratio"]) for r in rows]
    return rows, max(ratios) if ratios else 0.0


def needs_song(rows: list[dict]) -> tuple[bool, int]:
    high = sum(1 for r in rows if float(r["ratio"]) >= ENERGY_THR)
    return high > 0, high


def run_song(wav: Path, tmp: Path) -> int:
    stem = wav.name.split(".")[0]
    out = tmp / "song"
    last_err = ""
    for attempt in range(2):
        if out.exists():
            shutil.rmtree(out)
        out.mkdir()
        proc = subprocess.run(
            [str(DETECTOR), str(MODEL), str(out), str(wav)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            n = append_songs(wav, out / "decoded_songs.csv")
            place = out / f"placements_{stem}.csv"
            if n > 0 and place.exists():
                shutil.move(str(place), PLACEMENTS / f"placements_{stem}.csv")
            return n
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        last_err = err[-1] if err else f"exit {proc.returncode}"
        transient = "driver shutting down" in last_err or "CUDA" in last_err
        if not transient or attempt == 1:
            break
        log(f"retry song after GPU glitch: {last_err}")
        time.sleep(3)
    raise RuntimeError(last_err)


def process_one(wav: Path) -> tuple[str, int, float, float, int]:
    """Returns status, n_songs, seconds, max_ratio, n_high."""
    tmp = WORK / f"tmp_{os.getpid()}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    began = time.time()
    try:
        rows, max_ratio = run_energy(wav, tmp)
        gate, n_high = needs_song(rows)
        if not gate:
            return "ok_energy_only", 0, time.time() - began, max_ratio, n_high
        n = run_song(wav, tmp)
        return "ok_song", n, time.time() - began, max_ratio, n_high
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def list_eligible() -> tuple[list[Path], int]:
    wavs = sorted(WAV_ROOT.rglob("*.wav"))
    eligible_paths: list[Path] = []
    skipped = 0
    if not SKIP_FILE.exists():
        with SKIP_FILE.open("w", newline="") as skip_handle:
            skip_writer = csv.writer(skip_handle)
            skip_writer.writerow(["wav_path", "reason"])
            for wav in wavs:
                ok, reason = eligible(wav)
                if ok:
                    eligible_paths.append(wav)
                else:
                    skipped += 1
                    skip_writer.writerow([str(wav), reason])
    else:
        for wav in wavs:
            ok, _ = eligible(wav)
            if ok:
                eligible_paths.append(wav)
        skipped = max(sum(1 for _ in SKIP_FILE.open()) - 1, 0)
    return eligible_paths, skipped


def main() -> int:
    for path in (DETECTOR, MODEL, ENERGY):
        if not path.exists():
            print(f"missing {path}", file=sys.stderr)
            return 2
    HERE.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(exist_ok=True)
    PLACEMENTS.mkdir(exist_ok=True)
    ensure_headers()

    log("hybrid listing wavs under /mnt/sda …")
    eligible_paths, skipped = list_eligible()
    log(f"eligible {len(eligible_paths)}, ineligible {skipped}")

    done_stems = load_done()
    todo = [w for w in eligible_paths if w.name.split(".")[0] not in done_stems]
    already = len(eligible_paths) - len(todo)
    log(f"resume: ignoring {already} already-processed stems, {len(todo)} remaining")

    songs = 0
    if SONGS_CSV.exists():
        with SONGS_CSV.open() as handle:
            songs = max(sum(1 for _ in handle) - 1, 0)

    began = time.time()
    failed = 0
    completed = already
    song_runs = 0
    energy_only = 0
    last_ok = ""
    last_err = ""
    write_progress(
        status="running",
        total=len(eligible_paths),
        done=completed,
        already=already,
        skipped=skipped,
        failed=failed,
        songs=songs,
        song_runs=song_runs,
        energy_only=energy_only,
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
                already=already,
                skipped=skipped,
                failed=failed,
                songs=songs,
                song_runs=song_runs,
                energy_only=energy_only,
                current=str(wav),
                last_ok=last_ok,
                last_err=last_err,
                began=began,
            )
            try:
                status, n, sec, max_ratio, n_high = process_one(wav)
            except Exception as exc:
                status, n, sec, max_ratio, n_high = "fail", 0, 0.0, 0.0, 0
                last_err = f"{wav}: {exc}"
                log(last_err)
                traceback.print_exc()
                with (HERE / "failures.log").open("a") as fh:
                    fh.write(f"{wav}\t{exc}\n")

            if status.startswith("ok"):
                mark_done(stem)
                completed += 1
                songs += n
                if status == "ok_song":
                    song_runs += 1
                else:
                    energy_only += 1
                last_ok = (
                    f"{wav} ({sec:.1f}s, max_ratio={max_ratio:.3f}, "
                    f"high_segs={n_high}, +{n} songs, {status})"
                )
                writer.writerow([str(wav), stem, status, "", f"{sec:.2f}", n, f"{max_ratio:.6f}", n_high])
                handle.flush()
                log(
                    f"OK {stem}: {sec:.1f}s {status} max_ratio={max_ratio:.3f} "
                    f"high={n_high} songs +{n} (total {songs})"
                )
            else:
                failed += 1
                last_err = str(wav)
                writer.writerow([str(wav), stem, "fail", "see failures.log", f"{sec:.2f}", 0, "", ""])
                handle.flush()

            write_progress(
                status="running",
                total=len(eligible_paths),
                done=completed,
                already=already,
                skipped=skipped,
                failed=failed,
                songs=songs,
                song_runs=song_runs,
                energy_only=energy_only,
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
        already=already,
        skipped=skipped,
        failed=failed,
        songs=songs,
        song_runs=song_runs,
        energy_only=energy_only,
        current="",
        last_ok=last_ok,
        last_err=last_err,
        began=began,
    )
    log(f"finished: completed {completed}, failed {failed}, songs {songs}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
