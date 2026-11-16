#!/usr/bin/env python3
"""Build year-panel plots of whitened 50–500 Hz energy fraction vs time."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

HERE = Path(__file__).resolve().parent
CSV = HERE / "segments_monthly.csv"
FIG = HERE / "figures"
MD = HERE / "energy_ratio_by_year.md"
STAMP = re.compile(r"(?<!\d)(\d{6})_(\d{6})(?!\d)")
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


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


def load() -> pd.DataFrame:
    df = pd.read_csv(CSV)
    t0 = df["filename"].map(parse_time)
    df = df.assign(
        t0=t0,
        time=lambda d: d["t0"] + pd.to_timedelta(d["start_sec"], unit="s"),
    )
    df = df.dropna(subset=["time"])
    df["year"] = df["time"].dt.year
    return df


def plot_year(df: pd.DataFrame, year: int, out: Path) -> tuple[int, int]:
    sub = df[df["year"] == year].sort_values("time")
    fig, ax = plt.subplots(figsize=(11, 3.2))
    ax.plot(sub["time"], sub["ratio"], color="#4c78a8", lw=0.7, alpha=0.9)
    ax.set_ylim(0, min(1.0, max(0.35, float(sub["ratio"].quantile(0.995)) * 1.15)))
    ax.set_ylabel("50–500 Hz fraction")
    ax.set_xlabel("")
    ax.set_title(f"{year} — whitened band energy / total (10 Hz–Nyquist)")
    ax.set_xlim(datetime(year, 1, 1), datetime(year, 12, 31, 23, 59, 59))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.grid(True, axis="y", alpha=0.3)
    fig.autofmt_xdate(rotation=0, ha="center")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return len(sub), sub["filename"].nunique()


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = load()
    years = sorted(df["year"].unique())
    lines = [
        "# Whitened 50–500 Hz energy fraction by year",
        "",
        "Each point is a **600-second** segment. The ratio is the sum of squares of the "
        "GPL-whitened spectrogram in 50–500 Hz, divided by the sum of squares from 10 Hz "
        "to Nyquist (same definition as `Initial_plan.md`).",
        "",
        "Files are a monthly stratified sample of Site H `df20` recordings at 10 kHz or "
        "16 kHz (up to four files per calendar month, evenly spaced). Time comes from the "
        "`YYMMDD_HHMMSS` stamp in the file name plus `start_sec` within the file. "
        "X-axis ticks are month names.",
        "",
        f"**Segments:** {len(df):,} &nbsp;|&nbsp; **Files:** {df['filename'].nunique():,} "
        f"&nbsp;|&nbsp; **Years:** {years[0]}–{years[-1]}",
        "",
    ]
    for year in years:
        out = FIG / f"energy_{year}.png"
        n_seg, n_files = plot_year(df, year, out)
        lines += [
            f"## {year}",
            "",
            f"{n_files} files, {n_seg:,} segments.",
            "",
            f"![Energy ratio {year}](figures/energy_{year}.png)",
            "",
        ]
    lines += [
        "## Files",
        "",
        "- `segments_monthly.csv` — segment ratios used for the plots",
        "- `figures/energy_YYYY.png` — one panel per year",
        "- `build_sample_and_run.py` — sample selection and `energy_filter --cpu` runner",
        "- `plot_years.py` — this figure builder",
        "",
    ]
    MD.write_text("\n".join(lines))
    print("wrote", MD, "years", years)


if __name__ == "__main__":
    main()
