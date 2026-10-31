"""PDF report: motif-length scan (5–9 min) and 1-min vs 7-min placement.

Produces ``panmp_193500_length_scan.pdf`` in this directory. Motif discovery
reuses ``report_193500`` (one motif group per length, copy threshold lowered
until no 20-minute hole remains).
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec

from analysis.matrix_profile.replicate_report import report_193500 as R

HERE = Path(__file__).resolve().parent
OUT = HERE / "panmp_193500_length_scan.pdf"
FIGDIR = HERE / "cache_193500" / "figures"

LENGTHS = (5, 6, 7, 8, 9)
PAIRS = ((5, 6), (6, 7), (7, 8), (8, 9))
COLORS = {
    5: "#1f77b4",
    6: "#ff7f0e",
    7: "#bcbd22",
    8: "#9467bd",
    9: "#2ca02c",
    1: "#17becf",
}


def _wrap(text: str, width: int = 96) -> str:
    lines: list[str] = []
    for paragraph in text.split("\n"):
        if not paragraph.strip():
            lines.append("")
            continue
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    return "\n".join(lines)


def _iqr_text(gaps: np.ndarray) -> str:
    if len(gaps) == 0:
        return "—"
    q1, med, q3 = np.percentile(gaps, [25, 50, 75])
    return f"{med:.1f} ({q1:.1f}–{q3:.1f})"


def _count_diff_pct(n_a: int, n_b: int) -> float:
    base = max(n_a, n_b)
    if base == 0:
        return float("nan")
    return 100.0 * abs(n_a - n_b) / base


def _hist_gaps(gaps: np.ndarray, xmax: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
    """1-minute bins; last bin pools everything >= xmax - 1."""
    clipped = np.asarray(gaps, dtype=np.float64).copy()
    clipped[clipped >= xmax - 1] = xmax - 0.5
    edges = np.arange(0.0, xmax + 1.0, 1.0)
    counts, _ = np.histogram(clipped, bins=edges)
    centers = edges[:-1] + 0.5
    return centers, counts


def _ecdf(values: np.ndarray, xs: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.zeros_like(xs)
    ordered = np.sort(values)
    return np.searchsorted(ordered, xs, side="right") / len(ordered)


def scan_lengths(motif: np.ndarray, lengths: tuple[int, ...] = LENGTHS) -> dict:
    scans = {}
    for minutes in lengths:
        print(f"scan {minutes} min", flush=True)
        scans[minutes] = R._scan_one(motif, float(minutes), R.RATE)
    return scans


def length_summaries(scans: dict) -> list[dict]:
    rows = []
    for minutes in sorted(scans):
        s = scans[minutes]
        gaps = R.start_intervals_min(s["starts"], R.RATE)
        n = int(len(s["starts"]))
        under = int(np.sum(gaps < 10.0)) if len(gaps) else 0
        rows.append(
            {
                "minutes": minutes,
                "occurrences": n,
                "threshold": float(s["threshold"]),
                "best_r": float(s["best_r"]),
                "gaps": gaps,
                "median_iqr": _iqr_text(gaps),
                "under_10": under,
                "n_gaps": int(len(gaps)),
            }
        )
    return rows


def pair_summaries(scans: dict, pairs: tuple[tuple[int, int], ...] = PAIRS) -> list[dict]:
    rows = []
    for a, b in pairs:
        compared = R.compare_pair(
            scans[a]["starts"],
            scans[a]["w"],
            scans[b]["starts"],
            scans[b]["w"],
            R.RATE,
        )
        n_a, n_b = compared["n_a"], compared["n_b"]
        diff = _count_diff_pct(n_a, n_b)
        rows.append(
            {
                "pair": f"{a} vs {b}",
                "a": a,
                "b": b,
                "count_diff_pct": diff,
                "ks": compared["ks"],
                "w_min": compared["w_min"],
                "shared": compared["shared"],
                "n_a": n_a,
                "n_b": n_b,
            }
        )
    return rows


def plot_start_to_start(scans: dict, pairs: list[dict], out: Path) -> Path:
    lengths = sorted(scans)
    n_left = len(lengths)
    n_right = len(pairs)
    fig = plt.figure(figsize=(11.0, 12.5))
    fig.suptitle(
        "Start-to-start intervals of one motif group per length\n"
        "193500 (copies overlap ≤ 10%, no stretch > 20 min without a copy)",
        fontsize=12,
        y=0.995,
    )
    gs = GridSpec(
        max(n_left, n_right),
        2,
        figure=fig,
        left=0.08,
        right=0.98,
        top=0.93,
        bottom=0.05,
        hspace=0.55,
        wspace=0.22,
    )

    most_alike = min(pairs, key=lambda r: (r["ks"], r["w_min"]))["pair"]

    for i, minutes in enumerate(lengths):
        ax = fig.add_subplot(gs[i, 0])
        s = scans[minutes]
        gaps = R.start_intervals_min(s["starts"], R.RATE)
        centers, counts = _hist_gaps(gaps)
        color = COLORS.get(minutes, "#444444")
        ax.bar(centers, counts, width=1.0, color=color, edgecolor="white", linewidth=0.4)
        if len(gaps):
            med = float(np.median(gaps))
            ax.axvline(med, color="black", linestyle="--", linewidth=1.0)
            ymax = max(counts.max(), 1)
            ax.text(med, ymax * 1.02, f"median {med:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_xlim(0, 40)
        ax.set_ylim(0, max(counts.max() * 1.25, 5) if len(counts) else 5)
        ax.set_ylabel("count")
        if i == n_left - 1:
            ax.set_xlabel("start of one occurrence to start of the next (min; last bin ≥ 39)")
        else:
            ax.set_xticklabels([])
        thr = s["threshold"]
        ax.set_title(
            f"{minutes:g} min: {len(s['starts'])} occurrences, copies r ≥ {thr:.2f}",
            fontsize=9,
            loc="left",
        )

    for i, row in enumerate(pairs):
        ax = fig.add_subplot(gs[i, 1])
        a, b = row["a"], row["b"]
        gaps_a = R.start_intervals_min(scans[a]["starts"], R.RATE)
        gaps_b = R.start_intervals_min(scans[b]["starts"], R.RATE)
        xs = np.linspace(0, 40, 401)
        ax.plot(
            xs,
            _ecdf(gaps_a, xs),
            color=COLORS[a],
            label=f"{a} min",
            linewidth=1.6,
            drawstyle="steps-post",
        )
        ax.plot(
            xs,
            _ecdf(gaps_b, xs),
            color=COLORS[b],
            label=f"{b} min",
            linewidth=1.6,
            drawstyle="steps-post",
        )
        ax.set_xlim(0, 40)
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("fraction")
        if i == n_right - 1:
            ax.set_xlabel("start-to-start interval (min)")
        else:
            ax.set_xticklabels([])
        alike = " (most alike)" if row["pair"] == most_alike else ""
        ax.set_title(
            f"{a} vs {b} min: KS D = {row['ks']:.2f}, W = {row['w_min']:.1f} min, "
            f"counts differ {row['count_diff_pct']:.0f}%{alike}",
            fontsize=8,
            loc="left",
        )
        ax.legend(loc="lower right", fontsize=8, frameon=False)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def one_vs_seven(motif: np.ndarray, long: dict | None = None) -> dict:
    print("scan 1 min", flush=True)
    short = R._scan_one(motif, 1.0, R.RATE)
    if long is None:
        print("scan 7 min", flush=True)
        long = R._scan_one(motif, 7.0, R.RATE)
    mid_1 = R.midpoint_sec(short["starts"], short["w"], R.RATE)
    mid_7 = R.midpoint_sec(long["starts"], long["w"], R.RATE)
    if len(mid_7) == 0:
        offsets = np.full(len(mid_1), np.nan)
    else:
        offsets = np.empty(len(mid_1), dtype=np.float64)
        for i, t in enumerate(mid_1):
            offsets[i] = float(mid_7[np.argmin(np.abs(mid_7 - t))] - t)
    within = int(np.sum(np.abs(offsets) <= 60.0))
    return {
        "short": short,
        "long": long,
        "offsets_sec": offsets,
        "offsets_min": offsets / 60.0,
        "within_1min": within,
        "n_1": int(len(mid_1)),
        "n_7": int(len(mid_7)),
    }


def plot_one_vs_seven(rel: dict, out: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.5), gridspec_kw={"height_ratios": [2.2, 1.0]})
    fig.suptitle(
        "1-minute motif midpoints relative to nearest 7-minute motif\n"
        "193500 (same copy rules as the length scan)",
        fontsize=12,
    )

    ax = axes[0]
    offs = rel["offsets_min"]
    finite = offs[np.isfinite(offs)]
    # Keep a readable window around zero; pool extremes in the outer bins.
    lo, hi = -8.0, 8.0
    clipped = finite.copy()
    clipped[clipped < lo] = lo + 0.01
    clipped[clipped > hi] = hi - 0.01
    edges = np.arange(lo, hi + 0.25, 0.25)
    ax.hist(clipped, bins=edges, color=COLORS[1], edgecolor="white", linewidth=0.3)
    if len(finite):
        med = float(np.median(finite))
        ax.axvline(med, color="black", linestyle="--", linewidth=1.1)
        ax.text(med, ax.get_ylim()[1] * 0.92 if ax.get_ylim()[1] else 1, f"median {med:.2f} min",
                ha="center", va="bottom", fontsize=9)
    ax.axvline(-1.0, color="#444444", linestyle=":", linewidth=1.0)
    ax.axvline(1.0, color="#444444", linestyle=":", linewidth=1.0)
    ax.set_xlim(lo, hi)
    ax.set_xlabel("1-min midpoint − nearest 7-min midpoint (min)")
    ax.set_ylabel("count")
    ax.set_title(
        f"{rel['n_1']} occurrences of the 1-min motif; "
        f"{rel['within_1min']} of {rel['n_1']} within ±1 min of a 7-min midpoint "
        f"(7-min group: {rel['n_7']} occurrences, r ≥ {rel['long']['threshold']:.2f})",
        fontsize=9,
        loc="left",
    )

    ax = axes[1]
    mid_7 = R.midpoint_sec(rel["long"]["starts"], rel["long"]["w"], R.RATE) / 3600.0
    mid_1 = R.midpoint_sec(rel["short"]["starts"], rel["short"]["w"], R.RATE) / 3600.0
    ax.scatter(mid_7, np.full_like(mid_7, 1.0), s=18, color=COLORS[7], label="7 min", zorder=3)
    ax.scatter(mid_1, np.full_like(mid_1, 0.0), s=10, color=COLORS[1], alpha=0.7, label="1 min")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["1 min", "7 min"])
    ax.set_xlabel("time in recording (h)")
    ax.set_xlim(0, 10)
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.set_title("Occurrence midpoints across the recording", fontsize=9, loc="left")

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def _table(ax, headers: list[str], rows: list[list[str]], col_widths: list[float] | None = None) -> None:
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="upper center",
        cellLoc="center",
        colWidths=col_widths,
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    for (r, c), cell in table.get_celld().items():
        cell.set_edgecolor("none")
        cell.set_linewidth(0)
        if r == 0:
            cell.set_text_props(fontweight="bold")
            cell.visible_edges = "TB"
            cell.set_linewidth(1.1)
            cell.set_edgecolor("black")
        elif r == len(rows):
            cell.visible_edges = "B"
            cell.set_linewidth(1.1)
            cell.set_edgecolor("black")
        else:
            cell.visible_edges = ""


class Report:
    def __init__(self, pdf: PdfPages) -> None:
        self.pdf = pdf
        self.page = 0

    def new_page(self, title: str) -> plt.Figure:
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("white")
        self.page += 1
        fig.text(0.5, 0.02, str(self.page), ha="center", va="center", fontsize=9, color="#555555")
        fig.text(0.08, 0.955, title, fontsize=14, fontweight="bold", va="top", ha="left")
        self.y = 0.91
        self.fig = fig
        return fig

    def text(self, body: str, *, size: float = 10, gap: float = 0.01) -> None:
        wrapped = _wrap(body)
        n_lines = wrapped.count("\n") + 1
        height = n_lines * size * 1.35 / 72 / 11 + gap
        self.fig.text(0.08, self.y, wrapped, fontsize=size, va="top", ha="left", linespacing=1.35)
        self.y -= height

    def heading(self, title: str) -> None:
        self.fig.text(0.08, self.y, title, fontsize=11, fontweight="bold", va="top")
        self.y -= 0.026

    def image(self, path: Path, height: float = 0.55) -> None:
        ax = self.fig.add_axes([0.06, self.y - height, 0.88, height])
        ax.imshow(plt.imread(path))
        ax.axis("off")
        self.y -= height + 0.015

    def close(self) -> None:
        self.pdf.savefig(self.fig)
        plt.close(self.fig)


def write_pdf(
    summaries: list[dict],
    pairs: list[dict],
    fig_intervals: Path,
    rel: dict,
    fig_rel: Path,
    out: Path = OUT,
) -> Path:
    most = min(pairs, key=lambda r: (r["ks"], r["w_min"]))
    with PdfPages(out) as pdf:
        rep = Report(pdf)
        rep.new_page("Figure: Start-to-start intervals, 5–9 min")
        rep.text(
            "One motif group per length on the 2 kHz, 10-hour 193500 recording "
            "(MFCC coefficient 2 / c1 at 10 Hz). Seed is the best non-overlapping "
            "pair; copies keep correlation with the seed at a threshold lowered "
            "from 0.2 until no stretch longer than 20 minutes lacks a copy "
            "(overlap ≤ 10%).",
            size=9,
        )
        # Full remaining page for the multi-panel figure.
        rep.image(fig_intervals, height=min(0.78, rep.y - 0.04))
        rep.close()

        rep.new_page("Summary tables: 5–9 min length scan")
        rep.heading("Per length")
        headers = ["Length", "Occurrences", "Copy threshold", "Median interval (IQR)", "Intervals < 10 min"]
        rows = [
            [
                f"{r['minutes']} min",
                str(r["occurrences"]),
                f"{r['threshold']:.2f}",
                r["median_iqr"],
                f"{r['under_10']} of {r['n_gaps']}",
            ]
            for r in summaries
        ]
        ax = rep.fig.add_axes([0.08, rep.y - 0.28, 0.84, 0.28])
        _table(ax, headers, rows, col_widths=[0.12, 0.16, 0.18, 0.30, 0.24])
        rep.y -= 0.30

        rep.text(
            "“Same occurrences” counts shorter-length midpoints that lie within "
            "1 minute of a longer-length midpoint.",
            size=9,
        )
        rep.heading("Neighbouring lengths")
        headers2 = ["Pair", "Count difference", "KS D", "W", "Same occurrences"]
        rows2 = [
            [
                r["pair"],
                f"{r['count_diff_pct']:.0f}%",
                f"{r['ks']:.2f}",
                f"{r['w_min']:.1f} min",
                f"{r['shared']} of {r['n_a']}",
            ]
            for r in pairs
        ]
        ax = rep.fig.add_axes([0.08, rep.y - 0.26, 0.84, 0.26])
        _table(ax, headers2, rows2, col_widths=[0.16, 0.22, 0.14, 0.18, 0.30])
        rep.y -= 0.28

        notes = [
            f"• {most['pair']} min match most closely (KS D = {most['ks']:.2f}, "
            f"W = {most['w_min']:.1f} min, counts differ {most['count_diff_pct']:.0f}%).",
        ]
        shortish = [r for r in summaries if r["under_10"] / max(r["n_gaps"], 1) > 0.25]
        if shortish:
            names = ", ".join(f"{r['minutes']} min" for r in shortish)
            notes.append(
                f"• {names} still show many intervals under 10 minutes "
                "(half-song matches inside one cycle)."
            )
        longest = summaries[-1]
        notes.append(
            f"• At {longest['minutes']} min the median interval is {longest['median_iqr'].split()[0]} "
            f"min with {longest['under_10']} of {longest['n_gaps']} gaps under 10 minutes."
        )
        rep.text("\n".join(notes))
        rep.close()

        rep.new_page("1-minute motifs relative to 7-minute motifs")
        rep.text(
            "Both lengths use the same copy rules. For each 1-minute occurrence, "
            "the offset is its midpoint minus the midpoint of the nearest "
            "7-minute occurrence. Vertical dotted lines mark ±1 minute."
        )
        rep.image(fig_rel, height=0.52)
        pct = 100.0 * rel["within_1min"] / rel["n_1"] if rel["n_1"] else float("nan")
        offs = rel["offsets_min"][np.isfinite(rel["offsets_min"])]
        med = float(np.median(offs)) if len(offs) else float("nan")
        iqr = (
            f"{np.percentile(offs, 25):.2f}–{np.percentile(offs, 75):.2f}"
            if len(offs)
            else "—"
        )
        rep.heading("Counts")
        rep.text(
            f"1-min occurrences: {rel['n_1']} (copy threshold {rel['short']['threshold']:.2f})\n"
            f"7-min occurrences: {rel['n_7']} (copy threshold {rel['long']['threshold']:.2f})\n"
            f"Within ±1 min of a 7-min midpoint: {rel['within_1min']} of {rel['n_1']} "
            f"({pct:.0f}%)\n"
            f"Offset median (IQR): {med:.2f} min ({iqr} min)"
        )
        rep.close()
    return out


def main() -> Path:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    motif, _energy = R.load_features(R.WAV)
    scans = scan_lengths(motif, LENGTHS)
    summaries = length_summaries(scans)
    pairs = pair_summaries(scans, PAIRS)
    fig1 = plot_start_to_start(scans, pairs, FIGDIR / "start_to_start_5_9.png")
    rel = one_vs_seven(motif, long=scans.get(7))
    fig2 = plot_one_vs_seven(rel, FIGDIR / "one_vs_seven.png")
    path = write_pdf(summaries, pairs, fig1, rel, fig2, OUT)
    print(f"wrote {path}", flush=True)
    return path


if __name__ == "__main__":
    main()
