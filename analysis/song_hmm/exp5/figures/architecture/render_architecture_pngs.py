#!/usr/bin/env python3
"""Render Exp5 theme–slot HMM architecture diagrams to PNG.

Prefers system ``dot``; otherwise draws a layout matching the soft-HMM
Graphviz style (theme clusters, dashed repeat, next edges).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

HERE = Path(__file__).resolve().parent


def _try_graphviz(dot_path: Path, png_path: Path) -> bool:
    dot = shutil.which("dot")
    if not dot:
        return False
    subprocess.run([dot, "-Tpng", str(dot_path), "-o", str(png_path)], check=True)
    return True


def _rounded_box(ax, x, y, w, h, text, *, fc="#f6f6f6", ec="#333333"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=fc,
            edgecolor=ec,
            lw=1.15,
            zorder=3,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=8,
        zorder=4,
        linespacing=1.25,
    )


def _arrow(ax, x1, y1, x2, y2, *, style="-", color="#444444", lw=1.15, label=None, label_xy=None):
    ax.annotate(
        "",
        xy=(x2, y2),
        xytext=(x1, y1),
        arrowprops=dict(arrowstyle="->", color=color, lw=lw, linestyle=style),
        zorder=2,
    )
    if label:
        if label_xy is None:
            label_xy = ((x1 + x2) / 2 + 0.12, (y1 + y2) / 2)
        ax.text(label_xy[0], label_xy[1], label, fontsize=7.5, color=color, ha="left", va="center")


def draw_theme_slot_hmm(
    png_path: Path,
    *,
    title: str,
    subtitle: str,
    emission_line: str,
    box_fc: str,
    themes: list[int],
    last_theme_note: str = "repeat until song ends / next (learned)",
) -> None:
    """Soft-HMM style: stacked theme bands, dashed repeat, next between themes."""
    n_themes = len(themes)
    max_slots = max(themes)
    box_w, box_h = 1.35, 0.78
    gap_x = 0.28
    row_h = 1.85
    left = 1.55
    margin_top = 1.35
    top = margin_top + n_themes * row_h

    fig_w = left + max_slots * (box_w + gap_x) + 2.2
    fig_h = top + 1.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    ax.text(fig_w / 2, fig_h - 0.25, title, ha="center", va="top", fontsize=12, fontweight="bold")
    ax.text(fig_w / 2, fig_h - 0.55, subtitle, ha="center", va="top", fontsize=8.5, color="0.25")

    start_xy = (0.55, top - 0.55)
    ax.add_patch(Circle(start_xy, 0.11, facecolor="#222222", edgecolor="#222222", zorder=5))

    centers: list[list[tuple[float, float]]] = []
    for ti, n_slots in enumerate(themes):
        y = top - (ti + 1) * row_h + 0.35
        band_w = n_slots * (box_w + gap_x) + 0.4
        ax.add_patch(
            Rectangle(
                (left - 0.28, y - 0.35),
                band_w,
                box_h + 0.85,
                facecolor="#fafafa",
                edgecolor="#bbbbbb",
                lw=1.0,
                zorder=1,
            )
        )
        if ti + 1 < n_themes:
            band_title = f"Theme {ti + 1}    repeat p{ti+1} (learned)"
        else:
            band_title = f"Theme {ti + 1}    {last_theme_note}"
        ax.text(left - 0.18, y + box_h + 0.32, band_title, fontsize=9.5, va="bottom", color="#333333")

        row = []
        for si in range(n_slots):
            x = left + si * (box_w + gap_x)
            label = f"T{ti+1}.{si+1}\n{emission_line}"
            _rounded_box(ax, x, y, box_w, box_h, label, fc=box_fc)
            cx, cy = x + box_w / 2, y + box_h / 2
            row.append((cx, cy, x, y))
            if si > 0:
                prev = row[si - 1]
                _arrow(ax, prev[0] + box_w / 2 - 0.05, prev[1], x - 0.02, cy)
        centers.append(row)

        # dashed repeat: last → first
        x0, y0 = row[-1][0], row[-1][1] + box_h / 2 + 0.02
        x1, y1 = row[0][0], row[0][1] + box_h / 2 + 0.02
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->",
                color="#666666",
                lw=1.1,
                linestyle="--",
                connectionstyle="arc3,rad=-0.4",
            ),
            zorder=2,
        )

        if ti == 0:
            _arrow(ax, start_xy[0] + 0.11, start_xy[1], row[0][2], row[0][3] + box_h / 2)

    for ti in range(n_themes - 1):
        a = centers[ti][-1]
        b = centers[ti + 1][0]
        _arrow(
            ax,
            a[0],
            a[3] - 0.02,
            b[2] + box_w / 2,
            b[3] + box_h + 0.02,
            label=f"next 1−p{ti+1}",
            label_xy=(a[0] + 0.2, (a[3] + b[3] + box_h) / 2),
        )

    ax.text(
        0.35,
        0.35,
        "Solid: advance slot (or next theme).  Dashed: repeat phrase (last → first of same theme).  "
        "No skip edges inside a theme — a different motif sequence is a different theme.  "
        "T and #slots/theme are schematic; chosen on train.",
        fontsize=8,
        va="center",
        color="0.2",
        wrap=True,
    )

    fig.tight_layout()
    fig.savefig(png_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", png_path)


def draw_fixed_soft_hmm_t6(png_path: Path) -> None:
    """Reproduce the fixed soft-HMM reference (known units / durations / probs)."""
    themes = [
        [("T1.1", "U7 · 2.0 s"), ("T1.2", "U2 · 1.7 s"), ("T1.3", "U9 · 1.2 s")],
        [("T2.1", "U1 · 1.0 s"), ("T2.2", "U8 · 1.0 s"), ("T2.3", "U7 · 1.6 s")],
        [
            ("T3.1", "U9 · 1.3 s"),
            ("T3.2", "U11 · 1.2 s"),
            ("T3.3", "U9 · 1.0 s"),
            ("T3.4", "U6 · 1.0 s"),
            ("T3.5", "U8 · 1.0 s"),
            ("T3.6", "U7 · 1.1 s"),
        ],
        [("T4.1", "U2 · 1.2 s"), ("T4.2", "U9 · 1.0 s"), ("T4.3", "U6 · 1.0 s")],
        [
            ("T5.1", "U8 · 1.2 s"),
            ("T5.2", "U9 · 1.1 s"),
            ("T5.3", "U3 · 1.0 s"),
            ("T5.4", "U8 · 1.0 s"),
            ("T5.5", "U7 · 1.1 s"),
        ],
        [
            ("T6.1", "U9 · 1.1 s"),
            ("T6.2", "U3 · 1.0 s"),
            ("T6.3", "U8 · 1.6 s"),
            ("T6.4", "U2 · 2.0 s"),
            ("T6.5", "U7 · 1.1 s"),
            ("T6.6", "U9 · 2.0 s"),
            ("T6.7", "U11 · 1.0 s"),
            ("T6.8", "U2 · 1.3 s"),
        ],
    ]
    repeats = ["0.008", "0.984", "0.943", "0.585", "0.922", "1 until the song ends"]
    nexts = ["0.992", "0.016", "0.057", "0.415", "0.078"]

    n_themes = len(themes)
    max_slots = max(len(t) for t in themes)
    box_w, box_h = 1.2, 0.72
    gap_x = 0.22
    row_h = 1.7
    left = 1.4
    top = 0.9 + n_themes * row_h
    fig_w = left + max_slots * (box_w + gap_x) + 1.6
    fig_h = top + 0.6
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.axis("off")
    ax.text(
        fig_w / 2,
        fig_h - 0.2,
        "Fixed soft-HMM reference (193500) — NOT the Exp5 model",
        ha="center",
        va="top",
        fontsize=12,
        fontweight="bold",
    )

    start_xy = (0.5, top - 0.45)
    ax.add_patch(Circle(start_xy, 0.1, facecolor="#222222", edgecolor="#222222", zorder=5))

    centers = []
    for ti, slots in enumerate(themes):
        y = top - (ti + 1) * row_h + 0.3
        band_w = len(slots) * (box_w + gap_x) + 0.35
        ax.add_patch(
            Rectangle(
                (left - 0.25, y - 0.3),
                band_w,
                box_h + 0.75,
                facecolor="#fafafa",
                edgecolor="#bbbbbb",
                lw=1.0,
                zorder=1,
            )
        )
        ax.text(
            left - 0.15,
            y + box_h + 0.28,
            f"Theme {ti + 1}    repeat {repeats[ti]}",
            fontsize=9.5,
            va="bottom",
        )
        row = []
        for si, (name, em) in enumerate(slots):
            x = left + si * (box_w + gap_x)
            _rounded_box(ax, x, y, box_w, box_h, f"{name}\n{em}")
            cx, cy = x + box_w / 2, y + box_h / 2
            row.append((cx, cy, x, y))
            if si > 0:
                prev = row[si - 1]
                _arrow(ax, prev[0] + box_w / 2 - 0.04, prev[1], x - 0.02, cy)
        centers.append(row)
        x0, y0 = row[-1][0], row[-1][1] + box_h / 2 + 0.02
        x1, y1 = row[0][0], row[0][1] + box_h / 2 + 0.02
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->",
                color="#666666",
                lw=1.05,
                linestyle="--",
                connectionstyle="arc3,rad=-0.38",
            ),
            zorder=2,
        )
        if ti == 0:
            _arrow(ax, start_xy[0] + 0.1, start_xy[1], row[0][2], row[0][3] + box_h / 2)

    for ti in range(n_themes - 1):
        a = centers[ti][-1]
        b = centers[ti + 1][0]
        _arrow(
            ax,
            a[0],
            a[3] - 0.02,
            b[2] + box_w / 2,
            b[3] + box_h + 0.02,
            label=f"next {nexts[ti]}",
            label_xy=(a[0] + 0.15, (a[3] + b[3] + box_h) / 2),
        )

    fig.tight_layout()
    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", png_path)


def main() -> None:
    a_dot = HERE / "approachA_theme_slot_hmm.dot"
    b_dot = HERE / "approachB_theme_slot_hmm.dot"
    ref_dot = HERE / "song_soft_hmm_t6_reference.dot"
    a_png = HERE / "approachA_theme_slot_hmm.png"
    b_png = HERE / "approachB_theme_slot_hmm.png"
    ref_png = HERE / "song_soft_hmm_t6_reference.png"
    a_alias = HERE / "approachA_sticky_categorical_hmm.png"
    b_alias = HERE / "approachB_sticky_gaussian_hmm.png"

    # Schematic flexible Exp5 graphs: T=6 with soft-HMM-like slot lengths.
    # Actual T / lengths are chosen on train — this is layout only.
    themes = [3, 3, 6, 3, 5, 8]

    if not _try_graphviz(a_dot, a_png):
        draw_theme_slot_hmm(
            a_png,
            title="Approach A (Exp5): flexible theme–slot HMM — dictionary motifs",
            subtitle="Schematic T=6 (soft-HMM layout family); actual T / #slots / motif ids learned on train. No skip inside a theme.",
            emission_line="motif (learned)",
            box_fc="#f6f6f6",
            themes=themes,
            last_theme_note="repeat p₆ (learned) until song ends",
        )
    else:
        print("graphviz A")

    if not _try_graphviz(b_dot, b_png):
        draw_theme_slot_hmm(
            b_png,
            title="Approach B (Exp5): flexible theme–slot HMM — Gaussian on x_t",
            subtitle="Schematic T=6 (same topology as A); actual T / #slots / N(μ,Σ) learned on train. No skip inside a theme.",
            emission_line="N(μ,Σ)",
            box_fc="#eef6fb",
            themes=themes,
            last_theme_note="repeat p₆ (learned) until song ends",
        )
    else:
        print("graphviz B")

    if not _try_graphviz(ref_dot, ref_png):
        draw_fixed_soft_hmm_t6(ref_png)
    else:
        print("graphviz reference")

    shutil.copyfile(a_png, a_alias)
    shutil.copyfile(b_png, b_alias)
    print("also wrote aliases", a_alias.name, b_alias.name)


if __name__ == "__main__":
    main()
