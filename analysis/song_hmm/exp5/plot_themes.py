"""Plot each Approach A theme visit: spectrogram + motif bars below + boundaries."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from analysis.song_hmm.exp4.spectrogram import stitch_gpl
from analysis.song_hmm.exp5.paths import GPL_193500, WORK
from analysis.song_hmm.exp6.export_csvs import Placement

# Max seconds per panel so long T8 visits stay readable
DEFAULT_PANEL_S = 300.0  # 5 min
PAD_S = 2.0


def load_eval_themes(eval_path: Path | None = None) -> tuple[list[dict], list[float]]:
    path = Path(eval_path) if eval_path else WORK / "results" / "A" / "eval.json"
    data = json.loads(path.read_text())
    themes = [t for t in (data.get("themes") or []) if float(t["t1"]) > float(t["t0"])]
    bounds = [float(b) for b in (data.get("boundaries") or [])]
    return themes, bounds


def _motif_color_map(motif_ids: Sequence[int]) -> dict[int, tuple]:
    colors = plt.cm.tab20.colors
    uniq = sorted(set(int(m) for m in motif_ids))
    return {m: colors[i % len(colors)] for i, m in enumerate(uniq)}


def plot_theme_panel(
    placements: Sequence[Placement],
    *,
    theme_label: str,
    visit_t0: float,
    visit_t1: float,
    panel_t0: float,
    panel_t1: float,
    boundaries: Sequence[float],
    gpl_dir: Path = GPL_193500,
    ax_spec: plt.Axes | None = None,
    ax_motifs: plt.Axes | None = None,
    vmax_pct: float = 99.0,
) -> tuple[plt.Axes, plt.Axes]:
    """Spectrogram on top; motif bars on a strip below; theme edges + other bounds as lines."""
    if ax_spec is None or ax_motifs is None:
        fig, (ax_spec, ax_motifs) = plt.subplots(
            2,
            1,
            figsize=(14, 4.2),
            gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.05},
            sharex=True,
        )
    else:
        fig = ax_spec.figure

    # 75 s GPL windows — load only the panel neighborhood
    win_s = 75.0
    w0 = max(0, int(panel_t0 // win_s) - 1)
    n_w = int(np.ceil((panel_t1 - w0 * win_s) / win_s)) + 2
    spec, t0, t1, _ = stitch_gpl(
        gpl_dir, window_offset=w0, n_windows=n_w, t0=panel_t0, t1=panel_t1
    )
    finite = spec[np.isfinite(spec)]
    vmax = float(np.percentile(finite, vmax_pct)) if finite.size else 1.0
    vmin = float(np.percentile(finite, 1.0)) if finite.size else 0.0
    ax_spec.imshow(
        spec,
        origin="lower",
        aspect="auto",
        extent=[t0 / 60.0, t1 / 60.0, 0, spec.shape[0]],
        cmap="magma",
        vmin=vmin,
        vmax=max(vmax, vmin + 1e-6),
        interpolation="nearest",
    )
    # light theme tint over the visit ∩ panel
    tint0 = max(visit_t0, t0) / 60.0
    tint1 = min(visit_t1, t1) / 60.0
    if tint1 > tint0:
        ax_spec.axvspan(tint0, tint1, color="deepskyblue", alpha=0.12, lw=0)

    # boundaries: visit edges (yellow) + any decoded bounds in panel (cyan)
    for b, color, lw in (
        (visit_t0, "yellow", 2.0),
        (visit_t1, "yellow", 2.0),
    ):
        if t0 <= b <= t1:
            ax_spec.axvline(b / 60.0, color=color, lw=lw, alpha=0.95)
            ax_motifs.axvline(b / 60.0, color=color, lw=lw, alpha=0.95)
    for b in boundaries:
        if t0 < b < t1 and abs(b - visit_t0) > 0.05 and abs(b - visit_t1) > 0.05:
            ax_spec.axvline(b / 60.0, color="cyan", lw=1.2, alpha=0.9)
            ax_motifs.axvline(b / 60.0, color="cyan", lw=1.2, alpha=0.9)

    ax_spec.set_ylabel("freq bin")
    ax_spec.set_title(
        f"{theme_label}  visit {visit_t0/60:.1f}–{visit_t1/60:.1f} min  |  "
        f"panel {panel_t0/60:.1f}–{panel_t1/60:.1f} min   "
        f"(yellow = this theme edge, cyan = other theme boundaries)"
    )
    ax_spec.tick_params(labelbottom=False)

    # Motif bars on dedicated strip (below spectrogram)
    in_panel = [
        p
        for p in placements
        if p.end_sec >= t0 and p.start_sec <= t1 and p.end_sec > p.start_sec
    ]
    cmap = _motif_color_map([p.motif for p in in_panel])
    ax_motifs.set_xlim(t0 / 60.0, t1 / 60.0)
    ax_motifs.set_ylim(0, 1)
    ax_motifs.set_yticks([])
    ax_motifs.set_xlabel("time (min from wav start)")
    ax_motifs.set_ylabel("motifs")
    for p in in_panel:
        x0 = max(p.start_sec, t0) / 60.0
        x1 = min(p.end_sec, t1) / 60.0
        if x1 <= x0:
            continue
        ax_motifs.axvspan(x0, x1, ymin=0.15, ymax=0.85, color=cmap[p.motif], alpha=0.95, lw=0)
    # theme interval marker on motif strip
    if tint1 > tint0:
        ax_motifs.axvspan(tint0, tint1, ymin=0.0, ymax=0.12, color="deepskyblue", alpha=0.8, lw=0)

    n_mot = len(in_panel)
    n_ids = len(cmap)
    ax_motifs.set_title(f"{n_mot} motif placements ({n_ids} ids) in panel", fontsize=9, loc="left")
    if cmap:
        handles = [
            Patch(facecolor=c, edgecolor="none", label=f"M{m}")
            for m, c in sorted(cmap.items())[:14]
        ]
        ax_motifs.legend(
            handles=handles,
            loc="upper right",
            ncol=min(7, len(handles)),
            fontsize=6,
            framealpha=0.75,
        )
    return ax_spec, ax_motifs


def plot_all_theme_visits(
    placements: Sequence[Placement],
    *,
    eval_path: Path | None = None,
    out_dir: Path | None = None,
    gpl_dir: Path = GPL_193500,
    panel_s: float = DEFAULT_PANEL_S,
    dpi: int = 130,
) -> list[Path]:
    """Write one PNG per panel for every positive-duration theme visit. Returns paths."""
    themes, boundaries = load_eval_themes(eval_path)
    out_dir = Path(out_dir) if out_dir else WORK / "figures" / "themes_by_label"
    out_dir.mkdir(parents=True, exist_ok=True)

    # group visits by label, preserve time order
    by_label: dict[str, list[dict]] = {}
    for th in sorted(themes, key=lambda t: (t["theme"], float(t["t0"]))):
        by_label.setdefault(str(th["theme"]), []).append(th)

    written: list[Path] = []
    index_rows: list[str] = [
        "# Theme visits (Approach A)",
        "",
        "Yellow lines = edges of this theme visit. Cyan = other theme boundaries. "
        "Bottom strip = motif bars.",
        "",
    ]

    for label in sorted(by_label.keys(), key=lambda x: (len(x), x)):
        visits = by_label[label]
        label_dir = out_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        index_rows.append(f"## {label}  ({len(visits)} visits)")
        index_rows.append("")
        for vi, th in enumerate(visits, start=1):
            vt0, vt1 = float(th["t0"]), float(th["t1"])
            dur = vt1 - vt0
            index_rows.append(
                f"### {label} visit {vi:02d}  "
                f"({vt0/60:.2f}–{vt1/60:.2f} min, {dur:.1f} s, n_tokens={th.get('n_tokens')})"
            )
            index_rows.append("")
            # pad slightly, then chunk
            span0 = max(0.0, vt0 - PAD_S)
            span1 = vt1 + PAD_S
            part = 0
            t = span0
            while t < span1 - 1e-6:
                p0 = t
                p1 = min(t + panel_s, span1)
                part += 1
                fig, (ax_s, ax_m) = plt.subplots(
                    2,
                    1,
                    figsize=(14, 4.2),
                    gridspec_kw={"height_ratios": [3.2, 1.0], "hspace": 0.08},
                    sharex=True,
                )
                plot_theme_panel(
                    placements,
                    theme_label=f"{label} visit {vi:02d}",
                    visit_t0=vt0,
                    visit_t1=vt1,
                    panel_t0=p0,
                    panel_t1=p1,
                    boundaries=boundaries,
                    gpl_dir=gpl_dir,
                    ax_spec=ax_s,
                    ax_motifs=ax_m,
                )
                fig.subplots_adjust(hspace=0.12)
                name = f"visit_{vi:02d}_part_{part:02d}.png"
                path = label_dir / name
                fig.savefig(path, dpi=dpi, bbox_inches="tight")
                plt.close(fig)
                written.append(path)
                # relative path from notebook
                rel = f"../work/figures/themes_by_label/{label}/{name}"
                index_rows.append(f"![{label} v{vi} p{part}]({rel})")
                index_rows.append("")
                t = p1
        index_rows.append("")

    (out_dir / "README_gallery.md").write_text("\n".join(index_rows))
    # HTML gallery
    html = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'/>",
        "<title>Theme visits</title>",
        "<style>body{font-family:system-ui;margin:16px;background:#111;color:#eee}",
        "img{width:100%;max-width:1400px;display:block;margin:8px 0 20px;border:1px solid #333}",
        "a{color:#8cf} h1,h2,h3{margin:12px 0 4px}</style></head><body>",
        "<h1>Approach A — all theme visits</h1>",
        "<p>Yellow = theme visit edges. Cyan = other boundaries. Bottom = motif bars.</p>",
    ]
    for label in sorted(by_label.keys(), key=lambda x: (len(x), x)):
        html.append(f"<h2 id='{label}'>{label}</h2>")
        for p in sorted((out_dir / label).glob("visit_*.png")):
            html.append(f"<h3>{label} / {p.name}</h3>")
            html.append(f"<img src='{label}/{p.name}' loading='lazy'/>")
    html.append("</body></html>")
    (out_dir / "index.html").write_text("\n".join(html))
    print(f"wrote {len(written)} panels → {out_dir}")
    return written
