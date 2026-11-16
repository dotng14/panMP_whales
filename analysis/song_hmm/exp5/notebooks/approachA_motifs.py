"""Generate Approach A notebook figures/tables from work/results (run as script or paste into notebook)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

from analysis.song_hmm.exp4.spectrogram import plot_themes_on_spectrogram, stitch_gpl
from analysis.song_hmm.exp5.paths import GPL_193500, TRAIN_N_WINDOWS, TEST_N_WINDOWS, WORK


def main() -> None:
    eval_path = WORK / "results" / "A" / "eval.json"
    data = json.loads(eval_path.read_text())
    themes = data["themes"]
    boundaries = data["boundaries"]
    print(f"n_boundaries={len(boundaries)} n_theme_segments={len(themes)}")
    print("JS summary:", json.dumps(data.get("js_summary"), indent=2))
    print("Scores (provisional marks):", json.dumps(data.get("scores_vs_provisional_marks"), indent=2))
    print("\nEvery boundary (s / min / h):")
    for b in boundaries:
        print(f"  {b:.1f} s  |  {b/60:.2f} min  |  {b/3600:.3f} h")
    print("\nBoth themes at each boundary:")
    for b in boundaries:
        before = [t for t in themes if t["t0"] < b <= t["t1"] + 1e-9] or [
            t for t in themes if t["t1"] <= b
        ]
        after = [t for t in themes if t["t0"] >= b]
        left = before[-1] if before else None
        right = after[0] if after else None
        print(f"  t={b:.1f}: before={left} after={right}")

    # one overview figure of first 20 min of test
    spec, t0, t1, _ = stitch_gpl(
        GPL_193500, window_offset=TRAIN_N_WINDOWS, n_windows=min(16, TEST_N_WINDOWS)
    )
    fig, ax = plt.subplots(figsize=(14, 3.5))
    plot_themes_on_spectrogram(
        spec, t0, t1, themes, boundaries, title="Exp5 Approach A — test start", ax=ax
    )
    out = WORK / "figures" / "A" / "notebook_overview.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
