#!/usr/bin/env python3
"""Re-estimate a dictionary from prior-run matchings (LeiData wav1, 2 windows).

Loads matchings + widths from the latest stamped
``wav1_compare_build_pca_w2`` run, then calls ``match_to_dictionary`` with
``refine_iters=2`` on the same two windows.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from database import catalog  # noqa: E402
from analysis.dict_learning.code.core import Matching  # noqa: E402
from analysis.dict_learning.code.io import load_windows_from_paths  # noqa: E402
from analysis.dict_learning.code.learn.match_to_dictionary import (  # noqa: E402
    match_to_dictionary,
    write_match_to_dictionary_result,
)

LIBRARY_ROOT = Path("/home/yfreund/whale_processing")
DB = LIBRARY_ROOT / "result/dict_learning_catalog.sqlite"
WAV = LIBRARY_ROOT / "LeiData/SOCAL_H_76_230427_003345.x.wav"
SOURCE_ROOT = (
    LIBRARY_ROOT / "result/LeiData/dict_learning_templates/wav1_compare_build_pca_w2"
)
OUT = (
    LIBRARY_ROOT / "result/LeiData/dict_learning_templates/wav1_match_to_dictionary_w2"
)


def latest_stamped(root: Path) -> Path:
    stamped = sorted(
        [p for p in root.iterdir() if p.is_dir() and (p / "manifest.json").is_file()],
        key=lambda p: p.name,
    )
    if not stamped:
        raise FileNotFoundError(f"No stamped run under {root}")
    return stamped[-1]


def load_matchings_and_widths(manifest: dict) -> tuple[list[Matching], dict[int, int]]:
    widths = {
        int(t["template_id"]): int(t["width"]) for t in manifest.get("templates") or []
    }
    matchings: list[Matching] = []
    for m in manifest.get("matchings") or []:
        coeffs = m.get("coeffs")
        if coeffs is not None:
            matchings.append(
                Matching(
                    template_id=int(m["template_id"]),
                    window_id=str(m["window_id"]),
                    t_start=int(m["t_start"]),
                    coeffs=np.asarray(coeffs, dtype=np.float64),
                )
            )
        else:
            matchings.append(
                Matching(
                    template_id=int(m["template_id"]),
                    window_id=str(m["window_id"]),
                    t_start=int(m["t_start"]),
                    amplitude=float(m.get("amplitude", 1.0)),
                )
            )
    return matchings, widths


def load_first_n_windows(n: int = 2) -> list:
    with catalog.connect(DB) as conn:
        wav_row = catalog.find_wav_file(conn, str(WAV))
        if wav_row is None:
            raise FileNotFoundError(f"WAV not in catalog: {WAV}")
        segs = catalog.list_gpl_segmentations_for_wav(conn, wav_row["wav_id"])
        if not segs:
            raise FileNotFoundError(f"No segmentations for {WAV}")
        seg_id = int(segs[0]["segmentation_id"])
        window_rows = catalog.list_windows_for_segmentation(
            conn, seg_id, limit=n, offset=0
        )
        paths = [r["npy_path"] for r in window_rows]
    return load_windows_from_paths(paths)


def main() -> None:
    source = latest_stamped(SOURCE_ROOT)
    man = json.loads((source / "manifest.json").read_text())
    matchings, widths = load_matchings_and_widths(man)
    windows = load_first_n_windows(2)
    print(
        f"source={source} windows={len(windows)} "
        f"input_matchings={len(matchings)} widths={widths}",
        flush=True,
    )
    result = match_to_dictionary(
        windows,
        matchings,
        widths=widths,
        refine_iters=2,
        eps_step_fraction=0.2,
        template_method="pca",
        n_components=1,
        show_progress=True,
    )
    OUT.mkdir(parents=True, exist_ok=True)
    out_dir = write_match_to_dictionary_result(
        result,
        OUT,
        name="leidata_wav1_match_to_dictionary_w2",
        params={
            "source_manifest": str(source / "manifest.json"),
            "refine_iters": 2,
            "n_windows": 2,
            "widths": {str(k): v for k, v in widths.items()},
            "n_input_matchings": len(matchings),
        },
    )
    by_tid: dict[int, int] = {}
    for m in result.matchings:
        by_tid[int(m.template_id)] = by_tid.get(int(m.template_id), 0) + 1
    summary = {
        "source_dir": str(source),
        "out_dir": str(out_dir),
        "n_atoms": len(result.dictionary),
        "n_matchings": len(result.matchings),
        "matchings_per_template": by_tid,
        "iterations_completed": result.iterations_completed,
        "energy_before": result.energies[0] if result.energies else None,
        "energy_after": result.energies[-1] if result.energies else None,
    }
    text = json.dumps(summary, indent=2, default=str)
    print(text)
    (OUT / "summary.json").write_text(text)


if __name__ == "__main__":
    main()
