#!/usr/bin/env python3
"""Run matched extend_dictionary cold-start: 4 windows, max 8 atoms."""

from __future__ import annotations

import json
from pathlib import Path

from analysis.dict_learning.code.learn.extend_dictionary import extend_dictionary_for_wav

OUT = Path(
    "/home/yfreund/whale_processing/result/LeiData/dict_learning_templates/"
    "wav1_compare_extend_pca_w4_r8"
)
DB = Path("/home/yfreund/whale_processing/result/dict_learning_catalog.sqlite")
WAV = "/home/yfreund/whale_processing/LeiData/SOCAL_H_76_230427_003345.x.wav"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = extend_dictionary_for_wav(
        DB,
        WAV,
        out_dir=OUT,
        name="leidata_wav1_compare_extend_pca_w4_r8",
        n_windows=4,
        window_offset=0,
        min_coverage=0.999,
        max_new_atoms=8,
        eps_step_fraction=0.2,
        min_repeats=3,
        min_template_width=None,
        max_template_width=24,
        refine_iters=2,
        n_components=1,
        show_progress=True,
    )
    text = json.dumps(summary, indent=2, default=str)
    print(text)
    (OUT / "summary.json").write_text(text)


if __name__ == "__main__":
    main()
