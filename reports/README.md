# Reports

These are the reports for the humpback song-session work on branch `dat-hmm`, in the order the experiments were run.

Figures and code stay in each experiment's folder under `analysis/`, and the links in these reports point there. The reports are generated: rerunning an experiment's report command rewrites its file here.

| Report | What it covers | Regenerate with |
|---|---|---|
| [exp1_slot_hmm.md](exp1_slot_hmm.md) | **Experiment 1:** per-second slot HMM (28 phrase slots, 18-number per-second input) | `python -m analysis.song_hmm.report` |
| [exp2_phrase_hmm.md](exp2_phrase_hmm.md) | **Experiment 2:** phrase-window theme HMM, with Experiment 1 reviewed and compared | `python -m analysis.song_hmm_phrase.report` |
| [exp2_pipeline_check.md](exp2_pipeline_check.md) | Check of Experiment 2's wav → matrix processing: is it correct, and what does it lose | `python -m analysis.song_hmm_phrase.pipeline_check.render` (re-render), or `.check` (rerun everything) |
| [exp3_inputs_and_models.md](exp3_inputs_and_models.md) | **Experiment 3:** 120 input combinations and four models (M1–M4), with Experiment 2 reviewed and compared | `python -m analysis.song_exp3.evaluate`, then `python -m analysis.song_exp3.report` |
| [exp3_best_combination.md](exp3_best_combination.md) | The best setup (`S2_N1_X2_20s` + M1) on its own: how it works, what it does well and not well | Written by hand from `analysis/song_exp3/work/eval.json` |

The notebook that runs the best model on the wav and shows each detected song's spectrogram is [analysis/song_exp3/notebooks/detect_songs_best_model.ipynb](../analysis/song_exp3/notebooks/detect_songs_best_model.ipynb).
