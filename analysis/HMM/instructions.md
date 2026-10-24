# Use the fitted song HMM

The fitted model is [song_soft_hmm_t6.npz](song_soft_hmm_t6.npz). It contains the phrase-slot Gaussians, the duration and theme-repeat probabilities, the start distribution, the 12 motif prototypes, and the score mean and standard deviation from training.

Run the commands below from the repository root (`panMP_whales`), so Python imports this `analysis` package.

## What the fit measured

The numbers are in [work/train_report.json](work/train_report.json) and in [song_soft_hmm_t6_report.pdf](song_soft_hmm_t6_report.pdf).

| Metric | Meaning |
| --- | --- |
| Sequence log-likelihood | Sum over the 53 songs of the forward likelihood. Higher is a better explanation of the motif scores. |
| Nats per second | That sum divided by 36,000. Compared with 12 independent standard normals, which is what the z-scored scores score when every slot mean stays at zero. |
| Complete-data split | Posterior-weighted emission, transition, and start terms on the saved parameters. |
| Per-song nats per second | The same rate for each song. The min and max show whether one song is carrying the total, or being left out. |
| Posterior mass on the labeled theme | Average forward–backward probability on the theme the CSV wrote at that second. This is not a decoded path. |
| Theme occupancy | Expected seconds in each theme under the posterior, next to the labeled seconds. |
| Labeled phrases per song | How many times the CSV repeats each theme’s slot chain, divided by 53. Set next to the fitted repeat probability. |
| Theme-end and mean sensitivity | How far the sequence log-likelihood falls when one repeat probability or one slot mean is nudged. No refit. |

## Train again

```bash
python -m analysis.HMM.run_train
```

That reuses the GPL windows in `analysis/HMM/work/gpl_193500/` and the motif scores in `analysis/HMM/work/motif_scores.npz` when those files exist, then overwrites `song_soft_hmm_t6.npz`.

To recompute the metrics for the parameters already on disk, without another Baum–Welch run:

```bash
python -c "from analysis.HMM.run_train import refresh_metrics; refresh_metrics()"
```

The notebook [train_song_soft_hmm_t6.ipynb](train_song_soft_hmm_t6.ipynb) calls the same training entry and prints the metrics.
