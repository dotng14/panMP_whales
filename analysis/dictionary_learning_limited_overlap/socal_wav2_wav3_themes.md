# Themes on the joined wav 2 and wav 3 sequence

Wav 2, `SOCAL_H_76_230427_011115.x`, starts at 01:11:15 and lasts 37.5 minutes. Wav 3, `SOCAL_H_76_230427_014845.x`, starts when wav 2 ends, at 01:48:45. Times below are minutes from the start of wav 2. The join is at 37.5, and the sequence ends at 75.0.

Both files are scored with the 35 motifs of `SOCAL_H_76_230427_014845.x.limited_overlap_k1_w40_pc5_tau04_m8x32_refine2`. Each motif is placed by one pursuit on that file's GPL spectrogram. A column is covered at most once. A placement is kept when the unexplained fraction is at most 0.4. The joined sequence has 2367 placements of all 35 motifs.

## The model

The sequence is a hidden semi-Markov model with four themes. The themes follow one deterministic cycle. Theme 0 is followed by theme 1, theme 1 by theme 2, theme 2 by theme 3, and theme 3 by theme 0. The recording may begin at any theme. Once it has begun, the successor of each theme is fixed.

Theme 0 is the merger of the two neighboring themes whose motif distribution was led by motif 21. In the five-theme cycle those were the theme with motif 21 at probability 0.229 and the theme that followed it, with motif 21 at probability 0.128. Their visits that met were joined, and the motif and length distributions were estimated again on the four-theme cycle. Two cuts moved in that re-estimation: 37.0 became 36.5, and 67.0 became 66.0.

Each theme has two distributions of its own:

- A distribution over the 35 motifs. While the theme is active, each placement is drawn independently from that distribution.
- A distribution over the length of the visit, in minutes. The length is lognormal, truncated to the range from 1 to 30 minutes.

The observations are the placements in successive 30-second bins. A segment that contains counts \(n_m\) of motif \(m\), with \(N = \sum_m n_m\), has likelihood

\[
\frac{N!}{\prod_m n_m!} \prod_m p_{k,m}^{n_m}
\]

under theme \(k\). The factorial depends only on the counts, so the comparison between two themes is \(\sum_m n_m \log p_{k,m}\).

Every motif probability includes a prior count of 0.3, so a motif that does not occur in a theme still has a small probability.

## The four themes

| Theme | Visits | Length (min) | Motifs with probability at least 0.04 |
|---:|---:|---|---|
| 0 | 7 | 1.0–7.5, median 4.0 | 21 at 0.182, 46 at 0.060, 20 at 0.060, 33 at 0.050, 26 at 0.046, 3 at 0.045, 24 at 0.040, 38 at 0.040 |
| 1 | 7 | 1.5–2.5, median 2.0 | 49 at 0.117, 32 at 0.082, 0 at 0.068, 21 at 0.061, 24 at 0.057, 40 at 0.052, 46 at 0.052 |
| 2 | 7 | 1.5–3.5, median 2.5 | 43 at 0.156, 1 at 0.114, 21 at 0.067, 31 at 0.056, 44 at 0.056, 27 at 0.046, 46 at 0.041 |
| 3 | 6 | 1.5–3.0, median 1.8 | 49 at 0.104, 1 at 0.096, 39 at 0.078, 35 at 0.057, 32 at 0.052, 33 at 0.047, 46 at 0.045 |

Theme 0 is the motif-21 theme. Its longest visits are 8.0–15.0 and 52.5–60.0. Theme 2 carries motifs 43 and 1. Each of its seven visits overlaps one pile of motifs 0, 1, and 10, in order: 3.0–6.5, 17.0–18.5, 24.0–27.5, 34.5–36.5, 47.5–50.5, 62.5–64.5, and 72.5–75.0. Theme 1 precedes that pile theme and puts its largest probabilities on motifs 49, 32, and 0. Theme 3 follows the pile and puts its largest probabilities on motifs 49 and 1. Theme 3 occurs six times because the recording ends during theme 2 of the seventh pass.

## The cycle on the recording

| Pass | Theme 0 | Theme 1 | Theme 2 | Theme 3 |
|---:|---|---|---|---|
| 1 | 0.0–1.0 | 1.0–3.0 | 3.0–6.5 | 6.5–8.0 |
| 2 | 8.0–15.0 | 15.0–17.0 | 17.0–18.5 | 18.5–20.5 |
| 3 | 20.5–22.5 | 22.5–24.0 | 24.0–27.5 | 27.5–29.0 |
| 4 | 29.0–33.0 | 33.0–34.5 | 34.5–36.5 | 36.5–39.5 |
| 5 | 39.5–45.5 | 45.5–47.5 | 47.5–50.5 | 50.5–52.5 |
| 6 | 52.5–60.0 | 60.0–62.5 | 62.5–64.5 | 64.5–66.0 |
| 7 | 66.0–70.0 | 70.0–72.5 | 72.5–75.0 | |

The six complete passes last 8.0, 12.5, 8.5, 10.5, 13.0, and 13.5 minutes.

## Likelihood of the counts

For each decoded segment, the motif distribution of its theme was recomputed with that segment removed, and the segment's counts were scored under that distribution and under the other three. The assigned theme gives the higher likelihood for all 27 segments.

The median difference in log likelihood is 20.8, a factor of about \(1 \times 10^9\). Twenty-three segments have a difference larger than 5, a factor of about 150. The four smaller margins are 0.0–1.0 (3.7), 18.5–20.5 (0.7), 27.5–29.0 (4.2), and 60.0–62.5 (2.8).

## The decoded segments

| Time (min) | Theme | Placements | Leading motif | Leave-one-out margin |
|---|---:|---:|---:|---:|
| 0.0–1.0 | 0 | 13 | 21 | 3.7 |
| 1.0–3.0 | 1 | 45 | 32 | 21.8 |
| 3.0–6.5 | 2 | 85 | 1 | 53.0 |
| 6.5–8.0 | 3 | 47 | 49 | 9.0 |
| 8.0–15.0 | 0 | 214 | 21 | 108.8 |
| 15.0–17.0 | 1 | 45 | 49 | 14.1 |
| 17.0–18.5 | 2 | 33 | 43 | 16.1 |
| 18.5–20.5 | 3 | 48 | 49 | 0.7 |
| 20.5–22.5 | 0 | 52 | 46 | 28.4 |
| 22.5–24.0 | 1 | 45 | 21 | 7.1 |
| 24.0–27.5 | 2 | 96 | 21 | 53.5 |
| 27.5–29.0 | 3 | 49 | 1 | 4.2 |
| 29.0–33.0 | 0 | 115 | 21 | 79.7 |
| 33.0–34.5 | 1 | 41 | 21 | 13.9 |
| 34.5–36.5 | 2 | 54 | 43 | 31.3 |
| 36.5–39.5 | 3 | 107 | 49 | 35.3 |
| 39.5–45.5 | 0 | 218 | 21 | 95.9 |
| 45.5–47.5 | 1 | 69 | 49 | 13.6 |
| 47.5–50.5 | 2 | 108 | 43 | 34.2 |
| 50.5–52.5 | 3 | 72 | 33 | 43.7 |
| 52.5–60.0 | 0 | 283 | 20 | 108.6 |
| 60.0–62.5 | 1 | 84 | 40 | 2.8 |
| 62.5–64.5 | 2 | 73 | 1 | 7.4 |
| 64.5–66.0 | 3 | 55 | 35 | 9.7 |
| 66.0–70.0 | 0 | 141 | 21 | 49.1 |
| 70.0–72.5 | 1 | 89 | 0 | 20.8 |
| 72.5–75.0 | 2 | 86 | 31 | 19.5 |

## Precision of the boundaries

There are 26 boundaries, one between each successive pair of visits. The start and the end of the recording are fixed by the files.

For each boundary, the motif distributions of the two themes that meet there were estimated with both of those visits removed. The cut was then moved to every 30-second edge inside the two visits, and each placement in the two visits was scored under the distribution of the theme on its side of the cut. The log likelihood of a cut is the sum of those log probabilities. A cut belongs to the 95% set when its log likelihood is within 1.92 of the best cut. That is the chi-square threshold for one free parameter at 95%. The set is an interval on the 30-second grid. Its width is the distance from its earliest edge to its latest edge, so a set that contains one edge has width 0.

Eighteen boundaries have a 95% set of one edge. Six sets span 0.5 minute, and the sets at 17.0 and at 62.5 span 1.0 minute. Those two are the widest.

The fitted cut is the best cut at 17 boundaries. At the other nine, the best cut is 0.5 or 1.0 minute away. The largest gaps are 5.9 at the fitted cut 64.5, whose best cut is 63.5, and 3.9 at the fitted cut 29.0, whose best cut is 28.5.

A redraw check uses the same held-out distributions. The motif of each placement is drawn again from the theme on its side of the fitted cut, the number of placements in each 30-second bin stays as observed, and the cut is estimated again. In 2000 redraws, the central 95% of the estimated cuts covers at most 1.0 minute at every boundary, and a single edge at 19 boundaries. The fitted cut is the one chosen in 92% to 100% of the redraws.

The columns of the table are:

- **Fitted (min)** is the boundary in the decoded segmentation, in minutes from the start of wav 2.
- **Themes** names the theme before the cut and the theme after it. The arrow follows the cycle.
- **Best cut (min)** is the 30-second edge inside the two visits with the highest log likelihood. The motif distributions used for that score were estimated with both visits removed.
- **Gap** is how far the log likelihood of the fitted cut falls short of the best cut. It is 0 when the fitted cut is the best cut.
- **95% set (min)** lists every 30-second edge whose log likelihood is within 1.92 of the best cut.
- **Width (min)** is the latest edge in the 95% set minus the earliest. A set that contains one edge has width 0.

| Fitted (min) | Themes | Best cut (min) | Gap | 95% set (min) | Width (min) |
|---:|---|---:|---:|---|---:|
| 1.0 | 0 → 1 | 0.5 | 2.2 | 0.5 | 0.0 |
| 3.0 | 1 → 2 | 2.0 | 2.2 | 2.0–2.5 | 0.5 |
| 6.5 | 2 → 3 | 6.5 | 0.0 | 6.5 | 0.0 |
| 8.0 | 3 → 0 | 8.0 | 0.0 | 7.5–8.0 | 0.5 |
| 15.0 | 0 → 1 | 15.0 | 0.0 | 15.0 | 0.0 |
| 17.0 | 1 → 2 | 17.5 | 0.4 | 16.5–17.5 | 1.0 |
| 18.5 | 2 → 3 | 18.5 | 0.0 | 18.5 | 0.0 |
| 20.5 | 3 → 0 | 20.5 | 0.0 | 20.5 | 0.0 |
| 22.5 | 0 → 1 | 22.5 | 0.0 | 22.5 | 0.0 |
| 24.0 | 1 → 2 | 24.0 | 0.0 | 24.0 | 0.0 |
| 27.5 | 2 → 3 | 27.5 | 0.0 | 27.5 | 0.0 |
| 29.0 | 3 → 0 | 28.5 | 3.9 | 28.5 | 0.0 |
| 33.0 | 0 → 1 | 33.0 | 0.0 | 32.5–33.0 | 0.5 |
| 34.5 | 1 → 2 | 34.5 | 0.0 | 34.5 | 0.0 |
| 36.5 | 2 → 3 | 36.5 | 0.0 | 36.5 | 0.0 |
| 39.5 | 3 → 0 | 39.5 | 0.0 | 39.5 | 0.0 |
| 45.5 | 0 → 1 | 45.5 | 0.0 | 45.5 | 0.0 |
| 47.5 | 1 → 2 | 48.0 | 3.4 | 48.0 | 0.0 |
| 50.5 | 2 → 3 | 50.5 | 0.0 | 50.5 | 0.0 |
| 52.5 | 3 → 0 | 52.5 | 0.0 | 52.5 | 0.0 |
| 60.0 | 0 → 1 | 60.5 | 0.4 | 60.0–60.5 | 0.5 |
| 62.5 | 1 → 2 | 62.5 | 0.0 | 62.0–63.0 | 1.0 |
| 64.5 | 2 → 3 | 63.5 | 5.9 | 63.5 | 0.0 |
| 66.0 | 3 → 0 | 65.5 | 0.2 | 65.5–66.0 | 0.5 |
| 70.0 | 0 → 1 | 70.0 | 0.0 | 69.5–70.0 | 0.5 |
| 72.5 | 1 → 2 | 73.5 | 3.6 | 73.5 | 0.0 |
