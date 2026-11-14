# Themes and songs from motif placements

Fit a cyclic hidden semi-Markov model to the placements from step 2 and write the segments the frontend draws. The reference fit is the four-theme segmentation of wav 2 and wav 3 joined in time, recorded in `socal_wav2_wav3_themes.md` and in `wav32.csv`.

Wav 2 (`SOCAL_H_76_230427_011115.x`) starts at 01:11:15 UTC on 27 April 2023. That instant is unix time 1682557875. Wav 2 lasts 37.5 minutes. Wav 3 (`SOCAL_H_76_230427_014845.x`) starts when wav 2 ends, so its placements are shifted by 2250 seconds. The joined sequence ends at 75 minutes.

## Constants

| Name | Default | Meaning |
|---|---:|---|
| `n_themes` | 4 | Hidden states. The order is the cycle 0, 1, 2, 3, and then 0 |
| `bin_minutes` | 0.5 | Width of each observation bin, in minutes. 30 seconds |
| `alpha` | 0.3 | Count added to every motif before a theme's motif probabilities are normalized |
| `duration_min_bins` | 2 | Shortest visit, 1 minute |
| `duration_max_bins` | 60 | Longest visit, 30 minutes |
| `duration_sigma_floor` | 0.25 | Smallest standard deviation of log visit length, in bins |
| `likelihood_gap` | 1.92 | A boundary edge is inside the 95% set when its log likelihood is within this of the best edge. This is half of the chi-square 95% value for one parameter (3.841458820694124 / 2) |
| `n_themes_before_merge` | 5 | Themes fit before the two motif-21 neighbors are merged |

The transition into the next theme is certain. After theme `k` comes theme `(k + 1) mod n_themes`. A theme is not followed by itself; the duration distribution decides how long the visit lasts. The first theme is chosen from a uniform start distribution.

## Inputs

The kept placements of step 2 on wav 2 and on wav 3. Each placement has a motif id, a start time, and an end time. Times are seconds from the start of wav 2.

## Algorithm

### Bin the placements

There are `ceil(75 / bin_minutes)` bins, which is 150. A placement whose start falls in a bin adds one to that bin's count for its motif. The motif likelihood of a segment uses these counts. Empty bins do not change the motif likelihood.

### Visit length

A theme's length distribution is lognormal on the number of bins. For lengths `d = 1 ... duration_max_bins`, the unnormalized log probability is

```
-0.5 * ((log(d) - mu) / sigma)^2 - log(d * sigma)
```

Lengths below `duration_min_bins` have probability zero. The remaining values are normalized. `mu` and `sigma` are the mean and standard deviation of the log lengths of that theme's visits. `sigma` is at least `duration_sigma_floor`. A theme with no visits uses `mu = log(6)` and `sigma = 0.5`.

### Motif distribution

A theme's probability of motif `m` is `(alpha + count of m in that theme's visits) / (alpha * n_motifs + total count)`. `n_motifs` is 35 for the reference dictionary.

The log likelihood of a segment with counts `n_m` under theme `k` is `sum_m n_m log p_k,m`. The factorial term of a multinomial is the same for every theme and is not included, because each placement is one draw.

### Fit

Decode with Viterbi on the explicit-duration model. At bin `t` and theme `k`, the best path ends a visit of theme `k` whose length `d` is between `duration_min_bins` and `min(duration_max_bins, t)`. The visit's motif score is the sum of per-bin log probabilities of theme `k` over those `d` bins. The preceding state is theme `(k - 1) mod n_themes`, except for a visit that starts at the first bin, which uses the start distribution.

After each decode, re-estimate the motif probabilities and the length parameters from the visits, and decode again. Keep the path with the highest score.

The published four-theme path is reached as follows.

1. Fit `n_themes_before_merge` = 5 themes, with the same duration limits and the same cyclic transitions. Start from an equal division of the 75 minutes into repeating 15-minute cycles of five themes. The reference search also tried other periods and random initial distributions; the 15-minute start reaches the best score, and the other starts that match it reach the same segmentation.
2. Among those five themes, two neighbors have motif 21 as the motif of highest probability. Merge them into one theme. Join a visit of the second to the visit of the first when they meet. The cycle now has four themes.
3. Re-estimate and decode with `n_themes` = 4 until the path score stops rising. In the reference fit this moves two boundaries: 37.0 minutes becomes 36.5, and 67.0 minutes becomes 66.0.

Label the themes so the decoded path starts at theme 0 and then follows 1, 2, 3.

### Boundaries at the 95% set

The recording starts at 0 and ends at 75 minutes. Those two edges are fixed.

For every other boundary, between a visit of theme `k` on `[L, t)` and a visit of theme `k+1` on `[t, R)`:

1. Estimate `p_k` from the other visits of theme `k`, and `p_{k+1}` from the other visits of theme `k+1`. Both estimates include the prior `alpha`. Leave out the two visits that meet at this boundary.
2. Score every bin edge `c` strictly between `L` and `R` by the log likelihood of counts on `[L, c)` under `p_k` plus counts on `[c, R)` under `p_{k+1}`.
3. The best edge is the one with the highest score. The 95% set is every edge whose score is within `likelihood_gap` of that best score.
4. The boundary used for the segments is the best edge.

### Songs

A new song starts at every visit of theme 0. The song's end is the start of the next theme-0 visit, or the end of the recording for the last song. Song numbers start at 1.

## Outputs for the viewer

Unix times are `1682557875 + seconds from the start of wav 2`.

### Song lanes

Write `<stem>.motif_segments.csv`. The song view reads these columns and no others:

```
motif,song,theme,phrase_id,phrase_count,start_unix,end_unix
```

One row per kept placement. `motif` is the motif id. `song` is the song number. `theme` is 0, 1, 2, or 3, chosen by which theme interval contains the placement's start. `phrase_id` equals `theme`. `phrase_count` is 1, because each theme occurs once in a song. `start_unix` and `end_unix` are the placement's own times. The stem's date and time must parse to 1682557875, which `230427_011115` does.

### Segment table

Write a second CSV with the columns

```
type,segment index,start_unix,end_unix
```

- `type` `motif`: one row per placement. The segment index is the motif id. The times are the placement's start and end.
- `type` `theme`: one row per theme visit. The segment index cycles through 0, 1, 2, 3. The times are the 95% boundaries of that visit.
- `type` `song`: one row per song. The segment index is the song counter, starting at 1. The times are that song's start and end.

The reference file of this shape is `wav32.csv` next to the joined wav. It has 2367 motif rows, 27 theme rows, and 7 song rows.

## Acceptance

The seven songs, in minutes from the start of wav 2:

| Song | Interval |
|---:|---|
| 1 | 0.0–8.0 |
| 2 | 8.0–20.5 |
| 3 | 20.5–28.5 |
| 4 | 28.5–39.5 |
| 5 | 39.5–52.5 |
| 6 | 52.5–65.5 |
| 7 | 65.5–75.0 |

The theme visits inside those songs follow 0, 1, 2, 3, and the boundaries are the best edges of the 95% sets. Theme 2, the theme of motifs 43 and 1, has seven visits, and each one overlaps one pile of motifs 0, 1, and 10. With each visit left out of its own motif distribution, that distribution assigns the visit's counts a higher likelihood than the other three themes do, for all 27 visits.
