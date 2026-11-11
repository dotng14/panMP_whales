## Limited-overlap dictionary

```
counts ← 0 on every time column
dictionary ← empty

for pass = 1 .. max_outer:
    eligible ← detection crops with width in [w_min, w_max]
               and every column still uncovered
    if eligible is empty: stop
    seeds ← m crops drawn from eligible, each divided by its norm

    for round = 1 .. refine_iters:
        pursuit(seeds)
        drop seeds with fewer than n_min placements
        replace each remaining seed by the leading principal component
            of its patches, then divide by its norm
        drop placements with unexplained fraction r > tau
        drop seeds that now have fewer than n_min placements
        counts ← cover counts of the placements still kept

    add the surviving seeds and their placements to the dictionary
    if none survived: stop
    if fraction of columns with count ≥ 1 > min_coverage: stop

pursuit(motifs):
    repeat:
        best ← admissible placement with the largest explained fraction
        if none exists: return
        if this is not the first placement of this pursuit
           and its explained energy < eps × the first placement's: return
        keep best and add 1 to each column it covers
```

Learn rank-1 motifs on GPL whitened spectrogram windows. The loop is the same shape as a randomized dictionary build: sample seed motifs from detection crops, place them, rebuild each motif from the patches that selected it, and repeat on what is still available. The difference is placement. A placement does not subtract from the spectrogram. It blocks the columns it covers: a time column is covered by at most one placement, so its cover count is 0 or 1.

The spectrogram is never changed. The same peak therefore stays the best match until one of its columns is covered.

### Data

- Input: whitened windows, each `(frequency, time)`, and detection intervals `(window id, start column, end column)`. A motif never crosses a window boundary.
- Motif: `(F, W)`, stored at unit norm. `W` is its width in time columns. Motifs span all frequency rows.
- Placement: motif id, window id, start column `t`, amplitude `α`.
- Cover count: one integer per time column, initially 0. A placement of width `W` at `t` covers `[t, t+W)` and, while it is kept, contributes 1 to each of those columns. Counts are always recomputed from the placements still kept.

### Knobs

- `eps`. Stop a pursuit when the best explained energy falls below `eps` times the explained energy of that pursuit's first placement. Default 0.01.
- `w_min`, `w_max`. Inclusive width range of a detection crop that may become a seed. Default 10 and 40.
- `m`. Number of seed motifs sampled per pass. Default 10.
- `n_min`. Minimum number of placements a motif must keep. Default 10.
- `refine_iters`. Rebuild rounds inside one pass. Default 1.
- `tau`. Drop a placement whose unexplained fraction exceeds `tau`. Default 0.9.
- `max_outer`. Maximum number of passes. Default 50.
- `min_coverage`. Stop when the fraction of time columns with count at least 1 exceeds this. Default 0.5.

### Admission and score

Start `t` is admissible when `[t, t+W)` lies inside the window and every column there has count 0.

On the original spectrogram, let `X` be the patch at that start and `A` the unit-norm motif. `α = ⟨X, A⟩`. The explained fraction is `α² / ‖X‖²`. The unexplained fraction is `r = 1 - α² / ‖X‖²`. Among admissible starts, take the one with the largest explained fraction.

### Pursuit

Given motifs and the current counts:

1. Take the admissible placement with the largest explained fraction.
2. Stop if none exists, or if its explained energy `α²` is below `eps` times the explained energy of the first placement in this pursuit.
3. Record it and increment the covered columns.
4. Repeat.

### Create

A detection crop is eligible when its width is between `w_min` and `w_max` (inclusive) and the whole crop is admissible. Sample `m` eligible crops without replacement. Each sample divided by its norm is a seed motif. Stop the outer pass if none are eligible.

### Refine

Repeat `refine_iters` times:

1. Run a pursuit with the new motifs. Counts already produced by earlier passes stay occupied.
2. Drop a motif with fewer than `n_min` placements.
3. Replace each survivor by the leading principal component of its accepted patches, then renormalize.
4. Drop placements with `r > tau`, then drop motifs that fall below `n_min`. Recompute counts from the placements still kept.

### Outer loop

Up to `max_outer` passes. Each pass creates, refines, and merges the survivors into the accumulated dictionary. Coverage is the fraction of time columns with count at least 1. Stop when coverage exceeds `min_coverage`, when `max_outer` passes finish, when a pass keeps no motif, or when no seed is eligible.

### Output

Motifs, kept placements, per-column cover counts, and coverage. Every cover count is 0 or 1.
