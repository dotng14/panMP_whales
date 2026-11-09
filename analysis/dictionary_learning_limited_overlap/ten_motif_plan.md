# Twenty-motif dictionary with a residual gap

Build a 20-motif, 5-component limited-overlap dictionary on the 014845 recording whose placement counts stay within a factor of 4, coverage is at least 0.7, and every motif beats the runner-up by at least 0.1 unexplained fraction on average.

The five targets conflict on the dictionary we already have. After replacing motif 8 with 32 coefficient-halves and refining twice, coverage is 0.761 with 35 motifs, but only 8 motifs have a mean residual gap of at least 0.2, and those 8 have placement counts from 10 to 50 (a factor of 5) and cover far less than 0.7. The motifs that supply the coverage are near-duplicates, so their gaps are about 0.03–0.09. The search below keeps a candidate only when all five checks pass.

## Acceptance test

For each kept placement, score every motif at that start time on a patch of that motif's own width. The residual is the unexplained fraction `‖patch − reconstruction‖² / ‖patch‖²`, using the stored mean plus 5 components (`learn.py` `_fit`). For each motif, average `second_best − this_motif` over its own placements.

Accept the dictionary only when all of these hold:

- exactly 20 motifs, each with 5 components
- `max(n_placements) / min(n_placements) ≤ 4`
- coverage ≥ 0.7
- every motif's mean gap ≥ 0.1

## Methods combined

- Limited-overlap create, pursuit, 5-component rebuild, and `tau` filter from `learn.py`, on the 014845 GPL output, with `k=1`, widths 10–40, `tau=0.4`, `n_components=5`.
- The coefficient split used on motif 8: repeated median halves of a motif's placements along its component coefficients. Use it only to rebalance a motif that is already well separated from the others. Do not emit the 32-way split; that produced the small gaps.
- The binary split from the removed `analysis/dict_learning/code/learn/split_template.py` (deleted on 2026-10-25; the experiments that import it no longer run). Reimplement the algorithm in the limited-overlap package, scored with the 5-component residual instead of rank-1 PC1: for each of K trials, draw a perturbation `ε` with `‖ε‖ = 0.1 ‖v‖`, start from `v±ε`, assign the motif's patches to the child with higher explained energy, then refit and reassign for S steps. Keep the trial with the smallest residual SSE.

## Search

Start from a fresh limited-overlap run that stops once coverage exceeds 0.7. Then repeat a bounded number of edits. After every edit, run two refine rounds (pursuit, 5-component rebuild, drop placements with unexplained fraction above 0.4, drop motifs under `n_min`) and recompute the five checks.

- More than 20 motifs, or any motif with mean gap below 0.1: merge the motif into the motif that is most often its runner-up, so the near-duplicate stops competing.
- Fewer than 20 motifs, or the largest motif has more than 4 times the placements of the smallest: split the largest motif. Try the binary `ε` split first. If either child would still sit within 0.1 of another motif, try one median split along the component that most separates the patches. Keep a split only when both children survive the gap check after refine.
- If neither merge nor split improves the worst failing check, stop and report which constraint still fails. Do not relax a target to force a save.

`n_min` stays 10 during refine so a split child is not deleted before it can be judged. The factor-of-4 check, not `n_min`, is what rejects a tiny motif.

## Output

Write the accepted dictionary under `/home/yfreund/whale_processing/result/LeiData/` without overwriting the existing limited-overlap runs. Point `notebooks/014845_limited_overlap.ipynb` at it and re-execute. The notebook should show the five checks: motif count, components, placement counts, coverage, and per-motif mean gap.
