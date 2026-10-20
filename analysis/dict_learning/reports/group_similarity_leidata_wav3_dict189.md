# Group-mean template similarity — LeiData wav3 size-10 dictionary

## Run summary

| Field | Value |
|---|---|
| dictionary_id | **189** |
| name | `leidata_wav3_size10` |
| source | seed |
| parent | None |
| wav | LeiData **wav3** (`SOCAL_H_76_230427_014845.x`) |
| wav path | `/home/yfreund/whale_processing/LeiData/SOCAL_H_76_230427_014845.x.wav` |
| segmentation_id | 9 |
| windows | first 30 |
| atoms | 10: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] |
| catalog matchings | 943 |
| build | `--n-rounds 10 --n-windows 30 --eps-step-fraction 0.2 --min-repeats 3 --min-template-width 12 --max-template-width 24 --template-method pca --refine-iters 2` |
| note | refine aborted (atom left without matches); registered dictionary is the 10-round unrefined PCA result |
| templates on disk | `/home/yfreund/whale_processing/result/LeiData/dict_learning_templates/wav3_size10/` |

Similarity is asymmetric group-mean response correlation: partition matching patches by `template_id`, unit-L2 rank-1 atoms,

$$\mu(A,B)=\frac{1}{|G_A|}\sum_{P\in G_A}\langle T_B,P\rangle,\qquad S_{AB}=\mu(A,B)/\mu(A,A).$$

High \(S_{AB}\) means B scores about as well as A on patches assigned to A. Empty cells (—) mark insignificant self-response.

### Coverage (catalog matchings)

- windows: **30/30** (100.0%)
- time bins: **14892/18270** (81.5%)

### Coverage (DP re-peel, non-overlap, λ=0.2)

- windows: **30/30** (100.0%)
- time bins: **3244/18270** (17.8%)
- matchings: **222**
- E(R)/E(S): **0.7037** (energy explained 29.6%)

## Group-mean similarity (catalog matchings)

Registered learning assignment (may allow residual-stacked placements).

**Group sizes**

| tid | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| |G| | 174 | 133 | 138 | 103 | 130 | 61 | 66 | 54 | 51 | 33 |

**S matrix** (rows = owner A, columns = substitute B; **values > 0.8 bold**)

| A\\B | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0** | **1.00** | 0.21 | 0.16 | 0.12 | 0.06 | -0.02 | **0.81** | 0.31 | 0.00 | 0.23 |
| **1** | 0.37 | **1.00** | 0.42 | 0.07 | 0.33 | 0.01 | 0.10 | 0.32 | 0.02 | 0.12 |
| **2** | 0.13 | 0.08 | **1.00** | 0.05 | 0.03 | 0.02 | 0.18 | 0.43 | -0.00 | 0.20 |
| **3** | 0.37 | 0.14 | 0.12 | **1.00** | 0.05 | 0.00 | 0.33 | 0.16 | -0.01 | 0.08 |
| **4** | **2.31** | **1.11** | **2.60** | 0.20 | **1.00** | 0.30 | **1.33** | **2.32** | -0.03 | **1.36** |
| **5** | 0.26 | 0.10 | 0.48 | 0.00 | 0.01 | **1.00** | 0.23 | 0.56 | -0.09 | 0.23 |
| **6** | **1.03** | 0.08 | 0.35 | 0.11 | 0.11 | -0.05 | **1.00** | 0.66 | -0.06 | 0.56 |
| **7** | 0.43 | 0.16 | **1.33** | 0.10 | 0.01 | 0.04 | 0.35 | **1.00** | -0.07 | 0.27 |
| **8** | — | — | — | — | — | — | — | — | — | — |
| **9** | **1.02** | 0.31 | 0.57 | 0.05 | 0.06 | **0.83** | 0.46 | 0.67 | -0.08 | **1.00** |

**S[A,B] ≥ 0.8 (A ≠ B)**

| A | B | S | |G_A| |
|---:|---:|---:|---:|
| 0 | 6 | 0.814 | 174 |
| 4 | 0 | 2.311 | 130 |
| 4 | 1 | 1.107 | 130 |
| 4 | 2 | 2.604 | 130 |
| 4 | 6 | 1.329 | 130 |
| 4 | 7 | 2.321 | 130 |
| 4 | 9 | 1.359 | 130 |
| 6 | 0 | 1.032 | 66 |
| 7 | 2 | 1.334 | 54 |
| 9 | 0 | 1.022 | 33 |
| 9 | 5 | 0.833 | 33 |

## Group-mean similarity (DP re-peel matchings)

S[A,B] = μ(A,B)/μ(A,A); patches partitioned by matching `template_id` after non-overlap DP peel.

**Group sizes**

| tid | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| |G| | 79 | 24 | 27 | 6 | 5 | 10 | 18 | 26 | 3 | 24 |

**S matrix** (rows = owner A, columns = substitute B; **values > 0.8 bold**)

| A\\B | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **0** | **1.00** | 0.22 | 0.13 | 0.17 | 0.06 | -0.03 | 0.59 | 0.31 | 0.01 | 0.17 |
| **1** | 0.13 | **1.00** | 0.02 | 0.02 | 0.28 | -0.02 | 0.01 | 0.02 | 0.01 | 0.06 |
| **2** | 0.04 | 0.05 | **1.00** | 0.02 | 0.01 | 0.00 | 0.05 | 0.31 | 0.00 | 0.07 |
| **3** | 0.04 | 0.03 | 0.01 | **1.00** | 0.01 | -0.00 | 0.01 | 0.01 | -0.00 | -0.00 |
| **4** | 0.16 | 0.41 | 0.03 | 0.13 | **1.00** | -0.04 | 0.03 | 0.03 | 0.02 | 0.05 |
| **5** | 0.02 | 0.20 | 0.02 | 0.00 | -0.00 | **1.00** | 0.00 | 0.00 | -0.02 | 0.09 |
| **6** | **1.34** | 0.04 | 0.12 | -0.09 | 0.03 | -0.03 | **1.00** | 0.28 | 0.01 | 0.31 |
| **7** | 0.29 | 0.13 | **1.90** | 0.09 | 0.04 | -0.01 | 0.24 | **1.00** | -0.03 | 0.31 |
| **8** | -0.00 | 0.09 | -0.02 | -0.03 | -0.09 | 0.05 | 0.04 | -0.05 | **1.00** | -0.01 |
| **9** | **2.29** | 0.26 | 0.23 | 0.04 | 0.10 | -0.02 | **0.87** | 0.37 | 0.00 | **1.00** |

**S[A,B] ≥ 0.8 (A ≠ B)**

| A | B | S | |G_A| |
|---:|---:|---:|---:|
| 6 | 0 | 1.336 | 18 |
| 7 | 2 | 1.899 | 26 |
| 9 | 0 | 2.295 | 24 |
| 9 | 6 | 0.868 | 24 |

## Suggestions: which templates to remove

**Only criterion:** off-diagonal \(S_{AB} \ge 0.8\) (\(A \ne B\)).  
Interpret as: owner **A** is a **removal candidate**; **B** can stand in for A’s patches. No use of group size, empty rows, or coverage.

**Prefer the DP matrix.** Catalog matchings can be residual-stacked; row **4** is the clear example:

| Matrix | max off-diagonal \(S_{4,B}\) | \(S_{4,B} \ge 0.8\)? |
|---|---:|---|
| **DP re-peel** | 0.41 (B=1) | **no** — do **not** remove 4 from DP alone |
| Catalog | 2.60 (B=2) | yes (6 substitutes) — likely artifact of stacked residual placements (\|G₄\|=130 catalog vs 5 DP) |

### Candidates from DP re-peel (primary)

| Remove A | Substitutes B with \(S_{AB}\ge 0.8\) |
|---:|---|
| **6** | 0 (1.34) |
| **7** | 2 (1.90) |
| **9** | 0 (2.30), 6 (0.87) |

Not nominated on DP (no off-diagonal \(S \ge 0.8\) as owner): **0, 1, 2, 3, 4, 5, 8**.

### Candidates from catalog matchings (secondary)

| Remove A | Substitutes B with \(S_{AB}\ge 0.8\) | Notes |
|---:|---|---|
| **6** | 0 (1.03) | agrees with DP |
| **7** | 2 (1.33) | agrees with DP |
| **9** | 0 (1.02), 5 (0.83) | agrees with DP (B differs slightly) |
| **0** | 6 (0.81) | reverse of 6→0; keep **0**, drop **6** |
| **4** | 0, 1, 2, 6, 7, 9 | **catalog only** — DP falsifies; **do not drop 4** |

### Resolving pairs (DP + catalog when they agree)

| Pair | DP | Catalog | Suggestion |
|---|---|---|---|
| **6 → 0** | \(S_{6,0}=1.34\) | \(S_{6,0}=1.03\) | remove **6**, keep **0** |
| **7 → 2** | \(S_{7,2}=1.90\) | \(S_{7,2}=1.33\) | remove **7**, keep **2** |
| **9 → 0** | \(S_{9,0}=2.30\) | \(S_{9,0}=1.02\) | remove **9**, keep **0** |
| **4 → …** | none \(> 0.8\) | many catalog hits | **keep 4** under DP S-rule |

### Recommended drop set

| Scope | Drop | Keep |
|---|---|---|
| **DP S ≥ 0.8 only** | `6, 7, 9` | `0, 1, 2, 3, 4, 5, 8` |

After prune, recompute S; do not remove further without new off-diagonal \(S \ge 0.8\) evidence.

---

## After removing templates 6, 7, 9

Sequential `remove_template` from dictionary **189** → pruned **192** → DP re-peel registered as **193** (atoms [0, 1, 2, 3, 4, 5, 8]).

Full recalculation: [`group_similarity_leidata_wav3_dict193_drop679.md`](group_similarity_leidata_wav3_dict193_drop679.md).

### DP S (recalculated, values > 0.8 bold)

| tid | 0 | 1 | 2 | 3 | 4 | 5 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| |G| | 123 | 25 | 45 | 7 | 6 | 10 | 3 |

| A\\B | 0 | 1 | 2 | 3 | 4 | 5 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **0** | **1.00** | 0.19 | 0.14 | 0.12 | 0.06 | -0.03 | 0.00 |
| **1** | 0.13 | **1.00** | 0.02 | 0.02 | 0.28 | -0.02 | 0.01 |
| **2** | 0.05 | 0.05 | **1.00** | 0.02 | 0.02 | 0.00 | 0.00 |
| **3** | 0.05 | 0.03 | 0.01 | **1.00** | 0.01 | -0.00 | -0.00 |
| **4** | 0.14 | 0.41 | 0.03 | 0.12 | **1.00** | -0.03 | 0.02 |
| **5** | 0.02 | 0.20 | 0.02 | 0.00 | -0.00 | **1.00** | -0.02 |
| **8** | 0.03 | 0.00 | 0.10 | -0.05 | -0.11 | 0.06 | **1.00** |

No off-diagonal S ≥ 0.8 after prune + re-peel.

No further removal candidates under DP \(S\ge 0.8\).
