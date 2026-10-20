# Group-mean template similarity — LeiData wav3 dict 189 after drop 6,7,9

## Lineage

| Field | Value |
|---|---|
| parent | **189** (`leidata_wav3_size10`) |
| dropped | **6, 7, 9** (via sequential `remove_template`) |
| pruned dictionary_id | **192** (`dict189_drop_6_7_9`) |
| after DP re-peel | **193** (`dict189_drop_6_7_9_dp`) |
| wav | LeiData **wav3** (`SOCAL_H_76_230427_014845.x`), segmentation 9 |
| windows | first 30 |
| atoms kept | [0, 1, 2, 3, 4, 5, 8] |
| peel | DP non-overlap, λ=0.2, eps_step_fraction=0.2 |

Similarity: \(S_{AB}=\mu(A,B)/\mu(A,A)\) on unit-L2 atoms; **values > 0.8 bold**.

### Coverage (DP matchings on pruned dict)

- windows: **30/30** (100.0%)
- time bins: **3335/18270** (18.3%)
- matchings: **219**
- E(R)/E(S): **0.7259** (energy explained 27.4%)

## Group-mean similarity (DP re-peel)

**Group sizes**

| tid | 0 | 1 | 2 | 3 | 4 | 5 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| |G| | 123 | 25 | 45 | 7 | 6 | 10 | 3 |

**S matrix** (rows = owner A, columns = substitute B; **values > 0.8 bold**)

| A\\B | 0 | 1 | 2 | 3 | 4 | 5 | 8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **0** | **1.00** | 0.19 | 0.14 | 0.12 | 0.06 | -0.03 | 0.00 |
| **1** | 0.13 | **1.00** | 0.02 | 0.02 | 0.28 | -0.02 | 0.01 |
| **2** | 0.05 | 0.05 | **1.00** | 0.02 | 0.02 | 0.00 | 0.00 |
| **3** | 0.05 | 0.03 | 0.01 | **1.00** | 0.01 | -0.00 | -0.00 |
| **4** | 0.14 | 0.41 | 0.03 | 0.12 | **1.00** | -0.03 | 0.02 |
| **5** | 0.02 | 0.20 | 0.02 | 0.00 | -0.00 | **1.00** | -0.02 |
| **8** | 0.03 | 0.00 | 0.10 | -0.05 | -0.11 | 0.06 | **1.00** |

No off-diagonal entries with S ≥ 0.8.

## Suggestions: which templates to remove

**Only criterion:** off-diagonal \(S_{AB} \ge 0.8\) on the DP matrix above.

No removal candidates under \(S \ge 0.8\). Keep all: **0, 1, 2, 3, 4, 5, 8**.

Parent report (pre-drop): `group_similarity_leidata_wav3_dict189.md` / dictionary 189.
