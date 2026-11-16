# Theme motif probabilities

Motif probabilities of the fixed four-theme model in `results/theme_model.npz`, fitted on the joined wav 2 and wav 3 placements (27 April 2023, 01:11:15–02:26:15 UTC). Each theme’s distribution is a Dirichlet-smoothed multinomial over the 35 dictionary motifs (`alpha = 0.3`).

## Top motifs of each theme

| Rank | Theme 0 motif | Theme 0 *p* | Theme 1 motif | Theme 1 *p* | Theme 2 motif | Theme 2 *p* | Theme 3 motif | Theme 3 *p* |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 21 | 0.1460 | 32 | 0.1118 | 43 | 0.1373 | 1 | 0.0785 |
| 2 | 20 | 0.0762 | 49 | 0.1060 | 1 | 0.1206 | 32 | 0.0757 |
| 3 | 19 | 0.0529 | 0 | 0.0768 | 44 | 0.0589 | 33 | 0.0730 |
| 4 | 46 | 0.0519 | 46 | 0.0651 | 31 | 0.0555 | 35 | 0.0646 |
| 5 | 33 | 0.0468 | 21 | 0.0564 | 27 | 0.0522 | 49 | 0.0619 |
| 6 | 26 | 0.0458 | 24 | 0.0476 | 21 | 0.0489 | 39 | 0.0591 |
| 7 | 24 | 0.0408 | 41 | 0.0447 | 24 | 0.0389 | 22 | 0.0563 |
| 8 | 38 | 0.0387 | 18 | 0.0447 | 28 | 0.0322 | 21 | 0.0508 |
| 9 | 25 | 0.0377 | 14 | 0.0388 | 40 | 0.0322 | 31 | 0.0480 |
| 10 | 29 | 0.0357 | 31 | 0.0388 | 46 | 0.0305 | 10 | 0.0369 |

## Notes

- **Theme 0:** background / motif-21 family (from merging the two motif-21 neighbors in the five-theme fit).
- **Theme 1:** led by motifs 32 and 49; motif 0 (part of the pile) ranks third.
- **Theme 2:** led by motifs 43 and 1; visits often overlap the pile of motifs 0, 1, and 10.
- **Theme 3:** mixed (1, 32, 33, …); often the weakest margin in a song.

Probabilities within each theme sum to 1 over all 35 motifs.
