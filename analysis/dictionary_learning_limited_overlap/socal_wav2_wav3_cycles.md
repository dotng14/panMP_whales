# Cycles on wav 2 and wav 3

The dictionary is the 35 motifs in `SOCAL_H_76_230427_014845.x.limited_overlap_k1_w40_pc5_tau04_m8x32_refine2`. Each recording is the same length, 37.5 minutes, with the same GPL spectrogram (1637 frequency bins, hop 0.12288 s). The motifs are placed by one pursuit on that spectrogram: a column is covered at most once, and a placement is kept when its unexplained fraction is at most 0.4.

Wav 2 is `SOCAL_H_76_230427_011115.x`. Wav 3 is `SOCAL_H_76_230427_014845.x`, the file the dictionary was learned on. The earlier note [014845_m8x32_refine2_cycles.md](014845_m8x32_refine2_cycles.md) used the placements saved during learning. The numbers here come from placing the finished atoms, so the wav 3 counts differ slightly and the boundary times do not.

A cycle runs from the start of one pile of motifs 0, 1, and 10 to the start of the next pile. A pile is a run of those motifs with no gap longer than 2 minutes. Each cycle is split into five equal parts. A fifth is 20% of that cycle, so its length in minutes follows the cycle. The first fifth starts when the pile starts.

## Wav 3

The pursuit keeps 1350 placements of all 35 motifs. The four piles are the same times as in the learning placements.

| Boundary | Time (min) | Placements of 0, 1, 10 | What is in it |
|---|---|---:|---|
| 1 | 0.0–1.8 | 16 | All three, overlapping |
| 2 | 8.6–14.6 | 30 | Motif 0 opens it, motif 1 runs through it, motif 10 joins at 13.2–14.5 |
| 3 | 23.2–27.9 | 15 | Motif 0, then motif 1. Motif 10 is absent |
| 4 | 32.4–37.4 | 20 | Motif 0, then motif 1. Motif 10 is absent |

The gaps with none of these three are 6.8, 8.7, and 4.4 minutes. The three complete cycles last 8.6, 14.6, and 9.1 minutes.

The other motifs sit where the earlier note put them. Motif 49 is heaviest in the first fifth (22 of its placements in the three cycles). Motif 17 is at the opening of the second boundary only: 16 of its 18 placements are in cycle 2. Motif 43 is just after the marker, 18 of its cycle placements in the second fifth, mostly in cycle 2. Motif 21 runs through the body of all three cycles (24, 22, and 18 placements). Motif 20 peaks in phase 0.6–0.8 (25 placements there). Motifs 19, 40, 30, and 24 peak in the last fifth.

## Wav 2

The pursuit keeps 1017 placements of 34 motifs. Motif 23 is not placed. Motif 10 is placed once.

| Boundary | Time (min) | Placements of 0, 1, 10 | What is in it |
|---|---|---:|---|
| 1 | 1.2–7.6 | 34 | Motifs 0 and 1, overlapping. Motif 10 is absent |
| 2 | 16.2–20.4 | 11 | Motifs 0 and 1. The single motif 10 is at 20.4, the end of the pile |
| 3 | 23.1–28.5 | 23 | Motif 1 throughout, and one placement of motif 0. Motif 10 is absent |
| 4 | 33.6–37.5 | 23 | Motif 0, then motif 1. Motif 10 is absent |

The gaps with none of these three are 8.6, 2.7, and 5.1 minutes. The three complete cycles last 15.0, 6.9, and 10.6 minutes. The first pile is long, 6.5 minutes, so on the 15 minute first cycle it fills the first two fifths. The second cycle is short, and its pile fills most of it.

**With the marker.** Motif 43 repeats at the piles: 19, 7, 22, and 14 placements in the four boundaries. In the three complete cycles, 33 of its placements fall in the second fifth. Motif 44 does the same on a smaller scale, 14 of its cycle placements in that second fifth. Motif 17 does not repeat. All 5 of its placements are in the first cycle.

**Through the body.** Motif 21 is the motif of the recording, with 199 placements. In the gap after the first pile, 7.6–16.2 min, it leads every third: 29, then 40, then 15. In the gap after the third pile, 28.5–33.6 min, it leads again: 13, then 25, then 6. Across the three cycles its fifths are 24, 14, 37, 76, and 30, so the mass is past the middle and it is present the whole way. Motif 3 peaks in the same late-middle fifth as on wav 3: 15 of its cycle placements are in phase 0.6–0.8.

**Just before the next pile.** Motif 46 leans to the end of the gaps. Its cycle fifths are 7, 7, 9, 10, and 25. It is one of the last motifs in the gap after the first pile (9 placements) and in the gap after the third (7).

Motif 20 does not keep the late-middle place it has on wav 3. Its 40 placements are spread across the cycles. Motif 49 is also spread: 11 in the first fifth, 18 in the middle, and 16 in the last, rather than sitting mainly with the marker.
