# Subcycles of the joined wav 2 and wav 3 sequence

Wav 2, `SOCAL_H_76_230427_011115.x`, starts at 01:11:15 and lasts 37.5 minutes. Wav 3, `SOCAL_H_76_230427_014845.x`, starts when wav 2 ends, at 01:48:45. Times below are minutes from the start of wav 2. The join is at 37.5, and the sequence ends at 75.0.

Both files are scored with the 35 motifs of `SOCAL_H_76_230427_014845.x.limited_overlap_k1_w40_pc5_tau04_m8x32_refine2`. Each motif is placed by one pursuit on that file's GPL spectrogram. A column is covered at most once. A placement is kept when the unexplained fraction is at most 0.4. The joined sequence has 2367 placements.

A pile is a run of motifs 0, 1, and 10 with no internal gap longer than 2 minutes. A cycle runs from the start of one pile to the start of the next. The gap is the time after a pile ends and before the next pile starts. A motif leads a stretch when it has strictly more placements there than any other motif. A tie has no leader.

The question is whether each cycle splits into five subcycles whose leading motifs agree across cycles. The pile is the first subcycle, and it already has a different length in each cycle. The other four subcycles are a split of the gap. They may each take a different fraction of their own gap.

## The seven piles

| Pile | Time (min) | File | Placements of 0, 1, 10 | Cycle that follows | Cycle length (min) | Gap after the pile (min) |
|---|---|---|---:|---:|---:|---:|
| 1 | 1.2–7.6 | wav 2 | 34 | 1 | 15.0 | 7.6–16.2 (8.6) |
| 2 | 16.2–20.4 | wav 2 | 11 | 2 | 6.9 | 20.4–23.1 (2.7) |
| 3 | 23.1–28.5 | wav 2 | 23 | 3 | 10.6 | 28.5–33.6 (5.1) |
| 4 | 33.6–39.3 | both | 39 | 4 | 12.5 | 39.3–46.1 (6.8) |
| 5 | 46.1–52.1 | wav 3 | 30 | 5 | 14.6 | 52.1–60.7 (8.7) |
| 6 | 60.7–65.4 | wav 3 | 15 | 6 | 9.1 | 65.4–69.9 (4.4) |
| 7 | 69.9–74.9 | wav 3 | 20 | | | |

Pile 4 crosses the join. Motif 1 is placed at 37.46, still in wav 2, and motif 0 at 37.51, five seconds into wav 3. Motif 10 joins that pile at 38.2–39.3. Motif 10 is also at the end of pile 2 (one placement) and at the end of pile 5 (six placements). The other piles are motifs 0 and 1.

The pile occupies about two fifths to three fifths of its cycle. Motifs 0 and 1 occur in every pile. During the pile, other motifs are placed as well, so 0 and 1 are the motifs that mark the pile rather than the majority of every placement in it.

## How a split is scored

The six gaps and four subcycles in each gap give 24 slots. A shared assignment names one motif for each of the four subcycles. It scores one point when that motif is the leader of the slot. The highest possible score is 24, which would mean the same four leaders, in the same order, in every gap.

## Equal quarters

Cutting every gap into four equal parts scores 9 of 24. The leader of each quarter:

| Gap | Quarter 1 | Quarter 2 | Quarter 3 | Quarter 4 |
|---:|---|---|---|---|
| 1 | 21 | 21 | 21 | tie |
| 2 | 22 | 3 | 46 | tie |
| 3 | 21 | 21 | 21 | 49 |
| 4 | 21 | 21 | 46 | 30 |
| 5 | tie | 33 | 20 | 19 |
| 6 | tie | 38 | tie | 21 |

Motif 21 is the shared leader of the first quarter in 3 gaps and of the second quarter in 3 gaps. The fourth quarter has no motif that leads more than one gap.

## A different fraction in each gap

The four cuts were chosen separately in each gap. Each subcycle had to cover at least a sixth of that gap, so a stage cannot be a single short blip. The motifs tried as leaders were the ones that lead some stretch of a gap: 21, 46, 19, 20, 33, 49, 3, 35, 32, and 40.

The best shared assignment scores 12 of 24, three more than equal quarters. Its four names are motif 21, motif 35, motif 21 again, and motif 46. The extra points are the 21 and 46 slots. Motif 35 is the leader of its slot in one gap.

What those two motifs actually lead, when each is given the longest stretch of its gap on which it is the strict leader, and that stretch is at least a sixth of the gap:

| Gap | Time (min) | Motif 21 | Motif 46 |
|---:|---|---|---|
| 1 | 7.6–16.2 | first 83%, 37% of the 220 placements | last 17%, 20% of the 30 placements |
| 2 | 20.4–23.1 | | last 83%, 17% of the 58 placements |
| 3 | 28.5–33.6 | first 83%, 34% of the 123 placements | last 33%, 18% of the 38 placements |
| 4 | 39.3–46.1 | a long stretch, 11% of the placements | last 75%, 9% of the placements |
| 5 | 52.1–60.7 | | |
| 6 | 65.4–69.9 | a long stretch, 11% of the placements | |

Gaps 1 and 3 share an order: motif 21 for most of the gap, motif 46 at the end. In those two gaps motif 21 is about a third of the placements in its stretch. In gaps 4 and 6 it is the strict leader of a long stretch only because every other motif is rarer. It is 11% of the placements there.

Gap 5 does not follow that order. Its last fifth is led by motif 19, which is 12 of 64 placements there, 19%. Gap 6 ends on motif 21.

Naming the two middle subcycles 33, 35, or 20 raises the score in one gap and lowers it in the others. None of those motifs leads the same middle slot in more than one or two gaps.

If every subcycle must be at least a quarter of the gap, four subcycles fill the gap and the fractions can no longer move. The score is then the equal-quarter score, 9 of 24.

## Result

Letting the subcycles take different fractions raises the score from 9 of 24 to 12 of 24. The gain is the motif-21 stretch and the motif-46 ending in gaps 1 and 3. The other four gaps do not share that order, and no choice of the two middle leaders repeats across the six gaps. The pile remains the only subcycle whose motifs are the same in every cycle.
