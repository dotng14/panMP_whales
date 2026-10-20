# Payne & McVay (1971) — Songs of Humpback Whales

**Full citation:** Payne RS, McVay S. "Songs of Humpback Whales."
*Science* 173(3997):585–597, 13 August 1971.
DOI: [10.1126/science.173.3997.585](https://doi.org/10.1126/science.173.3997.585)
PMID: 17833100

> **Note:** Full text requires AAAS/Science subscription. PDF not included.
> Abstract and hierarchy are reproduced here from the published abstract (open access).

---

## Summary

The founding paper on humpback whale song. Payne & McVay established that
humpback vocalizations are not random but form a **nested, hierarchical, repeating
structure** — the framework used by every subsequent bioacoustics study of the species.

---

## The Song Hierarchy

```
Unit
  │  The shortest continuous, acoustically distinct sound.
  │  Characterized by slope (ascending/descending), frequency range,
  │  duration, and timbre (moan, shriek, groan, cry, etc.).
  ↓
Phrase
  │  A fixed small sequence of units, repeated consistently.
  │  The phrase is the basic repeating building block of a theme.
  ↓
Theme
  │  A succession of similar phrases; each theme has a characteristic
  │  phrase type.  A song contains multiple distinct themes.
  ↓
Song
  │  An ordered sequence of all themes; lasts 7–30 minutes.
  │  The sequence of themes is fixed and consistent within an individual.
  │  No theme is ever completely omitted.
  ↓
Song session
     The same song repeated continuously without pause, for several hours.
     All whales in a population sing the same song type within a season.
```

---

## Key findings (from published abstract)

1. Humpback whales produce sounds for 7–30 minutes and then repeat the same
   series with considerable precision. This performance is called **singing**;
   each repeated series is a **song**.

2. All prolonged sound patterns recorded are in song form. Each individual
   adheres to its own song type.

3. Several song types exist, but individual variation is pronounced; there is
   only a rough species-specific song pattern.

4. Songs are repeated without any obvious pause; **song sessions** may continue
   for several hours.

5. The **sequence of themes** in successive songs by the same individual is the
   same. Although the number of phrases per theme varies, no theme is ever
   completely omitted.

6. Loud sounds (e.g. dynamite blasts) do not disrupt singing.

7. The sex of the singer was unknown at time of publication (later confirmed
   to be exclusively adult males).

8. The function of singing was unknown at time of publication (later evidence
   points to mate attraction and male–male competition).

---

## Song evolution (established in later work building on this paper)

- All males in a population converge on the same song within a season
  (horizontal cultural transmission).
- Songs change gradually but continuously — "progressive change" (Payne & Payne, 1985).
- Occasionally a new song type spreads rapidly across a population — a
  **"song revolution"** (Noad et al., 2000; Garland et al., 2011).
- Different populations sing different song types; within the same ocean basin
  songs can spread between populations.

---

## Relevance to HumpbackSongs project

| Hierarchy level | Detection/classification target | Pipeline stage |
|-----------------|--------------------------------|----------------|
| **Unit** | Individual call detection (Helble GPL) | Stage 4 — Detector |
| **Phrase / Theme** | Sequence structure within an event | Stage 5 — Classifier (future) |
| **Song** | Song presence/absence per 75 s block (Allen CNN) | Stage 5 — Classifier (binary) |
| **Song session** | Seasonal/daily occurrence patterns | Downstream analysis |

The current pipeline targets **unit-level detection** (Stage 4, GPL) and
**song-level presence/absence** (Stage 5, CNN-style classifier). Phrase and theme
classification would require sequence modeling (HMM, transformer) as a future extension.

---

## Related papers in this directory

- [`HebleJASA2012.pdf`](HebleJASA2012.pdf) — GPL unit detector (Stage 4)
- [`fmars.md`](fmars.md) — Allen et al. CNN for song-level detection (Stage 5)
