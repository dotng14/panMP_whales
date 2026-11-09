# Phrase-pair judging (reference group P)

**Page:** https://claude.ai/artifact/2PHkxyNwseVqyTdkcd7dJM (private; share it from the page's Share menu with anyone else who should judge).

## What it shows

- **Queries:** 20 query windows from the test half.
- **Matches:** for each query, the nearest window (at least 5 minutes away) proposed by each of 5 inputs, plus one random window. Matches proposed by more than one input are shown once.
- **Total:** 112 pairs in a shuffled order.
- **Blind:** the page never says which input proposed a pair. That key is only in `work/pairs/key.json`, which is git-ignored.

## Judging

- **Buttons:** Same phrase / Different / Unsure, or keys 1 / 2 / 3; ← and → move between pairs.
- **Saving:** each judge's verdicts are saved in the page's database at `judgments/<judge id>`.

## Reading the judgements back

1. Claude reads the `judgments` collection with the ArtifactData tool and writes `work/pairs/judgments.json` in the form `{judge_id: {pair_id: verdict}}`.
2. Then run:

```bash
python -m analysis.song_hmm.exp3.report
```

The report fills the pair-precision table (group P), reported per input.

## Rebuild

```bash
python -m analysis.song_hmm.exp3.pairs.build
```

The seed is fixed, so it rebuilds the same pairs. Republishing the page keeps the saved judgements.
