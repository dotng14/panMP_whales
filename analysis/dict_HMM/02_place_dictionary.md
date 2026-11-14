# Place a dictionary on a wav file

Score a finished dictionary on a recording and keep the placements. This does not learn new motifs. The reference placements are the ones written for wav 2 (`SOCAL_H_76_230427_011115.x`) and wav 3 (`SOCAL_H_76_230427_014845.x`) from the 35-motif dictionary of step 1.

The spectrogram constants and the time formula are the same as in `01_build_dictionary.md`. Hop is 0.12288 seconds.

## Constants

| Name | Default | Meaning |
|---|---:|---|
| `k` | 1 | Maximum placements on one time column |
| `eps` | 0.01 | Stop when the best explained energy falls below this fraction of the first accepted placement |
| `tau` | 0.4 | Keep a placement only when its unexplained fraction is at most this |
| `n_min` | 1 | Drop a motif with fewer kept placements than this. Learning uses 10; placement uses 1 so a motif that occurs once remains |

## Inputs

- The GPL windows of the wav being scored: spectrogram, `window_id`, `start_sample`, `sample_freq`, `overlap`. The frequency count must equal the dictionary's frequency count (1637).
- For each motif, `atoms/template_XXX.npy` (the components) and `atoms/mean_XXX.npy` (the mean). Every motif of the reference dictionary has a mean. The explained-fraction score uses the mean. A motif loaded without its mean is a different score and will not match the reference placements.

## Algorithm

Cover counts start at zero.

Pursuit, once, over the whole dictionary:

1. For every motif and every window, compute the explained fraction at every start whose width fits in the window. The formula is the one in `01_build_dictionary.md` for a motif that has a mean: patch energy minus the residual of `mean + components`, divided by patch energy.
2. Repeatedly choose the admissible start with the largest explained fraction. A start is admissible when every column of its width has cover count less than `k`.
3. The first accepted placement sets `first_energy`. Stop when the next best explained energy is below `eps * first_energy`, or when no admissible start remains.
4. On acceptance, increment the cover counts and store `motif_id`, `window_id`, `t_start`, the unexplained fraction `1 - explained fraction`, and the leading coefficient.

Then rescore every accepted placement with the same reconstruction used during learning (`01_build_dictionary.md`, Rescore). Replace the unexplained fraction with that rescored value.

Keep a placement when its rescored unexplained fraction is at most `tau`. Then drop any motif that has fewer than `n_min` kept placements.

The start time of a kept placement, in seconds from the start of the wav, is

```
((start_sample - 1) / sample_freq) + t_start * hop
```

Its end time is that start plus `width * hop`.

## Outputs for the viewer

Same directory layout as step 1, written beside the recording rather than over the learning directory:

- `matchings.json` with one object per kept placement: `motif_id`, `window_id`, `t_start`, `width`, `amplitude`, `unexplained`.
- `windows.json` with `start_sample`, `sample_freq`, `overlap` for each window.
- `summary.json` recording the dictionary path, `k`, `eps`, `tau`, `n_min`, the number of motifs, and the number of kept placements.

Do not copy the atoms. The viewer reads the atoms from the dictionary directory and the placements from this directory.

## Acceptance

One pursuit of the 35-motif dictionary:

- Wav 2: 1447 placements before `tau`, 1017 kept, 34 motifs. Motif 23 is absent. Motif 10 is placed once.
- Wav 3: 1416 placements before `tau`, 1350 kept, all 35 motifs.

These counts are one fresh pursuit. They are not the 1336 placements accumulated while the dictionary was learned.
