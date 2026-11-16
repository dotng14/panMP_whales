# Spring energy filter

An energy filter for whale recordings. This first pass selects a random sample of spring files, measures a band energy ratio from the GPL whitened spectrogram in fixed segments, and summarizes the result in a notebook. The filter code is C and uses the CPUs and both GPUs on this machine.

## Recordings

Source root is `/mnt/sda/SiteH`. A file is spring when the `YYMMDD` stamp in its name has a month in March, April, or May. That set contains 1774 `df20` files (mostly 10 kHz, some 16 kHz, 16-bit mono) and 257 `df100` files.

Use only `df20`. The initial run draws 12 files with seed 0, spread across years when the year has a spring file. The count and seed are named constants in this plan and on the command line. The same draw with `--season winter` covers December–February.

## Filter

Read the `data` chunk of each wav as mono int16 and scale to float.

Cut the recording into 75-second GPL windows. In each window:

1. Take a Hamming STFT with GPL defaults for the file’s sample rate (`freq_resolution_hz = 10`, `skip_percent = 75`): at 10 kHz that is fftl 1024 and hop 768; at 16 kHz, fftl 2048 and hop 1536.
2. Keep magnitude rows from 10 Hz to Nyquist.
3. Whiten each row with the densest-half noise level (`base3x` / mean), then smooth with the GPL cross kernel `[[0,1,0],[1,4,1],[0,1,0]] / 8`.

A segment is 600 seconds. Segment **band energy** is the sum of squares of whitened cells whose rows fall in 50–500 Hz. Segment **overall energy** is the sum of squares over all kept rows. The reported value is

```
ratio = band_energy / overall_energy
```

## Hardware

This machine has an i9-13900K (32 threads) and two GeForce RTX 3080 Ti GPUs. The C program has two backends that compute the same whitened ratio:

- CPU: FFTW, one worker thread per recording (up to the core count).
- GPU: batched cuFFT for each window’s frames, one recording per GPU so both cards stay busy; whitening stays on the CPU.

Disk reads from `/mnt/sda` are about 100–210 MB/s. The program streams each file in 75-second windows rather than loading the whole file as floats.

## Outputs

The C program writes one CSV row per segment: site, filename, year, segment index, start time in seconds from the start of the file, band_energy, overall_energy, ratio.

The notebook `pre_processing/spring_energy.ipynb` reads the spring and winter CSVs and shows:

- ratio against time for each sampled file
- the distribution of segment ratios, drawn separately for each file and pooled by year

The notebook states the 12-file seed and that the last segment of a file may be shorter than 600 seconds.
