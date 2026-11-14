# Spring energy filter

An energy filter for whale recordings. This first pass selects a random sample of spring files, band-passes each file, measures energy in fixed segments, and summarizes the result in a notebook. The filter code is C and uses the CPUs and both GPUs on this machine.

## Recordings

Source root is `/mnt/sda/SiteH`. A file is spring when the `YYMMDD` stamp in its name has a month in March, April, or May. That set contains 1774 `df20` files (10 kHz, 16-bit mono, about 1.6 GB and 22 hours each) and 257 `df100` files.

Use only `df20`. Those files are sampled at 10 kHz, so the Nyquist frequency is 5 kHz. The requested band is 50–5000 Hz, which on these files is a high-pass at 50 Hz. The `df100` files are sampled at 2 kHz and cannot represent 5 kHz.

The initial run draws 12 files with seed 0, spread across years when the year has a spring file. The count and seed are named constants in this plan and on the command line.

## Filter

Read the `data` chunk of each wav as mono int16 and scale to float.

Band-pass by FFT, in chunks, keeping bins from 50 Hz through the Nyquist bin and zeroing the rest, then inverse FFT. Chunks overlap so the join does not create a step. The same bin mask is the CPU path and the GPU path.

A segment is 60 seconds of the filtered signal. Segment energy is the sum of squares. The mean square is stored too, so a short final chunk stays comparable. A 22-hour file yields about 1300 segments.

## Hardware

This machine has an i9-13900K (32 threads) and two GeForce RTX 3080 Ti GPUs. CuPy can see both GPUs. `nvcc` and `cufft.h` are not on the path, and `fftw3.h` is not installed, although the FFTW shared libraries are.

The C program has two backends that compute the same mask and the same segment energies:

- CPU: FFTW, installed as `libfftw3-dev` if the header is still missing, using the threaded FFTW library across the 32 threads.
- GPU: cuFFT, one file per GPU so both 3080 Ti cards stay busy. If `nvcc` is still absent, install the CUDA toolkit compiler that matches the driver already on the machine, then compile that backend.

Disk reads from `/mnt/sda` are about 100–210 MB/s, so twelve files are a few minutes of reading. The program streams each file in chunks rather than loading 6 GB of float samples at once.

## Outputs

The C program writes one CSV row per segment: site, filename, year, segment index, start time in seconds from the start of the file, energy, mean square.

The notebook `pre_processing/spring_energy.ipynb` reads that CSV and shows:

- energy against time for each sampled file
- the distribution of segment energies, drawn separately for each file and pooled by year

The notebook states the 12-file seed and that the last segment of a file is the start time of that file’s last 60 seconds, not the end of the deployment.
