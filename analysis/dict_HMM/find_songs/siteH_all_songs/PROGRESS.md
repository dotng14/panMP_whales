# Site H hybrid detection — progress

**Updated:** 2026-11-15 14:35:40  
**Status:** done  
**Mode:** energy (CPU) → song only if any 600 s ratio ≥ 0.15

```
[█████████████████████████████████████████████████·] 100.0%
```

| | |
|---|---:|
| Eligible 10 kHz wavs | 6797 |
| Completed | 6795 |
| Already done at resume | 1719 |
| Done this session | 5076 |
| Failed | 2 |
| Ineligible | 2473 |
| Song detector runs | 4465 |
| Energy-only (song skipped) | 611 |
| Conforming songs so far | 2546 |
| Elapsed this session | 17.57 h |
| Mean s / file this session | 12.5 s |
| ETA | 0.0 h |
| Last file seconds | — |

**Current file**  
`—`

**Last completed**  
`/mnt/sda/SiteH/SOCAL_H_81/SOCAL_H_81_disk10_df20/SOCAL_H_81_260923_004902_df20.x.wav (9.5s, max_ratio=0.464, high_segs=4, +0 songs, ok_song)`

**Last error**  
`/mnt/sda/SiteH/SOCAL_H_71/SOCAL_H_71_disk01_df20/SOCAL_H_71_210526_213500_df20.x.wav`

## Outputs

- `conforming_songs.csv` — conforming songs (extractable spans)
- `energy_segments.csv` — 600 s whitened 50–500 Hz ratios (new files only)
- `manifest.csv` — per-file status (`ok_song` / `ok_energy_only` / `fail`)
- `placements/` — kept when a file yields conforming songs
- `done_stems.txt` — completed stems (song-only history + hybrid)

Reload this file in an editor to watch progress. Restarting `hybrid_run.py`
resumes from `done_stems.txt` and ignores those files.
