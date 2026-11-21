# Component recording picker

The spectrogram viewer’s library mode no longer browses folders. Users **paste a
recording name** (or a partial prefix), the UI **parses it into component chips**,
and empty chips open **cascading dropdowns** of values that still match. A
**matching-results list** shows every WAV that fits; clicking a row selects it
and continues to the existing analysis overlay picker.

SOCAL Site H naming is supported in v1. The parser registry leaves room for
other families (e.g. CHNMS) later.

## UI

| Piece | Role |
|-------|------|
| Paste field | Full name, stem, path basename, or partial (e.g. `SOCAL_H_76_230426`) |
| **?** help | How to write a name, worked example, meaning of each component |
| Component chips | Site, deployment, disk, date, time, df, channel, resample — **Any** until set |
| Chip dropdown | Distinct values (+ counts) still available given the other chips |
| Matching list | Paginated WAVs; size / date / ready badge; click → `{ fileId, name }` |

After selection, [`AnalysisPicker`](src/components/AnalysisPicker.tsx) is unchanged.

### Example

```text
SOCAL_H_76_230426_193500_df100.x.x.wav
└SOCAL┘ └H┘ └76┘ └date─┘ └time─┘ └df─┘ └ch┘
```

| Component | Meaning |
|-----------|---------|
| Site | Hydrophone site letter (`H` for Site H) |
| Deployment | Deployment number (e.g. `76`) |
| Disk | Instrument disk / logger when present (`DL37`, `disk01`, `d03`) |
| Date | Start date `YYMMDD` (UTC) |
| Time | Start time `HHMMSS` (UTC) |
| df | Decimation: `20` ≈ 10 kHz, `100` ≈ 2 kHz, `native` = full rate |
| Channel | `x` or `x.x` (HARP / Triton channel id) |
| Resample | Optional later suffix such as `10kHz` |

Older stems such as `SOCAL45H_DL37_120131_042345.df20.x.wav` parse to the same
fields (deployment glued to the site letter; `df` after a dot).

Some SOCAL30H files on disk have **corrupted basenames** (e.g. `$0H_Q2 $_disk01_…`
instead of `SOCAL30H_Q2_…`). They are listed for substring search but do not
fill the SOCAL chips.

## Code map

| Path | Role |
|------|------|
| [`src/components/RecordingPicker.tsx`](src/components/RecordingPicker.tsx) | Paste, chips, help, matching list |
| [`src/lib/socalWavName.ts`](src/lib/socalWavName.ts) | Client-side SOCAL parse + help copy |
| [`src/api/library.ts`](src/api/library.ts) | `getFacets` / `matchRecordings` |
| [`../database/wav_name.py`](../database/wav_name.py) | Canonical parser (`PARSERS` registry) |
| [`../backend/app/services/wav_name.py`](../backend/app/services/wav_name.py) | Re-export for the API |
| [`../backend/app/services/library_service.py`](../backend/app/services/library_service.py) | Facets / match + fileId encoding |
| [`../backend/app/routes/library.py`](../backend/app/routes/library.py) | HTTP routes |
| [`../backend/scripts/index_siteh.py`](../backend/scripts/index_siteh.py) | Site H catalog fill from a path manifest |

`App.tsx` still expects `onSelectFile({ fileId, name })` as with the old folder
browser.

## API

Configure roots with `WAV_LIBRARY_ROOTS` in `backend/.env`.

| Endpoint | Purpose |
|----------|---------|
| `GET /api/library/roots` | Root labels |
| `GET /api/library/facets?…` | Distinct component values (+ counts) given the other filters |
| `GET /api/library/match?…` | Paginated matching files (`fileId`, name, size, mtime, cached) |
| `GET /api/library/browse` | Legacy directory browse (unused by this UI) |

Filter query params (all optional, AND’d): `family`, `site`, `deployment`,
`disk`, `date`, `time`, `df`, `channel`, `resample`, plus substring `q`.
Pagination: `offset`, `limit`.

Component fields live on `wav_files` in the SQLite catalog (see
[`../database/schema.sql`](../database/schema.sql)). Small library roots are
indexed on demand; **Site H on `/mnt/sda`** should be indexed from the known
path manifest (recursive listing on that volume can hang and return nothing):

```bash
python backend/scripts/index_siteh.py
# optional: path to another CSV with a wav_path column
python backend/scripts/index_siteh.py /path/to/manifest.csv
```

Default manifest:
`/home/yfreund/projects/panMP_whales/analysis/dict_HMM/find_songs/siteH_all_songs/manifest.csv`.

## Tests

```bash
python -m pytest analysis/test/unit/database/test_wav_name.py \
  analysis/test/integration/test_library_match_api.py
```

## Extending to another naming family

1. Add a parser to `database.wav_name.PARSERS` (same `RecordingComponents` fields
   or a documented mapping).
2. Mirror parse + chip labels in `src/lib/` (or choose chips from `family`).
3. Reindex so `wav_files` component columns update.
