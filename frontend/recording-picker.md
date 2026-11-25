# Universal site + datetime recording picker

Library mode finds recordings by a **universal key**: **site letter + wall-clock
start time** (`YYYY-MM-DD HH:MM:SS`), not by SOCAL filename chips or folder
trees. A **matching-results list** shows every catalogued WAV that fits
(including df / channel variants that share the same start); clicking a row
selects `{ fileId, name }` and continues to [`AnalysisPicker`](src/components/AnalysisPicker.tsx).

Site is fixed as **`H`** for now. Filename layout (deployment, disk, `df`, …)
stays an implementation detail used only to resolve and disambiguate files.

## UI

| Piece | Role |
|-------|------|
| **Site** | Fixed `H` |
| **When / paste** | Wall-clock start (`2023-04-26` …) or legacy SOCAL stem |
| **Component boxes** | Typeable **Year / Month / Day / Hour / Minute / Second** (same order as `date '+%Y-%m-%d %H:%M:%S'`) |
| **?** help | Site+time model + chip meanings |
| Matching list | Paginated; **startTime** primary, filename secondary; ready badge |

Chips stay so users can browse without typing (e.g. open **Date** → choose `230426` → see every recording that day). When and chips stay in sync when possible.

### Example

```text
When: 2023-04-26 19:35:00
→ same cluster as SOCAL_H_76_230426_193500_df100.x.x.wav / df20 / native
```

## Catalog vs disk (hang-safe)

Picker match/facets read **only SQLite**. They never `rglob` / `stat` /
`listdir` `/mnt/sdb` (or `/mnt/sda`).

| Situation | What to do |
|-----------|------------|
| Disk down / flaky | Ship from existing `catalog.sqlite`; `start_time` backfills from stem `date`/`time` |
| (Re)populate Site H names | [`../backend/scripts/index_siteh.py`](../backend/scripts/index_siteh.py) — **manifest + `skip_stat=True`** |
| Forbidden for Site H | `python -m database index --wav-root /mnt/sdb/SiteH` (in-process `rglob`+`stat` can hang) |

```bash
# Paths come from CSV (legacy /mnt/sda/SiteH prefixes are rewritten to /mnt/sdb/SiteH),
# with no metadata I/O on the volume:
python backend/scripts/index_siteh.py
python backend/scripts/index_siteh.py /path/to/manifest.csv
```

Default manifest:
`/home/yfreund/projects/panMP_whales/analysis/dict_HMM/find_songs/siteH_all_songs/manifest.csv`.

Opening a wav for analysis is a separate step; a missing file fails that job,
not the picker list.

## Code map

| Path | Role |
|------|------|
| [`src/components/RecordingPicker.tsx`](src/components/RecordingPicker.tsx) | Site + When + matching list |
| [`src/lib/recordingWhen.ts`](src/lib/recordingWhen.ts) | Client parse of ISO / SOCAL → site+start |
| [`src/lib/socalWavName.ts`](src/lib/socalWavName.ts) | Legacy stem parse (still used by When paste) |
| [`src/api/library.ts`](src/api/library.ts) | `matchRecordings` (+ optional facets) |
| [`../database/recording_when.py`](../database/recording_when.py) | Canonical site+start helpers |
| [`../database/wav_name.py`](../database/wav_name.py) | SOCAL stem parser |
| [`../database/catalog.py`](../database/catalog.py) | `start_time` column, match filter, SQL backfill |
| [`../backend/app/services/library_service.py`](../backend/app/services/library_service.py) | Match entries include `startTime`; no Site H stat |
| [`../backend/scripts/index_siteh.py`](../backend/scripts/index_siteh.py) | Hang-safe Site H ingest |

## API

Configure roots with `WAV_LIBRARY_ROOTS` in `backend/.env`.

| Endpoint | Purpose |
|----------|---------|
| `GET /api/library/roots` | Root labels |
| `GET /api/library/match?site=H&start=2023-04-26…` | Paginated matches (`fileId`, `name`, `startTime`, …) |
| `GET /api/library/facets?…` | Legacy component facets (still available) |
| `GET /api/library/browse` | Legacy directory browse (unused by this UI) |

Primary filter params: `site`, `start` (full second → equality; shorter →
`LIKE prefix%`). Legacy SOCAL component params still work. Substring `q` for
unparsed paste. Pagination: `offset`, `limit`.

`wav_files.start_time` is `YYYY-MM-DD HH:MM:SS` derived from stem date/time.

## Tests

```bash
python -m pytest analysis/test/unit/database/test_recording_when.py \
  analysis/test/unit/database/test_wav_name.py \
  analysis/test/integration/test_library_match_api.py
```
