# backend/ — spectrogram viewer API

A minimal FastAPI app: pick a WAV (from a library folder or upload), pick an
overlay from the catalog (a dictionary's MP placements or a song-structure
CSV), run the WAV through the GPL whitening pipeline, and view the spectrogram
with the overlay lanes under it. Dictionaries are learned offline with the
dict_learning CLIs, not from the web app.

## Flow

1. **Pick a file** — library mode uses the component recording picker
   (`GET /api/library/facets` + `GET /api/library/match` over WAVs under
   `WAV_LIBRARY_ROOTS`). See [`frontend/recording-picker.md`](../frontend/recording-picker.md).
   Upload mode still posts a WAV directly. `GET /api/library/browse` remains
   for legacy directory listing.
2. **Pick an overlay** — `GET /api/library/analyses?fileId=` lists the
   catalog's dictionaries and song-structure CSVs for that WAV's stem. The
   picker sorts them (A–Z, index, time), renames them in place
   (`PATCH /api/catalog/dictionaries/{id}`, `PATCH /api/catalog/csvs/{id}`),
   and has one checkbox per kind to hide dictionaries or CSVs. "Spectrogram
   only" shows no overlay. Upload mode skips this step.
3. **Process** — `POST /api/jobs/from-library` (or `POST /api/jobs` for an
   upload) creates a job and runs either the GPU hybrid song-detector
   (`app/services/hybrid_song_service.py` → `song_detector` with the 2023
   `song_model.bin` dictionary + 4-theme cyclic HMM, for 10 kHz mono PCM16
   Site H WAVs) or `analysis.GPL_Python.process_v4.ProcessingOrchestrator` in
   the background. For library files, a pre-existing `.gpl` cache
   (see `app/services/gpl_cache.py`) skips straight to "complete" instead of
   reprocessing.
4. **Watch progress** — `GET /api/jobs/{id}/events` is a WebSocket that
   reports status until the job completes.
5. **View** —
   - `GET /api/jobs/{id}/panel` serves the overview `panel.bin`
     (spectrogram magnitude + energy arrays); `GET /api/jobs/{id}/audio`
     streams the source WAV for playback alongside it.
   - `GET /api/jobs/{id}/spectrogram-slice?startSec=&endSec=` serves
     frame-aligned full-resolution (`0.123 s/col`, 736-row) tiles assembled
     from 610-frame (`74.9568 s`) cached blocks with block-interpolated
     whitening floors and fixed display scaling (`0.35`–`1.85`) so tiles join
     seamlessly. The frontend (`SpectrogramCanvas`) streams visible tiles and
     prefetches 2–12 tiles ahead based on scroll/playback speed.
   - `GET /api/jobs/{id}/song-structure?csvId=` serves motif / phrase / theme /
     song lanes (from the hybrid `song_structure.json` when `csvId` is omitted,
     or a catalog CSV).
   - `GET /api/jobs/{id}/repeats` (with optional `dictionaryId`) serves
     `RepeatMatches` for the **Repeated Matches** panel below the spectrogram:
     a catalog dictionary when `dictionaryId` is given, or the 2023 model's
     matching-pursuit placements (`placements_*.csv` + `song_model.bin`
     averages, top 200 per motif) when omitted.
   - `GET /api/jobs/{id}/repeat-strip?motifId=` serves the full-resolution
     (`0.123 s/col`) stitched spectrogram strip for one 2023 motif's matches,
     disk-cached under `output/repeat_strips/motif_<id>.bin` and pre-warmed in
     a background thread when `/repeats` is requested.
   - A catalog dictionary overlay on the main spectrogram comes from
     `GET /api/catalog/dictionaries/{id}/placements` (one atom lane, a box per
     placement).
6. **History (frontend)** — `HistoryButton` (next to the file picker) records
   `(file, timeSec, context)` in `localStorage` on user interactions (seek,
   song step, lane click, repeated-match click, or pause) so users can jump
   straight back to any previously observed recording and timestamp.

## Layout

| Path                            | Role                                                             |
| -------------------------------- | ----------------------------------------------------------------- |
| `app/routes/jobs.py`             | Upload/library job creation, panel + audio + high-res slice + repeat-strip serving, job events |
| `app/routes/library.py`          | Library root listing + directory browsing; the overlay picker's list |
| `app/routes/catalog.py`          | `/api/catalog/*`: dictionaries, runs, placements, CSVs, labels, feedback, reindex |
| `app/services/hybrid_song_service.py` | GPU `song_detector` runner, `song_structure.json` builder, seamless 610-frame high-res tile renderer, 2023 `RepeatMatches` + disk-cached `/repeat-strip` builder & pre-warmer |
| `app/services/catalog_service.py` | Adapter over [`database/`](../database/README.md); read-only for reads |
| `app/services/song_structure.py` | Song-structure CSV → lanes (by catalog id, or by WAV stem)       |
| `app/services/pipeline_service.py` | Runs the GPL pipeline for a job in a thread-pool executor      |
| `app/services/gpl_cache.py`      | Per-WAV `.gpl` cache (panel.bin + manifest) beside the library file |
| `app/services/library_service.py` | Library directory listing + file-id resolution                  |
| `preprocess.py`                  | Offline batch cache builder (thin shim over `analysis.GPL_Python.preprocess`), usually run via `scripts/preprocess_cron.sh` |

The spectrogram science itself (whitening, STFT, `panel.bin` format) lives in
[`analysis/GPL_Python/`](../analysis/GPL_Python/README.md); this app only
wires it up to HTTP.

## Config

Copy `env.example` to `.env` and set `WAV_LIBRARY_ROOTS` to the folder(s) you
want to browse (comma-separated). See `env.example` for the rest (GPU
compute, CORS, panel column cap, and the catalog: `CATALOG_DB`,
`CATALOG_RUNS_ROOTS`, `CATALOG_CSV_ROOTS`).

To fill the catalog, register runs as you make them (`--catalog` on the
dict_learning commands) or scan folders:
`python -m database index --db backend/data/catalog.sqlite --runs-root … --csv-root backend/data/song_structure`
(or `POST /api/catalog/reindex`, which scans the configured roots).

The app imports `analysis` and `database` from this checkout only when the
repository root is on `sys.path` (an editable install in this checkout's own
environment, or `PYTHONPATH`). In an environment where another checkout is
installed, both names resolve to that checkout instead.
