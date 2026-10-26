# backend/ — spectrogram viewer API

A minimal FastAPI app: pick a WAV (from a library folder or upload), pick an
overlay from the catalog (a dictionary's MP placements or a song-structure
CSV), run the WAV through the GPL whitening pipeline, and view the spectrogram
with the overlay lanes under it. Dictionaries are learned offline with the
dict_learning CLIs, not from the web app.

## Flow

1. **Pick a file** — `GET /api/library/roots` + `GET /api/library/browse`
   list WAVs under `WAV_LIBRARY_ROOTS` (configured in `.env`), or the
   frontend can upload one directly.
2. **Pick an overlay** — `GET /api/library/analyses?fileId=` lists the
   catalog's dictionaries and song-structure CSVs for that WAV's stem. The
   picker sorts them (A–Z, index, time), renames them in place
   (`PATCH /api/catalog/dictionaries/{id}`, `PATCH /api/catalog/csvs/{id}`),
   and has one checkbox per kind to hide dictionaries or CSVs. "Spectrogram
   only" shows no overlay. Upload mode skips this step.
3. **Process** — `POST /api/jobs/from-library` (or `POST /api/jobs` for an
   upload) creates a job and runs `analysis.GPL_Python.process_v4.ProcessingOrchestrator`
   on it in the background. For library files, a pre-existing `.gpl` cache
   (see `app/services/gpl_cache.py`, built by `preprocess.py`) skips straight
   to "complete" instead of reprocessing.
4. **Watch progress** — `GET /api/jobs/{id}/events` is a WebSocket that
   reports status until the job completes.
5. **View** — `GET /api/jobs/{id}/panel` serves the rendered `panel.bin`
   (spectrogram magnitude + energy arrays); `GET /api/jobs/{id}/audio`
   streams the source WAV for playback alongside it. A CSV overlay comes from
   `GET /api/jobs/{id}/song-structure?csvId=` (motif / phrase / theme / song
   lanes); a dictionary overlay from
   `GET /api/catalog/dictionaries/{id}/placements` (one atom lane, a box per
   placement).

## Layout

| Path                            | Role                                                             |
| -------------------------------- | ----------------------------------------------------------------- |
| `app/routes/jobs.py`             | Upload/library job creation, panel + audio serving, job events   |
| `app/routes/library.py`          | Library root listing + directory browsing; the overlay picker's list |
| `app/routes/catalog.py`          | `/api/catalog/*`: dictionaries, runs, placements, CSVs, labels, feedback, reindex |
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
