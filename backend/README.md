# backend/ — spectrogram viewer API

A minimal FastAPI app: pick a WAV (from a library folder or upload), run it
through the GPL whitening pipeline, and view the resulting spectrogram in the
browser. No dictionary learning, repeat detection, or review tooling — just
the spectrogram.

## Flow

1. **Pick a file** — `GET /api/library/roots` + `GET /api/library/browse`
   list WAVs under `WAV_LIBRARY_ROOTS` (configured in `.env`), or the
   frontend can upload one directly.
2. **Process** — `POST /api/jobs/from-library` (or `POST /api/jobs` for an
   upload) creates a job and runs `analysis.GPL_Python.process_v4.ProcessingOrchestrator`
   on it in the background. For library files, a pre-existing `.gpl` cache
   (see `app/services/gpl_cache.py`, built by `preprocess.py`) skips straight
   to "complete" instead of reprocessing.
3. **Watch progress** — `GET /api/jobs/{id}/events` is a WebSocket that
   reports status until the job completes.
4. **View** — `GET /api/jobs/{id}/panel` serves the rendered `panel.bin`
   (spectrogram magnitude + energy arrays); `GET /api/jobs/{id}/audio`
   streams the source WAV for playback alongside it.

## Layout

| Path                            | Role                                                             |
| -------------------------------- | ----------------------------------------------------------------- |
| `app/routes/jobs.py`             | Upload/library job creation, panel + audio serving, job events   |
| `app/routes/library.py`          | Library root listing + directory browsing                        |
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
compute, CORS, panel column cap).
