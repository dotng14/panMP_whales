# Plan: bring back the catalog, and one picker for dictionaries and CSVs

Status: c0–c5 implemented, 2026-10-26 (see Results). Decisions are at the end.

## Results (2026-10-26)

- `database/` (schema, `catalog.py`, `index_existing.py`, `python -m database
  {index,register}`), `--catalog` on both dict_learning commands, backend
  catalog routes, `/api/library/analyses`, CSV loading by catalog id,
  `AnalysisPicker` + `InlineRename` + the atom lane in the viewer.
- Integration gate (now also `database` and the backend catalog modules): 100%,
  113 passed, with GPUs visible and with them hidden.
- All of `analysis/`: 258 passed, 7 skipped (MATLAB), 1 known collection error.
- Frontend: `npm run build` passes. `npm run lint` reports the same rule
  categories as before (set-state-in-effect, react-refresh exports), now also
  in the copied `InlineRename.tsx` and for the two new lane exports. Not
  clicked through in a browser.

## Goal

1. A SQLite catalog in this repo that stores every dictionary and run we make
   (randomized build, matching pursuit), so runs can be kept apart, named and
   compared.
2. The gplrev3 indexing features: scan the WAV library, GPL outputs and run
   folders into the catalog (`python -m database.index_existing`,
   `POST /api/catalog/reindex`).
3. The song-structure CSVs (`*.motif_segments.csv`) indexed in the same catalog.
4. After picking a WAV, one picker lists that WAV's dictionaries **and** CSVs,
   with the gplrev3 sorting (A–Z, index, time), inline rename, the
   seed/refinement badge and the parent link. Two checkboxes in the picker
   header, **Dictionaries** and **CSVs**, hide or show each kind.

## What gplrev3 has, and what we keep

Source: `/home/rivo/gplrev3-webserver` (`database/`, `backend/app/routes/catalog.py`,
`backend/app/services/catalog_service.py`, `frontend/src/components/DictionaryPicker.tsx`,
`InlineRename.tsx`).

| gplrev3 piece | Keep? | Notes |
|---|---|---|
| `wav_files`, `gpl_segmentations`, `windows` | Keep | Indexing of WAVs, GPL output folders and windows, with absolute times |
| `learning_runs`, `learning_round_logs` | Keep, adapt | One row per randomized build; one log row per outer pass (`outer_logs.jsonl`) |
| `templates`, `dictionaries`, `dictionary_templates`, `dictionary_windows` | Keep | `source` stays `seed` / `refinement` / `matching` / `edit`, with the lineage check |
| `matchings`, `matching_feedback` | Keep | MP placements per dictionary; good/bad feedback API kept for later review UI |
| `tool_runs` | Keep, new tool list | `randomized_build`, `matching_pursuit` |
| Labels (`label`, `named`, `displayName`), template labels | Keep | Used by the picker's rename |
| `list_dictionaries_for_wav` and the `/api/catalog/*` read routes | Keep | |
| `index_existing.py` (`reindex`) | Keep, extend | Also scans run folders (`summary.json`) and CSV folders |
| `_migrate*` (upgrades of old gplrev3 DBs), `migrate_flat_dict_dirs.py` | Drop | Fresh schema, fresh DB file |
| `persist_*` for removed tools (average, build-from-detections, remove, merge), prune helpers, `action_history` | Drop | Their tools were removed in the baseline |
| `register_learning_run_from_dir` (old `manifest.json` + `round_XXX/` layout) | Replace | New registrar reads the `randomized-build` / `matching-pursuit` output layout |

New:

- **`annotation_csvs`** table: `csv_id`, `wav_id` (nullable), `stem`, `csv_path`
  (unique), `kind` (`motif_segments`), `name`, `label`, `n_rows`, `n_songs`,
  `created_at` (file mtime), `indexed_at`, `size_bytes`.
- **`register_run_dir(conn, out_dir)`**: reads `summary.json`, `atoms/`,
  `matchings.json`, `outer_logs.jsonl`; links the WAV and windows through the
  GPL folder named in `summary.json`. A randomized build becomes one
  `learning_run` + one `seed` dictionary + its matchings. A matching-pursuit run
  becomes a `matching` dictionary whose parent is the dictionary it used.
  Re-registering the same folder replaces its rows (idempotent).
- **CLI**: `--catalog DB` on `randomized-build` and `matching-pursuit` registers
  the output after writing it; `python -m database register DIR... --db DB` and
  `python -m database index --db DB --wav-root … --runs-root … --csv-root …`
  for existing folders.

## Layout

```
database/                 repo root, as in gplrev3
├── schema.sql
├── catalog.py            connect, upserts, register_run_dir, CSV upserts, list/query, labels
├── index_existing.py     reindex(): WAV roots, GPL outputs, run roots, CSV roots
├── __main__.py           python -m database {index,register}
└── README.md
backend/app/routes/catalog.py           /api/catalog/* (reads, label PATCH, feedback, reindex)
backend/app/services/catalog_service.py
backend/app/routes/library.py           + GET /api/library/analyses?fileId=
backend/app/services/song_structure.py  load by csv_id (path from catalog); stem lookup kept as fallback
frontend/src/components/AnalysisPicker.tsx   (from DictionaryPicker) dictionaries + CSVs
frontend/src/components/InlineRename.tsx     (copied)
frontend/src/api/catalog.ts
```

Config (`backend/app/config.py`, `env.example`): `CATALOG_DB` (default
`backend/data/catalog.sqlite`, git-ignored), `CATALOG_RUNS_ROOTS`,
`CATALOG_CSV_ROOTS` (default `backend/data/song_structure`).

## Picker

`GET /api/library/analyses?fileId=` returns one list:

```json
{"wavId": 3, "items": [
  {"kind": "dictionary", "id": 12, "name": "…", "label": null, "named": false,
   "displayName": "…", "source": "seed", "parentId": null, "createdAt": "…",
   "nTemplates": 9, "nMatchings": 412},
  {"kind": "csv", "id": 4, "name": "SOCAL_H_76_….motif_segments.csv", "label": "HMM v1",
   "named": true, "displayName": "HMM v1", "createdAt": "…", "nRows": 980, "nSongs": 7}
]}
```

- Rows get an icon per kind (📚 dictionary, 📄 CSV), the `#id` (with `← #parent`
  for dictionaries), the name with inline rename, a badge (`seed` / `refinement`
  / `matching` / `csv`), counts, and the timestamp.
- Sorting as in gplrev3: A–Z (named first), Index, Time; applied across both
  kinds together. Index sorts by kind, then id (ids are per table).
- Header checkboxes **Dictionaries** and **CSVs** filter the list; both start
  ticked; remembered in `localStorage`.
- Rename: `PATCH /api/catalog/dictionaries/{id}` and `PATCH /api/catalog/csvs/{id}`.

## Tests

- Unit (`analysis/test/unit/database/`, or `database/tests/`): port gplrev3's
  `test_catalog.py` and `test_catalog_api.py`, minus dropped features; new tests
  for `register_run_dir`, CSV upserts, `list_analyses_for_wav`, sorting fields.
- Integration: `randomized-build --catalog`, `matching-pursuit --catalog`,
  `python -m database index/register` on the synthetic fixtures. Add `database`
  to the 100% gate in `integration/.coveragerc`.
- Backend: FastAPI `TestClient` tests for `/api/library/analyses` and the label
  PATCH routes.
- Frontend: `npm run build` and lint only (no frontend test runner exists).

## Steps

| # | Step |
|---|---|
| c0 | Copy `database/` from gplrev3; strip migrations and removed-tool code; new tool list; `annotation_csvs` |
| c1 | `register_run_dir` + `--catalog` on both CLIs + `python -m database` |
| c2 | Indexer: run roots and CSV roots |
| c3 | Backend: catalog router, service, `/api/library/analyses`, CSV loading by id |
| c4 | Frontend: `AnalysisPicker`, `InlineRename`, App flow (WAV → picker → view) |
| c5 | Tests + coverage gate + docs (`database/README.md`, `analysis/test/README.md`, `backend/README.md`, dict_learning README "Removed" section) |

## Deployment note

`database` is a top-level package, like `analysis`. The shared `dev`
environment already has gplrev3's editable install, which provides both. As
with `analysis`, this repo's `database` only wins when this repo's root is on
`sys.path` first (pytest does this; production would need its own venv or
`PYTHONPATH`).

## Decisions (2026-10-26)

1. Choosing a dictionary shows an **atom lane**: one lane under the spectrogram,
   a box per MP placement labelled by atom id, built on the song-structure lane
   code. Choosing a CSV shows the song / theme / phrase / motif lanes as now.
2. The hide boxes are the two **per-kind filters** only (no per-row hide).
3. The catalog stores **randomized-build and matching-pursuit** runs. Pan runs
   stay as files.
4. A **new DB file** for this repo (`CATALOG_DB`, default
   `backend/data/catalog.sqlite`). gplrev3's catalog is untouched.
5. No "New dictionary" (learn from the web UI) row for now; a "Spectrogram only"
   row takes its place.
