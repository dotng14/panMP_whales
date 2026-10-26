-- SQLite catalog for the main application objects:
--   1. WAV file locations                  -> wav_files
--   2. GPL segmentation (set of windows)   -> gpl_segmentations (+ windows)
--   3. Randomized builds                   -> learning_runs (+ learning_round_logs)
--   4. Atoms                               -> templates
--   5. Dictionaries (with lineage)         -> dictionaries (+ dictionary_templates, dictionary_windows)
--   6. MP placements (per dictionary)      -> matchings (+ matching_feedback)
--   7. Song-structure CSVs                 -> annotation_csvs
--
-- Large arrays (atom .npy, spectrogram windows) stay on disk; this DB stores
-- metadata + absolute paths only. Fresh schema for this repo, adapted from
-- gplrev3-webserver's database/schema.sql (see database/README.md).

PRAGMA foreign_keys = ON;

-- 1. WAV locations -----------------------------------------------------------
CREATE TABLE IF NOT EXISTS wav_files (
    wav_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    library_root TEXT NOT NULL,
    rel_path     TEXT NOT NULL,
    abs_path     TEXT NOT NULL UNIQUE,
    stem         TEXT NOT NULL,
    size_bytes   INTEGER,
    mtime        TEXT,
    indexed_at   TEXT NOT NULL,
    UNIQUE (library_root, rel_path)
);

-- 2. GPL segmentation --------------------------------------------------------
-- The on-disk output of GPL_Python for one WAV: a set of windows. `csv_path`
-- points at the GPL detection CSV (CSV_*.csv); its rows are not indexed.
CREATE TABLE IF NOT EXISTS gpl_segmentations (
    segmentation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    wav_id          INTEGER REFERENCES wav_files(wav_id) ON DELETE CASCADE,
    output_dir      TEXT NOT NULL UNIQUE,
    npy_dir         TEXT,
    csv_path        TEXT,
    panel_path      TEXT,
    created_at      TEXT NOT NULL
);

-- A window is a fixed-length segment of one WAV file, indexed within its WAV
-- (`window_index`). `abs_start_time`/`abs_end_time` are wall-clock ISO times
-- (WAV start datetime + sample offset), NULL when the WAV start is unknown.
CREATE TABLE IF NOT EXISTS windows (
    window_id       TEXT PRIMARY KEY,
    segmentation_id INTEGER REFERENCES gpl_segmentations(segmentation_id) ON DELETE CASCADE,
    wav_id          INTEGER REFERENCES wav_files(wav_id) ON DELETE CASCADE,
    window_index    INTEGER,
    npy_path        TEXT NOT NULL,
    json_path       TEXT,
    start_sample    INTEGER,          -- 1-based sample offset within the WAV
    end_sample      INTEGER,
    abs_start_time  TEXT,
    abs_end_time    TEXT,
    overlap         INTEGER,          -- STFT hop in samples
    fftl            INTEGER,
    sample_freq     REAL,
    start_freq      REAL,
    end_freq        REAL,
    n_freq          INTEGER,
    n_time          INTEGER
);

-- 3. Randomized builds -------------------------------------------------------
-- One row per `randomized-build` output folder.
CREATE TABLE IF NOT EXISTS learning_runs (
    run_id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    label                  TEXT,
    out_dir                TEXT NOT NULL UNIQUE,
    gpl_dir                TEXT,
    n_windows              INTEGER,
    max_outer              INTEGER,
    n_outer_completed      INTEGER,
    n_atoms                INTEGER,
    n_matchings            INTEGER,
    energy_before          REAL,
    energy_after           REAL,
    time_coverage_fraction REAL,
    params_json            TEXT,
    status                 TEXT NOT NULL DEFAULT 'completed'
        CHECK (status IN ('running', 'completed', 'failed')),
    stop_reason            TEXT,
    created_at             TEXT NOT NULL,
    registered_at          TEXT NOT NULL
);

-- One row per outer pass (`outer_logs.jsonl`, RandomizedOuterPassLog).
CREATE TABLE IF NOT EXISTS learning_round_logs (
    log_id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id                   INTEGER NOT NULL REFERENCES learning_runs(run_id) ON DELETE CASCADE,
    round_index              INTEGER NOT NULL,
    pool_size                INTEGER,
    n_sampled                INTEGER,
    n_atoms_this_pass        INTEGER,
    n_matchings_this_pass    INTEGER,
    n_pruned                 INTEGER,
    n_distance_filtered      INTEGER,
    n_atoms                  INTEGER,
    n_matchings              INTEGER,
    time_coverage_fraction   REAL,
    window_coverage_fraction REAL,
    n_bins_covered           INTEGER,
    n_bins_total             INTEGER,
    n_windows_matched        INTEGER,
    n_windows_total          INTEGER,
    energy_before            REAL,
    energy_after             REAL,
    UNIQUE (run_id, round_index)
);

-- 4. Tool runs ---------------------------------------------------------------
CREATE TABLE IF NOT EXISTS tool_runs (
    tool_run_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tool                 TEXT NOT NULL
        CHECK (tool IN ('randomized_build', 'matching_pursuit')),
    input_dictionary_id  INTEGER REFERENCES dictionaries(dictionary_id) ON DELETE SET NULL,
    output_dictionary_id INTEGER REFERENCES dictionaries(dictionary_id) ON DELETE SET NULL,
    segmentation_id      INTEGER REFERENCES gpl_segmentations(segmentation_id) ON DELETE SET NULL,
    learning_run_id      INTEGER REFERENCES learning_runs(run_id) ON DELETE SET NULL,
    params_json          TEXT,
    energy_before        REAL,
    energy_after         REAL,
    stop_reason          TEXT,
    out_dir              TEXT NOT NULL UNIQUE,
    created_at           TEXT NOT NULL
);

-- 5. Atoms -------------------------------------------------------------------
-- Immutable arrays on disk. `template_pk` is the stable global id; the natural
-- key `(run_id, template_id)` is kept as provenance.
CREATE TABLE IF NOT EXISTS templates (
    template_pk       INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            INTEGER REFERENCES learning_runs(run_id) ON DELETE CASCADE,
    tool_run_id       INTEGER REFERENCES tool_runs(tool_run_id) ON DELETE SET NULL,
    template_id       INTEGER NOT NULL,
    npy_path          TEXT NOT NULL,
    n_freq            INTEGER,
    width             INTEGER,
    n_components      INTEGER,
    n_patches         INTEGER,
    source_segment_id TEXT
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_templates_run_template
    ON templates(run_id, template_id) WHERE run_id IS NOT NULL;

-- 6. Dictionaries ------------------------------------------------------------
-- A list of templates produced by one operation. source:
--   seed       a randomized build (no parent)
--   refinement a later build seeded from a parent (reserved for Stage 1)
--   matching   matching-pursuit placements of the parent's atoms
--   edit       hand-curated (reserved)
-- `wav_stem` links a dictionary to its WAV even before the WAV is indexed.
CREATE TABLE IF NOT EXISTS dictionaries (
    dictionary_id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name                   TEXT NOT NULL,
    label                  TEXT,             -- user alias; NULL = unnamed
    description            TEXT,
    source                 TEXT NOT NULL,
    parent_dictionary_id   INTEGER
        REFERENCES dictionaries(dictionary_id) ON DELETE CASCADE,
    learning_run_id        INTEGER REFERENCES learning_runs(run_id) ON DELETE CASCADE,
    segmentation_id        INTEGER REFERENCES gpl_segmentations(segmentation_id) ON DELETE SET NULL,
    wav_stem               TEXT,
    out_dir                TEXT UNIQUE,
    params_json            TEXT,
    energy_before          REAL,
    energy_after           REAL,
    time_coverage_fraction REAL,
    stop_reason            TEXT,
    created_at             TEXT NOT NULL,
    CHECK (source IN ('seed', 'refinement', 'matching', 'edit')),
    CHECK ((source = 'seed') = (parent_dictionary_id IS NULL))
);

CREATE TABLE IF NOT EXISTS dictionary_templates (
    dictionary_id INTEGER NOT NULL REFERENCES dictionaries(dictionary_id) ON DELETE CASCADE,
    template_pk   INTEGER NOT NULL REFERENCES templates(template_pk) ON DELETE CASCADE,
    position      INTEGER NOT NULL,
    label         TEXT,             -- user alias for this membership; NULL = unnamed
    PRIMARY KEY (dictionary_id, template_pk)
);

-- The set of windows a dictionary's MP ran over.
CREATE TABLE IF NOT EXISTS dictionary_windows (
    dictionary_id INTEGER NOT NULL REFERENCES dictionaries(dictionary_id) ON DELETE CASCADE,
    window_id     TEXT NOT NULL,     -- soft reference to windows(window_id)
    PRIMARY KEY (dictionary_id, window_id)
);

-- 7. MP placements -----------------------------------------------------------
-- `start_sec`/`end_sec` are seconds from the start of the WAV, computed at
-- registration from the window's sample offset and STFT hop.
CREATE TABLE IF NOT EXISTS matchings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    dictionary_id INTEGER NOT NULL REFERENCES dictionaries(dictionary_id) ON DELETE CASCADE,
    template_pk   INTEGER NOT NULL REFERENCES templates(template_pk) ON DELETE CASCADE,
    window_id     TEXT,
    t_start       INTEGER,
    amplitude     REAL,
    energy        REAL,
    coeffs_json   TEXT,
    start_sec     REAL,
    end_sec       REAL
);

-- One human verdict per placement ('good' | 'bad'), written through the API.
CREATE TABLE IF NOT EXISTS matching_feedback (
    matching_id INTEGER PRIMARY KEY REFERENCES matchings(id) ON DELETE CASCADE,
    label       TEXT NOT NULL CHECK (label IN ('good', 'bad')),
    note        TEXT,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

-- 8. Song-structure CSVs -----------------------------------------------------
-- `*.motif_segments.csv` files (motif, song, theme, phrase_id, phrase_count,
-- start_unix, end_unix). Linked to a WAV by stem.
CREATE TABLE IF NOT EXISTS annotation_csvs (
    csv_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    csv_path    TEXT NOT NULL UNIQUE,
    name        TEXT NOT NULL,
    label       TEXT,                -- user alias; NULL = unnamed
    kind        TEXT NOT NULL DEFAULT 'motif_segments' CHECK (kind IN ('motif_segments')),
    wav_stem    TEXT NOT NULL,
    n_rows      INTEGER,
    n_songs     INTEGER,
    size_bytes  INTEGER,
    created_at  TEXT NOT NULL,       -- file mtime
    indexed_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_windows_segmentation  ON windows(segmentation_id);
CREATE INDEX IF NOT EXISTS idx_windows_wav           ON windows(wav_id, window_index);
CREATE INDEX IF NOT EXISTS idx_templates_run         ON templates(run_id);
CREATE INDEX IF NOT EXISTS idx_round_logs_run        ON learning_round_logs(run_id, round_index);
CREATE INDEX IF NOT EXISTS idx_dictionaries_parent   ON dictionaries(parent_dictionary_id);
CREATE INDEX IF NOT EXISTS idx_dictionaries_run      ON dictionaries(learning_run_id);
CREATE INDEX IF NOT EXISTS idx_dictionaries_stem     ON dictionaries(wav_stem);
CREATE INDEX IF NOT EXISTS idx_dict_templates_dict   ON dictionary_templates(dictionary_id);
CREATE INDEX IF NOT EXISTS idx_dict_templates_tmpl   ON dictionary_templates(template_pk);
CREATE INDEX IF NOT EXISTS idx_dict_windows_window   ON dictionary_windows(window_id);
CREATE INDEX IF NOT EXISTS idx_matchings_dict        ON matchings(dictionary_id);
CREATE INDEX IF NOT EXISTS idx_matchings_dict_tmpl   ON matchings(dictionary_id, template_pk);
CREATE INDEX IF NOT EXISTS idx_matchings_window      ON matchings(window_id);
CREATE INDEX IF NOT EXISTS idx_annotation_csvs_stem  ON annotation_csvs(wav_stem);
