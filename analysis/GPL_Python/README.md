# GPL_Python

Object-oriented Python translation baseline of the `GPL_v4` core pipeline (`process_v4.m` + `GPL_v31` call chain), implemented from MATLAB behavior in `GPL_v4`.

## What it does

- Discovers WAV files (single file or directory)
- Builds GPL-like detector parameters from sample rate
- Splits each file into processing windows
- Runs GPL-like STFT + whitening + iterative threshold/crop/merge detector
- Aggregates detections with absolute sample/julian times
- Writes:
  - `Detections_<name>.mat`
  - `Labels_<name>.tlab`
  - `CSV_<name>.csv`
  - whitened spectrogram windows as `.npy` in `spectrogram_npy/` (with sidecar JSON)
- Extracts each call's contours and measures them (`gpl_contour.py`, from `GPL_template.m`,
  `GPL_contour.m`, `GPL_measurements.m`, `GPL_filter.m`), as the parameters ask

### Contours and measurements

Per detection, as MATLAB's `GPL_struct(k)`:

- **Contours**: `cm` (all islands of pixels over `contour_cutoff`), `cm_max` (the strongest
  island), `cm_max2` (the second). Each is kept sparse, as `GPL_sparse.m` keeps it
  (`values` uint16 of `scale`, 1-based column-major `index`, `size`), in the `.mat`'s `calls`.
- **Measurements** of each contour: `duration_sec`, `slope` (Hz/s) with `slope_ci` and
  `slope_y_int`, `freq_bandwidth_hz`, `start_freq_hz` / `end_freq_hz` / `min_freq_hz` /
  `max_freq_hz` / `abs_bandwidth_hz`; per call, `spec_rl` and `spec_noise`.
- **Filter flags** (`filter_v3`): 1 where a measurement passes its limit, 0 where not.

The CSV gets a column per measurement that is on (`cm_slope`, `cm_max_duration_sec`, ...,
flags as `filter_cm_slope`); `start_freq_hz` ... `max_freq_hz` come from `cm`, as MATLAB's
CSV does.

What is computed is set by the **parameters**: `DetectorParams.contour` (switches
`template_on`, `cm_on`, `cm_max_on`, `cm_max2_on`, `measurements_on`, `filter_on`;
`measure.*` per contour; `limits.*` per contour), defaulting to `GPL_parameter_Humpback.m`
(`cm` only; duration, slope, bandwidth and frequency limits; no RL or noise). Override them
with a JSON parameters file, `--params FILE` (`process_v4.py`, `preprocess.py`) or
`$GPL_PARAMS_FILE` (the web backend); `gpl_params.example.json` lists every key with its
default.

Differences from MATLAB, all in `gpl_contour.py`'s docstring: indices are uint32 (MATLAB's
uint16 saturates at 65,535, which a 200 kHz call passes after about 40 columns); the
slope filter tests each call's own slope (MATLAB tests the first call's against the upper
limit); only `island_radius = 1` is ported.

Downstream matching (`searchingForMatches`) uses GPL detection `start_time` / `end_time` from `CSV_*.csv` as **segment boundaries** (one detection = one segment) when `segmentation_mode=gpl`.

## Run

On LeiData (auto-detects ``shared_drive/LeiData`` or Google Drive):

```bash
cd searchingForMatches
python scripts/run_gpl_python_leidata.py
```

Or manually:

```bash
python3 "GPL_Python/process_v4.py" "/path/to/input.wav" --output-dir "GPL_Python_outputs"
```

Custom frequency band (Hz):

```bash
python3 "GPL_Python/process_v4.py" "/path/to/input.wav" \
  --output-dir "GPL_Python_outputs" \
  --freq-lo 100 --freq-hi 3000
```

Clip bright outliers before export (legacy MATLAB-style 99.9th percentile):

```bash
python3 "GPL_Python/process_v4.py" "/path/to/input.wav" \
  --intensity-clip-hi-percentile 99.9
```

Load exports from Python or notebooks via `gpl_npy_io`:

```python
from gpl_npy_io import load_display, load_sidecar, imshow_extent
```

### MATLAB parity connector (GPL_v4 / R2004-compatible)

Compare GPL_Python checkpoints against legacy MATLAB `GPL_v4`:

```bash
# Set MATLAB_R2004_EXECUTABLE or pass --matlab for Release-14-era installs.
python searchingForMatches/scripts/run_gpl_matlab_parity.py /path/to/input.wav

# Python-only checkpoint export (no MATLAB required)
python searchingForMatches/scripts/run_gpl_matlab_parity.py /path/to/input.wav --python-only
```

From Python:

```python
from matlab_connector import Matlab2004Connector, ParityConfig

connector = Matlab2004Connector()
mat_path = connector.run_parity_export(
    ParityConfig(wav_path=Path("input.wav"), out_mat_path=Path("parity_matlab.mat"))
)
parity = connector.load_parity(mat_path)
```

Configs use `.mat` by default (no `jsondecode`); JSON is available for modern MATLAB via `--json-config`.

Or folder mode:

```bash
python3 "GPL_Python/process_v4.py" "/path/to/wav_folder" --output-dir "GPL_Python_outputs"
```

## Notes

- This is a **core translation baseline** focused on orchestration, whitening, detection loop, and exports.
- STFT and frequency-band cropping follow `GPL_v4/GPL_fft.m` and `GPL_parameter_Humpback.m`: full `fftl` spectrogram rows, `linspace` crop indices (1-based `FreqBinLo`/`FreqBinHi`), and a separate `arange` grid for noise-level `K`.
- Contour extraction, measurements and the contour filter flags are ported (above). Deck-test jumps, waveform export (`ww365`), `radial_island` and the legacy analysis scripts are not.
- `.npy` output naming:
  - `spectrogram_npy/spwhiten_<wav_stem>_w<window>_s<start>_e<end>.npy`
  - `spectrogram_npy/spwhiten_*_display.npy` — cube-root display transform for plotting
  - paired metadata JSON with frequency bounds (`start_freq`, `end_freq` from `--freq-lo` / `--freq-hi`, default 10–10000 Hz)
  - intensities clipped to `[0, p]` before save (`p` = `--intensity-clip-hi-percentile`, default 100 = max, no upper trim); display `.npy` and `.bmp` are derived from the clipped raw export

## Downstream: LeiData template discovery

After `process_v4.py` writes whitened spectrograms, run the LeiData pipeline:

```bash
cd searchingForMatches
python scripts/leidata_run.py all --jobs 8
```

See [`searchingForMatches/docs/leidata_workflow.md`](../searchingForMatches/docs/leidata_workflow.md).

Legacy OO CLI (`old/run_leidata_analysis.py`):

```bash
python old/run_leidata_analysis.py \
  --display-npy-dir "/path/to/spectrogram_npy" \
  --output-dir "match_outputs/leidata_run" \
  --jobs 8
```

### Common options

| Flag | Default | Meaning |
|------|---------|---------|
| `--display-npy-dir` | (required) | Directory with `spwhiten_*.npy` windows |
| `--output-dir` | (required) | Root for `summary.txt`, `figures/`, peel artifacts |
| `--template-source-window` | `3` | Window used to seed the first template |
| `--template-set-idx` | `1` | Similarity set for the seed median |
| `--zncc-fraction-of-max` | `0.5` | Peak threshold as fraction of max ZNCC |
| `--zncc-zscore-min` | `3.0` | Z-score gate on local ZNCC peaks |
| `--no-zncc-zscore` | off | Disable z-score gating |
| `--peel-target-coverage` | `0.90` | Stop when this fraction of pixels is masked |
| `--no-peel-target` | off | Peel until max rounds or no detections |
| `--peel-max-rounds` | `100` | Upper bound on peel iterations |
| `--jobs` / `-j` | all cores | Parallel workers for load, cross-window ZNCC, masking |
| `--skip-discovery` | off | Run peel only (discovery must exist or peel re-discovers each round) |
| `--skip-peel` | off | Run single-pass discovery only |
| `--exploratory` | off | Also run segment analysis plots on the source window |

### Outputs

Under `--output-dir`:

- `summary.txt` — run configuration, discovery stats, peel coverage and audits
- `figures/` — PNG plots (template gallery, coverage curve, peak audit, histograms)
- `template_discovery/` — first-pass cross-window + hard-joint EM artifacts
- `iterative_peel/` — per-round templates, masked spectrograms, `peel_summary.json`

Example without z-score gating (legacy CLI):

```bash
python old/run_leidata_analysis.py \
  --display-npy-dir "/path/to/spectrogram_npy" \
  --output-dir "match_outputs/leidata_no_zscore" \
  --no-zncc-zscore \
  --jobs -1
```

Discovery-only (legacy CLI):

```bash
python old/run_leidata_analysis.py \
  --display-npy-dir "/path/to/spectrogram_npy" \
  --output-dir "match_outputs/leidata_discovery" \
  --skip-peel
```

More detail: [`../searchingForMatches/README.md`](../searchingForMatches/README.md).

