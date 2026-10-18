"""Offline GPL pre-processing: science + panel + catalog registration.

Owned by analysis. Writes whitened windows / CSV / ``panel.bin`` via
``process_v4``, indexes them in the catalog (analysis↔backend bus), then runs
``learn_averaged_templates_from_gpl`` and registers the dictionary.

Usage::

    python -m analysis.GPL_Python.preprocess /data/recordings/clip.wav \\
        --db /data/recordings/result/dict_learning_catalog.sqlite \\
        --library-root /data/recordings
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path

from analysis.GPL_Python.gpl_outputs import DEFAULT_INTENSITY_CLIP_HI_PERCENTILE
from analysis.GPL_Python.process_v4 import ProcessingOrchestrator, is_wav_file, normalize_input_path
from analysis.dict_learning.code.core import defaults
from analysis.dict_learning.code.learn.build_dictionary import register_averaged_result
from analysis.dict_learning.code.learn.learn_dictionary import learn_averaged_templates_from_gpl_paths
from database import catalog


@dataclass(frozen=True)
class PreprocessOptions:
    force: bool = False
    freq_lo: float = 10.0
    freq_hi: float = 10000.0
    intensity_clip_hi_percentile: float = DEFAULT_INTENSITY_CLIP_HI_PERCENTILE
    n_windows: int = defaults.N_WINDOWS
    window_offset: int = defaults.WINDOW_OFFSET
    n_rounds: int | None = defaults.N_ROUNDS
    eps_step_fraction: float = defaults.EPS_STEP_FRACTION
    min_repeats: int = defaults.MIN_REPEATS
    normalize: bool = defaults.NORMALIZE
    min_template_width: int | None = defaults.MIN_TEMPLATE_WIDTH
    max_template_width: int | None = defaults.MAX_TEMPLATE_WIDTH
    refine_iters: int = defaults.REFINE_ITERS
    skip_dictionary: bool = False
    # Re-run learning even when the segmentation already has a dictionary
    # (e.g. after changing GPL_REPEATS_N_ROUNDS / --n-rounds).
    relearn: bool = False


def cache_dir_for(wav_path: Path, library_roots: list[Path], result_dirname: str) -> Path:
    wav = wav_path.resolve()
    for root in library_roots:
        r = root.resolve()
        if wav.is_relative_to(r):
            rel = wav.relative_to(r)
            return r / result_dirname / rel.with_suffix(".gpl")
    return wav_path.with_suffix(".gpl")


def discover_wavs(input_path: Path, result_dirname: str) -> list[Path]:
    input_path = normalize_input_path(input_path)
    if is_wav_file(input_path):
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")
    return sorted(
        p
        for p in input_path.rglob("*.wav")
        if is_wav_file(p) and result_dirname not in p.relative_to(input_path).parts
    )


def _build_manifest(output_dir: Path, job_id: str) -> dict:
    """Minimal manifest for cache hit detection (full viewer manifest can be rebuilt by backend)."""
    npy_dir = output_dir / "spectrogram_npy"
    windows = {}
    for sc in sorted(npy_dir.glob("spwhiten_*.json")):
        meta = json.loads(sc.read_text())
        windows[sc.stem] = {
            "windowName": sc.stem,
            "sampleRate": meta.get("sample_freq"),
            "windowStartSample": meta.get("start_sample"),
            "windowEndSample": meta.get("end_sample"),
            "freqBins": meta.get("rows"),
            "timeCols": meta.get("cols"),
            "detections": [],
        }
    return {"jobId": job_id, "windows": windows}


def index_wav_into_catalog(
    conn,
    *,
    library_root: Path,
    wav_path: Path,
    output_dir: Path,
) -> int:
    """Upsert WAV + segmentation + windows + panel_path; return segmentation_id."""
    wav_id = catalog.upsert_wav_file(conn, library_root=library_root, abs_path=wav_path)
    npy_dir = output_dir / "spectrogram_npy"
    csvs = sorted(output_dir.glob("CSV_*.csv"))
    panel = output_dir / "panel.bin"
    segmentation_id = catalog.upsert_gpl_segmentation(
        conn,
        wav_id=wav_id,
        output_dir=output_dir,
        npy_dir=npy_dir if npy_dir.is_dir() else None,
        csv_path=csvs[0] if csvs else None,
        panel_path=panel if panel.is_file() else None,
    )
    if npy_dir.is_dir():
        catalog.upsert_windows_from_dir(conn, segmentation_id=segmentation_id, npy_dir=npy_dir)
    return segmentation_id


def learn_and_register(
    conn,
    *,
    output_dir: Path,
    segmentation_id: int,
    stem: str,
    opts: PreprocessOptions,
    dict_out: Path | None = None,
) -> tuple[int, int] | None:
    if opts.skip_dictionary:
        return None
    result = learn_averaged_templates_from_gpl_paths(
        npy_dir=output_dir / "spectrogram_npy",
        gpl_output_dir=output_dir,
        n_windows=opts.n_windows,
        window_offset=opts.window_offset,
        n_rounds=opts.n_rounds,
        eps_step_fraction=opts.eps_step_fraction,
        min_repeats=opts.min_repeats,
        normalize=opts.normalize,
        min_template_width=opts.min_template_width,
        max_template_width=opts.max_template_width,
        show_progress=True,
    )
    if not result.get_templates():
        return None
    if opts.refine_iters > 0:
        from analysis.dict_learning.code.learn.build_dictionary import apply_refine_iters
        from analysis.dict_learning.code.io.io_gpl import load_gpl_inputs

        windows, _ = load_gpl_inputs(
            npy_dir=output_dir / "spectrogram_npy",
            gpl_output_dir=output_dir,
        )
        end = opts.window_offset + opts.n_windows
        windows = list(windows)[opts.window_offset : end]
        result = apply_refine_iters(
            windows,
            result,
            refine_iters=opts.refine_iters,
            eps_step_fraction=opts.eps_step_fraction,
            normalize=opts.normalize,
            show_progress=True,
        )
    out = dict_out or (output_dir.parent / "dictionaries" / f"{stem}_averaged")
    label = f"{stem}_d{len(result.get_templates())}_w{opts.n_windows}"
    return register_averaged_result(
        conn,
        result=result,
        out_dir=out,
        label=label,
        segmentation_id=segmentation_id,
        params={
            "n_windows": opts.n_windows,
            "window_offset": opts.window_offset,
            "n_rounds": opts.n_rounds,
            "eps_step_fraction": opts.eps_step_fraction,
            "min_repeats": opts.min_repeats,
            "normalize": opts.normalize,
            "min_template_width": opts.min_template_width,
            "max_template_width": opts.max_template_width,
            "refine_iters": opts.refine_iters,
            "learner": "learn_averaged_templates_from_gpl",
        },
        name=label,
    )


def preprocess_one(
    wav_path: Path,
    *,
    library_roots: list[Path],
    result_dirname: str,
    db_path: Path | None,
    opts: PreprocessOptions,
) -> str:
    cache_dir = cache_dir_for(wav_path, library_roots, result_dirname)
    output_dir = cache_dir / "output"
    panel_path = output_dir / "panel.bin"
    manifest_path = cache_dir / "manifest.json"

    if not opts.force and panel_path.is_file() and manifest_path.is_file():
        # Still ensure catalog + dictionary are present when a DB is configured.
        if db_path is not None:
            root = next(
                (r for r in library_roots if wav_path.resolve().is_relative_to(r.resolve())),
                library_roots[0] if library_roots else wav_path.parent,
            )
            with closing(catalog.connect(db_path)) as conn:
                seg_id = index_wav_into_catalog(
                    conn, library_root=root, wav_path=wav_path, output_dir=output_dir
                )
                d = catalog.get_dictionary_for_segmentation(conn, seg_id)
                relearned = False
                if (d is None or opts.relearn) and not opts.skip_dictionary:
                    relearned = (
                        learn_and_register(
                            conn,
                            output_dir=output_dir,
                            segmentation_id=seg_id,
                            stem=wav_path.stem,
                            opts=opts,
                        )
                        is not None
                    )
                conn.commit()
            if relearned:
                return f"skipped (cached), dictionary relearned d{opts.n_rounds}_w{opts.n_windows}"
            return "skipped (cached, catalog refreshed)"
        return "skipped (cached)"

    output_dir.mkdir(parents=True, exist_ok=True)
    ProcessingOrchestrator(
        output_dir,
        freq_lo=opts.freq_lo,
        freq_hi=opts.freq_hi,
        intensity_clip_hi_percentile=opts.intensity_clip_hi_percentile,
    ).process(wav_path)

    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(_build_manifest(output_dir, wav_path.stem)))

    if db_path is not None:
        root = next(
            (r for r in library_roots if wav_path.resolve().is_relative_to(r.resolve())),
            library_roots[0] if library_roots else wav_path.parent,
        )
        with closing(catalog.connect(db_path)) as conn:
            seg_id = index_wav_into_catalog(
                conn, library_root=root, wav_path=wav_path, output_dir=output_dir
            )
            learn_and_register(
                conn, output_dir=output_dir, segmentation_id=seg_id, stem=wav_path.stem, opts=opts
            )
            conn.commit()
    return "done"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build GPL caches and register panel + dictionary in the catalog.")
    p.add_argument("input_path", type=Path)
    p.add_argument("--db", default=None, help="Catalog SQLite path (required to register panel/dictionary)")
    p.add_argument("--library-root", action="append", default=[], help="WAV library root (repeatable)")
    p.add_argument("--result-dirname", default=os.environ.get("GPL_RESULT_DIRNAME", "result"))
    p.add_argument("--force", action="store_true")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--freq-lo", type=float, default=10.0)
    p.add_argument("--freq-hi", type=float, default=10000.0)
    p.add_argument("--intensity-clip-hi-percentile", type=float, default=DEFAULT_INTENSITY_CLIP_HI_PERCENTILE)
    p.add_argument("--n-windows", type=int, default=defaults.N_WINDOWS)
    p.add_argument("--window-offset", type=int, default=defaults.WINDOW_OFFSET)
    p.add_argument("--n-rounds", type=int, default=defaults.N_ROUNDS,
                   help="Optional max outer rounds (default: grow until below min-repeats)")
    p.add_argument("--eps-step-fraction", type=float, default=defaults.EPS_STEP_FRACTION)
    p.add_argument("--min-repeats", type=int, default=defaults.MIN_REPEATS)
    p.add_argument("--min-template-width", type=int, default=defaults.MIN_TEMPLATE_WIDTH)
    p.add_argument("--max-template-width", type=int, default=defaults.MAX_TEMPLATE_WIDTH)
    p.add_argument("--refine-iters", type=int, default=defaults.REFINE_ITERS)
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--skip-dictionary", action="store_true")
    p.add_argument(
        "--relearn",
        action="store_true",
        help="Re-learn the dictionary for cached files (default: keep the existing one)",
    )
    args = p.parse_args(argv)

    roots = [Path(r) for r in args.library_root]
    if not roots:
        roots = [
            Path(x.strip())
            for x in os.environ.get("WAV_LIBRARY_ROOTS", "").split(",")
            if x.strip()
        ]
    scan = normalize_input_path(args.input_path)
    scan_root = (scan if scan.is_dir() else scan.parent).resolve()
    if not any(scan_root.is_relative_to(r.resolve()) for r in roots):
        roots.append(scan_root)

    db_path = Path(args.db) if args.db else None
    if db_path is None:
        env_db = os.environ.get("CATALOG_DB", "").strip()
        if env_db:
            db_path = Path(env_db)
        elif roots:
            db_path = roots[0] / args.result_dirname / "dict_learning_catalog.sqlite"

    opts = PreprocessOptions(
        force=args.force,
        freq_lo=args.freq_lo,
        freq_hi=args.freq_hi,
        intensity_clip_hi_percentile=args.intensity_clip_hi_percentile,
        n_windows=args.n_windows,
        window_offset=args.window_offset,
        n_rounds=args.n_rounds,
        eps_step_fraction=args.eps_step_fraction,
        min_repeats=args.min_repeats,
        normalize=not args.no_normalize,
        min_template_width=args.min_template_width,
        max_template_width=args.max_template_width,
        refine_iters=args.refine_iters,
        skip_dictionary=args.skip_dictionary,
        relearn=args.relearn,
    )

    wavs = discover_wavs(args.input_path, args.result_dirname)
    if not wavs:
        print(f"No WAV files found at {normalize_input_path(args.input_path)}")
        return 1

    print(f"Found {len(wavs)} WAV(s). db={db_path} workers={args.workers} force={args.force}")

    def _run(item: tuple[int, Path]) -> tuple[int, Path, str, float]:
        i, wav = item
        t0 = time.perf_counter()
        try:
            status = preprocess_one(
                wav,
                library_roots=roots,
                result_dirname=args.result_dirname,
                db_path=db_path,
                opts=opts,
            )
        except Exception as exc:  # noqa: BLE001
            status = f"ERROR: {exc}"
        return i, wav, status, time.perf_counter() - t0

    failures = 0
    if args.workers <= 1:
        for item in enumerate(wavs, start=1):
            i, wav, status, dt = _run(item)
            if status.startswith("ERROR"):
                failures += 1
            print(f"[{i}/{len(wavs)}] {wav.name} ... {status} ({dt:.1f}s)")
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_run, (i, w)): i for i, w in enumerate(wavs, start=1)}
            for fut in as_completed(futs):
                i, wav, status, dt = fut.result()
                if status.startswith("ERROR"):
                    failures += 1
                print(f"[{i}/{len(wavs)}] {wav.name} ... {status} ({dt:.1f}s)")

    print(f"\nComplete: {len(wavs) - failures} ok, {failures} failed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
