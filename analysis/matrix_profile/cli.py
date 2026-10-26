"""Command line for the pan matrix profile.

    python -m analysis.matrix_profile pan --series x.npy --out OUT
    python -m analysis.matrix_profile pan --wav clip.wav --rate 100 --signal bands --out OUT
    python -m analysis.matrix_profile pan --gpl-dir GPL_OUT --rate 10 --signal full --out OUT

Motif lengths (``--min-len``, ``--max-len``, ``--step``) are in samples of the
signal. Writes ``pan.npy`` (motif length × time), ``pan.json`` (lengths and
how the signal was built), ``pairs.json`` (most unusual pairs across lengths;
see ``pan_top_pairs``) and, with ``--plot``, ``pan.png``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from . import ENGINES, pan_lengths, pan_profile, pan_top_pairs
from .signals import GPL_KINDS, WAV_KINDS, load_gpl, load_series, load_wav


def _band_edges(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m analysis.matrix_profile",
        description="Pan matrix profile: the matrix profile over a sweep of motif lengths.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    pan = sub.add_parser(
        "pan",
        help="Run the pan matrix profile on a series, a WAV, or a GPL output folder",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = pan.add_mutually_exclusive_group(required=True)
    source.add_argument("--series", type=Path, help="1-D or (rows, T) .npy array")
    source.add_argument("--wav", type=Path, help="Mono WAV file")
    source.add_argument("--gpl-dir", type=Path, help="GPL output folder")
    pan.add_argument(
        "--signal",
        default=None,
        help=f"WAV: {'|'.join(WAV_KINDS)} (default rms); GPL: {'|'.join(GPL_KINDS)} (default energy)",
    )
    pan.add_argument(
        "--rate",
        type=float,
        default=None,
        help="Signal samples per second (required for --wav; GPL default: native STFT columns)",
    )
    pan.add_argument("--band-edges", type=_band_edges, default=None, help="GPL bands, Hz: 10,100,200")
    pan.add_argument("--min-len", type=int, default=32)
    pan.add_argument("--max-len", type=int, default=128)
    pan.add_argument("--step", type=int, default=32)
    pan.add_argument("--top-k", type=int, default=10)
    pan.add_argument("--engine", choices=ENGINES, default="fast")
    pan.add_argument("--threads", type=int, default=1)
    pan.add_argument("--plot", action="store_true", help="Also write pan.png")
    pan.add_argument("--out", type=Path, required=True)
    pan.set_defaults(func=run_pan)
    return parser


def _load_signal(args: argparse.Namespace):
    if args.series is not None:
        if not args.series.is_file():
            raise ValueError(f"series file not found: {args.series}")
        return load_series(args.series)
    if args.wav is not None:
        if not args.wav.is_file():
            raise ValueError(f"WAV file not found: {args.wav}")
        if args.rate is None:
            raise ValueError("--rate is required with --wav")
        return load_wav(args.wav, int(args.rate), args.signal or "rms")
    if not args.gpl_dir.is_dir():
        raise ValueError(f"GPL directory not found: {args.gpl_dir}")
    return load_gpl(args.gpl_dir, args.signal or "energy", rate=args.rate, band_edges=args.band_edges)


def _plot(surface: np.ndarray, lengths: list[int], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 3))
    image = ax.imshow(
        np.ma.masked_invalid(surface),
        aspect="auto",
        origin="lower",
        cmap="jet",
        extent=(0, surface.shape[1], lengths[0], lengths[-1]),
    )
    ax.set_xlabel("start (samples)")
    ax.set_ylabel("motif length (samples)")
    fig.colorbar(image, ax=ax)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def run_pan(args: argparse.Namespace) -> int:
    signal = _load_signal(args)
    if args.top_k < 1:
        raise ValueError(f"--top-k must be >= 1, got {args.top_k}")
    lengths = pan_lengths(args.min_len, args.max_len, args.step)
    profile = pan_profile(signal.data, lengths, engine=args.engine, nthreads=args.threads)
    pairs = pan_top_pairs(profile, args.top_k)
    args.out.mkdir(parents=True, exist_ok=True)
    np.save(args.out / "pan.npy", profile.surface)
    (args.out / "pairs.json").write_text(json.dumps(pairs, indent=2))
    meta = {
        "lengths": lengths,
        "rows": int(signal.data.shape[0]),
        "n_samples": int(signal.data.shape[1]),
        "rate": signal.rate,
        "engine": args.engine,
        "signal": signal.info,
    }
    (args.out / "pan.json").write_text(json.dumps(meta, indent=2))
    if args.plot:
        _plot(profile.surface, lengths, args.out / "pan.png")
    print(json.dumps({"out": str(args.out), "rows": meta["rows"], "n_pairs": len(pairs)}))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

