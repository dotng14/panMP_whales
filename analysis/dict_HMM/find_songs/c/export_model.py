"""Write the dictionary and the theme model as one binary file for the C detector.

The dictionary rows are cut to the band a 10 kHz file holds and the components
are made orthonormal, exactly as `place.load_dictionary` does, so the C program
scores with the same numbers.

usage: python3 export_model.py OUT_FILE
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import place  # noqa: E402


def main() -> None:
    out = Path(sys.argv[1])
    ids = sorted(
        int(p.stem.split("_")[1]) for p in (place.DICTIONARY / "atoms").glob("template_*.npy")
    )
    theme = np.load(HERE.parent / "results" / "theme_model.npz")
    if list(theme["motif_ids"]) != ids:
        raise SystemExit("theme model and dictionary list different motifs")
    means, bases, widths = [], [], []
    for motif_id in ids:
        components = np.load(place.DICTIONARY / "atoms" / f"template_{motif_id:03d}.npy")
        components = components[:, : place.N_ROWS, :].astype(np.float64)
        mean = np.load(place.DICTIONARY / "atoms" / f"mean_{motif_id:03d}.npy")[: place.N_ROWS, :].astype(np.float64)
        n_components, _rows, width = components.shape
        flat, _ = np.linalg.qr(components.reshape(n_components, -1).T)
        bases.append(np.ascontiguousarray(flat.T.reshape(n_components, place.N_ROWS, width)))
        means.append(np.ascontiguousarray(mean))
        widths.append(width)
    with open(out, "wb") as handle:
        handle.write(b"SONGMDL1")
        handle.write(struct.pack("<4i", len(ids), n_components, place.N_ROWS, theme["log_p"].shape[0]))
        handle.write(np.asarray(ids, dtype="<i4").tobytes())
        handle.write(np.asarray(widths, dtype="<i4").tobytes())
        for mean, basis in zip(means, bases):
            handle.write(mean.astype("<f8").tobytes())
            handle.write(basis.astype("<f8").tobytes())
        for name in ("log_p", "mu", "sigma"):
            handle.write(np.ascontiguousarray(theme[name], dtype="<f8").tobytes())
    print(f"wrote {out}: {len(ids)} motifs, {place.N_ROWS} rows, {out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
