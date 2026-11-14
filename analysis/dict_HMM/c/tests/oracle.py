"""Write a tiny spectrogram and the placements produced by the Python learner."""

import sys
from pathlib import Path

import numpy as np

from analysis.dictionary_learning_limited_overlap.learn import (
    Motif,
    _explain,
    _principal_components,
    _pursuit,
)


def main() -> None:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    spec = rng.normal(size=(5, 36))
    pattern = rng.normal(size=(5, 6))
    pattern /= np.linalg.norm(pattern)
    spec[:, 4:10] += 4.0 * pattern
    spec[:, 22:28] += 3.0 * pattern
    motif = Motif(motif_id=7, data=pattern, window_id="w0", t_start=4)
    fraction, explained, dots = _explain(spec, motif)
    counts = {"w0": np.zeros(spec.shape[1], dtype=np.int32)}
    placements = _pursuit([motif], {"w0": spec}, counts, k=1, eps=0.01)

    spec.astype(np.float64).tofile(out / "spec.bin")
    pattern.astype(np.float64).tofile(out / "pattern.bin")
    fraction.astype(np.float64).tofile(out / "fraction.bin")
    lines = [f"{spec.shape[0]} {spec.shape[1]} {pattern.shape[1]} {len(fraction)}"]
    lines.append(f"{len(placements)}")
    for placement in placements:
        lines.append(
            f"{placement.motif_id} {placement.t_start} {placement.unexplained:.10f} {placement.amplitude:.10f}"
        )
    (out / "pursuit.txt").write_text("\n".join(lines) + "\n")

    patches = rng.normal(size=(6, 4, 5))
    base = rng.normal(size=(4, 5))
    base /= np.linalg.norm(base)
    patches += np.linspace(0.2, 2.0, 6)[:, None, None] * base
    basis = _principal_components(list(patches), 2)
    patches.astype(np.float64).tofile(out / "patches.bin")
    basis.astype(np.float64).tofile(out / "basis.bin")
    (out / "basis.txt").write_text(f"{patches.shape[0]} {patches.shape[1]} {patches.shape[2]} {basis.shape[0]}\n")
    np.testing.assert_allclose(fraction, fraction)
    _ = explained, dots


if __name__ == "__main__":
    main()
