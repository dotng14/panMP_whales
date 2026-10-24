"""Compatibility / subspace-integration smoke tests for Template and Matching."""

from __future__ import annotations

import numpy as np

from analysis.dict_learning.code.learn.learn_dictionary import learn_averaged_templates_from_gpl
from analysis.dict_learning.code.mp import run_mp_loop, single_template_dictionary
from analysis.dict_learning.code.core import Matching, Segment, Template, Window
from analysis.dict_learning.code.learn.seed_from_gpl import mean_center
from analysis.dict_learning.code.tools import pca_basis, rebuild_template_from_matchings


def test_matching_amplitude_shim():
    m = Matching(0, "w0", 3, amplitude=1.5)
    assert m.amplitude == 1.5
    assert m.coeffs.shape == (1,)
    assert abs(float(m.coeffs[0]) - 1.5) < 1e-12
    assert abs(m.energy - 1.5**2) < 1e-12

    m2 = Matching(0, "w0", 3, coeffs=[3.0, 4.0])
    assert abs(m2.amplitude - 5.0) < 1e-12
    assert abs(m2.energy - 25.0) < 1e-12


def test_template_rank1_and_basis():
    data = np.ones((4, 6), dtype=np.float64)
    t = Template(0, data)
    assert t.n_components == 1
    assert t.width == 6 and t.n_freq == 4
    assert t.basis.shape == (1, 4, 6)
    assert t.as_rank1.shape == (4, 6)

    basis = np.stack([data, data * 0.5], axis=0)
    # Orthonormalize roughly for storage
    basis[0] /= np.linalg.norm(basis[0])
    basis[1] = basis[1] - np.sum(basis[1] * basis[0]) * basis[0]
    basis[1] /= np.linalg.norm(basis[1])
    t2 = Template(1, basis)
    assert t2.n_components == 2
    assert t2.basis.shape == (2, 4, 6)


def test_learn_n_components_2_synthetic():
    rng = np.random.default_rng(0)
    freq, width, tlen = 8, 6, 80
    v1 = rng.normal(size=(freq, width))
    v1 /= np.linalg.norm(v1)
    v2 = rng.normal(size=(freq, width))
    v2 = v2 - np.sum(v2 * v1) * v1
    v2 /= np.linalg.norm(v2)

    signal = np.zeros((freq, tlen), dtype=np.float64)
    starts = [5, 25, 45, 65]
    for i, t0 in enumerate(starts):
        a, b = (2.0, 0.5) if i % 2 == 0 else (1.5, -0.8)
        signal[:, t0 : t0 + width] += a * v1 + b * v2

    window = Window(
        window_id="w0",
        data=signal,
        start_sample=1,
        end_sample=tlen,
        overlap=1,
    )
    segments = [
        Segment(
            segment_id=f"s{i}",
            data=np.array(signal[:, t0 : t0 + width], copy=True),
            window_id="w0",
            t_start=t0,
            t_end=t0 + width,
            global_start_sample=t0,
            global_end_sample=t0 + width,
            detection_index=i,
        )
        for i, t0 in enumerate(starts)
    ]
    result = learn_averaged_templates_from_gpl(
        [window],
        segments,
        n_rounds=1,
        eps_step_fraction=1e-9,
        min_repeats=3,
        n_components=2,
        disallow_overlap=True,
        show_progress=False,
    )
    templates = result.get_templates()
    assert len(templates) == 1
    assert templates[0].n_components == 2
    assert templates[0].data.ndim == 3
    assert len(result.get_matchings()) >= 3
    for m in result.get_matchings():
        assert np.asarray(m.coeffs).size == 2


def test_rebuild_and_place_k2():
    rng = np.random.default_rng(1)
    freq, width = 6, 5
    patches = []
    matchings = []
    stack = {"w0": np.zeros((freq, 40), dtype=np.float64)}
    v1 = rng.normal(size=(freq, width))
    v1 /= np.linalg.norm(v1)
    v2 = rng.normal(size=(freq, width))
    v2 -= np.sum(v2 * v1) * v1
    v2 /= np.linalg.norm(v2)
    for i, t0 in enumerate((2, 12, 22)):
        coeffs = np.array([1.0 + 0.1 * i, 0.5 - 0.05 * i])
        stack["w0"][:, t0 : t0 + width] += coeffs[0] * v1 + coeffs[1] * v2
        matchings.append(Matching(0, "w0", t0, amplitude=1.0))
    tmpl = Template(0, v1)
    rebuilt = rebuild_template_from_matchings(
        stack, tmpl, matchings, n_components=2, method="pca"
    )
    assert rebuilt.n_components == 2
    loop = run_mp_loop(
        {k: np.array(v, copy=True) for k, v in stack.items()},
        single_template_dictionary(rebuilt),
        eps_step_fraction=1e-9,
        normalize=False,
        disallow_overlap=True,
        mp_mode="global",
        show_progress=False,
    )
    assert len(loop.matchings) >= 2
    e0 = float(np.sum(stack["w0"] ** 2))
    assert loop.energies[-1] < e0


if __name__ == "__main__":
    test_matching_amplitude_shim()
    test_template_rank1_and_basis()
    test_learn_n_components_2_synthetic()
    test_rebuild_and_place_k2()
    print("ok")
