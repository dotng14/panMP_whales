"""Tests for the normalized cross-atom placement score used by matching pursuit.

``_best_placement_on_window`` used to return raw amplitude / absolute
explained energy as its cross-atom comparison score. That let an atom with a
large norm "win" a patch by explaining a lot of *absolute* energy while
actually fitting the patch's *shape* worse (in relative terms) than a
smaller-norm atom would. The fix ranks candidates by the same scale-free
explained-energy fraction used elsewhere for display (``chunk_distance_norm``
/ ``patch_template_relative_distance``), with absolute energy only as a
tiebreak — see ``PlacementScore`` in ``mp_step.py``.
"""

from __future__ import annotations


import numpy as np


from analysis.dict_learning.code.core import Dictionary, Template  # noqa: E402
from analysis.dict_learning.code.mp import (  # noqa: E402
    _best_placement_on_window,
    _mp_step_one_window_inplace,
    _placement_beats,
    mp_step,
)


def test_placement_beats_prefers_higher_relative_fit_over_higher_absolute_energy():
    # Low absolute energy but a near-perfect relative fit should beat a huge
    # absolute energy that only explains a small fraction of its patch.
    small_but_precise = (0.99, 5.0)
    big_but_sloppy = (0.5, 1000.0)
    assert _placement_beats(small_but_precise, big_but_sloppy)
    assert not _placement_beats(big_but_sloppy, small_but_precise)


def test_placement_beats_uses_absolute_energy_as_tiebreak():
    higher_energy = (0.9, 50.0)
    lower_energy = (0.9, 10.0)
    assert _placement_beats(higher_energy, lower_energy)
    assert not _placement_beats(lower_energy, higher_energy)


def test_placement_beats_treats_near_equal_scores_as_tied():
    """Two independently-computed 'identical' atoms can land ~1e-15 apart from
    a perfect fit purely from floating-point rounding. Neither should be able
    to claim victory over the other from noise alone (see the loud/faint
    identical-atom refine tests in test_learn_dictionary.py, which rely on
    ties resolving by encounter order, not by whichever side of 1.0 the
    rounding error happened to land on).
    """
    almost_one = (1.0 - 1e-15, 90.0)
    exactly_one = (1.0, 89.99999999999997)
    assert not _placement_beats(almost_one, exactly_one)
    assert not _placement_beats(exactly_one, almost_one)


def _rank1_basis(row_index: int, row: list[float], width: int, n_freq: int) -> np.ndarray:
    basis = np.zeros((1, n_freq, width), dtype=np.float64)
    basis[0, row_index, :] = row
    return basis


def test_mp_step_prefers_the_atom_that_explains_more_of_its_own_patch():
    """Integration-level regression test for the normalization fix.

    ``atom_small`` and ``atom_big`` live in different (orthogonal) frequency
    rows, each with a patch of its own, so neither atom's own amplitude-based
    shift search is contaminated by the other's region. ``atom_small`` sits
    exactly on top of its patch (relative fit 1.0, tiny absolute energy).
    ``atom_big`` only explains ~12% of a much louder patch (most of that
    patch's energy is a third, orthogonal row neither atom can explain) —
    poor relative fit, but far larger absolute energy (675 vs 3). Under the
    old unnormalized score (raw amplitude/explained energy), atom_big would
    win outright. The fix must pick atom_small.
    """
    width, n_freq, n_cols = 3, 3, 20
    resid = np.zeros((n_freq, n_cols), dtype=np.float64)

    # Region A, row 0: exact match for atom_small (amp=1, explained=patch_energy=3).
    resid[0, 2:5] = [1.0, 1.0, 1.0]

    # Region B, row 1: atom_big's direction scaled by 3 (amp=3, explained=675);
    # row 2 is a much larger orthogonal burst atom_big can't explain, so the
    # patch's own energy (5475) leaves atom_big explaining only ~12.3% of it.
    resid[1, 10:13] = [15.0, 15.0, 15.0]
    resid[2, 10:13] = [40.0, 40.0, 40.0]

    atom_small = _rank1_basis(0, [1.0, 1.0, 1.0], width, n_freq)
    atom_big = _rank1_basis(1, [5.0, 5.0, 5.0], width, n_freq)

    placed_small = _best_placement_on_window(
        atom_small, resid, template_id=0, window_id="w0",
        normalize=True, occupied_intervals=None,
    )
    placed_big = _best_placement_on_window(
        atom_big, resid, template_id=1, window_id="w0",
        normalize=True, occupied_intervals=None,
    )
    assert placed_small is not None and placed_big is not None
    relative_fit_small, energy_small = placed_small[0]
    relative_fit_big, energy_big = placed_big[0]
    assert relative_fit_small == 1.0
    assert energy_small < energy_big  # confirms atom_big really is "louder"
    assert relative_fit_small > relative_fit_big  # but fits worse, relatively

    atoms = [(0, atom_small, width), (1, atom_big, width)]
    step = _mp_step_one_window_inplace(resid.copy(), atoms, window_id="w0")
    assert step.matching is not None
    assert step.matching.template_id == 0
    assert step.matching.t_start == 2


def test_mp_step_global_mode_also_prefers_the_better_relative_fit():
    """Same scenario as above, but through mp_step() (global cross-window
    selection) rather than _mp_step_one_window_inplace — a regression guard
    since these two call sites duplicate the winner-selection comparison and
    a fix to one does not automatically apply to the other.
    """
    width, n_freq, n_cols = 3, 3, 20
    resid = np.zeros((n_freq, n_cols), dtype=np.float64)
    resid[0, 2:5] = [1.0, 1.0, 1.0]
    resid[1, 10:13] = [15.0, 15.0, 15.0]
    resid[2, 10:13] = [40.0, 40.0, 40.0]

    dictionary = Dictionary()
    dictionary.add(Template(template_id=0, data=_rank1_basis(0, [1.0, 1.0, 1.0], width, n_freq)[0]))
    dictionary.add(Template(template_id=1, data=_rank1_basis(1, [5.0, 5.0, 5.0], width, n_freq)[0]))

    _resid_out, step = mp_step({"w0": resid}, dictionary)
    assert step.matching is not None
    assert step.matching.template_id == 0
    assert step.matching.t_start == 2
