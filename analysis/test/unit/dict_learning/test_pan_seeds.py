"""Pan matrix profile seeds (``learn/pan_seeds.py``) and the column map they use."""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from analysis.dict_learning.code.core import Dictionary, Template
from analysis.dict_learning.code.io import ColumnMap
from analysis.dict_learning.code.learn import pan_seeds
from analysis.dict_learning.code.learn.pan_seeds import PanSeedConfig, PanSeeder, shape_similarity
from analysis.dict_learning.code.learn.randomized_dictionary import learn_randomized_outer
from analysis.test.synthetic import planted_stack

FREQ = {"start_freq": 10.0, "end_freq": 130.0}


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


def _motif_of(atom: np.ndarray, motifs: list[np.ndarray]) -> tuple[int, float]:
    sims = [shape_similarity(atom, _unit(m)) for m in motifs]
    return int(np.argmax(sims)), float(max(sims))


# ── ColumnMap ────────────────────────────────────────────────────────────────


def test_column_map_round_trip_and_boundaries():
    stack = {"a": np.zeros((2, 5)), "b": np.ones((2, 3)), "c": np.full((2, 4), 2.0)}
    columns = ColumnMap.from_stack(stack)
    assert columns.starts == (0, 5, 8, 12) and columns.n_columns == 12
    assert columns.join(stack).shape == (2, 12)
    assert columns.locate(6, 2) == ("b", 1)
    assert columns.locate(4, 2) is None  # spans a and b
    assert columns.locate(8, 4) == ("c", 0)
    assert columns.locate(12, 1) is None
    np.testing.assert_array_equal(
        np.flatnonzero(columns.crosses_boundary(3)), [3, 4, 6, 7]
    )


# ── shape_similarity ─────────────────────────────────────────────────────────


def test_shape_similarity_matches_shifted_and_longer_cuts():
    rng = np.random.default_rng(0)
    call = rng.standard_normal((4, 10))
    padded = np.concatenate([np.zeros((4, 3)), call, np.zeros((4, 5))], axis=1)
    assert shape_similarity(call, call) == pytest.approx(1.0)
    assert shape_similarity(call, padded) == pytest.approx(1.0)
    assert shape_similarity(padded, call) == pytest.approx(1.0)
    other = np.zeros((4, 10))
    other[0] = 1.0
    call_elsewhere = np.zeros((4, 10))
    call_elsewhere[3] = rng.standard_normal(10)
    assert shape_similarity(other, call_elsewhere) == 0.0
    assert shape_similarity(np.zeros((4, 6)), call) == 0.0


# ── the seeder ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("seed", range(4))
def test_one_seed_per_motif_with_its_own_copies(seed, monkeypatch):
    """Motif 0 is twice as frequent and twice as loud; it must still give one seed."""
    planted = planted_stack(seed=seed)
    calls = []
    real = pan_seeds.pan_profile
    monkeypatch.setattr(pan_seeds, "pan_profile", lambda *a, **k: calls.append(1) or real(*a, **k))
    seeder = PanSeeder(PanSeedConfig(lengths=(12,), signal="full", **FREQ))
    result = seeder(planted.stack, 3, Dictionary())

    assert len(calls) == 1
    assert result.n_sampled == len(result.dictionary) == 3
    found = [_motif_of(t.data, planted.motifs) for t in result.dictionary]
    assert sorted(k for k, _ in found) == [0, 1, 2]
    assert all(sim > 0.99 for _, sim in found)
    for template, pick in zip(result.dictionary, seeder.passes[0].picks):
        motif = _motif_of(template.data, planted.motifs)[0]
        planted_at = {(f"w{w}", c) for w, c, k in planted.copies if k == motif}
        assert len(pick.copies) == len(planted_at)
        # Within a quarter of the width: z-normalized minima move a little with the noise.
        for window_id, t in pick.copies:
            assert any(w == window_id and abs(t - c) <= 3 for w, c in planted_at)
        assert pick.template_id == template.template_id
        assert np.linalg.norm(template.data) == pytest.approx(1.0)
        assert template.source_segment_id.startswith("pan:")


def test_a_call_already_in_the_dictionary_is_not_seeded_again():
    # Motif 0's copies stay in this residual (the outer loop would have
    # subtracted them), so its pair is the strongest and must be skipped.
    planted = planted_stack(seed=0)
    known = Dictionary()
    known.add(Template(template_id=7, data=_unit(planted.motifs[0])))
    seeder = PanSeeder(PanSeedConfig(lengths=(12,), signal="full", **FREQ))
    result = seeder(planted.stack, 2, known)
    assert seeder.passes[0].n_duplicates >= 1
    assert sorted(_motif_of(t.data, planted.motifs)[0] for t in result.dictionary) == [1, 2]
    assert min(t.template_id for t in result.dictionary) == 8


def test_too_few_copies_gives_no_seed_and_stops_trying():
    planted = planted_stack(seed=0)
    seeder = PanSeeder(PanSeedConfig(lengths=(12,), signal="full", min_copies=11, **FREQ))
    result = seeder(planted.stack, 2, Dictionary())
    log = seeder.passes[0]
    assert len(result.dictionary) == 0 and result.pool_size == log.n_pairs_tried
    assert log.n_too_few_copies > 0
    assert log.n_pairs_tried <= pan_seeds.TRIES_PER_SEED * 2


def test_decimated_search_still_finds_every_motif():
    planted = planted_stack(seed=1)
    seeder = PanSeeder(PanSeedConfig(lengths=(12,), signal="full", decimate=2, **FREQ))
    result = seeder(planted.stack, 3, Dictionary())
    assert seeder.passes[0].signal_columns == 450
    found = [_motif_of(t.data, planted.motifs) for t in result.dictionary]
    assert sorted(k for k, _ in found) == [0, 1, 2]
    assert all(sim > 0.95 for _, sim in found)
    assert all(t.width == 12 for t in result.dictionary)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"signal": "rms"}, "signal must be one of"),
        ({"lengths": ()}, "at least one motif length"),
        ({"decimate": 0}, "decimate must be >= 1"),
        ({"lengths": (6,), "decimate": 2}, "at least 4 search samples"),
        ({"copy_threshold": 1.0}, "copy threshold"),
        ({"min_copies": 1}, "min copies"),
        ({"dedupe_corr": 0.0}, "dedupe correlation"),
    ],
)
def test_config_rejects(kwargs, message):
    with pytest.raises(ValueError, match=message):
        PanSeedConfig(**{"lengths": (12,), **FREQ, **kwargs})


def test_search_lengths_round_to_distinct_samples():
    config = PanSeedConfig(lengths=(20, 12, 13), decimate=3, **FREQ)
    assert config.search_lengths == [4, 7]


# ── in the outer loop ────────────────────────────────────────────────────────


def test_outer_loop_with_pan_seeds_learns_every_motif():
    planted = planted_stack(seed=3)
    seeder = PanSeeder(PanSeedConfig(lengths=(12,), **FREQ))
    result = learn_randomized_outer(
        planted.stack, None, seeder=seeder, m_per_outer=3, n_min=3, min_coverage=0.3,
        max_outer=3, refine_iters=2, n_workers=1,
    )
    counts = Counter(m.template_id for m in result.matchings)
    found = {}
    for template in result.dictionary:
        motif, sim = _motif_of(template.data, planted.motifs)
        if sim > 0.95:
            found[motif] = counts[template.template_id]
    assert sorted(found) == [0, 1, 2]
    assert len(seeder.passes) == result.n_outer_completed


def test_outer_loop_needs_a_partition_or_a_seeder():
    with pytest.raises(ValueError, match="partition is required"):
        learn_randomized_outer(planted_stack().stack, None)


def test_lengths_take_turns_longest_first():
    planted = planted_stack(seed=0)
    seeder = PanSeeder(PanSeedConfig(lengths=(8, 12), signal="full", **FREQ))
    seeder(planted.stack, 3, Dictionary())
    widths = [pick.width for pick in seeder.passes[0].picks]
    assert widths[0] == 12 and set(widths) == {8, 12}
