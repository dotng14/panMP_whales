"""Bipartition a template's patches into two rank-1 subclusters.

Randomized 1→2 split: perturb the current dictionary word by ±ε, assign and
refine within the patch group, score residual SSE, and keep the best of
``n_trials`` (K) draws with up to ``n_refine`` / ``max_refine`` (S) iterations
each. Returned atoms are unit-L2. Used for exploratory splits (no catalog writes).

Prefer :func:`split_one_template` when you want the two templates plus the
patches assigned to each side.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from ..core.objects import Matching, Template, Window
from ..tools import _extract_patches


def _assert_unit_l2(name: str, arr: np.ndarray, *, atol: float = 1e-6) -> None:
    """Assert ``||arr||_2 ≈ 1``."""
    nrm = float(np.linalg.norm(np.asarray(arr, dtype=np.float64)))
    assert abs(nrm - 1.0) <= atol, f"{name} L2 norm is {nrm}, expected 1 (±{atol})"


@dataclass
class SplitResult:
    """Outcome of a 1→2 rank-1 bipartition search."""

    accepted: bool
    labels: np.ndarray  # int, shape (n,), values in {0, 1}
    template_a: np.ndarray  # (H, W) unit-oriented PC1 of cluster 0
    template_b: np.ndarray  # (H, W) unit-oriented PC1 of cluster 1
    sse_one: float
    sse_two: float
    relative_reduction: float
    n_a: int
    n_b: int
    n_pixels: int
    n_trials: int = 0
    best_trial: int = -1

    @property
    def rms_one(self) -> float:
        """Per-pixel RMS residual under the single shared atom."""
        denom = self.labels.size * self.n_pixels
        return float(np.sqrt(self.sse_one / denom)) if denom > 0 else 0.0

    @property
    def rms_two(self) -> float:
        """Per-pixel RMS residual under the two-atom partition."""
        denom = self.labels.size * self.n_pixels
        return float(np.sqrt(self.sse_two / denom)) if denom > 0 else 0.0


@dataclass
class SplitOneResult:
    """Best-of-K split: two unit-L2 templates and their assigned patches."""

    template_a: np.ndarray  # (H, W), ||·||_2 = 1
    template_b: np.ndarray  # (H, W), ||·||_2 = 1
    patches_a: list[np.ndarray]
    patches_b: list[np.ndarray]
    labels: np.ndarray  # int, shape (n,), values in {0, 1}
    sse_one: float
    sse_two: float
    relative_reduction: float
    best_trial: int
    n_trials: int
    n_refine: int


def _stack_patches(patches: Sequence[np.ndarray]) -> tuple[np.ndarray, int, int]:
    arrays = [np.asarray(p, dtype=np.float64) for p in patches]
    if not arrays:
        raise ValueError("patches must be non-empty")
    shape0 = arrays[0].shape
    if len(shape0) != 2:
        raise ValueError(f"each patch must be 2-D (H, W), got shape {shape0}")
    for i, arr in enumerate(arrays):
        if arr.shape != shape0:
            raise ValueError(
                f"patch {i} shape {arr.shape} != first patch shape {shape0}"
            )
    h, w = int(shape0[0]), int(shape0[1])
    data = np.stack([p.reshape(-1) for p in arrays], axis=0)
    return data, h, w


def _orient(vec: np.ndarray, mean_vec: np.ndarray) -> np.ndarray:
    return -vec if float(np.dot(vec, mean_vec)) < 0.0 else vec


def _as_unit(vec: np.ndarray, mean_vec: np.ndarray | None = None) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= 0.0:
        return np.zeros_like(vec, dtype=np.float64)
    unit = vec / norm
    if mean_vec is not None:
        unit = _orient(unit, mean_vec)
    return unit


def _pc1_unit(data: np.ndarray) -> np.ndarray:
    """Unit-norm leading PC of rows in ``data`` (n, D), mean-oriented."""
    n, d = data.shape
    mean_vec = np.mean(data, axis=0)
    if n == 1:
        return _as_unit(data[0], mean_vec)

    gram = data @ data.T
    eigvals, eigvecs = np.linalg.eigh(gram)
    idx = int(np.argmax(eigvals))
    lam = float(eigvals[idx])
    if lam <= 0.0:
        return _as_unit(mean_vec)
    s = float(np.sqrt(lam))
    u = eigvecs[:, idx]
    v = (data.T @ u) / s
    return _orient(v, mean_vec)


def _sse_vs_unit(data: np.ndarray, unit: np.ndarray) -> float:
    """Residual SSE of rows against a unit-norm template (optimal amplitude)."""
    total = float(np.sum(data * data))
    if unit.size == 0 or float(np.sum(unit * unit)) <= 0.0:
        return total
    projs = data @ unit
    explained = float(np.dot(projs, projs))
    return max(total - explained, 0.0)


def _sse_partition(
    data: np.ndarray, labels: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    """Fit PC1 on each label side; return SSE_two, unit_a, unit_b."""
    data_a = data[labels == 0]
    data_b = data[labels == 1]
    unit_a = _pc1_unit(data_a)
    unit_b = _pc1_unit(data_b)
    sse = _sse_vs_unit(data_a, unit_a) + _sse_vs_unit(data_b, unit_b)
    return sse, unit_a, unit_b


def _reassign(data: np.ndarray, unit_a: np.ndarray, unit_b: np.ndarray) -> np.ndarray:
    """Assign each row to the atom with larger explained energy ⟨x,v⟩²."""
    ea = (data @ unit_a) ** 2
    eb = (data @ unit_b) ** 2
    return (eb > ea).astype(np.int64)


def _refine_from_units(
    data: np.ndarray,
    unit_a: np.ndarray,
    unit_b: np.ndarray,
    *,
    min_size: int,
    max_refine: int,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray] | None:
    """Assign → PC1-refit loop starting from two candidate units.

    Returns ``None`` if either cluster falls below ``min_size``.
    """
    mean_vec = np.mean(data, axis=0)
    unit_a = _as_unit(unit_a, mean_vec)
    unit_b = _as_unit(unit_b, mean_vec)
    labels = _reassign(data, unit_a, unit_b)
    n_a = int(np.sum(labels == 0))
    n_b = int(np.sum(labels == 1))
    if n_a < min_size or n_b < min_size:
        return None

    best_sse, unit_a, unit_b = _sse_partition(data, labels)
    best_labels = labels.copy()
    best_a, best_b = unit_a, unit_b

    for _ in range(max(0, int(max_refine))):
        new_labels = _reassign(data, unit_a, unit_b)
        n_a = int(np.sum(new_labels == 0))
        n_b = int(np.sum(new_labels == 1))
        if n_a < min_size or n_b < min_size:
            break
        if np.array_equal(new_labels, labels):
            break
        sse, unit_a, unit_b = _sse_partition(data, new_labels)
        labels = new_labels
        if sse < best_sse:
            best_sse = sse
            best_labels = labels.copy()
            best_a, best_b = unit_a, unit_b
    return best_labels, best_sse, best_a, best_b


def _resolve_base(
    data: np.ndarray,
    h: int,
    w: int,
    base_template: np.ndarray | None,
) -> np.ndarray:
    """Flattened base dictionary word (data-like magnitude)."""
    if base_template is not None:
        base = np.asarray(base_template, dtype=np.float64)
        if base.ndim == 3:
            base = base[0]
        if base.shape != (h, w):
            raise ValueError(
                f"base_template shape {base.shape} does not match patch shape {(h, w)}"
            )
        return base.reshape(-1)

    # No catalog word: use mean patch (same geometry / scale as data).
    return np.mean(data, axis=0)


def _sample_pm_epsilon(
    base: np.ndarray,
    *,
    epsilon: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(base+ε, base-ε)`` with ``||ε|| = epsilon * ||base||``."""
    d = base.size
    g = rng.standard_normal(d)
    g_norm = float(np.linalg.norm(g))
    base_norm = float(np.linalg.norm(base))
    if g_norm <= 0.0 or base_norm <= 0.0:
        # Degenerate: tiny isotropic jitter so ± still differ numerically.
        eps_vec = rng.standard_normal(d) * (abs(float(epsilon)) * 1e-6 + 1e-12)
    else:
        eps_vec = g * ((float(epsilon) * base_norm) / g_norm)
    return base + eps_vec, base - eps_vec


def _empty_split(
    data: np.ndarray,
    h: int,
    w: int,
    *,
    sse_one: float,
    n_trials: int = 0,
    relative_reduction: float = 0.0,
) -> SplitResult:
    n = data.shape[0]
    unit = _pc1_unit(data) if n > 0 else np.zeros(h * w, dtype=np.float64)
    tmpl = unit.reshape(h, w)
    if n > 0 and float(np.linalg.norm(unit)) > 0.0:
        _assert_unit_l2("template_a", tmpl)
        _assert_unit_l2("template_b", tmpl)
    return SplitResult(
        accepted=False,
        labels=np.zeros(n, dtype=np.int64),
        template_a=np.array(tmpl, copy=True),
        template_b=np.array(tmpl, copy=True),
        sse_one=float(sse_one),
        sse_two=float(sse_one),
        relative_reduction=float(relative_reduction),
        n_a=n,
        n_b=0,
        n_pixels=h * w,
        n_trials=int(n_trials),
        best_trial=-1,
    )


def _rng(seed: np.random.Generator | int | None) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def split_patches_rank1(
    patches: Sequence[np.ndarray],
    *,
    base_template: np.ndarray | None = None,
    epsilon: float = 0.1,
    n_trials: int = 10,
    min_relative_reduction: float = 0.1,
    min_size: int = 2,
    max_refine: int = 10,
    rng: np.random.Generator | int | None = None,
) -> SplitResult:
    """Find a 2-way rank-1 partition via randomized ±ε perturbations.

    For each of ``n_trials``:

    1. Draw a random direction and set ``ε`` with
       ``||ε|| = epsilon * ||base||`` (``base`` is ``base_template`` or the
       mean patch).
    2. Form candidate words ``base+ε`` and ``base-ε``.
    3. Assign each patch to the candidate with larger explained energy, then
       refine by alternating PC1 refit and reassignment inside the patch group.
    4. Score residual SSE of the refined partition.

    Keep the trial with the smallest SSE. Accept if
    ``(sse_one - sse_two) / sse_one >= min_relative_reduction``, where
    ``sse_one`` is residual SSE under a single shared PC1 of all patches.
    """
    if min_size < 1:
        raise ValueError(f"min_size must be >= 1, got {min_size}")
    if not (0.0 <= float(min_relative_reduction) <= 1.0):
        raise ValueError(
            f"min_relative_reduction must be in [0, 1], got {min_relative_reduction}"
        )
    if max_refine < 0:
        raise ValueError(f"max_refine must be >= 0, got {max_refine}")
    if float(epsilon) < 0.0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon}")
    if int(n_trials) < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")

    data, h, w = _stack_patches(patches)
    n = data.shape[0]
    pc1_all = _pc1_unit(data)
    sse_one = _sse_vs_unit(data, pc1_all)

    if n < 2 * int(min_size):
        return _empty_split(data, h, w, sse_one=sse_one, n_trials=int(n_trials))

    base = _resolve_base(data, h, w, base_template)
    gen = _rng(rng)

    best_sse = sse_one
    best_labels: np.ndarray | None = None
    best_a = pc1_all
    best_b = pc1_all
    best_trial = -1
    best_rel = 0.0

    for trial in range(int(n_trials)):
        cand_a, cand_b = _sample_pm_epsilon(base, epsilon=float(epsilon), rng=gen)
        refined = _refine_from_units(
            data,
            cand_a,
            cand_b,
            min_size=int(min_size),
            max_refine=int(max_refine),
        )
        if refined is None:
            continue
        labels, sse_two, unit_a, unit_b = refined
        if sse_two < best_sse - 1e-12:
            best_sse = sse_two
            best_labels = labels
            best_a, best_b = unit_a, unit_b
            best_trial = trial
            best_rel = (
                0.0 if sse_one <= 0.0 else float((sse_one - sse_two) / sse_one)
            )

    if best_labels is None:
        return _empty_split(
            data, h, w, sse_one=sse_one, n_trials=int(n_trials), relative_reduction=0.0
        )

    n_a = int(np.sum(best_labels == 0))
    n_b = int(np.sum(best_labels == 1))
    accepted = best_rel >= float(min_relative_reduction) and best_sse < sse_one

    tmpl_a = pc1_all.reshape(h, w)
    tmpl_b = pc1_all.reshape(h, w)
    if not accepted:
        _assert_unit_l2("template_a", tmpl_a)
        _assert_unit_l2("template_b", tmpl_b)
        return SplitResult(
            accepted=False,
            labels=np.zeros(n, dtype=np.int64),
            template_a=tmpl_a,
            template_b=tmpl_b,
            sse_one=float(sse_one),
            sse_two=float(sse_one),
            relative_reduction=float(best_rel),
            n_a=n,
            n_b=0,
            n_pixels=h * w,
            n_trials=int(n_trials),
            best_trial=int(best_trial),
        )

    tmpl_a = best_a.reshape(h, w)
    tmpl_b = best_b.reshape(h, w)
    _assert_unit_l2("template_a", tmpl_a)
    _assert_unit_l2("template_b", tmpl_b)
    return SplitResult(
        accepted=True,
        labels=best_labels,
        template_a=tmpl_a,
        template_b=tmpl_b,
        sse_one=float(sse_one),
        sse_two=float(best_sse),
        relative_reduction=float(best_rel),
        n_a=n_a,
        n_b=n_b,
        n_pixels=h * w,
        n_trials=int(n_trials),
        best_trial=int(best_trial),
    )


def split_one_template(
    patches: Sequence[np.ndarray],
    *,
    base_template: np.ndarray | None = None,
    n_trials: int = 10,
    n_refine: int = 10,
    epsilon: float = 0.1,
    min_size: int = 2,
    rng: np.random.Generator | int | None = None,
) -> SplitOneResult:
    """Split one template's patches into two unit-L2 rank-1 atoms.

    Runs ``n_trials`` (K) independent ±ε initializations. Each trial performs
    up to ``n_refine`` (S) assign→PC1-refit iterations. Keeps the trial with
    the smallest residual SSE.

    Returns the two templates and the patches assigned to each. Asserts that
    both templates (and the unit-normalized base word) have L2 norm 1.

    Raises
    ------
    ValueError
        If fewer than ``2 * min_size`` patches, or no trial yields a valid
        bipartition with both sides at least ``min_size``.
    """
    if min_size < 1:
        raise ValueError(f"min_size must be >= 1, got {min_size}")
    if n_refine < 0:
        raise ValueError(f"n_refine must be >= 0, got {n_refine}")
    if float(epsilon) < 0.0:
        raise ValueError(f"epsilon must be >= 0, got {epsilon}")
    if int(n_trials) < 1:
        raise ValueError(f"n_trials must be >= 1, got {n_trials}")

    patch_list = [np.asarray(p, dtype=np.float64) for p in patches]
    data, h, w = _stack_patches(patch_list)
    n = data.shape[0]
    if n < 2 * int(min_size):
        raise ValueError(
            f"need at least {2 * int(min_size)} patches to split, got {n}"
        )

    pc1_all = _pc1_unit(data)
    sse_one = _sse_vs_unit(data, pc1_all)
    _assert_unit_l2("shared_pc1", pc1_all)

    base = _resolve_base(data, h, w, base_template)
    base_unit = _as_unit(base)
    if float(np.linalg.norm(base_unit)) <= 0.0:
        raise ValueError("base_template (or mean patch) has zero L2 norm")
    _assert_unit_l2("base_template", base_unit)

    gen = _rng(rng)
    best_sse = float("inf")
    best_labels: np.ndarray | None = None
    best_a = pc1_all
    best_b = pc1_all
    best_trial = -1

    for trial in range(int(n_trials)):
        cand_a, cand_b = _sample_pm_epsilon(
            base_unit, epsilon=float(epsilon), rng=gen
        )
        refined = _refine_from_units(
            data,
            cand_a,
            cand_b,
            min_size=int(min_size),
            max_refine=int(n_refine),
        )
        if refined is None:
            continue
        labels, sse_two, unit_a, unit_b = refined
        if sse_two < best_sse - 1e-12:
            best_sse = float(sse_two)
            best_labels = labels
            best_a, best_b = unit_a, unit_b
            best_trial = trial

    if best_labels is None:
        raise ValueError(
            f"no valid bipartition found in {int(n_trials)} trials "
            f"(min_size={int(min_size)})"
        )

    template_a = best_a.reshape(h, w)
    template_b = best_b.reshape(h, w)
    _assert_unit_l2("template_a", template_a)
    _assert_unit_l2("template_b", template_b)

    patches_a = [patch_list[i] for i, lab in enumerate(best_labels) if lab == 0]
    patches_b = [patch_list[i] for i, lab in enumerate(best_labels) if lab == 1]
    rel = 0.0 if sse_one <= 0.0 else float((sse_one - best_sse) / sse_one)

    return SplitOneResult(
        template_a=template_a,
        template_b=template_b,
        patches_a=patches_a,
        patches_b=patches_b,
        labels=best_labels,
        sse_one=float(sse_one),
        sse_two=float(best_sse),
        relative_reduction=rel,
        best_trial=int(best_trial),
        n_trials=int(n_trials),
        n_refine=int(n_refine),
    )


def split_template_matchings(
    signal: Sequence[Window] | dict[str, np.ndarray],
    template: Template,
    matchings: Sequence[Matching],
    *,
    epsilon: float = 0.1,
    n_trials: int = 10,
    min_relative_reduction: float = 0.1,
    min_size: int = 2,
    max_refine: int = 10,
    rng: np.random.Generator | int | None = None,
) -> tuple[SplitResult, list[Matching], list[Matching]]:
    """Extract patches for ``template`` and run :func:`split_patches_rank1`.

    Uses the catalog / input ``template`` data as the base dictionary word for
    ±ε perturbations. Returns ``(result, matchings_a, matchings_b)``. When the
    split is rejected, ``matchings_a`` is all input matchings for the template
    and ``matchings_b`` is empty.
    """
    used = [m for m in matchings if int(m.template_id) == int(template.template_id)]
    if not used:
        used = list(matchings)
    if not used:
        raise ValueError(
            f"template_id {template.template_id} has no matchings to split"
        )
    width = int(template.width)
    n_freq = int(template.n_freq)
    patches = _extract_patches(signal, used, width=width, n_freq=n_freq)
    if len(patches) != len(used):
        kept: list[Matching] = []
        kept_patches: list[np.ndarray] = []
        for matching in used:
            one = _extract_patches(signal, [matching], width=width, n_freq=n_freq)
            if one:
                kept.append(matching)
                kept_patches.append(one[0])
        used, patches = kept, kept_patches
    if not patches:
        raise ValueError(
            f"template_id {template.template_id}: no patches could be extracted"
        )

    base = template.as_rank1 if template.n_components == 1 else template.basis[0]
    result = split_patches_rank1(
        patches,
        base_template=base,
        epsilon=epsilon,
        n_trials=n_trials,
        min_relative_reduction=min_relative_reduction,
        min_size=min_size,
        max_refine=max_refine,
        rng=rng,
    )
    if not result.accepted:
        return result, list(used), []
    matchings_a = [m for m, lab in zip(used, result.labels) if lab == 0]
    matchings_b = [m for m, lab in zip(used, result.labels) if lab == 1]
    return result, matchings_a, matchings_b
