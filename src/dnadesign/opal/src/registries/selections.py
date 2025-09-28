"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/registries/selections.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np

# Registry: name -> (either)
#   (A) selection_fn(ids, scores, *, top_k, tie_handling, **extra) -> dict
#   (B) factory(params) -> selection_fn(...)
_REG_S: Dict[str, Callable[..., Any]] = {}


def register_selection(name: str):
    """Decorator to register a selection strategy (either as a function or as a factory)."""

    def _wrap(obj: Callable[..., Any]):
        if name in _REG_S:
            raise ValueError(f"selection '{name}' already registered")
        _REG_S[name] = obj
        return obj

    return _wrap


def get_selection(
    name: str, params: Dict[str, Any] | None = None
) -> Callable[..., Any]:
    """Return a selection callable, invoking a factory if needed."""
    if name not in _REG_S:
        raise KeyError(f"selection '{name}' not found. Available: {sorted(_REG_S)}")
    obj = _REG_S[name]
    # Try as factory(params) -> callable
    try:
        return obj(params or {})  # type: ignore[misc]
    except TypeError:
        # Not a factory; return as-is (plain callable selection)
        return obj  # type: ignore[return-value]


def list_selections() -> List[str]:
    return sorted(_REG_S)


# --- canonical result adapter for selection plugins --------------


def _stable_sort_indices(
    ids: np.ndarray, scores: np.ndarray, objective: str
) -> np.ndarray:
    """
    Stable deterministic sort:
      - maximize: (-score, id)
      - minimize: ( score, id)
    """
    ids = ids.astype(str)
    scores = scores.astype(float)
    # push NaNs to the bottom deterministically
    if objective == "maximize":
        primary = np.where(np.isfinite(scores), -scores, np.inf)
    else:
        primary = np.where(np.isfinite(scores), scores, np.inf)
    # np.lexsort: last key is primary
    return np.lexsort((ids, primary))


def _ranks_from_sorted_scores(sorted_scores: np.ndarray, method: str) -> np.ndarray:
    """
    Compute ranks for already-sorted scores (descending if maximize, ascending if minimize).
    method: 'competition_rank' | 'dense_rank' | 'ordinal'
    """
    n = len(sorted_scores)
    if n == 0:
        return np.array([], dtype=int)

    if method == "ordinal":
        return np.arange(1, n + 1, dtype=int)

    # group ties by isclose
    ranks = np.empty(n, dtype=int)
    i = 0
    current_rank = 1
    while i < n:
        j = i + 1
        # treat near-equal as ties
        while (
            j < n
            and np.isfinite(sorted_scores[i])
            and np.isfinite(sorted_scores[j])
            and np.isclose(sorted_scores[j], sorted_scores[i], rtol=1e-12, atol=1e-12)
        ):
            j += 1
        # also group NaNs together (they already sit at the end)
        if not np.isfinite(sorted_scores[i]):
            # all remaining are NaN → give ordinal ranks
            ranks[i:] = np.arange(i + 1, n + 1, dtype=int)
            break
        ranks[i:j] = current_rank
        if method == "dense_rank":
            current_rank += 1
        else:  # competition_rank
            current_rank = j + 1  # jump by tie size
        i = j
    return ranks


def normalize_selection_result(
    raw: Dict[str, Any],
    *,
    ids: np.ndarray,
    scores: np.ndarray,
    top_k: int,
    tie_handling: str = "competition_rank",
    objective: str = "maximize",
) -> Dict[str, np.ndarray]:
    """
    Normalize arbitrary plugin outputs into the canonical shape:
      { 'order_idx': np.ndarray[int],
        'ranks': np.ndarray[int],
        'selected_bool': np.ndarray[bool] }
    If a field is missing, we recompute it deterministically.
    """
    n = len(ids)
    ids = np.asarray(ids)
    scores = np.asarray(scores, dtype=float)

    # 1) order_idx
    order = (
        np.asarray(raw.get("order_idx"))
        if "order_idx" in raw and raw.get("order_idx") is not None
        else _stable_sort_indices(ids, scores, objective)
    )
    # sanity: make order a valid permutation
    if not (order.ndim == 1 and order.size == n):
        order = _stable_sort_indices(ids, scores, objective)

    # 2) ranks (aligned to order)
    ranks = raw.get("ranks")
    if ranks is None:
        # try common aliases
        ranks = raw.get("rank") or raw.get("rank_competition") or raw.get("dense_ranks")
    if ranks is None:
        # recompute from scores in the order we choose
        # compute "sorted" scores in the chosen order, respecting objective
        sorted_scores = scores[order] if objective == "minimize" else scores[order]
        # NOTE: _ranks_from_sorted_scores expects the array already ordered according to the primary sort,
        #       which `order` is by objective + id.
        ranks = _ranks_from_sorted_scores(sorted_scores, tie_handling)
    else:
        ranks = np.asarray(ranks)
        if ranks.ndim != 1 or ranks.size != n:
            # fall back to recompute
            sorted_scores = scores[order]
            ranks = _ranks_from_sorted_scores(sorted_scores, tie_handling)

    # 3) selected_bool (aligned to order)
    selected = raw.get("selected_bool")
    if selected is None:
        selected = ranks <= int(top_k)
    else:
        selected = np.asarray(selected, dtype=bool)
        if selected.ndim != 1 or selected.size != n:
            selected = ranks <= int(top_k)

    return {
        "order_idx": order.astype(int, copy=False),
        "ranks": ranks.astype(int, copy=False),
        "selected_bool": selected.astype(bool, copy=False),
    }
