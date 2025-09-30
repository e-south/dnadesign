"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/registries/selections.py

Selection registry with RoundCtx contract enforcement.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List

import numpy as np

from ..round_context import PluginCtx

# Registry: name -> (either)
#   (A) selection_fn(...)
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


def _wrap_for_ctx_enforcement(name: str, fn: Callable[..., Any]) -> Callable[..., Any]:
    """
    Return a function that enforces PluginCtx contract pre/post checks if a ctx is provided.
    We preserve the original __opal_contract__ so run_round can derive a PluginCtx from it.
    """
    contract = getattr(fn, "__opal_contract__", None)

    def _wrapped(
        *,
        ids,
        scores,
        top_k,
        objective="maximize",
        tie_handling="competition_rank",
        ctx: PluginCtx | None = None,
        **kw,
    ):
        if ctx is not None:
            # Ensure caller passed a PluginCtx for this plugin instance
            try:
                ctx.precheck_requires()
            except Exception:
                raise
        out = fn(
            ids=ids,
            scores=scores,
            top_k=top_k,
            objective=objective,
            tie_handling=tie_handling,
            ctx=ctx,
            **kw,
        )
        if ctx is not None:
            try:
                ctx.postcheck_produces()
            except Exception:
                raise
        return out

    # propagate the contract so rctx.for_plugin(...) sees it
    if contract is not None:
        setattr(_wrapped, "__opal_contract__", contract)
    return _wrapped


def get_selection(
    name: str, params: Dict[str, Any] | None = None
) -> Callable[..., Any]:
    """Return a selection callable, invoking a factory if needed, wrapped for ctx enforcement."""
    if name not in _REG_S:
        raise KeyError(f"selection '{name}' not found. Available: {sorted(_REG_S)}")
    obj = _REG_S[name]
    # Try as factory(params) -> callable
    try:
        fn = obj(params or {})  # type: ignore[misc]
    except TypeError:
        fn = obj  # not a factory
    return _wrap_for_ctx_enforcement(name, fn)


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
    if str(objective).lower().startswith("max"):
        primary = np.where(np.isfinite(scores), -scores, np.inf)
    else:
        primary = np.where(np.isfinite(scores), scores, np.inf)
    return np.lexsort((ids, primary))  # last key wins


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

    ranks = np.empty(n, dtype=int)
    i = 0
    current_rank = 1
    while i < n:
        j = i + 1
        while (
            j < n
            and np.isfinite(sorted_scores[i])
            and np.isfinite(sorted_scores[j])
            and np.isclose(sorted_scores[j], sorted_scores[i], rtol=1e-12, atol=1e-12)
        ):
            j += 1
        if not np.isfinite(sorted_scores[i]):
            ranks[i:] = np.arange(i + 1, n + 1, dtype=int)
            break
        ranks[i:j] = current_rank
        if method == "dense_rank":
            current_rank += 1
        else:  # competition_rank
            current_rank = j + 1
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
    Normalize arbitrary plugin outputs into the canonical, **original-id order**:
      {
        'order_idx'        : np.ndarray[int]  (best-first sort indices)
        'ranks'            : np.ndarray[int]  (tie_handling-specific)
        'rank_competition' : np.ndarray[int]  (always competition-style)
        'selected_bool'    : np.ndarray[bool] (based on 'ranks' and top_k)
      }
    The normalizer computes both rank styles deterministically and aligns all
    arrays back to the original `ids` order (not the sorted order).
    """
    ids = np.asarray(ids, dtype=str)
    scores = np.asarray(scores, dtype=float)
    n = ids.shape[0]

    # Resolve or compute stable order
    order = raw.get("order_idx")
    if order is None:
        order = _stable_sort_indices(ids, scores, objective)
    else:
        order = np.asarray(order)
        if order.ndim != 1 or order.size != n:
            order = _stable_sort_indices(ids, scores, objective)
    order = order.astype(int, copy=False)

    # Sorted view of scores (objective already baked into order)
    sorted_scores = scores[order]

    # Always compute both rank flavors in the **sorted** view
    ranks_sorted_current = _ranks_from_sorted_scores(sorted_scores, tie_handling)
    ranks_sorted_comp = _ranks_from_sorted_scores(sorted_scores, "competition_rank")

    # Selection uses the requested tie_handling
    selected_sorted = ranks_sorted_current <= int(top_k)

    # Map back to original id order
    ranks_current = np.empty(n, dtype=int)
    ranks_comp = np.empty(n, dtype=int)
    selected_bool = np.empty(n, dtype=bool)
    ranks_current[order] = ranks_sorted_current
    ranks_comp[order] = ranks_sorted_comp
    selected_bool[order] = selected_sorted

    return {
        "order_idx": order,
        "ranks": ranks_current,
        "rank_competition": ranks_comp,
        "selected_bool": selected_bool,
    }
