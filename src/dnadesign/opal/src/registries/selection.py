"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/registries/selection.py

Registers selection strategies and loads built-in selection modules. Provides
selection lookup and contract enforcement helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import inspect
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Dict, List

import numpy as np

from ..core.round_context import PluginCtx
from ..core.selection_contracts import (
    extract_selection_plugin_params,
    resolve_selection_objective_mode,
)
from ..core.utils import OpalError
from .loader import load_builtin_modules


@dataclass(frozen=True)
class _SelectionRegistration:
    obj: Callable[..., Any]
    is_factory: bool


# Registry: name -> explicit registration mode + callable
_REG_S: Dict[str, _SelectionRegistration] = {}
_BUILTINS_LOADED = False
_REQUIRED_SELECTION_ARGS = ("ids", "scores", "top_k", "objective", "tie_handling")
_RUNTIME_SELECTION_ARGS = frozenset(
    {
        "ids",
        "scores",
        "top_k",
        "objective",
        "tie_handling",
        "ctx",
        "scalar_uncertainty",
    }
)


def _dbg(msg: str) -> None:
    if str(os.getenv("OPAL_DEBUG", "")).strip().lower() in ("1", "true", "yes", "on"):
        print(f"[opal.debug.selection] {msg}", file=sys.stderr)


def _normalize_objective_mode(objective: str) -> str:
    return resolve_selection_objective_mode(
        {"objective_mode": objective},
        error_cls=ValueError,
        field_prefix="objective",
    )


def _ensure_builtins_loaded() -> None:
    """Import package-shipped selection modules that self-register via @register_selection."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    load_builtin_modules("dnadesign.opal.src.selection", label="selection", debug=_dbg)
    _BUILTINS_LOADED = True


def register_selection(name: str, *, factory: bool = False):
    """Decorator to register a selection strategy or an explicit factory."""

    def _wrap(obj: Callable[..., Any]):
        if name in _REG_S:
            raise ValueError(f"selection '{name}' already registered")
        _REG_S[name] = _SelectionRegistration(obj=obj, is_factory=bool(factory))
        return obj

    return _wrap


def _looks_like_selection_callable(obj: Callable[..., Any]) -> bool:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return False
    for arg in _REQUIRED_SELECTION_ARGS:
        param = sig.parameters.get(arg)
        if param is None:
            return False
        if param.kind is inspect.Parameter.POSITIONAL_ONLY:
            return False
    return True


def _missing_selection_signature_args(obj: Callable[..., Any]) -> list[str] | None:
    try:
        sig = inspect.signature(obj)
    except (TypeError, ValueError):
        return None
    names = set(sig.parameters.keys())
    seen = [arg for arg in _REQUIRED_SELECTION_ARGS if arg in names]
    if not seen:
        return None
    return [arg for arg in _REQUIRED_SELECTION_ARGS if arg not in names]


def _resolve_selection_callable(
    name: str,
    obj: Callable[..., Any],
    params: Dict[str, Any] | None,
    *,
    is_factory: bool,
) -> Callable[..., Any]:
    def _validate_required_params(fn: Callable[..., Any]) -> None:
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError) as e:
            raise TypeError(f"selection '{name}' has unreadable callable signature.") from e
        provided = set(_RUNTIME_SELECTION_ARGS)
        provided.update(extract_selection_plugin_params(params or {}).keys())
        missing: list[str] = []
        for p in sig.parameters.values():
            if p.kind is inspect.Parameter.VAR_POSITIONAL:
                continue
            if p.kind is inspect.Parameter.VAR_KEYWORD:
                continue
            if p.kind is inspect.Parameter.POSITIONAL_ONLY:
                if p.default is inspect.Parameter.empty:
                    missing.append(p.name)
                continue
            if p.default is inspect.Parameter.empty and p.name not in provided:
                missing.append(p.name)
        if missing:
            raise TypeError(
                (
                    f"selection '{name}' has required parameter(s) not provided by runtime/config: {missing}. "
                    "Pass required plugin params in selection.params."
                )
            )

    if is_factory:
        factory_contract = getattr(obj, "__opal_contract__", None)
        fn = obj(params or {})  # may raise, intentionally not masked
        if not callable(fn):
            raise TypeError(f"selection '{name}' factory must return a callable, got {type(fn).__name__}.")
        if factory_contract is not None and getattr(fn, "__opal_contract__", None) is None:
            setattr(fn, "__opal_contract__", factory_contract)
        if not _looks_like_selection_callable(fn):
            raise TypeError(
                (
                    f"selection '{name}' resolved to an invalid callable signature; "
                    f"expected parameters {_REQUIRED_SELECTION_ARGS}."
                )
            )
        _validate_required_params(fn)
        return fn

    if _looks_like_selection_callable(obj):
        _validate_required_params(obj)
        return obj
    missing = _missing_selection_signature_args(obj)
    if missing:
        raise TypeError(
            (f"selection '{name}' has invalid callable signature; missing required parameter(s): {missing}.")
        )
    raise TypeError(
        (
            f"selection '{name}' appears to be a factory. "
            "Register factories with @register_selection(name, factory=True)."
        )
    )


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
        objective,
        tie_handling,
        ctx: PluginCtx | None = None,
        **kw,
    ):
        if ctx is not None:
            # Ensure caller passed a PluginCtx for this plugin instance
            ctx.precheck_requires(stage="selection")
        try:
            out = fn(
                ids=ids,
                scores=scores,
                top_k=top_k,
                objective=objective,
                tie_handling=tie_handling,
                ctx=ctx,
                **kw,
            )
        except Exception:
            if ctx is not None:
                ctx.reset_stage_state()
            raise
        if ctx is not None:
            ctx.postcheck_produces(stage="selection")
        return out

    # propagate the contract so rctx.for_plugin(...) sees it
    if contract is not None:
        setattr(_wrapped, "__opal_contract__", contract)
    return _wrapped


def get_selection(name: str, params: Dict[str, Any] | None = None) -> Callable[..., Any]:
    """Return a selection callable for the registered strategy."""
    _ensure_builtins_loaded()
    if name not in _REG_S:
        raise KeyError(f"selection '{name}' not found. Available: {sorted(_REG_S)}")
    entry = _REG_S[name]
    fn = _resolve_selection_callable(name, entry.obj, params, is_factory=entry.is_factory)
    return _wrap_for_ctx_enforcement(name, fn)


def list_selections() -> List[str]:
    _ensure_builtins_loaded()
    return sorted(_REG_S)


# --- canonical result adapter for selection plugins --------------


@dataclass(frozen=True)
class SelectionResultV2:
    order_idx: np.ndarray
    score: np.ndarray


def _coerce_selection_order_idx(raw_order_idx: Any, *, plugin_name: str) -> np.ndarray:
    order_raw = np.asarray(raw_order_idx)
    order_flat = order_raw.reshape(-1)
    if order_flat.dtype.kind == "b":
        raise OpalError(f"Selection plugin '{plugin_name}' must return integer order_idx; got boolean values.")
    if order_flat.dtype.kind not in {"i", "u", "f"}:
        raise OpalError(f"Selection plugin '{plugin_name}' must return numeric order_idx values.")
    if order_flat.dtype.kind == "f":
        if not np.all(np.isfinite(order_flat)):
            raise OpalError(f"Selection plugin '{plugin_name}' returned non-finite order_idx values.")
        if not np.all(np.equal(order_flat, np.floor(order_flat))):
            raise OpalError(f"Selection plugin '{plugin_name}' returned non-integral order_idx values.")
    return order_flat.astype(int, copy=False)


def _coerce_selection_score(raw_score: Any, *, plugin_name: str) -> np.ndarray:
    score_raw = np.asarray(raw_score)
    score_flat = score_raw.reshape(-1)
    if score_flat.dtype.kind == "b":
        raise OpalError(f"Selection plugin '{plugin_name}' must return numeric selection scores, not booleans.")
    if score_flat.dtype.kind not in {"i", "u", "f"}:
        raise OpalError(f"Selection plugin '{plugin_name}' must return numeric selection scores.")
    score = score_flat.astype(float, copy=False)
    if not np.all(np.isfinite(score)):
        raise OpalError(f"Selection plugin '{plugin_name}' returned non-finite selection scores.")
    return score


def validate_selection_result(raw: Dict[str, Any], *, plugin_name: str, expected_len: int) -> SelectionResultV2:
    if not isinstance(raw, dict):
        raise OpalError(f"Selection plugin '{plugin_name}' must return a mapping.")
    if "order_idx" not in raw:
        raise OpalError(f"Selection plugin '{plugin_name}' must return 'order_idx'.")
    if "score" not in raw:
        raise OpalError(f"Selection plugin '{plugin_name}' must return 'score'.")

    order_idx = _coerce_selection_order_idx(raw["order_idx"], plugin_name=plugin_name)
    score = _coerce_selection_score(raw["score"], plugin_name=plugin_name)
    if order_idx.size != expected_len:
        raise OpalError(
            f"Selection plugin '{plugin_name}' returned order_idx length {order_idx.size}, expected {expected_len}."
        )
    if score.size != expected_len:
        raise OpalError(
            f"Selection plugin '{plugin_name}' returned score length {score.size}, expected {expected_len}."
        )
    if expected_len == 0:
        return SelectionResultV2(order_idx=order_idx, score=score)
    if np.unique(order_idx).size != expected_len:
        raise OpalError(f"Selection plugin '{plugin_name}' returned non-unique order_idx entries.")
    if int(np.min(order_idx)) < 0 or int(np.max(order_idx)) >= expected_len:
        raise OpalError(f"Selection plugin '{plugin_name}' returned out-of-range order_idx entries.")

    return SelectionResultV2(order_idx=order_idx, score=score)


def _stable_sort_indices(ids: np.ndarray, scores: np.ndarray, objective: str) -> np.ndarray:
    """
    Stable deterministic sort:
      - maximize: (-score, id)
      - minimize: ( score, id)
    """
    ids = ids.astype(str)
    scores = scores.astype(float)
    mode = _normalize_objective_mode(objective)
    # push NaNs to the bottom deterministically
    if mode == "maximize":
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

    if method not in {"competition_rank", "dense_rank", "ordinal"}:
        raise ValueError(
            f"Invalid tie_handling mode: {method!r}. Expected one of ['competition_rank', 'dense_rank', 'ordinal']."
        )
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
    tie_handling: str,
    objective: str,
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
        order_raw = np.asarray(order)
        if order_raw.ndim != 1:
            raise ValueError("order_idx must be 1D when provided.")
        if order_raw.size != n:
            raise ValueError(f"order_idx length mismatch: got {order_raw.size}, expected {n}.")
        if order_raw.dtype.kind == "b":
            raise ValueError("order_idx must be an integer permutation, not boolean.")
        if order_raw.dtype.kind not in {"i", "u", "f"}:
            raise ValueError("order_idx must be numeric when provided.")
        if order_raw.dtype.kind == "f":
            if not np.all(np.isfinite(order_raw)):
                raise ValueError("order_idx contains non-finite values.")
            if not np.all(np.equal(order_raw, np.floor(order_raw))):
                raise ValueError("order_idx must contain integral values.")
        order = order_raw.astype(int, copy=False)
        if np.unique(order).size != n or int(np.min(order)) < 0 or int(np.max(order)) >= n:
            raise ValueError("order_idx must be a permutation of [0..n-1].")

    # Sorted view of scores (objective already baked into order)
    sorted_scores = scores[order]

    # Always compute both rank flavors in the **sorted** view.
    # Skip duplicate computation when requested tie mode is already competition_rank.
    if tie_handling == "competition_rank":
        ranks_sorted_comp = _ranks_from_sorted_scores(sorted_scores, tie_handling)
        ranks_sorted_current = ranks_sorted_comp
    else:
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
