"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/round_context.py

OPAL Round Context (runtime carrier) + plugin contract DSL.

Design goals (pragmatic & assertive):
- JSON-centric: accept any JSON-serializable value (nested dicts/lists).
- Normalize common scientific types (NumPy/Pandas/Path/dataclasses) to plain Python.
- Reject non-finite numbers and non-deterministic/opaque types (sets, bytes, callables).
- Keep contracts strict: plugins may only write under their own namespace and to
  keys they declared in 'produces'.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import dataclasses as _dc
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

__all__ = [
    "RoundCtx",
    "PluginCtx",
    "roundctx_contract",
    "RoundCtxPathError",
    "RoundCtxTypeError",
    "RoundCtxContractError",
    "PluginRegistryView",
]

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class RoundCtxPathError(RuntimeError):
    def __init__(self, path: str, reason: str) -> None:
        super().__init__(f"[RoundCtx] Invalid path '{path}': {reason}")
        self.path = path
        self.reason = reason


class RoundCtxTypeError(RuntimeError):
    def __init__(self, path: str, value: Any, expected: str) -> None:
        t = type(value).__name__
        super().__init__(
            f"[RoundCtx] Unsupported type for '{path}': {t} (expected {expected})"
        )
        self.path = path
        self.value = value
        self.expected = expected


class RoundCtxContractError(RuntimeError):
    def __init__(
        self,
        *,
        category: str,
        name: str,
        missing_requires: Optional[List[str]] = None,
        missing_produces: Optional[List[str]] = None,
        illegal_writes: Optional[List[str]] = None,
        msg: Optional[str] = None,
    ) -> None:
        parts = [f"[RoundCtx] Contract error for {category}/{name}"]
        if missing_requires:
            parts.append(f"missing requires={missing_requires}")
        if missing_produces:
            parts.append(f"missing produces={missing_produces}")
        if illegal_writes:
            parts.append(f"illegal writes={illegal_writes}")
        if msg:
            parts.append(msg)
        super().__init__(", ".join(parts))
        self.category = category
        self.name = name
        self.missing_requires = missing_requires or []
        self.illegal_writes = illegal_writes or []
        self.missing_produces = missing_produces or []


# ---------------------------------------------------------------------------
# Contracts & plugin registry view
# ---------------------------------------------------------------------------

_ALLOWED_ROOTS = {
    "core",
    "model",
    "objective",
    "selection",
    "transform_x",
    "transform_y",
    "yops",  # training-time Y-ops shared space (pipelines, stats, etc.)
}
_PATH_RE = re.compile(r"^(?P<root>[a-z0-9_]+)(?:/[a-z0-9_\-]+)+$")


@dataclass(frozen=True)
class Contract:
    category: str
    requires: Tuple[str, ...]
    produces: Tuple[str, ...]


def roundctx_contract(
    *,
    category: str,
    requires: Optional[List[str]] = None,
    produces: Optional[List[str]] = None,
):
    if category not in _ALLOWED_ROOTS - {"core", "yops"}:
        raise ValueError(f"roundctx_contract: invalid category={category!r}")
    req = tuple(requires or ())
    prod = tuple(produces or ())

    def _wrap(obj: Any) -> Any:
        setattr(
            obj,
            "__opal_contract__",
            Contract(category=category, requires=req, produces=prod),
        )
        return obj

    return _wrap


@dataclass(frozen=True)
class PluginRegistryView:
    model: str
    objective: str
    selection: str
    transform_x: str
    transform_y: str

    def name_for(self, category: str) -> str:
        try:
            return getattr(self, category)
        except AttributeError as e:
            raise ValueError(f"Unknown plugin category '{category}'") from e


# ---------------------------------------------------------------------------
# JSON normalization
# ---------------------------------------------------------------------------


def _to_jsonable(path: str, val: Any) -> Any:
    """
    Recursively normalize 'val' to a JSON-serializable structure:
      - primitives: None, bool, int, float (finite), str
      - lists/tuples → list
      - dict with string keys → dict
      - NumPy scalars/arrays → Python scalar/list
      - Pandas Series/DataFrame → list / list-of-dicts
      - dataclasses → asdict
      - pathlib.Path → str
    Rejects: bytes, sets, callables, non-string dict keys, NaN/Inf floats.
    """
    import math as _math

    # Dataclasses
    if _dc.is_dataclass(val):
        val = _dc.asdict(val)

    # pathlib.Path
    if isinstance(val, Path):
        return str(val)

    # NumPy family
    try:
        import numpy as _np  # type: ignore

        if isinstance(val, (_np.floating, _np.integer, _np.bool_)):  # type: ignore[attr-defined]
            val = val.item()
        elif isinstance(val, _np.ndarray):
            val = val.tolist()
    except Exception:
        pass

    # Pandas family
    try:
        import pandas as _pd  # type: ignore

        if isinstance(val, _pd.Series):
            val = val.tolist()
        elif isinstance(val, _pd.DataFrame):
            # deterministic order: records
            val = val.to_dict(orient="records")
    except Exception:
        pass

    # Primitives
    if val is None:
        return None
    if isinstance(val, bool):
        return bool(val)
    if isinstance(val, int):
        return int(val)
    if isinstance(val, float):
        if not _math.isfinite(val):
            raise RoundCtxTypeError(path, val, "finite float")
        return float(val)
    if isinstance(val, str):
        return val

    # Containers
    if isinstance(val, (list, tuple)):
        return [_to_jsonable(f"{path}[{i}]", v) for i, v in enumerate(list(val))]

    if isinstance(val, dict):
        out: Dict[str, Any] = {}
        for k, v in val.items():
            if not isinstance(k, str):
                raise RoundCtxTypeError(path, val, "dict with string keys")
            out[k] = _to_jsonable(f"{path}.{k}", v)
        return out

    # Explicit rejections for non-deterministic/opaque types
    if isinstance(val, (set, frozenset)):
        raise RoundCtxTypeError(path, val, "list (sets are not allowed)")
    if isinstance(val, (bytes, bytearray, memoryview)):
        raise RoundCtxTypeError(path, val, "str (bytes not allowed)")
    if callable(val):
        raise RoundCtxTypeError(path, val, "JSON-serializable value (not a callable)")

    # Last resort: reject
    raise RoundCtxTypeError(
        path,
        val,
        "JSON-serializable (None|bool|int|float(str); list/tuple; dict[str, ...])",
    )


def _assert_path(path: str) -> Tuple[str, List[str]]:
    m = _PATH_RE.match(path)
    if not m:
        raise RoundCtxPathError(
            path, "must match '<root>/<seg>[/<seg>...]' and use lowercase + [_-]"
        )
    root = m.group("root")
    if root not in _ALLOWED_ROOTS:
        raise RoundCtxPathError(path, f"root must be one of {sorted(_ALLOWED_ROOTS)}")
    parts = path.split("/")
    if len(parts) < 2:
        raise RoundCtxPathError(path, "must have at least two segments")
    return root, parts


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


@dataclass
class _Audit:
    consumed: Dict[Tuple[str, str], Set[str]] = field(default_factory=dict)
    produced: Dict[Tuple[str, str], Set[str]] = field(default_factory=dict)

    def add_consumed(self, cat: str, name: str, path: str) -> None:
        self.consumed.setdefault((cat, name), set()).add(path)

    def add_produced(self, cat: str, name: str, path: str) -> None:
        self.produced.setdefault((cat, name), set()).add(path)

    def snapshot(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for (cat, name), paths in sorted(self.consumed.items()):
            out[f"core/contracts/{cat}/{name}/consumed"] = sorted(paths)
        for (cat, name), paths in sorted(self.produced.items()):
            out[f"core/contracts/{cat}/{name}/produced"] = sorted(paths)
        return out


# ---------------------------------------------------------------------------
# RoundCtx & PluginCtx
# ---------------------------------------------------------------------------


class RoundCtx:
    """
    Flat, strict runtime store. Plugins should use PluginCtx.
    System components (e.g., Y-ops pipeline) may write to shared 'yops/*' keys via RoundCtx.
    """

    def __init__(self, *, core: Mapping[str, Any], registry: PluginRegistryView):
        self._store: Dict[str, Any] = {}
        self._audit = _Audit()
        self._registry = registry
        for k, v in core.items():
            self.set_core(k, v)

    # ----- core -----
    def set_core(self, key: str, value: Any) -> None:
        if not key.startswith("core/"):
            raise RoundCtxPathError(key, "core keys must start with 'core/'")
        _assert_path(key)
        self._store[key] = _to_jsonable(key, value)

    # ----- generic -----
    def get(self, path: str, default: Any = None) -> Any:
        _assert_path(path)
        if path in self._store:
            return self._store[path]
        if default is not None:
            return default
        raise KeyError(f"[RoundCtx] Missing key: {path}")

    def has(self, path: str) -> bool:
        _assert_path(path)
        return path in self._store

    def set(self, path: str, value: Any, *, allow_overwrite: bool = False) -> None:
        _assert_path(path)
        normalized = _to_jsonable(path, value)
        if (
            not allow_overwrite
            and path in self._store
            and self._store[path] != normalized
        ):
            raise RoundCtxPathError(path, "overwrite not allowed (value differs)")
        self._store[path] = normalized

    # ----- plugin view -----
    def for_plugin(
        self, *, category: str, name: str, plugin: Optional[Any] = None
    ) -> "PluginCtx":
        if category not in _ALLOWED_ROOTS - {"core", "yops"}:
            raise ValueError(f"Unknown plugin category '{category}'")
        contract: Optional[Contract] = (
            getattr(plugin, "__opal_contract__", None) if plugin is not None else None
        )
        if contract is not None and contract.category != category:
            raise RoundCtxContractError(
                category=category, name=name, msg="decorator category/name mismatch"
            )
        effective = contract or Contract(
            category=category, requires=tuple(), produces=tuple()
        )
        return PluginCtx(
            round_ctx=self, category=category, name=name, contract=effective
        )

    # ----- snapshot -----
    def snapshot(self) -> Dict[str, Any]:
        out = dict(self._store)
        out.update(self._audit.snapshot())
        return out


@dataclass
class PluginCtx:
    round_ctx: RoundCtx
    category: str
    name: str
    contract: Contract

    def _expand(self, path: str) -> str:
        if "<self>" in path:
            path = path.replace("<self>", self.name)
        _assert_path(path)
        return path

    def _ensure_own_namespace(self, path: str) -> None:
        if not path.startswith(f"{self.category}/{self.name}/"):
            raise RoundCtxPathError(
                path, f"plugin may only write under '{self.category}/{self.name}/...'"
            )

    def get(self, path: str, default: Any = None) -> Any:
        p = self._expand(path)
        val = self.round_ctx.get(p, default=default)
        self.round_ctx._audit.add_consumed(self.category, self.name, p)
        return val

    def set(self, path: str, value: Any) -> None:
        p = self._expand(path)
        self._ensure_own_namespace(p)
        expanded_produces = [self._expand(x) for x in self.contract.produces]
        if p not in expanded_produces:
            raise RoundCtxContractError(
                category=self.category,
                name=self.name,
                illegal_writes=[p],
                msg="write to a key not declared in produces",
            )
        self.round_ctx.set(p, value, allow_overwrite=False)
        self.round_ctx._audit.add_produced(self.category, self.name, p)

    def precheck_requires(self) -> None:
        missing: List[str] = []
        for req in self.contract.requires:
            p = self._expand(req)
            try:
                self.round_ctx.get(p)
            except KeyError:
                missing.append(p)
        if missing:
            raise RoundCtxContractError(
                category=self.category, name=self.name, missing_requires=missing
            )

    def postcheck_produces(self) -> None:
        missing: List[str] = []
        for prod in self.contract.produces:
            p = self._expand(prod)
            try:
                self.round_ctx.get(p)
            except KeyError:
                missing.append(p)
        if missing:
            raise RoundCtxContractError(
                category=self.category, name=self.name, missing_produces=missing
            )
