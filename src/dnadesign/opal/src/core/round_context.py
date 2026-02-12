"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/core/round_context.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import dataclasses as _dc
import os
import re
import sys
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
    "RoundCtxStageError",
    "PluginRegistryView",
]


def _dbg(msg: str) -> None:
    if str(os.getenv("OPAL_DEBUG", "")).strip().lower() in ("1", "true", "yes", "on"):
        print(f"[opal.debug.round_ctx] {msg}", file=sys.stderr)


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
        super().__init__(f"[RoundCtx] Unsupported type for '{path}': {t} (expected {expected})")
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


class RoundCtxStageError(RuntimeError):
    def __init__(self, *, active_stage: Optional[str], requested_stage: str, action: str) -> None:
        active = active_stage if active_stage is not None else "<none>"
        if action == "enter":
            msg = (
                f"RoundCtx stage mismatch: already in stage '{active}', "
                f"cannot enter '{requested_stage}' without closing."
            )
        else:
            msg = f"RoundCtx stage mismatch: active stage '{active}', cannot postcheck '{requested_stage}'."
        super().__init__(msg)
        self.active_stage = active_stage
        self.requested_stage = requested_stage
        self.action = action


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
    requires_by_stage: Optional[Dict[str, Tuple[str, ...]]] = None
    produces_by_stage: Optional[Dict[str, Tuple[str, ...]]] = None


def _assert_path_template(path: str) -> None:
    sample = path.replace("<self>", "self")
    _assert_path(sample)


def _normalize_stage_map(
    stage_map: Optional[Mapping[str, List[str]]], *, label: str
) -> Optional[Dict[str, Tuple[str, ...]]]:
    if stage_map is None:
        return None
    if not isinstance(stage_map, Mapping):
        raise ValueError(f"roundctx_contract: {label} must be a mapping of stage -> list[path]")
    out: Dict[str, Tuple[str, ...]] = {}
    for stage, paths in stage_map.items():
        if not isinstance(stage, str) or not stage.strip():
            raise ValueError(f"roundctx_contract: {label} stage name must be a non-empty string")
        if paths is None:
            path_list: List[str] = []
        elif isinstance(paths, (list, tuple)):
            path_list = list(paths)
        else:
            raise ValueError(f"roundctx_contract: {label} for stage '{stage}' must be a list of paths")
        for p in path_list:
            if not isinstance(p, str) or not p.strip():
                raise ValueError(f"roundctx_contract: {label} entries must be non-empty strings (stage '{stage}')")
            _assert_path_template(p)
        out[stage] = tuple(path_list)
    return out


def roundctx_contract(
    *,
    category: str,
    requires: Optional[List[str]] = None,
    produces: Optional[List[str]] = None,
    requires_by_stage: Optional[Mapping[str, List[str]]] = None,
    produces_by_stage: Optional[Mapping[str, List[str]]] = None,
):
    if category not in _ALLOWED_ROOTS - {"core"}:
        raise ValueError(f"roundctx_contract: invalid category={category!r}")
    req = tuple(requires or ())
    prod = tuple(produces or ())
    req_stage = _normalize_stage_map(requires_by_stage, label="requires_by_stage")
    prod_stage = _normalize_stage_map(produces_by_stage, label="produces_by_stage")
    if prod_stage is not None and prod:
        raise ValueError("roundctx_contract: produces must be empty when produces_by_stage is provided")

    def _wrap(obj: Any) -> Any:
        setattr(
            obj,
            "__opal_contract__",
            Contract(
                category=category,
                requires=req,
                produces=prod,
                requires_by_stage=req_stage,
                produces_by_stage=prod_stage,
            ),
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
        raise RoundCtxPathError(path, "must match '<root>/<seg>[/<seg>...]' and use lowercase + [_-]")
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

    @classmethod
    def from_snapshot(
        cls,
        snapshot: Mapping[str, Any],
        *,
        registry: Optional[PluginRegistryView] = None,
    ) -> "RoundCtx":
        """
        Rehydrate a RoundCtx from a persisted snapshot (round_ctx.json).
        Useful for inspection or for passing into y-ops inversion in predict.
        """
        snap = dict(snapshot or {})
        if registry is None:

            def _get(name: str) -> str:
                v = snap.get(name)
                return str(v) if v is not None else "unknown"

            registry = PluginRegistryView(
                model=_get("core/plugins/model/name"),
                objective=_get("core/plugins/objective/name"),
                selection=_get("core/plugins/selection/name"),
                transform_x=_get("core/plugins/transforms_x/name"),
                transform_y=_get("core/plugins/transforms_y/name"),
            )

        ctx = cls(core={}, registry=registry)
        for k, v in snap.items():
            ctx.set(k, v, allow_overwrite=False)
        return ctx

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
        if not allow_overwrite and path in self._store and self._store[path] != normalized:
            raise RoundCtxPathError(path, "overwrite not allowed (value differs)")
        self._store[path] = normalized

    # ----- plugin view -----
    def for_plugin(
        self,
        *,
        category: str,
        name: str,
        plugin: Optional[Any] = None,
        contract: Optional[Contract] = None,
    ) -> "PluginCtx":
        if category not in _ALLOWED_ROOTS - {"core"}:
            raise ValueError(f"Unknown plugin category '{category}'")
        effective_contract: Optional[Contract] = contract
        if effective_contract is None and plugin is not None:
            effective_contract = getattr(plugin, "__opal_contract__", None)
        if effective_contract is not None and effective_contract.category != category:
            raise RoundCtxContractError(category=category, name=name, msg="decorator category/name mismatch")
        effective = effective_contract or Contract(category=category, requires=tuple(), produces=tuple())
        return PluginCtx(round_ctx=self, category=category, name=name, contract=effective)

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
    _active_stage: Optional[str] = field(init=False, default=None, repr=False)
    _stage_buffer: Dict[str, Any] = field(init=False, default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.contract.produces_by_stage is not None and self.contract.produces:
            raise RoundCtxContractError(
                category=self.category,
                name=self.name,
                msg="stage-scoped contracts require produces to be empty; use produces_by_stage instead",
            )

    def _expand(self, path: str) -> str:
        if "<self>" in path:
            path = path.replace("<self>", self.name)
        _assert_path(path)
        return path

    def _ensure_own_namespace(self, path: str) -> None:
        if not path.startswith(f"{self.category}/{self.name}/"):
            raise RoundCtxPathError(path, f"plugin may only write under '{self.category}/{self.name}/...'")

    def _has_stage_maps(self) -> bool:
        return bool(self.contract.requires_by_stage or self.contract.produces_by_stage)

    def _require_stage(self, stage: Optional[str]) -> str:
        if self._has_stage_maps() and stage is None:
            raise RoundCtxContractError(
                category=self.category,
                name=self.name,
                msg="stage is required for stage-scoped contract enforcement",
            )
        if stage is None:
            return ""
        if not isinstance(stage, str) or not stage.strip():
            raise RoundCtxContractError(
                category=self.category,
                name=self.name,
                msg="stage must be a non-empty string",
            )
        return stage

    def _stage_requires(self, stage: Optional[str]) -> List[str]:
        if self._has_stage_maps():
            stage_key = self._require_stage(stage)
            stage_req = (self.contract.requires_by_stage or {}).get(stage_key, tuple())
            reqs = list(self.contract.requires) + list(stage_req)
        else:
            reqs = list(self.contract.requires)
        return [self._expand(r) for r in reqs]

    def _stage_produces(self, stage: Optional[str]) -> List[str]:
        if self._has_stage_maps():
            stage_key = self._require_stage(stage)
            stage_prod = (self.contract.produces_by_stage or {}).get(stage_key, tuple())
            prods = list(stage_prod)
        else:
            prods = list(self.contract.produces)
        return [self._expand(p) for p in prods]

    def _allowed_produces(self) -> Set[str]:
        out: Set[str] = set()
        for p in self.contract.produces:
            out.add(self._expand(p))
        stage_map = self.contract.produces_by_stage or {}
        for paths in stage_map.values():
            for p in paths:
                out.add(self._expand(p))
        return out

    def _is_stage_buffer_eligible(self, path: str) -> bool:
        if self._active_stage is None:
            return False
        stage_map = self.contract.produces_by_stage
        if not stage_map:
            return False
        stage_paths = stage_map.get(self._active_stage, tuple())
        return path in {self._expand(p) for p in stage_paths}

    def _reset_stage_state(self) -> None:
        self._stage_buffer.clear()
        self._active_stage = None

    def reset_stage_state(self) -> None:
        self._reset_stage_state()

    def get(self, path: str, default: Any = None) -> Any:
        p = self._expand(path)
        if self._active_stage is not None and p in self._stage_buffer:
            val = self._stage_buffer[p]
        else:
            val = self.round_ctx.get(p, default=default)
        self.round_ctx._audit.add_consumed(self.category, self.name, p)
        return val

    def set(self, path: str, value: Any) -> None:
        p = self._expand(path)
        self._ensure_own_namespace(p)
        if p not in self._allowed_produces():
            raise RoundCtxContractError(
                category=self.category,
                name=self.name,
                illegal_writes=[p],
                msg="write to a key not declared in produces",
            )
        if self._is_stage_buffer_eligible(p):
            _dbg(f"staging write for {self.category}/{self.name} stage={self._active_stage} path={p}")
            self._stage_buffer[p] = _to_jsonable(p, value)
        else:
            _dbg(f"persisting write for {self.category}/{self.name} path={p}")
            self.round_ctx.set(p, value, allow_overwrite=False)
        self.round_ctx._audit.add_produced(self.category, self.name, p)

    def precheck_requires(self, stage: Optional[str] = None) -> None:
        if stage is not None and self._has_stage_maps():
            stage_key = self._require_stage(stage)
            if self._active_stage is None:
                self._active_stage = stage_key
                self._stage_buffer.clear()
            elif self._active_stage != stage_key:
                raise RoundCtxStageError(active_stage=self._active_stage, requested_stage=stage_key, action="enter")
        missing: List[str] = []
        for req in self._stage_requires(stage):
            p = req
            try:
                self.round_ctx.get(p)
            except KeyError:
                missing.append(p)
        if missing:
            raise RoundCtxContractError(category=self.category, name=self.name, missing_requires=missing)

    def postcheck_produces(self, stage: Optional[str] = None) -> None:
        if stage is not None and self._has_stage_maps():
            stage_key = self._require_stage(stage)
            if self._active_stage is None:
                raise RoundCtxStageError(active_stage=None, requested_stage=stage_key, action="postcheck")
            if self._active_stage != stage_key:
                raise RoundCtxStageError(
                    active_stage=self._active_stage,
                    requested_stage=stage_key,
                    action="postcheck",
                )
            try:
                _dbg(
                    "committing staged writes for "
                    f"{self.category}/{self.name} stage={stage_key} keys={len(self._stage_buffer)}"
                )
                for p in sorted(self._stage_buffer.keys()):
                    self.round_ctx.set(p, self._stage_buffer[p], allow_overwrite=False)
            finally:
                self._reset_stage_state()
        missing: List[str] = []
        for prod in self._stage_produces(stage):
            p = prod
            try:
                self.round_ctx.get(p)
            except KeyError:
                missing.append(p)
        if missing:
            raise RoundCtxContractError(category=self.category, name=self.name, missing_produces=missing)
