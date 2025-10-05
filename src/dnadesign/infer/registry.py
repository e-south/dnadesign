"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/infer/registry.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict, Type

from ._logging import get_logger
from .errors import ConfigError

_LOG = get_logger(__name__)

_MODEL_REGISTRY: Dict[str, Type] = {}
_FN_REGISTRY: Dict[str, str] = {}


def register_model(model_id: str, adapter_cls: Type) -> None:
    if model_id in _MODEL_REGISTRY:
        _LOG.warning(f"Model '{model_id}' already registered; overriding.")
    _MODEL_REGISTRY[model_id] = adapter_cls


def get_adapter_cls(model_id: str) -> Type:
    try:
        return _MODEL_REGISTRY[model_id]
    except KeyError as e:
        raise ConfigError(
            f"Unknown model id '{model_id}'. Is the adapter registered?"
        ) from e


def list_models() -> Dict[str, Type]:
    return dict(_MODEL_REGISTRY)


def get_namespace_for_model(model_id: str) -> str:
    """Infer adapter namespace from model_id. 'evo2_7b' -> 'evo2'."""
    return model_id.split("_", 1)[0] if "_" in model_id else model_id


def register_fn(namespaced: str, method_name: str) -> None:
    if namespaced in _FN_REGISTRY:
        _LOG.warning(f"Function '{namespaced}' already registered; overriding.")
    _FN_REGISTRY[namespaced] = method_name


def resolve_fn(namespaced: str) -> str:
    try:
        return _FN_REGISTRY[namespaced]
    except KeyError as e:
        raise ConfigError(f"Unknown function '{namespaced}'.") from e


def list_fns() -> Dict[str, str]:
    return dict(_FN_REGISTRY)
