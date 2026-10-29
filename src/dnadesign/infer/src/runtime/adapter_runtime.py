"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/runtime/adapter_runtime.py

Provides adapter cache/loading and runtime policy helpers for infer execution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import inspect
import os
from typing import Callable, Dict, Tuple

from ..config import ModelConfig
from ..errors import InferError, ModelLoadError, ValidationError
from ..registry import get_adapter_cls

_ADAPTER_CACHE: Dict[Tuple[str, str, str, str, Tuple[int, ...]], object] = {}


def clear_adapter_cache() -> None:
    _ADAPTER_CACHE.clear()


def _supported_parallelism_strategies(adapter_cls: object) -> tuple[str, ...]:
    declared = getattr(adapter_cls, "supported_parallelism_strategies", None)
    if declared is None:
        return ("single_device",)
    normalized = tuple(str(item).strip() for item in declared if str(item).strip())
    return normalized or ("single_device",)


def validate_adapter_runtime_contract(model: ModelConfig, resolver: Callable[[str], object] = get_adapter_cls) -> None:
    adapter_cls = resolver(model.id)
    supported = _supported_parallelism_strategies(adapter_cls)
    if model.parallelism.strategy not in supported:
        supported_text = ", ".join(sorted(supported))
        raise ValidationError(
            "ADAPTER_CONTRACT_FAIL "
            f"model_id={model.id} "
            f"parallelism.strategy={model.parallelism.strategy} "
            f"supported_parallelism={supported_text}"
        )


def _adapter_init_kwargs(adapter_cls: object, model: ModelConfig) -> dict[str, object]:
    try:
        parameters = inspect.signature(adapter_cls).parameters
    except (TypeError, ValueError):
        return {}
    accepts_kwargs = any(param.kind is inspect.Parameter.VAR_KEYWORD for param in parameters.values())
    if accepts_kwargs or "parallelism" in parameters:
        return {"parallelism": model.parallelism}
    return {}


def get_adapter(model: ModelConfig, resolver: Callable[[str], object] = get_adapter_cls):
    validate_adapter_runtime_contract(model=model, resolver=resolver)
    key = (
        model.id,
        model.device,
        model.precision,
        model.parallelism.strategy,
        tuple(model.parallelism.gpu_ids or ()),
    )
    if key in _ADAPTER_CACHE:
        return _ADAPTER_CACHE[key]
    adapter_cls = resolver(model.id)
    try:
        adapter = adapter_cls(model.id, model.device, model.precision, **_adapter_init_kwargs(adapter_cls, model))
    except InferError:
        raise
    except Exception as exc:
        raise ModelLoadError(str(exc))
    _ADAPTER_CACHE[key] = adapter
    return adapter


def is_oom(error: BaseException) -> bool:
    text = str(error).lower()
    return "out of memory" in text or "canuse32bitindexmath" in text


def auto_derate_enabled() -> bool:
    return os.environ.get("INFER_AUTO_DERATE_OOM", "1").lower() not in {
        "0",
        "false",
        "off",
        "no",
    }
