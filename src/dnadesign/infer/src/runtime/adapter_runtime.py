"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/runtime/adapter_runtime.py

Provides adapter cache/loading and runtime policy helpers for infer execution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from typing import Callable, Dict, Tuple

from ..config import ModelConfig
from ..errors import InferError, ModelLoadError
from ..registry import get_adapter_cls

_ADAPTER_CACHE: Dict[Tuple[str, str, str], object] = {}


def clear_adapter_cache() -> None:
    _ADAPTER_CACHE.clear()


def get_adapter(*, model: ModelConfig, resolver: Callable[[str], object] = get_adapter_cls):
    key = (model.id, model.device, model.precision)
    if key in _ADAPTER_CACHE:
        return _ADAPTER_CACHE[key]
    adapter_cls = resolver(model.id)
    try:
        adapter = adapter_cls(model.id, model.device, model.precision)
    except InferError:
        raise
    except Exception as exc:
        raise ModelLoadError(str(exc))
    _ADAPTER_CACHE[key] = adapter
    return adapter


def is_oom(error: BaseException) -> bool:
    return "out of memory" in str(error).lower()


def auto_derate_enabled() -> bool:
    return os.environ.get("INFER_AUTO_DERATE_OOM", "1").lower() not in {
        "0",
        "false",
        "off",
        "no",
    }
