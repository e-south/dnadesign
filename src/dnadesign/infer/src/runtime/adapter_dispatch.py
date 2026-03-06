"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/infer/src/runtime/adapter_dispatch.py

Resolves and invokes infer adapter callables for extract and generate operations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

from ..errors import CapabilityError
from ..registry import resolve_fn

ExtractCallable = Callable[..., List[object]]
GenerateCallable = Callable[..., Dict[str, List[object]]]


def resolve_extract_callable(*, adapter: object, namespaced_fn: str) -> Tuple[str, ExtractCallable]:
    method_name = resolve_fn(namespaced_fn)
    fn = getattr(adapter, method_name, None)
    if fn is None:
        raise CapabilityError(f"Adapter does not implement '{namespaced_fn}'")
    return method_name, fn


def invoke_extract_callable(
    *,
    fn: ExtractCallable,
    method_name: str,
    chunk: List[str],
    params: Dict[str, Any],
    output_format: str,
) -> List[object]:
    if method_name == "log_likelihood":
        return fn(chunk, **params)
    if method_name in {"logits", "embedding"}:
        return fn(chunk, **params, fmt=output_format)
    raise CapabilityError(f"Unsupported extract function '{method_name}' in v1")


def resolve_generate_callable(*, adapter: object, namespaced_fn: str) -> GenerateCallable:
    method_name = resolve_fn(namespaced_fn)
    fn = getattr(adapter, method_name, None)
    if fn is None:
        raise CapabilityError(f"Adapter does not support generation '{namespaced_fn}'")
    return fn
