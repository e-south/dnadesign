"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/src/contracts.py

Explicit runtime contracts for infer namespaces and USR output column naming.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List

from .config import OutputSpec
from .errors import ConfigError
from .registry import get_namespace_for_model


def infer_usr_column_name(*, model_id: str, job_id: str, out_id: str) -> str:
    return f"infer__{model_id}__{job_id}__{out_id}"


def namespace_from_fn(namespaced_fn: str) -> str:
    text = str(namespaced_fn or "").strip()
    if "." not in text:
        raise ConfigError(f"function '{namespaced_fn}' must be namespaced (expected '<namespace>.<fn>')")
    namespace, fn_name = text.split(".", 1)
    namespace = namespace.strip()
    fn_name = fn_name.strip()
    if not namespace or not fn_name:
        raise ConfigError(f"function '{namespaced_fn}' must be namespaced (expected '<namespace>.<fn>')")
    return namespace


def validate_extract_output_namespace(*, model_id: str, outputs: List[OutputSpec]) -> str:
    if not outputs:
        raise ConfigError("extract job requires one or more outputs")
    namespaces = {namespace_from_fn(out.fn) for out in outputs}
    if len(namespaces) != 1:
        raise ConfigError("All outputs in a job must share the same output namespace")

    output_namespace = next(iter(namespaces))
    expected_namespace = get_namespace_for_model(model_id)
    if output_namespace != expected_namespace:
        raise ConfigError(
            f"extract output namespace '{output_namespace}' does not match model namespace '{expected_namespace}'"
        )
    return output_namespace


def resolve_generate_namespaced_fn(*, model_id: str, fn: str | None) -> str:
    expected_namespace = get_namespace_for_model(model_id)
    if fn is None:
        return f"{expected_namespace}.generate"

    provided_namespace = namespace_from_fn(fn)
    if provided_namespace != expected_namespace:
        raise ConfigError(
            f"generate namespace '{provided_namespace}' does not match model namespace '{expected_namespace}'"
        )
    return fn
