"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/infer/tests/test_adapter_dispatch.py

Unit tests for infer adapter dispatch invariants.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dnadesign.infer.adapter_dispatch import (
    invoke_extract_callable,
    resolve_extract_callable,
    resolve_generate_callable,
)
from dnadesign.infer.errors import CapabilityError
from dnadesign.infer.registry import register_fn


def test_resolve_extract_callable_returns_method_and_callable() -> None:
    register_fn("audit_dispatch.logits", "logits")
    adapter = SimpleNamespace(logits=lambda chunk, **_kwargs: [[1.0] for _ in chunk])

    method_name, fn = resolve_extract_callable(adapter=adapter, namespaced_fn="audit_dispatch.logits")

    assert method_name == "logits"
    assert callable(fn)


def test_resolve_extract_callable_fails_when_method_is_missing() -> None:
    register_fn("audit_dispatch_missing.logits", "logits")
    adapter = SimpleNamespace()

    with pytest.raises(CapabilityError, match="does not implement"):
        resolve_extract_callable(adapter=adapter, namespaced_fn="audit_dispatch_missing.logits")


def test_invoke_extract_callable_passes_fmt_for_logits() -> None:
    observed: dict[str, object] = {}

    def _logits(chunk, **kwargs):
        observed["chunk"] = list(chunk)
        observed["fmt"] = kwargs.get("fmt")
        observed["topk"] = kwargs.get("topk")
        return [[3.14] for _ in chunk]

    out = invoke_extract_callable(
        fn=_logits,
        method_name="logits",
        chunk=["ACGT"],
        params={"topk": 5},
        output_format="list",
    )

    assert out == [[3.14]]
    assert observed == {"chunk": ["ACGT"], "fmt": "list", "topk": 5}


def test_invoke_extract_callable_rejects_unsupported_method() -> None:
    with pytest.raises(CapabilityError, match="Unsupported extract function"):
        invoke_extract_callable(
            fn=lambda _chunk, **_kwargs: [],
            method_name="unknown_method",
            chunk=["ACGT"],
            params={},
            output_format="float",
        )


def test_resolve_generate_callable_fails_when_method_missing() -> None:
    register_fn("audit_dispatch_generate.generate", "generate")
    adapter = SimpleNamespace()

    with pytest.raises(CapabilityError, match="does not support generation"):
        resolve_generate_callable(adapter=adapter, namespaced_fn="audit_dispatch_generate.generate")
