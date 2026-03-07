"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/test_adapter_runtime.py

Contract tests for infer adapter runtime cache/loading helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dnadesign.infer.src.config import ModelConfig
from dnadesign.infer.src.errors import InferError, ModelLoadError
from dnadesign.infer.src.runtime.adapter_runtime import (
    auto_derate_enabled,
    clear_adapter_cache,
    get_adapter,
    is_oom,
)


def _model() -> ModelConfig:
    return ModelConfig(id="evo2_7b", device="cpu", precision="fp32", alphabet="dna")


def test_get_adapter_caches_by_model_device_precision() -> None:
    clear_adapter_cache()
    calls = {"count": 0}

    class _Adapter:
        def __init__(self, model_id: str, device: str, precision: str) -> None:
            calls["count"] += 1
            self.key = (model_id, device, precision)

    def _resolver(_model_id: str):
        return _Adapter

    first = get_adapter(model=_model(), resolver=_resolver)
    second = get_adapter(model=_model(), resolver=_resolver)
    assert first is second
    assert calls["count"] == 1
    assert first.key == ("evo2_7b", "cpu", "fp32")


def test_get_adapter_accepts_positional_model_argument() -> None:
    clear_adapter_cache()

    class _Adapter:
        def __init__(self, model_id: str, device: str, precision: str) -> None:
            self.key = (model_id, device, precision)

    def _resolver(_model_id: str):
        return _Adapter

    adapter = get_adapter(_model(), resolver=_resolver)
    assert adapter.key == ("evo2_7b", "cpu", "fp32")


def test_get_adapter_re_raises_infer_error() -> None:
    clear_adapter_cache()

    class _ExpectedInferError(InferError):
        pass

    class _Adapter:
        def __init__(self, *_args, **_kwargs) -> None:
            raise _ExpectedInferError("known infer error")

    def _resolver(_model_id: str):
        return _Adapter

    try:
        get_adapter(model=_model(), resolver=_resolver)
        raise AssertionError("expected infer error")
    except _ExpectedInferError as exc:
        assert "known infer error" in str(exc)


def test_get_adapter_wraps_non_infer_errors_as_model_load_error() -> None:
    clear_adapter_cache()

    class _Adapter:
        def __init__(self, *_args, **_kwargs) -> None:
            raise RuntimeError("boom")

    def _resolver(_model_id: str):
        return _Adapter

    try:
        get_adapter(model=_model(), resolver=_resolver)
        raise AssertionError("expected model load error")
    except ModelLoadError as exc:
        assert "boom" in str(exc)


def test_is_oom_matches_case_insensitive_phrase() -> None:
    assert is_oom(RuntimeError("CUDA Out Of Memory")) is True
    assert is_oom(RuntimeError("other failure")) is False


def test_auto_derate_enabled_contract(monkeypatch) -> None:
    monkeypatch.delenv("INFER_AUTO_DERATE_OOM", raising=False)
    assert auto_derate_enabled() is True

    monkeypatch.setenv("INFER_AUTO_DERATE_OOM", "0")
    assert auto_derate_enabled() is False

    monkeypatch.setenv("INFER_AUTO_DERATE_OOM", "false")
    assert auto_derate_enabled() is False

    monkeypatch.setenv("INFER_AUTO_DERATE_OOM", "1")
    assert auto_derate_enabled() is True
