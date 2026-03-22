"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/infer/tests/test_adapter_runtime.py

Contract tests for infer adapter runtime cache/loading helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.infer.src.config import ModelConfig, ModelParallelismConfig
from dnadesign.infer.src.errors import InferError, ModelLoadError, ValidationError
from dnadesign.infer.src.runtime.adapter_runtime import (
    auto_derate_enabled,
    clear_adapter_cache,
    get_adapter,
    is_oom,
    validate_adapter_runtime_contract,
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


def test_get_adapter_passes_parallelism_to_adapter_when_supported() -> None:
    clear_adapter_cache()
    calls = {"count": 0}

    class _Adapter:
        supported_parallelism_strategies = ("single_device", "multi_gpu_vortex")

        def __init__(self, model_id: str, device: str, precision: str, *, parallelism) -> None:
            calls["count"] += 1
            self.key = (model_id, device, precision, parallelism.strategy, tuple(parallelism.gpu_ids or ()))

    def _resolver(_model_id: str):
        return _Adapter

    model = ModelConfig(
        id="evo2_20b",
        device="cuda:0",
        precision="bf16",
        alphabet="dna",
        parallelism=ModelParallelismConfig(strategy="multi_gpu_vortex", min_gpus=2, gpu_ids=[0, 1]),
    )
    first = get_adapter(model=model, resolver=_resolver)
    second = get_adapter(model=model, resolver=_resolver)

    assert first is second
    assert calls["count"] == 1
    assert first.key == ("evo2_20b", "cuda:0", "bf16", "multi_gpu_vortex", (0, 1))


def test_validate_adapter_runtime_contract_rejects_unwired_parallelism() -> None:
    class _Adapter:
        supported_parallelism_strategies = ("single_device",)

    def _resolver(_model_id: str):
        return _Adapter

    model = ModelConfig(
        id="evo2_20b",
        device="cuda:0",
        precision="bf16",
        alphabet="dna",
        parallelism=ModelParallelismConfig(strategy="multi_gpu_vortex", min_gpus=2),
    )

    with pytest.raises(ValidationError, match="ADAPTER_CONTRACT_FAIL"):
        validate_adapter_runtime_contract(model=model, resolver=_resolver)


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
