"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/infer/tests/runtime/test_extract_params.py

Unit tests for extract-parameter resolution contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.infer.src.errors import CapabilityError
from dnadesign.infer.src.runtime.extract_params import resolve_extract_params


def test_resolve_extract_params_returns_copy_for_non_embedding_methods() -> None:
    params = {"topk": 8}

    resolved = resolve_extract_params(
        model_id="evo2_7b",
        method_name="logits",
        params=params,
    )

    assert resolved == params
    assert resolved is not params


def test_resolve_extract_params_sets_evo2_default_embedding_layer_when_missing() -> None:
    params = {"pool": {"method": "mean", "dim": 1}}

    resolved = resolve_extract_params(
        model_id="evo2_7b",
        method_name="embedding",
        params=params,
    )

    assert resolved["layer"] == "blocks.20.mlp.l3"
    assert "layer" not in params


def test_resolve_extract_params_preserves_explicit_embedding_layer() -> None:
    resolved = resolve_extract_params(
        model_id="agnostic_1b",
        method_name="embedding",
        params={"layer": "encoder.layer.11"},
    )

    assert resolved["layer"] == "encoder.layer.11"


def test_resolve_extract_params_fails_without_layer_when_no_default_registered() -> None:
    with pytest.raises(CapabilityError, match="no default embedding layer"):
        resolve_extract_params(
            model_id="agnostic_1b",
            method_name="embedding",
            params={},
        )
