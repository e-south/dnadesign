"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_runtime_and_public_api.py

Tests for explicit runtime bootstrap and stable public API helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

import dnadesign.baserender as baserender
from dnadesign.baserender import render_parquet_record_figure
from dnadesign.baserender.src.core import ContractError, RenderingError
from dnadesign.baserender.src.core.registry import (
    clear_feature_effect_contracts,
    get_effect_contract,
    get_feature_contract,
)
from dnadesign.baserender.src.render.effects.registry import clear_effect_drawers, get_effect_drawer
from dnadesign.baserender.src.runtime import initialize_runtime

from .conftest import write_parquet


def test_runtime_bootstrap_is_explicit_and_idempotent() -> None:
    clear_feature_effect_contracts()
    clear_effect_drawers()
    import dnadesign.baserender.src.render as _render  # noqa: F401

    with pytest.raises(ContractError, match="Unknown feature kind"):
        get_feature_contract("kmer")
    with pytest.raises(ContractError, match="Unknown effect kind"):
        get_effect_contract("span_link")
    with pytest.raises(RenderingError, match="Unknown effect kind"):
        get_effect_drawer("span_link")

    initialize_runtime()
    assert get_feature_contract("kmer")
    assert get_effect_contract("span_link")
    assert get_effect_drawer("span_link")

    initialize_runtime()
    assert get_feature_contract("kmer")
    assert get_effect_contract("motif_logo")


def test_public_parquet_render_helper_renders_record_figure(tmp_path) -> None:
    parquet = write_parquet(
        tmp_path / "input.parquet",
        [
            {
                "id": "r1",
                "sequence": "TTGACAAAAAAAAAAAAAAAATATAAT",
                "densegen__used_tfbs_detail": [
                    {"tf": "lexA", "orientation": "fwd", "tfbs": "TTGACA", "offset": 0},
                    {"tf": "cpxR", "orientation": "fwd", "tfbs": "TATAAT", "offset": 23},
                ],
                "details": "row1",
            }
        ],
    )

    fig = render_parquet_record_figure(
        dataset_path=parquet,
        record_id="r1",
        adapter_kind="densegen_tfbs",
        adapter_columns={
            "sequence": "sequence",
            "annotations": "densegen__used_tfbs_detail",
            "id": "id",
            "details": "details",
        },
    )
    assert fig is not None
    plt.close(fig)


def test_public_api_does_not_export_tool_specific_helpers() -> None:
    assert not hasattr(baserender, "render_densegen_record_figure")
