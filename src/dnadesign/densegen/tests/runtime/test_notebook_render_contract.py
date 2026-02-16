"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/runtime/test_notebook_render_contract.py

Tests for the DenseGen notebook render contract used by BaseRender integration.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from dnadesign.baserender import render_parquet_record_figure
from dnadesign.densegen.notebook_render_contract import densegen_notebook_render_contract


def test_notebook_render_contract_renders_without_optional_details_column(tmp_path: Path) -> None:
    records_path = tmp_path / "records.parquet"
    pd.DataFrame(
        [
            {
                "id": "row1",
                "sequence": "AAAAAA",
                "densegen__used_tfbs_detail": [
                    {"tf": "lexA", "orientation": "fwd", "tfbs": "AAA", "offset": 0},
                ],
            }
        ]
    ).to_parquet(records_path)

    contract = densegen_notebook_render_contract()
    fig = render_parquet_record_figure(
        dataset_path=records_path,
        record_id="row1",
        adapter_kind=contract.adapter_kind,
        adapter_columns=contract.adapter_columns,
        adapter_policies=contract.adapter_policies,
        style_preset=contract.style_preset,
    )
    assert fig is not None
    plt.close(fig)


def test_notebook_render_contract_is_explicit_and_complete() -> None:
    contract = densegen_notebook_render_contract()
    assert contract.adapter_kind == "densegen_tfbs"
    assert contract.adapter_columns == {
        "id": "id",
        "sequence": "sequence",
        "annotations": "densegen__used_tfbs_detail",
    }
    assert contract.adapter_policies == {}
    assert contract.style_preset == "presentation_default"
    assert contract.record_window_limit == 500
