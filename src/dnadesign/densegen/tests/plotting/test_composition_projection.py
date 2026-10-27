"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/plotting/test_composition_projection.py

Projection contract tests for composition.parquet schema consumption in plotting.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dnadesign.densegen.src.viz import plotting as plotting_module


def test_plotting_composition_projection_maps_regulator_sequence_columns(tmp_path: Path) -> None:
    composition_path = tmp_path / "composition.parquet"
    pd.DataFrame(
        [
            {
                "solution_id": "sol-1",
                "input_name": "demo_input",
                "plan_name": "demo_plan",
                "regulator": "TF_A",
                "sequence": "AAAA",
                "offset": 0,
                "length": 4,
                "end": 4,
            }
        ]
    ).to_parquet(composition_path, index=False)

    frame = plotting_module._read_composition_parquet(
        composition_path,
        columns=["solution_id", "input_name", "plan_name", "tf", "tfbs", "offset", "length", "end"],
    )

    assert "tf" in frame.columns
    assert "tfbs" in frame.columns
    assert frame.loc[0, "tf"] == "TF_A"
    assert frame.loc[0, "tfbs"] == "AAAA"
