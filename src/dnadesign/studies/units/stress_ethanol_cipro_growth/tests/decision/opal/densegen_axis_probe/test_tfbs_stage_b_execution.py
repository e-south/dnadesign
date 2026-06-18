"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/densegen_axis_probe/test_tfbs_stage_b_execution.py

Regression tests for TFBS stage b execution studies units stress ethanol.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from .probe_modules import probe_module

_manifest = probe_module("tfbs.stage_b.execution.manifest")
_selection = probe_module("tfbs.stage_b.execution.selection")
_label_inputs = probe_module("tfbs.stage_b.execution.label_inputs")
assert_selection_budget = _selection.assert_selection_budget
selected_campaign_rows = _manifest.selected_campaign_rows
write_label_input_for_ids = _label_inputs.write_label_input_for_ids


def test_write_label_input_for_ids_preserves_selection_order(tmp_path: Path) -> None:
    label_table = tmp_path / "labels.parquet"
    records = tmp_path / "records.parquet"
    out = tmp_path / "inputs" / "labels-b1.parquet"
    pd.DataFrame({"id": ["a", "b", "c"], "lexA_present": [1.0, 0.0, 1.0]}).to_parquet(label_table, index=False)
    pd.DataFrame({"id": ["a", "b", "c"], "sequence": ["AAA", "BBB", "CCC"]}).to_parquet(records, index=False)

    write_label_input_for_ids(
        path=out,
        label_table_path=label_table,
        records_path=records,
        label_name="lexA_present",
        ids=["c", "a"],
    )

    frame = pd.read_parquet(out)
    assert frame.to_dict(orient="list") == {
        "id": ["c", "a"],
        "sequence": ["CCC", "AAA"],
        "lexA_present": [1.0, 1.0],
    }


def test_selected_campaign_rows_fail_on_missing_requested_key() -> None:
    manifest = {"campaigns": [{"campaign_key": "present"}]}

    with pytest.raises(ValueError, match="missing requested campaign"):
        selected_campaign_rows(manifest, ["missing"])


def test_assert_selection_budget_fails_for_ordinal_tie_expansion(tmp_path: Path) -> None:
    selection_path = tmp_path / "outputs" / "rounds" / "round_0" / "selection" / "selection_top_k.csv"
    selection_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": ["a", "b", "c", "d", "e", "f", "g"]}).to_csv(selection_path, index=False)

    with pytest.raises(RuntimeError, match="exact-budget selection contract failed"):
        assert_selection_budget(workdir=tmp_path, round_index=0, selection_k=6, tie_handling="ordinal")


def test_assert_selection_budget_allows_exact_ordinal_budget(tmp_path: Path) -> None:
    selection_path = tmp_path / "outputs" / "rounds" / "round_0" / "selection" / "selection_top_k.csv"
    selection_path.parent.mkdir(parents=True)
    pd.DataFrame({"id": ["a", "b", "c", "d", "e", "f"]}).to_csv(selection_path, index=False)

    assert_selection_budget(workdir=tmp_path, round_index=0, selection_k=6, tie_handling="ordinal")
