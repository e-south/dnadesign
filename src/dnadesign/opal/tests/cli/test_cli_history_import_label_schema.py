"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/cli/test_cli_history_import_label_schema.py

Checks canonical label-ledger schemas after campaign-history relocation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.opal.src.storage.parquet_io import read_parquet_df
from dnadesign.opal.tests._cli_helpers import write_ledger_labels
from dnadesign.opal.tests.cli.test_cli_history_import import _workspace
from dnadesign.opal.tests.cli.test_cli_history_import_provenance import _invoke_import


def test_history_import_unifies_label_parts_with_different_optional_columns(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    _workspace(source, round_index=0, run_id="run-0", with_state=True)
    target_campaign, _ = _workspace(target, round_index=1, run_id="run-1", with_state=False)
    write_ledger_labels(source, round_index=0)
    write_ledger_labels(target, round_index=1)
    source_part = next((source / "outputs/ledger/labels.parquet").glob("*.parquet"))
    target_part = next((target / "outputs/ledger/labels.parquet").glob("*.parquet"))
    source_labels = read_parquet_df(source_part)
    source_labels["note"] = "source-note"
    source_labels.to_parquet(source_part, index=False)
    target_labels = read_parquet_df(target_part)
    target_labels["sequence"] = "AAA"
    target_labels["observed_round"] = target_labels["observed_round"].astype("int32")
    target_labels.to_parquet(target_part, index=False)

    result = _invoke_import(source, target_campaign)

    assert result.exit_code == 0, result.output
    canonical_root = target / "outputs/ledger/labels.parquet"
    canonical_parts = list(canonical_root.glob("*.parquet"))
    assert len(canonical_parts) == 1
    canonical = read_parquet_df(canonical_root).set_index("observed_round")
    assert set(canonical.columns) >= {"note", "sequence"}
    assert canonical.at[0, "note"] == "source-note"
    assert canonical.at[1, "sequence"] == "AAA"
