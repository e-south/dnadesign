"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/cli/test_cli_history_import_retention.py

Exercises prediction-retention identity contracts during history relocation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from dnadesign.opal.src.core.utils import OpalError
from dnadesign.opal.src.storage.history_relocation.prediction_retention import (
    FULL,
    SELECTED_HISTORY,
    prediction_dataset_sha256,
    validate_prediction_retention,
)
from dnadesign.opal.src.storage.parquet_io import read_parquet_df
from dnadesign.opal.tests.cli.test_cli_history_import import _workspace
from dnadesign.opal.tests.cli.test_cli_history_import_provenance import (
    _apply_selected_history_retention,
    _invoke_import,
)


def test_history_import_keeps_the_last_full_candidate_universe_across_selected_history(tmp_path: Path) -> None:
    source = tmp_path / "source"
    canonical = tmp_path / "canonical"
    future = tmp_path / "future"
    _workspace(source, round_index=0, run_id="run-0", with_state=True)
    canonical_campaign, _ = _workspace(canonical, round_index=1, run_id="run-1", with_state=False)
    _apply_selected_history_retention(canonical_campaign)
    assert _invoke_import(source, canonical_campaign).exit_code == 0
    future_campaign, _ = _workspace(future, round_index=2, run_id="run-2", with_state=False)
    prediction_part = next((future / "outputs/ledger/predictions").glob("*.parquet"))
    predictions = read_parquet_df(prediction_part)
    predictions.loc[predictions["id"].eq("b"), "id"] = "c"
    predictions.to_parquet(prediction_part, index=False)

    result = _invoke_import(canonical, future_campaign)

    assert result.exit_code == 4, result.output
    assert "introduces candidate ids absent from the prior campaign universe" in result.output.lower()


def test_history_import_rejects_duplicate_retained_selection_memberships(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source_campaign, _ = _workspace(source, round_index=0, run_id="run-0", with_state=True)
    _apply_selected_history_retention(source_campaign)
    prediction_root = source / "outputs/ledger/predictions"
    prediction_part = next(prediction_root.glob("*.parquet"))
    predictions = read_parquet_df(prediction_part)
    memberships = predictions.at[0, "pred__selection_views"]
    if hasattr(memberships, "tolist"):
        memberships = memberships.tolist()
    predictions.at[0, "pred__selection_views"] = [*memberships, dict(memberships[0])]
    predictions.to_parquet(prediction_part, index=False)
    manifest_path = source / "outputs/retention_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["actions"][0]["sha256"] = prediction_dataset_sha256(prediction_root)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    target_campaign, _ = _workspace(target, round_index=1, run_id="run-1", with_state=False)

    result = _invoke_import(source, target_campaign)

    assert result.exit_code == 4, result.output
    assert "retained prediction memberships must be unique" in result.output.lower()


def test_selected_history_compares_candidate_ids_verbatim() -> None:
    predictions = pd.DataFrame(
        {
            "id": [" candidate-a "],
            "pred__selection_views": [[{"selection_view_id": "primary", "is_selected": True}]],
        }
    )
    selections = pd.DataFrame({"id": [" candidate-a "], "selection_view_id": ["primary"]})

    validate_prediction_retention(
        predictions,
        expected_scored_rows=1,
        mode=SELECTED_HISTORY,
        label="Round 0",
        selections=selections,
    )


def test_full_history_rejects_memberships_that_disagree_with_immutable_selections() -> None:
    predictions = pd.DataFrame(
        {
            "id": ["candidate-a"],
            "pred__selection_views": [[{"selection_view_id": "primary", "is_selected": True}]],
        }
    )
    selections = pd.DataFrame({"id": ["candidate-b"], "selection_view_id": ["primary"]})

    with pytest.raises(OpalError, match="retained prediction memberships differ from immutable selections"):
        validate_prediction_retention(
            predictions,
            expected_scored_rows=1,
            mode=FULL,
            label="Round 0",
            selections=selections,
        )


def test_history_import_rejects_duplicate_unselected_view_memberships() -> None:
    predictions = pd.DataFrame(
        {
            "id": ["candidate-a"],
            "pred__selection_views": [
                [
                    {"selection_view_id": "primary", "is_selected": True},
                    {"selection_view_id": "secondary", "is_selected": False},
                    {"selection_view_id": "secondary", "is_selected": False},
                ]
            ],
        }
    )
    selections = pd.DataFrame({"id": ["candidate-a"], "selection_view_id": ["primary"]})

    with pytest.raises(OpalError, match="retained prediction memberships must be unique"):
        validate_prediction_retention(
            predictions,
            expected_scored_rows=3,
            mode=SELECTED_HISTORY,
            label="Round 0",
            selections=selections,
        )
