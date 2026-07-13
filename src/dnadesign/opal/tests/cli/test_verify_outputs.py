"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/cli/test_verify_outputs.py

Selection-view verification against shared OPAL prediction ledgers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest
from typer.testing import CliRunner

from dnadesign.opal.src.analysis.ledger import read_selection_view_predictions
from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.src.core.utils import OpalError
from dnadesign.opal.src.reporting.verify_outputs import compare_selection_to_ledger
from dnadesign.opal.src.storage.artifacts import write_selection_parquet
from dnadesign.opal.src.storage.ledger import LedgerWriter
from dnadesign.opal.src.storage.workspace import CampaignWorkspace
from dnadesign.opal.src.storage.writebacks import (
    SelectionViewEmit,
    build_run_meta_event,
    build_run_pred_events,
)


def test_compare_selection_to_view_ledger_matches() -> None:
    selection_df = pd.DataFrame({"id": ["a", "b"], "selection_score": [1.0, 2.0]})
    ledger_df = pd.DataFrame({"id": ["a", "b"], "view__selection_score": [1.0, 2.0]})

    summary, mismatches = compare_selection_to_ledger(selection_df, ledger_df, eps=1e-9)

    assert summary["mismatch_count"] == 0
    assert mismatches.empty


def test_compare_selection_to_view_ledger_rejects_unknown_ids() -> None:
    selection_df = pd.DataFrame({"id": ["a", "missing"], "selection_score": [1.0, 2.0]})
    ledger_df = pd.DataFrame({"id": ["a", "b"], "view__selection_score": [1.0, 2.0]})

    with pytest.raises(OpalError, match="selected_ids_outside_eval"):
        compare_selection_to_ledger(selection_df, ledger_df, eps=1e-9)


def test_verify_outputs_integration_uses_named_view(tmp_path: Path) -> None:
    workdir = tmp_path / "workdir"
    workdir.mkdir(parents=True)
    ws = CampaignWorkspace(config_path=tmp_path / "campaign.yaml", workdir=workdir)
    run_id = "run-1"
    view = SelectionViewEmit(
        selection_view_id="ethanol",
        objective_name="scalar_identity_v1",
        selection_name="top_n",
        score=np.asarray([0.5, 0.6]),
        score_ref="ethanol/scalar",
        selection_score=np.asarray([0.5, 0.6]),
        ranks_competition=np.asarray([2, 1]),
        selected_bool=np.asarray([False, True]),
        top_k=1,
        diagnostics={},
    )
    predictions = build_run_pred_events(
        run_id=run_id,
        as_of_round=1,
        ids=["a", "b"],
        sequences=["AAA", "BBB"],
        y_hat_model=np.asarray([[0.1], [0.2]]),
        y_dim=1,
        selection_views=[view],
    )
    selection_path = workdir / "outputs" / "rounds" / "round_1" / "selection" / "selections.parquet"
    write_selection_parquet(
        selection_path,
        pd.DataFrame(
            {
                "selection_view_id": ["ethanol"],
                "id": ["b"],
                "score": [0.6],
                "selection_score": [0.6],
            }
        ),
    )
    metadata = build_run_meta_event(
        run_id=run_id,
        as_of_round=1,
        model_name="dummy",
        model_params={},
        y_ops=[],
        x_transform_name="identity",
        x_transform_params={},
        y_ingest_transform_name="scalar_from_table_v1",
        y_ingest_transform_params={},
        objective_defs=[{"selection_view_id": "ethanol", "objective_name": "scalar_identity_v1"}],
        selection_view_defs=[{"selection_view_id": "ethanol", "selection_name": "top_n"}],
        stats_n_train=2,
        stats_n_scored=2,
        pred_rows_df=predictions,
        artifact_paths_and_hashes={
            "selection/selections.parquet": ("sha", str(selection_path)),
        },
    )
    writer = LedgerWriter(ws)
    writer.append_run_pred(predictions)
    writer.append_run_meta(metadata)

    projected = read_selection_view_predictions(
        ws.ledger_predictions_dir,
        selection_view_id="ethanol",
        columns=["id", "view__selection_score"],
        round_selector=1,
        run_id=run_id,
        runs_df=pl.read_parquet(ws.ledger_runs_path),
        require_run_id=False,
    ).to_pandas()
    selected = pd.read_parquet(selection_path)
    summary, mismatches = compare_selection_to_ledger(selected, projected, eps=1e-9)

    assert summary["mismatch_count"] == 0
    assert mismatches.empty


def test_verify_outputs_requires_selection_view_argument() -> None:
    result = CliRunner().invoke(_build(), ["verify-outputs", "-c", "campaign.yaml"])

    assert result.exit_code == 2
    assert "--view" in result.output
