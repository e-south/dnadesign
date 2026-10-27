"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/analysis/test_selection_view_predictions.py

Explicit selection-view projection from shared prediction ledgers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dnadesign.opal.src.analysis.ledger import read_selection_view_predictions
from dnadesign.opal.src.core.utils import OpalError
from dnadesign.opal.src.storage.ledger import LedgerWriter
from dnadesign.opal.src.storage.workspace import CampaignWorkspace
from dnadesign.opal.src.storage.writebacks import (
    SelectionViewEmit,
    build_run_meta_event,
    build_run_pred_events,
)


def _workspace_with_two_views(tmp_path: Path) -> CampaignWorkspace:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True)
    ws = CampaignWorkspace(config_path=workdir / "campaign.yaml", workdir=workdir)
    views = [
        SelectionViewEmit(
            selection_view_id=view_id,
            objective_name="scalar_identity_v1",
            selection_name="top_n",
            score=np.asarray(scores),
            score_ref=f"{view_id}/scalar",
            selection_score=np.asarray(scores),
            ranks_competition=np.asarray(ranks),
            selected_bool=np.asarray(selected),
            top_k=1,
            diagnostics={"component": np.asarray(scores)},
        )
        for view_id, scores, ranks, selected in [
            ("ethanol", [0.1, 0.2], [2, 1], [False, True]),
            ("ciprofloxacin", [0.3, 0.1], [1, 2], [True, False]),
        ]
    ]
    predictions = build_run_pred_events(
        run_id="r0",
        as_of_round=0,
        ids=["a", "b"],
        sequences=["AAA", "BBB"],
        y_hat_model=np.asarray([[0.1], [0.2]]),
        y_dim=1,
        selection_views=views,
    )
    metadata = build_run_meta_event(
        run_id="r0",
        as_of_round=0,
        model_name="random_forest",
        model_params={},
        y_ops=[],
        x_transform_name="identity",
        x_transform_params={},
        y_ingest_transform_name="scalar_from_table_v1",
        y_ingest_transform_params={},
        objective_defs=[
            {"selection_view_id": view.selection_view_id, "objective_name": view.objective_name} for view in views
        ],
        selection_view_defs=[
            {"selection_view_id": view.selection_view_id, "selection_name": view.selection_name} for view in views
        ],
        stats_n_train=2,
        stats_n_scored=2,
        pred_rows_df=predictions,
        artifact_paths_and_hashes={},
    )
    writer = LedgerWriter(ws)
    writer.append_run_pred(predictions)
    writer.append_run_meta(metadata)
    return ws


def test_read_selection_view_predictions_projects_only_requested_view(tmp_path: Path) -> None:
    ws = _workspace_with_two_views(tmp_path)

    frame = read_selection_view_predictions(
        ws.ledger_predictions_dir,
        selection_view_id="ethanol",
        round_selector="latest",
        runs_df=None,
        require_run_id=False,
    )

    assert frame["selection_view_id"].to_list() == ["ethanol", "ethanol"]
    assert frame["view__score"].to_list() == [0.1, 0.2]
    assert frame["view__rank_competition"].to_list() == [2, 1]
    assert frame["view__is_selected"].to_list() == [False, True]
    assert "pred__y_hat_model" in frame.columns


def test_read_selection_view_predictions_rejects_unknown_view(tmp_path: Path) -> None:
    ws = _workspace_with_two_views(tmp_path)

    with pytest.raises(OpalError, match="selection view 'and'.*not present"):
        read_selection_view_predictions(
            ws.ledger_predictions_dir,
            selection_view_id="and",
            round_selector="latest",
            runs_df=None,
            require_run_id=False,
        )
