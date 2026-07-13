"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/analysis/test_analysis_ledger_setpoint.py

Selection-view setpoint joins from ledger v2.0 metadata.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dnadesign.opal.src.analysis.ledger import load_predictions_with_setpoint
from dnadesign.opal.src.core.utils import OpalError
from dnadesign.opal.src.storage.ledger import LedgerWriter
from dnadesign.opal.src.storage.workspace import CampaignWorkspace
from dnadesign.opal.src.storage.writebacks import (
    SelectionViewEmit,
    build_run_meta_event,
    build_run_pred_events,
)


def _write_run(tmp_path: Path, *, include_setpoint: bool) -> CampaignWorkspace:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    ws = CampaignWorkspace(config_path=workdir / "campaign.yaml", workdir=workdir)
    view = SelectionViewEmit(
        selection_view_id="ethanol",
        objective_name="sfxi_v1",
        selection_name="top_n",
        score=np.asarray([0.1, 0.2]),
        score_ref="ethanol/sfxi",
        selection_score=np.asarray([0.1, 0.2]),
        ranks_competition=np.asarray([2, 1]),
        selected_bool=np.asarray([False, True]),
        top_k=1,
        diagnostics={},
    )
    predictions = build_run_pred_events(
        run_id="r0",
        as_of_round=0,
        ids=["a", "b"],
        sequences=["AAA", "BBB"],
        y_hat_model=np.asarray([[0.1], [0.2]]),
        y_dim=1,
        selection_views=[view],
    )
    params = {"setpoint_vector": [0, 1, 0, 1]} if include_setpoint else {}
    metadata = build_run_meta_event(
        run_id="r0",
        as_of_round=0,
        model_name="random_forest",
        model_params={},
        y_ops=[],
        x_transform_name="identity",
        x_transform_params={},
        y_ingest_transform_name="sfxi_vec8_from_table_v1",
        y_ingest_transform_params={},
        objective_defs=[
            {
                "selection_view_id": "ethanol",
                "objective_name": "sfxi_v1",
                "params": params,
                "score_channels": ["ethanol/sfxi"],
            }
        ],
        selection_view_defs=[
            {
                "selection_view_id": "ethanol",
                "objective_name": "sfxi_v1",
                "objective_params": params,
                "selection_name": "top_n",
                "score_ref": "ethanol/sfxi",
                "top_k": 1,
            }
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


def test_load_predictions_joins_requested_selection_view_setpoint(tmp_path: Path) -> None:
    ws = _write_run(tmp_path, include_setpoint=True)

    frame = load_predictions_with_setpoint(
        ws.outputs_dir,
        {"as_of_round"},
        selection_view_id="ethanol",
        round_selector="latest",
    )

    assert frame["obj__diag__setpoint"].to_list() == [[0.0, 1.0, 0.0, 1.0]] * 2


def test_load_predictions_rejects_missing_view_setpoint(tmp_path: Path) -> None:
    ws = _write_run(tmp_path, include_setpoint=False)

    with pytest.raises(OpalError, match="ethanol.*setpoint_vector"):
        load_predictions_with_setpoint(
            ws.outputs_dir,
            {"as_of_round"},
            selection_view_id="ethanol",
            round_selector="latest",
        )
