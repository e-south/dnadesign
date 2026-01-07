"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_analysis_facade_setpoint.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dnadesign.opal.src import LEDGER_SCHEMA_VERSION
from dnadesign.opal.src import __version__ as OPAL_VERSION
from dnadesign.opal.src.analysis.facade import load_predictions_with_setpoint
from dnadesign.opal.src.core.utils import OpalError
from dnadesign.opal.src.storage.ledger import LedgerWriter
from dnadesign.opal.src.storage.workspace import CampaignWorkspace
from dnadesign.opal.src.storage.writebacks import SelectionEmit, build_run_meta_event, build_run_pred_events

from ._cli_helpers import write_campaign_yaml, write_records


def test_load_predictions_with_setpoint_requires_setpoint(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)

    ws = CampaignWorkspace(config_path=campaign, workdir=workdir)
    writer = LedgerWriter(ws)

    y_hat = np.array([[0.1], [0.2]])
    y_obj = np.array([0.1, 0.2])
    sel_emit = SelectionEmit(
        ranks_competition=np.array([1, 2]),
        selected_bool=np.array([True, False]),
    )
    run_pred_ok = build_run_pred_events(
        run_id="r0",
        as_of_round=0,
        ids=["a", "b"],
        sequences=["AAA", "BBB"],
        y_hat_model=y_hat,
        y_obj_scalar=y_obj,
        y_dim=1,
        y_hat_model_sd=None,
        y_obj_scalar_sd=None,
        obj_diagnostics=None,
        sel_emit=sel_emit,
    )
    run_meta_ok = build_run_meta_event(
        run_id="r0",
        as_of_round=0,
        model_name="random_forest",
        model_params={},
        y_ops=[],
        x_transform_name="identity",
        x_transform_params={},
        y_ingest_transform_name="sfxi_vec8_from_table_v1",
        y_ingest_transform_params={},
        objective_name="sfxi_v1",
        objective_params={"setpoint_vector": [0, 0, 0, 1]},
        selection_name="top_n",
        selection_params={"top_k": 1},
        selection_score_field="pred__y_obj_scalar",
        selection_objective_mode="maximize",
        sel_tie_handling="competition_rank",
        stats_n_train=2,
        stats_n_scored=2,
        unc_mean_sd=None,
        pred_rows_df=run_pred_ok,
        artifact_paths_and_hashes={},
        objective_summary_stats={"score_min": 0.1, "score_median": 0.2, "score_max": 0.3},
    )
    run_meta_ok["schema__version"] = LEDGER_SCHEMA_VERSION
    run_meta_ok["opal__version"] = OPAL_VERSION

    run_pred_missing = build_run_pred_events(
        run_id="r1",
        as_of_round=0,
        ids=["c", "d"],
        sequences=["CCC", "DDD"],
        y_hat_model=y_hat,
        y_obj_scalar=y_obj,
        y_dim=1,
        y_hat_model_sd=None,
        y_obj_scalar_sd=None,
        obj_diagnostics=None,
        sel_emit=sel_emit,
    )
    run_meta_missing = build_run_meta_event(
        run_id="r1",
        as_of_round=0,
        model_name="random_forest",
        model_params={},
        y_ops=[],
        x_transform_name="identity",
        x_transform_params={},
        y_ingest_transform_name="sfxi_vec8_from_table_v1",
        y_ingest_transform_params={},
        objective_name="sfxi_v1",
        objective_params={},  # missing setpoint_vector should be rejected
        selection_name="top_n",
        selection_params={"top_k": 1},
        selection_score_field="pred__y_obj_scalar",
        selection_objective_mode="maximize",
        sel_tie_handling="competition_rank",
        stats_n_train=2,
        stats_n_scored=2,
        unc_mean_sd=None,
        pred_rows_df=run_pred_missing,
        artifact_paths_and_hashes={},
        objective_summary_stats={"score_min": 0.1, "score_median": 0.2, "score_max": 0.3},
    )
    run_meta_missing["schema__version"] = LEDGER_SCHEMA_VERSION
    run_meta_missing["opal__version"] = OPAL_VERSION

    writer.append_run_pred(run_pred_ok)
    writer.append_run_pred(run_pred_missing)
    writer.append_run_meta(run_meta_ok)
    writer.append_run_meta(run_meta_missing)

    with pytest.raises(OpalError) as exc:
        load_predictions_with_setpoint(
            ws.outputs_dir,
            {"as_of_round", "pred__y_obj_scalar"},
            round_selector="latest",
        )

    assert "Missing objective__params.setpoint_vector" in str(exc.value)
