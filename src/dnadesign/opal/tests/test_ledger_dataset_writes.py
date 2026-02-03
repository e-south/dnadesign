# ABOUTME: Ensures ledger run/label sinks write append-only dataset parts.
# ABOUTME: Validates run/label ledger paths are directories with parquet parts.
"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_ledger_dataset_writes.py

Module Author(s): Eric J. South (extended by Codex)
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dnadesign.opal.src import LEDGER_SCHEMA_VERSION
from dnadesign.opal.src import __version__ as OPAL_VERSION
from dnadesign.opal.src.storage.ledger import LedgerReader, LedgerWriter
from dnadesign.opal.src.storage.workspace import CampaignWorkspace
from dnadesign.opal.src.storage.writebacks import SelectionEmit, build_run_meta_event, build_run_pred_events


def _workspace(tmp_path: Path) -> CampaignWorkspace:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    return CampaignWorkspace(config_path=workdir / "campaign.yaml", workdir=workdir)


def test_ledger_run_meta_writes_dataset_parts(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    writer = LedgerWriter(ws)

    y_hat = np.array([[0.1], [0.2]])
    y_obj = np.array([0.1, 0.2])
    sel_emit = SelectionEmit(
        ranks_competition=np.array([1, 2]),
        selected_bool=np.array([True, False]),
    )
    run_pred = build_run_pred_events(
        run_id="run-0",
        as_of_round=0,
        ids=["a", "b"],
        sequences=["AAA", "BBB"],
        y_hat_model=y_hat,
        y_obj_scalar=y_obj,
        y_dim=1,
        y_hat_model_sd=None,
        y_obj_scalar_sd=None,
        obj_diagnostics={},
        sel_emit=sel_emit,
    )
    run_meta = build_run_meta_event(
        run_id="run-0",
        as_of_round=0,
        model_name="random_forest",
        model_params={},
        y_ops=[],
        x_transform_name="identity",
        x_transform_params={},
        y_ingest_transform_name="scalar_from_table_v1",
        y_ingest_transform_params={},
        objective_name="scalar_identity_v1",
        objective_params={},
        selection_name="top_n",
        selection_params={"top_k": 1},
        selection_score_field="pred__y_obj_scalar",
        selection_objective_mode="maximize",
        sel_tie_handling="competition_rank",
        stats_n_train=2,
        stats_n_scored=2,
        unc_mean_sd=None,
        pred_rows_df=run_pred,
        artifact_paths_and_hashes={},
        objective_summary_stats=None,
    )
    run_meta["schema__version"] = LEDGER_SCHEMA_VERSION
    run_meta["opal__version"] = OPAL_VERSION

    writer.append_run_meta(run_meta)

    runs_path = ws.ledger_runs_path
    assert runs_path.exists()
    assert runs_path.is_dir()
    assert list(runs_path.rglob("*.parquet"))

    reader = LedgerReader(ws)
    runs_df = reader.read_runs()
    assert len(runs_df) == 1


def test_ledger_labels_writes_dataset_parts(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    writer = LedgerWriter(ws)

    labels = pd.DataFrame(
        {
            "event": ["label"],
            "observed_round": [0],
            "id": ["a"],
            "y_obs": [[0.1]],
            "src": ["test"],
        }
    )
    writer.append_label(labels)

    labels_path = ws.ledger_labels_path
    assert labels_path.exists()
    assert labels_path.is_dir()
    assert list(labels_path.rglob("*.parquet"))

    reader = LedgerReader(ws)
    labels_df = reader.read_labels()
    assert len(labels_df) == 1


def test_run_pred_diagnostics_column_order_is_deterministic() -> None:
    y_hat = np.array([[0.1], [0.2]])
    y_obj = np.array([0.1, 0.2])
    sel_emit = SelectionEmit(
        ranks_competition=np.array([1, 2]),
        selected_bool=np.array([True, False]),
    )
    diag = {
        "effect_raw": np.array([1.0, 2.0]),
        "effect_scaled": np.array([0.1, 0.2]),
        "logic_fidelity": np.array([0.5, 0.6]),
        "clip_lo_mask": np.array([0.0, 1.0]),
        "clip_hi_mask": np.array([1.0, 0.0]),
    }
    run_pred = build_run_pred_events(
        run_id="run-0",
        as_of_round=0,
        ids=["a", "b"],
        sequences=["AAA", "BBB"],
        y_hat_model=y_hat,
        y_obj_scalar=y_obj,
        y_dim=1,
        y_hat_model_sd=None,
        y_obj_scalar_sd=None,
        obj_diagnostics=diag,
        sel_emit=sel_emit,
    )
    expected = [
        "event",
        "run_id",
        "as_of_round",
        "id",
        "sequence",
        "pred__y_dim",
        "pred__y_hat_model",
        "pred__y_obj_scalar",
        "sel__rank_competition",
        "sel__is_selected",
        "obj__logic_fidelity",
        "obj__effect_raw",
        "obj__effect_scaled",
        "obj__clip_lo_mask",
        "obj__clip_hi_mask",
    ]
    assert list(run_pred.columns) == expected
