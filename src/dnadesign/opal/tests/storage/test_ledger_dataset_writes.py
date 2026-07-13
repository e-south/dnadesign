"""Ledger v2.0 selection-view dataset contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from dnadesign.opal.src.storage.ledger import LedgerReader, LedgerWriter
from dnadesign.opal.src.storage.workspace import CampaignWorkspace
from dnadesign.opal.src.storage.writebacks import (
    SelectionViewEmit,
    build_run_meta_event,
    build_run_pred_events,
)


def _workspace(tmp_path: Path) -> CampaignWorkspace:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    return CampaignWorkspace(config_path=workdir / "campaign.yaml", workdir=workdir)


def _view(*, view_id: str = "primary", uncertainty=None, uncertainty_ref=None) -> SelectionViewEmit:
    return SelectionViewEmit(
        selection_view_id=view_id,
        objective_name="scalar_identity_v1",
        selection_name="top_n",
        score=np.asarray([0.1, 0.2]),
        score_ref=f"{view_id}/scalar",
        selection_score=np.asarray([0.1, 0.2]),
        ranks_competition=np.asarray([2, 1]),
        selected_bool=np.asarray([False, True]),
        top_k=1,
        diagnostics={"component": np.asarray([0.3, 0.4])},
        uncertainty=uncertainty,
        uncertainty_ref=uncertainty_ref,
    )


def _predictions(*, views=None):
    return build_run_pred_events(
        run_id="run-0",
        as_of_round=0,
        ids=["a", "b"],
        sequences=["AAA", "BBB"],
        y_hat_model=np.asarray([[0.1], [0.2]]),
        y_dim=1,
        selection_views=list(views or [_view()]),
        score_channels={"primary/scalar": np.asarray([0.1, 0.2])},
    )


def _run_meta(predictions):
    return build_run_meta_event(
        run_id="run-0",
        as_of_round=0,
        model_name="random_forest",
        model_params={},
        y_ops=[],
        x_transform_name="identity",
        x_transform_params={},
        y_ingest_transform_name="scalar_from_table_v1",
        y_ingest_transform_params={},
        objective_defs=[
            {
                "selection_view_id": "primary",
                "objective_name": "scalar_identity_v1",
                "score_channels": ["primary/scalar"],
            }
        ],
        selection_view_defs=[
            {
                "selection_view_id": "primary",
                "selection_name": "top_n",
                "score_ref": "primary/scalar",
                "top_k": 1,
            }
        ],
        stats_n_train=2,
        stats_n_scored=2,
        pred_rows_df=predictions,
        artifact_paths_and_hashes={},
    )


def test_ledger_run_meta_writes_dataset_parts(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    writer = LedgerWriter(ws)
    predictions = _predictions()
    writer.append_run_pred(predictions)
    writer.append_run_meta(_run_meta(predictions))

    assert ws.ledger_runs_path.is_dir()
    assert list(ws.ledger_runs_path.rglob("*.parquet"))
    assert len(LedgerReader(ws).read_runs()) == 1


def test_run_predictions_store_view_payloads_without_duplicate_y_hat() -> None:
    frame = _predictions(views=[_view(view_id="a"), _view(view_id="b")])

    assert len(frame) == 2
    assert frame.columns.tolist().count("pred__y_hat_model") == 1
    assert [item["selection_view_id"] for item in frame.iloc[0]["pred__selection_views"]] == ["a", "b"]


def test_run_predictions_reject_duplicate_view_ids() -> None:
    with pytest.raises(ValueError, match="selection view ids must be unique"):
        _predictions(views=[_view(view_id="a"), _view(view_id="a")])


def test_run_predictions_reject_misaligned_view_arrays() -> None:
    view = _view()
    bad = SelectionViewEmit(**{**view.__dict__, "score": np.asarray([0.1])})
    with pytest.raises(ValueError, match="arrays must match 2 prediction rows"):
        _predictions(views=[bad])


def test_run_predictions_require_uncertainty_and_ref_together() -> None:
    with pytest.raises(ValueError, match="uncertainty and uncertainty_ref must be paired"):
        _predictions(views=[_view(uncertainty=np.asarray([0.1, 0.1]))])


def test_run_meta_requires_objective_and_selection_view_definitions() -> None:
    predictions = _predictions()
    with pytest.raises(ValueError, match="objective_defs"):
        build_run_meta_event(
            run_id="run-0",
            as_of_round=0,
            model_name="random_forest",
            model_params={},
            y_ops=[],
            x_transform_name="identity",
            x_transform_params={},
            y_ingest_transform_name="scalar_from_table_v1",
            y_ingest_transform_params={},
            objective_defs=[],
            selection_view_defs=[],
            stats_n_train=2,
            stats_n_scored=2,
            pred_rows_df=predictions,
            artifact_paths_and_hashes={},
        )
