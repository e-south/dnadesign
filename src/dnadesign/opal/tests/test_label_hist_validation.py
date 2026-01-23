# ABOUTME: Validates label_history schema parsing and strictness for OPAL records.
# ABOUTME: Ensures label/pred entries meet required contracts for auditability.
"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_label_hist_validation.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd
import pytest

from dnadesign.opal.src.core.utils import OpalError
from dnadesign.opal.src.storage.data_access import RecordsStore


def _store(tmp_path):
    return RecordsStore(
        kind="local",
        records_path=tmp_path / "records.parquet",
        campaign_slug="demo",
        x_col="X",
        y_col="Y",
        x_transform_name="identity",
        x_transform_params={},
    )


def test_label_hist_validation_rejects_malformed_entries(tmp_path):
    df = pd.DataFrame(
        {
            "id": ["a"],
            "bio_type": ["dna"],
            "sequence": ["AAA"],
            "alphabet": ["dna_4"],
            "opal__demo__label_hist": [
                [
                    {
                        "kind": "label",
                        "y_obs": {"value": [1.0, 2.0], "dtype": "vector"},
                    }
                ]
            ],  # missing observed_round
        }
    )
    store = _store(tmp_path)
    with pytest.raises(OpalError):
        store.validate_label_hist(df, require=True)


def test_label_hist_validation_accepts_pred_entries(tmp_path):
    df = pd.DataFrame(
        {
            "id": ["a"],
            "bio_type": ["dna"],
            "sequence": ["AAA"],
            "alphabet": ["dna_4"],
            "opal__demo__label_hist": [
                [
                    {
                        "kind": "label",
                        "observed_round": 0,
                        "y_obs": {"value": [0.1], "dtype": "vector"},
                        "src": "ingest_y",
                    },
                    {
                        "kind": "pred",
                        "as_of_round": 1,
                        "run_id": "run-1",
                        "y_pred": {"value": [0.2], "dtype": "vector"},
                        "y_space": "objective",
                        "objective": {"name": "sfxi_v1", "params": {"setpoint_vector": [0, 0, 0, 1]}},
                        "metrics": {"score": 0.5},
                        "selection": {"rank": 1, "top_k": True},
                    },
                ]
            ],
        }
    )
    store = _store(tmp_path)
    store.validate_label_hist(df, require=True)


def test_label_hist_validation_rejects_pred_missing_run_id(tmp_path):
    df = pd.DataFrame(
        {
            "id": ["a"],
            "bio_type": ["dna"],
            "sequence": ["AAA"],
            "alphabet": ["dna_4"],
            "opal__demo__label_hist": [
                [
                    {
                        "kind": "pred",
                        "as_of_round": 1,
                        "y_pred": {"value": [0.2], "dtype": "vector"},
                        "y_space": "objective",
                    }
                ]
            ],
        }
    )
    store = _store(tmp_path)
    with pytest.raises(OpalError):
        store.validate_label_hist(df, require=True)


def test_label_hist_validation_accepts_pred_nonnumeric_payload(tmp_path):
    df = pd.DataFrame(
        {
            "id": ["a"],
            "bio_type": ["dna"],
            "sequence": ["AAA"],
            "alphabet": ["dna_4"],
            "opal__demo__label_hist": [
                [
                    {
                        "kind": "pred",
                        "as_of_round": 1,
                        "run_id": "run-1",
                        "y_pred": {"value": {"v": [1, 2, 3]}, "dtype": "object"},
                        "y_space": "objective",
                        "objective": {"name": "sfxi_v1", "params": {"setpoint_vector": [0, 0, 0, 1]}},
                        "metrics": {"score": 0.5},
                        "selection": {"rank": 1, "top_k": True},
                    }
                ]
            ],
        }
    )
    store = _store(tmp_path)
    store.validate_label_hist(df, require=True)


def test_label_hist_validation_allows_missing_when_not_required(tmp_path):
    df = pd.DataFrame({"id": ["a"], "sequence": ["AAA"], "bio_type": ["dna"], "alphabet": ["dna_4"]})
    store = _store(tmp_path)
    store.validate_label_hist(df, require=False)


def test_append_predictions_coerces_objective_params(tmp_path):
    store = _store(tmp_path)
    label_hist_col = store.label_hist_col()
    df = pd.DataFrame(
        {
            "id": ["a"],
            "bio_type": ["dna"],
            "sequence": ["AAA"],
            "alphabet": ["dna_4"],
            label_hist_col: [None],
        }
    )
    out = store.append_predictions_from_arrays(
        df,
        ids=["a"],
        y_hat=np.array([[0.1, 0.2]]),
        as_of_round=0,
        run_id="run-1",
        objective={
            "name": "sfxi_v1",
            "mode": "maximize",
            "params": {"setpoint_vector": np.array([0, 0, 0, 1])},
        },
        metrics_by_name={"score": [0.5]},
        selection_rank=np.array([1]),
        selection_top_k=np.array([True]),
    )
    cell = out[label_hist_col].iloc[0][0]
    params = cell["objective"]["params"]
    assert isinstance(params["setpoint_vector"], list)
