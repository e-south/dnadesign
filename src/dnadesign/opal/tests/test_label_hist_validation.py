"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_label_hist_validation.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

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
            "opal__demo__label_hist": [[{"y": [1.0, 2.0]}]],  # missing observed_round
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
                    {"kind": "label", "observed_round": 0, "y_obs": [0.1], "src": "ingest_y"},
                    {
                        "kind": "pred",
                        "as_of_round": 1,
                        "run_id": "run-1",
                        "y_hat": [0.2],
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
                        "y_hat": [0.2],
                    }
                ]
            ],
        }
    )
    store = _store(tmp_path)
    with pytest.raises(OpalError):
        store.validate_label_hist(df, require=True)


def test_label_hist_validation_allows_missing_when_not_required(tmp_path):
    df = pd.DataFrame({"id": ["a"], "sequence": ["AAA"], "bio_type": ["dna"], "alphabet": ["dna_4"]})
    store = _store(tmp_path)
    store.validate_label_hist(df, require=False)
