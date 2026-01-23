"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_training_policy.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import pandas as pd
import pytest

from dnadesign.opal.src.core.utils import OpalError
from dnadesign.opal.src.storage.data_access import RecordsStore
from dnadesign.opal.src.transforms_x import identity  # noqa: F401


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


def _df():
    return pd.DataFrame(
        {
            "id": ["a"],
            "bio_type": ["dna"],
            "sequence": ["AAA"],
            "alphabet": ["dna_4"],
            "X": [[0.1]],
            "opal__demo__label_hist": [
                [
                    {"kind": "label", "observed_round": 0, "y_obs": [0.0]},
                    {
                        "kind": "pred",
                        "as_of_round": 1,
                        "run_id": "run-1",
                        "y_hat": [0.5],
                        "metrics": {"score": 0.2},
                    },
                    {"kind": "label", "observed_round": 1, "y_obs": [1.0]},
                ]
            ],
        }
    )


def test_training_policy_latest_only(tmp_path):
    store = _store(tmp_path)
    df = _df()
    out = store.training_labels_with_round(
        df,
        as_of_round=1,
        cumulative_training=True,
        dedup_policy="latest_only",
    )
    assert len(out) == 1
    assert out.iloc[0]["r"] == 1
    assert out.iloc[0]["y"] == [1.0]


def test_training_policy_all_rounds(tmp_path):
    store = _store(tmp_path)
    df = _df()
    out = store.training_labels_with_round(
        df,
        as_of_round=1,
        cumulative_training=True,
        dedup_policy="all_rounds",
    )
    assert len(out) == 2
    assert sorted(out["r"].tolist()) == [0, 1]


def test_training_policy_ignores_pred_entries(tmp_path):
    store = _store(tmp_path)
    df = _df()
    out = store.training_labels_with_round(
        df,
        as_of_round=1,
        cumulative_training=True,
        dedup_policy="all_rounds",
    )
    assert out["y"].tolist() == [[0.0], [1.0]]


def test_training_policy_error_on_duplicate(tmp_path):
    store = _store(tmp_path)
    df = _df()
    with pytest.raises(OpalError):
        store.training_labels_with_round(
            df,
            as_of_round=1,
            cumulative_training=True,
            dedup_policy="error_on_duplicate",
        )
