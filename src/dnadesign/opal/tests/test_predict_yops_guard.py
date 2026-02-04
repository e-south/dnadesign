# ABOUTME: Tests predict Y-ops guardrails for missing round context metadata.
# ABOUTME: Ensures prediction fails fast when Y-ops inversion lacks round_ctx.
"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_predict_yops_guard.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import json

import numpy as np
import pandas as pd
import pytest

from dnadesign.opal.src.core.utils import OpalError
from dnadesign.opal.src.models.random_forest import RandomForestModel  # noqa: F401
from dnadesign.opal.src.runtime.predict import run_predict_ephemeral
from dnadesign.opal.src.storage.data_access import RecordsStore
from dnadesign.opal.src.transforms_x import identity  # noqa: F401 (registers)


def test_predict_requires_round_ctx_when_yops(tmp_path):
    df = pd.DataFrame(
        {
            "id": ["a", "b"],
            "sequence": ["AAA", "BBB"],
            "bio_type": ["dna", "dna"],
            "alphabet": ["dna_4", "dna_4"],
            "X": [[0.1, 0.2], [0.2, 0.3]],
        }
    )

    X_train = np.array([[0.1, 0.2], [0.2, 0.3]])
    Y_train = np.array([[1.0], [2.0]])
    model = RandomForestModel(
        params={
            "n_estimators": 5,
            "random_state": 1,
            "bootstrap": True,
            "oob_score": False,
        }
    )
    model.fit(X_train, Y_train)
    model_dir = tmp_path / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.joblib"
    model.save(str(model_path))

    # model_meta declares Y-ops but no round_ctx.json is present
    meta = {
        "model__name": "random_forest",
        "model__params": model.get_params(),
        "training__y_ops": [{"name": "intensity_median_iqr", "params": {"min_labels": 5}}],
    }
    (model_dir / "model_meta.json").write_text(json.dumps(meta))

    store = RecordsStore(
        kind="local",
        records_path=tmp_path / "records.parquet",
        campaign_slug="demo",
        x_col="X",
        y_col="Y",
        x_transform_name="identity",
        x_transform_params={},
    )

    with pytest.raises(OpalError):
        run_predict_ephemeral(store, df, model_path)

    preds = run_predict_ephemeral(store, df, model_path, assume_no_yops=True)
    assert "y_pred_vec" in preds.columns
