"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_predict_yops_inversion.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from pydantic import BaseModel

from dnadesign.opal.src.models.random_forest import RandomForestModel  # noqa: F401
from dnadesign.opal.src.registries.transforms_y import list_y_ops, register_y_op
from dnadesign.opal.src.runtime.predict import run_predict_ephemeral
from dnadesign.opal.src.storage.data_access import RecordsStore
from dnadesign.opal.src.transforms_x import identity  # noqa: F401 (registers)


class _Params(BaseModel):
    pass


def _ensure_yop_add1() -> None:
    if "test_add1_invert" in list_y_ops():
        return

    @register_y_op("test_add1_invert")
    def _yop():
        def fit(Y, params, ctx=None):
            return None

        def transform(Y, params, ctx=None):
            return np.asarray(Y, dtype=float) + 1.0

        def inverse(Y, params, ctx=None):
            return np.asarray(Y, dtype=float) - 1.0

        return fit, transform, inverse, _Params


def test_predict_inverts_yops_when_round_ctx_present(tmp_path):
    _ensure_yop_add1()
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
    model = RandomForestModel(params={"n_estimators": 5, "random_state": 1, "bootstrap": True, "oob_score": False})
    model.fit(X_train, Y_train)
    model_path = tmp_path / "model.joblib"
    model.save(str(model_path))

    # model_meta declares Y-ops
    meta = {
        "model__name": "random_forest",
        "model__params": model.get_params(),
        "training__y_ops": [{"name": "test_add1_invert", "params": {}}],
    }
    (tmp_path / "model_meta.json").write_text(json.dumps(meta))

    # round_ctx.json with Y-ops pipeline
    ctx = {"yops/pipeline/names": ["test_add1_invert"], "yops/pipeline/params": [{}]}
    (tmp_path / "round_ctx.json").write_text(json.dumps(ctx))

    store = RecordsStore(
        kind="local",
        records_path=tmp_path / "records.parquet",
        campaign_slug="demo",
        x_col="X",
        y_col="Y",
        x_transform_name="identity",
        x_transform_params={},
    )

    yhat_raw = model.predict(X_train)
    preds = run_predict_ephemeral(store, df, model_path)
    y_pred = np.asarray(preds["y_pred_vec"].tolist(), dtype=float)
    assert np.allclose(y_pred, yhat_raw - 1.0)
