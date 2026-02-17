"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_predict_robustness.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from dnadesign.opal.src.registries.models import list_models, register_model
from dnadesign.opal.src.runtime.predict import run_predict_ephemeral
from dnadesign.opal.src.storage.data_access import RecordsStore
from dnadesign.opal.src.transforms_x import identity  # noqa: F401 (registers)


class _ListModel:
    def __init__(self, params: dict) -> None:
        self._params = dict(params or {})

    def fit(self, X, Y, ctx=None):  # noqa: ANN001
        _ = X, Y, ctx
        return None

    def predict(self, X, ctx=None):  # noqa: ANN001
        _ = ctx
        n = int(getattr(X, "shape", [len(X)])[0])
        return [[float(i)] for i in range(n)]

    def get_params(self) -> dict:
        return dict(self._params)


def _register_list_model() -> str:
    name = "test_list_model_v1"
    if name in list_models():
        return name

    @register_model(name)
    def _factory(params: dict) -> _ListModel:
        return _ListModel(params)

    def _load(path: str, params: dict | None = None) -> _ListModel:
        _ = path
        return _ListModel(params or {})

    _factory.load = staticmethod(_load)  # type: ignore[attr-defined]
    return name


def test_predict_accepts_list_outputs_and_preserves_null_sequence(tmp_path: Path) -> None:
    model_name = _register_list_model()
    df = pd.DataFrame(
        {
            "id": ["a", "b"],
            "sequence": ["AAA", None],
            "bio_type": ["dna", "dna"],
            "alphabet": ["dna_4", "dna_4"],
            "X": [[0.1, 0.2], [0.2, 0.3]],
        }
    )

    store = RecordsStore(
        kind="local",
        records_path=tmp_path / "records.parquet",
        campaign_slug="demo",
        x_col="X",
        y_col="Y",
        x_transform_name="identity",
        x_transform_params={},
    )

    preds = run_predict_ephemeral(
        store,
        df,
        model_path=tmp_path / "model.joblib",
        model_name=model_name,
        model_params={},
    )
    assert isinstance(preds["y_pred_vec"].iloc[0], list)
    assert np.isclose(preds["y_pred_vec"].iloc[1][0], 1.0)
    assert pd.isna(preds.loc[1, "sequence"])
