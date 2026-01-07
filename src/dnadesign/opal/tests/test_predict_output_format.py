"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_predict_output_format.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import numpy as np
import pandas as pd

from dnadesign.opal.src.models.random_forest import RandomForestModel  # noqa: F401
from dnadesign.opal.src.runtime.predict import run_predict_ephemeral
from dnadesign.opal.src.storage.data_access import RecordsStore
from dnadesign.opal.src.transforms_x import identity  # noqa: F401 (registers)


def test_predict_returns_list_vectors(tmp_path):
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
    model_path = tmp_path / "model.joblib"
    model.save(str(model_path))

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
        model_path,
        model_name="random_forest",
        model_params=model.get_params(),
    )
    assert isinstance(preds["y_pred_vec"].iloc[0], list)
