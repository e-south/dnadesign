"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/predict.py

Ephemeral prediction helper.

Loads a frozen model and produces Ŷ for a set of ids (or all rows).
Validates that the configured representation column is present and coercible
to a fixed-width matrix. Never writes back; intended for quick scoring or
downstream analysis.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .data_access import RecordsStore
from .models.random_forest import RandomForestModel
from .utils import ExitCodes, OpalError


def run_predict_ephemeral(
    store: RecordsStore,
    df: pd.DataFrame,
    model_path: Path,
    ids: list[str] | None = None,
) -> pd.DataFrame:
    mdl = RandomForestModel.load(str(model_path))
    if ids is None:
        ids = df["id"].tolist()
    X, _ = store.transform_matrix(df, ids)
    if X.shape[0] != len(ids):
        raise OpalError(
            "Mismatch between inputs and transformed matrix count.",
            ExitCodes.INTERNAL_ERROR,
        )
    yhat = mdl.predict(X)
    # normalize to list for dataframe export
    y_list = [
        json.dumps(list(map(float, row))) if yhat.ndim == 2 else float(row)
        for row in yhat
    ]
    out = pd.DataFrame({"id": ids, "y_pred_vec": y_list})
    # add sequence
    seq_map = df.set_index("id")["sequence"].to_dict()
    out["sequence"] = [seq_map.get(i, "") for i in ids]
    return out[["id", "sequence", "y_pred_vec"]]
