"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/predict.py

Ephemeral prediction helper.

Loads a frozen model and produces Ŷ for a set of ids (or all rows).
Validates that the configured representation column is present and coercible
to a fixed-width matrix. Never writes back; intended for quick scoring or
downstream analysis.

Requires `model_meta.json` next to the model, unless an explicit model name is
provided. If a `round_ctx.json` is present, the recorded Y-ops pipeline is
strictly inverted on Ŷ before returning (so outputs match objective space).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .data_access import RecordsStore
from .registries.models import load_model
from .registries.transforms_x import get_transform_x
from .registries.transforms_y import run_y_ops_pipeline
from .round_context import PluginRegistryView, RoundCtx
from .utils import ExitCodes, OpalError


def _inverse_yops_if_present(yhat: np.ndarray, model_path: Path) -> np.ndarray:
    ctx_path = model_path.parent / "round_ctx.json"
    if not ctx_path.exists():
        return yhat
    try:
        ctx_data: Dict[str, Any] = json.loads(ctx_path.read_text())
        if "yops/pipeline/names" not in ctx_data or "yops/pipeline/params" not in ctx_data:
            raise OpalError(f"round_ctx.json missing Y-ops pipeline keys: {ctx_path}")
        Y = np.asarray(yhat, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        rctx = RoundCtx.from_snapshot(ctx_data)
        return run_y_ops_pipeline(stage="inverse", y_ops=[], Y=Y, ctx=rctx)
    except Exception as e:
        raise OpalError(f"Failed to invert Y-ops from {ctx_path}: {e}") from e


def run_predict_ephemeral(
    store: RecordsStore,
    df: pd.DataFrame,
    model_path: Path,
    ids: List[str] | None = None,
    *,
    model_name: str | None = None,
    model_params: Dict[str, Any] | None = None,
    id_column: str = "id",
    sequence_column: str = "sequence",
    generate_id_from_sequence: bool = False,
) -> pd.DataFrame:
    meta_path = model_path.parent / "model_meta.json"
    if model_name is None:
        if not meta_path.exists():
            raise OpalError(
                f"model_meta.json not found next to {model_path}. "
                "Provide --model-name/--model-params or re-run a round."
            )
        meta = json.loads(meta_path.read_text())
        model_name = meta.get("model__name")
        model_params = meta.get("model__params")
        if not model_name:
            raise OpalError(f"model_meta.json missing model__name: {meta_path}")
    mdl = load_model(str(model_name), str(model_path), params=model_params)
    df_work = df.copy()
    if id_column not in df_work.columns:
        if generate_id_from_sequence:
            if sequence_column not in df_work.columns:
                raise OpalError("Cannot generate ids: sequence column missing.")
            df_work["id"] = df_work[sequence_column].map(store.deterministic_id_from_sequence)
        else:
            raise OpalError(f"Input missing id column '{id_column}'. Use --generate-id-from-sequence if needed.")
    else:
        if id_column != "id":
            df_work["id"] = df_work[id_column].astype(str)
        else:
            df_work["id"] = df_work["id"].astype(str)

    if ids is None:
        ids = df_work["id"].astype(str).tolist()
    reg = PluginRegistryView(
        model=str(model_name),
        objective="unknown",
        selection="unknown",
        transform_x=store.x_transform_name,
        transform_y="unknown",
    )
    rctx = RoundCtx(
        core={
            "core/run_id": "predict-ephemeral",
            "core/round_index": -1,
            "core/campaign_slug": store.campaign_slug,
            "core/labels_as_of_round": -1,
            "core/plugins/transforms_x/name": store.x_transform_name,
            "core/plugins/model/name": str(model_name),
        },
        registry=reg,
    )
    tx = get_transform_x(store.x_transform_name, store.x_transform_params)
    tctx = rctx.for_plugin(category="transform_x", name=store.x_transform_name, plugin=tx)
    X, id_order = store.transform_matrix(df_work, ids, ctx=tctx)
    if X.shape[0] != len(id_order):
        raise OpalError(
            "Mismatch between inputs and transformed matrix count.",
            ExitCodes.INTERNAL_ERROR,
        )

    yhat = mdl.predict(X)
    yhat = _inverse_yops_if_present(yhat, Path(model_path))

    # normalize to list for dataframe export (list[float], parquet-friendly)
    y_list = [list(map(float, row)) if yhat.ndim == 2 else [float(row)] for row in yhat]
    out = pd.DataFrame({"id": id_order, "y_pred_vec": y_list})
    # add sequence
    seq_map = (
        df_work.set_index("id")[sequence_column].astype(str).to_dict() if sequence_column in df_work.columns else {}
    )
    out["sequence"] = [seq_map.get(i, "") for i in id_order]
    return out[["id", "sequence", "y_pred_vec"]]
