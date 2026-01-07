"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/runtime/predict.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from ..core.round_context import PluginRegistryView, RoundCtx
from ..core.utils import ExitCodes, OpalError
from ..registries.models import load_model
from ..registries.transforms_x import get_transform_x
from ..registries.transforms_y import run_y_ops_pipeline
from ..storage.data_access import RecordsStore


def _load_model_meta(meta_path: Path) -> Optional[Dict[str, Any]]:
    if not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except Exception as e:
        raise OpalError(f"model_meta.json is not valid JSON: {meta_path}. {e}") from e
    if not isinstance(meta, dict):
        raise OpalError(f"model_meta.json must be a JSON object: {meta_path}")
    return meta


def _extract_y_ops(meta: Optional[Dict[str, Any]]) -> List[Any]:
    if not meta:
        return []
    yops = meta.get("training__y_ops") or meta.get("y_ops") or []
    if not isinstance(yops, list):
        raise OpalError("model_meta.json has invalid training__y_ops (expected list).")
    return yops


def _inverse_yops_if_present(
    yhat: np.ndarray,
    model_path: Path,
    *,
    require_ctx_if_yops: bool,
    y_ops_declared: bool,
) -> np.ndarray:
    ctx_path = model_path.parent / "round_ctx.json"
    if not ctx_path.exists():
        if require_ctx_if_yops and y_ops_declared:
            raise OpalError(
                "round_ctx.json is missing next to the model, but training used Y-ops. "
                "Provide --assume-no-yops to bypass inversion or copy the round_ctx.json alongside the model."
            )
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
    assume_no_yops: bool = False,
) -> pd.DataFrame:
    meta_path = model_path.parent / "model_meta.json"
    meta = _load_model_meta(meta_path)
    if model_name is None:
        if meta is None:
            raise OpalError(
                f"model_meta.json not found next to {model_path}. "
                "Provide --model-name/--model-params or re-run a round."
            )
        model_name = meta.get("model__name")
        model_params = meta.get("model__params")
        if not model_name:
            raise OpalError(f"model_meta.json missing model__name: {meta_path}")
    else:
        if meta is not None:
            meta_name = meta.get("model__name")
            if meta_name and str(meta_name) != str(model_name):
                raise OpalError(
                    f"Model name mismatch: model_meta.json declares '{meta_name}', "
                    f"but --model-name provided '{model_name}'."
                )
            if model_params is not None:
                raise OpalError(
                    "model_meta.json exists; --model-params is only allowed when model_meta.json is missing."
                )
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
    meta_x_dim = meta.get("x_dim") if meta is not None else None
    if meta_x_dim is not None:
        try:
            meta_x_dim = int(meta_x_dim)
        except Exception as e:
            raise OpalError(f"model_meta.json has non-integer x_dim: {meta_x_dim}") from e
        if X.shape[1] != meta_x_dim:
            raise OpalError(
                f"X dimension mismatch: model_meta.json expects x_dim={meta_x_dim}, "
                f"but transform_x produced {X.shape[1]} columns. Check that your config matches the model."
            )

    yhat = mdl.predict(X)
    y_ops_declared = bool(_extract_y_ops(meta))
    yhat = _inverse_yops_if_present(
        yhat,
        Path(model_path),
        require_ctx_if_yops=not assume_no_yops,
        y_ops_declared=y_ops_declared,
    )
    meta_y_dim = meta.get("y_dim") if meta is not None else None
    if meta_y_dim is not None:
        try:
            meta_y_dim = int(meta_y_dim)
        except Exception as e:
            raise OpalError(f"model_meta.json has non-integer y_dim: {meta_y_dim}") from e
        yhat_arr = np.asarray(yhat)
        if yhat_arr.ndim == 1:
            pred_dim = 1
        elif yhat_arr.ndim == 2:
            pred_dim = int(yhat_arr.shape[1])
        else:
            raise OpalError(f"Predicted Y has unexpected ndim={yhat_arr.ndim}.")
        if pred_dim != meta_y_dim:
            raise OpalError(
                f"Y dimension mismatch: model_meta.json expects y_dim={meta_y_dim}, but prediction produced {pred_dim}."
            )

    # normalize to list for dataframe export (list[float], parquet-friendly)
    y_list = [list(map(float, row)) if yhat.ndim == 2 else [float(row)] for row in yhat]
    out = pd.DataFrame({"id": id_order, "y_pred_vec": y_list})
    # add sequence
    seq_map = (
        df_work.set_index("id")[sequence_column].astype(str).to_dict() if sequence_column in df_work.columns else {}
    )
    out["sequence"] = [seq_map.get(i, "") for i in id_order]
    return out[["id", "sequence", "y_pred_vec"]]
