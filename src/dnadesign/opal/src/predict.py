"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/predict.py

Ephemeral prediction helper.

Loads a frozen model and produces Ŷ for a set of ids (or all rows).
Validates that the configured representation column is present and coercible
to a fixed-width matrix. Never writes back; intended for quick scoring or
downstream analysis.

If a 'round_ctx.json' is present next to the model, any recorded training-time
Y-ops pipeline is inverted on Ŷ before returning (so outputs match objective
space used downstream).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from .data_access import RecordsStore
from .models.random_forest import RandomForestModel
from .registries.transforms_y import get_y_op
from .utils import ExitCodes, OpalError


def _load_yops_pipeline_from_ctx(ctx: Dict[str, Any]) -> List[Tuple[str, dict]]:
    names = ctx.get("yops/pipeline/names") or []
    params = ctx.get("yops/pipeline/params") or []
    out: List[Tuple[str, dict]] = []
    for i, n in enumerate(names):
        p = params[i] if i < len(params) else {}
        out.append((str(n), dict(p or {})))
    return out


class _CtxShim:
    """Tiny adapter so Y-op inverse() can ctx.get(...) from saved round_ctx.json."""

    def __init__(self, store: Dict[str, Any]):
        self._s = store

    def get(self, path: str, default: Any = None) -> Any:
        if path in self._s:
            return self._s[path]
        if default is not None:
            return default
        raise KeyError(path)

    # inverse() never writes, but be defensive
    def set(self, *args, **kwargs):  # noqa: D401 (no-op)
        raise RuntimeError("ctx.set is not supported in prediction shim")


def _inverse_yops_if_present(yhat: np.ndarray, model_path: Path) -> np.ndarray:
    ctx_path = model_path.parent / "round_ctx.json"
    if not ctx_path.exists():
        return yhat
    try:
        ctx_data: Dict[str, Any] = json.loads(ctx_path.read_text())
        pipeline = _load_yops_pipeline_from_ctx(ctx_data)
        if not pipeline:
            return yhat
        Y = np.asarray(yhat, dtype=float)
        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        shim = _CtxShim(ctx_data)
        for name, param_dict in reversed(pipeline):
            _, _, inv_fn, ParamT = get_y_op(name)
            params = ParamT(**param_dict) if ParamT is not None else param_dict
            Y = inv_fn(Y, params, ctx=shim)
        return Y
    except Exception:
        # best-effort: leave yhat as-is if anything goes wrong
        return yhat


def run_predict_ephemeral(
    store: RecordsStore,
    df: pd.DataFrame,
    model_path: Path,
    ids: List[str] | None = None,
) -> pd.DataFrame:
    mdl = RandomForestModel.load(str(model_path))
    if ids is None:
        ids = df["id"].astype(str).tolist()
    X, id_order = store.transform_matrix(df, ids)
    if X.shape[0] != len(id_order):
        raise OpalError(
            "Mismatch between inputs and transformed matrix count.",
            ExitCodes.INTERNAL_ERROR,
        )

    yhat = mdl.predict(X)
    yhat = _inverse_yops_if_present(yhat, Path(model_path))

    # normalize to list for dataframe export
    y_list = [list(map(float, row)) if yhat.ndim == 2 else [float(row)] for row in yhat]
    out = pd.DataFrame({"id": id_order, "y_pred_vec": [json.dumps(v) for v in y_list]})
    # add sequence
    seq_map = (
        df.set_index("id")["sequence"].astype(str).to_dict()
        if "sequence" in df.columns
        else {}
    )
    out["sequence"] = [seq_map.get(i, "") for i in id_order]
    return out[["id", "sequence", "y_pred_vec"]]
