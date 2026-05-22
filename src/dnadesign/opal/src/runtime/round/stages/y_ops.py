"""
Y-op inverse helpers for OPAL round scoring.
"""

from __future__ import annotations

from typing import Any, List

import numpy as np

from ....core.round_context import Contract, RoundCtx
from ....core.utils import OpalError
from ....registries.transforms_y import get_y_op


def coalesce_uncertainty_chunks(std_payload: Any) -> np.ndarray:
    if isinstance(std_payload, list):
        if len(std_payload) == 0:
            raise OpalError("model/<self>/std_devs payload is empty after prediction.")
        chunks = [np.asarray(chunk, dtype=float) for chunk in std_payload]
        dims = {arr.ndim for arr in chunks}
        if dims == {1}:
            return np.concatenate([arr.reshape(-1) for arr in chunks], axis=0)
        if dims == {2}:
            widths = {arr.shape[1] for arr in chunks}
            if len(widths) != 1:
                raise OpalError("model/<self>/std_devs chunks have inconsistent column widths.")
            return np.vstack(chunks)
        raise OpalError("model/<self>/std_devs chunks must be all 1D or all 2D arrays.")
    arr = np.asarray(std_payload, dtype=float)
    if arr.ndim in (1, 2):
        return arr
    raise OpalError("model/<self>/std_devs payload must be a 1D or 2D numeric array.")


def inverse_yops_outputs(
    *,
    rctx: RoundCtx,
    y_ops_cfg: List[Any],
    y_pred: np.ndarray,
    y_pred_std: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    y_inv = np.asarray(y_pred, dtype=float)
    std_inv = None if y_pred_std is None else np.asarray(y_pred_std, dtype=float)
    if not y_ops_cfg:
        return y_inv, std_inv

    names = rctx.get("yops/pipeline/names", default=[])
    params_used = rctx.get("yops/pipeline/params", default=[])
    if len(names) != len(params_used):
        raise OpalError("Malformed Y-ops pipeline: names/params length mismatch.")

    for name, params in zip(reversed(names), reversed(params_used)):
        spec = get_y_op(name)
        ParamT = spec.ParamModel
        params_obj = ParamT(**params) if ParamT is not None else params
        inv_contract = Contract(
            category="yops",
            requires=spec.requires + spec.produces,
            produces=tuple(),
        )
        inv_ctx = rctx.for_plugin(category="yops", name=name, contract=inv_contract)
        inv_ctx.precheck_requires(stage="inverse")

        y_before = y_inv
        if std_inv is not None:
            inverse_std_fn = getattr(spec.inverse_fn, "inverse_std", None)
            if inverse_std_fn is None:
                raise OpalError(
                    "[round] y_ops are active but "
                    f"{name} does not support inverse_std; uncertainty channels require objective-space units."
                )
            std_inv = np.asarray(
                inverse_std_fn(std_inv, params_obj, ctx=inv_ctx, y_pred_transformed=y_before),
                dtype=float,
            )
            if std_inv.shape != y_before.shape:
                raise OpalError(
                    f"Y-op inverse_std shape mismatch for {name}: expected {y_before.shape}, got {std_inv.shape}."
                )
            if not np.all(np.isfinite(std_inv)):
                raise OpalError(f"Y-op inverse_std produced non-finite values for {name}.")
            if np.any(std_inv < 0.0):
                raise OpalError(f"Y-op inverse_std produced negative standard deviations for {name}.")

        y_inv = np.asarray(spec.inverse_fn(y_before, params_obj, ctx=inv_ctx), dtype=float)

    return y_inv, std_inv
