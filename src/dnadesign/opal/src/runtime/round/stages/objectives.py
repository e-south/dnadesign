"""
Objective-channel evaluation for an OPAL round.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np

from ....core.objective_result import validate_objective_result_v2
from ....core.round_context import RoundCtx
from ....core.utils import OpalError
from ....registries.objectives import get_objective
from ..contracts import RoundInputs
from .telemetry import format_summary_stats_for_log, log


@dataclass(frozen=True)
class ObjectiveEvaluation:
    score_channels: Dict[str, np.ndarray]
    uncertainty_channels: Dict[str, np.ndarray]
    channel_modes: Dict[str, str]
    diagnostics_by_objective: Dict[str, Dict[str, Any]]
    objective_defs: List[Dict[str, Any]]


class _TrainView:
    def __init__(self, Y: np.ndarray, R: np.ndarray, as_of_round: int) -> None:
        self._Y = np.asarray(Y, dtype=float)
        self._R = np.asarray(R, dtype=int)
        self._as = int(as_of_round)

    def labels_count(self) -> int:
        return int(self._Y.shape[0])

    def iter_labels_y(self):
        for i in range(self._Y.shape[0]):
            yield self._Y[i, :]

    def iter_labels_y_current_round(self):
        mask = self._R == self._as
        for i in np.where(mask)[0].tolist():
            yield self._Y[i, :]


def evaluate_objectives(
    *,
    inputs: RoundInputs,
    rctx: RoundCtx,
    Y_hat: np.ndarray,
    y_pred_std: np.ndarray | None,
    Y_train: np.ndarray,
    R_train: np.ndarray,
    id_order_pool: List[str],
) -> ObjectiveEvaluation:
    cfg = inputs.cfg
    req = inputs.req
    train_view = _TrainView(Y_train, R_train, int(req.as_of_round))

    score_channels: Dict[str, np.ndarray] = {}
    uncertainty_channels: Dict[str, np.ndarray] = {}
    channel_modes: Dict[str, str] = {}
    diagnostics_by_objective: Dict[str, Dict[str, Any]] = {}
    objective_defs: List[Dict[str, Any]] = []

    for obj_ref in cfg.objectives.objectives:
        obj_name_i = obj_ref.name
        obj_params_i = dict(obj_ref.params)
        obj_fn = get_objective(obj_name_i)
        octx = rctx.for_plugin(category="objective", name=obj_name_i, plugin=obj_fn)
        try:
            raw_obj = obj_fn(
                y_pred=Y_hat,
                params=obj_params_i,
                ctx=octx,
                train_view=train_view,
                y_pred_std=y_pred_std,
            )
        except OpalError:
            raise
        except Exception as e:
            raise OpalError(f"Objective plugin '{obj_name_i}' failed: {e}") from e
        obj_res = validate_objective_result_v2(
            result=raw_obj,
            objective_name=obj_name_i,
            n_rows=len(id_order_pool),
        )
        diagnostics_by_objective[obj_name_i] = dict(obj_res.diagnostics or {})

        score_refs_for_obj: List[str] = []
        for channel_name, arr in obj_res.scores_by_name.items():
            ref = f"{obj_name_i}/{channel_name}"
            if ref in score_channels:
                raise OpalError(f"Duplicate score channel reference generated: {ref}")
            score_channels[ref] = arr
            channel_modes[ref] = obj_res.modes_by_name[channel_name]
            score_refs_for_obj.append(ref)

        uncertainty_refs_for_obj: List[str] = []
        for channel_name, arr in obj_res.uncertainty_by_name.items():
            ref = f"{obj_name_i}/{channel_name}"
            if ref in uncertainty_channels:
                raise OpalError(f"Duplicate uncertainty channel reference generated: {ref}")
            uncertainty_channels[ref] = arr
            uncertainty_refs_for_obj.append(ref)

        objective_defs.append(
            {
                "name": obj_name_i,
                "params": obj_params_i,
                "score_channels": score_refs_for_obj,
                "uncertainty_channels": uncertainty_refs_for_obj,
                "diagnostics_summary_keys": list((obj_res.diagnostics or {}).keys()),
            }
        )

        summary_stats = (obj_res.diagnostics or {}).get("summary_stats", {})
        if isinstance(summary_stats, dict) and summary_stats:
            kvs = format_summary_stats_for_log(summary_stats)
            log(req.verbose, f"[objective:{obj_name_i}] " + " | ".join(kvs))

    return ObjectiveEvaluation(
        score_channels=score_channels,
        uncertainty_channels=uncertainty_channels,
        channel_modes=channel_modes,
        diagnostics_by_objective=diagnostics_by_objective,
        objective_defs=objective_defs,
    )
