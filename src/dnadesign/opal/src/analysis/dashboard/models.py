"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/dashboard/models.py

Loads model artifacts and round context for dashboard overlays. Centralizes.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from ...core.round_context import RoundCtx
from ...models.random_forest import RandomForestModel


def load_model_artifact(path: Path):
    try:
        import joblib

        return joblib.load(path), None
    except Exception as exc:
        return None, str(exc)


def unwrap_artifact_model(obj):
    if obj is None:
        return None
    rf_wrapped = RandomForestModel.from_artifact(obj)
    if rf_wrapped is not None:
        return rf_wrapped
    if hasattr(obj, "predict"):
        return obj
    model_attr = getattr(obj, "model", None)
    if model_attr is not None and hasattr(model_attr, "predict"):
        rf_wrapped = RandomForestModel.from_artifact(model_attr)
        return rf_wrapped if rf_wrapped is not None else model_attr
    if isinstance(obj, dict):
        for key in ("model", "estimator", "rf", "random_forest"):
            candidate = obj.get(key)
            if candidate is not None and hasattr(candidate, "predict"):
                rf_wrapped = RandomForestModel.from_artifact(candidate)
                return rf_wrapped if rf_wrapped is not None else candidate
    return None


def load_round_ctx_from_dir(round_dir: Path) -> tuple[RoundCtx | None, str | None]:
    ctx_path = round_dir / "metadata" / "round_ctx.json"
    if not ctx_path.is_file():
        return None, f"round_ctx.json not found under {round_dir}"
    try:
        snapshot = json.loads(ctx_path.read_text())
    except Exception as exc:
        return None, f"round_ctx.json read failed: {exc}"
    try:
        return RoundCtx.from_snapshot(snapshot), None
    except Exception as exc:
        return None, f"round_ctx.json parse failed: {exc}"
