"""Model and artifact helpers used by dashboard overlays."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ...core.round_context import RoundCtx
from .artifacts import resolve_round_artifacts
from .datasets import CampaignInfo


def load_model_artifact(path: Path):
    try:
        import joblib

        return joblib.load(path), None
    except Exception as exc:
        return None, str(exc)


def unwrap_artifact_model(obj):
    if obj is None:
        return None
    if hasattr(obj, "predict"):
        return obj
    model_attr = getattr(obj, "model", None)
    if model_attr is not None and hasattr(model_attr, "predict"):
        return model_attr
    if isinstance(obj, dict):
        for key in ("model", "estimator", "rf", "random_forest"):
            candidate = obj.get(key)
            if candidate is not None and hasattr(candidate, "predict"):
                return candidate
    return None


def get_feature_importances(model):
    if model is None:
        return None
    if hasattr(model, "feature_importances_"):
        return getattr(model, "feature_importances_")
    if hasattr(model, "feature_importances"):
        try:
            return model.feature_importances()
        except Exception:
            return None
    return None


def load_feature_importances_from_artifact(round_dir: Path | None) -> tuple[np.ndarray | None, str | None]:
    if round_dir is None:
        return None, "Round directory unavailable for feature importance."
    path = Path(round_dir) / "feature_importance.csv"
    if not path.is_file():
        return None, f"feature_importance.csv not found under {round_dir}"
    try:
        df = pl.read_csv(path)
    except Exception as exc:
        return None, f"feature_importance.csv read failed: {exc}"
    if "importance" not in df.columns:
        return None, "feature_importance.csv missing required 'importance' column."
    try:
        arr = np.asarray(df.get_column("importance").to_numpy(), dtype=float).ravel()
    except Exception as exc:
        return None, f"feature_importance.csv parse failed: {exc}"
    if arr.size == 0:
        return None, "feature_importance.csv is empty."
    if not np.all(np.isfinite(arr)):
        return None, "feature_importance.csv contains non-finite values."
    return arr, None


def load_intensity_params_from_round_ctx(round_dir: Path, *, eps_default: float):
    ctx_path = round_dir / "round_ctx.json"
    if not ctx_path.is_file():
        return None
    try:
        data = json.loads(ctx_path.read_text())
    except Exception:
        return None
    med = data.get("yops/intensity_median_iqr/center")
    scale = data.get("yops/intensity_median_iqr/scale")
    if med is None or scale is None:
        return None
    med_arr = np.asarray(med, dtype=float)
    scale_arr = np.asarray(scale, dtype=float)
    if med_arr.shape != (4,) or scale_arr.shape != (4,):
        return None
    enabled = bool(data.get("yops/intensity_median_iqr/enabled", False))
    eps_val = float(data.get("yops/intensity_median_iqr/eps", eps_default))
    return med_arr, scale_arr, enabled, eps_val


def load_round_ctx_from_dir(round_dir: Path) -> tuple[RoundCtx | None, str | None]:
    ctx_path = round_dir / "round_ctx.json"
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


@dataclass(frozen=True)
class ArtifactModelResult:
    requested_artifact: bool
    use_artifact: bool
    model: Any | None
    model_path: Path | None
    round_dir: Path | None
    note: str
    warning: str | None = None


def resolve_artifact_model(
    *,
    use_artifact: bool,
    campaign_info: CampaignInfo | None,
    as_of_round: int | None,
    run_id: str | None,
) -> ArtifactModelResult:
    if not use_artifact:
        return ArtifactModelResult(
            requested_artifact=False,
            use_artifact=False,
            model=None,
            model_path=None,
            round_dir=None,
            note="Artifact loading disabled.",
        )
    if campaign_info is None:
        return ArtifactModelResult(
            requested_artifact=True,
            use_artifact=False,
            model=None,
            model_path=None,
            round_dir=None,
            note="No campaign selected; artifact unavailable.",
        )

    if as_of_round is None:
        return ArtifactModelResult(
            requested_artifact=True,
            use_artifact=False,
            model=None,
            model_path=None,
            round_dir=None,
            note="Artifact selection requires a round; select an as-of round first.",
        )

    artifacts, artifact_warning = resolve_round_artifacts(campaign_info.workdir, as_of_round=as_of_round)
    if not artifacts or "model.joblib" not in artifacts:
        note = f"Artifacts missing for round `R={as_of_round}`."
        if artifact_warning:
            note = f"{note} ({artifact_warning})"
        return ArtifactModelResult(
            requested_artifact=True,
            use_artifact=False,
            model=None,
            model_path=None,
            round_dir=None,
            note=note,
            warning=artifact_warning,
        )

    model_path = Path(artifacts["model.joblib"])
    round_dir = Path(artifacts.get("round_dir", model_path.parent))
    if not model_path.exists():
        return ArtifactModelResult(
            requested_artifact=True,
            use_artifact=False,
            model=None,
            model_path=model_path,
            round_dir=round_dir,
            note=f"Artifact path missing on disk: `{model_path}`.",
            warning=artifact_warning,
        )
    obj, err = load_model_artifact(model_path)
    model = unwrap_artifact_model(obj)
    if model is None:
        msg = err or "Unsupported artifact format"
        return ArtifactModelResult(
            requested_artifact=True,
            use_artifact=False,
            model=None,
            model_path=model_path,
            round_dir=round_dir,
            note=f"Artifact load failed: {msg}.",
            warning=artifact_warning,
        )
    note = f"Loaded artifact for round `R={as_of_round}`: `{model_path}`"
    if run_id is not None:
        note = f"{note} (run_id `{run_id}`)"
    if artifact_warning:
        note += f" (note: {artifact_warning})"
    return ArtifactModelResult(
        requested_artifact=True,
        use_artifact=True,
        model=model,
        model_path=model_path,
        round_dir=round_dir,
        note=note,
        warning=artifact_warning,
    )


def resolve_artifact_state(
    *,
    campaign_info: CampaignInfo | None,
    as_of_round: int | None,
    run_id: str | None,
) -> ArtifactModelResult:
    return resolve_artifact_model(
        use_artifact=True,
        campaign_info=campaign_info,
        as_of_round=as_of_round,
        run_id=run_id,
    )
