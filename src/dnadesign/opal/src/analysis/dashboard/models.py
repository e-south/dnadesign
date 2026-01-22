"""Model and artifact helpers used by dashboard overlays."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from ...core.round_context import RoundCtx
from .artifacts import resolve_run_artifacts
from .datasets import CampaignInfo, resolve_campaign_workdir


def find_latest_model_artifact(info: CampaignInfo) -> tuple[Path | None, Path | None]:
    outputs_dir = resolve_campaign_workdir(info) / "outputs"
    if not outputs_dir.exists():
        return None, None
    candidates: list[tuple[int, Path, Path]] = []
    for round_dir in outputs_dir.glob("round_*"):
        model_path = round_dir / "model.joblib"
        if not model_path.is_file():
            continue
        try:
            round_idx = int(round_dir.name.split("_")[1])
        except Exception:
            round_idx = -1
        candidates.append((round_idx, model_path, round_dir))
    if not candidates:
        return None, None
    _, model_path, round_dir = max(candidates, key=lambda item: item[0])
    return model_path, round_dir


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
    run_id: str | None,
    ledger_runs_df: pl.DataFrame | None,
    allow_fallback: bool,
) -> ArtifactModelResult:
    if not use_artifact:
        return ArtifactModelResult(
            requested_artifact=False,
            use_artifact=False,
            model=None,
            model_path=None,
            round_dir=None,
            note="Overlay RF (session-scoped). Use the notebook compute button to run predictions.",
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

    artifact_warning = None
    artifacts = None
    if run_id:
        artifacts, artifact_warning = resolve_run_artifacts(ledger_runs_df, run_id=run_id)
        if artifacts and "model.joblib" not in artifacts:
            artifact_warning = "Artifacts missing model.joblib entry for selected run_id."

    if artifacts and "model.joblib" in artifacts:
        model_path = Path(artifacts["model.joblib"])
        round_ctx_path = artifacts.get("round_ctx.json")
        round_dir = Path(round_ctx_path).parent if round_ctx_path else model_path.parent
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
        note = f"Loaded artifact for run_id `{run_id}`: `{model_path}`"
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

    if not allow_fallback:
        if not run_id:
            note = "Artifact selection requires a run_id; no fallback allowed."
        else:
            note = f"Run-aware artifact not resolved for run_id `{run_id}`; no fallback allowed."
        return ArtifactModelResult(
            requested_artifact=True,
            use_artifact=False,
            model=None,
            model_path=None,
            round_dir=None,
            note=note,
            warning=artifact_warning,
        )

    try:
        model_path, round_dir = find_latest_model_artifact(campaign_info)
    except Exception as exc:
        return ArtifactModelResult(
            requested_artifact=True,
            use_artifact=False,
            model=None,
            model_path=None,
            round_dir=None,
            note=f"Artifact lookup failed: {exc}.",
            warning=artifact_warning,
        )
    if model_path is None:
        return ArtifactModelResult(
            requested_artifact=True,
            use_artifact=False,
            model=None,
            model_path=None,
            round_dir=None,
            note="No model.joblib found for this campaign.",
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
    note = f"Loaded artifact (latest round, not run-aware): `{model_path}`"
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
    run_id: str | None,
    ledger_runs_df: pl.DataFrame | None,
    allow_fallback: bool,
) -> ArtifactModelResult:
    return resolve_artifact_model(
        use_artifact=True,
        campaign_info=campaign_info,
        run_id=run_id,
        ledger_runs_df=ledger_runs_df,
        allow_fallback=allow_fallback,
    )
