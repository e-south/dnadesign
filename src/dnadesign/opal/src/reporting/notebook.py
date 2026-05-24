"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/reporting/notebook.py

Manifest-backed view-model helpers for generated OPAL campaign notebooks.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..analysis.campaign import CampaignAnalysis
from ..analysis.notebook_scope import resolve_notebook_run_scope
from ..core.utils import ExitCodes, OpalError, now_iso
from ..plots.config import list_configured_plot_specs, load_plot_config
from ..plots.manifests import (
    PLOT_MANIFEST_INDEX_SCHEMA_VERSION,
    load_plot_artifact_manifest,
    load_plot_manifest_index,
)
from .artifact_garden import build_artifact_garden_audit
from .progress import build_campaign_progress
from .review import load_review_manifest

NOTEBOOK_VIEW_MODEL_SCHEMA_VERSION = "opal.notebook_view_model.v1"


def build_notebook_view_model(
    config_path: str | Path | None,
    *,
    round_selector: str | None = "latest",
    run_id: str | None = None,
    review_manifest_path: str | Path | None = None,
    plot_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    analysis = CampaignAnalysis.from_config_path(Path(config_path) if config_path is not None else None, allow_dir=True)
    cfg = analysis.config
    ws = analysis.workspace
    warnings: list[dict[str, Any]] = []
    resolved_round_selector, resolved_run_id = resolve_notebook_run_scope(
        analysis,
        round_selector=round_selector,
        run_id=run_id,
    )

    try:
        progress = build_campaign_progress(
            analysis.config_path,
            round_selector=resolved_round_selector,
            run_id=resolved_run_id,
        )
    except Exception as exc:
        progress = {
            "schema_version": "opal.campaign_progress.unavailable",
            "status": "attention",
            "rounds": [],
            "error": str(exc),
        }
        warnings.append(_warning("ProgressContractError", str(exc), severity="error"))

    review_manifest = None
    review_path = (
        Path(review_manifest_path) if review_manifest_path is not None else ws.outputs_dir / "review" / "manifest.json"
    )
    if review_path.exists():
        try:
            review_manifest = load_review_manifest(review_path)
        except Exception as exc:
            warnings.append(_warning("ReviewManifestError", str(exc), path=review_path, severity="error"))
    else:
        warnings.append(
            _warning(
                "ReviewManifestWarning",
                f"Review manifest not found: {review_path}",
                path=review_path,
            )
        )

    plot_manifests = _load_plot_manifests(
        ws.outputs_dir / "plots",
        plot_manifest_path=plot_manifest_path,
        warnings=warnings,
    )
    configured_plots = _load_configured_plot_specs(
        campaign_cfg=analysis.read_config_dict(),
        config_path=analysis.config_path,
        campaign_dir=ws.workdir,
        warnings=warnings,
    )
    stale_artifacts = []
    if review_manifest is not None:
        stale_artifacts.extend(review_manifest.get("stale_artifacts") or [])
    stale_artifacts.extend(_detect_unmanifested_plot_outputs(ws.outputs_dir / "plots", plot_manifests))

    artifact_garden = None
    try:
        artifact_garden = build_artifact_garden_audit(analysis.config_path)
    except Exception as exc:
        warnings.append(_warning("ArtifactGardenWarning", str(exc)))

    return {
        "schema_version": NOTEBOOK_VIEW_MODEL_SCHEMA_VERSION,
        "generated_at": now_iso(),
        "campaign": {
            "name": cfg.campaign.name,
            "slug": cfg.campaign.slug,
            "description": cfg.campaign.description,
            "description_source": "config" if str(cfg.campaign.description or "").strip() else "derived",
            "metadata": dict(getattr(cfg.campaign, "metadata", {}) or {}),
            "workdir": str(ws.workdir),
            "config_path": str(analysis.config_path),
            "records_path": str(analysis.records_store().records_path),
            "x_column": cfg.data.x_column_name,
            "y_column": cfg.data.y_column_name,
            "label_source": getattr(cfg.labels.source, "kind", "campaign_history"),
            "model": cfg.model.name,
            "selection": cfg.selection.selection.name,
            "objectives": [objective.name for objective in cfg.objectives.objectives],
        },
        "status": {
            "progress_status": progress.get("status"),
            "round_selector": resolved_round_selector or "latest",
            "run_id_selector": resolved_run_id,
            "round_count": progress.get("round_count", 0),
            "latest_run_id": _latest_run_id(progress),
        },
        "progress": progress,
        "review_manifest_path": str(review_path),
        "review_manifest": review_manifest,
        "configured_plots": configured_plots,
        "plot_manifests": plot_manifests,
        "artifact_garden": artifact_garden,
        "stale_artifacts": stale_artifacts,
        "warnings": warnings,
    }


def smoke_check_notebook(path: str | Path, *, run_marimo_check: bool = True) -> dict[str, Any]:
    notebook_path = Path(path)
    text = notebook_path.read_text(encoding="utf-8")
    ast.parse(text)
    result: dict[str, Any] = {
        "schema_version": "opal.notebook_smoke.v1",
        "path": str(notebook_path),
        "python_parse_ok": True,
        "marimo_check_ok": None,
    }
    if run_marimo_check and importlib.util.find_spec("marimo") is not None:
        proc = subprocess.run(
            [sys.executable, "-m", "marimo", "check", str(notebook_path)],
            check=False,
            capture_output=True,
            text=True,
        )
        result.update(
            {
                "marimo_check_ok": proc.returncode == 0,
                "marimo_returncode": int(proc.returncode),
                "marimo_stdout": proc.stdout,
                "marimo_stderr": proc.stderr,
            }
        )
        if proc.returncode != 0:
            raise OpalError(f"marimo check failed for {notebook_path}", ExitCodes.CONTRACT_VIOLATION)
    return result


def _load_plot_manifests(
    plots_dir: Path,
    *,
    plot_manifest_path: str | Path | None,
    warnings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    index_path = Path(plot_manifest_path) if plot_manifest_path is not None else plots_dir / "plot_manifest.json"
    manifests: list[dict[str, Any]] = []
    if index_path.exists():
        try:
            index = load_plot_manifest_index(index_path)
            if index.get("schema_version") != PLOT_MANIFEST_INDEX_SCHEMA_VERSION:
                raise OpalError(f"Unsupported plot manifest index schema: {index.get('schema_version')!r}")
            for row in index.get("manifests") or []:
                manifest_path = row.get("manifest_path")
                if manifest_path and Path(str(manifest_path)).exists():
                    manifests.append(load_plot_artifact_manifest(str(manifest_path)))
                elif isinstance(row, dict):
                    manifests.append(row)
                    warnings.append(
                        _warning(
                            "PlotManifestError",
                            "Plot manifest index references a missing manifest.",
                            path=manifest_path,
                            severity="error",
                        )
                    )
        except Exception as exc:
            warnings.append(_warning("PlotManifestError", str(exc), path=index_path, severity="error"))
    elif plots_dir.exists():
        warnings.append(
            _warning(
                "StaleArtifactWarning",
                f"Plot manifest index not found: {index_path}",
                path=index_path,
            )
        )
    return manifests


def _load_configured_plot_specs(
    *,
    campaign_cfg: dict[str, Any],
    config_path: Path,
    campaign_dir: Path,
    warnings: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    try:
        plot_cfg = load_plot_config(
            campaign_cfg=campaign_cfg,
            campaign_yaml=config_path,
            campaign_dir=campaign_dir,
            plot_config_opt=None,
        )
        return [
            spec
            for spec in list_configured_plot_specs(
                plots_cfg=plot_cfg.plots,
                plot_presets=plot_cfg.plot_presets,
            )
            if spec.get("enabled")
        ]
    except Exception as exc:
        if "[plot] No plots found" in str(exc):
            return []
        warnings.append(_warning("PlotConfigWarning", str(exc), path=config_path))
        return []


def _detect_unmanifested_plot_outputs(plots_dir: Path, manifests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    referenced = {
        str(Path(str(output.get("path"))).resolve())
        for manifest in manifests
        for output in (manifest.get("outputs") or [])
        if isinstance(output, dict) and output.get("path")
    }
    if not plots_dir.exists():
        return []
    stale = []
    for path in sorted(plots_dir.iterdir()):
        if not path.is_file() or path.name == "plot_manifest.json" or path.name.endswith(".manifest.json"):
            continue
        if path.suffix.lower() not in {".png", ".svg", ".pdf", ".csv"}:
            continue
        if str(path.resolve()) not in referenced:
            stale.append(
                {
                    "category": "StaleArtifactWarning",
                    "severity": "warning",
                    "path": str(path),
                    "message": "Plot output exists on disk but is absent from the active plot manifest.",
                }
            )
    return stale


def _latest_run_id(progress: dict[str, Any]) -> str | None:
    for row in reversed(progress.get("rounds") or []):
        summary = row.get("summary") or {}
        run_scope = summary.get("run_scope") or {}
        if run_scope.get("resolved_run_id"):
            return str(run_scope["resolved_run_id"])
        run_ids = run_scope.get("run_ids") or []
        if run_ids:
            return str(run_ids[-1])
    return None


def _warning(
    category: str,
    message: str,
    *,
    path: str | Path | None = None,
    severity: str = "warning",
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "category": category,
        "severity": severity,
        "message": message,
    }
    if path is not None:
        row["path"] = str(path)
    return row
