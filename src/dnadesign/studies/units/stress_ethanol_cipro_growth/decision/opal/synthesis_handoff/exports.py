"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/synthesis_handoff/exports.py

Generated synthesis-handoff output layout.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
import shutil
import tempfile
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from ..source_evidence import sfxi_round0_source_evidence_dir
from .azenta import render_azenta_workbook, validate_azenta_workbook
from .genbank import build_genbank_feature_table, render_genbank_record_set, validate_genbank_record_set

CAMPAIGN_ROOT = Path("src/dnadesign/opal/campaigns")
_SAFE_PATH_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def _safe_path_component(value: object, *, label: str) -> str:
    text = str(value).strip()
    if _SAFE_PATH_COMPONENT.fullmatch(text) is None:
        raise ValueError(f"{label} must be a non-empty safe path component: {text!r}")
    return text


def campaign_synthesis_output_dir(repo_root: str | Path, *, campaign_slug: str, batch_id: str) -> Path:
    """Return the default campaign-local synthesis handoff directory."""

    slug = _safe_path_component(campaign_slug, label="campaign_slug")
    batch = _safe_path_component(batch_id, label="batch_id")
    return Path(repo_root) / CAMPAIGN_ROOT / slug / "outputs" / "synthesis_handoff" / batch


def source_evidence_synthesis_output_dir(repo_root: str | Path, *, campaign_slug: str, batch_id: str) -> Path:
    """Return the synthesis handoff directory owned by one SFXI source artifact."""

    campaign = _safe_path_component(campaign_slug, label="campaign_slug")
    batch = _safe_path_component(batch_id, label="batch_id")
    source_dir = sfxi_round0_source_evidence_dir(repo_root, source_slug=campaign)
    return source_dir / "outputs" / "synthesis_handoff" / batch


def campaign_synthesis_artifact_paths(
    export_dir: str | Path,
    *,
    batch_id: str,
    campaign_slug: str,
) -> dict[str, Path]:
    """Return detached-safe artifact paths for one campaign handoff export."""

    batch = _safe_path_component(batch_id, label="batch_id")
    campaign = _safe_path_component(campaign_slug, label="campaign_slug")
    prefix = f"{batch}__{campaign}"
    root = Path(export_dir)
    return {
        "manifest": root / f"{prefix}__synthesis_manifest.csv",
        "azenta_workbook": root / f"{prefix}__azenta_gene_synthesis.xlsx",
        "genbank_dir": root / f"{prefix}__genbank_inserts",
        "genbank_feature_table": root / f"{prefix}__genbank_features.csv",
        "stale_genbank_aggregate": root / f"{prefix}__annotated_inserts.gb",
    }


def _require_campaign_manifest_columns(manifest: pd.DataFrame) -> None:
    missing = [
        column for column in ("campaign_slug", "synthesis_name", "final_sequence") if column not in manifest.columns
    ]
    if missing:
        raise ValueError("synthesis manifest missing required campaign export columns: " + ", ".join(missing))


def _export_dir_for(
    *,
    repo_root: str | Path | None,
    output_root: str | Path | None,
    campaign_slug: str,
    batch_id: str,
    output_owner: Literal["campaign", "source_evidence"],
) -> Path:
    campaign_slug = _safe_path_component(campaign_slug, label="campaign_slug")
    batch_id = _safe_path_component(batch_id, label="batch_id")
    if output_root is not None:
        return Path(output_root) / campaign_slug
    if repo_root is None:
        raise ValueError("repo_root is required when output_root is not provided")
    if output_owner == "source_evidence":
        return source_evidence_synthesis_output_dir(
            repo_root,
            campaign_slug=campaign_slug,
            batch_id=batch_id,
        )
    if output_owner == "campaign":
        return campaign_synthesis_output_dir(
            repo_root,
            campaign_slug=campaign_slug,
            batch_id=batch_id,
        )
    raise ValueError(f"unknown synthesis output owner: {output_owner!r}")


def render_campaign_scoped_exports(
    manifest: pd.DataFrame,
    *,
    batch_id: str,
    output_owner: Literal["campaign", "source_evidence"],
    repo_root: str | Path | None = None,
    output_root: str | Path | None = None,
    candidate_records_path: str | Path | None = None,
) -> pd.DataFrame:
    """Validate then atomically publish campaign-scoped handoff artifacts."""

    _require_campaign_manifest_columns(manifest)
    feature_table = build_genbank_feature_table(manifest, candidate_records_path=candidate_records_path)
    campaign_groups = [(str(slug), rows.copy()) for slug, rows in manifest.groupby("campaign_slug", sort=False)]
    final_dirs = {
        campaign: _export_dir_for(
            repo_root=repo_root,
            output_root=output_root,
            campaign_slug=campaign,
            batch_id=batch_id,
            output_owner=output_owner,
        )
        for campaign, _ in campaign_groups
    }
    existing_dirs = [str(path) for path in final_dirs.values() if path.exists()]
    if existing_dirs:
        raise ValueError("synthesis handoff export directories already exist: " + ", ".join(existing_dirs))
    final_index_path = None if output_root is None else Path(output_root) / "handoff_index.csv"
    if final_index_path is not None and final_index_path.exists():
        raise ValueError(f"synthesis handoff index already exists: {final_index_path}")

    staging_parent = Path(output_root) if output_root is not None else Path(repo_root or ".")
    staging_parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(tempfile.mkdtemp(prefix=".synthesis-handoff-", dir=staging_parent))
    published_dirs: list[Path] = []
    try:
        result_rows: list[dict[str, Any]] = []
        staged_dirs: dict[str, Path] = {}
        for campaign, campaign_manifest in campaign_groups:
            stage_dir = staging_root / campaign
            stage_dir.mkdir(parents=True)
            staged_dirs[campaign] = stage_dir
            staged_paths = campaign_synthesis_artifact_paths(
                stage_dir,
                batch_id=batch_id,
                campaign_slug=campaign,
            )
            final_paths = campaign_synthesis_artifact_paths(
                final_dirs[campaign],
                batch_id=batch_id,
                campaign_slug=campaign,
            )
            campaign_feature_table = feature_table.loc[
                feature_table["campaign_slug"].astype(str) == campaign
            ].reset_index(drop=True)
            campaign_manifest.to_csv(staged_paths["manifest"], index=False)
            render_azenta_workbook(campaign_manifest, staged_paths["azenta_workbook"])
            campaign_feature_table.to_csv(staged_paths["genbank_feature_table"], index=False)
            render_genbank_record_set(campaign_manifest, campaign_feature_table, staged_paths["genbank_dir"])
            workbook_validation = validate_azenta_workbook(campaign_manifest, staged_paths["azenta_workbook"])
            genbank_validation = validate_genbank_record_set(
                campaign_manifest,
                staged_paths["genbank_dir"],
                feature_table=staged_paths["genbank_feature_table"],
            )
            result_rows.append(
                {
                    "campaign_slug": campaign,
                    "batch_id": batch_id,
                    "row_count": int(len(campaign_manifest)),
                    "manifest_path": str(final_paths["manifest"]),
                    "azenta_workbook_path": str(final_paths["azenta_workbook"]),
                    "azenta_validation_status": workbook_validation["status"],
                    "genbank_dir_path": str(final_paths["genbank_dir"]),
                    "genbank_feature_table_path": str(final_paths["genbank_feature_table"]),
                    "genbank_validation_status": genbank_validation["status"],
                }
            )

        exports = pd.DataFrame(result_rows)
        staged_index_path = staging_root / "handoff_index.csv"
        if final_index_path is not None:
            exports.to_csv(staged_index_path, index=False)
        for campaign, _ in campaign_groups:
            final_dir = final_dirs[campaign]
            final_dir.parent.mkdir(parents=True, exist_ok=True)
            staged_dirs[campaign].replace(final_dir)
            published_dirs.append(final_dir)
        if final_index_path is not None:
            staged_index_path.replace(final_index_path)
        return exports
    except Exception:
        for published_dir in reversed(published_dirs):
            shutil.rmtree(published_dir, ignore_errors=True)
        raise
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
