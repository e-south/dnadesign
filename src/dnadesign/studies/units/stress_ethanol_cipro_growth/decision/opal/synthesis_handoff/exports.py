"""Generated synthesis-handoff output layout."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .azenta import render_azenta_workbook, validate_azenta_workbook
from .genbank import build_genbank_feature_table, render_genbank_record_set, validate_genbank_record_set

CAMPAIGN_ROOT = Path("src/dnadesign/opal/campaigns")

_STALE_GENERIC_ARTIFACT_NAMES = (
    "synthesis_manifest.csv",
    "azenta_gene_synthesis.xlsx",
)


def campaign_synthesis_output_dir(repo_root: str | Path, *, campaign_slug: str, batch_id: str) -> Path:
    """Return the default campaign-local synthesis handoff directory."""

    slug = str(campaign_slug).strip()
    batch = str(batch_id).strip()
    if not slug:
        raise ValueError("campaign_slug must be non-empty")
    if not batch:
        raise ValueError("batch_id must be non-empty")
    return Path(repo_root) / CAMPAIGN_ROOT / slug / "outputs" / "synthesis_handoff" / batch


def campaign_synthesis_artifact_paths(
    export_dir: str | Path,
    *,
    batch_id: str,
    campaign_slug: str,
) -> dict[str, Path]:
    """Return detached-safe artifact paths for one campaign handoff export."""

    batch = str(batch_id).strip()
    campaign = str(campaign_slug).strip()
    if not batch:
        raise ValueError("batch_id must be non-empty")
    if not campaign:
        raise ValueError("campaign_slug must be non-empty")
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
) -> Path:
    if output_root is not None:
        return Path(output_root) / campaign_slug
    if repo_root is None:
        raise ValueError("repo_root is required when output_root is not provided")
    return campaign_synthesis_output_dir(repo_root, campaign_slug=campaign_slug, batch_id=batch_id)


def render_campaign_scoped_exports(
    manifest: pd.DataFrame,
    *,
    batch_id: str,
    repo_root: str | Path | None = None,
    output_root: str | Path | None = None,
    candidate_records_path: str | Path | None = None,
) -> pd.DataFrame:
    """Write campaign-scoped synthesis handoff artifacts."""

    _require_campaign_manifest_columns(manifest)
    feature_table = build_genbank_feature_table(manifest, candidate_records_path=candidate_records_path)
    rows: list[dict[str, Any]] = []
    for campaign_slug, campaign_manifest in manifest.groupby("campaign_slug", sort=False):
        campaign = str(campaign_slug)
        export_dir = _export_dir_for(
            repo_root=repo_root,
            output_root=output_root,
            campaign_slug=campaign,
            batch_id=batch_id,
        )
        export_dir.mkdir(parents=True, exist_ok=True)
        for stale_name in _STALE_GENERIC_ARTIFACT_NAMES:
            stale_path = export_dir / stale_name
            if stale_path.exists():
                stale_path.unlink()
        artifact_paths = campaign_synthesis_artifact_paths(
            export_dir,
            batch_id=batch_id,
            campaign_slug=campaign,
        )
        manifest_path = artifact_paths["manifest"]
        workbook_path = artifact_paths["azenta_workbook"]
        genbank_dir_path = artifact_paths["genbank_dir"]
        feature_table_path = artifact_paths["genbank_feature_table"]
        stale_genbank_aggregate_path = artifact_paths["stale_genbank_aggregate"]
        if stale_genbank_aggregate_path.exists():
            stale_genbank_aggregate_path.unlink()
        campaign_feature_table = feature_table.loc[feature_table["campaign_slug"].astype(str) == campaign].reset_index(
            drop=True
        )
        campaign_manifest.to_csv(manifest_path, index=False)
        render_azenta_workbook(campaign_manifest, workbook_path)
        campaign_feature_table.to_csv(feature_table_path, index=False)
        render_genbank_record_set(campaign_manifest, campaign_feature_table, genbank_dir_path)
        workbook_validation = validate_azenta_workbook(campaign_manifest, workbook_path)
        genbank_validation = validate_genbank_record_set(campaign_manifest, genbank_dir_path)
        rows.append(
            {
                "campaign_slug": campaign,
                "batch_id": batch_id,
                "row_count": int(len(campaign_manifest)),
                "manifest_path": str(manifest_path),
                "azenta_workbook_path": str(workbook_path),
                "azenta_validation_status": workbook_validation["status"],
                "genbank_dir_path": str(genbank_dir_path),
                "genbank_feature_table_path": str(feature_table_path),
                "genbank_validation_status": genbank_validation["status"],
            }
        )

    exports = pd.DataFrame(rows)
    if output_root is not None:
        index_path = Path(output_root) / "handoff_index.csv"
        exports.to_csv(index_path, index=False)
    return exports
