"""Generated synthesis-handoff output layout."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .azenta import render_azenta_workbook, validate_azenta_workbook

CAMPAIGN_ROOT = Path("src/dnadesign/opal/campaigns")


def campaign_synthesis_output_dir(repo_root: str | Path, *, campaign_slug: str, batch_id: str) -> Path:
    """Return the default campaign-local synthesis handoff directory."""

    slug = str(campaign_slug).strip()
    batch = str(batch_id).strip()
    if not slug:
        raise ValueError("campaign_slug must be non-empty")
    if not batch:
        raise ValueError("batch_id must be non-empty")
    return Path(repo_root) / CAMPAIGN_ROOT / slug / "outputs" / "synthesis_handoff" / batch


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
) -> pd.DataFrame:
    """Write one manifest and one Azenta/GeneWiz workbook per campaign."""

    _require_campaign_manifest_columns(manifest)
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
        manifest_path = export_dir / "synthesis_manifest.csv"
        workbook_path = export_dir / "azenta_gene_synthesis.xlsx"
        campaign_manifest.to_csv(manifest_path, index=False)
        render_azenta_workbook(campaign_manifest, workbook_path)
        validation = validate_azenta_workbook(campaign_manifest, workbook_path)
        rows.append(
            {
                "campaign_slug": campaign,
                "batch_id": batch_id,
                "row_count": int(len(campaign_manifest)),
                "manifest_path": str(manifest_path),
                "azenta_workbook_path": str(workbook_path),
                "azenta_validation_status": validation["status"],
            }
        )

    exports = pd.DataFrame(rows)
    if output_root is not None:
        index_path = Path(output_root) / "handoff_index.csv"
        exports.to_csv(index_path, index=False)
    return exports
