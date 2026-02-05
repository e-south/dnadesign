"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/analyze/manifests.py

Build analysis plot/table manifests for the curated v3 analysis suite.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from dnadesign.cruncher.analysis.layout import (
    ANALYSIS_LAYOUT_VERSION,
    analysis_manifest_path,
    plot_manifest_path,
    table_manifest_path,
)


@dataclass
class AnalysisManifestBundle:
    plot_manifest: dict[str, object]
    plot_manifest_file: Path
    table_manifest: dict[str, object]
    table_manifest_file: Path
    analysis_manifest_payload: dict[str, object]
    analysis_manifest_file: Path


def build_analysis_manifests(
    *,
    analysis_id: str,
    created_at: str,
    analysis_root: Path,
    analysis_used_file: Path,
    plot_entries: list[dict[str, object]],
    table_entries: list[dict[str, object]],
    analysis_artifacts: list[dict[str, object]],
) -> AnalysisManifestBundle:
    plot_manifest = {
        "analysis_id": analysis_id,
        "created_at": created_at,
        "plots": plot_entries,
    }
    plot_manifest_file = plot_manifest_path(analysis_root)
    plot_manifest_file.parent.mkdir(parents=True, exist_ok=True)
    plot_manifest_file.write_text(json.dumps(plot_manifest, indent=2))

    table_manifest = {
        "analysis_id": analysis_id,
        "created_at": created_at,
        "tables": table_entries,
    }
    table_manifest_file = table_manifest_path(analysis_root)
    table_manifest_file.parent.mkdir(parents=True, exist_ok=True)
    table_manifest_file.write_text(json.dumps(table_manifest, indent=2))

    analysis_manifest_payload = {
        "analysis_id": analysis_id,
        "created_at": created_at,
        "analysis_layout_version": ANALYSIS_LAYOUT_VERSION,
        "analysis_used": str(analysis_used_file.relative_to(analysis_root.parent)),
        "plot_manifest": str(plot_manifest_file.relative_to(analysis_root.parent)),
        "table_manifest": str(table_manifest_file.relative_to(analysis_root.parent)),
        "artifacts": analysis_artifacts,
    }
    analysis_manifest_file = analysis_manifest_path(analysis_root)
    analysis_manifest_file.parent.mkdir(parents=True, exist_ok=True)
    analysis_manifest_file.write_text(json.dumps(analysis_manifest_payload, indent=2))

    return AnalysisManifestBundle(
        plot_manifest=plot_manifest,
        plot_manifest_file=plot_manifest_file,
        table_manifest=table_manifest,
        table_manifest_file=table_manifest_file,
        analysis_manifest_payload=analysis_manifest_payload,
        analysis_manifest_file=analysis_manifest_file,
    )
