"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/foldcheck_review/visuals.py

Visual manifest orchestration for Eco1 fold-check review.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_review.constants import (
    CHIMERAX_DIR_NAME,
    NOTEBOOKS_DIR_NAME,
    PLOTS_DIR_NAME,
    REVIEW_NOTEBOOK_FILE_NAME,
    VISUAL_MANIFEST_FILE_NAME,
    VISUAL_MANIFEST_SCHEMA_ID,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_review.models import PanelEntry
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_review.notebook import (
    write_review_notebook,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_review.plots import (
    write_review_plot_rows,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.foldcheck_review.structure_overlay import (
    write_structure_overlay_plot_row,
)


def write_review_visuals(
    *,
    review_root: Path,
    output_root: Path,
    ranking_rows: list[dict[str, Any]],
    reference_local_path: Path,
    panel_entries: list[PanelEntry],
    source_request_hash: str,
) -> tuple[Path, Path, int]:
    """Write compact review plots, a visual manifest, and a scoped marimo notebook."""

    if not ranking_rows:
        raise ValueError("fold-check review visuals require at least one ranking row")
    plots = write_review_plot_rows(
        plot_root=review_root / PLOTS_DIR_NAME,
        output_root=output_root,
        ranking_rows=ranking_rows,
    )
    plots.append(
        write_structure_overlay_plot_row(
            plot_root=review_root / PLOTS_DIR_NAME,
            chimerax_root=review_root / CHIMERAX_DIR_NAME,
            reference_local_path=reference_local_path,
            entries=panel_entries,
        )
    )
    notebook_path = review_root / NOTEBOOKS_DIR_NAME / REVIEW_NOTEBOOK_FILE_NAME
    write_review_notebook(notebook_path)

    manifest_path = review_root / VISUAL_MANIFEST_FILE_NAME
    manifest_root = manifest_path.parent
    relative_plots = [_with_manifest_relative_path(plot, manifest_root) for plot in plots]
    manifest = {
        "schema_id": VISUAL_MANIFEST_SCHEMA_ID,
        "schema_version": 1,
        "status": "materialized",
        "path_policy": "manifest_relative",
        "source_request_hash": source_request_hash,
        "plot_count": len(relative_plots),
        "plots": relative_plots,
        "notebook": {
            "path": _manifest_relative_path(notebook_path, manifest_root),
            "scope": "eco1_rt_repack fold-check review",
            "entrypoint": "marimo run eco1_foldcheck_review.py",
            "input_manifest": manifest_path.name,
            "description": (
                "Scoped marimo review notebook for fold metrics, cryoEM-reference "
                "comparison, and Biohub ESMC SAE coverage summaries."
            ),
        },
        "visual_policy": {
            "candidate_acceptance_gate": False,
            "requires_alt_text": True,
            "plain_language_limit": (
                "Plots summarize model-derived metrics for review. They do not measure "
                "RT activity, processivity, strand displacement, or hairpin readthrough."
            ),
        },
    }
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    return manifest_path, notebook_path, len(plots)


def _with_manifest_relative_path(row: dict[str, Any], manifest_root: Path) -> dict[str, Any]:
    normalized = dict(row)
    normalized["path"] = _manifest_relative_path(Path(str(row["path"])), manifest_root)
    return normalized


def _manifest_relative_path(path: Path, manifest_root: Path) -> str:
    if not path.is_absolute():
        return str(path)
    return os.path.relpath(path, start=manifest_root)
