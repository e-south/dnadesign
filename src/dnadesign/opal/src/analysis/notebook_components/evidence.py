"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/evidence.py

Notebook component builders for evidence OPAL analysis notebook components.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Mapping

from ._support import compact_path, join_list, mapping, sequence


def build_notebook_evidence_rows(view_model: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return warning and stale-artifact rows for notebook evidence tables."""

    campaign = mapping(view_model.get("campaign"))
    workdir = campaign.get("workdir")
    rows: list[dict[str, Any]] = []
    for label, path in (
        ("config", campaign.get("config_path")),
        ("workdir", workdir),
        ("records", campaign.get("records_path")),
        ("review_manifest", view_model.get("review_manifest_path")),
    ):
        if path:
            rows.append(
                {
                    "source": "path",
                    "category": label,
                    "severity": None,
                    "message": compact_path(path, base=workdir),
                    "path": compact_path(path, base=workdir),
                }
            )
    for warning in sequence(view_model.get("warnings")):
        if isinstance(warning, Mapping):
            rows.append(
                {
                    "source": "warning",
                    "category": warning.get("category"),
                    "severity": warning.get("severity"),
                    "message": warning.get("message"),
                    "path": compact_path(warning.get("path"), base=workdir),
                }
            )
    for artifact in sequence(view_model.get("stale_artifacts")):
        if isinstance(artifact, Mapping):
            rows.append(
                {
                    "source": "stale_artifact",
                    "category": artifact.get("category"),
                    "severity": artifact.get("severity"),
                    "message": artifact.get("message"),
                    "path": compact_path(artifact.get("path"), base=workdir),
                }
            )
    return rows


def build_notebook_metric_definition_rows(view_model: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return plot metric/data-shape definitions for notebook evidence tables."""

    rows: list[dict[str, Any]] = []
    for manifest in sequence(view_model.get("plot_manifests")):
        if not isinstance(manifest, Mapping):
            continue
        metadata = mapping(manifest.get("metadata"))
        freshness = mapping(manifest.get("freshness"))
        purpose = manifest.get("review_purpose") or manifest.get("caption") or metadata.get("summary") or "not recorded"
        rows.append(
            {
                "plot": manifest.get("name"),
                "kind": manifest.get("kind"),
                "data_shape": metadata.get("data_shape") or "not recorded",
                "tidy_schema": join_list(metadata.get("tidy_schema"), sep=", "),
                "failure_modes": join_list(metadata.get("failure_modes"), sep="; "),
                "freshness": freshness.get("status") or "unknown",
                "purpose": purpose,
            }
        )
    return rows


__all__ = ["build_notebook_evidence_rows", "build_notebook_metric_definition_rows"]
