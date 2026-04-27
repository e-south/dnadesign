"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/released_snapback/test_app_catalogs.py

Shared app-side catalog resolution tests for released-product Snapback.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.cruncher.app.snapback_released_catalogs import (
    released_catalog_sources_summary,
    resolve_released_catalogs,
)
from dnadesign.cruncher.snapback.models import CatalogSources
from dnadesign.cruncher.snapback.released_models import ReleaseCatalogSources


def _write_workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "workspaces" / "demo_released"
    nick_catalog_path = workspace / "inputs" / "nickases" / "local.nickases.yaml"
    release_catalog_path = workspace / "inputs" / "release_enzymes" / "local.release.yaml"
    nick_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    release_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    nick_catalog_path.write_text(
        yaml.safe_dump(
            {
                "nickases": {
                    "schema_version": 1,
                    "entries": [
                        {
                            "id": "Nx.Exact7",
                            "specificity_id": "Nx.Exact7",
                            "motif_top_5to3": "AACGTTG",
                            "top_cut_offset": 0,
                        }
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    release_catalog_path.write_text(
        yaml.safe_dump(
            {
                "release_enzymes": {
                    "schema_version": 1,
                    "entries": [
                        {
                            "variant_id": "Re.Exact",
                            "display_name": "Re.Exact",
                            "recognition_sequence": "CCAA",
                            "top_cut_offset": 1,
                            "bottom_cut_offset": 0,
                            "class_label": "other_ds_re",
                            "commercial_confidence": "primary_vendor_current",
                            "source_catalog_id": "local_release",
                        }
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return workspace


def test_released_catalog_sources_summary_uses_declared_sources_before_load() -> None:
    summary = released_catalog_sources_summary(
        nick_sources=CatalogSources(additional_paths=[Path("inputs/nickases/local.nickases.yaml")]),
        release_sources=ReleaseCatalogSources(additional_paths=[Path("inputs/release_enzymes/local.release.yaml")]),
    )

    assert summary.nick_catalog_source == "inputs/nickases/local.nickases.yaml"
    assert summary.release_catalog_source == "inputs/release_enzymes/local.release.yaml"


def test_resolve_released_catalogs_returns_loaded_catalogs_labels_and_snapshots(tmp_path: Path) -> None:
    workspace = _write_workspace(tmp_path)

    resolved = resolve_released_catalogs(
        nick_sources=CatalogSources(additional_paths=[Path("inputs/nickases/local.nickases.yaml")]),
        release_sources=ReleaseCatalogSources(additional_paths=[Path("inputs/release_enzymes/local.release.yaml")]),
        workspace_root=workspace,
    )

    assert resolved.nick_catalog.by_id()["Nx.Exact7"].id == "Nx.Exact7"
    assert resolved.release_catalog.by_id()["Re.Exact"].variant_id == "Re.Exact"
    assert "local.nickases.yaml" in resolved.nick_catalog_source
    assert "local.release.yaml" in resolved.release_catalog_source
    assert "Nx.Exact7" in resolved.nick_catalog_yaml
    assert "Re.Exact" in resolved.release_catalog_yaml
