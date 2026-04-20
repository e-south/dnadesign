"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/nickases/test_catalog.py

Contract tests for the shared nickase catalog seam.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.cruncher.nickases.catalog import (
    load_builtin_nickase_catalog_preset,
    load_merged_nickase_catalog,
    load_nickase_catalog,
)
from dnadesign.cruncher.nickases.errors import NickaseCatalogError


def test_shared_catalog_parses_raw_cut_notation_into_normalized_offsets(tmp_path: Path) -> None:
    catalog_path = tmp_path / "nickases.yaml"
    catalog_path.write_text(
        yaml.safe_dump(
            {
                "nickases": {
                    "schema_version": 1,
                    "entries": [
                        {
                            "id": "Nt.BbvCI",
                            "specificity_id": "BbvCI",
                            "raw_cut_notation": "CCTCAGC(-5/none)",
                            "source": "neb",
                        },
                        {
                            "id": "Nb.BbvCI",
                            "specificity_id": "BbvCI",
                            "raw_cut_notation": "CCTCAGC(none/-2)",
                            "source": "neb",
                        },
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    catalog = load_nickase_catalog(catalog_path)
    entries = catalog.by_id()

    assert entries["Nt.BbvCI"].motif_top_5to3 == "CCTCAGC"
    assert entries["Nt.BbvCI"].top_cut_offset == -5
    assert entries["Nb.BbvCI"].bottom_cut_offset == -2


def test_shared_builtin_neb_preset_loads_and_preserves_product_alias_metadata() -> None:
    catalog = load_builtin_nickase_catalog_preset("neb_nicking_v1")

    entries = catalog.by_id()

    assert catalog.preset_id == "neb_nicking_v1"
    assert "Nt.BbvCI" in entries
    assert entries["Nt.BstNBI"].metadata["vendor_catalog_number"] == "R0607"
    assert any(alias.alias_id == "WarmStart Nt.BstNBI" for alias in catalog.product_aliases)


def test_shared_preset_overlay_merge_rejects_duplicate_variant_ids(tmp_path: Path) -> None:
    workspace = tmp_path / "workspaces" / "demo_snapback"
    overlay_path = workspace / "inputs" / "nickases" / "overlay.yaml"
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_path.write_text(
        yaml.safe_dump(
            {
                "nickases": {
                    "schema_version": 1,
                    "entries": [
                        {
                            "id": "Nt.BbvCI",
                            "specificity_id": "BbvCI",
                            "raw_cut_notation": "CCTCAGC(-5/none)",
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(NickaseCatalogError, match="Duplicate nickase id"):
        load_merged_nickase_catalog(
            preset_id="neb_nicking_v1",
            additional_paths=[Path("inputs/nickases/overlay.yaml")],
            workspace_root=workspace,
        )
