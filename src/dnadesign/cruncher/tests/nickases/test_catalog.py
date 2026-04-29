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
from dnadesign.cruncher.snapback.models import build_catalog_info


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


def test_shared_catalog_normalizes_outside_site_raw_cut_offsets_relative_to_motif_end(tmp_path: Path) -> None:
    catalog_path = tmp_path / "nickases.yaml"
    catalog_path.write_text(
        yaml.safe_dump(
            {
                "nickases": {
                    "schema_version": 1,
                    "entries": [
                        {
                            "id": "Nt.BsmAI",
                            "specificity_id": "BsmAI",
                            "raw_cut_notation": "GTCTC(1/none)",
                            "selection": {"outside_site": True},
                            "source": "neb",
                        },
                        {
                            "id": "Nt.BstNBI",
                            "specificity_id": "BstNBI",
                            "raw_cut_notation": "GAGTC(4/none)",
                            "selection": {"outside_site": True},
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

    assert entries["Nt.BsmAI"].top_cut_offset == 6
    assert entries["Nt.BstNBI"].top_cut_offset == 9
    assert entries["Nt.BsmAI"].resolved_vendor_diagram_top_5to3 == "GTCTCNN"
    assert entries["Nt.BstNBI"].resolved_vendor_diagram_top_5to3 == "GAGTCNNNNN"


def test_shared_catalog_normalizes_vendor_cut_notation_from_motif_end_when_declared(tmp_path: Path) -> None:
    catalog_path = tmp_path / "nickases.yaml"
    catalog_path.write_text(
        yaml.safe_dump(
            {
                "nickases": {
                    "schema_version": 1,
                    "entries": [
                        {
                            "id": "Nb.BtsI",
                            "specificity_id": "BtsI",
                            "motif_top_5to3": "GCAGTG",
                            "vendor_diagram_top_5to3": "GCAGTGNN",
                            "raw_cut_notation": "GCAGTG(none/0)",
                            "raw_cut_offset_reference": "motif_end",
                            "source": "neb",
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    catalog = load_nickase_catalog(catalog_path)
    entry = catalog.by_id()["Nb.BtsI"]

    assert entry.bottom_cut_offset == 6
    assert entry.raw_cut_offset_reference == "motif_end"
    assert entry.resolved_vendor_diagram_top_5to3 == "GCAGTGNN"


def test_shared_builtin_neb_preset_loads_and_preserves_product_alias_metadata() -> None:
    catalog = load_builtin_nickase_catalog_preset("neb_nicking_v1")

    entries = catalog.by_id()
    aliases = {alias.alias_id: alias for alias in catalog.product_aliases}

    assert catalog.preset_id == "neb_nicking_v1"
    assert catalog.preset_ids == ["neb_nicking_v1"]
    assert "Nt.BbvCI" in entries
    assert entries["Nt.BstNBI"].vendor_catalog_number == "R0607"
    assert entries["Nt.BstNBI"].source_url == "https://www.neb.com/en-us/products/r0607-ntbstnbi"
    assert entries["Nb.BsrDI"].bottom_cut_offset == 6
    assert entries["Nb.BtsI"].bottom_cut_offset == 6
    assert entries["Nt.AlwI"].top_cut_offset == 9
    assert entries["Nt.BsmAI"].top_cut_offset == 6
    assert entries["Nt.BstNBI"].resolved_vendor_diagram_top_5to3 == "GAGTCNNNNN"
    assert entries["Nb.BsrDI"].resolved_vendor_diagram_top_5to3 == "GCAATGNN"
    assert entries["Nb.BtsI"].resolved_vendor_diagram_top_5to3 == "GCAGTGNN"
    assert entries["Nt.AlwI"].resolved_vendor_diagram_top_5to3 == "GGATCNNNNN"
    assert entries["Nt.BsmAI"].resolved_vendor_diagram_top_5to3 == "GTCTCNN"
    assert entries["Nt.BstNBI"].selection is not None
    assert entries["Nt.BstNBI"].selection.outside_site is True
    assert entries["Nb.BsmI"].selection is not None
    assert entries["Nb.BsmI"].selection.warning_codes == ["STAR_ACTIVITY_RISK", "DOUBLE_STRAND_CLEAVAGE_RISK"]
    assert any(alias.alias_id == "WarmStart Nt.BstNBI" for alias in catalog.product_aliases)
    assert aliases["SibEnzyme N.Bst9 I"].canonical_variant_id == "Nt.BstNBI"
    assert aliases["SibEnzyme N.Bst9 I"].source_url == "https://sibenzyme.com/product/n-bst9-i/"


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


def test_shared_multiple_builtin_presets_merge_and_preserve_typed_selection_metadata(tmp_path: Path) -> None:
    workspace = tmp_path / "workspaces" / "demo_snapback"
    workspace.mkdir(parents=True, exist_ok=True)

    catalog, resolved_paths = load_merged_nickase_catalog(
        preset_id="neb_nicking_v1",
        additional_preset_ids=["thermo_nicking_v1"],
        additional_paths=[],
        workspace_root=workspace,
    )

    entries = catalog.by_id()

    assert resolved_paths == []
    assert catalog.preset_ids == ["neb_nicking_v1", "thermo_nicking_v1"]
    assert entries["Nt.Bpu10I"].top_cut_offset == 2
    assert entries["Nb.Bpu10I"].bottom_cut_offset == -2
    assert entries["Nb.Mva1269I"].operational is not None
    assert entries["Nb.Mva1269I"].operational.buffer_family == "O"
    assert entries["Nb.Mva1269I"].source_url == "https://www.thermofisher.com/order/catalog/product/ER2051"
    assert entries["Nt.Bpu10I"].selection is not None
    assert entries["Nt.Bpu10I"].selection.warning_codes == ["NONSPECIFIC_NICKING_ASSAY_SIGNAL"]


def test_shared_catalog_info_builder_preserves_nickase_source_url() -> None:
    catalog = load_builtin_nickase_catalog_preset("neb_nicking_v1")

    info = build_catalog_info(catalog.by_id()["Nt.BstNBI"])

    assert info.source_url == "https://www.neb.com/en-us/products/r0607-ntbstnbi"
