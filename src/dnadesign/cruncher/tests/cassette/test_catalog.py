"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cassette/test_catalog.py

Contract tests for the cassette nickase catalog.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.cruncher.cassette.catalog import (
    load_builtin_nickase_catalog_preset,
    load_merged_nickase_catalog,
    load_nickase_catalog,
)
from dnadesign.cruncher.cassette.errors import NickaseCatalogError


def test_catalog_parses_raw_cut_notation_into_normalized_offsets(tmp_path: Path) -> None:
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
    assert entries["Nt.BbvCI"].bottom_cut_offset is None
    assert entries["Nb.BbvCI"].motif_top_5to3 == "CCTCAGC"
    assert entries["Nb.BbvCI"].top_cut_offset is None
    assert entries["Nb.BbvCI"].bottom_cut_offset == -2


def test_catalog_accepts_iupac_motifs(tmp_path: Path) -> None:
    catalog_path = tmp_path / "nickases.yaml"
    catalog_path.write_text(
        yaml.safe_dump(
            {
                "nickases": {
                    "schema_version": 1,
                    "entries": [
                        {
                            "id": "Nt.ambiguous",
                            "specificity_id": "ambiguous",
                            "motif_top_5to3": "CCANNNG",
                            "top_cut_offset": 2,
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    catalog = load_nickase_catalog(catalog_path)

    assert catalog.entries[0].motif_top_5to3 == "CCANNNG"
    assert catalog.entries[0].motif_len == 7


def test_catalog_rejects_raw_cut_notation_that_disagrees_with_explicit_motif(tmp_path: Path) -> None:
    catalog_path = tmp_path / "nickases.yaml"
    catalog_path.write_text(
        yaml.safe_dump(
            {
                "nickases": {
                    "schema_version": 1,
                    "entries": [
                        {
                            "id": "Nt.bad",
                            "specificity_id": "bad",
                            "motif_top_5to3": "AACGA",
                            "raw_cut_notation": "CCTCAGC(-5/none)",
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(NickaseCatalogError, match="raw_cut_notation"):
        load_nickase_catalog(catalog_path)


def test_catalog_rejects_variants_with_both_cut_offsets(tmp_path: Path) -> None:
    catalog_path = tmp_path / "nickases.yaml"
    catalog_path.write_text(
        yaml.safe_dump(
            {
                "nickases": {
                    "schema_version": 1,
                    "entries": [
                        {
                            "id": "bad_variant",
                            "specificity_id": "bad_variant",
                            "motif_top_5to3": "AACGA",
                            "top_cut_offset": 2,
                            "bottom_cut_offset": 3,
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(NickaseCatalogError, match="exactly one"):
        load_nickase_catalog(catalog_path)


def test_builtin_neb_preset_loads_and_preserves_product_alias_metadata() -> None:
    catalog = load_builtin_nickase_catalog_preset("neb_nicking_v1")

    entries = catalog.by_id()

    assert catalog.preset_id == "neb_nicking_v1"
    assert catalog.preset_ids == ["neb_nicking_v1"]
    assert "Nt.BbvCI" in entries
    assert entries["Nt.BbvCI"].top_cut_offset == 2
    assert entries["Nb.BbvCI"].bottom_cut_offset == 5
    assert entries["Nb.BssSI"].bottom_cut_offset == 5
    assert entries["Nt.BstNBI"].vendor_catalog_number == "R0607"
    assert entries["Nt.BstNBI"].selection is not None
    assert entries["Nt.BstNBI"].selection.outside_site is True
    assert any(alias.alias_id == "WarmStart Nt.BstNBI" for alias in catalog.product_aliases)
    warmstart = next(alias for alias in catalog.product_aliases if alias.alias_id == "WarmStart Nt.BstNBI")
    assert warmstart.canonical_variant_id == "Nt.BstNBI"
    assert warmstart.alias_kind == "formulation"


def test_preset_overlay_merge_rejects_duplicate_variant_ids(tmp_path: Path) -> None:
    workspace = tmp_path / "workspaces" / "demo_cassette"
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
