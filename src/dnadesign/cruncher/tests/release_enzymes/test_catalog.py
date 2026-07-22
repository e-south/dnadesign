"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/tests/release_enzymes/test_catalog.py

Contract tests for release-enzyme catalog loading and cut normalization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.cruncher.release_enzymes.catalog import (
    load_builtin_release_enzyme_catalog_preset,
    load_release_enzyme_catalog,
)
from dnadesign.cruncher.release_enzymes.errors import ReleaseEnzymeCatalogError
from dnadesign.cruncher.release_enzymes.models import ReleaseEnzymeEntry
from dnadesign.cruncher.release_enzymes.scanning import derive_release_cut


def _write_catalog(tmp_path: Path, entries: list[dict[str, object]]) -> Path:
    path = tmp_path / "catalog.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "release_enzymes": {
                    "schema_version": 1,
                    "entries": entries,
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def test_builtin_release_catalog_loads_current_type_iis_starting_set() -> None:
    catalog = load_builtin_release_enzyme_catalog_preset("type_iis_release_v1")

    by_id = catalog.by_id()
    assert {"BsaI-HFv2", "BsmBI-v2", "BbsI", "BbsI-HF", "PaqCI", "SapI", "BspQI"} <= set(by_id)
    assert by_id["BsaI-HFv2"].recognition_sequence == "GGTCTC"
    assert by_id["BsaI-HFv2"].recommended_5prime_flanking_bases == 6
    assert by_id["BsaI-HFv2"].outside_site is True
    assert by_id["BbsI-HF"].recognition_sequence == "GAAGAC"
    assert by_id["BbsI-HF"].top_cut_offset == 8
    assert by_id["BbsI-HF"].bottom_cut_offset == 12
    assert by_id["PaqCI"].recognition_sequence == "CACCTGC"
    assert by_id["PaqCI"].top_cut_offset == 11
    assert by_id["PaqCI"].bottom_cut_offset == 15
    assert "MULTI_SITE_REQUIRED" in by_id["PaqCI"].warning_codes
    assert "ACTIVATOR_REQUIRED" in by_id["PaqCI"].warning_codes
    assert by_id["BspQI"].recognition_sequence == "GCTCTTC"
    assert by_id["BspQI"].top_cut_offset == 8
    assert by_id["BspQI"].bottom_cut_offset == 11
    assert by_id["BspQI"].source_url == "https://www.neb.com/en-us/products/r3712-bspqi-hf"


def test_release_catalog_rejects_missing_second_cut_offset(tmp_path: Path) -> None:
    catalog_path = _write_catalog(
        tmp_path,
        [
            {
                "variant_id": "Re.Test",
                "display_name": "Re.Test",
                "recognition_sequence": "CCAA",
                "top_cut_offset": 0,
                "class_label": "other_ds_re",
                "commercial_confidence": "primary_vendor_current",
                "source_catalog_id": "local",
            }
        ],
    )

    with pytest.raises(ReleaseEnzymeCatalogError):
        load_release_enzyme_catalog(catalog_path)


def test_release_catalog_rejects_duplicate_variant_ids(tmp_path: Path) -> None:
    catalog_path = _write_catalog(
        tmp_path,
        [
            {
                "variant_id": "Re.Test",
                "display_name": "Re.Test",
                "recognition_sequence": "CCAA",
                "top_cut_offset": 0,
                "bottom_cut_offset": 1,
                "class_label": "other_ds_re",
                "commercial_confidence": "primary_vendor_current",
                "source_catalog_id": "local",
            },
            {
                "variant_id": "Re.Test",
                "display_name": "Re.Test Duplicate",
                "recognition_sequence": "GGTT",
                "top_cut_offset": 0,
                "bottom_cut_offset": 1,
                "class_label": "other_ds_re",
                "commercial_confidence": "primary_vendor_current",
                "source_catalog_id": "local",
            },
        ],
    )

    with pytest.raises(ReleaseEnzymeCatalogError):
        load_release_enzyme_catalog(catalog_path)


def test_release_cut_normalizes_reverse_orientation_to_upstream_coordinates() -> None:
    entry = ReleaseEnzymeEntry(
        variant_id="BsaI-HFv2",
        display_name="BsaI-HFv2",
        recognition_sequence="GGTCTC",
        top_cut_offset=7,
        bottom_cut_offset=11,
        class_label="type_iis",
        commercial_confidence="primary_vendor_current",
        source_catalog_id="type_iis_release_v1",
    )

    cut = derive_release_cut(entry=entry, start=12, orientation="reverse")

    assert cut.top_cut_boundary == 7
    assert cut.bottom_cut_boundary == 11
