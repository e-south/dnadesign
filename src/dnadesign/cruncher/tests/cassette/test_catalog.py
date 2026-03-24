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

from dnadesign.cruncher.cassette.catalog import load_nickase_catalog
from dnadesign.cruncher.cassette.errors import NickaseCatalogError


def test_catalog_rejects_palindromic_recognition_sites(tmp_path: Path) -> None:
    catalog_path = tmp_path / "nickases.yaml"
    catalog_path.write_text(
        yaml.safe_dump(
            {
                "nickases": {
                    "schema_version": 1,
                    "entries": [
                        {
                            "id": "bad_palindrome",
                            "recognition_sequence": "GAATTC",
                            "nicked_site_strand": "forward",
                            "cut_offset": 2,
                        }
                    ],
                }
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(NickaseCatalogError, match="palindromic sites are ambiguous"):
        load_nickase_catalog(catalog_path)
