"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/scripts/test_export_genbank_baserender_projection.py

Tests for USR-owned GenBank BaseRender projection helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.usr.scripts.export_genbank_baserender_projection import project_genbank_baserender_rows
from dnadesign.usr.src.contracts import SchemaError


def test_project_genbank_baserender_rows_keeps_only_annotation_backed_rows() -> None:
    rows = [
        {
            "id": "annotated",
            "sequence": "ACGT",
            "usr_label__primary": "demoP",
            "seq_annot__source_file": "demo.gb",
            "seq_annot__features": [{"feature_id": "f1", "start_0": 0, "end_0": 4}],
            "derived__product_kind": "selected_region",
            "unrelated": "drop-me",
        },
        {
            "id": "unannotated",
            "sequence": "ACGT",
            "usr_label__primary": "unannotated_reference",
            "seq_annot__source_file": None,
            "seq_annot__features": None,
            "derived__product_kind": None,
        },
    ]

    projected, skipped = project_genbank_baserender_rows(rows)

    assert skipped == 1
    assert projected == [
        {
            "id": "annotated",
            "sequence": "ACGT",
            "usr_label__primary": "demoP",
            "seq_annot__source_file": "demo.gb",
            "seq_annot__features": [{"feature_id": "f1", "start_0": 0, "end_0": 4}],
            "derived__product_kind": "selected_region",
        }
    ]


def test_project_genbank_baserender_rows_rejects_malformed_annotation_payloads() -> None:
    rows = [
        {
            "id": "malformed",
            "sequence": "ACGT",
            "usr_label__primary": "demoP",
            "seq_annot__source_file": "demo.gb",
            "seq_annot__features": "not-a-feature-list",
            "derived__product_kind": "selected_region",
        }
    ]

    with pytest.raises(SchemaError, match="seq_annot__features must be a list"):
        project_genbank_baserender_rows(rows)
