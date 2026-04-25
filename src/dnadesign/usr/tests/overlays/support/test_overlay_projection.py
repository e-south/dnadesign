"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/overlays/support/test_overlay_projection.py

Tests for projecting namespaced overlay columns across USR datasets.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pyarrow as pa

from dnadesign.testsupport.usr import register_test_namespace
from dnadesign.usr import Dataset
from dnadesign.usr.src.overlays.support.projection import project_namespace_overlay


def _register_projection_namespaces(root: Path) -> None:
    register_test_namespace(
        root,
        namespace="densegen",
        columns_spec="densegen__plan:string,densegen__required_regulators:list<string>",
    )
    register_test_namespace(root, namespace="infer", columns_spec="infer__score:float64")
    register_test_namespace(root, namespace="construct", columns_spec="construct__anchor_id:string")


def _make_dataset(root: Path, name: str, sequences: list[str]) -> Dataset:
    dataset = Dataset(root, name)
    dataset.init(source="unit-test")
    dataset.import_rows(
        [
            {"sequence": sequence, "bio_type": "dna", "alphabet": "dna_4", "source": "unit-test"}
            for sequence in sequences
        ],
        source="unit-test",
    )
    return dataset


def test_project_namespace_overlay_preserves_existing_infer_overlay(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    _register_projection_namespaces(root)
    src = _make_dataset(root, "densegen_source", ["ACGT", "TGCA"])
    dest = _make_dataset(root, "anchor_dest", ["ACGT", "TGCA", "GGGG"])

    src_rows = src.head(10, columns=["id", "sequence"]).to_dict(orient="records")
    src_by_sequence = {str(row["sequence"]): str(row["id"]) for row in src_rows}
    dest_rows = dest.head(10, columns=["id", "sequence"]).to_dict(orient="records")
    dest_by_sequence = {str(row["sequence"]): str(row["id"]) for row in dest_rows}

    src.write_overlay(
        "densegen",
        pa.table(
            {
                "id": [src_by_sequence["ACGT"], src_by_sequence["TGCA"]],
                "densegen__plan": ["ethanol_f", "ciprofloxacin_b"],
                "densegen__required_regulators": [["cpxR"], ["lexA"]],
            }
        ),
        key="id",
        overwrite=True,
    )
    dest.write_overlay(
        "infer",
        pa.table(
            {
                "id": [
                    dest_by_sequence["ACGT"],
                    dest_by_sequence["TGCA"],
                    dest_by_sequence["GGGG"],
                ],
                "infer__score": [0.1, 0.2, 0.3],
            }
        ),
        key="id",
        overwrite=True,
    )

    preview = project_namespace_overlay(
        root=root,
        src_dataset_name=src.name,
        dest_dataset_name=dest.name,
        namespace="densegen",
        allow_missing=True,
    )

    rows = dest.head(
        10,
        columns=["sequence", "densegen__plan", "densegen__required_regulators", "infer__score"],
    ).to_dict(orient="records")
    by_sequence = {str(row["sequence"]): row for row in rows}

    assert preview.matched_rows == 2
    assert preview.missing_rows == 1
    assert by_sequence["ACGT"]["densegen__plan"] == "ethanol_f"
    assert by_sequence["ACGT"]["densegen__required_regulators"] == ["cpxR"]
    assert by_sequence["TGCA"]["densegen__plan"] == "ciprofloxacin_b"
    assert by_sequence["TGCA"]["infer__score"] == 0.2
    assert by_sequence["GGGG"]["infer__score"] == 0.3
    assert pd.isna(by_sequence["GGGG"]["densegen__plan"])


def test_project_namespace_overlay_supports_alternate_destination_join(tmp_path: Path) -> None:
    root = tmp_path / "datasets"
    _register_projection_namespaces(root)
    src = _make_dataset(root, "densegen_source", ["ACGT", "TGCA"])
    dest = _make_dataset(root, "construct_dest", ["AAAA", "CCCC", "TTTT"])

    src_rows = src.head(10, columns=["id", "sequence"]).to_dict(orient="records")
    src_by_sequence = {str(row["sequence"]): str(row["id"]) for row in src_rows}
    dest_rows = dest.head(10, columns=["id", "sequence"]).to_dict(orient="records")
    dest_by_sequence = {str(row["sequence"]): str(row["id"]) for row in dest_rows}

    src.write_overlay(
        "densegen",
        pa.table(
            {
                "id": [src_by_sequence["ACGT"], src_by_sequence["TGCA"]],
                "densegen__plan": ["ethanol_ciprofloxacin_f", "background_only_c"],
                "densegen__required_regulators": [["cpxR", "lexA"], []],
            }
        ),
        key="id",
        overwrite=True,
    )
    dest.write_overlay(
        "construct",
        pa.table(
            {
                "id": [
                    dest_by_sequence["AAAA"],
                    dest_by_sequence["CCCC"],
                    dest_by_sequence["TTTT"],
                ],
                "construct__anchor_id": [
                    src_by_sequence["ACGT"],
                    src_by_sequence["TGCA"],
                    "missing-anchor",
                ],
            }
        ),
        key="id",
        overwrite=True,
    )
    dest.write_overlay(
        "infer",
        pa.table(
            {
                "id": [
                    dest_by_sequence["AAAA"],
                    dest_by_sequence["CCCC"],
                    dest_by_sequence["TTTT"],
                ],
                "infer__score": [0.7, 0.8, 0.9],
            }
        ),
        key="id",
        overwrite=True,
    )

    preview = project_namespace_overlay(
        root=root,
        src_dataset_name=src.name,
        dest_dataset_name=dest.name,
        namespace="densegen",
        dest_join="construct__anchor_id",
        allow_missing=True,
    )

    rows = dest.head(
        10,
        columns=["sequence", "construct__anchor_id", "densegen__plan", "infer__score"],
    ).to_dict(orient="records")
    by_sequence = {str(row["sequence"]): row for row in rows}

    assert preview.matched_rows == 2
    assert preview.missing_rows == 1
    assert by_sequence["AAAA"]["densegen__plan"] == "ethanol_ciprofloxacin_f"
    assert by_sequence["AAAA"]["infer__score"] == 0.7
    assert by_sequence["CCCC"]["densegen__plan"] == "background_only_c"
    assert by_sequence["TTTT"]["infer__score"] == 0.9
    assert pd.isna(by_sequence["TTTT"]["densegen__plan"])
