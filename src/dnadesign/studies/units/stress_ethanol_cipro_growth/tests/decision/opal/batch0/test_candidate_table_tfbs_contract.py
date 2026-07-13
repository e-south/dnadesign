"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/batch0/test_candidate_table_tfbs_contract.py

DenseGen TFBS metadata contracts for the stress OPAL candidate table.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from dnadesign import baserender
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.batch0.candidate_table import (
    validate_candidate_feature_table,
)

X_COLUMN = "latentdna__evo2_7b__context_anchor_mean_bidir_concat"
DETAIL_COLUMN = "densegen__used_tfbs_detail"
REGULATORS_COLUMN = "densegen__required_regulators"
DETAIL_TYPE = pa.list_(
    pa.struct(
        [
            pa.field("part_kind", pa.string()),
            pa.field("regulator", pa.string()),
            pa.field("sequence", pa.string()),
            pa.field("orientation", pa.string()),
            pa.field("offset", pa.int64()),
        ]
    )
)


def _detail(regulator: str = "BaeR") -> list[dict[str, object]]:
    return [
        {
            "part_kind": "tfbs",
            "regulator": regulator,
            "sequence": "ACGT",
            "orientation": "fwd",
            "offset": 0,
        }
    ]


def _write_candidate_table(
    path: Path,
    *,
    details: list[object] | None,
    regulators: list[object] | None,
    densegen_keys: bool = True,
    source_class: str | None = None,
) -> None:
    resolved_source_class = source_class or ("densegen" if densegen_keys else "construct_derived")
    columns: dict[str, pa.Array] = {
        "id": pa.array(["candidate-a"]),
        "bio_type": pa.array(["dna"]),
        "sequence": pa.array(["ACGT"]),
        "alphabet": pa.array(["dna_4"]),
        "opal_candidate__role": pa.array(["opal_candidate_feature_table"]),
        "opal_candidate__x_source_view_id": pa.array(["bidir"]),
        "opal_candidate__source_class": pa.array([resolved_source_class]),
        "opal_candidate__design_family": pa.array(["ethanol" if densegen_keys else "control"]),
        "opal_candidate__sfxi_ref__collection_id": pa.array([None], type=pa.string()),
        "densegen__plan": pa.array(["ethanol" if densegen_keys else None]),
        "densegen__run_id": pa.array(["run-a" if densegen_keys else None]),
        "densegen__sampling_library_hash": pa.array(["hash-a" if densegen_keys else None]),
    }
    if details is not None:
        columns[DETAIL_COLUMN] = pa.array(details, type=DETAIL_TYPE)
    if regulators is not None:
        columns[REGULATORS_COLUMN] = pa.array(regulators, type=pa.list_(pa.string()))
    columns[X_COLUMN] = pa.array([[0.1, 0.2]], type=pa.list_(pa.float32(), list_size=2))
    pq.write_table(pa.table(columns), path)


def test_candidate_table_requires_baserender_tfbs_columns(tmp_path: Path) -> None:
    records = tmp_path / "records.parquet"
    _write_candidate_table(records, details=None, regulators=None)

    with pytest.raises(ValueError, match="densegen__used_tfbs_detail"):
        validate_candidate_feature_table(records_path=records, x_column=X_COLUMN)


def test_candidate_table_rejects_null_tfbs_detail_for_densegen_rows(tmp_path: Path) -> None:
    records = tmp_path / "records.parquet"
    _write_candidate_table(records, details=[None], regulators=[["BaeR"]])

    with pytest.raises(ValueError, match="DenseGen-backed rows.*densegen__used_tfbs_detail"):
        validate_candidate_feature_table(records_path=records, x_column=X_COLUMN)


def test_candidate_table_allows_explicit_non_densegen_metadata_exemption(tmp_path: Path) -> None:
    records = tmp_path / "records.parquet"
    _write_candidate_table(records, details=[None], regulators=[None], densegen_keys=False)

    report = validate_candidate_feature_table(records_path=records, x_column=X_COLUMN)

    assert report["densegen_metadata_row_count"] == 0
    assert report["densegen_metadata_exempt_row_count"] == 1


def test_candidate_table_rejects_densegen_source_without_identity_metadata(tmp_path: Path) -> None:
    records = tmp_path / "records.parquet"
    _write_candidate_table(
        records,
        details=[None],
        regulators=[None],
        densegen_keys=False,
        source_class="densegen",
    )

    with pytest.raises(ValueError, match="DenseGen source rows require complete DenseGen identity metadata"):
        validate_candidate_feature_table(records_path=records, x_column=X_COLUMN)


def test_candidate_table_rejects_non_densegen_source_with_identity_metadata(tmp_path: Path) -> None:
    records = tmp_path / "records.parquet"
    _write_candidate_table(
        records,
        details=[_detail()],
        regulators=[["BaeR"]],
        densegen_keys=True,
        source_class="construct_derived",
    )

    with pytest.raises(ValueError, match="non-DenseGen source rows must not carry DenseGen identity metadata"):
        validate_candidate_feature_table(records_path=records, x_column=X_COLUMN)


def test_candidate_table_tfbs_metadata_satisfies_public_baserender_adapter(tmp_path: Path) -> None:
    records = tmp_path / "records.parquet"
    _write_candidate_table(records, details=[_detail()], regulators=[["BaeR"]])
    row = pq.read_table(records).to_pylist()[0]
    config = baserender.sequence_panel_config_for_adapter("densegen_tfbs")

    record = baserender.adapt_record(
        row,
        adapter_kind=config.adapter_kind,
        adapter_columns=config.adapter_columns,
        adapter_policies=config.adapter_policies,
    )

    assert record.id == "candidate-a"
    assert len(record.features) == 1


def test_sampling_contract_materializes_renderable_densegen_metadata() -> None:
    config_path = Path("src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/batch0/sampling.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    columns = config["candidate_feature_table"]["materialization"]["densegen_sidecar_columns"]
    assert DETAIL_COLUMN in columns
    assert REGULATORS_COLUMN in columns
