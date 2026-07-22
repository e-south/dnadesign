"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/batch0/candidate_metadata.py

DenseGen metadata contract for the stress OPAL candidate universe.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from dnadesign.baserender import DENSEGEN_TFBS_REQUIRED_KEYS

DENSEGEN_SOURCE_CLASS_COLUMN = "opal_candidate__source_class"
DENSEGEN_SOURCE_CLASS_VALUE = "densegen"
DENSEGEN_KEY_COLUMNS: tuple[str, ...] = (
    "densegen__plan",
    "densegen__run_id",
    "densegen__sampling_library_hash",
)
DENSEGEN_TFBS_METADATA_COLUMNS: tuple[str, ...] = (
    "densegen__used_tfbs_detail",
    "densegen__required_regulators",
)
DENSEGEN_MATERIALIZATION_COLUMNS: tuple[str, ...] = (
    *DENSEGEN_KEY_COLUMNS,
    *DENSEGEN_TFBS_METADATA_COLUMNS,
)


def validate_candidate_densegen_metadata(records_path: str | Path) -> dict[str, int]:
    """Require renderable TFBS metadata for every DenseGen-backed candidate."""

    path = Path(records_path)
    schema = pq.ParquetFile(path).schema_arrow
    required = ("id", DENSEGEN_SOURCE_CLASS_COLUMN, *DENSEGEN_MATERIALIZATION_COLUMNS)
    missing = [column for column in required if column not in schema.names]
    if missing:
        raise ValueError(f"candidate feature table missing required DenseGen metadata columns: {missing}")
    _validate_tfbs_schema(schema)

    table = pq.read_table(path, columns=list(required))
    key_frame = table.select(DENSEGEN_KEY_COLUMNS).to_pandas()
    key_present = pd.DataFrame({column: ~_blank_text_mask(key_frame[column]) for column in DENSEGEN_KEY_COLUMNS})
    any_key = key_present.any(axis=1).to_numpy(dtype=bool)
    all_keys = key_present.all(axis=1).to_numpy(dtype=bool)
    partial_keys = any_key & ~all_keys
    if partial_keys.any():
        raise ValueError(
            "candidate feature table has partially populated DenseGen identity metadata "
            f"(sample_ids={_sample_ids(table, partial_keys)})"
        )

    source_class = table[DENSEGEN_SOURCE_CLASS_COLUMN].to_pandas().astype("string").str.strip()
    blank_source_class = source_class.isna() | source_class.eq("")
    if blank_source_class.any():
        raise ValueError(
            "candidate feature table has null/blank source-class metadata "
            f"(sample_ids={_sample_ids(table, blank_source_class.to_numpy(dtype=bool))})"
        )
    densegen_source = source_class.eq(DENSEGEN_SOURCE_CLASS_VALUE).to_numpy(dtype=bool)
    missing_identity = densegen_source & ~all_keys
    if missing_identity.any():
        raise ValueError(
            "DenseGen source rows require complete DenseGen identity metadata "
            f"(sample_ids={_sample_ids(table, missing_identity)})"
        )
    unowned_identity = ~densegen_source & all_keys
    if unowned_identity.any():
        raise ValueError(
            "non-DenseGen source rows must not carry DenseGen identity metadata "
            f"(sample_ids={_sample_ids(table, unowned_identity)})"
        )

    detail = table["densegen__used_tfbs_detail"]
    detail_lengths = pc.fill_null(pc.list_value_length(detail), 0).to_numpy(zero_copy_only=False)
    detail_present = (~pc.is_null(detail).to_numpy(zero_copy_only=False)) & (detail_lengths > 0)
    regulators = table["densegen__required_regulators"]
    regulators_present = ~pc.is_null(regulators).to_numpy(zero_copy_only=False)

    for column, present in (
        ("densegen__used_tfbs_detail", detail_present),
        ("densegen__required_regulators", regulators_present),
    ):
        bad = densegen_source & ~present
        if bad.any():
            raise ValueError(f"DenseGen-backed rows require non-null {column} (sample_ids={_sample_ids(table, bad)})")

    unexpected = ~densegen_source & (detail_present | regulators_present)
    if unexpected.any():
        raise ValueError(
            "non-DenseGen candidate rows must not carry unowned DenseGen TFBS metadata "
            f"(sample_ids={_sample_ids(table, unexpected)})"
        )

    densegen_rows = int(densegen_source.sum())
    return {
        "densegen_metadata_row_count": densegen_rows,
        "densegen_metadata_exempt_row_count": int(len(all_keys) - densegen_rows),
    }


def _validate_tfbs_schema(schema: pa.Schema) -> None:
    detail_type = schema.field("densegen__used_tfbs_detail").type
    if not (pa.types.is_list(detail_type) or pa.types.is_large_list(detail_type)):
        raise ValueError("densegen__used_tfbs_detail must use a list<struct> parquet schema")
    item_type = detail_type.value_type
    if not pa.types.is_struct(item_type):
        raise ValueError("densegen__used_tfbs_detail must use a list<struct> parquet schema")
    missing_fields = [field for field in DENSEGEN_TFBS_REQUIRED_KEYS if field not in item_type.names]
    if missing_fields:
        raise ValueError(
            "densegen__used_tfbs_detail is not compatible with the BaseRender densegen_tfbs adapter; "
            f"missing fields={missing_fields}"
        )

    regulators_type = schema.field("densegen__required_regulators").type
    if not (pa.types.is_list(regulators_type) or pa.types.is_large_list(regulators_type)):
        raise ValueError("densegen__required_regulators must use a list<string> parquet schema")
    if not (pa.types.is_string(regulators_type.value_type) or pa.types.is_large_string(regulators_type.value_type)):
        raise ValueError("densegen__required_regulators must use a list<string> parquet schema")


def _blank_text_mask(series: pd.Series) -> pd.Series:
    text = series.astype("string").str.strip()
    return text.isna() | text.eq("")


def _sample_ids(table: pa.Table, mask: np.ndarray, *, limit: int = 5) -> list[str]:
    indices = np.flatnonzero(mask)[:limit]
    return [str(table["id"][int(index)].as_py()) for index in indices]


__all__ = [
    "DENSEGEN_KEY_COLUMNS",
    "DENSEGEN_MATERIALIZATION_COLUMNS",
    "DENSEGEN_SOURCE_CLASS_COLUMN",
    "DENSEGEN_SOURCE_CLASS_VALUE",
    "DENSEGEN_TFBS_METADATA_COLUMNS",
    "validate_candidate_densegen_metadata",
]
