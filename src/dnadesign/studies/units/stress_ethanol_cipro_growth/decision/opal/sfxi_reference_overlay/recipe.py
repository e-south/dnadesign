"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/sfxi_reference_overlay/recipe.py

Build the stress-study SFXI overlay from neutral Reader four-state vectors.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pyarrow as pa

from dnadesign.opal.api.sfxi import (
    SFXIScoringConfig,
    score_vec8,
    to_sfxi_reference_overlay_records,
    validate_sfxi_reference_overlay_records,
)
from dnadesign.usr import Dataset, SchemaError

from .reader_records import default_selection_path, load_verified_reader_selection

DEFAULT_OUTPUT_DATASET = "usr_sfxi_pdual10_densegen_promoters"
DEFAULT_COLLECTION_ID = "reader_sfxi_pdual10_latest"
DEFAULT_CAMPAIGN_ID = "20260501_sfxi_promoter_setpoint_scatter"
DEFAULT_SETPOINT_NAME = "and"
DEFAULT_SETPOINT_VECTOR = (0.0, 0.0, 0.0, 1.0)
DEFAULT_METRIC_ID = "sfxi_v1/and/sfxi"
# This value is part of the already-published historical overlay. It is retained
# as immutable study evidence, not as the name of a current Reader contract.
HISTORICAL_METRIC_PROVENANCE = "reader.vec8.sfxi_setpoint_scatter+dnadesign.opal.api.sfxi"
DEFAULT_SCORE_REF = "dnadesign.opal.api.sfxi.score_vec8"
FIXED_SCORING_CONFIG = SFXIScoringConfig(
    setpoint_vector=DEFAULT_SETPOINT_VECTOR,
    scaling_percentile=95,
    scaling_min_n=5,
    scaling_eps=1.0e-8,
    logic_exponent_beta=1.0,
    intensity_exponent_gamma=1.0,
    intensity_log2_offset_delta=0.0,
)
VEC8_COLUMNS = ("v00", "v10", "v01", "v11", "y00_star", "y10_star", "y01_star", "y11_star")
REQUIRED_READER_COLUMNS = (
    "design_id",
    "sequence",
    "time_selected_h",
    "reference_design_id",
    "r_logic",
    "flat_logic",
    *VEC8_COLUMNS,
)


@dataclass(frozen=True, slots=True)
class OverlayPreview:
    """Validated in-memory overlay; construction never writes dataset state."""

    table: pa.Table
    source_ref: str
    record_digests: tuple[str, ...]
    dataset_name: str


def _normalized_sequence(value: object) -> str:
    sequence = "".join(str(value).split()).upper()
    if not sequence:
        raise SchemaError("SFXI reference sequences must be non-empty.")
    return sequence


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], *, context: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise SchemaError(f"{context} missing required columns: {missing}")


def _base_records(dataset: Dataset) -> pd.DataFrame:
    batches = list(dataset.scan(columns=["id", "sequence"], include_overlays=False))
    return pa.Table.from_batches(batches).to_pandas()


def _join_usr_ids(*, base: pd.DataFrame, reader: pd.DataFrame) -> list[str]:
    base = base.loc[:, ["id", "sequence"]].copy()
    base["sequence_norm"] = base["sequence"].map(_normalized_sequence)
    reader_norm = reader["sequence"].map(_normalized_sequence)
    if base["sequence_norm"].duplicated().any():
        raise SchemaError("USR base records contain duplicate normalized sequences.")
    if base["id"].astype(str).duplicated().any():
        raise SchemaError("USR base records contain duplicate USR ids.")
    if reader_norm.duplicated().any():
        raise SchemaError("Reader four-state vectors contain duplicate normalized sequences.")
    ids = reader_norm.map(base.set_index("sequence_norm")["id"])
    if ids.isna().any():
        missing = reader.loc[ids.isna(), "design_id"].astype(str).tolist()
        raise SchemaError(f"Reader four-state designs do not map to USR records by sequence: {missing[:5]}")
    return ids.astype(str).tolist()


def build_overlay_preview(
    *,
    usr_root: Path,
    dataset_name: str,
    reader_root: Path,
    selection_path: Path | None = None,
    collection_id: str = DEFAULT_COLLECTION_ID,
    campaign_id: str = DEFAULT_CAMPAIGN_ID,
) -> OverlayPreview:
    """Apply study-owned SFXI scoring to verified neutral Reader measurements."""

    verified = load_verified_reader_selection(
        reader_root=reader_root,
        selection_path=selection_path or default_selection_path(),
    )
    reader = verified.frame
    _require_columns(reader, REQUIRED_READER_COLUMNS, context="Reader four-state vector")
    if reader["design_id"].astype(str).duplicated().any():
        raise SchemaError("Reader four-state vectors contain duplicate design ids.")
    dataset = Dataset.open(usr_root.expanduser().resolve(), dataset_name)
    usr_ids = _join_usr_ids(base=_base_records(dataset), reader=reader)
    result = score_vec8(reader.loc[:, VEC8_COLUMNS].to_numpy(float), FIXED_SCORING_CONFIG)
    rows = to_sfxi_reference_overlay_records(
        result,
        metric_id=DEFAULT_METRIC_ID,
        metric_provenance=HISTORICAL_METRIC_PROVENANCE,
        source_ref=verified.source_ref,
        score_ref=DEFAULT_SCORE_REF,
        reference_instance_id=reader["design_id"].astype(str).tolist(),
        collection_id=collection_id,
        batch_id=reader["experiment_id"].astype(str).tolist(),
        campaign_id=campaign_id,
        reader_experiment_id=reader["experiment_id"].astype(str).tolist(),
        reader_experiment_date=reader["experiment_date"].astype(int).tolist(),
        setpoint_name=DEFAULT_SETPOINT_NAME,
        r_logic=reader["r_logic"].astype(float).tolist(),
        time_selected_h=reader["time_selected_h"].astype(float).tolist(),
        reference_design_id=reader["reference_design_id"].astype(str).tolist(),
        sequence_source_id=usr_ids,
        flat_logic=reader["flat_logic"].astype(bool).tolist(),
    )
    validate_sfxi_reference_overlay_records(
        rows,
        expected_setpoint_vector=DEFAULT_SETPOINT_VECTOR,
        metric_id=DEFAULT_METRIC_ID,
    )
    table = pa.Table.from_pylist([{"id": usr_id, **row} for usr_id, row in zip(usr_ids, rows, strict=True)])
    table = table.sort_by([("id", "ascending")])
    return OverlayPreview(
        table=table,
        source_ref=verified.source_ref,
        record_digests=verified.record_digests,
        dataset_name=dataset_name,
    )


def publish_overlay(*, usr_root: Path, preview: OverlayPreview) -> int:
    """Publish a validated preview through the public USR Dataset API."""

    dataset = Dataset.open(usr_root.expanduser().resolve(), preview.dataset_name)
    return dataset.create_overlay("sfxi_ref", preview.table, key="id", allow_missing=False)
