"""Runtime contract for OPAL label ingest memory posture."""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pandas as pd

INGEST_RUNTIME_SCHEMA_VERSION = "opal.ingest_runtime.v1"


@dataclass(frozen=True)
class IngestRuntimeContract:
    """Machine-readable ingest runtime posture for CLI and notebook surfaces."""

    mode: str
    label_source_kind: str
    records_load_mode: str
    full_records_loaded: bool
    fixed_candidate_universe: bool
    records_path: Path
    records_row_count: int
    identity_columns: tuple[str, ...]
    records_columns_loaded: tuple[str, ...]
    candidate_x_column: str
    candidate_x_column_loaded: bool
    candidate_index_rows: int
    records_frame_bytes: int
    estimated_memory_bytes: int
    peak_rss_bytes: int | None
    unknown_sequences_policy: str
    write_scope: str
    input_rows: int | None = None
    transformed_label_rows: int | None = None
    unknown_count_initial: int | None = None
    unknown_count_after_policy: int | None = None
    labels_after_unknown_policy: int | None = None
    schema_version: str = INGEST_RUNTIME_SCHEMA_VERSION

    def with_preview_counts(
        self,
        *,
        input_rows: int,
        transformed_label_rows: int,
        unknown_count_initial: int,
    ) -> IngestRuntimeContract:
        return replace(
            self,
            input_rows=int(input_rows),
            transformed_label_rows=int(transformed_label_rows),
            unknown_count_initial=int(unknown_count_initial),
        )

    def with_policy_counts(
        self,
        *,
        unknown_count_after_policy: int,
        labels_after_unknown_policy: int,
    ) -> IngestRuntimeContract:
        return replace(
            self,
            unknown_count_after_policy=int(unknown_count_after_policy),
            labels_after_unknown_policy=int(labels_after_unknown_policy),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mode": self.mode,
            "label_source_kind": self.label_source_kind,
            "records_load_mode": self.records_load_mode,
            "full_records_loaded": self.full_records_loaded,
            "fixed_candidate_universe": self.fixed_candidate_universe,
            "records_path": str(self.records_path),
            "records_row_count": int(self.records_row_count),
            "identity_columns": list(self.identity_columns),
            "records_columns_loaded": list(self.records_columns_loaded),
            "candidate_x_column": self.candidate_x_column,
            "candidate_x_column_loaded": self.candidate_x_column_loaded,
            "candidate_index_rows": int(self.candidate_index_rows),
            "records_frame_bytes": int(self.records_frame_bytes),
            "estimated_memory_bytes": int(self.estimated_memory_bytes),
            "peak_rss_bytes": self.peak_rss_bytes,
            "unknown_sequences_policy": self.unknown_sequences_policy,
            "write_scope": self.write_scope,
            "input_rows": self.input_rows,
            "transformed_label_rows": self.transformed_label_rows,
            "unknown_count_initial": self.unknown_count_initial,
            "unknown_count_after_policy": self.unknown_count_after_policy,
            "labels_after_unknown_policy": self.labels_after_unknown_policy,
        }


def build_ingest_runtime_contract(
    *,
    frame: pd.DataFrame,
    records_path: Path,
    records_row_count: int,
    candidate_x_column: str,
    label_source_kind: str,
    fixed_candidate_universe: bool,
    unknown_sequences_policy: str,
) -> IngestRuntimeContract:
    columns = tuple(map(str, frame.columns))
    identity_columns = tuple(column for column in ("id", "sequence") if column in frame.columns)
    records_frame_bytes = int(frame.memory_usage(deep=True).sum())
    records_load_mode = "identity_frame" if fixed_candidate_universe else "full_records"
    return IngestRuntimeContract(
        mode="identity_index" if fixed_candidate_universe else "full_records",
        label_source_kind=str(label_source_kind),
        records_load_mode=records_load_mode,
        full_records_loaded=not fixed_candidate_universe,
        fixed_candidate_universe=bool(fixed_candidate_universe),
        records_path=Path(records_path),
        records_row_count=int(records_row_count),
        identity_columns=identity_columns,
        records_columns_loaded=columns,
        candidate_x_column=str(candidate_x_column),
        candidate_x_column_loaded=str(candidate_x_column) in columns,
        candidate_index_rows=int(len(frame)),
        records_frame_bytes=records_frame_bytes,
        estimated_memory_bytes=records_frame_bytes,
        peak_rss_bytes=current_peak_rss_bytes(),
        unknown_sequences_policy=str(unknown_sequences_policy),
        write_scope="label_sidecar" if fixed_candidate_universe else "records",
    )


def current_peak_rss_bytes() -> int | None:
    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    if peak <= 0:
        return None
    if sys.platform == "darwin":
        return peak
    return peak * 1024
