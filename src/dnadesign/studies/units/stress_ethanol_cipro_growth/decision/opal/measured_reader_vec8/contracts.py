"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/measured_reader_vec8/contracts.py

Defines data contracts for measured Reader vec8 staging.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


class MeasuredReaderVec8Error(RuntimeError):
    """Raised when measured reader vec8 staging inputs violate the contract."""


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class ReaderVec8Source:
    experiment_id: str
    config_path: Path
    table_path: Path
    row_count: int
    records_path: Path
    plot_files_by_record_id: dict[str, tuple[str, ...]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "config_path": str(self.config_path),
            "table_path": str(self.table_path),
            "row_count": int(self.row_count),
            "records_path": str(self.records_path),
            "plot_files_by_record_id": {
                record_id: list(files) for record_id, files in sorted(self.plot_files_by_record_id.items())
            },
        }


@dataclass(frozen=True)
class MeasuredReaderVec8Staging:
    audit_frame: pd.DataFrame
    measured_frame: pd.DataFrame
    duplicate_frame: pd.DataFrame
    source_records: tuple[ReaderVec8Source, ...]

    @property
    def summary(self) -> dict[str, int]:
        return {
            "reader_vec8_rows": int(len(self.audit_frame)),
            "measured_candidate_rows": int(len(self.measured_frame)),
            "duplicate_candidate_rows": int(len(self.duplicate_frame)),
            "reader_sources": int(len(self.source_records)),
        }


@dataclass(frozen=True)
class MeasuredReaderVec8WriteResult:
    staging: MeasuredReaderVec8Staging
    audit_csv: Path
    manifest_json: Path
    campaign_inputs: dict[str, Path]
    campaign_evidence_manifests: dict[str, Path]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "stress_ethanol_cipro_growth.measured_reader_vec8.write.v1",
            "summary": self.staging.summary,
            "audit_csv": str(self.audit_csv),
            "manifest_json": str(self.manifest_json),
            "campaign_inputs": {slug: str(path) for slug, path in sorted(self.campaign_inputs.items())},
            "campaign_evidence_manifests": {
                slug: str(path) for slug, path in sorted(self.campaign_evidence_manifests.items())
            },
        }
