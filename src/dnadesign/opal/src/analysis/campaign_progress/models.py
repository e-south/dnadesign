"""Data contracts for the checked-in campaign progress notebook."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

OPAL_RECORD_IDENTITY_COLUMNS = ("id", "bio_type", "sequence", "alphabet")


@dataclass(frozen=True)
class RecordsContractReport:
    row_count: int
    column_count: int
    required_columns: tuple[str, ...]
    missing_required_columns: tuple[str, ...]
    x_column: str | None
    label_hist_column: str | None
    x_values_loaded: bool = True

    @property
    def ready(self) -> bool:
        return not self.missing_required_columns


@dataclass(frozen=True)
class OptionalTableRead:
    name: str
    path: Path | None
    df: pl.DataFrame
    status: str
    message: str

    @property
    def available(self) -> bool:
        return self.status == "available"
