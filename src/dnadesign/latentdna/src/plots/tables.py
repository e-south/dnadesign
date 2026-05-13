"""Table-loading and schema contracts for plot rendering."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from ..contracts.errors import ContractViolationError, MissingArtifactError
from ..contracts.plot import ResolvedPlotSpec
from ..workspaces.loader import WorkspaceContext


def table_artifact_path(context: WorkspaceContext, spec: ResolvedPlotSpec) -> tuple[str, str, Path]:
    """Resolve the single table-backed artifact declared by a plot spec."""

    candidates = [
        (
            "scalar_table",
            spec.scalar_id,
            context.output_root / "scalars" / spec.scalar_id / "table.parquet" if spec.scalar_id is not None else None,
        ),
        (
            "distance_set",
            spec.distance_id,
            context.output_root / "distances" / spec.distance_id / "table.parquet"
            if spec.distance_id is not None
            else None,
        ),
        (
            "enrichment_set",
            spec.enrichment_id,
            context.output_root / "enrichments" / spec.enrichment_id / "table.parquet"
            if spec.enrichment_id is not None
            else None,
        ),
        (
            "agreement_set",
            spec.agreement_id,
            context.output_root / "agreements" / spec.agreement_id / "table.parquet"
            if spec.agreement_id is not None
            else None,
        ),
    ]
    selected = [(kind, artifact_id, path) for kind, artifact_id, path in candidates if artifact_id is not None]
    if len(selected) != 1:
        raise ContractViolationError(
            "plot rendering requires exactly one table-backed artifact input for this plot kind"
        )
    artifact_kind, artifact_id, artifact_path = selected[0]
    assert artifact_path is not None
    if not artifact_path.exists():
        raise MissingArtifactError(f"{artifact_kind} artifact is missing for plot rendering: {artifact_id}")
    return artifact_kind, str(artifact_id), artifact_path


def require_table_columns(
    table: pa.Table,
    columns: Iterable[str | None],
    *,
    artifact_label: str,
) -> None:
    """Fail fast when a table artifact lacks required semantic columns."""

    required = [str(column) for column in columns if column is not None and str(column)]
    if not required:
        return
    available = set(table.schema.names)
    missing = [column for column in required if column not in available]
    if missing:
        raise ContractViolationError(f"{artifact_label} is missing required column(s): {missing}")


def read_table_rows(
    table_path: Path,
    *,
    required_columns: Iterable[str | None] = (),
    artifact_label: str | None = None,
) -> list[dict[str, object]]:
    """Read a Parquet table and validate required schema before row conversion."""

    table = pq.read_table(table_path)
    require_table_columns(table, required_columns, artifact_label=artifact_label or table_path.as_posix())
    return table.to_pylist()


def numeric_table_columns(table: pa.Table) -> list[str]:
    """Return numeric schema columns in table order."""

    numeric: list[str] = []
    for field in table.schema:
        if pa.types.is_integer(field.type) or pa.types.is_floating(field.type):
            numeric.append(field.name)
    return numeric


def secondary_numeric_column(table: pa.Table, *, primary: str) -> str:
    """Return the first numeric column that is not the selected primary axis."""

    for candidate in numeric_table_columns(table):
        if candidate != primary:
            return candidate
    raise ContractViolationError(
        f"plot rendering requires at least two numeric columns when {primary!r} is used as the first axis"
    )


def require_row_columns(
    rows: Sequence[Mapping[str, object]],
    columns: Iterable[str | None],
    *,
    context: str,
) -> None:
    """Validate required columns for row-dict inputs used by low-level render helpers."""

    required = [str(column) for column in columns if column is not None and str(column)]
    if not required or not rows:
        return
    missing_examples: dict[str, int] = {}
    for row_index, row in enumerate(rows):
        for column in required:
            if column not in row and column not in missing_examples:
                missing_examples[column] = row_index
    if missing_examples:
        details = ", ".join(
            f"{column!r} first missing at row {row_index}" for column, row_index in missing_examples.items()
        )
        raise ContractViolationError(f"{context} is missing required column(s): {details}")


def require_unique_grid_cell(
    seen_cells: dict[tuple[str, str], int],
    *,
    row_key: str,
    column_key: str,
    row_number: int,
    context: str,
) -> None:
    """Fail fast on duplicate semantic heatmap cells instead of overwriting them."""

    cell_key = (row_key, column_key)
    if cell_key not in seen_cells:
        seen_cells[cell_key] = row_number
        return
    first_row_number = seen_cells[cell_key]
    raise ContractViolationError(
        f"{context} contains duplicate heatmap cell ({row_key!r}, {column_key!r}) "
        f"at rows {first_row_number} and {row_number}; aggregate upstream before rendering"
    )
