"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/analysis/notebook_components/baserender_records.py

Notebook component builders for BaseRender records OPAL analysis notebook components.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

from .baserender import NO_RENDERABLE_RECORDS_LABEL
from .baserender_record_sources import (
    annotation_column,
    annotation_count_expr,
    compact_record_id,
    contract_valid_filters,
    id_column,
    join_metadata_ids,
    join_metadata_rows,
    label_ids_for_round,
    metadata_records_path,
    metadata_source_columns,
    normalise_record_ids,
    record_source_columns,
    require_unique_record_ids,
)
from .baserender_record_sources import (
    source_columns as contract_source_columns,
)
from .baserender_record_validation import public_adapter_valid_record_ids


def build_notebook_baserender_record_options(
    records_path: str | Path,
    contract: Mapping[str, Any],
    *,
    labels_df: Any | None = None,
    round_value: Any | None = None,
    record_ids: Iterable[Any] | None = None,
    limit: int = 500,
    require_all_record_ids: bool = True,
) -> list[str]:
    """Return renderable record ids for the active BaseRender contract."""

    if not bool(contract.get("available")):
        return [NO_RENDERABLE_RECORDS_LABEL]

    import polars as pl

    source_columns = record_source_columns(contract)
    if not source_columns:
        return [NO_RENDERABLE_RECORDS_LABEL]
    identifier = id_column(contract)
    if identifier not in source_columns:
        source_columns.append(identifier)
    scan = pl.scan_parquet(str(records_path)).select(source_columns)
    schema = scan.collect_schema()
    for expr in contract_valid_filters(pl, contract, schema):
        scan = scan.filter(expr)
    metadata_path = metadata_records_path(contract)
    if metadata_path is not None:
        scan = join_metadata_ids(pl, scan, metadata_path, id_column_name=identifier)
    selected_ids = normalise_record_ids(record_ids)
    if record_ids is not None:
        if not selected_ids:
            return [NO_RENDERABLE_RECORDS_LABEL]
        scan = scan.filter(pl.col(identifier).cast(pl.Utf8).is_in(selected_ids))
    label_ids = label_ids_for_round(labels_df, round_value=round_value)
    if label_ids:
        scan = scan.filter(pl.col(identifier).cast(pl.Utf8).is_in(label_ids))
    records_scan = (
        scan.select(pl.col(identifier).cast(pl.Utf8).alias("__record_id")).drop_nulls().unique(maintain_order=True)
    )
    if record_ids is None:
        records_scan = records_scan.limit(max(1, int(limit)))
    records = records_scan.collect()
    options = records.get_column("__record_id").to_list() if "__record_id" in records.columns else []
    renderable = public_adapter_valid_record_ids(
        records_path,
        contract,
        record_ids=[str(item) for item in options if str(item).strip()],
    )
    available = set(renderable)
    if record_ids is not None:
        unavailable = [record_id for record_id in selected_ids if record_id not in available]
        if unavailable and require_all_record_ids:
            raise ValueError(
                "BaseRender candidate evidence is incomplete; contract-invalid or missing record ids: "
                f"{unavailable[:10]}."
            )
        ordered = [record_id for record_id in selected_ids if record_id in available]
        return ordered or [NO_RENDERABLE_RECORDS_LABEL]
    return renderable or [NO_RENDERABLE_RECORDS_LABEL]


def build_notebook_baserender_record_choices(record_ids: Iterable[Any]) -> list[dict[str, str]]:
    """Return stable dropdown labels for renderable record ids."""

    values = normalise_record_ids(record_ids)
    if not values:
        return [{"label": NO_RENDERABLE_RECORDS_LABEL, "record_id": NO_RENDERABLE_RECORDS_LABEL}]
    return [
        {
            "label": f"{index}. {compact_record_id(record_id)}",
            "record_id": record_id,
        }
        for index, record_id in enumerate(values, start=1)
    ]


def has_notebook_baserender_record_options(record_ids: Iterable[Any]) -> bool:
    """Return whether a record-option collection contains a renderable identity."""

    values = normalise_record_ids(record_ids)
    return bool(values) and NO_RENDERABLE_RECORDS_LABEL not in values


def build_notebook_baserender_record_annotation_counts(
    records_path: str | Path,
    contract: Mapping[str, Any],
    *,
    record_ids: Iterable[Any] | None = None,
) -> dict[str, int]:
    """Return per-record annotation counts for campaign-candidate lookup context."""

    if not bool(contract.get("available")):
        return {}

    import polars as pl

    identifier = id_column(contract)
    annotation = annotation_column(contract)
    if annotation is None:
        return {}
    source_columns = (
        metadata_source_columns(contract)
        if metadata_records_path(contract) is not None
        else contract_source_columns(contract)
    )
    if identifier not in source_columns:
        source_columns.append(identifier)
    if annotation not in source_columns:
        source_columns.append(annotation)
    metadata_path = metadata_records_path(contract)
    source_path = metadata_path or str(records_path)
    scan = pl.scan_parquet(source_path).select(source_columns)
    schema = scan.collect_schema()
    if metadata_path is None:
        for expr in contract_valid_filters(pl, contract, schema):
            scan = scan.filter(expr)
    else:
        scan = scan.filter(pl.col(identifier).is_not_null())
    selected_ids = normalise_record_ids(record_ids)
    if record_ids is not None:
        if not selected_ids:
            return {}
        scan = scan.filter(pl.col(identifier).cast(pl.Utf8).is_in(selected_ids))
    count_expr = annotation_count_expr(pl, annotation, schema)
    counts_df = (
        scan.select(pl.col(identifier).cast(pl.Utf8).alias("__record_id"), count_expr)
        .drop_nulls(subset=["__record_id"])
        .collect()
    )
    return {
        str(row["__record_id"]): max(0, int(row["__annotation_count"] or 0))
        for row in counts_df.to_dicts()
        if str(row["__record_id"]).strip()
    }


def build_notebook_baserender_record_choices_with_counts(
    record_ids: Iterable[Any],
    annotation_counts: Mapping[str, int],
    *,
    annotation_label: str = "annotations",
    display_aliases: Mapping[str, str] | None = None,
    candidate_evidence: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """Return searchable candidate labels with annotation and campaign evidence."""

    rows = build_notebook_baserender_record_choices(record_ids)
    if not rows or rows[0]["record_id"] == NO_RENDERABLE_RECORDS_LABEL:
        return rows
    label = str(annotation_label or "annotations").strip() or "annotations"
    out: list[dict[str, str]] = []
    for row in rows:
        record_id = str(row["record_id"])
        count = max(0, int(annotation_counts.get(record_id, 0)))
        alias = str((display_aliases or {}).get(record_id) or compact_record_id(record_id)).strip()
        identity = alias if alias == compact_record_id(record_id) else f"{alias} · {compact_record_id(record_id)}"
        evidence_label = _candidate_evidence_label((candidate_evidence or {}).get(record_id))
        display = (
            f"{evidence_label} · {identity} · {count} {label}" if evidence_label else f"{identity} · {count} {label}"
        )
        out.append(
            {
                "label": display,
                "record_id": record_id,
            }
        )
    label_counts: dict[str, int] = {}
    for row in out:
        row_label = str(row["label"])
        label_counts[row_label] = label_counts.get(row_label, 0) + 1
    for row in out:
        row_label = str(row["label"])
        if label_counts[row_label] > 1:
            row["label"] = f"{row_label} · ID {row['record_id']}"
    if len({row["label"] for row in out}) != len(out):
        raise ValueError("BaseRender candidate labels must be unique after identity disambiguation.")
    return out


def _candidate_evidence_label(evidence: Mapping[str, Any] | None) -> str:
    if not evidence:
        return ""
    active_rank = evidence.get("active_view_rank")
    memberships = [row for row in evidence.get("selection_memberships") or () if isinstance(row, Mapping)]
    observed_rounds = [int(value) for value in evidence.get("observed_rounds") or ()]
    parts: list[str] = []
    if active_rank is not None:
        parts.append(f"Selected rank {int(active_rank)}")
    elif memberships:
        views = ", ".join(sorted({str(row.get("selection_view_id") or "").strip() for row in memberships}))
        parts.append(f"Selected in {views}")
    if observed_rounds:
        parts.append("Observed " + ", ".join(f"R{value}" for value in sorted(set(observed_rounds))))
    return " + ".join(parts)


def select_notebook_baserender_default_record_id(
    record_ids: Iterable[Any],
    annotation_counts: Mapping[str, int] | None = None,
) -> str:
    """Choose the first annotated candidate, falling back to the first candidate."""

    values = normalise_record_ids(record_ids)
    if not values:
        return NO_RENDERABLE_RECORDS_LABEL
    counts = dict(annotation_counts or {})
    for record_id in values:
        if int(counts.get(record_id, 0)) > 0:
            return record_id
    return values[0]


def load_notebook_baserender_record_row(
    records_path: str | Path,
    record_id: str,
    contract: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Load one contract-valid record row for notebook BaseRender display."""

    if not bool(contract.get("available")) or str(record_id) == NO_RENDERABLE_RECORDS_LABEL:
        return None

    import polars as pl

    identifier = id_column(contract)
    source_columns = record_source_columns(contract)
    if identifier not in source_columns:
        source_columns.append(identifier)
    scan = pl.scan_parquet(str(records_path)).select(source_columns)
    scan = scan.filter(pl.col(identifier).cast(pl.Utf8) == str(record_id))
    require_unique_record_ids(pl, scan, id_column_name=identifier)
    schema = scan.collect_schema()
    for expr in contract_valid_filters(pl, contract, schema):
        scan = scan.filter(expr)
    metadata_path = metadata_records_path(contract)
    if metadata_path is not None:
        scan = join_metadata_rows(pl, scan, metadata_path, id_column_name=identifier, contract=contract)
    require_unique_record_ids(pl, scan, id_column_name=identifier)
    row_df = scan.collect()
    if row_df.is_empty():
        return None
    return row_df.to_dicts()[0]


def build_notebook_baserender_label_rows(
    labels_df: Any | None,
    *,
    record_id: str,
    round_value: Any | None = None,
    limit: int = 5,
) -> list[dict[str, Any]]:
    """Return compact observed-label rows for the selected rendered record."""

    if labels_df is None or str(record_id) == NO_RENDERABLE_RECORDS_LABEL:
        return []
    if not hasattr(labels_df, "columns") or not hasattr(labels_df, "is_empty"):
        return []
    if "id" not in labels_df.columns or labels_df.is_empty():
        return []
    try:
        import polars as pl

        filtered = labels_df.filter(pl.col("id").cast(pl.Utf8) == str(record_id))
        if round_value is not None and "observed_round" in filtered.columns:
            filtered = filtered.filter(pl.col("observed_round") == int(round_value))
        label_columns = [
            column
            for column in (
                "observed_round",
                "id",
                "y_space",
                "y_obs",
                "src",
                "label_src",
                "note",
                "ts",
            )
            if column in filtered.columns
        ]
        if not label_columns or filtered.is_empty():
            return []
        return filtered.select(label_columns).head(max(1, int(limit))).to_dicts()
    except Exception:
        return []
