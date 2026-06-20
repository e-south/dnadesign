"""Mestre roster-table parsing for Eco1 conservation source caches."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.source_sequences.roster_cache.models import (
    RosterRow,
)

_NODE_ALIASES = ("Node", "node", "Nodo", "node_id")
_SUBTYPE_ALIASES = ("Retron subtype", "retron_subtype", "Subtype", "Type", "subtype")
_CLUSTER_ALIASES = ("Cluster/domain", "cluster_domain", "Cluster", "Domain", "domain")
_CLADE_ALIASES = ("RT clade", "rt_clade", "Clade", "clade")
_STATUS_ALIASES = ("source_cache_status", "cache_status", "Status", "status")
_EXCLUSION_REASON_ALIASES = ("exclusion_reason", "Exclusion reason", "exclude_reason", "reason")
_RECORD_STATUSES = {"included", "excluded"}


def load_roster_rows(path: Path, *, accession_field: str) -> list[RosterRow]:
    """Load a CSV/TSV/XLSX roster table into normalized rows."""

    raw_rows = _read_table_rows(path)
    if not raw_rows:
        raise ValueError(f"roster table has no rows: {path}")

    columns = _column_map(raw_rows[0])
    accession_column = _resolve_column(columns, (accession_field,))
    node_column = _resolve_column(columns, _NODE_ALIASES, required=False)
    subtype_column = _resolve_column(columns, _SUBTYPE_ALIASES)
    cluster_column = _resolve_column(columns, _CLUSTER_ALIASES)
    clade_column = _resolve_column(columns, _CLADE_ALIASES)
    status_column = _resolve_column(columns, _STATUS_ALIASES, required=False)
    exclusion_reason_column = _resolve_column(columns, _EXCLUSION_REASON_ALIASES, required=False)

    rows: list[RosterRow] = []
    for index, raw_row in enumerate(raw_rows, start=1):
        accession = _cell(raw_row, accession_column)
        if not accession:
            continue
        status = _status(raw_row, status_column)
        exclusion_reason = _cell(raw_row, exclusion_reason_column) if exclusion_reason_column else ""
        if status == "excluded" and not exclusion_reason:
            raise ValueError(f"excluded roster row {index} must include exclusion_reason")
        rows.append(
            RosterRow(
                row_index=index,
                node_id=_cell(raw_row, node_column) if node_column else f"row_{index}",
                accession=accession,
                retron_subtype=_cell(raw_row, subtype_column),
                cluster_domain=_cell(raw_row, cluster_column),
                rt_clade=_cell(raw_row, clade_column),
                status=status,
                exclusion_reason=exclusion_reason or None,
            )
        )
    if not rows:
        raise ValueError(f"roster table has no rows with accession field {accession_field!r}")
    return rows


def select_profile_rows(
    rows: Sequence[RosterRow],
    *,
    profile_id: str,
    source_group: Mapping[str, Any],
) -> list[RosterRow]:
    """Select roster rows for one conservation source profile."""

    selection_rule = _require_mapping(source_group.get("selection_rule"), "selection_rule")
    included_records = str(selection_rule.get("included_records", ""))
    if included_records == "all_mestre_s1_rt_records_after_filters":
        return list(rows)
    if included_records == "mestre_s1_retron_subtype_ii_a3_cluster_42_1_after_filters":
        expected_subtype = str(selection_rule.get("retron_subtype", ""))
        expected_cluster = str(selection_rule.get("cluster_domain", ""))
        expected_clade = str(selection_rule.get("parent_rt_clade", ""))
        selected = [
            row
            for row in rows
            if row.retron_subtype == expected_subtype
            and row.cluster_domain == expected_cluster
            and row.rt_clade == expected_clade
        ]
        if not selected:
            raise ValueError(f"profile {profile_id!r} selected zero rows from roster table")
        return selected
    raise ValueError(f"profile {profile_id!r} has unsupported included_records rule {included_records!r}")


def _read_table_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        import pandas as pd

        frame = pd.read_excel(path)
        return [_normalize_row(row) for row in frame.to_dict(orient="records")]
    if suffix in {".csv", ".tsv"}:
        import pandas as pd

        sep = "\t" if suffix == ".tsv" else ","
        frame = pd.read_csv(path, sep=sep)
        return [_normalize_row(row) for row in frame.to_dict(orient="records")]
    raise ValueError(f"unsupported roster table format {suffix!r}; expected .csv, .tsv, .xlsx, or .xls")


def _normalize_row(row: Mapping[str, Any]) -> dict[str, str]:
    return {str(key): "" if _is_nullish(value) else str(value).strip() for key, value in row.items()}


def _is_nullish(value: Any) -> bool:
    if value is None:
        return True
    try:
        import pandas as pd

        return bool(pd.isna(value))
    except Exception:
        return False


def _column_map(row: Mapping[str, Any]) -> dict[str, str]:
    return {_normalize_column(key): key for key in row}


def _resolve_column(columns: Mapping[str, str], aliases: Sequence[str], *, required: bool = True) -> str | None:
    for alias in aliases:
        column = columns.get(_normalize_column(alias))
        if column is not None:
            return column
    if required:
        raise ValueError(f"roster table is missing required column matching {aliases[0]!r}")
    return None


def _normalize_column(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _cell(row: Mapping[str, Any], column: str) -> str:
    return str(row.get(column, "")).strip()


def _status(row: Mapping[str, Any], column: str | None) -> str:
    status = _cell(row, column).lower() if column else "included"
    if not status:
        status = "included"
    if status not in _RECORD_STATUSES:
        raise ValueError(f"unsupported roster row status {status!r}")
    return status


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value
