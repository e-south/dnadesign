"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/formatting/renderers/ingest.py

Renders ingest-y command output for OPAL CLI. Formats ingest preview and commit
summaries.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

from ...tui import kv_table, tui_enabled
from ..core import _as_dict, _truncate, bullet_list, kv_block, short_array


def _render_id_value(value: Any) -> str:
    if value is None:
        return "<unresolved>"
    try:
        if isinstance(value, float) and value != value:
            return "<unresolved>"
    except Exception:
        pass
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "nat"}:
        return "<unresolved>"
    return text


def render_ingest_preview_text(
    preview: Any,
    sample_rows: Sequence[Mapping[str, Any]],
    *,
    transform_name: Optional[str] = None,
) -> str:
    p = _as_dict(preview)
    if tui_enabled():
        from rich import box
        from rich.console import Group
        from rich.table import Table

        head = kv_table(
            f"[Preview] ingest-y (transform={transform_name or '(from YAML)'})",
            {
                "rows in input table": p.get("total_rows_in_csv"),
                "columns present": f"id={'yes' if p.get('rows_with_id') else 'no'}, "
                f"sequence={'yes' if p.get('rows_with_sequence') else 'no'}",
                "resolved ids by sequence": p.get("resolved_ids_by_sequence"),
                "unknown sequences": p.get("unknown_sequences"),
                "y_expected_length": p.get("y_expected_length"),
                "y_length_ok (sampled)": p.get("y_length_ok"),
                "y_length_bad (sampled)": p.get("y_length_bad"),
                "duplicate policy": p.get("duplicate_policy"),
                "duplicate key": p.get("duplicate_key"),
                "duplicates found": p.get("duplicates_found"),
                "duplicates dropped": p.get("duplicates_dropped"),
                "warnings": ", ".join(p.get("warnings") or []) or "(none)",
            },
        )
        blocks = [head] if head is not None else []
        if sample_rows:
            table = Table(title="Sample (first 5)", box=box.ASCII)
            table.add_column("id", style="bold")
            table.add_column("sequence")
            table.add_column("y")
            for r in sample_rows[:5]:
                seq = _truncate(r.get("sequence", ""))
                rid = _render_id_value(r.get("id", ""))
                y = r.get("y", "")
                y_str = short_array(y, maxlen=6) if isinstance(y, (list, tuple)) else _truncate(str(y), 64)
                table.add_row(rid, str(seq), str(y_str))
            blocks.append(table)
        return Group(*blocks)

    head = kv_block(
        f"[Preview] ingest-y (transform={transform_name or '(from YAML)'})",
        {
            "rows in input table": p.get("total_rows_in_csv"),
            "columns present": f"id={'yes' if p.get('rows_with_id') else 'no'}, "
            f"sequence={'yes' if p.get('rows_with_sequence') else 'no'}",
            "resolved ids by sequence": p.get("resolved_ids_by_sequence"),
            "unknown sequences": p.get("unknown_sequences"),
            "y_expected_length": p.get("y_expected_length"),
            "y_length_ok (sampled)": p.get("y_length_ok"),
            "y_length_bad (sampled)": p.get("y_length_bad"),
            "duplicate policy": p.get("duplicate_policy"),
            "duplicate key": p.get("duplicate_key"),
            "duplicates found": p.get("duplicates_found"),
            "duplicates dropped": p.get("duplicates_dropped"),
            "warnings": ", ".join(p.get("warnings") or []) or "(none)",
        },
    )
    if not sample_rows:
        return head

    lines: List[str] = []
    for r in sample_rows[:5]:
        seq = _truncate(r.get("sequence", ""))
        rid = _render_id_value(r.get("id", ""))
        y = r.get("y", "")
        y_str = short_array(y, maxlen=6) if isinstance(y, (list, tuple)) else _truncate(str(y), 64)
        lines.append(f"id={rid}  sequence={seq}  y={y_str}")

    return head + "\n\n" + bullet_list("Sample (first 5)", lines)


def render_ingest_commit_text(
    *,
    round_index: int,
    labels_appended: int,
    labels_skipped: int = 0,
    y_column_updated: str,
) -> str:
    payload = {
        "round": round_index,
        "labels appended": labels_appended,
        "y column updated": y_column_updated,
        "ledger labels appended": "yes",
    }
    if labels_skipped:
        payload["labels skipped"] = labels_skipped
    if tui_enabled():
        table = kv_table("[Committed] ingest-y", payload)
        if table is not None:
            return table
    return kv_block("[Committed] ingest-y", payload)


def render_ingest_runtime_text(runtime: Any) -> str:
    payload = _as_dict(runtime)
    rows = payload.get("candidate_index_rows")
    records_rows = payload.get("records_row_count")
    memory = _format_bytes(payload.get("estimated_memory_bytes"))
    peak = _format_bytes(payload.get("peak_rss_bytes"))
    fields = {
        "schema": payload.get("schema_version"),
        "mode": payload.get("mode"),
        "label source": payload.get("label_source_kind"),
        "write scope": payload.get("write_scope"),
        "fixed universe": _yes_no(payload.get("fixed_candidate_universe")),
        "full records loaded": _yes_no(payload.get("full_records_loaded")),
        "loaded columns": _format_columns(payload.get("records_columns_loaded") or []),
        "X column loaded": _yes_no(payload.get("candidate_x_column_loaded")),
        "candidate index rows": rows,
        "records rows": records_rows,
        "estimated frame memory": memory,
        "process peak RSS": peak,
        "unknown policy": payload.get("unknown_sequences_policy"),
    }
    if tui_enabled():
        table = kv_table("[Runtime] ingest-y", fields)
        if table is not None:
            return table
    return kv_block("[Runtime] ingest-y", fields)


def _yes_no(value: Any) -> str:
    return "yes" if bool(value) else "no"


def _format_bytes(value: Any) -> str:
    if value is None:
        return "unknown"
    try:
        size = float(value)
    except Exception:
        return str(value)
    units = ("B", "KiB", "MiB", "GiB")
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    if idx == 0:
        return f"{int(size)} {units[idx]}"
    return f"{size:.2f} {units[idx]}"


def _format_columns(columns: Any) -> str:
    try:
        values = [str(column) for column in columns]
    except TypeError:
        values = [str(columns)]
    return ", ".join(values) if values else "(none)"
