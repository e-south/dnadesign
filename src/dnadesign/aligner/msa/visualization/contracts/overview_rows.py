"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/aligner/msa/visualization/contracts/overview_rows.py

Overview-row resolution for generic MSA visualization panels.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from string import Formatter
from typing import Any

import yaml

from dnadesign.aligner.msa.visualization.contracts.models import ExemplarRow
from dnadesign.aligner.msa.visualization.contracts.panel_spec import MsaPanelSpec

_TEMPLATE_FIELDS = {"record_id", "node", "row_index", "accession", "provider_id"}


def build_overview_rows(
    *,
    profile_id: str,
    records: Mapping[str, str],
    target_row_id: str,
    exemplar_rows: tuple[ExemplarRow, ...],
    panel_spec: MsaPanelSpec,
) -> tuple[ExemplarRow, ...]:
    """Resolve whole-alignment overview rows from the declared row source."""

    if panel_spec.overview_row_source == "exemplar_rows":
        selected = exemplar_rows
        if panel_spec.max_display_rows is not None:
            selected = selected[: panel_spec.max_display_rows]
        return selected
    if panel_spec.overview_row_source == "all_records":
        return _all_record_rows(
            profile_id=profile_id,
            records=records,
            target_row_id=target_row_id,
            panel_spec=panel_spec,
        )
    raise ValueError(f"unsupported overview row source {panel_spec.overview_row_source!r}")


def _all_record_rows(
    *,
    profile_id: str,
    records: Mapping[str, str],
    target_row_id: str,
    panel_spec: MsaPanelSpec,
) -> tuple[ExemplarRow, ...]:
    profile_spec = panel_spec.overview_profile_spec(profile_id)
    metadata = _source_manifest_metadata(profile_spec.source_manifest_path)
    _validate_template(profile_spec.label_template)
    rows: list[ExemplarRow] = []
    for record_id in records:
        if record_id == target_row_id:
            rows.append(
                ExemplarRow(
                    record_id=record_id,
                    label=profile_spec.target_label or record_id,
                    group="target",
                )
            )
            continue
        record_metadata = metadata.get(record_id, {})
        if profile_spec.source_manifest_path is not None and not record_metadata:
            raise ValueError(f"{profile_id} overview row {record_id!r} is missing source-manifest metadata")
        label = _format_label(record_id=record_id, metadata=record_metadata, template=profile_spec.label_template)
        rows.append(
            ExemplarRow(
                record_id=record_id,
                label=_trim_label(label, max_chars=profile_spec.label_max_chars),
                group=profile_spec.group or profile_id,
            )
        )
    return tuple(rows)


def _source_manifest_metadata(path: Path | None) -> dict[str, dict[str, str]]:
    if path is None:
        return {}
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"source manifest must be a YAML mapping: {path}")
    raw_records = payload.get("included_records")
    if not isinstance(raw_records, list):
        raise ValueError(f"source manifest must declare included_records as a list: {path}")
    metadata: dict[str, dict[str, str]] = {}
    for index, raw_record in enumerate(raw_records):
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"source manifest included_records[{index}] must be a mapping")
        record_id = _required_string(raw_record, "record_id", context=f"included_records[{index}]")
        node, row_index = _record_id_parts(record_id)
        metadata[record_id] = {
            "record_id": record_id,
            "node": node,
            "row_index": row_index,
            "accession": _optional_string(raw_record.get("accession")),
            "provider_id": _optional_string(raw_record.get("provider_id")),
        }
    return metadata


def _format_label(*, record_id: str, metadata: Mapping[str, str], template: str) -> str:
    context = {
        "record_id": record_id,
        "node": "",
        "row_index": "",
        "accession": "",
        "provider_id": "",
        **metadata,
    }
    return re.sub(r"\s+", " ", template.format(**context)).strip() or record_id


def _validate_template(template: str) -> None:
    formatter = Formatter()
    fields = {field for _, field, _, _ in formatter.parse(template) if field}
    unknown = sorted(fields - _TEMPLATE_FIELDS)
    if unknown:
        raise ValueError(f"overview label_template contains unsupported field {unknown[0]!r}")


def _record_id_parts(record_id: str) -> tuple[str, str]:
    parts = record_id.rsplit("__", 2)
    if len(parts) == 3:
        return parts[1], str(int(parts[2])) if parts[2].isdigit() else parts[2]
    return "", ""


def _trim_label(label: str, *, max_chars: int) -> str:
    if len(label) <= max_chars:
        return label
    if max_chars <= 3:
        return label[:max_chars]
    return label[: max_chars - 3].rstrip() + "..."


def _required_string(payload: Mapping[str, Any], key: str, *, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"source manifest {context}.{key} must be a non-empty string")
    return value.strip()


def _optional_string(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""
