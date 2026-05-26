"""USR dataset and overlay writers for RT-lnRNA Construct materialization."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

import pyarrow as pa
import yaml

from dnadesign.usr import Dataset, ensure_sequence_contract_namespaces

from ..infer_readiness import ConstructInferReadinessError, require_construct_infer_readiness
from .contracts import (
    _CONSTRUCT_SUBJECT_BIOLOGICAL_SEQUENCE_FIELDS,
    _CONSTRUCT_SUBJECT_INT_FIELDS,
    _CONSTRUCT_SUBJECT_OVERLAY,
    _INPUT_DATASET,
    _MATERIALIZATION_SOURCE,
    _OUTPUT_DATASET,
    MaterializationContractError,
)


def _write_construct_subject_dataset(*, usr_root: Path, rows: list[dict[str, object]]) -> dict[str, str]:
    field_names = _construct_subject_overlay_fields(rows)
    _ensure_construct_subject_overlay_namespace(usr_root, field_names=field_names)
    dataset = Dataset(usr_root, _INPUT_DATASET)
    dataset.init(source=_MATERIALIZATION_SOURCE, notes="Temp RT-lnRNA Construct materialization inputs.")
    carrier_sequences = [
        _construct_subject_envelope_carrier_sequence(index) for index, _row in enumerate(rows, start=1)
    ]
    add_result = dataset.add_sequences(
        carrier_sequences,
        bio_type="dna",
        alphabet="dna_4",
        source=_MATERIALIZATION_SOURCE,
    )
    input_ids_by_subject_id = {str(row["id"]): input_id for row, input_id in zip(rows, add_result.ids, strict=True)}
    input_ids = [input_ids_by_subject_id[str(row["id"])] for row in rows]
    columns: dict[str, pa.Array] = {
        "id": pa.array(input_ids, type=pa.string()),
        "construct_subject__id": pa.array([str(row["id"]) for row in rows], type=pa.string()),
    }
    for field_name in field_names:
        columns[field_name] = pa.array([row.get(field_name) for row in rows])
    dataset.write_overlay(_CONSTRUCT_SUBJECT_OVERLAY, pa.table(columns), key="id", overwrite=True)
    dataset.write_overlay(
        "usr_label",
        pa.table(
            {
                "id": pa.array(input_ids, type=pa.string()),
                "usr_label__primary": pa.array([str(row["id"]) for row in rows], type=pa.string()),
                "usr_label__aliases": pa.array([[] for _row in rows], type=pa.list_(pa.string())),
            }
        ),
        key="id",
        overwrite=True,
    )
    return input_ids_by_subject_id


def _construct_subject_envelope_carrier_sequence(index: int) -> str:
    if index < 1:
        raise MaterializationContractError("Construct-subject envelope carrier index must be positive.")
    alphabet = "ACGT"
    n = index - 1
    encoded: list[str] = []
    for _digit in range(10):
        n, remainder = divmod(n, len(alphabet))
        encoded.append(alphabet[remainder])
    if n:
        raise MaterializationContractError(
            "Construct-subject envelope carrier index exceeds synthetic policy capacity."
        )
    return "ACGT" + "".join(reversed(encoded))


def _write_construct_output_subject_bridge(
    *,
    usr_root: Path,
    input_ids_by_subject_id: Mapping[str, str],
) -> None:
    if not input_ids_by_subject_id:
        raise MaterializationContractError("Cannot bridge Construct outputs without construct-subject input ids.")

    input_dataset = Dataset(usr_root, _INPUT_DATASET)
    output_dataset = Dataset(usr_root, _OUTPUT_DATASET)
    input_frame = input_dataset.head(n=max(len(input_ids_by_subject_id) + 20, 1000))
    output_frame = output_dataset.head(n=max(len(input_ids_by_subject_id) * 4 + 20, 1000))
    construct_subject_columns = tuple(
        column for column in input_frame.columns if column.startswith("construct_subject__")
    )
    if "construct_subject__id" not in construct_subject_columns:
        raise MaterializationContractError("Construct input dataset is missing construct_subject__id overlay.")
    if output_frame.empty:
        raise MaterializationContractError("Construct output dataset has no rows to bridge.")

    input_by_id = {str(row["id"]): row for row in input_frame.to_dict(orient="records")}
    expected_input_ids = set(input_ids_by_subject_id.values())
    missing_inputs = sorted(expected_input_ids - set(input_by_id))
    if missing_inputs:
        raise MaterializationContractError(
            "Construct input dataset is missing construct subject bridge row(s): " + ", ".join(missing_inputs)
        )

    for construct_subject_id, input_id in input_ids_by_subject_id.items():
        input_construct_subject_id = str(input_by_id[input_id].get("construct_subject__id") or "")
        if input_construct_subject_id != construct_subject_id:
            raise MaterializationContractError(
                f"Construct input construct subject bridge mismatch for {input_id}: "
                f"expected {construct_subject_id}, found {input_construct_subject_id or '<missing>'}."
            )

    bridge_rows: list[dict[str, object]] = []
    seen_output_ids: set[str] = set()
    seen_input_ids: set[str] = set()
    for output_row in output_frame.to_dict(orient="records"):
        output_id = str(output_row.get("id") or "")
        input_id = str(output_row.get("construct__input_id") or "")
        if not output_id:
            raise MaterializationContractError("Construct output row is missing id.")
        if output_id in seen_output_ids:
            raise MaterializationContractError(
                f"Duplicate Construct output id while bridging construct subjects: {output_id}"
            )
        seen_output_ids.add(output_id)
        input_row = input_by_id.get(input_id)
        if input_row is None:
            raise MaterializationContractError(
                f"Construct output row {output_id} references unknown construct__input_id {input_id or '<missing>'}."
            )
        seen_input_ids.add(input_id)
        bridge_row = {"id": output_id}
        for column in construct_subject_columns:
            bridge_row[column] = _clean_overlay_value(input_row.get(column), field_name=column)
        bridge_row["construct_subject__record_kind"] = "construct_output"
        bridge_row["construct_subject__sequence_authority"] = "realized_construct_sequence"
        bridge_rows.append(bridge_row)

    missing_outputs = sorted(expected_input_ids - seen_input_ids)
    if missing_outputs:
        raise MaterializationContractError(
            "Construct output construct subject bridge has no realized output row(s) for input id(s): "
            + ", ".join(missing_outputs)
        )

    output_dataset.write_overlay(
        _CONSTRUCT_SUBJECT_OVERLAY,
        pa.table(
            {
                column: pa.array([row.get(column) for row in bridge_rows])
                for column in ("id", *construct_subject_columns)
            }
        ),
        key="id",
        overwrite=True,
    )
    with input_dataset.maintenance(reason="construct_output_subject_bridge_registry_refresh"):
        input_dataset.refresh_overlay_metadata(_CONSTRUCT_SUBJECT_OVERLAY)
        input_dataset.refresh_overlay_metadata("usr_label")
    with output_dataset.maintenance(reason="construct_output_subject_bridge_registry_refresh"):
        output_dataset.refresh_overlay_metadata(_CONSTRUCT_SUBJECT_OVERLAY)


def _require_construct_infer_ready(
    *,
    usr_root: Path,
    input_ids_by_subject_id: Mapping[str, str],
) -> None:
    try:
        require_construct_infer_readiness(
            usr_root=usr_root,
            input_dataset=_INPUT_DATASET,
            output_dataset=_OUTPUT_DATASET,
            expected_construct_subject_ids=tuple(input_ids_by_subject_id),
        )
    except ConstructInferReadinessError as exc:
        raise MaterializationContractError(str(exc)) from exc


def _clean_overlay_value(value: object, *, field_name: str) -> object:
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is not None and field_name in _CONSTRUCT_SUBJECT_INT_FIELDS:
        return int(value)
    return value


def _construct_subject_overlay_fields(rows: list[dict[str, object]]) -> tuple[str, ...]:
    required = _CONSTRUCT_SUBJECT_BIOLOGICAL_SEQUENCE_FIELDS
    extras = tuple(
        sorted(
            {
                key
                for row in rows
                for key in row
                if key.startswith("construct_subject__") and key not in {*required, "construct_subject__id"}
            }
        )
    )
    return (*required, *extras)


def _ensure_construct_subject_overlay_namespace(usr_root: Path, *, field_names: tuple[str, ...]) -> None:
    ensure_sequence_contract_namespaces(usr_root)
    registry_path = usr_root / "registry.yaml"
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise MaterializationContractError(f"{registry_path}: expected registry mapping")
    namespaces = payload.setdefault("namespaces", {})
    if not isinstance(namespaces, dict):
        raise MaterializationContractError(f"{registry_path}: namespaces must be a mapping")
    namespaces[_CONSTRUCT_SUBJECT_OVERLAY] = {
        "owner": "study",
        "description": "RT-lnRNA construct subjects and their slot sequences.",
        "columns": [{"name": "construct_subject__id", "type": "string"}]
        + [{"name": field_name, "type": _construct_subject_field_type(field_name)} for field_name in field_names],
    }
    registry_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def _construct_subject_field_type(field_name: str) -> str:
    if field_name in _CONSTRUCT_SUBJECT_INT_FIELDS:
        return "int64"
    if field_name in {"construct_subject__biological_sequence_fields", "construct_subject__permuter_modifications"}:
        return "list<string>"
    return "string"
