"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/output_store.py

Construct registry and output persistence helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pyarrow as pa
import yaml

from dnadesign.usr import Dataset, ensure_sequence_contract_namespaces
from dnadesign.usr.src.registry.models import DERIVED_COLUMNS

from .errors import ValidationError

_USR_STATE_COLUMNS = [
    {"name": "usr_state__masked", "type": "bool"},
    {"name": "usr_state__qc_status", "type": "string"},
    {"name": "usr_state__split", "type": "string"},
    {"name": "usr_state__supersedes", "type": "string"},
    {"name": "usr_state__lineage", "type": "list<string>"},
]

_CONSTRUCT_COLUMNS = [
    {"name": "construct__job", "type": "string"},
    {"name": "construct__spec_id", "type": "string"},
    {"name": "construct__context_id", "type": "string"},
    {"name": "construct__context_kind", "type": "string"},
    {"name": "construct__template_id", "type": "string"},
    {"name": "construct__template_kind", "type": "string"},
    {"name": "construct__template_source", "type": "string"},
    {"name": "construct__template_dataset", "type": "string"},
    {"name": "construct__template_field", "type": "string"},
    {"name": "construct__template_record_id", "type": "string"},
    {"name": "construct__template_sha256", "type": "string"},
    {"name": "construct__template_length", "type": "int64"},
    {"name": "construct__template_circular", "type": "bool"},
    {"name": "construct__input_dataset", "type": "string"},
    {"name": "construct__input_fields", "type": "list<string>"},
    {"name": "construct__input_id", "type": "string"},
    {"name": "construct__input_length", "type": "int64"},
    {"name": "construct__anchor_id", "type": "string"},
    {"name": "construct__anchor_orientation", "type": "string"},
    {"name": "construct__anchor_start", "type": "int64"},
    {"name": "construct__anchor_end", "type": "int64"},
    {"name": "construct__mode", "type": "string"},
    {"name": "construct__focal_part", "type": "string"},
    {"name": "construct__focal_part_length", "type": "int64"},
    {"name": "construct__window_semantics", "type": "string"},
    {"name": "construct__window_reference", "type": "string"},
    {"name": "construct__window_direction", "type": "string"},
    {"name": "construct__window_size_bp", "type": "int64"},
    {"name": "construct__window_upstream_bp", "type": "int64"},
    {"name": "construct__window_downstream_bp", "type": "int64"},
    {"name": "construct__window_offset_bp", "type": "int64"},
    {"name": "construct__window_start", "type": "int64"},
    {"name": "construct__window_end", "type": "int64"},
    {"name": "construct__resolved_length", "type": "int64"},
    {"name": "construct__full_construct_length", "type": "int64"},
    {
        "name": "construct__parts",
        "type": (
            "list<struct<name:string,role:string,sequence_source:string,sequence_field:string,"
            "placement_kind:string,orientation:string,template_start:int64,template_end:int64,"
            "realized_start:int64,realized_end:int64,length:int64>>"
        ),
    },
    {"name": "construct__orientation", "type": "string"},
    {"name": "construct__forward_anchor_start", "type": "int64"},
    {"name": "construct__forward_anchor_end", "type": "int64"},
    {"name": "construct__parent_forward_construct_id", "type": "string"},
]

_CONSTRUCT_SEED_COLUMNS = [
    {"name": "construct_seed__label", "type": "string"},
    {"name": "construct_seed__manifest_id", "type": "string"},
    {"name": "construct_seed__role", "type": "string"},
    {"name": "construct_seed__source_ref", "type": "string"},
    {"name": "construct_seed__topology", "type": "string"},
    {"name": "construct_seed__sha256", "type": "string"},
]

_USR_LABEL_COLUMNS = [
    {"name": "usr_label__primary", "type": "string"},
    {"name": "usr_label__aliases", "type": "list<string>"},
]


def _registry_path(root: Path) -> Path:
    return root / "registry.yaml"


def _load_registry_payload(root: Path) -> dict:
    path = _registry_path(root)
    if not path.exists():
        return {"namespaces": {}}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except OSError as exc:
        raise ValidationError(f"USR registry could not be read: {path}") from exc
    namespaces = data.get("namespaces") or {}
    if not isinstance(namespaces, dict):
        raise ValidationError(f"USR registry at {path} must contain a 'namespaces' mapping.")
    return {"namespaces": namespaces}


def _validated_registry_columns(namespace_name: str, payload: dict) -> dict[str, str]:
    columns = payload.get("columns")
    if columns is None:
        payload["columns"] = []
        return {}
    if not isinstance(columns, list):
        raise ValidationError(f"USR registry namespace '{namespace_name}' must define columns as a list.")
    observed: dict[str, str] = {}
    for index, item in enumerate(columns):
        if not isinstance(item, dict):
            raise ValidationError(f"USR registry namespace '{namespace_name}' column #{index + 1} must be a mapping.")
        name = str(item.get("name") or "").strip()
        type_name = str(item.get("type") or "").strip()
        if not name or not type_name:
            raise ValidationError(
                f"USR registry namespace '{namespace_name}' column #{index + 1} must define name and type."
            )
        if name in observed:
            raise ValidationError(f"USR registry namespace '{namespace_name}' duplicates column '{name}'.")
        observed[name] = type_name
    return observed


def _ensure_registry_namespace(
    *,
    namespace_name: str,
    namespaces: dict,
    owner: str,
    description: str,
    expected_columns: list[dict[str, str]],
) -> None:
    payload = namespaces.setdefault(
        namespace_name,
        {
            "owner": owner,
            "description": description,
            "columns": [],
        },
    )
    if not isinstance(payload, dict):
        raise ValidationError(f"USR registry namespace '{namespace_name}' must be a mapping.")
    payload.setdefault("owner", owner)
    payload.setdefault("description", description)
    observed = _validated_registry_columns(namespace_name, payload)
    missing = []
    for column in expected_columns:
        observed_type = observed.get(column["name"])
        if observed_type is None:
            missing.append(column)
            continue
        if observed_type != column["type"]:
            raise ValidationError(
                f"USR registry namespace '{namespace_name}' column '{column['name']}' has type "
                f"'{observed_type}', expected '{column['type']}'."
            )
    if missing:
        payload["columns"] = list(payload.get("columns", [])) + [dict(column) for column in missing]


def _ensure_construct_registry(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = _load_registry_payload(root)
    original_payload = deepcopy(payload)
    namespaces = payload["namespaces"]
    _ensure_registry_namespace(
        namespace_name="usr_state",
        namespaces=namespaces,
        owner="usr",
        description="Reserved record-state overlay (masked/qc/split/lineage).",
        expected_columns=_USR_STATE_COLUMNS,
    )
    _ensure_registry_namespace(
        namespace_name="construct",
        namespaces=namespaces,
        owner="construct",
        description="Construct lineage overlays for realized DNA sequences.",
        expected_columns=_CONSTRUCT_COLUMNS,
    )
    _ensure_registry_namespace(
        namespace_name="construct_seed",
        namespaces=namespaces,
        owner="construct",
        description="Construct bootstrap/import metadata for seeded input datasets.",
        expected_columns=_CONSTRUCT_SEED_COLUMNS,
    )
    _ensure_registry_namespace(
        namespace_name="usr_label",
        namespaces=namespaces,
        owner="usr",
        description="Human-readable labels and aliases for canonical sequence records.",
        expected_columns=_USR_LABEL_COLUMNS,
    )
    if payload != original_payload:
        _registry_path(root).write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")
    ensure_sequence_contract_namespaces(root)


def _existing_output_ids(root: Path, dataset_name: str) -> set[str]:
    ds = Dataset(root, dataset_name)
    if not ds.records_path.exists():
        return set()
    ids: set[str] = set()
    for batch in ds.scan(columns=["id"], include_overlays=False):
        ids.update(str(value) for value in batch.column("id").to_pylist())
    return ids


def _split_top_level_registry_fields(text: str) -> list[str]:
    out: list[str] = []
    depth = 0
    token: list[str] = []
    for char in text:
        if char == "," and depth == 0:
            piece = "".join(token).strip()
            if piece:
                out.append(piece)
            token = []
            continue
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
        token.append(char)
    piece = "".join(token).strip()
    if piece:
        out.append(piece)
    return out


def _registry_arrow_type(type_name: str) -> pa.DataType:
    text = type_name.strip()
    mapping: dict[str, pa.DataType] = {
        "bool": pa.bool_(),
        "string": pa.string(),
        "int64": pa.int64(),
    }
    if text in mapping:
        return mapping[text]
    if text.startswith("list<") and text.endswith(">"):
        return pa.list_(_registry_arrow_type(text[5:-1]))
    if text.startswith("struct<") and text.endswith(">"):
        fields = []
        for item in _split_top_level_registry_fields(text[len("struct<") : -1]):
            if ":" not in item:
                raise ValidationError(f"Unsupported registry column type '{type_name}' for construct overlay attach.")
            name, inner = item.split(":", 1)
            fields.append(pa.field(name.strip(), _registry_arrow_type(inner.strip())))
        return pa.struct(fields)
    raise ValidationError(f"Unsupported registry column type '{type_name}' for construct overlay attach.")


def _construct_metadata_table(metadata_rows: list[dict[str, object]]) -> pa.Table:
    if not metadata_rows:
        return pa.table(
            {"id": pa.array([], type=pa.string())},
            schema=pa.schema([pa.field("id", pa.string())]),
        )
    schema = pa.schema(
        [pa.field("id", pa.string())]
        + [pa.field(col["name"], _registry_arrow_type(col["type"])) for col in _CONSTRUCT_COLUMNS]
    )
    return pa.table(
        {
            field.name: pa.array(
                [row.get(field.name) for row in metadata_rows],
                type=field.type,
            )
            for field in schema
        },
        schema=schema,
    )


def _derived_metadata_table(metadata_rows: list[dict[str, object]]) -> pa.Table:
    if not metadata_rows:
        return pa.table(
            {"id": pa.array([], type=pa.string())},
            schema=pa.schema([pa.field("id", pa.string())]),
        )
    schema = pa.schema(
        [pa.field("id", pa.string())]
        + [pa.field(column.name, _registry_arrow_type(column.type)) for column in DERIVED_COLUMNS]
    )
    return pa.table(
        {
            field.name: pa.array(
                [row.get(field.name) for row in metadata_rows],
                type=field.type,
            )
            for field in schema
        },
        schema=schema,
    )


def _usr_label_table(label_rows: list[dict[str, object]]) -> pa.Table:
    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("usr_label__primary", pa.string()),
            pa.field("usr_label__aliases", pa.list_(pa.string())),
        ]
    )
    return pa.table(
        {
            "id": pa.array([row.get("id") for row in label_rows], type=pa.string()),
            "usr_label__primary": pa.array([row.get("usr_label__primary") for row in label_rows], type=pa.string()),
            "usr_label__aliases": pa.array(
                [list(row.get("usr_label__aliases") or []) for row in label_rows],
                type=pa.list_(pa.string()),
            ),
        },
        schema=schema,
    )
