"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/construct/src/runtime.py

Construct runtime: template loading, realization, and USR persistence.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from dnadesign.usr import Dataset, compute_id, default_usr_root, normalize_sequence, normalize_usr_root

from .config import JobConfig, PartConfig, load_job_config
from .errors import ValidationError

_DNA_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")

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
    {"name": "construct__input_field", "type": "string"},
    {"name": "construct__input_fields", "type": "list<string>"},
    {"name": "construct__anchor_id", "type": "string"},
    {"name": "construct__anchor_length", "type": "int64"},
    {"name": "construct__mode", "type": "string"},
    {"name": "construct__focal_part", "type": "string"},
    {"name": "construct__window_bp", "type": "int64"},
    {"name": "construct__window_start", "type": "int64"},
    {"name": "construct__window_end", "type": "int64"},
    {"name": "construct__full_construct_length", "type": "int64"},
    {"name": "construct__part_count", "type": "int64"},
    {"name": "construct__part_names", "type": "list<string>"},
    {"name": "construct__part_roles", "type": "list<string>"},
    {"name": "construct__part_kinds", "type": "list<string>"},
    {"name": "construct__part_starts", "type": "list<int64>"},
    {"name": "construct__part_ends", "type": "list<int64>"},
    {"name": "construct__part_orientations", "type": "list<string>"},
    {"name": "construct__part_template_starts", "type": "list<int64>"},
    {"name": "construct__part_template_ends", "type": "list<int64>"},
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


@dataclass(frozen=True)
class RunResult:
    job_id: str
    input_dataset: str
    output_dataset: str
    output_root: Path
    records_total: int
    records_written: int
    records_skipped_existing: int
    spec_id: str
    dry_run: bool


@dataclass(frozen=True)
class PlannedRow:
    input_id: str
    output_id: str
    anchor_length: int
    output_length: int
    full_construct_length: int


@dataclass(frozen=True)
class PlannedPlacement:
    part_name: str
    part_role: str
    sequence_source: str
    sequence_field: str | None
    placement_kind: str
    template_start: int
    template_end: int
    orientation: str
    expected_template_sequence: str | None


@dataclass(frozen=True)
class PreflightResult:
    job_id: str
    input_dataset: str
    output_dataset: str
    input_root: Path
    output_root: Path
    template_id: str
    template_kind: str
    template_source: str
    template_dataset: str | None
    template_field: str | None
    template_record_id: str | None
    template_sha256: str
    template_length: int
    template_circular: bool
    realize_mode: str
    focal_part: str | None
    focal_point: str
    anchor_offset_bp: int
    window_bp: int | None
    spec_id: str
    records_total: int
    existing_output_collisions: int
    output_on_conflict: str
    placements: List[PlannedPlacement]
    planned_rows: List[PlannedRow]


@dataclass(frozen=True)
class _ResolvedPart:
    name: str
    role: str
    kind: str
    orientation: str
    start: int
    end: int
    sequence: str
    realized_start: int
    realized_end: int


@dataclass(frozen=True)
class _BuiltRecord:
    output_id: str
    sequence: str
    alphabet: str
    metadata: Dict[str, object]


@dataclass(frozen=True)
class _ResolvedTemplate:
    id: str
    kind: str
    sequence: str
    source: str
    dataset: str | None
    field: str | None
    record_id: str | None
    circular: bool


def _default_usr_root() -> Path:
    return default_usr_root()


def _resolve_optional_path(base_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _resolve_usr_root(base_dir: Path, value: str | None) -> Path:
    return normalize_usr_root(_resolve_optional_path(base_dir, value))


def _ensure_dna_text(text: str, *, label: str) -> str:
    seq = str(text or "").strip()
    if not seq:
        raise ValidationError(f"{label} cannot be empty.")
    try:
        alphabet = _alphabet_for_sequence(seq)
        normalize_sequence(seq, "dna", alphabet)
    except ValueError as exc:
        raise ValidationError(f"{label} must be valid DNA (ACGT or ACGTN).") from exc
    return seq


def _alphabet_for_sequence(sequence: str) -> str:
    return "dna_5" if "N" in sequence.upper() else "dna_4"


def _reverse_complement(sequence: str) -> str:
    return sequence.translate(_DNA_COMPLEMENT)[::-1]


def _expected_template_sequence(part: PartConfig) -> str | None:
    expected = part.placement.expected_template_sequence
    if expected is None:
        return None
    return _ensure_dna_text(
        str(expected),
        label=f"placement.expected_template_sequence for part '{part.name}'",
    )


def _load_template_sequence(base_dir: Path, cfg: JobConfig) -> _ResolvedTemplate:
    template = cfg.job.template
    if template.kind == "literal":
        if template.sequence is None:
            raise ValidationError("template.sequence is required when template.kind='literal'.")
        seq = _ensure_dna_text(template.sequence, label="template.sequence")
        return _ResolvedTemplate(
            id=template.id,
            kind="literal",
            sequence=seq,
            source=template.source or "template.sequence",
            dataset=None,
            field=None,
            record_id=None,
            circular=bool(template.circular),
        )

    if template.kind == "path":
        path = _resolve_optional_path(base_dir, template.path)
        if path is None or not path.exists():
            raise ValidationError(f"Template path not found: {template.path}")
        if not path.is_file():
            raise ValidationError(f"Template path must resolve to a readable file: {path}")
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise ValidationError(f"Template path could not be read: {path}") from exc
        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        if not lines:
            raise ValidationError(f"Template file is empty: {path}")
        if lines[0].startswith(">"):
            header_count = sum(1 for line in lines if line.startswith(">"))
            if header_count != 1:
                raise ValidationError(f"Template FASTA must contain exactly one record. Found {header_count}: {path}")
            seq_lines = [line for line in lines if not line.startswith(">")]
            if not seq_lines:
                raise ValidationError(f"Template FASTA does not contain sequence lines: {path}")
            seq = "".join(seq_lines)
        else:
            seq = "".join(lines)
        return _ResolvedTemplate(
            id=template.id,
            kind="path",
            sequence=_ensure_dna_text(seq, label=f"template.path ({path})"),
            source=template.source or str(path),
            dataset=None,
            field=None,
            record_id=None,
            circular=bool(template.circular),
        )

    if template.kind != "usr":
        raise ValidationError(f"Unsupported template.kind '{template.kind}'.")

    template_root = _resolve_usr_root(base_dir, template.root or cfg.job.input.root)
    template_ds = Dataset(template_root, str(template.dataset))
    if not template_ds.records_path.exists():
        raise ValidationError(f"Template dataset not initialized: {template_ds.records_path}")
    rows = _scan_usr_rows(
        template_ds,
        columns=["id", str(template.field)],
        ids=[str(template.record_id)],
    )
    if len(rows) != 1:
        raise ValidationError(f"Template selection must resolve exactly one row in dataset '{template.dataset}'.")
    row = rows[0]
    raw = row.get(str(template.field))
    if raw is None:
        raise ValidationError(f"Template record '{template.record_id}' is missing field '{template.field}'.")
    seq = _ensure_dna_text(str(raw), label=f"template field '{template.field}' in dataset '{template.dataset}'")
    return _ResolvedTemplate(
        id=template.id,
        kind="usr",
        sequence=seq,
        source=template.source or f"usr:{template.dataset}:{template.record_id}",
        dataset=str(template.dataset),
        field=str(template.field),
        record_id=str(template.record_id),
        circular=bool(template.circular),
    )


def _scan_usr_rows(ds: Dataset, *, columns: List[str], ids: List[str] | None) -> List[dict[str, object]]:
    wanted = [str(value) for value in (ids or []) if str(value).strip()]
    wanted_set = set(wanted) if wanted else None
    found: dict[str, dict[str, object]] = {}
    ordered: list[dict[str, object]] = []

    for batch in ds.scan(columns=columns, include_overlays=True):
        payload = batch.to_pydict()
        row_count = batch.num_rows
        for idx in range(row_count):
            row = {name: payload[name][idx] for name in payload}
            row_id = str(row["id"])
            if wanted_set is not None:
                if row_id not in wanted_set:
                    continue
                found[row_id] = row
            else:
                ordered.append(row)

    if wanted_set is not None:
        missing = [row_id for row_id in wanted if row_id not in found]
        if missing:
            preview = ", ".join(missing[:5])
            raise ValidationError(f"{len(missing)} requested input id(s) were not found. Sample: {preview}.")
        return [found[row_id] for row_id in wanted]
    return ordered


def _input_fields(cfg: JobConfig) -> List[str]:
    fields = {"id", cfg.job.input.field}
    for part in cfg.job.parts:
        if part.sequence.source == "input_field":
            fields.add(str(part.sequence.field))
    return sorted(fields)


def _planned_placements(parts: Iterable[PartConfig]) -> List[PlannedPlacement]:
    return [
        PlannedPlacement(
            part_name=part.name,
            part_role=part.role,
            sequence_source=part.sequence.source,
            sequence_field=str(part.sequence.field) if part.sequence.field is not None else None,
            placement_kind=part.placement.kind,
            template_start=part.placement.start,
            template_end=part.placement.end,
            orientation=part.placement.orientation,
            expected_template_sequence=_expected_template_sequence(part),
        )
        for part in parts
    ]


def _part_sequence(part: PartConfig, row: dict[str, object]) -> str:
    if part.sequence.source == "literal":
        seq = _ensure_dna_text(str(part.sequence.literal), label=f"literal for part '{part.name}'")
    else:
        raw = row.get(str(part.sequence.field))
        if raw is None:
            raise ValidationError(
                f"Input row '{row.get('id')}' is missing field '{part.sequence.field}' for part '{part.name}'."
            )
        seq = _ensure_dna_text(str(raw), label=f"input field '{part.sequence.field}' for part '{part.name}'")
    if part.placement.orientation == "reverse_complement":
        return _reverse_complement(seq)
    return seq


def _validate_placements(template_len: int, parts: Iterable[PartConfig]) -> List[PartConfig]:
    indexed_parts = list(enumerate(parts))
    ordered = [
        part
        for _, part in sorted(
            indexed_parts,
            key=lambda item: (item[1].placement.start, item[0]),
        )
    ]
    prior_end = -1
    prior_name = None
    prior_start = None
    prior_template_end = None
    for part in ordered:
        start = part.placement.start
        end = part.placement.end
        if end > template_len:
            raise ValidationError(f"Part '{part.name}' placement end {end} exceeds template length {template_len}.")
        if prior_start is not None and start == prior_start and end != prior_template_end:
            raise ValidationError(
                f"Part '{part.name}' shares template start {start} with part '{prior_name}' but uses a different "
                "template end. Same-start placements with different intervals are ambiguous; use distinct start "
                "coordinates or split them into separate construct jobs."
            )
        if start < prior_end:
            raise ValidationError(
                f"Part '{part.name}' overlaps prior placement '{prior_name}'. Placements must not overlap."
            )
        prior_end = end
        prior_name = part.name
        prior_start = start
        prior_template_end = end
    return ordered


def _assemble_full_construct(
    template_seq: str,
    parts: List[PartConfig],
    row: dict[str, object],
) -> tuple[str, List[_ResolvedPart], Dict[str, _ResolvedPart]]:
    ordered = _validate_placements(len(template_seq), parts)
    cursor = 0
    out: list[str] = []
    out_len = 0
    realized: Dict[str, _ResolvedPart] = {}
    realized_ordered: list[_ResolvedPart] = []

    for part in ordered:
        expected_template = _expected_template_sequence(part)
        template_interval = template_seq[part.placement.start : part.placement.end]
        if expected_template is not None and template_interval.upper() != expected_template.upper():
            raise ValidationError(
                f"Part '{part.name}' expected template interval "
                f"[{part.placement.start}, {part.placement.end}) to match the configured incumbent sequence."
            )
        prefix = template_seq[cursor : part.placement.start]
        out.append(prefix)
        out_len += len(prefix)

        seq = _part_sequence(part, row)
        realized_start = out_len
        out.append(seq)
        out_len += len(seq)
        realized_end = out_len

        resolved_part = _ResolvedPart(
            name=part.name,
            role=part.role,
            kind=part.placement.kind,
            orientation=part.placement.orientation,
            start=part.placement.start,
            end=part.placement.end,
            sequence=seq,
            realized_start=realized_start,
            realized_end=realized_end,
        )
        realized[part.name] = resolved_part
        realized_ordered.append(resolved_part)
        cursor = part.placement.end

    out.append(template_seq[cursor:])
    return "".join(out), realized_ordered, realized


def _focal_index(part: _ResolvedPart, *, focal_point: str) -> int:
    if focal_point == "start":
        return part.realized_start
    if focal_point == "end":
        return part.realized_end - 1
    return part.realized_start + (len(part.sequence) // 2)


def _extract_output_sequence(
    *,
    full_construct: str,
    realized_parts: Dict[str, _ResolvedPart],
    cfg: JobConfig,
) -> tuple[str, int, int]:
    if cfg.job.realize.mode == "full_construct":
        return full_construct, 0, len(full_construct)

    focal = realized_parts[cfg.job.realize.focal_part]
    point = _focal_index(focal, focal_point=cfg.job.realize.focal_point)
    window_bp = int(cfg.job.realize.window_bp)
    if window_bp > len(full_construct):
        raise ValidationError(
            f"Requested window_bp={window_bp} exceeds realized construct length {len(full_construct)}."
        )
    start_raw = point - (window_bp // 2) + int(cfg.job.realize.anchor_offset_bp)
    end_raw = start_raw + window_bp
    if cfg.job.template.circular:
        seq = "".join(full_construct[(start_raw + idx) % len(full_construct)] for idx in range(window_bp))
        start = start_raw % len(full_construct)
        end = (start + window_bp) % len(full_construct)
        return seq, start, end
    if start_raw < 0 or end_raw > len(full_construct):
        raise ValidationError(
            "Requested window extends beyond the linear construct boundaries. "
            "Adjust window_bp, anchor_offset_bp, or choose a circular template."
        )
    return full_construct[start_raw:end_raw], start_raw, end_raw


def _spec_id(
    cfg: JobConfig,
    *,
    template: _ResolvedTemplate,
    template_sha256: str,
    input_root: Path,
    output_root: Path,
) -> str:
    payload = {
        "job_id": cfg.job.id,
        "input": {
            "dataset": cfg.job.input.dataset,
            "field": cfg.job.input.field,
            "ids": list(cfg.job.input.ids or []),
            "root": str(input_root),
        },
        "template": {
            "id": cfg.job.template.id,
            "kind": template.kind,
            "circular": template.circular,
            "source": template.source,
            "dataset": template.dataset,
            "field": template.field,
            "record_id": template.record_id,
            "sha256": template_sha256,
        },
        "parts": [
            {
                "name": part.name,
                "role": part.role,
                "sequence": {
                    "source": part.sequence.source,
                    "field": part.sequence.field,
                    "literal": part.sequence.literal,
                },
                "placement": {
                    "kind": part.placement.kind,
                    "start": part.placement.start,
                    "end": part.placement.end,
                    "orientation": part.placement.orientation,
                    "expected_template_sequence": part.placement.expected_template_sequence,
                },
            }
            for part in cfg.job.parts
        ],
        "realize": {
            "mode": cfg.job.realize.mode,
            "focal_part": cfg.job.realize.focal_part,
            "focal_point": cfg.job.realize.focal_point,
            "anchor_offset_bp": cfg.job.realize.anchor_offset_bp,
            "window_bp": cfg.job.realize.window_bp,
        },
        "output": {
            "dataset": cfg.job.output.dataset,
            "root": str(output_root),
            "source": cfg.job.output.source,
            "on_conflict": cfg.job.output.on_conflict,
            "allow_same_as_input": cfg.job.output.allow_same_as_input,
        },
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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
        payload["columns"] = list(payload.get("columns", [])) + missing


def _ensure_construct_registry(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = _load_registry_payload(root)
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
    path = _registry_path(root)
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def _build_record(
    *,
    row: dict[str, object],
    cfg: JobConfig,
    template: _ResolvedTemplate,
    template_sha256: str,
    spec_id: str,
    ordered_parts: List[PartConfig],
) -> _BuiltRecord:
    full_construct, ordered_realized_parts, realized_parts = _assemble_full_construct(
        template.sequence,
        ordered_parts,
        row,
    )
    output_sequence, window_start, window_end = _extract_output_sequence(
        full_construct=full_construct,
        realized_parts=realized_parts,
        cfg=cfg,
    )
    alphabet = _alphabet_for_sequence(output_sequence)
    sequence_norm = normalize_sequence(output_sequence, "dna", alphabet)
    output_id = compute_id("dna", sequence_norm)

    input_fields = [str(part.sequence.field) for part in cfg.job.parts if part.sequence.source == "input_field"]
    metadata = {
        "id": output_id,
        "construct__job": cfg.job.id,
        "construct__spec_id": spec_id,
        "construct__template_id": template.id,
        "construct__template_kind": template.kind,
        "construct__template_source": template.source,
        "construct__template_dataset": template.dataset or "",
        "construct__template_field": template.field or "",
        "construct__template_record_id": template.record_id or "",
        "construct__template_sha256": template_sha256,
        "construct__template_length": len(template.sequence),
        "construct__template_circular": bool(template.circular),
        "construct__input_dataset": cfg.job.input.dataset,
        "construct__input_field": cfg.job.input.field,
        "construct__input_fields": input_fields,
        "construct__anchor_id": str(row["id"]),
        "construct__anchor_length": len(str(row[cfg.job.input.field]).strip()),
        "construct__mode": cfg.job.realize.mode,
        "construct__focal_part": cfg.job.realize.focal_part or "",
        "construct__window_bp": int(cfg.job.realize.window_bp) if cfg.job.realize.window_bp is not None else -1,
        "construct__window_start": window_start,
        "construct__window_end": window_end,
        "construct__full_construct_length": len(full_construct),
        "construct__part_count": len(ordered_realized_parts),
        "construct__part_names": [part.name for part in ordered_realized_parts],
        "construct__part_roles": [part.role for part in ordered_realized_parts],
        "construct__part_kinds": [part.kind for part in ordered_realized_parts],
        "construct__part_starts": [part.realized_start for part in ordered_realized_parts],
        "construct__part_ends": [part.realized_end for part in ordered_realized_parts],
        "construct__part_orientations": [part.orientation for part in ordered_realized_parts],
        "construct__part_template_starts": [part.start for part in ordered_realized_parts],
        "construct__part_template_ends": [part.end for part in ordered_realized_parts],
    }
    return _BuiltRecord(output_id=output_id, sequence=output_sequence, alphabet=alphabet, metadata=metadata)


def _attach_construct_metadata(ds: Dataset, metadata_rows: List[dict[str, object]]) -> None:
    if not metadata_rows:
        return
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "construct_attach.parquet"
        schema = pa.schema(
            [pa.field("id", pa.string())]
            + [pa.field(col["name"], _registry_arrow_type(col["type"])) for col in _CONSTRUCT_COLUMNS]
        )
        table = pa.table(
            {
                field.name: pa.array(
                    [row.get(field.name) for row in metadata_rows],
                    type=field.type,
                )
                for field in schema
            },
            schema=schema,
        )
        pq.write_table(table, path)
        ds.attach(
            path,
            namespace="construct",
            key="id",
            key_col="id",
            columns=[field.name for field in schema if field.name != "id"],
            allow_overwrite=True,
            note="dnadesign.construct lineage attach",
        )


def _existing_output_ids(root: Path, dataset_name: str) -> set[str]:
    ds = Dataset(root, dataset_name)
    if not ds.records_path.exists():
        return set()
    return {str(row["id"]) for row in _scan_usr_rows(ds, columns=["id"], ids=None)}


def _registry_arrow_type(type_name: str) -> pa.DataType:
    mapping: dict[str, pa.DataType] = {
        "bool": pa.bool_(),
        "string": pa.string(),
        "int64": pa.int64(),
        "list<string>": pa.list_(pa.string()),
        "list<int64>": pa.list_(pa.int64()),
    }
    if type_name not in mapping:
        raise ValidationError(f"Unsupported registry column type '{type_name}' for construct overlay attach.")
    return mapping[type_name]


def _plan_from_config(path: str | Path) -> tuple[PreflightResult, List[_BuiltRecord]]:
    cfg, config_path = load_job_config(path)
    base_dir = config_path.parent
    input_root = _resolve_usr_root(base_dir, cfg.job.input.root)
    output_root = _resolve_usr_root(base_dir, cfg.job.output.root or cfg.job.input.root)

    input_ds = Dataset(input_root, cfg.job.input.dataset)
    if not input_ds.records_path.exists():
        raise ValidationError(f"Input dataset not initialized: {input_ds.records_path}")
    if (
        input_root == output_root
        and cfg.job.input.dataset == cfg.job.output.dataset
        and not cfg.job.output.allow_same_as_input
    ):
        raise ValidationError(
            "Output dataset resolves to the same root/dataset as input. "
            "Set output.allow_same_as_input=true only when recursive accumulation is intentional."
        )

    template = _load_template_sequence(base_dir, cfg)
    ordered_parts = _validate_placements(len(template.sequence), cfg.job.parts)
    template_sha256 = hashlib.sha256(template.sequence.encode("utf-8")).hexdigest()
    spec_id = _spec_id(
        cfg,
        template=template,
        template_sha256=template_sha256,
        input_root=input_root,
        output_root=output_root,
    )

    rows = _scan_usr_rows(input_ds, columns=_input_fields(cfg), ids=cfg.job.input.ids)
    if not rows:
        raise ValidationError("Input selection resolved to zero rows.")

    built = [
        _build_record(
            row=row,
            cfg=cfg,
            template=template,
            template_sha256=template_sha256,
            spec_id=spec_id,
            ordered_parts=ordered_parts,
        )
        for row in rows
    ]
    duplicate_output_ids = sorted(
        output_id for output_id, count in Counter(record.output_id for record in built).items() if count > 1
    )
    if duplicate_output_ids:
        preview = ", ".join(duplicate_output_ids[:5])
        raise ValidationError(
            f"{len(duplicate_output_ids)} duplicate planned output id(s) were generated within this construct run. "
            f"Sample: {preview}. Deduplicate input.ids or route the colliding outputs into separate construct jobs."
        )
    existing_ids = _existing_output_ids(output_root, cfg.job.output.dataset)
    collision_count = sum(1 for record in built if record.output_id in existing_ids)
    if collision_count and cfg.job.output.on_conflict == "error":
        raise ValidationError(
            f"{collision_count} planned output id(s) already exist in dataset '{cfg.job.output.dataset}'. "
            "Choose a different output dataset, change the construct spec, or set output.on_conflict='ignore'."
        )
    planned_rows = [
        PlannedRow(
            input_id=str(row["id"]),
            output_id=record.output_id,
            anchor_length=int(record.metadata["construct__anchor_length"]),
            output_length=len(record.sequence),
            full_construct_length=int(record.metadata["construct__full_construct_length"]),
        )
        for row, record in zip(rows, built)
    ]
    preflight = PreflightResult(
        job_id=cfg.job.id,
        input_dataset=cfg.job.input.dataset,
        output_dataset=cfg.job.output.dataset,
        input_root=input_root,
        output_root=output_root,
        template_id=template.id,
        template_kind=template.kind,
        template_source=template.source,
        template_dataset=template.dataset,
        template_field=template.field,
        template_record_id=template.record_id,
        template_sha256=template_sha256,
        template_length=len(template.sequence),
        template_circular=bool(template.circular),
        realize_mode=cfg.job.realize.mode,
        focal_part=cfg.job.realize.focal_part,
        focal_point=cfg.job.realize.focal_point,
        anchor_offset_bp=cfg.job.realize.anchor_offset_bp,
        window_bp=cfg.job.realize.window_bp,
        spec_id=spec_id,
        records_total=len(built),
        existing_output_collisions=collision_count,
        output_on_conflict=cfg.job.output.on_conflict,
        placements=_planned_placements(ordered_parts),
        planned_rows=planned_rows,
    )
    return preflight, built


def preflight_from_config(path: str | Path) -> PreflightResult:
    preflight, _ = _plan_from_config(path)
    return preflight


def run_from_config(path: str | Path, *, dry_run: bool = False) -> RunResult:
    cfg, _ = load_job_config(path)
    preflight, built = _plan_from_config(path)

    if dry_run:
        return RunResult(
            job_id=cfg.job.id,
            input_dataset=cfg.job.input.dataset,
            output_dataset=cfg.job.output.dataset,
            output_root=preflight.output_root,
            records_total=preflight.records_total,
            records_written=0,
            records_skipped_existing=preflight.existing_output_collisions,
            spec_id=preflight.spec_id,
            dry_run=True,
        )

    _ensure_construct_registry(preflight.output_root)
    output_ds = Dataset(preflight.output_root, cfg.job.output.dataset)
    if not output_ds.records_path.exists():
        output_ds.init(source="construct", notes=f"Initialized by construct job {cfg.job.id}.")

    existing_ids = _existing_output_ids(preflight.output_root, cfg.job.output.dataset)
    built_to_write = [
        record for record in built if cfg.job.output.on_conflict != "ignore" or record.output_id not in existing_ids
    ]

    base_rows = [
        {
            "sequence": record.sequence,
            "bio_type": "dna",
            "alphabet": record.alphabet,
            "source": cfg.job.output.source or f"construct run {cfg.job.id}",
        }
        for record in built_to_write
    ]
    if base_rows:
        output_ds.import_rows(
            base_rows,
            default_bio_type="dna",
            source=cfg.job.output.source or f"construct run {cfg.job.id}",
        )
        _attach_construct_metadata(output_ds, [record.metadata for record in built_to_write])

    return RunResult(
        job_id=cfg.job.id,
        input_dataset=cfg.job.input.dataset,
        output_dataset=cfg.job.output.dataset,
        output_root=preflight.output_root,
        records_total=preflight.records_total,
        records_written=len(built_to_write),
        records_skipped_existing=preflight.records_total - len(built_to_write),
        spec_id=preflight.spec_id,
        dry_run=False,
    )
