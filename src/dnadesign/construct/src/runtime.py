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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from dnadesign.usr import Dataset, compute_id, normalize_sequence

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
    {"name": "construct__template_source", "type": "string"},
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


@dataclass(frozen=True)
class RunResult:
    job_id: str
    input_dataset: str
    output_dataset: str
    output_root: Path
    records_total: int
    dry_run: bool


@dataclass(frozen=True)
class PlannedRow:
    input_id: str
    output_id: str
    anchor_length: int
    output_length: int
    full_construct_length: int


@dataclass(frozen=True)
class PreflightResult:
    job_id: str
    input_dataset: str
    output_dataset: str
    input_root: Path
    output_root: Path
    template_id: str
    template_source: str
    template_sha256: str
    template_length: int
    template_circular: bool
    spec_id: str
    records_total: int
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


def _default_usr_root() -> Path:
    import dnadesign.usr as usr_pkg

    return (Path(usr_pkg.__file__).resolve().parent / "datasets").resolve()


def _resolve_optional_path(base_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _resolve_usr_root(base_dir: Path, value: str | None) -> Path:
    path = _resolve_optional_path(base_dir, value)
    return path if path is not None else _default_usr_root()


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


def _load_template_sequence(base_dir: Path, cfg: JobConfig) -> tuple[str, str]:
    template = cfg.job.template
    if template.sequence is not None:
        seq = _ensure_dna_text(template.sequence, label="template.sequence")
        return seq, template.source or "template.sequence"

    path = _resolve_optional_path(base_dir, template.path)
    if path is None or not path.exists():
        raise ValidationError(f"Template path not found: {template.path}")
    raw = path.read_text(encoding="utf-8")
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if not lines:
        raise ValidationError(f"Template file is empty: {path}")
    if lines[0].startswith(">"):
        seq_lines = [line for line in lines if not line.startswith(">")]
        if not seq_lines:
            raise ValidationError(f"Template FASTA does not contain sequence lines: {path}")
        seq = "".join(seq_lines)
    else:
        seq = "".join(lines)
    return _ensure_dna_text(seq, label=f"template.path ({path})"), template.source or str(path)


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
    ordered = sorted(parts, key=lambda part: (part.placement.start, part.placement.end, part.name))
    prior_end = -1
    prior_name = None
    for part in ordered:
        start = part.placement.start
        end = part.placement.end
        if end > template_len:
            raise ValidationError(
                f"Part '{part.name}' placement end {end} exceeds template length {template_len}."
            )
        if start < prior_end:
            raise ValidationError(
                f"Part '{part.name}' overlaps prior placement '{prior_name}'. Placements must not overlap."
            )
        prior_end = end
        prior_name = part.name
    return ordered


def _assemble_full_construct(
    template_seq: str,
    parts: List[PartConfig],
    row: dict[str, object],
) -> tuple[str, Dict[str, _ResolvedPart]]:
    ordered = _validate_placements(len(template_seq), parts)
    cursor = 0
    out: list[str] = []
    out_len = 0
    realized: Dict[str, _ResolvedPart] = {}

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

        realized[part.name] = _ResolvedPart(
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
        cursor = part.placement.end

    out.append(template_seq[cursor:])
    return "".join(out), realized


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
        return seq, start_raw, end_raw
    if start_raw < 0 or end_raw > len(full_construct):
        raise ValidationError(
            "Requested window extends beyond the linear construct boundaries. "
            "Adjust window_bp, anchor_offset_bp, or choose a circular template."
        )
    return full_construct[start_raw:end_raw], start_raw, end_raw


def _spec_id(cfg: JobConfig, *, template_sha256: str) -> str:
    payload = {
        "job_id": cfg.job.id,
        "input": {
            "dataset": cfg.job.input.dataset,
            "field": cfg.job.input.field,
        },
        "template": {
            "id": cfg.job.template.id,
            "circular": cfg.job.template.circular,
            "source": cfg.job.template.source,
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
    }
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _registry_path(root: Path) -> Path:
    return root / "registry.yaml"


def _load_registry_payload(root: Path) -> dict:
    path = _registry_path(root)
    if not path.exists():
        return {"namespaces": {}}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    namespaces = data.get("namespaces") or {}
    if not isinstance(namespaces, dict):
        raise ValidationError(f"USR registry at {path} must contain a 'namespaces' mapping.")
    return {"namespaces": namespaces}


def _ensure_construct_registry(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    payload = _load_registry_payload(root)
    namespaces = payload["namespaces"]
    usr_state = namespaces.setdefault(
        "usr_state",
        {
            "owner": "usr",
            "description": "Reserved record-state overlay (masked/qc/split/lineage).",
            "columns": [],
        },
    )
    construct = namespaces.setdefault(
        "construct",
        {
            "owner": "construct",
            "description": "Construct lineage overlays for realized DNA sequences.",
            "columns": [],
        },
    )
    usr_state.setdefault("owner", "usr")
    usr_state.setdefault("description", "Reserved record-state overlay (masked/qc/split/lineage).")
    construct.setdefault("owner", "construct")
    construct.setdefault("description", "Construct lineage overlays for realized DNA sequences.")
    existing_usr_state = {col["name"] for col in usr_state.get("columns", [])}
    existing_construct = {col["name"] for col in construct.get("columns", [])}
    usr_state["columns"] = list(usr_state.get("columns", [])) + [
        col for col in _USR_STATE_COLUMNS if col["name"] not in existing_usr_state
    ]
    construct["columns"] = list(construct.get("columns", [])) + [
        col for col in _CONSTRUCT_COLUMNS if col["name"] not in existing_construct
    ]
    path = _registry_path(root)
    path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")


def _build_record(
    *,
    row: dict[str, object],
    cfg: JobConfig,
    template_seq: str,
    template_source: str,
    template_sha256: str,
    spec_id: str,
) -> _BuiltRecord:
    full_construct, realized_parts = _assemble_full_construct(template_seq, cfg.job.parts, row)
    output_sequence, window_start, window_end = _extract_output_sequence(
        full_construct=full_construct,
        realized_parts=realized_parts,
        cfg=cfg,
    )
    alphabet = _alphabet_for_sequence(output_sequence)
    sequence_norm = normalize_sequence(output_sequence, "dna", alphabet)
    output_id = compute_id("dna", sequence_norm)

    input_fields = [
        str(part.sequence.field)
        for part in cfg.job.parts
        if part.sequence.source == "input_field"
    ]
    ordered_parts = [realized_parts[part.name] for part in cfg.job.parts]
    metadata = {
        "id": output_id,
        "construct__job": cfg.job.id,
        "construct__spec_id": spec_id,
        "construct__template_id": cfg.job.template.id,
        "construct__template_source": template_source,
        "construct__template_sha256": template_sha256,
        "construct__template_length": len(template_seq),
        "construct__template_circular": bool(cfg.job.template.circular),
        "construct__input_dataset": cfg.job.input.dataset,
        "construct__input_field": cfg.job.input.field,
        "construct__input_fields": input_fields,
        "construct__anchor_id": str(row["id"]),
        "construct__anchor_length": len(str(row[cfg.job.input.field]).strip()),
        "construct__mode": cfg.job.realize.mode,
        "construct__focal_part": cfg.job.realize.focal_part,
        "construct__window_bp": cfg.job.realize.window_bp,
        "construct__window_start": window_start,
        "construct__window_end": window_end,
        "construct__full_construct_length": len(full_construct),
        "construct__part_count": len(ordered_parts),
        "construct__part_names": [part.name for part in ordered_parts],
        "construct__part_roles": [part.role for part in ordered_parts],
        "construct__part_kinds": [part.kind for part in ordered_parts],
        "construct__part_starts": [part.realized_start for part in ordered_parts],
        "construct__part_ends": [part.realized_end for part in ordered_parts],
        "construct__part_orientations": [part.orientation for part in ordered_parts],
        "construct__part_template_starts": [part.start for part in ordered_parts],
        "construct__part_template_ends": [part.end for part in ordered_parts],
    }
    return _BuiltRecord(output_id=output_id, sequence=output_sequence, alphabet=alphabet, metadata=metadata)


def _attach_construct_metadata(ds: Dataset, metadata_rows: List[dict[str, object]]) -> None:
    if not metadata_rows:
        return
    frame = pd.DataFrame(metadata_rows)
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "construct_attach.parquet"
        table = pa.Table.from_pandas(frame, preserve_index=False)
        pq.write_table(table, path)
        ds.attach(
            path,
            namespace="construct",
            key="id",
            key_col="id",
            columns=[col for col in frame.columns if col != "id"],
            allow_overwrite=False,
            note="dnadesign.construct lineage attach",
        )


def _plan_from_config(path: str | Path) -> tuple[PreflightResult, List[_BuiltRecord]]:
    cfg, config_path = load_job_config(path)
    base_dir = config_path.parent
    input_root = _resolve_usr_root(base_dir, cfg.job.input.root)
    output_root = _resolve_usr_root(base_dir, cfg.job.output.root or cfg.job.input.root)

    input_ds = Dataset(input_root, cfg.job.input.dataset)
    if not input_ds.records_path.exists():
        raise ValidationError(f"Input dataset not initialized: {input_ds.records_path}")

    template_seq, template_source = _load_template_sequence(base_dir, cfg)
    template_sha256 = hashlib.sha256(template_seq.encode("utf-8")).hexdigest()
    spec_id = _spec_id(cfg, template_sha256=template_sha256)

    rows = _scan_usr_rows(input_ds, columns=_input_fields(cfg), ids=cfg.job.input.ids)
    if not rows:
        raise ValidationError("Input selection resolved to zero rows.")

    built = [
        _build_record(
            row=row,
            cfg=cfg,
            template_seq=template_seq,
            template_source=template_source,
            template_sha256=template_sha256,
            spec_id=spec_id,
        )
        for row in rows
    ]
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
        template_id=cfg.job.template.id,
        template_source=template_source,
        template_sha256=template_sha256,
        template_length=len(template_seq),
        template_circular=bool(cfg.job.template.circular),
        spec_id=spec_id,
        records_total=len(built),
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
            dry_run=True,
        )

    _ensure_construct_registry(preflight.output_root)
    output_ds = Dataset(preflight.output_root, cfg.job.output.dataset)
    if not output_ds.records_path.exists():
        output_ds.init(source="construct", notes=f"Initialized by construct job {cfg.job.id}.")

    base_rows = [
        {
            "sequence": record.sequence,
            "bio_type": "dna",
            "alphabet": record.alphabet,
            "source": cfg.job.output.source or f"construct run {cfg.job.id}",
        }
        for record in built
    ]
    output_ds.import_rows(
        base_rows,
        default_bio_type="dna",
        source=cfg.job.output.source or f"construct run {cfg.job.id}",
    )
    _attach_construct_metadata(output_ds, [record.metadata for record in built])

    return RunResult(
        job_id=cfg.job.id,
        input_dataset=cfg.job.input.dataset,
        output_dataset=cfg.job.output.dataset,
        output_root=preflight.output_root,
        records_total=preflight.records_total,
        dry_run=False,
    )
