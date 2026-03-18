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
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from dnadesign.usr import Dataset, compute_id, default_usr_root, normalize_sequence, normalize_usr_root

from .config import JobConfig, PartConfig, WindowConfig, load_job_config
from .errors import ValidationError
from .output_store import _construct_metadata_table, _ensure_construct_registry, _existing_output_ids, _usr_label_table

_DNA_COMPLEMENT = str.maketrans("ACGTNacgtn", "TGCANtgcan")


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
    input_length: int
    focal_part_length: int | None
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
    window_semantics: str | None
    window_reference: str | None
    window_direction: str | None
    window_size_bp: int | None
    window_upstream_bp: int | None
    window_downstream_bp: int | None
    window_offset_bp: int | None
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
    sequence_source: str
    sequence_field: str | None
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
    label_primary: str | None
    label_aliases: List[str]


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


@dataclass(frozen=True)
class _WindowGeometry:
    start_raw: int
    end_raw: int
    start: int
    end: int
    span_bp: int


@dataclass(frozen=True)
class _PlannedRun:
    cfg: JobConfig
    preflight: PreflightResult
    built: List[_BuiltRecord]


def _default_usr_root() -> Path:
    return default_usr_root()


def _resolve_optional_path(base_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _resolve_usr_root(base_dir: Path, value: str | None, *, label: str) -> Path:
    resolved = _resolve_optional_path(base_dir, value)
    if resolved is None:
        raise ValidationError(f"{label} is required for USR-backed construct jobs.")
    return normalize_usr_root(resolved)


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

    template_root = _resolve_usr_root(
        base_dir,
        template.root or cfg.job.input.root,
        label="template.root or job.input.root",
    )
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


def _input_scan_fields(ds: Dataset, cfg: JobConfig) -> List[str]:
    fields = set(_input_fields(cfg))
    available = set(ds.schema().names)
    if "usr_label__primary" in available:
        fields.add("usr_label__primary")
    if "usr_label__aliases" in available:
        fields.add("usr_label__aliases")
    return sorted(fields)


def _input_usr_labels(row: dict[str, object]) -> tuple[str | None, List[str]]:
    primary_raw = row.get("usr_label__primary")
    primary = str(primary_raw).strip() if primary_raw is not None and str(primary_raw).strip() else None

    aliases_raw = row.get("usr_label__aliases")
    aliases: list[str] = []
    if isinstance(aliases_raw, list):
        raw_values = aliases_raw
    elif aliases_raw is None:
        raw_values = []
    else:
        raw_values = [aliases_raw]
    for value in raw_values:
        text = str(value or "").strip()
        if not text or text == primary or text in aliases:
            continue
        aliases.append(text)
    return primary, aliases


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
            sequence_source=part.sequence.source,
            sequence_field=str(part.sequence.field) if part.sequence.field is not None else None,
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


def _window_reference_index(part: _ResolvedPart, *, reference: str) -> int:
    if reference == "start":
        return part.realized_start
    if reference == "end":
        return part.realized_end - 1
    return part.realized_start + (len(part.sequence) // 2)


def _orientation_step(*, orientation: str, direction: str) -> int:
    if direction == "five_prime":
        return -1 if orientation == "forward" else 1
    if direction == "three_prime":
        return 1 if orientation == "forward" else -1
    raise ValidationError(f"Unsupported window direction '{direction}'.")


def _window_raw_bounds(
    *,
    full_construct_length: int,
    focal: _ResolvedPart,
    window: WindowConfig,
) -> tuple[int, int]:
    if window.semantics == "fixed_total":
        window_bp = int(window.size_bp)
        if window_bp > full_construct_length:
            raise ValidationError(
                f"Requested fixed_total window size_bp={window_bp} exceeds realized construct length "
                f"{full_construct_length}."
            )
        if len(focal.sequence) > window_bp:
            raise ValidationError(
                f"Focal part '{focal.name}' length {len(focal.sequence)} exceeds "
                f"fixed_total window size_bp={window_bp}. "
                "Choose a larger fixed_total window or use anchor_plus_context semantics."
            )
        point = _window_reference_index(focal, reference=window.reference)
        offset_bp = int(window.offset_bp)
        if window.direction == "symmetric":
            start_raw = point - (window_bp // 2) + offset_bp
            return start_raw, start_raw + window_bp

        step = _orientation_step(orientation=focal.orientation, direction=window.direction)
        if step > 0:
            start_raw = point + offset_bp
            return start_raw, start_raw + window_bp

        end_raw = point + 1 + offset_bp
        return end_raw - window_bp, end_raw

    upstream_bp = int(window.upstream_bp)
    downstream_bp = int(window.downstream_bp)
    window_bp = len(focal.sequence) + upstream_bp + downstream_bp
    if window_bp > full_construct_length:
        raise ValidationError(
            f"Requested anchor_plus_context window length {window_bp} exceeds realized construct length "
            f"{full_construct_length}."
        )
    if focal.orientation == "forward":
        return focal.realized_start - upstream_bp, focal.realized_end + downstream_bp
    return focal.realized_start - downstream_bp, focal.realized_end + upstream_bp


def _normalize_window_geometry(
    *,
    full_construct_length: int,
    template_circular: bool,
    focal: _ResolvedPart,
    window: WindowConfig,
) -> _WindowGeometry:
    start_raw, end_raw = _window_raw_bounds(
        full_construct_length=full_construct_length,
        focal=focal,
        window=window,
    )
    span_bp = end_raw - start_raw
    if span_bp > full_construct_length:
        raise ValidationError(
            f"Requested window span {span_bp} exceeds realized construct length {full_construct_length}."
        )
    if template_circular:
        start = start_raw % full_construct_length
        end = (start + span_bp) % full_construct_length
        return _WindowGeometry(
            start_raw=start_raw,
            end_raw=end_raw,
            start=start,
            end=end,
            span_bp=span_bp,
        )
    if start_raw < 0 or end_raw > full_construct_length:
        raise ValidationError(
            "Requested window extends beyond the linear construct boundaries. "
            "Adjust the window settings or choose a circular template."
        )
    return _WindowGeometry(
        start_raw=start_raw,
        end_raw=end_raw,
        start=start_raw,
        end=end_raw,
        span_bp=span_bp,
    )


def _extract_output_sequence(
    *,
    full_construct: str,
    realized_parts: Dict[str, _ResolvedPart],
    cfg: JobConfig,
) -> tuple[str, int, int]:
    if cfg.job.realize.mode == "full_construct":
        return full_construct, 0, len(full_construct)

    window = cfg.job.realize.window
    if window is None:
        raise ValidationError("realize.window must resolve before runtime extraction.")
    focal = realized_parts[cfg.job.realize.focal_part]
    geometry = _normalize_window_geometry(
        full_construct_length=len(full_construct),
        template_circular=cfg.job.template.circular,
        focal=focal,
        window=window,
    )
    if cfg.job.template.circular:
        seq = "".join(
            full_construct[(geometry.start_raw + idx) % len(full_construct)] for idx in range(geometry.span_bp)
        )
        return seq, geometry.start, geometry.end
    return full_construct[geometry.start : geometry.end], geometry.start, geometry.end


def _resolve_anchor_part(
    *,
    realized_parts: Dict[str, _ResolvedPart],
    ordered_realized_parts: List[_ResolvedPart],
    focal_part_name: str | None,
) -> _ResolvedPart | None:
    if focal_part_name:
        candidate = realized_parts.get(focal_part_name)
        if candidate is not None:
            return candidate
    for part in ordered_realized_parts:
        if part.role == "anchor" or part.name == "anchor":
            return part
    return None


def _relative_anchor_bounds(
    *,
    anchor_part: _ResolvedPart | None,
    output_length: int,
    full_construct_length: int,
    window_start: int,
    mode: str,
) -> tuple[int | None, int | None]:
    if anchor_part is None:
        return None, None
    if mode == "full_construct":
        return anchor_part.realized_start, anchor_part.realized_end

    anchor_start = (anchor_part.realized_start - window_start) % full_construct_length
    anchor_end = anchor_start + len(anchor_part.sequence)
    if anchor_end > output_length:
        return None, None
    return anchor_start, anchor_end


def _spec_id(
    cfg: JobConfig,
    *,
    template: _ResolvedTemplate,
    template_sha256: str,
    input_root: Path,
    output_root: Path,
) -> str:
    window = cfg.job.realize.window
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
            "window": (
                {
                    "semantics": window.semantics,
                    "reference": window.reference,
                    "direction": window.direction,
                    "size_bp": window.size_bp,
                    "upstream_bp": window.upstream_bp,
                    "downstream_bp": window.downstream_bp,
                    "offset_bp": window.offset_bp,
                }
                if window is not None
                else None
            ),
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
    window = cfg.job.realize.window
    output_sequence, window_start, window_end = _extract_output_sequence(
        full_construct=full_construct,
        realized_parts=realized_parts,
        cfg=cfg,
    )
    alphabet = _alphabet_for_sequence(output_sequence)
    sequence_norm = normalize_sequence(output_sequence, "dna", alphabet)
    output_id = compute_id("dna", sequence_norm)
    label_primary, label_aliases = _input_usr_labels(row)

    input_fields = [field for field in _input_fields(cfg) if field != "id"]
    focal_part = realized_parts.get(cfg.job.realize.focal_part or "")
    anchor_part = _resolve_anchor_part(
        realized_parts=realized_parts,
        ordered_realized_parts=ordered_realized_parts,
        focal_part_name=cfg.job.realize.focal_part,
    )
    anchor_start, anchor_end = _relative_anchor_bounds(
        anchor_part=anchor_part,
        output_length=len(output_sequence),
        full_construct_length=len(full_construct),
        window_start=window_start,
        mode=cfg.job.realize.mode,
    )
    metadata = {
        "id": output_id,
        "construct__job": cfg.job.id,
        "construct__spec_id": spec_id,
        "construct__context_id": f"{cfg.job.id}:{template.id}",
        "construct__context_kind": "template",
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
        "construct__input_fields": input_fields,
        "construct__input_id": str(row["id"]),
        "construct__input_length": len(str(row[cfg.job.input.field]).strip()),
        "construct__anchor_id": str(row["id"]),
        "construct__anchor_orientation": anchor_part.orientation if anchor_part is not None else "",
        "construct__anchor_start": anchor_start,
        "construct__anchor_end": anchor_end,
        "construct__mode": cfg.job.realize.mode,
        "construct__focal_part": cfg.job.realize.focal_part or "",
        "construct__focal_part_length": len(focal_part.sequence) if focal_part is not None else None,
        "construct__window_semantics": window.semantics if window is not None else "",
        "construct__window_reference": window.reference if window is not None else "",
        "construct__window_direction": window.direction if window is not None else "",
        "construct__window_size_bp": int(window.size_bp) if window is not None and window.size_bp is not None else None,
        "construct__window_upstream_bp": (
            int(window.upstream_bp) if window is not None and window.upstream_bp is not None else None
        ),
        "construct__window_downstream_bp": (
            int(window.downstream_bp) if window is not None and window.downstream_bp is not None else None
        ),
        "construct__window_offset_bp": (
            int(window.offset_bp) if window is not None and window.semantics == "fixed_total" else None
        ),
        "construct__window_start": window_start,
        "construct__window_end": window_end,
        "construct__resolved_length": len(output_sequence),
        "construct__full_construct_length": len(full_construct),
        "construct__parts": [
            {
                "name": part.name,
                "role": part.role,
                "sequence_source": part.sequence_source,
                "sequence_field": part.sequence_field or "",
                "placement_kind": part.kind,
                "orientation": part.orientation,
                "template_start": part.start,
                "template_end": part.end,
                "realized_start": part.realized_start,
                "realized_end": part.realized_end,
                "length": len(part.sequence),
            }
            for part in ordered_realized_parts
        ],
    }
    return _BuiltRecord(
        output_id=output_id,
        sequence=output_sequence,
        alphabet=alphabet,
        metadata=metadata,
        label_primary=label_primary,
        label_aliases=label_aliases,
    )


def _plan_loaded_config(
    cfg: JobConfig,
    *,
    config_path: Path,
) -> tuple[PreflightResult, List[_BuiltRecord]]:
    base_dir = config_path.parent
    input_root = _resolve_usr_root(base_dir, cfg.job.input.root, label="job.input.root")
    output_root = _resolve_usr_root(
        base_dir,
        cfg.job.output.root or cfg.job.input.root,
        label="job.output.root or job.input.root",
    )

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

    rows = _scan_usr_rows(input_ds, columns=_input_scan_fields(input_ds, cfg), ids=cfg.job.input.ids)
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
            input_length=int(record.metadata["construct__input_length"]),
            focal_part_length=(
                int(record.metadata["construct__focal_part_length"])
                if record.metadata["construct__focal_part_length"] is not None
                else None
            ),
            output_length=len(record.sequence),
            full_construct_length=int(record.metadata["construct__full_construct_length"]),
        )
        for row, record in zip(rows, built)
    ]
    window = cfg.job.realize.window
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
        window_semantics=window.semantics if window is not None else None,
        window_reference=window.reference if window is not None else None,
        window_direction=window.direction if window is not None else None,
        window_size_bp=int(window.size_bp) if window is not None and window.size_bp is not None else None,
        window_upstream_bp=(int(window.upstream_bp) if window is not None and window.upstream_bp is not None else None),
        window_downstream_bp=(
            int(window.downstream_bp) if window is not None and window.downstream_bp is not None else None
        ),
        window_offset_bp=int(window.offset_bp) if window is not None else None,
        spec_id=spec_id,
        records_total=len(built),
        existing_output_collisions=collision_count,
        output_on_conflict=cfg.job.output.on_conflict,
        placements=_planned_placements(ordered_parts),
        planned_rows=planned_rows,
    )
    return preflight, built


def _planned_run_from_config(path: str | Path) -> _PlannedRun:
    cfg, config_path = load_job_config(path)
    preflight, built = _plan_loaded_config(cfg, config_path=config_path)
    return _PlannedRun(cfg=cfg, preflight=preflight, built=built)


def _plan_from_config(path: str | Path) -> tuple[PreflightResult, List[_BuiltRecord]]:
    planned = _planned_run_from_config(path)
    return planned.preflight, planned.built


def _dry_run_result(planned: _PlannedRun) -> RunResult:
    cfg = planned.cfg
    preflight = planned.preflight
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


def _ensure_output_dataset(planned: _PlannedRun) -> Dataset:
    cfg = planned.cfg
    preflight = planned.preflight
    _ensure_construct_registry(preflight.output_root)
    return Dataset(preflight.output_root, cfg.job.output.dataset)


def _records_to_write(planned: _PlannedRun) -> List[_BuiltRecord]:
    cfg = planned.cfg
    preflight = planned.preflight
    existing_ids = _existing_output_ids(preflight.output_root, cfg.job.output.dataset)
    return [
        record
        for record in planned.built
        if cfg.job.output.on_conflict != "ignore" or record.output_id not in existing_ids
    ]


def _write_output_records(output_ds: Dataset, *, cfg: JobConfig, records: List[_BuiltRecord]) -> None:
    with output_ds.write_session() as session:
        session.init_if_missing(source="construct", notes=f"Initialized by construct job {cfg.job.id}.")
        if not records:
            return
        source = cfg.job.output.source or f"construct run {cfg.job.id}"
        session.import_rows(
            [
                {
                    "sequence": record.sequence,
                    "bio_type": "dna",
                    "alphabet": record.alphabet,
                    "source": source,
                }
                for record in records
            ],
            default_bio_type="dna",
            source=source,
        )
        session.write_overlay(
            "construct",
            _construct_metadata_table([record.metadata for record in records]),
            key="id",
            overwrite=True,
            note="dnadesign.construct lineage attach",
        )
        label_rows = [
            {
                "id": record.output_id,
                "usr_label__primary": record.label_primary,
                "usr_label__aliases": record.label_aliases,
            }
            for record in records
            if record.label_primary is not None or record.label_aliases
        ]
        if label_rows:
            session.write_overlay(
                "usr_label",
                _usr_label_table(label_rows),
                overwrite=True,
                note="dnadesign.construct upstream label carry-through",
            )


def _persist_construct_run(planned: _PlannedRun) -> RunResult:
    cfg = planned.cfg
    preflight = planned.preflight
    output_ds = _ensure_output_dataset(planned)
    built_to_write = _records_to_write(planned)
    _write_output_records(output_ds, cfg=cfg, records=built_to_write)
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


def preflight_from_config(path: str | Path) -> PreflightResult:
    return _planned_run_from_config(path).preflight


def run_from_config(path: str | Path, *, dry_run: bool = False) -> RunResult:
    planned = _planned_run_from_config(path)
    if dry_run:
        return _dry_run_result(planned)
    return _persist_construct_run(planned)
