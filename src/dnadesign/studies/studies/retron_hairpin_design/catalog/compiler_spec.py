"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/studies/retron_hairpin_design/catalog/compiler_spec.py

Typed Retron MSD compiler-spec boundary.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dnadesign.contracts.sequence import MsdDesignCatalogV1, MsdDesignReferenceV1

from .msd_ids import MsdDesignPartInput, parse_msd_construct_label, parse_msd_design_parts
from .registry import load_retron_msd_registry


class MsdCompilerSpecError(ValueError):
    """Raised when a Retron MSD compiler spec is missing required intent."""


class MsdCompilerSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RankedPrimitiveSelectorSpec(MsdCompilerSpecModel):
    mode: Literal["rank"] = "rank"
    rank: int = Field(ge=1)

    @model_validator(mode="after")
    def _validate_mode_fields(self) -> "RankedPrimitiveSelectorSpec":
        if self.mode != "rank":
            raise ValueError("selector supports only mode=rank.")
        return self

    def requested_ranks(self) -> list[int]:
        return [self.rank]


class SnapbackCapSourceSpec(MsdCompilerSpecModel):
    kind: Literal["snapback_released_solve_cap"]
    run_dir: Path
    selector: RankedPrimitiveSelectorSpec


class ScarNickStemBaseSourceSpec(MsdCompilerSpecModel):
    kind: Literal["scar_nick_stem_bases"]
    run_dir: Path
    selector: RankedPrimitiveSelectorSpec


class SequenceInputSpec(MsdCompilerSpecModel):
    sequence: str | None = None
    source: SnapbackCapSourceSpec | None = None

    @model_validator(mode="after")
    def _validate_sequence_or_source(self) -> "SequenceInputSpec":
        if self.sequence is None and self.source is None:
            raise ValueError("sequence input requires sequence or source.")
        if self.sequence is not None and self.source is not None:
            raise ValueError("sequence input must not mix literal sequence and source.")
        return self


class MsdDesignInputSpec(MsdCompilerSpecModel):
    construct_id: str
    payload_id: str
    cap_id: str
    left_base: str | None = None
    right_base: str | None = None
    profile_s3s2s1s0: str | None = None
    stem_base_source: ScarNickStemBaseSourceSpec | None = None
    source_notes: str | None = None

    @field_validator("construct_id", "payload_id", "cap_id", "source_notes")
    @classmethod
    def _optional_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            raise ValueError("MSD design fields cannot be blank.")
        return text

    @model_validator(mode="after")
    def _validate_base_source(self) -> "MsdDesignInputSpec":
        has_explicit_bases = self.left_base is not None or self.right_base is not None
        if has_explicit_bases and (self.left_base is None or self.right_base is None):
            raise ValueError("MSD design must provide both left_base and right_base, or neither.")
        if not has_explicit_bases and self.stem_base_source is None:
            raise ValueError("MSD design requires explicit left/right bases or stem_base_source.")
        return self


class RetronMsdCompilerSpecV1(MsdCompilerSpecModel):
    contract: Literal["retron_msd_compiler_spec_v1"]
    schema_version: Literal[1] = 1
    labels: list[str] = Field(default_factory=list)
    designs: list[MsdDesignInputSpec] = Field(default_factory=list)
    payload_sequences: dict[str, SequenceInputSpec] = Field(default_factory=dict)
    cap_sequences: dict[str, SequenceInputSpec] = Field(default_factory=dict)

    @field_validator("labels")
    @classmethod
    def _labels_are_not_blank(cls, value: list[str]) -> list[str]:
        labels = [str(item).strip() for item in value if str(item).strip()]
        if len(labels) != len(value):
            raise ValueError("labels must not contain blank entries.")
        return labels

    @field_validator("payload_sequences", "cap_sequences", mode="before")
    @classmethod
    def _coerce_sequence_inputs(cls, value: Any) -> Any:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("sequence maps must be mappings.")
        return {key: {"sequence": item} if isinstance(item, str) else item for key, item in value.items()}

    @model_validator(mode="after")
    def _has_design_inputs(self) -> "RetronMsdCompilerSpecV1":
        if not self.labels and not self.designs:
            raise ValueError("compiler spec requires labels or designs.")
        return self


@dataclass(frozen=True)
class ResolvedMsdCompilerSpec:
    spec_path: Path
    catalog: MsdDesignCatalogV1
    payload_sequences: dict[str, str]
    cap_sequences: dict[str, str]


def load_msd_compiler_spec(path: str | Path, *, study_dir: str | Path) -> ResolvedMsdCompilerSpec:
    spec_path = Path(path).expanduser().resolve()
    payload = _load_mapping(spec_path)
    spec = RetronMsdCompilerSpecV1.model_validate(payload)
    registry = load_retron_msd_registry(study_dir)

    cap_sequences, cap_metadata = _resolve_cap_sequences(spec.cap_sequences)
    payload_sequences = _resolve_literal_sequence_map(spec.payload_sequences, label="payload_sequences")
    records: list[MsdDesignReferenceV1] = []

    for label in spec.labels:
        records.append(registry.build_reference(parse_msd_construct_label(label)))
    for design in spec.designs:
        parts, scar_nick_metadata = _resolve_design_parts(design)
        parsed = parse_msd_design_parts(parts)
        records.append(
            registry.build_reference_from_parts(
                parsed,
                cap_metadata=cap_metadata.get(parts.cap_id),
                scar_nick_metadata=scar_nick_metadata,
                source_notes=design.source_notes,
            )
        )

    _reject_duplicate_design_ids(records)
    return ResolvedMsdCompilerSpec(
        spec_path=spec_path,
        catalog=MsdDesignCatalogV1(records=records),
        payload_sequences=payload_sequences,
        cap_sequences=cap_sequences,
    )


def _load_mapping(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise MsdCompilerSpecError(f"Retron MSD compiler spec not found: {path}")
    try:
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, yaml.YAMLError) as exc:
        raise MsdCompilerSpecError(f"Retron MSD compiler spec is invalid: {path}") from exc
    if not isinstance(payload, dict):
        raise MsdCompilerSpecError(f"Retron MSD compiler spec must be a mapping: {path}")
    return payload


def _resolve_literal_sequence_map(values: dict[str, SequenceInputSpec], *, label: str) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for key, entry in values.items():
        if not isinstance(key, str) or not key.strip():
            raise MsdCompilerSpecError(f"{label} contains a blank key.")
        if entry.source is not None:
            raise MsdCompilerSpecError(f"{label}.{key} does not support primitive sources; provide a literal sequence.")
        if entry.sequence is None:
            raise MsdCompilerSpecError(f"{label}.{key} requires a literal sequence.")
        resolved[key.strip()] = _dna_sequence(entry.sequence, label=f"{label}.{key}.sequence")
    return resolved


def _resolve_cap_sequences(
    values: dict[str, SequenceInputSpec],
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    sequences: dict[str, str] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for key, entry in values.items():
        if not isinstance(key, str) or not key.strip():
            raise MsdCompilerSpecError("cap_sequences contains a blank key.")
        cap_id = key.strip()
        if entry.sequence is not None:
            sequences[cap_id] = _dna_sequence(entry.sequence, label=f"cap_sequences.{cap_id}.sequence")
            continue
        if entry.source is None:
            raise MsdCompilerSpecError(f"cap_sequences.{cap_id} requires sequence or source.")
        primitive = _select_single_snapback_cap(entry.source, label=f"cap_sequences.{cap_id}.source")
        sequences[cap_id] = primitive.sequence
        metadata[cap_id] = {
            "source_construct": primitive.primitive_id,
            "display_name": f"{cap_id} {primitive.primitive_id}",
            "snapback_topology": primitive.snapback_topology,
        }
    return sequences, metadata


def _resolve_design_parts(design: MsdDesignInputSpec) -> tuple[MsdDesignPartInput, dict[str, Any] | None]:
    left_base = design.left_base
    right_base = design.right_base
    scar_nick_metadata: dict[str, Any] | None = None
    if design.stem_base_source is not None:
        primitive = _select_single_stem_base(
            design.stem_base_source,
            label=f"designs.{design.construct_id}.stem_base_source",
        )
        if left_base is not None and left_base.upper() != primitive.left_base:
            raise MsdCompilerSpecError(
                f"Design {design.construct_id} left_base {left_base.upper()} does not match selected "
                f"scar_nick primitive {primitive.primitive_id} left_base {primitive.left_base}."
            )
        if right_base is not None and right_base.upper() != primitive.right_base:
            raise MsdCompilerSpecError(
                f"Design {design.construct_id} right_base {right_base.upper()} does not match selected "
                f"scar_nick primitive {primitive.primitive_id} right_base {primitive.right_base}."
            )
        left_base = primitive.left_base
        right_base = primitive.right_base
        if primitive.nicked_strand not in {"top", "bottom"}:
            raise MsdCompilerSpecError(
                f"Selected scar_nick primitive {primitive.primitive_id} must expose nicked_strand top or bottom."
            )
        scar_nick_metadata = {
            "route_status": "resolved",
            "nick_orientation": primitive.nicked_strand,
            "nickase": primitive.nickase_variant_id,
            "route_note": f"scar_nick primitive {primitive.primitive_id} from {primitive.source_table}",
        }
    if left_base is None or right_base is None:
        raise MsdCompilerSpecError(f"Design {design.construct_id} requires resolved left/right bases.")
    return (
        MsdDesignPartInput(
            construct_id=design.construct_id,
            payload_id=design.payload_id,
            cap_id=design.cap_id,
            left_base=left_base,
            right_base=right_base,
            profile_s3s2s1s0=design.profile_s3s2s1s0,
        ),
        scar_nick_metadata,
    )


def _select_single_snapback_cap(source: SnapbackCapSourceSpec, *, label: str):
    from dnadesign.cruncher.snapback import load_released_solve_cap_primitives

    primitives = load_released_solve_cap_primitives(source.run_dir)
    selected = _select_ranked(primitives, selector=source.selector, label=label)
    if len(selected) != 1:
        raise MsdCompilerSpecError(
            f"{label} selected {len(selected)} Snapback foldback primitives. "
            "Use selector mode=rank for the preferred explicit combination; no implicit combinatoric expansion is run."
        )
    return selected[0]


def _select_single_stem_base(source: ScarNickStemBaseSourceSpec, *, label: str):
    from dnadesign.cruncher.scar_nick import load_scar_nick_stem_base_primitives

    primitives = load_scar_nick_stem_base_primitives(source.run_dir)
    selected = _select_ranked(primitives, selector=source.selector, label=label)
    if len(selected) != 1:
        raise MsdCompilerSpecError(
            f"{label} selected {len(selected)} scar_nick stem-base primitives. "
            "Use selector mode=rank for the preferred explicit combination; no implicit combinatoric expansion is run."
        )
    return selected[0]


def _select_ranked(primitives: list[Any], *, selector: RankedPrimitiveSelectorSpec, label: str) -> list[Any]:
    by_rank: dict[int, Any] = {}
    for primitive in primitives:
        rank = int(primitive.rank)
        if rank in by_rank:
            raise MsdCompilerSpecError(f"{label} has duplicate primitive rank {rank}.")
        by_rank[rank] = primitive
    if not by_rank:
        raise MsdCompilerSpecError(f"{label} found no primitive options.")
    requested = selector.requested_ranks()
    missing = [rank for rank in requested if rank not in by_rank]
    if missing:
        raise MsdCompilerSpecError(f"{label} requested missing primitive rank(s): {', '.join(map(str, missing))}.")
    return [by_rank[rank] for rank in requested]


def _dna_sequence(value: str, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise MsdCompilerSpecError(f"{label} cannot be empty.")
    invalid = sorted(set(text.upper()) - {"A", "C", "G", "T"})
    if invalid:
        raise MsdCompilerSpecError(f"{label} contains non-DNA bases: {''.join(invalid)}.")
    return text


def _reject_duplicate_design_ids(records: list[MsdDesignReferenceV1]) -> None:
    duplicate_ids = sorted(
        {
            record.msd_design_id
            for record in records
            if [candidate.msd_design_id for candidate in records].count(record.msd_design_id) > 1
        }
    )
    if duplicate_ids:
        raise MsdCompilerSpecError(f"Compiler spec emits duplicate MSD design id(s): {', '.join(duplicate_ids)}")


__all__ = [
    "MsdCompilerSpecError",
    "ResolvedMsdCompilerSpec",
    "RetronMsdCompilerSpecV1",
    "load_msd_compiler_spec",
]
