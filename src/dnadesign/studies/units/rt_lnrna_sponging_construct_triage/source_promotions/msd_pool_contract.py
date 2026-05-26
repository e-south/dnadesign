"""
Typed MSD variant-pool contract for RT-lnRNA source promotions.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from dnadesign.studies.units.retron_hairpin_design.catalog.strict_mapping_io import (
    DuplicateMappingKeyError,
    load_unique_yaml,
)

from .contracts import SourcePromotionContractError


class MsdPoolContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TemplateLnrnaSpec(MsdPoolContractModel):
    sequence_ref: str
    genbank_path: str
    sequence_span_0: tuple[int, int]

    @field_validator("sequence_ref", "genbank_path")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        return _not_blank(value)

    @field_validator("sequence_span_0")
    @classmethod
    def _valid_span(cls, value: tuple[int, int]) -> tuple[int, int]:
        start, end = value
        if start < 0 or end <= start:
            raise ValueError("sequence_span_0 must be a zero-based half-open span.")
        return value


class TemplateMsdDesignSpec(MsdPoolContractModel):
    construct_id: str
    payload_id: str
    cap_id: str
    left_base: str
    right_base: str
    profile_s3s2s1s0: str | None = None
    source_notes: str | None = None

    @field_validator(
        "construct_id",
        "payload_id",
        "cap_id",
        "left_base",
        "right_base",
        "profile_s3s2s1s0",
        "source_notes",
    )
    @classmethod
    def _optional_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _not_blank(value)


class PlacementSpec(MsdPoolContractModel):
    expected_5p_flank: str
    expected_3p_flank: str

    @field_validator("expected_5p_flank", "expected_3p_flank")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        return _not_blank(value)


class CompilerInputsSpec(MsdPoolContractModel):
    payload_sequences: dict[str, Mapping[str, Any]]
    cap_sequences: dict[str, Mapping[str, Any]]

    @field_validator("payload_sequences", "cap_sequences")
    @classmethod
    def _not_empty_mapping(cls, value: dict[str, Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
        if not value:
            raise ValueError("compiler sequence maps must not be empty.")
        for key, item in value.items():
            _not_blank(str(key))
            if not isinstance(item, Mapping):
                raise ValueError(f"{key} must map to a sequence input mapping.")
        return value


class DesignSpaceStemBaseSpec(MsdPoolContractModel):
    stem_base_id: str
    left_base: str
    right_base: str
    profile_s3s2s1s0: str | None = None
    source_ref: str | None = None
    nick_orientation: Literal["top", "bottom"] | None = None
    nickase: str | None = None

    @field_validator("stem_base_id", "left_base", "right_base", "profile_s3s2s1s0", "source_ref", "nickase")
    @classmethod
    def _optional_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _not_blank(value)

    @field_validator("left_base", "right_base", mode="after")
    @classmethod
    def _uppercase_base(cls, value: str) -> str:
        return value.upper()

    @model_validator(mode="after")
    def _validate_literal_provenance(self) -> "DesignSpaceStemBaseSpec":
        provenance_fields = (self.source_ref, self.nick_orientation, self.nickase)
        if any(value is not None for value in provenance_fields) and not all(
            value is not None for value in provenance_fields
        ):
            raise ValueError(
                "literal stem-base provenance requires source_ref, nick_orientation, and nickase together."
            )
        return self

    def compiler_design_fields(self) -> dict[str, Any]:
        fields: dict[str, Any] = {
            "left_base": self.left_base,
            "right_base": self.right_base,
            "profile_s3s2s1s0": self.profile_s3s2s1s0,
            "source_notes": self.source_ref,
        }
        if self.source_ref is not None:
            fields.update(
                {
                    "literal_stem_base_source_id": self.source_ref,
                    "nick_orientation": self.nick_orientation,
                    "nickase": self.nickase,
                }
            )
        return {key: value for key, value in fields.items() if value is not None}


class DesignSpacePrimitiveRankedSourceSpec(MsdPoolContractModel):
    source_id: str
    run_dir: str
    ranks: list[int]
    expected_primitive_count: int | None = Field(default=None, ge=1)

    @field_validator("source_id", "run_dir")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        return _not_blank(value)

    @field_validator("ranks")
    @classmethod
    def _valid_ranks(cls, value: list[int]) -> list[int]:
        if not value:
            raise ValueError("primitive ranks must not be empty.")
        parsed: list[int] = []
        for rank in value:
            if isinstance(rank, bool) or int(rank) < 1:
                raise ValueError("primitive ranks must be positive integers.")
            parsed.append(int(rank))
        if len(set(parsed)) != len(parsed):
            raise ValueError("primitive ranks must be unique.")
        return parsed


class DesignSpaceCapPrimitiveSourceSpec(DesignSpacePrimitiveRankedSourceSpec):
    kind: Literal["snapback_released_solve_cap"]
    cap_id_prefix: str

    @field_validator("cap_id_prefix")
    @classmethod
    def _valid_cap_id_prefix(cls, value: str) -> str:
        text = _not_blank(value)
        if not text.startswith("C"):
            raise ValueError("cap_id_prefix must start with C to match Retron compiler cap IDs.")
        _validate_identifier_prefix(text, label="cap_id_prefix")
        return text


class DesignSpaceStemBasePrimitiveSourceSpec(DesignSpacePrimitiveRankedSourceSpec):
    kind: Literal["scar_nick_stem_bases"]
    stem_base_id_prefix: str

    @field_validator("stem_base_id_prefix")
    @classmethod
    def _valid_stem_base_id_prefix(cls, value: str) -> str:
        text = _not_blank(value)
        _validate_identifier_prefix(text, label="stem_base_id_prefix")
        return text


class DesignSpaceSpec(MsdPoolContractModel):
    construct_id_prefix: str
    payload_ids: list[str]
    cap_ids: list[str] = Field(default_factory=list)
    cap_primitives: list[DesignSpaceCapPrimitiveSourceSpec] = Field(default_factory=list)
    stem_bases: list[DesignSpaceStemBaseSpec] = Field(default_factory=list)
    stem_base_primitives: list[DesignSpaceStemBasePrimitiveSourceSpec] = Field(default_factory=list)

    @field_validator("construct_id_prefix")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        return _not_blank(value)

    @field_validator("payload_ids")
    @classmethod
    def _not_empty_str_list(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("design_space.payload_ids must not be empty.")
        return [_not_blank(item) for item in value]

    @field_validator("cap_ids")
    @classmethod
    def _str_list(cls, value: list[str]) -> list[str]:
        return [_not_blank(item) for item in value]

    @model_validator(mode="after")
    def _validate_design_sources(self) -> "DesignSpaceSpec":
        if not self.cap_ids and not self.cap_primitives:
            raise ValueError("design_space requires cap_ids or cap_primitives.")
        if not self.stem_bases and not self.stem_base_primitives:
            raise ValueError("design_space requires stem_bases or stem_base_primitives.")
        return self


class RtLnrnaMsdVariantPoolSpecV1(MsdPoolContractModel):
    contract: Literal["rt_lnrna_msd_variant_pool_spec_v1"]
    schema_version: Literal[1]
    pool_id: str
    study_id: Literal["rt_lnrna_sponging_construct_triage"]
    payload_program_id: str
    max_variant_count: int = Field(ge=1)
    expected_variant_count: int | None = Field(default=None, ge=1)
    dedupe_policy: Literal["fail"]
    template_lnrna: TemplateLnrnaSpec
    template_msd_design: TemplateMsdDesignSpec
    placement: PlacementSpec
    compiler_inputs: CompilerInputsSpec
    compiler_spec: dict[str, Any] | None = None
    design_space: DesignSpaceSpec | None = None
    allow_non_ligatable_s0: bool = False
    source_refs: list[str] = Field(default_factory=list)

    @field_validator("pool_id", "payload_program_id")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        return _not_blank(value)

    @field_validator("source_refs")
    @classmethod
    def _source_refs_not_blank(cls, value: list[str]) -> list[str]:
        return [_not_blank(item) for item in value]

    @model_validator(mode="after")
    def _validate_variant_source(self) -> "RtLnrnaMsdVariantPoolSpecV1":
        if (self.compiler_spec is None) == (self.design_space is None):
            raise ValueError("Pool spec requires exactly one of compiler_spec or design_space.")
        if self.expected_variant_count is not None and self.expected_variant_count > self.max_variant_count:
            raise ValueError("expected_variant_count must be <= max_variant_count.")
        return self


def load_msd_variant_pool_spec(path: Path) -> RtLnrnaMsdVariantPoolSpecV1:
    if not path.is_file():
        raise SourcePromotionContractError(f"MSD compiler pool spec is missing: {path}")
    try:
        payload = load_unique_yaml(path) or {}
    except DuplicateMappingKeyError as exc:
        raise SourcePromotionContractError(f"MSD compiler pool spec contains {exc}") from exc
    if not isinstance(payload, Mapping):
        raise SourcePromotionContractError(f"MSD compiler pool spec must be a mapping: {path}")
    try:
        return RtLnrnaMsdVariantPoolSpecV1.model_validate(payload)
    except ValidationError as exc:
        raise SourcePromotionContractError(_format_validation_error(exc)) from exc


def _format_validation_error(exc: ValidationError) -> str:
    messages = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error["loc"]) or "<root>"
        messages.append(f"{location}: {error['msg']}")
    return "MSD compiler pool spec is invalid: " + "; ".join(messages)


def _not_blank(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("field must be non-empty.")
    return text


def _validate_identifier_prefix(value: str, *, label: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise ValueError(f"{label} must contain only letters, digits, underscore, dot, or hyphen.")


__all__ = [
    "RtLnrnaMsdVariantPoolSpecV1",
    "load_msd_variant_pool_spec",
]
