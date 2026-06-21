"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/catalog/specs/variant_metadata.py

Optional Retron MSD payload-trim and design-role metadata specs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RetronMsdMetadataSpecModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PayloadSequenceMetadataSpec(RetronMsdMetadataSpecModel):
    display_name: str | None = None
    parent_payload_id: str | None = None
    payload_trim_id: str | None = None
    trim_class: Literal["full", "conservative", "aggressive"] | None = None
    trim_5p_nt: int | None = Field(default=None, ge=0)
    trim_3p_nt: int | None = Field(default=None, ge=0)
    retained_parent_span_0: dict[str, int] | None = None
    pwm_source_ref: str | None = None
    information_content_parent: float | None = Field(default=None, ge=0)
    information_content_retained: float | None = Field(default=None, ge=0)
    retained_information_fraction: float | None = Field(default=None, ge=0, le=1)
    selection_basis: str | None = None
    protected_positions_or_reason: str | None = None

    @field_validator(
        "display_name",
        "parent_payload_id",
        "payload_trim_id",
        "pwm_source_ref",
        "selection_basis",
        "protected_positions_or_reason",
    )
    @classmethod
    def _optional_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            raise ValueError("payload sequence metadata fields cannot be blank.")
        return text

    def metadata_payload(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class DesignVariantMetadataSpec(RetronMsdMetadataSpecModel):
    _METADATA_FIELDS: ClassVar[tuple[str, ...]] = (
        "variant_role",
        "scaffold_context",
        "payload_trim_id",
        "cap_selector_id",
        "stem_base_selector_id",
        "rt_mode",
        "decision_group",
        "control_id",
        "rationale",
    )

    variant_role: Literal["control", "scaffold_target", "rescue_candidate"] | None = None
    scaffold_context: Literal["retron26", "retron43", "de033_selected"] | None = None
    payload_trim_id: str | None = None
    cap_selector_id: str | None = None
    stem_base_selector_id: str | None = None
    rt_mode: Literal["wt_eco1"] | None = None
    decision_group: str | None = None
    control_id: str | None = None
    rationale: str | None = None

    @field_validator(
        "payload_trim_id",
        "cap_selector_id",
        "stem_base_selector_id",
        "decision_group",
        "control_id",
        "rationale",
    )
    @classmethod
    def _optional_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            raise ValueError("design variant metadata fields cannot be blank.")
        return text

    def variant_metadata_payload(self) -> dict[str, Any] | None:
        payload = {field: getattr(self, field) for field in self._METADATA_FIELDS if getattr(self, field) is not None}
        return payload or None


__all__ = [
    "DesignVariantMetadataSpec",
    "PayloadSequenceMetadataSpec",
]
