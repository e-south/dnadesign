"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/source_ingest/selected_lineage/contracts.py

Typed contracts for selected materialized MSD variant lineage.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
StemBase = Annotated[str, StringConstraints(pattern=r"^[ACGT]+$")]
ScarNickProfile = Annotated[str, StringConstraints(pattern=r"^[MWX]{4}$")]

_VARIANT_ID_RE = re.compile(r"^retron(?P<number>\d+)$")
_DISPLAY_ID_RE = re.compile(r"^pES-retron-(?P<number>\d+)$")
_SOURCE_RECORD_ID_RE = re.compile(r"^msd-retron-(?P<number>\d+)$")


class MaterializedVariantLineageError(ValueError):
    """Raised when a selected materialized-variant lineage is inconsistent."""


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MsdStructuralPrimitiveRefsV1(_StrictModel):
    """Stable identifiers for the payload and structural primitives used by one MSD."""

    scaffold_context_id: NonBlank
    payload_id: NonBlank
    cap_id: NonBlank
    cap_selector_id: NonBlank
    stem_base_selector_id: NonBlank
    left_stem_base_5to3: StemBase
    right_stem_base_5to3: StemBase
    scar_nick_profile_s3s2s1s0: ScarNickProfile
    literal_stem_base_source_id: NonBlank | None = None


class MaterializedVariantLineageEntryV1(_StrictModel):
    """Exact source-to-materialized-variant mapping for one MSD record."""

    variant_id: NonBlank
    display_id: NonBlank
    source_record_id: NonBlank
    design_set_ref: NonBlank
    compiler_spec_ref: NonBlank
    deliverable_plan_ref: NonBlank
    deliverable_variant_key: NonBlank
    source_construct_id: NonBlank
    source_msd_design_id: NonBlank
    source_precedent_id: NonBlank
    primitives: MsdStructuralPrimitiveRefsV1
    source_genbank_ref: NonBlank
    source_genbank_sha256: Sha256
    source_sequence_sha256: Sha256
    msd_region_record_ref: NonBlank
    msd_sequence_sha256: Sha256

    @model_validator(mode="after")
    def _identity_numbers_agree(self) -> "MaterializedVariantLineageEntryV1":
        variant = _matched_number(_VARIANT_ID_RE, self.variant_id, field="variant_id")
        display = _matched_number(_DISPLAY_ID_RE, self.display_id, field="display_id")
        source_record = _matched_number(_SOURCE_RECORD_ID_RE, self.source_record_id, field="source_record_id")
        if len({variant, display, source_record}) != 1:
            raise ValueError("variant_id, display_id, and source_record_id must encode the same retron number.")
        return self


class MaterializedVariantLineageV1(_StrictModel):
    """Hairpin-study projection from a selected cohort to source-owned records."""

    contract: Literal["retron_hairpin_materialized_variant_lineage_v1"]
    schema_version: Literal[1] = 1
    owner_study_id: Literal["retron_hairpin_design"]
    source_bundle_manifest_ref: NonBlank
    selected_variant_ids: tuple[NonBlank, ...]
    expected_selected_variant_count: int = Field(gt=0)
    entries: tuple[MaterializedVariantLineageEntryV1, ...]

    @model_validator(mode="after")
    def _selection_and_entries_are_complete_and_unique(self) -> "MaterializedVariantLineageV1":
        if len(self.selected_variant_ids) != self.expected_selected_variant_count:
            raise ValueError(
                "expected_selected_variant_count="
                f"{self.expected_selected_variant_count} but found {len(self.selected_variant_ids)} selected IDs."
            )
        if len(self.selected_variant_ids) != len(set(self.selected_variant_ids)):
            raise ValueError("selected_variant_ids contain duplicates.")
        if len(self.entries) != self.expected_selected_variant_count:
            raise ValueError(
                "expected_selected_variant_count="
                f"{self.expected_selected_variant_count} but found {len(self.entries)} entries."
            )
        for field in (
            "variant_id",
            "display_id",
            "source_record_id",
            "source_genbank_ref",
            "msd_region_record_ref",
        ):
            values = [getattr(entry, field) for entry in self.entries]
            if len(values) != len(set(values)):
                raise ValueError(f"entries contain duplicate {field} values.")
        entry_ids = {entry.variant_id for entry in self.entries}
        selected_ids = set(self.selected_variant_ids)
        if entry_ids != selected_ids:
            raise ValueError(
                "selected_variant_ids must exactly match entry variant IDs: "
                f"missing={sorted(selected_ids - entry_ids)}, unselected={sorted(entry_ids - selected_ids)}."
            )
        return self


def _matched_number(pattern: re.Pattern[str], value: str, *, field: str) -> str:
    match = pattern.fullmatch(value)
    if match is None:
        raise ValueError(f"{field} has invalid form: {value!r}.")
    return match.group("number")
