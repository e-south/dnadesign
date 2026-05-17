"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/sequence/msd_design_reference_v1.py

Retron MSD design reference contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_DNA4_RE = re.compile(r"^[ACGT]{4}$")
_PROFILE_RE = re.compile(r"^[MWX]{4}$")
_WC_PAIRS = {("A", "T"), ("T", "A"), ("C", "G"), ("G", "C")}
_WOBBLE_PAIRS = {("G", "T"), ("T", "G")}


class MsdDesignContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _not_blank(value: str, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} cannot be empty.")
    return text


def _dna4(value: str, *, label: str) -> str:
    text = _not_blank(value, label=label).upper()
    if not _DNA4_RE.fullmatch(text):
        raise ValueError(f"{label} must contain exactly four A/C/G/T bases.")
    return text


def _profile(value: str, *, label: str) -> str:
    text = _not_blank(value, label=label).upper()
    if not _PROFILE_RE.fullmatch(text):
        raise ValueError(f"{label} must contain exactly four M/W/X symbols.")
    return text


def classify_scar_nick_pair(left_nt: str, right_nt: str) -> Literal["M", "W", "X"]:
    pair = (left_nt.upper(), right_nt.upper())
    if pair in _WC_PAIRS:
        return "M"
    if pair in _WOBBLE_PAIRS:
        return "W"
    return "X"


def compute_scar_nick_profile_s3s2s1s0(*, left_base: str, right_base: str) -> str:
    left = _dna4(left_base, label="left_base")
    right = _dna4(right_base, label="right_base")
    return "".join(
        classify_scar_nick_pair(left_nt, right_nt)
        for left_nt, right_nt in (
            (left[0], right[3]),
            (left[1], right[2]),
            (left[2], right[1]),
            (left[3], right[0]),
        )
    )


class MsdPayloadOrTargetV1(MsdDesignContractModel):
    id: str
    display_name: str | None = None

    @field_validator("id", "display_name")
    @classmethod
    def _optional_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _not_blank(value, label="payload_or_target field")


class MsdCapTopologySpanV1(MsdDesignContractModel):
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "MsdCapTopologySpanV1":
        if self.end <= self.start:
            raise ValueError("snapback topology span end must be > start.")
        return self


class MsdCapSnapbackTopologyV1(MsdDesignContractModel):
    kind: Literal["snapback_foldback_geometry_v1"] = "snapback_foldback_geometry_v1"
    retained_stem_span: MsdCapTopologySpanV1
    cap_span: MsdCapTopologySpanV1
    foldback_return_span: MsdCapTopologySpanV1
    source: str | None = None

    @field_validator("source")
    @classmethod
    def _optional_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _not_blank(value, label="snapback topology source")

    @model_validator(mode="after")
    def _validate_foldback_geometry(self) -> "MsdCapSnapbackTopologyV1":
        if self.retained_stem_span.start != 0:
            raise ValueError("snapback topology retained_stem_span.start must be 0.")
        if self.retained_stem_span.end != self.cap_span.start:
            raise ValueError("snapback topology retained_stem_span.end must equal cap_span.start.")
        if self.cap_span.end != self.foldback_return_span.start:
            raise ValueError("snapback topology cap_span.end must equal foldback_return_span.start.")
        if self.cap_span.end - self.cap_span.start != 3:
            raise ValueError("snapback topology cap_span must be exactly 3 nt.")
        return self


class MsdCapReferenceV1(MsdDesignContractModel):
    id: str
    source_construct: str | None = None
    display_name: str | None = None
    snapback_topology: MsdCapSnapbackTopologyV1 | None = None

    @field_validator("id", "source_construct", "display_name")
    @classmethod
    def _optional_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _not_blank(value, label="cap field")


class MsdScarNickReferenceV1(MsdDesignContractModel):
    left_base: str
    right_base: str
    profile_s3s2s1s0: str
    s0_match_required: bool = True
    route_status: Literal["resolved", "note_only", "unresolved"] = "unresolved"
    nick_orientation: Literal["top", "bottom"] | None = None
    nickase: str | None = None
    route_note: str | None = None

    @field_validator("left_base", "right_base")
    @classmethod
    def _base_is_dna4(cls, value: str) -> str:
        return _dna4(value, label="scar_nick base")

    @field_validator("profile_s3s2s1s0")
    @classmethod
    def _profile_is_valid(cls, value: str) -> str:
        return _profile(value, label="profile_s3s2s1s0")

    @field_validator("nickase", "route_note")
    @classmethod
    def _optional_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _not_blank(value, label="scar_nick route field")

    @model_validator(mode="after")
    def _validate_profile(self) -> "MsdScarNickReferenceV1":
        observed = compute_scar_nick_profile_s3s2s1s0(left_base=self.left_base, right_base=self.right_base)
        if observed != self.profile_s3s2s1s0:
            raise ValueError(
                f"profile_s3s2s1s0 does not match left/right bases: observed {observed}, "
                f"declared {self.profile_s3s2s1s0}."
            )
        if self.s0_match_required and self.profile_s3s2s1s0[3] != "M":
            raise ValueError("profile_s3s2s1s0 must have S0=M when s0_match_required=true.")
        if self.route_status == "resolved" and (self.nick_orientation is None or self.nickase is None):
            raise ValueError("resolved scar_nick route_status requires nick_orientation and nickase.")
        return self


class MsdSequenceSummaryV1(MsdDesignContractModel):
    length: int | None = Field(default=None, ge=0)
    sha256: str | None = None

    @field_validator("sha256")
    @classmethod
    def _sha_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _not_blank(value, label="sequence.sha256")


class MsdDesignSourceV1(MsdDesignContractModel):
    dnadesign_bundle: str | None = None
    composition_id: str | None = None

    @field_validator("dnadesign_bundle", "composition_id")
    @classmethod
    def _optional_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _not_blank(value, label="source field")


class MsdDesignArtifactsV1(MsdDesignContractModel):
    composition_overview_svg: str | None = None
    composition_overview_png: str | None = None
    secondary_structure_native_png: str | None = None
    secondary_structure_svg: str | None = None
    component_span_png: str | None = None
    component_span_svg: str | None = None
    features_csv: str | None = None
    visual_contract: str | None = None
    genbank: str | None = None
    reverse_complement_genbank: str | None = None
    forward_fasta: str | None = None
    reverse_complement_fasta: str | None = None
    folding_prediction: str | None = None
    folding_png: str | None = None
    combined_plot_png: str | None = None

    @field_validator(
        "composition_overview_svg",
        "composition_overview_png",
        "secondary_structure_native_png",
        "secondary_structure_svg",
        "component_span_png",
        "component_span_svg",
        "combined_plot_png",
        "features_csv",
        "folding_png",
        "folding_prediction",
        "visual_contract",
        "genbank",
        "reverse_complement_genbank",
        "forward_fasta",
        "reverse_complement_fasta",
    )
    @classmethod
    def _optional_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _not_blank(value, label="artifact path")


class MsdDesignReferenceV1(MsdDesignContractModel):
    contract: Literal["msd_design_reference_v1"] = "msd_design_reference_v1"
    schema_version: Literal[1] = 1
    construct_id: str
    construct_label: str
    msd_design_id: str
    design_family: Literal["retron_msd"] = "retron_msd"
    payload_or_target: MsdPayloadOrTargetV1
    cap: MsdCapReferenceV1
    scar_nick: MsdScarNickReferenceV1
    source_notes: str | None = None
    sequence: MsdSequenceSummaryV1 = Field(default_factory=MsdSequenceSummaryV1)
    source: MsdDesignSourceV1 = Field(default_factory=MsdDesignSourceV1)
    artifacts: MsdDesignArtifactsV1 = Field(default_factory=MsdDesignArtifactsV1)

    @field_validator("construct_id", "construct_label", "msd_design_id", "source_notes")
    @classmethod
    def _optional_not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _not_blank(value, label="msd design reference field")


class MsdDesignCatalogV1(MsdDesignContractModel):
    contract: Literal["msd_design_catalog_v1"] = "msd_design_catalog_v1"
    schema_version: Literal[1] = 1
    records: list[MsdDesignReferenceV1] = Field(min_length=1)

    @model_validator(mode="after")
    def _validate_unique_ids(self) -> "MsdDesignCatalogV1":
        design_ids: set[str] = set()
        construct_ids: set[str] = set()
        for record in self.records:
            if record.msd_design_id in design_ids:
                raise ValueError(f"Duplicate msd_design_id '{record.msd_design_id}'.")
            if record.construct_id in construct_ids:
                raise ValueError(f"Duplicate construct_id '{record.construct_id}'.")
            design_ids.add(record.msd_design_id)
            construct_ids.add(record.construct_id)
        return self


__all__ = [
    "MsdDesignCatalogV1",
    "MsdDesignReferenceV1",
    "compute_scar_nick_profile_s3s2s1s0",
    "classify_scar_nick_pair",
]
