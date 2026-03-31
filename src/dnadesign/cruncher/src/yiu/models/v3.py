"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/models/v3.py

Canonical YIU v3 schema for the ship-ready circularized workflow.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from dnadesign.cruncher.bio.iupac import normalize_iupac, reverse_complement_iupac
from dnadesign.cruncher.config.schema_v3 import StrictBaseModel
from dnadesign.cruncher.yiu.models.catalogs import OutputSpecV2
from dnadesign.cruncher.yiu.models.common import _validate_slug
from dnadesign.cruncher.yiu.models.v2_steps import CatalogRefsV2

YIU_V3_SOURCE_OWNER_IDS: tuple[str, ...] = (
    "source_fwd_primer_binding_region",
    "payload_left_half",
    "sacrificial_region_long",
    "tether_dock_complement",
    "tether_cap",
    "tether_dock",
    "snapback_stem",
    "payload_right_half",
    "source_rev_primer_binding_region",
)

YIU_V3_OPERATIONAL_OWNER_IDS: tuple[str, ...] = (
    *YIU_V3_SOURCE_OWNER_IDS,
    "retained_region",
    "sacrificial_region_short",
    "y_adapter_complementary_arm",
    "y_adapter_noncomplementary_arm",
)

YIU_V3_EFFECT_TAG_CLASSES: frozenset[str] = frozenset(
    {
        "type_iis_recognition_left",
        "type_iis_recognition_right",
        "payload_overhang_left",
        "payload_overhang_right",
        "primer_bindable_by_source_forward",
        "primer_bindable_by_source_reverse",
        "primer_bindable_by_hairpin_pcr_forward",
        "primer_bindable_by_hairpin_pcr_reverse",
        "hairpin_pcr_forward_binding_region",
        "hairpin_pcr_reverse_binding_region",
        "pairs_with_tether_dock",
        "pairs_with_tether_dock_complement",
        "nt_bpu10i_snapback_site",
        "nb_bsssi_array_member",
        "retained",
        "sacrificial",
        "introduced_late",
        "y_adapter_binding",
        "payload_bulge_position",
        "payload_bulge_partner",
        "left_payload_overlap_unit",
        "bsssi_site_member_left",
        "bsai_site_member",
        "bsssi_site_member_right",
    }
)

YIU_V3_STATE_IDS: tuple[str, ...] = (
    "source_oligo_ssdna",
    "pcr_linear_duplex",
    "type_iis_digest_linear_duplex",
    "circularized_payload_candidate",
    "post_exonuclease_cleanup",
    "post_sacrificial_fragmentation",
    "post_fragment_cleanup",
    "snapback_adapter_complex",
    "ligated_ssdna_hairpin",
    "hairpin_pcr_linear_insert",
)

YIU_V3_ALLOWED_SOLVE_WINDOW_OWNER_IDS: frozenset[str] = frozenset({"payload_left_half", "payload_right_half"})
YIU_V3_ALLOWED_SOLVE_WINDOW_TAG_CLASSES: frozenset[str] = frozenset({"sacrificial"})

_PAYLOAD_ASSEMBLY_REQUIRED_STATES = (
    "circularized_payload_candidate",
    "post_exonuclease_cleanup",
    "post_fragment_cleanup",
    "ligated_ssdna_hairpin",
    "hairpin_pcr_linear_insert",
)
_NT_BPU10I_REQUIRED_STATES = (
    "post_fragment_cleanup",
    "snapback_adapter_complex",
    "ligated_ssdna_hairpin",
    "hairpin_pcr_linear_insert",
)
_SACRIFICIAL_FRAGMENTATION_REQUIRED_STATES = (
    "post_sacrificial_fragmentation",
    "post_fragment_cleanup",
)

_LEFT_PAYLOAD_OVERLAP_SEQUENCE = "CACGAGAGGTCTCACGAG"
_TYPE_IIS_LEFT_SEQUENCE = "GGTCTC"
_TYPE_IIS_RIGHT_SEQUENCE = "GAGACC"
_BSSSI_MEMBER_SEQUENCE = "CACGAG"
_NT_BPU10I_LOCAL_CONTEXT = "CCTCAGCCCGCTGA"
_NT_BPU10I_RECOGNITION_SEQUENCE = "CCTCAGC"

_OVERLAP_POLICIES: dict[tuple[str, str, str, str], str] = {
    (
        "structural_owner",
        "source_fwd_primer_binding_region",
        "effect_tag",
        "primer_bindable_by_source_forward",
    ): "allowed_if_identical_span",
    (
        "structural_owner",
        "source_rev_primer_binding_region",
        "effect_tag",
        "primer_bindable_by_source_reverse",
    ): "allowed_if_identical_span",
    (
        "structural_owner",
        "source_rev_primer_binding_region",
        "effect_tag",
        "type_iis_recognition_right",
    ): "allowed_if_identical_span",
    (
        "structural_owner",
        "source_fwd_primer_binding_region",
        "effect_tag",
        "bsssi_site_member_left",
    ): "allowed_if_identical_span",
    (
        "structural_owner",
        "source_fwd_primer_binding_region",
        "effect_tag",
        "left_payload_overlap_unit",
    ): "allowed_if_nested",
    ("structural_owner", "payload_left_half", "effect_tag", "payload_overhang_left"): "allowed_if_nested_exact",
    ("structural_owner", "payload_right_half", "effect_tag", "payload_overhang_right"): "allowed_if_nested_exact",
    ("structural_owner", "payload_left_half", "effect_tag", "type_iis_recognition_left"): "allowed_if_nested",
    ("structural_owner", "payload_left_half", "effect_tag", "bsai_site_member"): "allowed_if_nested",
    ("structural_owner", "payload_left_half", "effect_tag", "bsssi_site_member_right"): "allowed_if_transform_derived",
    ("structural_owner", "payload_left_half", "effect_tag", "left_payload_overlap_unit"): "allowed_if_nested",
    ("structural_owner", "sacrificial_region_long", "effect_tag", "sacrificial"): "allowed_if_nested",
    (
        "structural_owner",
        "sacrificial_region_long",
        "effect_tag",
        "bsssi_site_member_right",
    ): "allowed_if_transform_derived",
    (
        "structural_owner",
        "sacrificial_region_long",
        "effect_tag",
        "left_payload_overlap_unit",
    ): "allowed_if_transform_derived",
    ("structural_owner", "tether_dock_complement", "effect_tag", "nt_bpu10i_snapback_site"): "allowed_if_nested",
    ("structural_owner", "tether_cap", "effect_tag", "nt_bpu10i_snapback_site"): "allowed_if_nested",
    ("structural_owner", "tether_dock", "effect_tag", "nt_bpu10i_snapback_site"): "allowed_if_nested",
    ("structural_owner", "snapback_stem", "effect_tag", "nt_bpu10i_snapback_site"): "allowed_if_nested",
    (
        "effect_tag",
        "primer_bindable_by_source_forward",
        "effect_tag",
        "bsssi_site_member_left",
    ): "allowed_if_identical_span",
    (
        "effect_tag",
        "primer_bindable_by_source_reverse",
        "effect_tag",
        "type_iis_recognition_right",
    ): "allowed_if_identical_span",
    ("effect_tag", "primer_bindable_by_source_forward", "effect_tag", "left_payload_overlap_unit"): "allowed_if_nested",
    ("effect_tag", "left_payload_overlap_unit", "effect_tag", "bsssi_site_member_left"): "allowed_if_nested",
    ("effect_tag", "left_payload_overlap_unit", "effect_tag", "bsai_site_member"): "allowed_if_nested",
    ("effect_tag", "left_payload_overlap_unit", "effect_tag", "bsssi_site_member_right"): "allowed_if_nested",
    ("effect_tag", "left_payload_overlap_unit", "effect_tag", "payload_overhang_left"): "allowed_if_nested",
    ("effect_tag", "left_payload_overlap_unit", "effect_tag", "type_iis_recognition_left"): "allowed_if_nested",
    ("effect_tag", "type_iis_recognition_left", "effect_tag", "bsai_site_member"): "allowed_if_identical_span",
    ("effect_tag", "payload_overhang_left", "effect_tag", "type_iis_recognition_left"): "allowed_if_transform_derived",
    ("effect_tag", "payload_overhang_left", "effect_tag", "bsai_site_member"): "allowed_if_transform_derived",
    ("effect_tag", "bsai_site_member", "effect_tag", "bsssi_site_member_right"): "allowed_if_exact_partner",
    ("effect_tag", "type_iis_recognition_left", "effect_tag", "bsssi_site_member_right"): "allowed_if_exact_partner",
}

_EXACT_PARTNER_OVERLAP_COUNTS: dict[frozenset[str], int] = {
    frozenset({"bsai_site_member", "bsssi_site_member_right"}): 1,
    frozenset({"type_iis_recognition_left", "bsssi_site_member_right"}): 1,
    frozenset({"pairs_with_tether_dock", "tether_dock_complement"}): 4,
    frozenset({"pairs_with_tether_dock_complement", "tether_dock"}): 4,
}


def _validate_state_id(value: str, *, label: str) -> str:
    state_id = _validate_slug(value, label=label)
    if state_id not in YIU_V3_STATE_IDS:
        raise ValueError(f"{label} must be one of {list(YIU_V3_STATE_IDS)}")
    return state_id


def _state_index(state_id: str) -> int:
    return YIU_V3_STATE_IDS.index(state_id)


def _ranges_overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return left_start < right_end and right_start < left_end


def _is_nested(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return right[0] <= left[0] and left[1] <= right[1]


def _lookup_overlap_policy(left_kind: str, left_id: str, right_kind: str, right_id: str) -> str | None:
    direct = (left_kind, left_id, right_kind, right_id)
    if direct in _OVERLAP_POLICIES:
        return _OVERLAP_POLICIES[direct]
    reverse = (right_kind, right_id, left_kind, left_id)
    return _OVERLAP_POLICIES.get(reverse)


def _allows_overlap(
    policy: str,
    *,
    left_span: tuple[int, int],
    right_span: tuple[int, int],
    left_id: str,
    right_id: str,
) -> bool:
    if policy == "forbidden":
        return False
    if policy == "allowed_if_identical_span":
        return left_span == right_span
    if policy == "allowed_if_nested":
        return _is_nested(left_span, right_span) or _is_nested(right_span, left_span)
    if policy == "allowed_if_nested_exact":
        if not (_is_nested(left_span, right_span) or _is_nested(right_span, left_span)):
            return False
        return (
            left_span[0] == right_span[0]
            or left_span[1] == right_span[1]
            or left_span[0] == right_span[1]
            or left_span[1] == right_span[0]
        )
    if policy == "allowed_if_exact_partner":
        shared_nt_count = min(left_span[1], right_span[1]) - max(left_span[0], right_span[0])
        expected = _EXACT_PARTNER_OVERLAP_COUNTS.get(frozenset({left_id, right_id}))
        return expected is not None and shared_nt_count == expected
    if policy == "allowed_if_transform_derived":
        return True
    raise ValueError(f"unknown overlap policy {policy!r}")


class YiuDerivationV3(StrictBaseModel):
    kind: Literal["literal", "reverse_complement_projection", "transform_projection", "late_introduction"]
    from_state: str | None = None
    from_owner: str | None = None

    @field_validator("from_state")
    @classmethod
    def _validate_optional_from_state(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _validate_state_id(value, label="derivation.from_state")

    @field_validator("from_owner")
    @classmethod
    def _validate_optional_from_owner(cls, value: str | None) -> str | None:
        if value is None:
            return value
        normalized = _validate_slug(value, label="derivation.from_owner")
        if normalized not in YIU_V3_OPERATIONAL_OWNER_IDS:
            raise ValueError(f"derivation.from_owner must be one of {list(YIU_V3_OPERATIONAL_OWNER_IDS)}")
        return normalized


class YiuStateLifecycleV3(StrictBaseModel):
    first_state: str
    last_state: str | None = None
    disposition: Literal["retained", "transformed", "removed", "introduced"]

    @field_validator("first_state")
    @classmethod
    def _validate_first_state(cls, value: str) -> str:
        return _validate_state_id(value, label="state_lifecycle.first_state")

    @field_validator("last_state")
    @classmethod
    def _validate_last_state(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _validate_state_id(value, label="state_lifecycle.last_state")

    @model_validator(mode="after")
    def _validate_order(self) -> "YiuStateLifecycleV3":
        if self.last_state is not None and _state_index(self.last_state) < _state_index(self.first_state):
            raise ValueError("state_lifecycle.last_state must not precede state_lifecycle.first_state")
        return self


class YiuOwnerProvenanceV3(StrictBaseModel):
    origin_state: Literal["source_oligo_ssdna", "introduced_late"]
    origin_owner: str | None = None
    derivation: YiuDerivationV3

    @field_validator("origin_owner")
    @classmethod
    def _validate_optional_origin_owner(cls, value: str | None) -> str | None:
        if value is None:
            return value
        normalized = _validate_slug(value, label="origin_owner")
        if normalized not in YIU_V3_OPERATIONAL_OWNER_IDS:
            raise ValueError(f"origin_owner must be one of {list(YIU_V3_OPERATIONAL_OWNER_IDS)}")
        return normalized

    @model_validator(mode="after")
    def _validate_origin_contract(self) -> "YiuOwnerProvenanceV3":
        if self.origin_state == "introduced_late" and self.derivation.kind != "late_introduction":
            raise ValueError("origin_state=introduced_late requires derivation.kind=late_introduction")
        if self.origin_state == "source_oligo_ssdna" and self.derivation.kind == "late_introduction":
            raise ValueError("late_introduction provenance requires origin_state=introduced_late")
        return self


class YiuStructuralOwnerSpecV3(StrictBaseModel):
    id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    provenance: YiuOwnerProvenanceV3
    state_lifecycle: YiuStateLifecycleV3

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        normalized = _validate_slug(value, label="structural_owner.id")
        if normalized not in YIU_V3_SOURCE_OWNER_IDS:
            raise ValueError(f"structural_owner.id must be one of {list(YIU_V3_SOURCE_OWNER_IDS)}")
        return normalized

    @model_validator(mode="after")
    def _validate_bounds(self) -> "YiuStructuralOwnerSpecV3":
        if self.end <= self.start:
            raise ValueError("structural_owner.end must be > structural_owner.start")
        if self.provenance.origin_state != "source_oligo_ssdna":
            raise ValueError("source-side structural owners must originate from source_oligo_ssdna")
        if self.provenance.derivation.kind != "literal":
            raise ValueError("source-side structural owners must use provenance.derivation.kind=literal")
        if self.state_lifecycle.first_state != "source_oligo_ssdna":
            raise ValueError("source-side structural owners must first exist in source_oligo_ssdna")
        return self


class YiuEffectTagSpecV3(StrictBaseModel):
    id: str
    class_: str = Field(alias="class")
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    provenance: YiuOwnerProvenanceV3
    state_lifecycle: YiuStateLifecycleV3

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _validate_slug(value, label="effect_tag.id")

    @field_validator("class_")
    @classmethod
    def _validate_class(cls, value: str) -> str:
        normalized = _validate_slug(value, label="effect_tag.class")
        if normalized not in YIU_V3_EFFECT_TAG_CLASSES:
            raise ValueError(f"effect_tag.class must be one of {sorted(YIU_V3_EFFECT_TAG_CLASSES)}")
        return normalized

    @model_validator(mode="after")
    def _validate_bounds(self) -> "YiuEffectTagSpecV3":
        if self.end <= self.start:
            raise ValueError("effect_tag.end must be > effect_tag.start")
        return self


class SourceOligoSpecV3(StrictBaseModel):
    sequence: str | None = None
    authored_sequence: str | None = None
    structural_owners: list[YiuStructuralOwnerSpecV3] = Field(default_factory=list)
    effect_tags: list[YiuEffectTagSpecV3] = Field(default_factory=list)

    @field_validator("sequence")
    @classmethod
    def _validate_sequence(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return normalize_iupac(value)

    @model_validator(mode="after")
    def _normalize_and_validate(self) -> "SourceOligoSpecV3":
        normalized_sequence = self.sequence
        if normalized_sequence is None and self.authored_sequence is not None:
            normalized_sequence = normalize_iupac(self.authored_sequence)
        if normalized_sequence is None:
            raise ValueError("source_oligo requires sequence or authored_sequence")
        if self.sequence is not None and self.authored_sequence is not None:
            authored_normalized = normalize_iupac(self.authored_sequence)
            if self.sequence != authored_normalized:
                raise ValueError("source_oligo.sequence must match authored_sequence after normalization")
        self.sequence = normalized_sequence

        if len(self.structural_owners) != len(YIU_V3_SOURCE_OWNER_IDS):
            raise ValueError(
                f"source_oligo.structural_owners must declare each canonical source owner exactly once: "
                f"{list(YIU_V3_SOURCE_OWNER_IDS)}"
            )
        owner_ids = [owner.id for owner in self.structural_owners]
        if tuple(owner_ids) != YIU_V3_SOURCE_OWNER_IDS:
            raise ValueError(
                f"source_oligo.structural_owners must appear in canonical 5'->3' order {list(YIU_V3_SOURCE_OWNER_IDS)}"
            )
        if self.structural_owners[0].start != 0:
            raise ValueError("source_oligo.structural_owners must start at coordinate 0")
        cursor = 0
        for owner in self.structural_owners:
            if owner.start != cursor:
                raise ValueError("source_oligo.structural_owners must be contiguous and gap-free")
            cursor = owner.end
        if cursor != len(self.sequence):
            raise ValueError("source_oligo.structural_owners must cover the entire source_oligo.sequence")

        tag_ids: list[str] = []
        for tag in self.effect_tags:
            if tag.end > len(self.sequence):
                raise ValueError(f"effect tag {tag.id} exceeds source_oligo.sequence length")
            tag_ids.append(tag.id)
        if len(set(tag_ids)) != len(tag_ids):
            raise ValueError("source_oligo.effect_tags ids must be unique")

        tag_by_id = {tag.id: tag for tag in self.effect_tags}
        required_ids = {
            "type_iis_recognition_left": "type_iis_recognition_left",
            "type_iis_recognition_right": "type_iis_recognition_right",
            "payload_overhang_left": "payload_overhang_left",
            "payload_overhang_right": "payload_overhang_right",
            "primer_bindable_by_source_forward": "primer_bindable_by_source_forward",
            "primer_bindable_by_source_reverse": "primer_bindable_by_source_reverse",
            "nt_bpu10i_snapback_site": "nt_bpu10i_snapback_site",
            "left_payload_overlap_unit": "left_payload_overlap_unit",
            "bsssi_site_member_left": "bsssi_site_member_left",
            "bsai_site_member": "bsai_site_member",
            "bsssi_site_member_right": "bsssi_site_member_right",
        }
        for tag_id, expected_class in required_ids.items():
            tag = tag_by_id.get(tag_id)
            if tag is None:
                raise ValueError(f"canonical v3 source_oligo.effect_tags must include {tag_id!r}")
            if tag.class_ != expected_class:
                raise ValueError(f"effect tag {tag_id!r} must use class {expected_class!r}")
        if not any(tag.class_ == "sacrificial" for tag in self.effect_tags):
            raise ValueError("canonical v3 source_oligo.effect_tags must declare at least one sacrificial spacer tag")

        self._validate_effect_tag_sequences(tag_by_id)
        self._validate_overlap_legality()
        return self

    def _validate_effect_tag_sequences(self, tag_by_id: dict[str, YiuEffectTagSpecV3]) -> None:
        assert self.sequence is not None

        def _tag_sequence(tag_id: str) -> str:
            tag = tag_by_id[tag_id]
            return self.sequence[tag.start : tag.end]

        if _tag_sequence("left_payload_overlap_unit") != _LEFT_PAYLOAD_OVERLAP_SEQUENCE:
            raise ValueError("left_payload_overlap_unit must match the canonical CACGAGaGGTCTCACGAG overlap unit")
        if _tag_sequence("bsssi_site_member_left") != _BSSSI_MEMBER_SEQUENCE:
            raise ValueError("bsssi_site_member_left must match CACGAG")
        if _tag_sequence("bsssi_site_member_right") != _BSSSI_MEMBER_SEQUENCE:
            raise ValueError("bsssi_site_member_right must match CACGAG")
        if _tag_sequence("bsai_site_member") != _TYPE_IIS_LEFT_SEQUENCE:
            raise ValueError("bsai_site_member must match GGTCTC")
        if _tag_sequence("type_iis_recognition_left") != _TYPE_IIS_LEFT_SEQUENCE:
            raise ValueError("type_iis_recognition_left must match GGTCTC")
        if _tag_sequence("type_iis_recognition_right") != _TYPE_IIS_RIGHT_SEQUENCE:
            raise ValueError("type_iis_recognition_right must match GAGACC")
        if _tag_sequence("nt_bpu10i_snapback_site") != _NT_BPU10I_LOCAL_CONTEXT:
            raise ValueError("nt_bpu10i_snapback_site must match the canonical CCTCAGCccgctga local context")
        if len(_tag_sequence("payload_overhang_left")) != 4 or len(_tag_sequence("payload_overhang_right")) != 4:
            raise ValueError("payload overhang effect tags must be exactly 4 nt")
        if _tag_sequence("primer_bindable_by_source_forward") != self.sequence[0:6]:
            raise ValueError("primer_bindable_by_source_forward must match the source forward primer owner span")
        if _tag_sequence("primer_bindable_by_source_reverse") != self.sequence[51:57]:
            raise ValueError("primer_bindable_by_source_reverse must match the source reverse primer owner span")

    def _validate_overlap_legality(self) -> None:
        owners = list(self.structural_owners)
        tags = list(self.effect_tags)

        for owner in owners:
            owner_span = (owner.start, owner.end)
            for tag in tags:
                tag_span = (tag.start, tag.end)
                if not _ranges_overlap(*owner_span, *tag_span):
                    continue
                policy = _lookup_overlap_policy("structural_owner", owner.id, "effect_tag", tag.class_)
                if policy is None:
                    raise ValueError(
                        f"overlap between structural_owner {owner.id!r} and effect_tag {tag.class_!r} is not declared"
                    )
                if not _allows_overlap(
                    policy, left_span=owner_span, right_span=tag_span, left_id=owner.id, right_id=tag.class_
                ):
                    raise ValueError(
                        f"overlap between structural_owner {owner.id!r} and effect_tag {tag.class_!r} violates {policy}"
                    )

        for index, left in enumerate(tags):
            left_span = (left.start, left.end)
            for right in tags[index + 1 :]:
                right_span = (right.start, right.end)
                if not _ranges_overlap(*left_span, *right_span):
                    continue
                policy = _lookup_overlap_policy("effect_tag", left.class_, "effect_tag", right.class_)
                if policy is None:
                    raise ValueError(
                        f"overlap between effect_tag {left.class_!r} and effect_tag {right.class_!r} is not declared"
                    )
                if not _allows_overlap(
                    policy,
                    left_span=left_span,
                    right_span=right_span,
                    left_id=left.class_,
                    right_id=right.class_,
                ):
                    raise ValueError(
                        f"overlap between effect_tag {left.class_!r} and effect_tag {right.class_!r} violates {policy}"
                    )


class YiuExternalPartsV3(StrictBaseModel):
    primer_source_forward: str
    primer_source_reverse: str
    hairpin_pcr_forward: str
    hairpin_pcr_reverse: str
    y_adapter: str

    @field_validator(
        "primer_source_forward",
        "primer_source_reverse",
        "hairpin_pcr_forward",
        "hairpin_pcr_reverse",
        "y_adapter",
    )
    @classmethod
    def _validate_part_ids(cls, value: str, info) -> str:
        return _validate_slug(value, label=str(info.field_name))


class YiuSourcePcrStepSpecV3(StrictBaseModel):
    forward_primer_id: str
    reverse_primer_id: str

    @field_validator("forward_primer_id", "reverse_primer_id")
    @classmethod
    def _validate_part_ids(cls, value: str, info) -> str:
        return _validate_slug(value, label=str(info.field_name))


class YiuTypeIisDigestStepSpecV3(StrictBaseModel):
    enzyme_id: str
    left_site_ref: str
    right_site_ref: str
    left_orientation: Literal["forward", "reverse"] = "forward"
    right_orientation: Literal["forward", "reverse"] = "reverse"
    top_cut_offset: int = Field(ge=0)
    bottom_cut_offset: int = Field(ge=0)

    @field_validator("enzyme_id", "left_site_ref", "right_site_ref")
    @classmethod
    def _validate_ids(cls, value: str, info) -> str:
        return _validate_slug(value, label=str(info.field_name))


class YiuCircularizationStepSpecV3(StrictBaseModel):
    ligation_mode: Literal["exact_complement", "bulged"] = "exact_complement"
    left_overhang_ref: str
    right_overhang_ref: str

    @field_validator("left_overhang_ref", "right_overhang_ref")
    @classmethod
    def _validate_refs(cls, value: str, info) -> str:
        return _validate_slug(value, label=str(info.field_name))


class YiuExonucleaseCleanupStepSpecV3(StrictBaseModel):
    enabled: bool = True
    enzyme: str | None = None

    @field_validator("enzyme")
    @classmethod
    def _validate_optional_enzyme(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _validate_slug(value, label="exonuclease_cleanup.enzyme")


class YiuSacrificialDigestStepSpecV3(StrictBaseModel):
    enzyme_id: str
    site_ref: str

    @field_validator("enzyme_id", "site_ref")
    @classmethod
    def _validate_ids(cls, value: str, info) -> str:
        return _validate_slug(value, label=str(info.field_name))


class YiuFragmentCleanupStepSpecV3(StrictBaseModel):
    enabled: bool = True
    max_fragment_nt: int = Field(default=12, ge=1)
    min_retained_nt: int = Field(default=1, ge=1)


class YiuSnapbackAdapterEngagementStepSpecV3(StrictBaseModel):
    adapter_id: str

    @field_validator("adapter_id")
    @classmethod
    def _validate_adapter_id(cls, value: str) -> str:
        return _validate_slug(value, label="snapback_adapter_engagement.adapter_id")


class YiuHairpinLigationStepSpecV3(StrictBaseModel):
    ligase: str | None = None
    require_5p_phosphate: bool = False

    @field_validator("ligase")
    @classmethod
    def _validate_optional_ligase(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _validate_slug(value, label="hairpin_ligation.ligase")


class YiuHairpinPcrStepSpecV3(StrictBaseModel):
    forward_primer_id: str
    reverse_primer_id: str

    @field_validator("forward_primer_id", "reverse_primer_id")
    @classmethod
    def _validate_part_ids(cls, value: str, info) -> str:
        return _validate_slug(value, label=str(info.field_name))


class YiuStepsSpecV3(StrictBaseModel):
    source_pcr: YiuSourcePcrStepSpecV3
    type_iis_digest: YiuTypeIisDigestStepSpecV3
    circularization: YiuCircularizationStepSpecV3
    exonuclease_cleanup: YiuExonucleaseCleanupStepSpecV3 = Field(default_factory=YiuExonucleaseCleanupStepSpecV3)
    sacrificial_digest: YiuSacrificialDigestStepSpecV3
    fragment_cleanup: YiuFragmentCleanupStepSpecV3 = Field(default_factory=YiuFragmentCleanupStepSpecV3)
    snapback_adapter_engagement: YiuSnapbackAdapterEngagementStepSpecV3
    hairpin_ligation: YiuHairpinLigationStepSpecV3 = Field(default_factory=YiuHairpinLigationStepSpecV3)
    hairpin_pcr: YiuHairpinPcrStepSpecV3


class YiuPayloadBulgeMaskSpecV3(StrictBaseModel):
    left_index: int = Field(ge=0)
    right_index: int = Field(ge=0)
    mismatch_kind: Literal["bulged_unpaired", "mismatch"]


class YiuPayloadOverhangGeometrySpecV3(StrictBaseModel):
    left_overhang_ref: str
    right_overhang_ref: str
    bulge_mask: list[YiuPayloadBulgeMaskSpecV3] = Field(default_factory=list)

    @field_validator("left_overhang_ref", "right_overhang_ref")
    @classmethod
    def _validate_refs(cls, value: str, info) -> str:
        return _validate_slug(value, label=str(info.field_name))


class YiuPayloadGoalSpecV3(StrictBaseModel):
    assembled_payload_sequence: str
    payload_overhang_geometry: YiuPayloadOverhangGeometrySpecV3

    @field_validator("assembled_payload_sequence")
    @classmethod
    def _validate_sequence(cls, value: str) -> str:
        return normalize_iupac(value)


class YiuHardInvariantV3(StrictBaseModel):
    id: str
    class_: Literal["payload_assembly", "nt_bpu10i_snapback_site", "sacrificial_fragmentation"] = Field(alias="class")
    required: bool = True
    required_states: list[str] = Field(default_factory=list)
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _validate_slug(value, label="hard_invariant.id")

    @field_validator("required_states")
    @classmethod
    def _validate_required_states(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("hard_invariant.required_states must be non-empty")
        return [_validate_state_id(item, label="hard_invariant.required_states") for item in value]

    @model_validator(mode="after")
    def _validate_params(self) -> "YiuHardInvariantV3":
        if self.class_ == "payload_assembly":
            if self.params:
                raise ValueError("payload_assembly must not define params")
        elif self.class_ == "nt_bpu10i_snapback_site":
            required_keys = {
                "enzyme_variant",
                "recognized_sequence",
                "local_context_sequence",
                "nicked_strand",
                "nick_offset",
            }
            missing = required_keys - set(self.params)
            if missing:
                raise ValueError(f"nt_bpu10i_snapback_site.params is missing required keys {sorted(missing)}")
            self.params["recognized_sequence"] = normalize_iupac(str(self.params["recognized_sequence"]))
            self.params["local_context_sequence"] = normalize_iupac(str(self.params["local_context_sequence"]))
            self.params["enzyme_variant"] = _validate_slug(
                str(self.params["enzyme_variant"]), label="params.enzyme_variant"
            )
            if str(self.params["nicked_strand"]) not in {"primary", "complement"}:
                raise ValueError("nt_bpu10i_snapback_site.params.nicked_strand must be 'primary' or 'complement'")
            try:
                self.params["nick_offset"] = int(self.params["nick_offset"])
            except (TypeError, ValueError) as exc:
                raise ValueError("nt_bpu10i_snapback_site.params.nick_offset must be an integer") from exc
        elif self.class_ == "sacrificial_fragmentation":
            required_keys = {
                "max_fragment_nt",
                "threshold_mode",
                "require_retained_survival",
                "allow_single_payload_adjacent_retained_nt",
            }
            missing = required_keys - set(self.params)
            if missing:
                raise ValueError(f"sacrificial_fragmentation.params is missing required keys {sorted(missing)}")
            self.params["max_fragment_nt"] = int(self.params["max_fragment_nt"])
            if self.params["max_fragment_nt"] < 1:
                raise ValueError("sacrificial_fragmentation.params.max_fragment_nt must be >= 1")
            if str(self.params["threshold_mode"]) != "less_or_equal":
                raise ValueError("sacrificial_fragmentation.params.threshold_mode must be 'less_or_equal'")
            self.params["require_retained_survival"] = bool(self.params["require_retained_survival"])
            self.params["allow_single_payload_adjacent_retained_nt"] = bool(
                self.params["allow_single_payload_adjacent_retained_nt"]
            )
        return self


class YiuProcessSpecV3(StrictBaseModel):
    schema_version: int = 3
    family: Literal["yiu"] = "yiu"
    protocol_template: Literal["yiu_circularized_payload_v1"] = "yiu_circularized_payload_v1"
    name: str
    source_oligo: SourceOligoSpecV3
    external_parts: YiuExternalPartsV3
    steps: YiuStepsSpecV3
    payload_goal: YiuPayloadGoalSpecV3
    hard_invariants: list[YiuHardInvariantV3] = Field(default_factory=list)
    catalogs: CatalogRefsV2 = Field(default_factory=CatalogRefsV2)
    output: OutputSpecV2 = Field(default_factory=OutputSpecV2)

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        if int(value) != 3:
            raise ValueError("yiu.schema_version must be 3")
        return int(value)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        return _validate_slug(value, label="yiu.name")

    @model_validator(mode="after")
    def _validate_v3_contract(self) -> "YiuProcessSpecV3":
        effect_tag_ids = {tag.id for tag in self.source_oligo.effect_tags}
        owner_ids = {owner.id for owner in self.source_oligo.structural_owners}
        for ref in (
            self.steps.type_iis_digest.left_site_ref,
            self.steps.type_iis_digest.right_site_ref,
            self.steps.circularization.left_overhang_ref,
            self.steps.circularization.right_overhang_ref,
            self.payload_goal.payload_overhang_geometry.left_overhang_ref,
            self.payload_goal.payload_overhang_geometry.right_overhang_ref,
            self.steps.sacrificial_digest.site_ref,
        ):
            if ref not in effect_tag_ids:
                raise ValueError(f"canonical v3 reference {ref!r} must point to a declared source_oligo.effect_tags id")

        if owner_ids != set(YIU_V3_SOURCE_OWNER_IDS):
            raise ValueError("canonical v3 specs must declare the full source-side owner vocabulary")

        if self.steps.type_iis_digest.left_site_ref != "type_iis_recognition_left":
            raise ValueError("steps.type_iis_digest.left_site_ref must be type_iis_recognition_left")
        if self.steps.type_iis_digest.right_site_ref != "type_iis_recognition_right":
            raise ValueError("steps.type_iis_digest.right_site_ref must be type_iis_recognition_right")
        if self.steps.circularization.left_overhang_ref != "payload_overhang_left":
            raise ValueError("steps.circularization.left_overhang_ref must be payload_overhang_left")
        if self.steps.circularization.right_overhang_ref != "payload_overhang_right":
            raise ValueError("steps.circularization.right_overhang_ref must be payload_overhang_right")
        if self.steps.sacrificial_digest.site_ref != "nt_bpu10i_snapback_site":
            raise ValueError("steps.sacrificial_digest.site_ref must be nt_bpu10i_snapback_site")

        required_invariants = {invariant.id: invariant for invariant in self.hard_invariants}
        for required_id in ("payload_assembly", "nt_bpu10i_snapback_site", "sacrificial_fragmentation"):
            if required_id not in required_invariants:
                raise ValueError(f"hard_invariants must declare canonical required invariant {required_id!r}")

        payload_invariant = required_invariants["payload_assembly"]
        if tuple(payload_invariant.required_states) != _PAYLOAD_ASSEMBLY_REQUIRED_STATES:
            raise ValueError(f"payload_assembly.required_states must match {list(_PAYLOAD_ASSEMBLY_REQUIRED_STATES)}")
        nt_bpu10i_invariant = required_invariants["nt_bpu10i_snapback_site"]
        if tuple(nt_bpu10i_invariant.required_states) != _NT_BPU10I_REQUIRED_STATES:
            raise ValueError(f"nt_bpu10i_snapback_site.required_states must match {list(_NT_BPU10I_REQUIRED_STATES)}")
        if nt_bpu10i_invariant.params["recognized_sequence"] != _NT_BPU10I_RECOGNITION_SEQUENCE:
            raise ValueError("nt_bpu10i_snapback_site.params.recognized_sequence must be CCTCAGC")
        if nt_bpu10i_invariant.params["local_context_sequence"] != _NT_BPU10I_LOCAL_CONTEXT:
            raise ValueError("nt_bpu10i_snapback_site.params.local_context_sequence must be CCTCAGCCCGCTGA")
        if nt_bpu10i_invariant.params["enzyme_variant"] != "Nt.Bpu10I":
            raise ValueError("nt_bpu10i_snapback_site.params.enzyme_variant must be Nt.Bpu10I")
        if nt_bpu10i_invariant.params["nicked_strand"] != "primary":
            raise ValueError("nt_bpu10i_snapback_site.params.nicked_strand must be primary")
        if nt_bpu10i_invariant.params["nick_offset"] != 2:
            raise ValueError("nt_bpu10i_snapback_site.params.nick_offset must be 2")

        fragmentation_invariant = required_invariants["sacrificial_fragmentation"]
        if tuple(fragmentation_invariant.required_states) != _SACRIFICIAL_FRAGMENTATION_REQUIRED_STATES:
            raise ValueError(
                "sacrificial_fragmentation.required_states must match "
                f"{list(_SACRIFICIAL_FRAGMENTATION_REQUIRED_STATES)}"
            )

        if self.output.publish_contract_version != 3:
            raise ValueError("output.publish_contract_version must be 3 for canonical YIU v3 specs")
        if self.output.emit_baserender_jobs and not self.output.emit_view_contracts:
            raise ValueError("output.emit_baserender_jobs requires output.emit_view_contracts=true")

        source_part_requirements = {
            self.steps.source_pcr.forward_primer_id: self.external_parts.primer_source_forward,
            self.steps.source_pcr.reverse_primer_id: self.external_parts.primer_source_reverse,
            self.steps.hairpin_pcr.forward_primer_id: self.external_parts.hairpin_pcr_forward,
            self.steps.hairpin_pcr.reverse_primer_id: self.external_parts.hairpin_pcr_reverse,
            self.steps.snapback_adapter_engagement.adapter_id: self.external_parts.y_adapter,
        }
        for step_value, external_part in source_part_requirements.items():
            if step_value != external_part:
                raise ValueError(
                    "canonical v3 steps must reference the matching external_parts ids for the ship workflow"
                )

        overhang_left = next(tag for tag in self.source_oligo.effect_tags if tag.id == "payload_overhang_left")
        overhang_right = next(tag for tag in self.source_oligo.effect_tags if tag.id == "payload_overhang_right")
        source_sequence = self.source_oligo.sequence or ""
        left_sequence = source_sequence[overhang_left.start : overhang_left.end]
        right_sequence = source_sequence[overhang_right.start : overhang_right.end]
        if left_sequence != reverse_complement_iupac(right_sequence):
            raise ValueError("payload overhangs must satisfy exact-complement circularization")
        return self


class YiuSpecDocumentV3(StrictBaseModel):
    yiu: YiuProcessSpecV3
