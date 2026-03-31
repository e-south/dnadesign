"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/yiu/models/v4.py

Canonical YIU v4 schema for the replay/validation workflow.

Module Author(s): OpenAI Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from dnadesign.cruncher.bio.iupac import normalize_iupac, reverse_complement_iupac
from dnadesign.cruncher.config.schema_v3 import StrictBaseModel
from dnadesign.cruncher.yiu.models.common import _validate_slug
from dnadesign.cruncher.yiu.models.v2_steps import CatalogRefsV2

YIU_V4_SOURCE_OWNER_IDS: tuple[str, ...] = (
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

YIU_V4_OPERATIONAL_OWNER_IDS: tuple[str, ...] = (
    *YIU_V4_SOURCE_OWNER_IDS,
    "retained_region",
    "sacrificial_region_short",
    "y_adapter_complementary_arm",
    "y_adapter_noncomplementary_arm",
    "hairpin_pcr_forward_binding_region",
    "hairpin_pcr_reverse_binding_region",
)

YIU_V4_EFFECT_TAG_CLASSES: frozenset[str] = frozenset(
    {
        "type_iis_recognition_left",
        "type_iis_recognition_right",
        "payload_overhang_left",
        "payload_overhang_right",
        "nt_bpu10i_snapback_site",
        "nb_bsssi_array_member",
        "left_bsssi_bsai_overlap_unit",
        "pairs_with",
        "primer_bindable_by_source_forward",
        "primer_bindable_by_source_reverse",
        "primer_bindable_by_hairpin_pcr_forward",
        "primer_bindable_by_hairpin_pcr_reverse",
        "retained",
        "sacrificial",
        "introduced_late",
        "y_adapter_binding",
        "ligation_junction_member",
        "cut_boundary_anchor",
        "nick_boundary_anchor",
        "payload_bulge_position",
    }
)

YIU_V4_STATE_IDS: tuple[str, ...] = (
    "source_oligo_ssdna",
    "pcr_linear_duplex",
    "type_iis_cut_product_duplex",
    "circularized_payload_candidate",
    "post_sacrificial_fragmentation",
    "post_fragment_cleanup",
    "snapback_adapter_complex",
    "ligated_ssdna_hairpin",
    "hairpin_pcr_linear_insert",
)

YIU_V4_ALLOWED_SOLVE_WINDOW_OWNER_IDS: frozenset[str] = frozenset({"sacrificial_region_long"})

YIU_V4_LEFT_OVERLAP_SEQUENCE = "CACGAGAGGTCTCACGAG"
YIU_V4_TYPE_IIS_LEFT_SEQUENCE = "GGTCTC"
YIU_V4_TYPE_IIS_RIGHT_SEQUENCE = "GAGACC"
YIU_V4_NB_BSSSI_MEMBER_SEQUENCE = "CACGAG"
YIU_V4_NT_BPU10I_LOCAL_CONTEXT = "CCTCAGCCCGCTGA"
YIU_V4_NT_BPU10I_RECOGNITION_SEQUENCE = "CCTCAGC"

_OWNER_TAG_COMPATIBILITY: dict[str, frozenset[str]] = {
    "source_fwd_primer_binding_region": frozenset(
        {"primer_bindable_by_source_forward", "nb_bsssi_array_member", "left_bsssi_bsai_overlap_unit"}
    ),
    "payload_left_half": frozenset(
        {
            "payload_overhang_left",
            "type_iis_recognition_left",
            "nb_bsssi_array_member",
            "left_bsssi_bsai_overlap_unit",
        }
    ),
    "sacrificial_region_long": frozenset({"sacrificial", "nb_bsssi_array_member", "left_bsssi_bsai_overlap_unit"}),
    "tether_dock_complement": frozenset({"nt_bpu10i_snapback_site"}),
    "tether_cap": frozenset({"nt_bpu10i_snapback_site"}),
    "tether_dock": frozenset({"nt_bpu10i_snapback_site"}),
    "snapback_stem": frozenset({"nt_bpu10i_snapback_site"}),
    "payload_right_half": frozenset({"payload_overhang_right"}),
    "source_rev_primer_binding_region": frozenset({"primer_bindable_by_source_reverse", "type_iis_recognition_right"}),
}

_TAG_TAG_COMPATIBILITY: frozenset[frozenset[str]] = frozenset(
    {
        frozenset({"primer_bindable_by_source_forward", "nb_bsssi_array_member"}),
        frozenset({"primer_bindable_by_source_forward", "left_bsssi_bsai_overlap_unit"}),
        frozenset({"left_bsssi_bsai_overlap_unit", "nb_bsssi_array_member"}),
        frozenset({"left_bsssi_bsai_overlap_unit", "type_iis_recognition_left"}),
        frozenset({"left_bsssi_bsai_overlap_unit", "payload_overhang_left"}),
        frozenset({"left_bsssi_bsai_overlap_unit", "sacrificial"}),
        frozenset({"type_iis_recognition_left", "nb_bsssi_array_member"}),
        frozenset({"payload_overhang_left", "type_iis_recognition_left"}),
        frozenset({"nb_bsssi_array_member", "sacrificial"}),
        frozenset({"primer_bindable_by_source_reverse", "type_iis_recognition_right"}),
    }
)


def _validate_state_id(value: str, *, label: str) -> str:
    state_id = _validate_slug(value, label=label)
    if state_id not in YIU_V4_STATE_IDS:
        raise ValueError(f"{label} must be one of {list(YIU_V4_STATE_IDS)}")
    return state_id


def _ranges_overlap(left_start: int, left_end: int, right_start: int, right_end: int) -> bool:
    return left_start < right_end and right_start < left_end


class YiuOutputSpecV4(StrictBaseModel):
    run_dir: Path = Path("outputs/yiu/explicit")
    emit_view_contracts: bool = True
    publish_contract_version: int = 4
    persist_render_jobs_debug: bool = False

    @field_validator("run_dir")
    @classmethod
    def _validate_run_dir(cls, value: Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            raise ValueError("output.run_dir must be relative to the workspace root")
        if ".." in path.parts:
            raise ValueError("output.run_dir must stay inside the workspace root")
        return path

    @field_validator("publish_contract_version")
    @classmethod
    def _validate_publish_contract_version(cls, value: int) -> int:
        if int(value) != 4:
            raise ValueError("output.publish_contract_version must be 4 for canonical YIU v4 specs")
        return int(value)


class YiuOwnerProjectionSpecV4(StrictBaseModel):
    state: str
    strand: Literal["primary", "complement"]
    provenance_mode: Literal[
        "literal_source",
        "amplification_projection",
        "cut_product_projection",
        "ligation_assembly",
        "retained_projection",
        "sacrificial_projection",
        "late_introduction",
        "ligated_projection",
        "derived_binding_region",
    ]

    @field_validator("state")
    @classmethod
    def _validate_state(cls, value: str) -> str:
        return _validate_state_id(value, label="owner_lifecycle.projected_to.state")


class YiuOwnerLifecycleEntryV4(StrictBaseModel):
    owner_id: str
    appears_in: list[str] = Field(default_factory=list)
    projected_to: list[YiuOwnerProjectionSpecV4] = Field(default_factory=list)
    disappears_after: str | None = None

    @field_validator("owner_id")
    @classmethod
    def _validate_owner_id(cls, value: str) -> str:
        normalized = _validate_slug(value, label="owner_lifecycle.owner_id")
        if normalized not in YIU_V4_OPERATIONAL_OWNER_IDS:
            raise ValueError(f"owner_lifecycle.owner_id must be one of {list(YIU_V4_OPERATIONAL_OWNER_IDS)}")
        return normalized

    @field_validator("appears_in")
    @classmethod
    def _validate_appears_in(cls, value: list[str]) -> list[str]:
        return [_validate_state_id(item, label="owner_lifecycle.appears_in") for item in value]

    @field_validator("disappears_after")
    @classmethod
    def _validate_disappears_after(cls, value: str | None) -> str | None:
        if value is None:
            return value
        return _validate_state_id(value, label="owner_lifecycle.disappears_after")


class YiuStructuralOwnerSpecV4(StrictBaseModel):
    id: str
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        normalized = _validate_slug(value, label="structural_owner.id")
        if normalized not in YIU_V4_SOURCE_OWNER_IDS:
            raise ValueError(f"structural_owner.id must be one of {list(YIU_V4_SOURCE_OWNER_IDS)}")
        return normalized

    @model_validator(mode="after")
    def _validate_bounds(self) -> "YiuStructuralOwnerSpecV4":
        if self.end <= self.start:
            raise ValueError("structural_owner.end must be > structural_owner.start")
        return self


class YiuEffectTagSpecV4(StrictBaseModel):
    id: str
    class_: str = Field(alias="class")
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        return _validate_slug(value, label="effect_tag.id")

    @field_validator("class_")
    @classmethod
    def _validate_class(cls, value: str) -> str:
        normalized = _validate_slug(value, label="effect_tag.class")
        if normalized not in YIU_V4_EFFECT_TAG_CLASSES:
            raise ValueError(f"effect_tag.class must be one of {sorted(YIU_V4_EFFECT_TAG_CLASSES)}")
        return normalized

    @model_validator(mode="after")
    def _validate_bounds(self) -> "YiuEffectTagSpecV4":
        if self.end <= self.start:
            raise ValueError("effect_tag.end must be > effect_tag.start")
        return self


class SourceOligoSpecV4(StrictBaseModel):
    authored_sequence: str
    structural_owners: list[YiuStructuralOwnerSpecV4] = Field(default_factory=list)
    effect_tags: list[YiuEffectTagSpecV4] = Field(default_factory=list)

    @field_validator("authored_sequence")
    @classmethod
    def _validate_sequence(cls, value: str) -> str:
        return normalize_iupac(value)

    @model_validator(mode="after")
    def _validate_source(self) -> "SourceOligoSpecV4":
        sequence = self.authored_sequence
        if len(self.structural_owners) != len(YIU_V4_SOURCE_OWNER_IDS):
            raise ValueError(
                f"source_oligo.structural_owners must declare each canonical source owner exactly once: "
                f"{list(YIU_V4_SOURCE_OWNER_IDS)}"
            )
        owner_ids = [owner.id for owner in self.structural_owners]
        if tuple(owner_ids) != YIU_V4_SOURCE_OWNER_IDS:
            raise ValueError(
                f"source_oligo.structural_owners must appear in canonical 5'->3' order {list(YIU_V4_SOURCE_OWNER_IDS)}"
            )
        if self.structural_owners[0].start != 0:
            raise ValueError("source_oligo.structural_owners must start at coordinate 0")
        cursor = 0
        for owner in self.structural_owners:
            if owner.start != cursor:
                raise ValueError("source_oligo.structural_owners must be contiguous and gap-free")
            cursor = owner.end
        if cursor != len(sequence):
            raise ValueError("source_oligo.structural_owners must cover the entire source_oligo.authored_sequence")

        tag_ids = [tag.id for tag in self.effect_tags]
        if len(set(tag_ids)) != len(tag_ids):
            raise ValueError("source_oligo.effect_tags ids must be unique")
        for tag in self.effect_tags:
            if tag.end > len(sequence):
                raise ValueError(f"effect tag {tag.id} exceeds source_oligo.authored_sequence length")

        tag_by_id = {tag.id: tag for tag in self.effect_tags}
        required_classes = {
            "source_forward_primer_bindable": "primer_bindable_by_source_forward",
            "left_nb_bsssi_member": "nb_bsssi_array_member",
            "left_bsssi_bsai_overlap_unit": "left_bsssi_bsai_overlap_unit",
            "payload_overhang_left": "payload_overhang_left",
            "type_iis_recognition_left": "type_iis_recognition_left",
            "right_nb_bsssi_member": "nb_bsssi_array_member",
            "sacrificial_region_long_tag": "sacrificial",
            "nt_bpu10i_snapback_site": "nt_bpu10i_snapback_site",
            "payload_overhang_right": "payload_overhang_right",
            "source_reverse_primer_bindable": "primer_bindable_by_source_reverse",
            "type_iis_recognition_right": "type_iis_recognition_right",
        }
        for tag_id, expected_class in required_classes.items():
            tag = tag_by_id.get(tag_id)
            if tag is None:
                raise ValueError(f"canonical v4 source_oligo.effect_tags must include {tag_id!r}")
            if tag.class_ != expected_class:
                raise ValueError(f"effect tag {tag_id!r} must use class {expected_class!r}")

        self._validate_effect_tag_sequences()
        self._validate_overlap_legality()
        return self

    def _tag_sequence(self, tag_id: str) -> str:
        tag = next(item for item in self.effect_tags if item.id == tag_id)
        return self.authored_sequence[tag.start : tag.end]

    def _validate_effect_tag_sequences(self) -> None:
        if self._tag_sequence("left_bsssi_bsai_overlap_unit") != YIU_V4_LEFT_OVERLAP_SEQUENCE:
            raise ValueError("left_bsssi_bsai_overlap_unit must match CACGAGaGGTCTCACGAG")
        if self._tag_sequence("left_nb_bsssi_member") != YIU_V4_NB_BSSSI_MEMBER_SEQUENCE:
            raise ValueError("left_nb_bsssi_member must match CACGAG")
        if self._tag_sequence("right_nb_bsssi_member") != YIU_V4_NB_BSSSI_MEMBER_SEQUENCE:
            raise ValueError("right_nb_bsssi_member must match CACGAG")
        if self._tag_sequence("type_iis_recognition_left") != YIU_V4_TYPE_IIS_LEFT_SEQUENCE:
            raise ValueError("type_iis_recognition_left must match GGTCTC")
        if self._tag_sequence("type_iis_recognition_right") != YIU_V4_TYPE_IIS_RIGHT_SEQUENCE:
            raise ValueError("type_iis_recognition_right must match GAGACC")
        if self._tag_sequence("nt_bpu10i_snapback_site") != YIU_V4_NT_BPU10I_LOCAL_CONTEXT:
            raise ValueError("nt_bpu10i_snapback_site must match CCTCAGCCCGCTGA")
        if (
            len(self._tag_sequence("payload_overhang_left")) != 4
            or len(self._tag_sequence("payload_overhang_right")) != 4
        ):
            raise ValueError("payload overhang effect tags must be exactly 4 nt")

    def _validate_overlap_legality(self) -> None:
        owners = list(self.structural_owners)
        tags = list(self.effect_tags)
        for owner in owners:
            for tag in tags:
                if not _ranges_overlap(owner.start, owner.end, tag.start, tag.end):
                    continue
                allowed = _OWNER_TAG_COMPATIBILITY.get(owner.id, frozenset())
                if tag.class_ not in allowed:
                    raise ValueError(
                        f"overlap between structural_owner {owner.id!r} and effect_tag {tag.class_!r} is not declared"
                    )
        for index, left in enumerate(tags):
            for right in tags[index + 1 :]:
                if not _ranges_overlap(left.start, left.end, right.start, right.end):
                    continue
                if frozenset({left.class_, right.class_}) not in _TAG_TAG_COMPATIBILITY:
                    raise ValueError(
                        f"overlap between effect_tag {left.class_!r} and effect_tag {right.class_!r} is not declared"
                    )


class YiuExternalPartsV4(StrictBaseModel):
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
    def _validate_part_ids(cls, value: str) -> str:
        return _validate_slug(value, label="external_part")


class YiuEnzymesSpecV4(StrictBaseModel):
    left_type_iis: str = "BsmBI"
    right_type_iis: str = "BsmBI"
    snapback_nickase: str = "Nt.Bpu10I"
    sacrificial_nickase: str = "Nb.BssSI"

    @field_validator("left_type_iis", "right_type_iis", "snapback_nickase", "sacrificial_nickase")
    @classmethod
    def _validate_ids(cls, value: str, info) -> str:
        return _validate_slug(value, label=str(info.field_name))

    @model_validator(mode="after")
    def _validate_fixed_ids(self) -> "YiuEnzymesSpecV4":
        if self.left_type_iis != "BsmBI" or self.right_type_iis != "BsmBI":
            raise ValueError("enzymes.left_type_iis and enzymes.right_type_iis must be BsmBI")
        if self.snapback_nickase != "Nt.Bpu10I":
            raise ValueError("enzymes.snapback_nickase must be Nt.Bpu10I")
        if self.sacrificial_nickase != "Nb.BssSI":
            raise ValueError("enzymes.sacrificial_nickase must be Nb.BssSI")
        return self


class YiuPayloadSpecV4(StrictBaseModel):
    target_sequence: str
    bulge_mask: list[int] = Field(default_factory=list)

    @field_validator("target_sequence")
    @classmethod
    def _validate_sequence(cls, value: str) -> str:
        return normalize_iupac(value)

    @field_validator("bulge_mask")
    @classmethod
    def _validate_bulge_mask(cls, value: list[int]) -> list[int]:
        normalized = [int(item) for item in value]
        if len(set(normalized)) != len(normalized):
            raise ValueError("payload.bulge_mask positions must be unique")
        if any(item not in {1, 2} for item in normalized):
            raise ValueError("payload.bulge_mask positions are allowed only at indices 1 and 2")
        return normalized


class YiuProcessSpecV4(StrictBaseModel):
    schema_version: int = 4
    family: Literal["yiu"] = "yiu"
    protocol_template: Literal["yiu_circularized_payload_v1"] = "yiu_circularized_payload_v1"
    name: str
    source_oligo: SourceOligoSpecV4
    owner_lifecycle: list[YiuOwnerLifecycleEntryV4] = Field(default_factory=list)
    external_parts: YiuExternalPartsV4
    enzymes: YiuEnzymesSpecV4 = Field(default_factory=YiuEnzymesSpecV4)
    payload: YiuPayloadSpecV4
    catalogs: CatalogRefsV2 = Field(default_factory=CatalogRefsV2)
    output: YiuOutputSpecV4 = Field(default_factory=YiuOutputSpecV4)

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        if int(value) != 4:
            raise ValueError("yiu.schema_version must be 4")
        return int(value)

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        return _validate_slug(value, label="yiu.name")

    @model_validator(mode="after")
    def _validate_contract(self) -> "YiuProcessSpecV4":
        lifecycle_by_owner = {entry.owner_id: entry for entry in self.owner_lifecycle}
        if set(lifecycle_by_owner) != set(YIU_V4_OPERATIONAL_OWNER_IDS):
            raise ValueError("owner_lifecycle must declare every canonical structural owner exactly once")

        left_half = self.owner_sequence("payload_left_half")
        right_half = self.owner_sequence("payload_right_half")
        if self.payload.target_sequence != left_half + right_half:
            raise ValueError("payload.target_sequence must equal payload_left_half + payload_right_half")

        right_overhang = self.effect_sequence("payload_overhang_right")
        left_overhang = self.effect_sequence("payload_overhang_left")
        if left_overhang != reverse_complement_iupac(right_overhang):
            raise ValueError("payload overhangs must satisfy exact circularization complementarity")
        return self

    def owner_sequence(self, owner_id: str) -> str:
        owner = next(item for item in self.source_oligo.structural_owners if item.id == owner_id)
        return self.source_oligo.authored_sequence[owner.start : owner.end]

    def effect_sequence(self, tag_id: str) -> str:
        tag = next(item for item in self.source_oligo.effect_tags if item.id == tag_id)
        return self.source_oligo.authored_sequence[tag.start : tag.end]


class YiuSpecDocumentV4(StrictBaseModel):
    yiu: YiuProcessSpecV4
