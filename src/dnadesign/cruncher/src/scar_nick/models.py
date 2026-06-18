"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/scar_nick/models.py

Schema and report contracts for terminal scar-nick processing design.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dnadesign.cruncher.nickases.models import normalize_dna
from dnadesign.cruncher.scar_nick.policy import (
    ProfilePolicyStatus,
    classify_profile_policy,
)
from dnadesign.cruncher.scar_nick.policy import (
    normalize_profile as _normalize_profile,
)
from dnadesign.cruncher.scar_nick.policy import (
    profile_effective_disruption as _profile_effective_disruption,
)
from dnadesign.cruncher.scar_nick.policy import (
    profile_ligation_support as _profile_ligation_support,
)
from dnadesign.cruncher.scar_nick.semantics import PROFILE_ORDER_S3S2S1S0


class StrictScarNickModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ScarNickHeader(StrictScarNickModel):
    schema_version: Literal[1]
    contract: Literal["terminal_type_iis_scar_nick_v1"]
    name: str

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("scar_nick.name must be non-empty.")
        return text


class JunctionSpec(StrictScarNickModel):
    left_base: str
    right_base: str
    profile_order: Literal["S3_S2_S1_S0"] = PROFILE_ORDER_S3S2S1S0
    s0_match_required: bool = True
    overhang_length: int = Field(default=4, ge=1)

    @field_validator("left_base", "right_base")
    @classmethod
    def _validate_four_nt(cls, value: str) -> str:
        text = normalize_dna(value)
        if len(text) != 4:
            raise ValueError("scar-nick junction bases must be exactly 4 nt.")
        return text

    @field_validator("overhang_length")
    @classmethod
    def _validate_overhang_length(cls, value: int) -> int:
        length = int(value)
        if length != 4:
            raise ValueError("scar-nick currently supports a 4 nt terminal scar/overhang.")
        return length

    @model_validator(mode="after")
    def _validate_s0_match_required(self) -> "JunctionSpec":
        if not self.s0_match_required:
            raise ValueError("scar-nick contract requires junction.s0_match_required=true.")
        return self


class CatalogRef(StrictScarNickModel):
    preset: str | None = None
    additional_presets: list[str] = Field(default_factory=list)
    additional_paths: list[Path] = Field(default_factory=list)

    @field_validator("preset")
    @classmethod
    def _validate_optional_preset(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value or "").strip()
        if not text:
            raise ValueError("catalog.preset must be non-empty when provided.")
        return text

    @field_validator("additional_presets")
    @classmethod
    def _validate_additional_presets(cls, value: list[str]) -> list[str]:
        normalized = [str(item or "").strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("catalog.additional_presets must not contain blank values.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("catalog.additional_presets must be unique.")
        return normalized

    @model_validator(mode="after")
    def _validate_has_source(self) -> "CatalogRef":
        if self.preset is None and not self.additional_presets and not self.additional_paths:
            raise ValueError("catalog must define a preset, additional_presets, or additional_paths.")
        return self


class ReleaseProcessingSpec(StrictScarNickModel):
    variant_id: str
    catalog: CatalogRef
    required_terminal_scar_nt: int = Field(default=4, ge=1)
    recognition_site_must_be_excised: bool = True

    @field_validator("variant_id")
    @classmethod
    def _validate_variant_id(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("processing.release.variant_id must be non-empty.")
        return text


class NickProcessingSpec(StrictScarNickModel):
    target_strand: Literal["top", "bottom", "either"] = "either"
    terminal_nick_required: bool = True
    downstream_protected_nt_allowed: Literal[0] = 0
    downstream_must_be_degenerate: Literal[True] = True
    catalog: CatalogRef

    @model_validator(mode="after")
    def _validate_terminal_nick_required(self) -> "NickProcessingSpec":
        if not self.terminal_nick_required:
            raise ValueError("scar-nick contract requires processing.nick.terminal_nick_required=true.")
        return self


class ProcessingSpec(StrictScarNickModel):
    release: ReleaseProcessingSpec
    nick: NickProcessingSpec


class ReferenceProfile(StrictScarNickModel):
    id: str
    left_base: str
    right_base: str
    profile_s3s2s1s0: str

    @field_validator("id")
    @classmethod
    def _validate_id(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("reference profile ids must be non-empty.")
        return text

    @field_validator("left_base", "right_base")
    @classmethod
    def _validate_base(cls, value: str) -> str:
        text = normalize_dna(value)
        if len(text) != 4:
            raise ValueError("reference profile bases must be exactly 4 nt.")
        return text

    @field_validator("profile_s3s2s1s0")
    @classmethod
    def _validate_profile(cls, value: str) -> str:
        return _normalize_profile(value)


class CandidateRankingContext(StrictScarNickModel):
    optional_reference_profiles: dict[str, ReferenceProfile] = Field(default_factory=dict)
    anchor_mode: Literal["exact_left", "exact_right", "profile_analog", "surviving_strand"] = "profile_analog"
    target_profile_buckets: list[str] = Field(default_factory=list)
    reject_profiles: list[str] = Field(default_factory=list)
    reserve_profiles: list[str] = Field(default_factory=list)
    allow_gt_wobble: bool = True
    active_max_hard_mismatches: int = Field(default=2, ge=0, le=4)
    active_max_non_watson_crick_pairs: int = Field(default=2, ge=0, le=4)
    forbid_active_middle_middle_double_hard: bool = True
    min_ligation_support: float = Field(default=2.0, ge=0.0, le=4.0)
    max_effective_disruption: float = Field(default=2.5, ge=0.0, le=4.0)
    prefer_lower_middle_hard_mismatch_tier: bool = True
    prefer_lower_hard_mismatch_tier: bool = True
    reduce_gc_when_tied: bool = True

    @field_validator("target_profile_buckets", "reject_profiles", "reserve_profiles")
    @classmethod
    def _validate_profiles(cls, value: list[str]) -> list[str]:
        normalized = [_normalize_profile(item) for item in value]
        if len(set(normalized)) != len(normalized):
            raise ValueError("profile lists must not repeat values.")
        return normalized

    @model_validator(mode="after")
    def _validate_reference_profile_semantics(self) -> "CandidateRankingContext":
        from dnadesign.cruncher.scar_nick.profiles import classify_pair_profile

        targeted = set(self.target_profile_buckets)
        rejected = set(self.reject_profiles)
        reserve = set(self.reserve_profiles)
        rejected_conflicts = sorted(rejected & targeted)
        if rejected_conflicts:
            raise ValueError("ranking profiles cannot be both rejected and targeted: " + ", ".join(rejected_conflicts))
        reserve_conflicts = sorted(reserve & targeted)
        if reserve_conflicts:
            raise ValueError("ranking profiles cannot be both reserved and targeted: " + ", ".join(reserve_conflicts))
        reject_reserve_conflicts = sorted(rejected & reserve)
        if reject_reserve_conflicts:
            raise ValueError(
                "ranking profiles cannot be both rejected and reserved: " + ", ".join(reject_reserve_conflicts)
            )

        for label, reference in sorted(self.optional_reference_profiles.items()):
            observed = classify_pair_profile(
                reference.left_base,
                reference.right_base,
                allow_gt_wobble=self.allow_gt_wobble,
            )
            if observed.profile_s3s2s1s0 != reference.profile_s3s2s1s0:
                raise ValueError(
                    "reference profile mismatch for "
                    f"{label!r}: declared {reference.profile_s3s2s1s0}, "
                    f"observed {observed.profile_s3s2s1s0}."
                )
        return self

    @model_validator(mode="after")
    def _validate_target_profiles_fit_hard_gates(self) -> "CandidateRankingContext":
        invalid_profiles: list[str] = []
        for profile in self.target_profile_buckets:
            reasons: list[str] = []
            decision = classify_profile_policy(profile, context=self)
            if decision.status != "active":
                reasons.append(f"profile_policy={decision.status}:{decision.reason}")
            ligation_support = _profile_ligation_support(profile)
            if ligation_support < self.min_ligation_support:
                reasons.append(f"ligation_support {ligation_support:.1f} < min_ligation_support")
            effective_disruption = _profile_effective_disruption(profile)
            if effective_disruption > self.max_effective_disruption:
                reasons.append(f"effective_disruption {effective_disruption:.1f} > max_effective_disruption")
            if reasons:
                invalid_profiles.append(f"{profile} ({', '.join(reasons)})")

        if invalid_profiles:
            raise ValueError(
                "target_profile_buckets conflict with scar-nick hard gates: " + "; ".join(invalid_profiles)
            )
        return self


class SearchSpec(StrictScarNickModel):
    mode: Literal["curated_panel"] = "curated_panel"
    max_hits: int = Field(default=16, ge=1)
    materialize_top_k: int = Field(default=8, ge=0)
    min_nickase_recognition_nt: int = Field(default=4, ge=4)
    disallowed_nickase_warning_codes: list[str] = Field(default_factory=lambda: ["FREQUENT_CUTTER"])

    @field_validator("disallowed_nickase_warning_codes")
    @classmethod
    def _validate_disallowed_nickase_warning_codes(cls, value: list[str]) -> list[str]:
        normalized = [str(item or "").strip().upper() for item in value]
        if any(not item for item in normalized):
            raise ValueError("search.disallowed_nickase_warning_codes must not contain blank values.")
        if len(set(normalized)) != len(normalized):
            raise ValueError("search.disallowed_nickase_warning_codes must be unique.")
        return normalized

    @model_validator(mode="after")
    def _validate_materialize_bound(self) -> "SearchSpec":
        if self.materialize_top_k > self.max_hits:
            raise ValueError("search.materialize_top_k must be <= search.max_hits.")
        return self


class OutputSpec(StrictScarNickModel):
    run_dir: Path

    @field_validator("run_dir")
    @classmethod
    def _validate_run_dir(cls, value: Path) -> Path:
        path = Path(value)
        if path.is_absolute():
            raise ValueError("output.run_dir must be a relative path inside the workspace.")
        if any(part == ".." for part in path.parts):
            raise ValueError("output.run_dir must not traverse outside the workspace.")
        if not str(path).strip():
            raise ValueError("output.run_dir must be non-empty.")
        if len(path.parts) < 3 or path.parts[0] != "outputs" or path.parts[1] != "scar_nick":
            raise ValueError("output.run_dir must live under outputs/scar_nick/<name>.")
        return path


class ScarNickSpecDocument(StrictScarNickModel):
    scar_nick: ScarNickHeader
    junction: JunctionSpec
    processing: ProcessingSpec
    ranking_context: CandidateRankingContext
    search: SearchSpec
    output: OutputSpec

    @model_validator(mode="after")
    def _validate_bucket_capacity(self) -> "ScarNickSpecDocument":
        target_count = len(self.ranking_context.target_profile_buckets)
        if target_count and self.search.max_hits < target_count:
            raise ValueError("search.max_hits must be >= the number of target_profile_buckets.")
        return self


class ValidationIssue(StrictScarNickModel):
    code: str
    message: str
    details: dict[str, Any] = Field(default_factory=dict)


class PairClass(StrictScarNickModel):
    position: int = Field(ge=0, le=3)
    site: Literal["S3", "S2", "S1", "S0"]
    source_offset: int = Field(ge=0, le=3)
    left_base: str
    right_base: str
    aligned_right_base: str
    class_label: Literal["M", "W", "X"]
    position_class: Literal["edge", "middle"]
    canonical_mismatch_class: str | None = None
    class_tier_t4: int = Field(default=0, ge=0)

    @field_validator("left_base", "right_base", "aligned_right_base")
    @classmethod
    def _validate_base_symbol(cls, value: str) -> str:
        text = normalize_dna(value)
        if len(text) != 1:
            raise ValueError("pair class bases must be one A/C/G/T base.")
        return text

    @model_validator(mode="after")
    def _validate_pair_label_matches_pair(self) -> "PairClass":
        if self.class_label == "M" and self.left_base != self.aligned_right_base:
            raise ValueError("M pair classes require identical left_base and aligned_right_base.")
        if self.class_label == "W" and (self.left_base, self.right_base) not in {("G", "T"), ("T", "G")}:
            raise ValueError("W pair classes require a G:T or T:G physical pair.")
        if self.class_label == "X" and self.left_base == self.aligned_right_base:
            raise ValueError("X pair classes require a non-identical aligned pair.")
        return self


class PairProfile(StrictScarNickModel):
    profile_s3s2s1s0: str
    profile_payload_outward: str
    pairs: list[PairClass]
    hard_mismatch_count: int = Field(ge=0, le=4)
    wobble_count: int = Field(ge=0, le=4)
    non_watson_crick_count: int = Field(ge=0, le=4)
    watson_crick_count: int = Field(ge=0, le=4)
    middle_hard_count: int = Field(ge=0, le=2)
    middle_wobble_count: int = Field(ge=0, le=2)
    worst_hard_mismatch_tier: int = Field(default=0, ge=0)
    hard_mismatch_tier_sum: int = Field(default=0, ge=0)
    middle_hard_mismatch_tier_sum: int = Field(default=0, ge=0)
    edge_hard_mismatch_tier_sum: int = Field(default=0, ge=0)
    ligation_support: float = Field(ge=0.0, le=4.0)
    effective_disruption: float = Field(ge=0.0, le=4.0)
    s0_class: Literal["M", "W", "X"]


class ReleasePlacement(StrictScarNickModel):
    variant_id: str
    orientation: Literal["forward"]
    recognition_sequence: str
    source_catalog_id: str
    source_url: str
    commercial_confidence: str
    warning_codes: list[str] = Field(default_factory=list)
    recognition_site_start: int
    recognition_site_end: int
    top_cut_boundary: int
    bottom_cut_boundary: int
    retained_scar_start: int
    retained_scar_end: int
    retained_scar_nt: int = Field(ge=1)
    recognition_site_excised: bool


class NickasePlacement(StrictScarNickModel):
    variant_id: str
    specificity_id: str
    orientation: str
    motif_top_5to3: str
    canonical_motif_top_5to3: str
    vendor: str
    source_url: str
    source_family: Literal["nicking_endonuclease"]
    commercial_confidence: str
    warning_codes: list[str] = Field(default_factory=list)
    source_site_start: int
    source_site_end: int
    strand: Literal["top", "bottom"]
    boundary: int
    terminal_boundary: int
    boundary_distance: int = Field(ge=0)
    exact_terminal: bool


class NickaseDownstreamSymbol(StrictScarNickModel):
    raw_coordinate: int
    symbol: str
    fully_degenerate: bool


class NickaseReleaseOverlapConflict(StrictScarNickModel):
    raw_coordinate: int
    nickase_symbol: str
    release_symbol: str


class RetainedScarDomain(StrictScarNickModel):
    raw_coordinate: int
    bases: list[str]


class NickaseGeometryAuditEntry(StrictScarNickModel):
    variant_id: str
    specificity_id: str
    orientation: str | None = None
    motif_top_5to3: str
    terminal_candidate: bool = False
    source_site_start: int | None = None
    source_site_end: int | None = None
    boundary: int | None = None
    terminal_boundary: int
    exact_terminal: bool = False
    strand: Literal["top", "bottom"] | None = None
    policy_rejection_reasons: list[str] = Field(default_factory=list)
    rejection_reasons: list[str] = Field(default_factory=list)
    downstream_symbols: list[NickaseDownstreamSymbol] = Field(default_factory=list)
    release_overlap_conflicts: list[NickaseReleaseOverlapConflict] = Field(default_factory=list)
    retained_scar_domains: list[RetainedScarDomain] = Field(default_factory=list)
    feasible_scar_count: int = Field(default=0, ge=0)
    upstream_flank_sequence: str = ""
    type_iis_offset_sequence: str = ""
    compatible: bool = False


class ScarNickCandidate(StrictScarNickModel):
    rank: int | None = None
    candidate_id: str
    left_base: str
    right_base: str
    retained_scar: str
    retained_product_sequence: str
    profile_s3s2s1s0: str
    profile_payload_outward: str
    profile_order: Literal["S3_S2_S1_S0"] = PROFILE_ORDER_S3S2S1S0
    profile_policy_status: ProfilePolicyStatus
    profile_policy_reason: str
    s0_match_required: bool = True
    pair_classes: list[PairClass]
    s3_pair_identity: str
    s2_pair_identity: str
    s1_pair_identity: str
    s0_pair_identity: str
    m_count: int = Field(ge=0, le=4)
    w_count: int = Field(ge=0, le=4)
    x_count: int = Field(ge=0, le=4)
    non_watson_crick_count: int = Field(ge=0, le=4)
    middle_hard_count: int = Field(ge=0, le=2)
    middle_wobble_count: int = Field(ge=0, le=2)
    worst_hard_mismatch_tier: int = Field(default=0, ge=0)
    hard_mismatch_tier_sum: int = Field(default=0, ge=0)
    middle_hard_mismatch_tier_sum: int = Field(default=0, ge=0)
    edge_hard_mismatch_tier_sum: int = Field(default=0, ge=0)
    ligation_support: float = Field(ge=0.0, le=4.0)
    effective_disruption: float = Field(ge=0.0, le=4.0)
    tnna_flag: bool = False
    nicked_strand: Literal["top", "bottom"] | None = None
    surviving_strand: Literal["top", "bottom"] | None = None
    retained_scar_source: str
    discarded_strand_enzyme_burden: str | None = None
    release_placement: ReleasePlacement | None = None
    retained_scar_nt: int = Field(default=4, ge=1)
    nickase_placement: NickasePlacement | None = None
    nickase_site: str | None = None
    nick_boundary: int
    terminal_boundary: int
    nick_distance: int = Field(ge=0)
    gc_fraction: float = Field(ge=0.0, le=1.0)
    reference_control_distance: int | None = None
    reference_distances: dict[str, int] = Field(default_factory=dict)
    rejection_reasons: list[str] = Field(default_factory=list)
    rank_key: list[Any] = Field(default_factory=list)


class ScarNickReportMetadata(StrictScarNickModel):
    spec_schema_version: int
    contract: str
    terminal_boundary: int
    release_variant_id: str
    nick_target_strand: Literal["top", "bottom", "either"]
    release_catalog_preset_ids: list[str] = Field(default_factory=list)
    nickase_catalog_preset_ids: list[str] = Field(default_factory=list)
    enumerated_candidate_count: int = Field(default=0, ge=0)
    accepted_candidate_count: int = Field(default=0, ge=0)
    materialized_candidate_count: int = Field(default=0, ge=0)
    compatible_nickase_placement_count: int = Field(default=0, ge=0)
    enzyme_compatible_scar_count: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)


class ScarNickEvaluationReport(StrictScarNickModel):
    schema_version: Literal[1] = 1
    workflow: Literal["scar_nick"] = "scar_nick"
    status: Literal["satisfied", "unsatisfied"]
    spec_name: str
    workspace_root: str
    spec_path: str
    run_dir: str | None = None
    metadata: ScarNickReportMetadata
    release_placement: ReleasePlacement | None = None
    issues: list[ValidationIssue] = Field(default_factory=list)
    nickase_geometry_audit: list[NickaseGeometryAuditEntry] = Field(default_factory=list)
    candidates: list[ScarNickCandidate] = Field(default_factory=list)
    reserve_candidates: list[ScarNickCandidate] = Field(default_factory=list)
    rejected_reference_candidates: list[ScarNickCandidate] = Field(default_factory=list)


__all__ = [
    "CandidateRankingContext",
    "CatalogRef",
    "JunctionSpec",
    "NickaseDownstreamSymbol",
    "NickaseGeometryAuditEntry",
    "NickasePlacement",
    "NickaseReleaseOverlapConflict",
    "PairClass",
    "PairProfile",
    "ProcessingSpec",
    "ReferenceProfile",
    "ReleasePlacement",
    "RetainedScarDomain",
    "ScarNickCandidate",
    "ScarNickEvaluationReport",
    "ScarNickReportMetadata",
    "ScarNickSpecDocument",
    "SearchSpec",
    "ValidationIssue",
]
