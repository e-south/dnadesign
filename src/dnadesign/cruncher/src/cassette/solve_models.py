"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cassette/solve_models.py

Schema and report contracts for cassette solve/search workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator

from dnadesign.cruncher.cassette.models import (
    BoundedSegmentLength,
    CatalogNormalizationInfo,
    NickWindow,
    StrictCassetteModel,
    ValidationIssue,
    normalize_dna,
    normalize_iupac,
)

MAX_SOLVE_STEM_LENGTH_NT = 64
MAX_SOLVE_LOOP_LENGTH_NT = 32
MAX_SOLVE_ASSIGNMENT_PAIR_SPACE = 256
MAX_SOLVE_MAX_HITS = 128
MAX_SOLVE_MAX_ENUMERATED_CANDIDATES = 250000
MAX_SOLVE_MAX_SEARCH_NODES = 500000
MAX_SOLVE_MAX_MATERIALIZE_TOP_K = 32
DEFAULT_SELECTION_POOL_MIN = 64
DEFAULT_SELECTION_POOL_MULTIPLIER = 8


def _duplicate_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


class HairpinCassetteSolveTopologySpec(StrictCassetteModel):
    stem5p_arm_pattern: str
    loop_pattern: str

    @field_validator("stem5p_arm_pattern", "loop_pattern")
    @classmethod
    def _validate_pattern(cls, value: str) -> str:
        return normalize_iupac(value)

    @model_validator(mode="after")
    def _validate_lengths(self) -> "HairpinCassetteSolveTopologySpec":
        if len(self.stem5p_arm_pattern) == 0:
            raise ValueError("topology.stem5p_arm_pattern must be non-empty.")
        if len(self.loop_pattern) == 0:
            raise ValueError("topology.loop_pattern must be non-empty.")
        if len(self.stem5p_arm_pattern) > MAX_SOLVE_STEM_LENGTH_NT:
            raise ValueError(
                "topology.stem5p_arm_pattern exceeds the current first-phase solve safety limit "
                f"({MAX_SOLVE_STEM_LENGTH_NT} nt)."
            )
        if len(self.loop_pattern) > MAX_SOLVE_LOOP_LENGTH_NT:
            raise ValueError(
                "topology.loop_pattern exceeds the current first-phase solve safety limit "
                f"({MAX_SOLVE_LOOP_LENGTH_NT} nt)."
            )
        return self


class SolveConstructContextSpec(StrictCassetteModel):
    left_flank: str = ""
    right_flank: str = ""
    evaluation_scope: Literal["cassette_plus_flanks"] = "cassette_plus_flanks"

    @field_validator("left_flank", "right_flank")
    @classmethod
    def _validate_dna(cls, value: str) -> str:
        return normalize_dna(value, allow_empty=True)


class NickGoalSpec(StrictCassetteModel):
    target_strand: Literal["primary", "complement"]
    left_nick_window: NickWindow
    right_nick_window: NickWindow
    bounded_segment_length: BoundedSegmentLength | None = None


class AssignmentPolicySpec(StrictCassetteModel):
    allowed_left_variant_ids: list[str]
    allowed_right_variant_ids: list[str]
    forbidden_intended_variant_ids: list[str] = Field(default_factory=list)
    forbidden_intended_specificity_ids: list[str] = Field(default_factory=list)
    allow_same_variant: bool = True
    allow_same_specificity_opposite_variant: bool = True

    @field_validator(
        "allowed_left_variant_ids",
        "allowed_right_variant_ids",
        "forbidden_intended_variant_ids",
        "forbidden_intended_specificity_ids",
    )
    @classmethod
    def _validate_ids(cls, value: list[str]) -> list[str]:
        normalized = [str(item or "").strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("assignment policy ids must be non-empty strings.")
        duplicates = _duplicate_values(normalized)
        if duplicates:
            raise ValueError(f"assignment policy ids must not repeat values: {', '.join(duplicates)}")
        return normalized

    @model_validator(mode="after")
    def _validate_allowed_lists(self) -> "AssignmentPolicySpec":
        if not self.allowed_left_variant_ids:
            raise ValueError("assignment_policy.allowed_left_variant_ids must be non-empty.")
        if not self.allowed_right_variant_ids:
            raise ValueError("assignment_policy.allowed_right_variant_ids must be non-empty.")
        pair_space = len(self.allowed_left_variant_ids) * len(self.allowed_right_variant_ids)
        if pair_space > MAX_SOLVE_ASSIGNMENT_PAIR_SPACE:
            raise ValueError(
                "assignment pair space exceeds the current first-phase solve safety limit "
                f"({pair_space} > {MAX_SOLVE_ASSIGNMENT_PAIR_SPACE}). "
                "Narrow assignment_policy.allowed_left_variant_ids or allowed_right_variant_ids."
            )
        return self


class SiteBlacklistPolicySpec(StrictCassetteModel):
    forbidden_any_site_specificity_ids: list[str] = Field(default_factory=list)
    forbidden_unintended_site_specificity_ids: list[str] = Field(default_factory=list)
    forbidden_any_site_variant_ids: list[str] = Field(default_factory=list)
    scope: Literal["cassette_only", "evaluation_context"] = "evaluation_context"

    @field_validator(
        "forbidden_any_site_specificity_ids",
        "forbidden_unintended_site_specificity_ids",
        "forbidden_any_site_variant_ids",
    )
    @classmethod
    def _validate_ids(cls, value: list[str]) -> list[str]:
        normalized = [str(item or "").strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("site blacklist ids must be non-empty strings.")
        return normalized


class SequenceBlacklistPolicySpec(StrictCassetteModel):
    forbidden_literals: list[str] = Field(default_factory=list)
    forbidden_iupac_motifs: list[str] = Field(default_factory=list)
    forbid_reverse_complements: bool = True
    scope: Literal["cassette_only", "evaluation_context"] = "evaluation_context"

    @field_validator("forbidden_literals")
    @classmethod
    def _validate_literals(cls, value: list[str]) -> list[str]:
        return [normalize_dna(item) for item in value]

    @field_validator("forbidden_iupac_motifs")
    @classmethod
    def _validate_motifs(cls, value: list[str]) -> list[str]:
        return [normalize_iupac(item) for item in value]


class GCFractionRange(StrictCassetteModel):
    min: float = Field(ge=0.0, le=1.0)
    max: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_bounds(self) -> "GCFractionRange":
        if self.max < self.min:
            raise ValueError("gc_fraction.max must be >= gc_fraction.min.")
        return self


class SequenceQualitySpec(StrictCassetteModel):
    gc_fraction: GCFractionRange | None = None
    max_homopolymer_run: int | None = Field(default=None, ge=1)


class SolveCatalogConfig(StrictCassetteModel):
    preset: str | None = None
    additional_paths: list[Path] = Field(default_factory=list)

    @field_validator("preset")
    @classmethod
    def _validate_preset(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            raise ValueError("catalog.preset must be non-empty when provided.")
        return text

    @field_validator("additional_paths")
    @classmethod
    def _validate_paths(cls, value: list[Path]) -> list[Path]:
        paths = [Path(path) for path in value]
        if any(not str(path).strip() for path in paths):
            raise ValueError("catalog.additional_paths must not contain empty paths.")
        return paths

    @model_validator(mode="after")
    def _validate_sources(self) -> "SolveCatalogConfig":
        if self.preset is None and not self.additional_paths:
            raise ValueError("catalog must define at least one of preset or additional_paths.")
        return self


def _default_selection_pool_size(*, max_hits: int, materialize_top_k: int) -> int:
    return max(
        DEFAULT_SELECTION_POOL_MIN,
        max_hits * DEFAULT_SELECTION_POOL_MULTIPLIER,
        materialize_top_k * DEFAULT_SELECTION_POOL_MULTIPLIER,
    )


class SearchSelectionSpec(StrictCassetteModel):
    policy: Literal["score_only", "greedy_hamming", "mmr"] = "greedy_hamming"
    pool_size: int | None = Field(default=None, ge=1)
    distance_metric: Literal["hamming"] = "hamming"
    min_pairwise_distance: int = Field(ge=0, default=0)
    diversity_weight: float | None = Field(default=None, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_policy_fields(self) -> "SearchSelectionSpec":
        if self.policy == "score_only":
            if self.diversity_weight is not None:
                raise ValueError("search.selection.diversity_weight must be omitted when policy=score_only.")
            if self.min_pairwise_distance != 0:
                raise ValueError("search.selection.min_pairwise_distance must be 0 when policy=score_only.")
        if self.policy == "greedy_hamming" and self.distance_metric != "hamming":
            raise ValueError("search.selection.distance_metric must be hamming when policy=greedy_hamming.")
        if self.policy != "mmr" and self.diversity_weight is not None:
            raise ValueError("search.selection.diversity_weight must be omitted unless policy=mmr.")
        if self.policy == "mmr" and self.diversity_weight is None:
            raise ValueError("search.selection.diversity_weight is required when policy=mmr.")
        return self


class SearchSettingsSpec(StrictCassetteModel):
    max_hits: int = Field(ge=1, default=25)
    max_enumerated_candidates: int = Field(ge=1, default=100000)
    max_search_nodes: int = Field(ge=1, default=250000)
    bounded_segment_target: int | None = Field(default=None, ge=0)
    gc_target: float | None = Field(default=None, ge=0.0, le=1.0)
    materialize_top_k: int = Field(ge=0, default=5)
    selection: SearchSelectionSpec = Field(default_factory=SearchSelectionSpec)
    selection_policy_defaulted: bool = Field(default=False, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _normalize_selection(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        normalized = dict(value)
        legacy_min_pairwise_distance = normalized.pop("min_pairwise_hamming_distance", None)
        raw_selection = normalized.get("selection")
        selection_payload: dict[str, object]
        if raw_selection is None:
            selection_payload = {}
            normalized["selection_policy_defaulted"] = True
        elif isinstance(raw_selection, dict):
            selection_payload = dict(raw_selection)
            normalized["selection_policy_defaulted"] = False
        else:
            return value
        if legacy_min_pairwise_distance is not None:
            if (
                "min_pairwise_distance" in selection_payload
                and selection_payload["min_pairwise_distance"] != legacy_min_pairwise_distance
            ):
                raise ValueError(
                    "search.min_pairwise_hamming_distance conflicts with search.selection.min_pairwise_distance."
                )
            selection_payload.setdefault("min_pairwise_distance", legacy_min_pairwise_distance)
        normalized["selection"] = selection_payload
        return normalized

    @model_validator(mode="after")
    def _validate_bounds(self) -> "SearchSettingsSpec":
        if self.materialize_top_k > self.max_hits:
            raise ValueError("search.materialize_top_k must be <= search.max_hits.")
        if self.max_hits > MAX_SOLVE_MAX_HITS:
            raise ValueError(
                "search.max_hits exceeds the current first-phase solve safety limit "
                f"({self.max_hits} > {MAX_SOLVE_MAX_HITS})."
            )
        if self.max_enumerated_candidates > MAX_SOLVE_MAX_ENUMERATED_CANDIDATES:
            raise ValueError(
                "search.max_enumerated_candidates exceeds the current first-phase solve safety limit "
                f"({self.max_enumerated_candidates} > {MAX_SOLVE_MAX_ENUMERATED_CANDIDATES})."
            )
        if self.max_search_nodes > MAX_SOLVE_MAX_SEARCH_NODES:
            raise ValueError(
                "search.max_search_nodes exceeds the current first-phase solve safety limit "
                f"({self.max_search_nodes} > {MAX_SOLVE_MAX_SEARCH_NODES})."
            )
        if self.materialize_top_k > MAX_SOLVE_MAX_MATERIALIZE_TOP_K:
            raise ValueError(
                "search.materialize_top_k exceeds the current first-phase solve safety limit "
                f"({self.materialize_top_k} > {MAX_SOLVE_MAX_MATERIALIZE_TOP_K})."
            )
        resolved_pool_size = self.selection.pool_size or _default_selection_pool_size(
            max_hits=self.max_hits,
            materialize_top_k=self.materialize_top_k,
        )
        if resolved_pool_size < self.max_hits:
            raise ValueError("search.selection.pool_size must be >= search.max_hits.")
        self.selection = self.selection.model_copy(update={"pool_size": resolved_pool_size})
        return self


class SolveOutputConfig(StrictCassetteModel):
    run_dir: Path = Path("outputs/cassette_solves")
    write_render_contract: bool = True

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
        return path


class HairpinCassetteSolveSpec(StrictCassetteModel):
    schema_version: int
    topology: HairpinCassetteSolveTopologySpec
    construct_context: SolveConstructContextSpec = Field(default_factory=SolveConstructContextSpec)
    nick_goal: NickGoalSpec
    assignment_policy: AssignmentPolicySpec
    site_blacklist: SiteBlacklistPolicySpec = Field(default_factory=SiteBlacklistPolicySpec)
    sequence_blacklist: SequenceBlacklistPolicySpec = Field(default_factory=SequenceBlacklistPolicySpec)
    sequence_quality: SequenceQualitySpec = Field(default_factory=SequenceQualitySpec)
    catalog: SolveCatalogConfig
    search: SearchSettingsSpec = Field(default_factory=SearchSettingsSpec)
    output: SolveOutputConfig = Field(default_factory=SolveOutputConfig)

    @field_validator("schema_version")
    @classmethod
    def _validate_schema_version(cls, value: int) -> int:
        version = int(value)
        if version != 1:
            raise ValueError("cassette_solve.schema_version must be 1.")
        return version

    @property
    def stem_length_nt(self) -> int:
        return len(self.topology.stem5p_arm_pattern)

    @property
    def loop_length_nt(self) -> int:
        return len(self.topology.loop_pattern)

    @property
    def cassette_length_nt(self) -> int:
        return (2 * self.stem_length_nt) + self.loop_length_nt


class HairpinCassetteSolveSpecDocument(StrictCassetteModel):
    cassette_solve: HairpinCassetteSolveSpec


class CandidateScoreBreakdown(StrictCassetteModel):
    extra_site_count: int = Field(ge=0)
    bounded_segment_distance: float = Field(ge=0.0)
    gc_distance: float = Field(ge=0.0)
    homopolymer_penalty: int = Field(ge=0)


class CandidateHit(StrictCassetteModel):
    rank: int = Field(ge=1)
    score: list[float | int | str]
    base_penalty_vector: list[float | int]
    hit_id: str
    cassette_sequence: str
    stem5p_arm: str
    loop: str
    left_variant_id: str
    right_variant_id: str
    left_nick_boundary: int
    right_nick_boundary: int
    target_strand: Literal["primary", "complement"]
    bounded_segment_length: int = Field(ge=0)
    extra_site_count: int = Field(ge=0)
    gc_fraction: float = Field(ge=0.0, le=1.0)
    score_breakdown: CandidateScoreBreakdown
    selection_rank_reason: str | None = None
    distance_to_previous_selected: float | None = Field(default=None, ge=0.0)
    report_status: Literal["satisfied"] = "satisfied"
    materialized_run_dir: str | None = None

    @field_validator("cassette_sequence", "stem5p_arm", "loop")
    @classmethod
    def _validate_sequence(cls, value: str) -> str:
        return normalize_dna(value)


class SolveReportMetadata(StrictCassetteModel):
    catalog_variants: list[CatalogNormalizationInfo] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    warning_codes: list[str] = Field(default_factory=list)
    enumerated_candidate_count: int = Field(ge=0, default=0)
    accepted_candidate_count: int = Field(ge=0, default=0)
    considered_variant_pair_count: int = Field(ge=0, default=0)
    visited_search_node_count: int = Field(ge=0, default=0)
    materialized_hit_count: int = Field(ge=0, default=0)
    catalog_preset: str | None = None
    catalog_additional_paths: list[str] = Field(default_factory=list)


class PairwiseDistanceSummary(StrictCassetteModel):
    min: float | None = None
    max: float | None = None
    mean: float | None = None


class SolveSelectionSummary(StrictCassetteModel):
    policy: Literal["score_only", "greedy_hamming", "mmr"]
    distance_metric: Literal["hamming"]
    diversity_weight: float | None = Field(default=None, ge=0.0, le=1.0)
    max_hits: int = Field(ge=0)
    pool_size: int = Field(ge=0)
    accepted_candidate_count: int = Field(ge=0)
    accepted_pool_size: int = Field(ge=0)
    accepted_pool_admitted_count: int = Field(ge=0)
    accepted_pool_rejected_count: int = Field(ge=0)
    accepted_pool_truncated: bool = False
    accepted_pool_worst_score_at_close: list[float | int | str] | None = None
    search_truncated: bool = False
    selected_hit_count: int = Field(ge=0)
    selected_hit_ids: list[str] = Field(default_factory=list)
    selection_policy_defaulted: bool = False
    selection_pool_non_exhaustive_reason: str | None = None
    policy_limited_hit_count: int = Field(default=0, ge=0)
    policy_underfilled: bool = False
    policy_underfilled_reason: str | None = None
    pairwise_distance_summary: PairwiseDistanceSummary = Field(default_factory=PairwiseDistanceSummary)


class SolveReport(StrictCassetteModel):
    schema_version: int = 1
    workflow: Literal["cassette_solve"] = "cassette_solve"
    status: Literal["solved", "no_hits", "invalid_spec", "invalid_catalog"]
    workspace_root: str | None = None
    spec_path: str
    solve_id: str | None = None
    run_dir: str | None = None
    metadata: SolveReportMetadata = Field(default_factory=SolveReportMetadata)
    issues: list[ValidationIssue] = Field(default_factory=list)
    hits: list[CandidateHit] = Field(default_factory=list)
    selection_summary: SolveSelectionSummary | None = None
    materialized_hit_runs: list[str] = Field(default_factory=list)
    render_contracts_written: bool = False
    baserender_hits_contract_path: str | None = None
    notes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_status(self) -> "SolveReport":
        if self.status == "solved" and not self.hits:
            raise ValueError("Solved reports must include at least one hit.")
        if self.status == "no_hits" and self.hits:
            raise ValueError("no_hits reports must not include hits.")
        return self
