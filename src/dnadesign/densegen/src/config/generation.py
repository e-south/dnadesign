"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/config/generation.py

DenseGen generation and constraint schemas.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Literal


class PromoterConstraint(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Optional[str] = None
    upstream: Optional[str] = None
    downstream: Optional[str] = None
    spacer_length: Optional[tuple[int, int]] = None
    upstream_pos: Optional[tuple[int, int]] = None
    downstream_pos: Optional[tuple[int, int]] = None

    @field_validator("upstream", "downstream")
    @classmethod
    def _motif_ok(cls, v: Optional[str], info):
        if v is None:
            return v
        if not isinstance(v, str):
            raise ValueError(f"{info.field_name} must be a string")
        s = v.strip().upper()
        if not s:
            raise ValueError(f"{info.field_name} must be a non-empty motif string")
        if any(ch not in {"A", "C", "G", "T"} for ch in s):
            raise ValueError(f"{info.field_name} must contain only A/C/G/T characters")
        return s

    @field_validator("spacer_length", "upstream_pos", "downstream_pos")
    @classmethod
    def _pair_ok(cls, v: Optional[tuple[int, int]]):
        if v is None:
            return v
        if len(v) != 2:
            raise ValueError("Expected a 2-tuple (min, max)")
        lo, hi = v
        if lo < 0 or hi < 0:
            raise ValueError("Range bounds must be >= 0")
        if lo > hi:
            raise ValueError("Range must be min <= max")
        return v


class SideBiases(BaseModel):
    model_config = ConfigDict(extra="forbid")
    left: List[str] = Field(default_factory=list)
    right: List[str] = Field(default_factory=list)

    @field_validator("left", "right")
    @classmethod
    def _motifs_ok(cls, v: List[str], info):
        motifs = []
        for motif in v:
            if not isinstance(motif, str):
                raise ValueError(f"{info.field_name} motifs must be strings")
            s = motif.strip().upper()
            if not s:
                raise ValueError(f"{info.field_name} motifs must be non-empty")
            if any(ch not in {"A", "C", "G", "T"} for ch in s):
                raise ValueError(f"{info.field_name} motifs must contain only A/C/G/T characters")
            motifs.append(s)
        if len(set(motifs)) != len(motifs):
            raise ValueError(f"{info.field_name} motifs must be unique")
        return motifs

    @model_validator(mode="after")
    def _left_right_disjoint(self):
        overlap = set(self.left) & set(self.right)
        if overlap:
            joined = ", ".join(sorted(overlap))
            raise ValueError(f"side_biases left/right overlap: {joined}")
        return self


class FixedElements(BaseModel):
    model_config = ConfigDict(extra="forbid")
    promoter_constraints: List[PromoterConstraint] = Field(default_factory=list)
    side_biases: Optional[SideBiases] = None


class RegulatorGroup(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    members: List[str]
    min_required: int = 1

    @field_validator("name")
    @classmethod
    def _group_name_ok(cls, v: str):
        text = str(v).strip()
        if not text:
            raise ValueError("regulator group name must be a non-empty string")
        return text

    @field_validator("members")
    @classmethod
    def _group_members_ok(cls, v: List[str]):
        if not v:
            raise ValueError("regulator group members must be non-empty")
        cleaned = []
        for raw in v:
            text = str(raw).strip()
            if not text:
                raise ValueError("regulator group members must be non-empty strings")
            cleaned.append(text)
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("regulator group members must be unique")
        return cleaned

    @field_validator("min_required")
    @classmethod
    def _group_min_required_ok(cls, v: int):
        if int(v) <= 0:
            raise ValueError("regulator group min_required must be > 0")
        return int(v)

    @model_validator(mode="after")
    def _group_min_required_size(self):
        if self.members and int(self.min_required) > len(self.members):
            raise ValueError(
                f"regulator group min_required cannot exceed group size ({self.min_required} > {len(self.members)})."
            )
        return self


class RegulatorConstraints(BaseModel):
    model_config = ConfigDict(extra="forbid")
    groups: List[RegulatorGroup]
    min_count_by_regulator: Dict[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _constraints_ok(self):
        if not self.groups and self.min_count_by_regulator:
            raise ValueError("regulator_constraints.groups must be non-empty when min_count_by_regulator is set")
        if not self.groups:
            self.min_count_by_regulator = {}
            return self
        names = []
        member_set: set[str] = set()
        for group in self.groups:
            names.append(group.name)
            for member in group.members:
                if member in member_set:
                    raise ValueError(f"regulator group members must be disjoint; duplicate '{member}' found")
                member_set.add(member)
        if len(set(names)) != len(names):
            raise ValueError("regulator group names must be unique")
        cleaned_counts: Dict[str, int] = {}
        for key, val in (self.min_count_by_regulator or {}).items():
            name = str(key).strip()
            if not name:
                raise ValueError("min_count_by_regulator keys must be non-empty strings")
            if name in cleaned_counts:
                raise ValueError("min_count_by_regulator keys must be unique after trimming")
            count = int(val)
            if count <= 0:
                raise ValueError(f"min_count_by_regulator[{name!r}] must be > 0")
            if name not in member_set:
                raise ValueError(f"min_count_by_regulator includes unknown regulator '{name}'")
            cleaned_counts[name] = count
        self.min_count_by_regulator = cleaned_counts
        return self


class PlanSamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    include_inputs: List[str]

    @field_validator("include_inputs")
    @classmethod
    def _include_inputs_ok(cls, v: List[str]):
        if not v:
            raise ValueError("plan.sampling.include_inputs must be a non-empty list")
        cleaned = []
        for raw in v:
            name = str(raw).strip()
            if not name:
                raise ValueError("plan.sampling.include_inputs must contain non-empty strings")
            cleaned.append(name)
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("plan.sampling.include_inputs must be unique")
        return cleaned


class PlanItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    quota: int
    sampling: PlanSamplingConfig
    fixed_elements: FixedElements = Field(default_factory=FixedElements)
    regulator_constraints: RegulatorConstraints

    @field_validator("quota")
    @classmethod
    def _quota_ok(cls, v: int):
        if int(v) <= 0:
            raise ValueError("Plan item quota must be > 0")
        return int(v)


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pool_strategy: Literal["full", "subsample", "iterative_subsample"] = "subsample"
    library_source: Literal["build", "artifact"] = "build"
    library_artifact_path: Optional[str] = None
    library_size: int = 16
    library_sampling_strategy: Literal[
        "tf_balanced",
        "uniform_over_pairs",
        "coverage_weighted",
    ] = "tf_balanced"
    coverage_boost_alpha: float = 0.15
    coverage_boost_power: float = 1.0
    avoid_failed_motifs: bool = False
    failure_penalty_alpha: float = 0.5
    failure_penalty_power: float = 1.0
    cover_all_regulators: bool = False
    unique_binding_sites: bool = True
    unique_binding_cores: bool = True
    max_sites_per_regulator: Optional[int] = None
    relax_on_exhaustion: bool = False
    iterative_max_libraries: int = 0
    iterative_min_new_solutions: int = 0

    @field_validator("library_size", "iterative_max_libraries", "iterative_min_new_solutions")
    @classmethod
    def _nonneg(cls, v: int):
        if v < 0:
            raise ValueError("Value must be >= 0")
        return v

    @field_validator("max_sites_per_regulator")
    @classmethod
    def _max_sites_ok(cls, v: Optional[int]):
        if v is not None and v <= 0:
            raise ValueError("max_sites_per_regulator must be > 0 when set")
        return v

    @field_validator("library_size")
    @classmethod
    def _subsample_size_ok(cls, v: int):
        if v <= 0:
            raise ValueError("library_size must be > 0")
        return v

    @field_validator("coverage_boost_alpha", "coverage_boost_power", "failure_penalty_alpha", "failure_penalty_power")
    @classmethod
    def _coverage_boost_ok(cls, v: float, info):
        if v < 0:
            raise ValueError(f"{info.field_name} must be >= 0")
        if info.field_name in {"coverage_boost_power", "failure_penalty_power"} and v == 0:
            raise ValueError(f"{info.field_name} must be > 0")
        return v

    @model_validator(mode="after")
    def _pool_strategy_rules(self):
        if self.pool_strategy == "iterative_subsample" and self.iterative_max_libraries <= 0:
            raise ValueError("iterative_max_libraries must be > 0 when pool_strategy=iterative_subsample")
        if self.pool_strategy != "iterative_subsample":
            if self.iterative_max_libraries > 0:
                raise ValueError("iterative_max_libraries is only valid when pool_strategy=iterative_subsample")
            if self.iterative_min_new_solutions > 0:
                raise ValueError("iterative_min_new_solutions is only valid when pool_strategy=iterative_subsample")
        return self

    @model_validator(mode="after")
    def _library_source_rules(self):
        if self.library_source == "artifact":
            if self.library_artifact_path is None or not str(self.library_artifact_path).strip():
                raise ValueError("sampling.library_artifact_path is required when sampling.library_source=artifact")
        else:
            if self.library_artifact_path is not None:
                raise ValueError("sampling.library_artifact_path is only valid when sampling.library_source=artifact")
        return self


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sequence_length: int
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    plan: List[PlanItem]

    @field_validator("sequence_length")
    @classmethod
    def _positive(cls, v: int):
        if v <= 0:
            raise ValueError("Value must be > 0")
        return v

    @model_validator(mode="after")
    def _plan_mode(self):
        if not self.plan:
            raise ValueError("generation.plan must contain at least one item")
        return self

    def resolve_plan(self) -> List["ResolvedPlanItem"]:
        return [
            ResolvedPlanItem(
                name=p.name,
                quota=int(p.quota),
                include_inputs=list(p.sampling.include_inputs),
                fixed_elements=p.fixed_elements,
                regulator_constraints=p.regulator_constraints,
            )
            for p in self.plan
        ]

    def total_quota(self) -> int:
        return sum(int(item.quota) for item in self.plan)


@dataclass(frozen=True)
class ResolvedPlanItem:
    name: str
    quota: int
    include_inputs: list[str]
    fixed_elements: FixedElements
    regulator_constraints: RegulatorConstraints
