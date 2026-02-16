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
    upstream_variant_id: Optional[str] = None
    downstream_variant_id: Optional[str] = None
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

    @field_validator("upstream_variant_id", "downstream_variant_id")
    @classmethod
    def _variant_id_ok(cls, v: Optional[str], info):
        if v is None:
            return v
        text = str(v).strip()
        if not text:
            raise ValueError(f"{info.field_name} must be a non-empty string when set")
        return text

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


class PlanTemplatePair(BaseModel):
    model_config = ConfigDict(extra="forbid")
    up: str
    down: str

    @field_validator("up", "down")
    @classmethod
    def _name_ok(cls, v: str, info):
        text = str(v).strip()
        if not text:
            raise ValueError(f"{info.field_name} must be a non-empty string")
        return text


class PromoterMatrixPairing(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["zip", "cross_product", "explicit_pairs"] = "zip"
    pairs: List[PlanTemplatePair] = Field(default_factory=list)

    @model_validator(mode="after")
    def _pairs_mode_rules(self):
        if self.mode == "explicit_pairs":
            if not self.pairs:
                raise ValueError("promoter_matrix.pairing.pairs must be set when mode=explicit_pairs")
        elif self.pairs:
            raise ValueError("promoter_matrix.pairing.pairs is only valid when mode=explicit_pairs")
        return self


class PromoterMatrix(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    upstream_from_set: str
    downstream_from_set: str
    pairing: PromoterMatrixPairing = Field(default_factory=PromoterMatrixPairing)
    spacer_length: tuple[int, int]
    upstream_pos: tuple[int, int]
    downstream_pos: Optional[tuple[int, int]] = None

    @field_validator("name", "upstream_from_set", "downstream_from_set")
    @classmethod
    def _nonempty(cls, v: str, info):
        text = str(v).strip()
        if not text:
            raise ValueError(f"{info.field_name} must be a non-empty string")
        return text

    @field_validator("spacer_length", "upstream_pos", "downstream_pos")
    @classmethod
    def _range_ok(cls, v: Optional[tuple[int, int]], info):
        if v is None:
            return v
        if len(v) != 2:
            raise ValueError(f"{info.field_name} must be a 2-tuple (min, max)")
        lo, hi = int(v[0]), int(v[1])
        if lo < 0 or hi < 0:
            raise ValueError(f"{info.field_name} values must be >= 0")
        if lo > hi:
            raise ValueError(f"{info.field_name} must be min <= max")
        return lo, hi


class TemplateFixedElements(BaseModel):
    model_config = ConfigDict(extra="forbid")
    promoter_matrix: Optional[PromoterMatrix] = None
    promoter_constraints: List[PromoterConstraint] = Field(default_factory=list)
    side_biases: Optional[SideBiases] = None


class PlanTemplate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    base_name: str
    quota_per_variant: Optional[int] = None
    total_quota: Optional[int] = None
    distribution_policy: Optional[Literal["uniform"]] = None
    sampling: PlanSamplingConfig
    fixed_elements: TemplateFixedElements = Field(default_factory=TemplateFixedElements)
    regulator_constraints: RegulatorConstraints

    @field_validator("base_name")
    @classmethod
    def _base_name_ok(cls, v: str):
        text = str(v).strip()
        if not text:
            raise ValueError("plan_templates[].base_name must be a non-empty string")
        return text

    @field_validator("quota_per_variant", "total_quota")
    @classmethod
    def _quota_positive(cls, v: Optional[int], info):
        if v is None:
            return v
        value = int(v)
        if value <= 0:
            raise ValueError(f"plan_templates[].{info.field_name} must be > 0")
        return value

    @model_validator(mode="after")
    def _quota_mode_rules(self):
        has_per_variant = self.quota_per_variant is not None
        has_total = self.total_quota is not None
        if has_per_variant == has_total:
            raise ValueError("plan_templates[] requires exactly one of quota_per_variant or total_quota")
        if has_total and self.distribution_policy != "uniform":
            raise ValueError("plan_templates[].distribution_policy must be 'uniform' when total_quota is set")
        if has_per_variant and self.distribution_policy is not None:
            raise ValueError("plan_templates[].distribution_policy is only valid when total_quota is set")
        return self


class SequenceConstraintForbidKmers(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    patterns_from_motif_sets: List[str]
    include_reverse_complements: bool = False
    scope: Literal["outside_allowed_placements"] = "outside_allowed_placements"
    strands: Literal["forward", "both"] = "both"

    @field_validator("name")
    @classmethod
    def _name_ok(cls, v: str):
        text = str(v).strip()
        if not text:
            raise ValueError("sequence_constraints.forbid_kmers[].name must be non-empty")
        return text

    @field_validator("patterns_from_motif_sets")
    @classmethod
    def _patterns_from_sets_ok(cls, v: List[str]):
        if not v:
            raise ValueError("sequence_constraints.forbid_kmers[].patterns_from_motif_sets must be non-empty")
        cleaned: list[str] = []
        for raw in v:
            text = str(raw).strip()
            if not text:
                raise ValueError(
                    "sequence_constraints.forbid_kmers[].patterns_from_motif_sets entries must be non-empty"
                )
            cleaned.append(text)
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("sequence_constraints.forbid_kmers[].patterns_from_motif_sets must be unique")
        return cleaned


class SequenceConstraintAllowSelector(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fixed_element: Literal["promoter"]
    component: List[Literal["upstream", "downstream"]] = Field(default_factory=lambda: ["upstream", "downstream"])

    @field_validator("component")
    @classmethod
    def _component_ok(cls, v: List[str]):
        if not v:
            raise ValueError("sequence_constraints.allowlist[].selector.component must be non-empty")
        cleaned = [str(item).strip() for item in v]
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("sequence_constraints.allowlist[].selector.component must be unique")
        return cleaned


class SequenceConstraintAllowlistItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Literal["fixed_element_instance"]
    selector: SequenceConstraintAllowSelector
    match_exact_coordinates: bool = True

    @model_validator(mode="after")
    def _strict_coordinate_match(self):
        if not bool(self.match_exact_coordinates):
            raise ValueError("sequence_constraints.allowlist[].match_exact_coordinates must be true")
        return self


class SequenceConstraintsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    forbid_kmers: List[SequenceConstraintForbidKmers] = Field(default_factory=list)
    allowlist: List[SequenceConstraintAllowlistItem] = Field(default_factory=list)

    @model_validator(mode="after")
    def _rules_ok(self):
        if not self.forbid_kmers:
            return self
        if not self.allowlist:
            raise ValueError("sequence_constraints.allowlist must be set when sequence_constraints.forbid_kmers is set")
        return self


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
    plan: List[PlanItem] = Field(default_factory=list)
    plan_templates: List[PlanTemplate] = Field(default_factory=list)
    plan_template_max_expanded_plans: int = 256
    sequence_constraints: Optional[SequenceConstraintsConfig] = None

    @field_validator("sequence_length")
    @classmethod
    def _positive(cls, v: int):
        if v <= 0:
            raise ValueError("Value must be > 0")
        return v

    @field_validator("plan_template_max_expanded_plans")
    @classmethod
    def _max_expanded_ok(cls, v: int):
        if int(v) <= 0:
            raise ValueError("generation.plan_template_max_expanded_plans must be > 0")
        return int(v)

    @model_validator(mode="after")
    def _plan_mode(self):
        has_plan = bool(self.plan)
        has_templates = bool(self.plan_templates)
        if has_plan == has_templates:
            raise ValueError("generation requires exactly one of plan or plan_templates")
        return self

    def resolve_plan(self) -> List["ResolvedPlanItem"]:
        if not self.plan:
            raise ValueError("generation.plan is empty; plan_templates must be expanded before runtime")
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


def _spacer_min(spacer: tuple[int, int] | None) -> int:
    if spacer is None:
        return 0
    return int(min(spacer))


def _normalize_motif_set_values(values: Dict[str, str], *, set_name: str) -> Dict[str, str]:
    cleaned: Dict[str, str] = {}
    for raw_key, raw_seq in values.items():
        key = str(raw_key).strip()
        if not key:
            raise ValueError(f"motif_sets.{set_name} contains an empty variant id")
        seq = str(raw_seq).strip().upper()
        if not seq:
            raise ValueError(f"motif_sets.{set_name}.{key} must be a non-empty motif string")
        if any(ch not in {"A", "C", "G", "T"} for ch in seq):
            raise ValueError(f"motif_sets.{set_name}.{key} must contain only A/C/G/T characters")
        cleaned[key] = seq
    if not cleaned:
        raise ValueError(f"motif_sets.{set_name} must contain at least one motif")
    return cleaned


def normalize_motif_sets(motif_sets: Dict[str, Dict[str, str]] | None) -> Dict[str, Dict[str, str]]:
    if motif_sets is None:
        return {}
    cleaned: Dict[str, Dict[str, str]] = {}
    for raw_set_name, raw_values in motif_sets.items():
        set_name = str(raw_set_name).strip()
        if not set_name:
            raise ValueError("motif_sets keys must be non-empty strings")
        if not isinstance(raw_values, dict):
            raise ValueError(f"motif_sets.{set_name} must map variant ids to motif strings")
        cleaned[set_name] = _normalize_motif_set_values(raw_values, set_name=set_name)
    return cleaned


def _validate_promoter_geometry(*, plan_name: str, sequence_length: int, constraint: PromoterConstraint) -> None:
    if constraint.upstream is None or constraint.downstream is None:
        return
    spacer_min = _spacer_min(constraint.spacer_length)
    required = len(constraint.upstream) + spacer_min + len(constraint.downstream)
    pos = constraint.upstream_pos or (0, sequence_length - 1)
    lo, hi = int(pos[0]), int(pos[1])
    if lo > hi:
        raise ValueError(
            f"generation geometry invalid for plan '{plan_name}': upstream_pos has min > max ({lo} > {hi})."
        )
    if lo + required > int(sequence_length):
        raise ValueError(
            f"generation geometry invalid for plan '{plan_name}': "
            f"upstream_pos.min={lo}, required_bp={required}, sequence_length={sequence_length}."
        )


def _expand_matrix_pairs(
    *,
    matrix: PromoterMatrix,
    motif_sets: Dict[str, Dict[str, str]],
) -> List[tuple[str, str, str, str]]:
    if matrix.upstream_from_set not in motif_sets:
        raise ValueError(f"plan_templates.promoter_matrix references unknown motif set: {matrix.upstream_from_set}")
    if matrix.downstream_from_set not in motif_sets:
        raise ValueError(f"plan_templates.promoter_matrix references unknown motif set: {matrix.downstream_from_set}")
    up_set = motif_sets[matrix.upstream_from_set]
    down_set = motif_sets[matrix.downstream_from_set]
    pairing = matrix.pairing
    if pairing.mode == "zip":
        up_keys = sorted(up_set)
        down_keys = sorted(down_set)
        if set(up_keys) != set(down_keys):
            raise ValueError(
                "plan_templates.promoter_matrix pairing mode=zip requires matching keys between motif sets"
            )
        return [(key, up_set[key], key, down_set[key]) for key in sorted(set(up_keys))]
    if pairing.mode == "cross_product":
        return [
            (up_key, up_set[up_key], down_key, down_set[down_key])
            for up_key in sorted(up_set)
            for down_key in sorted(down_set)
        ]
    pairs: List[tuple[str, str, str, str]] = []
    for pair in pairing.pairs:
        if pair.up not in up_set:
            raise ValueError(f"plan_templates.promoter_matrix explicit pair references unknown upstream id: {pair.up}")
        if pair.down not in down_set:
            raise ValueError(
                f"plan_templates.promoter_matrix explicit pair references unknown downstream id: {pair.down}"
            )
        pairs.append((pair.up, up_set[pair.up], pair.down, down_set[pair.down]))
    return pairs


def expand_plan_templates(
    *,
    plan_templates: List[PlanTemplate],
    motif_sets: Dict[str, Dict[str, str]],
    sequence_length: int,
    max_expanded_plans: int,
) -> List[PlanItem]:
    expanded: List[PlanItem] = []
    for template in plan_templates:
        fixed = template.fixed_elements
        matrix = fixed.promoter_matrix
        if matrix is None:
            quota = (
                int(template.quota_per_variant) if template.quota_per_variant is not None else int(template.total_quota)
            )
            plan = PlanItem(
                name=str(template.base_name),
                quota=quota,
                sampling=template.sampling,
                fixed_elements=FixedElements(
                    promoter_constraints=list(fixed.promoter_constraints or []),
                    side_biases=fixed.side_biases,
                ),
                regulator_constraints=template.regulator_constraints,
            )
            for constraint in plan.fixed_elements.promoter_constraints:
                _validate_promoter_geometry(
                    plan_name=plan.name,
                    sequence_length=int(sequence_length),
                    constraint=constraint,
                )
            expanded.append(plan)
            continue

        matrix_pairs = _expand_matrix_pairs(matrix=matrix, motif_sets=motif_sets)
        if len(matrix_pairs) > int(max_expanded_plans):
            raise ValueError(
                "plan_templates expansion exceeds generation.plan_template_max_expanded_plans "
                f"({len(matrix_pairs)} > {int(max_expanded_plans)})."
            )
        if template.quota_per_variant is not None:
            quotas = [int(template.quota_per_variant)] * len(matrix_pairs)
        else:
            total_quota = int(template.total_quota)
            if total_quota % len(matrix_pairs) != 0:
                raise ValueError(
                    "plan_templates total_quota must divide evenly across expanded variants when "
                    "distribution_policy=uniform"
                )
            quotas = [int(total_quota // len(matrix_pairs))] * len(matrix_pairs)

        for (up_id, up_seq, down_id, down_seq), quota in zip(matrix_pairs, quotas):
            name = f"{template.base_name}__up={up_id}__down={down_id}"
            matrix_constraint = PromoterConstraint(
                name=matrix.name,
                upstream=up_seq,
                downstream=down_seq,
                upstream_variant_id=up_id,
                downstream_variant_id=down_id,
                spacer_length=matrix.spacer_length,
                upstream_pos=matrix.upstream_pos,
                downstream_pos=matrix.downstream_pos,
            )
            constraints = [matrix_constraint, *list(fixed.promoter_constraints or [])]
            fixed_elements = FixedElements(
                promoter_constraints=constraints,
                side_biases=fixed.side_biases,
            )
            plan = PlanItem(
                name=name,
                quota=int(quota),
                sampling=template.sampling,
                fixed_elements=fixed_elements,
                regulator_constraints=template.regulator_constraints,
            )
            for constraint in plan.fixed_elements.promoter_constraints:
                _validate_promoter_geometry(
                    plan_name=plan.name,
                    sequence_length=int(sequence_length),
                    constraint=constraint,
                )
            expanded.append(plan)
    return expanded
