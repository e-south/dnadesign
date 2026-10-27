"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/config/inputs.py

DenseGen input source schemas.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Annotated, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Literal

from .base import _deep_merge_dicts


class BindingSitesColumns(BaseModel):
    model_config = ConfigDict(extra="forbid")
    regulator: str = "tf"
    sequence: str = "tfbs"
    site_id: Optional[str] = None
    source: Optional[str] = None


class BindingSitesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    type: Literal["binding_sites"]
    path: str
    format: Optional[Literal["csv", "parquet", "xlsx"]] = None
    columns: BindingSitesColumns = Field(default_factory=BindingSitesColumns)


class SequenceLibraryInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    type: Literal["sequence_library"]
    path: str
    format: Optional[Literal["csv", "parquet"]] = None
    sequence_column: str = "sequence"


class BackgroundPoolMiningBudgetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["fixed_candidates"]
    candidates: int

    @field_validator("candidates")
    @classmethod
    def _candidates_ok(cls, v: int):
        if int(v) <= 0:
            raise ValueError("background_pool.sampling.mining.budget.candidates must be > 0")
        return int(v)


class BackgroundPoolMiningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    batch_size: int
    budget: BackgroundPoolMiningBudgetConfig
    log_every_batches: int = 1

    @field_validator("batch_size")
    @classmethod
    def _batch_size_ok(cls, v: int):
        if int(v) <= 0:
            raise ValueError("background_pool.sampling.mining.batch_size must be > 0")
        return int(v)

    @field_validator("log_every_batches")
    @classmethod
    def _log_every_ok(cls, v: int):
        if int(v) <= 0:
            raise ValueError("background_pool.sampling.mining.log_every_batches must be > 0")
        return int(v)


class BackgroundPoolLengthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    policy: Literal["exact", "range"] = "range"
    range: Optional[tuple[int, int]] = (16, 20)
    exact: Optional[int] = None

    @field_validator("range")
    @classmethod
    def _range_ok(cls, v: Optional[tuple[int, int]]):
        if v is None:
            return v
        if len(v) != 2:
            raise ValueError("background_pool.sampling.length.range must be a 2-tuple (min, max)")
        lo, hi = v
        if lo <= 0 or hi <= 0:
            raise ValueError("background_pool.sampling.length.range values must be > 0")
        if lo > hi:
            raise ValueError("background_pool.sampling.length.range must be min <= max")
        return v

    @field_validator("exact")
    @classmethod
    def _exact_ok(cls, v: Optional[int]):
        if v is None:
            return v
        if int(v) <= 0:
            raise ValueError("background_pool.sampling.length.exact must be > 0")
        return int(v)


class BackgroundPoolUniquenessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: Literal["sequence"] = "sequence"


class BackgroundPoolGCConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min: Optional[float] = None
    max: Optional[float] = None

    @model_validator(mode="after")
    def _gc_ok(self):
        if self.min is None and self.max is None:
            return self
        if self.min is not None:
            value = float(self.min)
            if value < 0 or value > 1:
                raise ValueError("background_pool.sampling.gc.min must be in [0, 1]")
            self.min = value
        if self.max is not None:
            value = float(self.max)
            if value < 0 or value > 1:
                raise ValueError("background_pool.sampling.gc.max must be in [0, 1]")
            self.max = value
        if self.min is not None and self.max is not None and float(self.min) > float(self.max):
            raise ValueError("background_pool.sampling.gc.min must be <= background_pool.sampling.gc.max")
        return self


class BackgroundPoolFimoExcludeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pwms_input: List[str]
    max_score_norm: Optional[float] = None
    allow_zero_hit_only: bool = False

    @field_validator("pwms_input", mode="before")
    @classmethod
    def _coerce_pwms_input(cls, v):
        if isinstance(v, str):
            return [v]
        return v

    @field_validator("pwms_input")
    @classmethod
    def _pwms_input_ok(cls, v: List[str]):
        if not v:
            raise ValueError("background_pool.sampling.filters.fimo_exclude.pwms_input must be non-empty")
        cleaned: list[str] = []
        for raw in v:
            name = str(raw).strip()
            if not name:
                raise ValueError("background_pool.sampling.filters.fimo_exclude.pwms_input must be non-empty")
            cleaned.append(name)
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("background_pool.sampling.filters.fimo_exclude.pwms_input must be unique")
        return cleaned

    @field_validator("max_score_norm")
    @classmethod
    def _max_score_norm_ok(cls, v: Optional[float]):
        if v is None:
            return v
        value = float(v)
        if value <= 0 or value > 1:
            raise ValueError("background_pool.sampling.filters.fimo_exclude.max_score_norm must be in (0, 1]")
        return value

    @model_validator(mode="after")
    def _mode_ok(self):
        if bool(self.allow_zero_hit_only) and self.max_score_norm is not None:
            raise ValueError(
                "background_pool.sampling.filters.fimo_exclude.max_score_norm "
                "must be unset when allow_zero_hit_only=true"
            )
        if not bool(self.allow_zero_hit_only) and self.max_score_norm is None:
            raise ValueError(
                "background_pool.sampling.filters.fimo_exclude.max_score_norm "
                "is required when allow_zero_hit_only=false"
            )
        return self


class BackgroundPoolForbidKmersRule(BaseModel):
    model_config = ConfigDict(extra="forbid")
    patterns_from_motif_sets: List[str]
    include_reverse_complements: bool = False
    strands: Literal["forward", "both"] = "both"

    @field_validator("patterns_from_motif_sets")
    @classmethod
    def _patterns_from_sets_ok(cls, v: List[str]):
        if not v:
            raise ValueError(
                "background_pool.sampling.filters.forbid_kmers[].patterns_from_motif_sets must be non-empty"
            )
        cleaned: list[str] = []
        for raw in v:
            text = str(raw).strip()
            if not text:
                raise ValueError(
                    "background_pool.sampling.filters.forbid_kmers[].patterns_from_motif_sets entries must be non-empty"
                )
            cleaned.append(text)
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("background_pool.sampling.filters.forbid_kmers[].patterns_from_motif_sets must be unique")
        return cleaned


class BackgroundPoolFiltersConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    forbid_kmers: List[Union[str, BackgroundPoolForbidKmersRule]] = Field(default_factory=list)
    include_reverse_complements: bool = False
    strands: Literal["forward", "both"] = "forward"
    fimo_exclude: Optional[BackgroundPoolFimoExcludeConfig] = None

    @field_validator("forbid_kmers")
    @classmethod
    def _forbid_kmers_ok(cls, v: List[Union[str, BackgroundPoolForbidKmersRule]]):
        literal_seen: set[str] = set()
        cleaned: list[Union[str, BackgroundPoolForbidKmersRule]] = []
        for raw in v:
            if isinstance(raw, BackgroundPoolForbidKmersRule):
                cleaned.append(raw)
                continue
            if not isinstance(raw, str):
                raise ValueError(
                    "background_pool.sampling.filters.forbid_kmers entries must be strings or motif-set rule objects"
                )
            seq = raw.strip().upper()
            if not seq:
                raise ValueError("background_pool.sampling.filters.forbid_kmers must be non-empty")
            if any(ch not in {"A", "C", "G", "T"} for ch in seq):
                raise ValueError("background_pool.sampling.filters.forbid_kmers must contain only A/C/G/T")
            if seq in literal_seen:
                raise ValueError("background_pool.sampling.filters.forbid_kmers literal kmers must be unique")
            literal_seen.add(seq)
            cleaned.append(seq)
        return cleaned


class BackgroundPoolSamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    n_sites: int
    mining: BackgroundPoolMiningConfig
    length: BackgroundPoolLengthConfig = Field(default_factory=BackgroundPoolLengthConfig)
    uniqueness: BackgroundPoolUniquenessConfig = Field(default_factory=BackgroundPoolUniquenessConfig)
    gc: Optional[BackgroundPoolGCConfig] = None
    filters: BackgroundPoolFiltersConfig = Field(default_factory=BackgroundPoolFiltersConfig)

    @field_validator("n_sites")
    @classmethod
    def _n_sites_ok(cls, v: int):
        if int(v) <= 0:
            raise ValueError("background_pool.sampling.n_sites must be > 0")
        return int(v)

    @model_validator(mode="after")
    def _sampling_rules(self):
        length_fields = set(getattr(self.length, "model_fields_set", set()))
        if self.length.policy == "exact":
            if "range" in length_fields and self.length.range is not None:
                raise ValueError("background_pool.sampling.length.range is not allowed when policy=exact")
            self.length.range = None
            if self.length.exact is None:
                raise ValueError("background_pool.sampling.length.exact is required when policy=exact")
        if self.length.policy == "range":
            if self.length.range is None:
                raise ValueError("background_pool.sampling.length.range is required when policy=range")
            if self.length.exact is not None:
                raise ValueError("background_pool.sampling.length.exact is not allowed when policy=range")
        return self


class BackgroundPoolInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    type: Literal["background_pool"]
    sampling: BackgroundPoolSamplingConfig


class PWMMiningBudgetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["tier_target", "fixed_candidates"]
    target_tier_fraction: Optional[float] = None
    candidates: Optional[int] = None
    max_candidates: Optional[int] = None
    max_seconds: Optional[float] = None
    min_candidates: Optional[int] = None
    growth_factor: float = 1.25

    @field_validator("growth_factor")
    @classmethod
    def _growth_factor_ok(cls, v: float):
        if float(v) <= 1.0:
            raise ValueError("pwm.sampling.mining.budget.growth_factor must be > 1.0")
        return float(v)

    @model_validator(mode="after")
    def _budget_rules(self):
        if self.mode == "fixed_candidates":
            if self.candidates is None:
                raise ValueError("pwm.sampling.mining.budget.candidates is required when mode=fixed_candidates")
            if int(self.candidates) <= 0:
                raise ValueError("pwm.sampling.mining.budget.candidates must be > 0")
        else:
            if self.target_tier_fraction is None:
                raise ValueError("pwm.sampling.mining.budget.target_tier_fraction is required for tier_target")
            if float(self.target_tier_fraction) <= 0 or float(self.target_tier_fraction) > 1:
                raise ValueError("pwm.sampling.mining.budget.target_tier_fraction must be in (0, 1]")
            if self.max_candidates is None and self.max_seconds is None:
                raise ValueError("pwm.sampling.mining.budget requires max_candidates or max_seconds for tier_target")
        if self.max_candidates is not None and int(self.max_candidates) <= 0:
            raise ValueError("pwm.sampling.mining.budget.max_candidates must be > 0 when set")
        if self.min_candidates is not None and int(self.min_candidates) <= 0:
            raise ValueError("pwm.sampling.mining.budget.min_candidates must be > 0 when set")
        if self.max_seconds is not None and float(self.max_seconds) <= 0:
            raise ValueError("pwm.sampling.mining.budget.max_seconds must be > 0 when set")
        if (
            self.min_candidates is not None
            and self.max_candidates is not None
            and int(self.min_candidates) > int(self.max_candidates)
        ):
            raise ValueError("pwm.sampling.mining.budget.min_candidates must be <= max_candidates")
        return self


class PWMMiningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    batch_size: int
    budget: PWMMiningBudgetConfig
    log_every_batches: int = 1

    @field_validator("batch_size")
    @classmethod
    def _batch_size_ok(cls, v: int):
        if v <= 0:
            raise ValueError("pwm.sampling.mining.batch_size must be > 0")
        return v

    @field_validator("log_every_batches")
    @classmethod
    def _log_every_batches_ok(cls, v: int):
        if v <= 0:
            raise ValueError("pwm.sampling.mining.log_every_batches must be > 0")
        return v


class PWMLengthConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    policy: Literal["exact", "range"] = "range"
    range: Optional[tuple[int, int]] = (16, 20)

    @field_validator("range")
    @classmethod
    def _range_ok(cls, v: Optional[tuple[int, int]]):
        if v is None:
            return v
        if len(v) != 2:
            raise ValueError("pwm.sampling.length.range must be a 2-tuple (min, max)")
        lo, hi = v
        if lo <= 0 or hi <= 0:
            raise ValueError("pwm.sampling.length.range values must be > 0")
        if lo > hi:
            raise ValueError("pwm.sampling.length.range must be min <= max")
        return v


class PWMTrimmingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    window_length: Optional[int] = None
    window_strategy: Literal["max_info"] = "max_info"

    @field_validator("window_length")
    @classmethod
    def _trim_length_ok(cls, v: Optional[int]):
        if v is None:
            return v
        if not isinstance(v, int) or v <= 0:
            raise ValueError("pwm.sampling.trimming.window_length must be a positive integer")
        return v


class PWMUniquenessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    key: Literal["sequence", "core"] = "core"
    cross_regulator_core_collisions: Literal["allow", "warn", "error"] = "warn"


class PWMSelectionPoolConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    min_score_norm: Optional[float] = None
    max_candidates: Optional[int] = None
    relevance_norm: Literal["percentile", "minmax_raw_score"] = "minmax_raw_score"

    @field_validator("min_score_norm")
    @classmethod
    def _min_score_norm_ok(cls, v: Optional[float]):
        if v is None:
            return v
        value = float(v)
        if value <= 0.0 or value > 1.0:
            raise ValueError("pwm.sampling.selection.pool.min_score_norm must be in (0, 1]")
        return value

    @field_validator("max_candidates")
    @classmethod
    def _max_candidates_ok(cls, v: Optional[int]):
        if v is None:
            return v
        if int(v) <= 0:
            raise ValueError("pwm.sampling.selection.pool.max_candidates must be > 0 when set")
        return int(v)


class PWMSelectionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    policy: Literal["top_score", "mmr"] = "top_score"
    rank_by: Literal["score", "score_norm"] = "score"
    alpha: float = 0.9
    pool: Optional[PWMSelectionPoolConfig] = None

    @field_validator("alpha")
    @classmethod
    def _alpha_ok(cls, v: float):
        if float(v) <= 0 or float(v) > 1:
            raise ValueError("pwm.sampling.selection.alpha must be in (0, 1]")
        return float(v)

    @model_validator(mode="after")
    def _defaults_for_policy(self):
        if self.policy == "mmr":
            if self.pool is None:
                raise ValueError("pwm.sampling.selection.pool is required when policy=mmr.")
        elif self.pool is not None:
            raise ValueError("pwm.sampling.selection.pool is only valid when policy=mmr.")
        return self


class PWMSamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy: Literal["consensus", "stochastic", "background"] = "stochastic"
    n_sites: int
    mining: PWMMiningConfig
    bgfile: Optional[str] = None
    keep_all_candidates_debug: bool = False
    include_matched_sequence: bool = True
    tier_fractions: Optional[List[float]] = None
    length: PWMLengthConfig = Field(default_factory=PWMLengthConfig)
    trimming: PWMTrimmingConfig = Field(default_factory=PWMTrimmingConfig)
    uniqueness: PWMUniquenessConfig = Field(default_factory=PWMUniquenessConfig)
    selection: PWMSelectionConfig = Field(default_factory=PWMSelectionConfig)

    @field_validator("n_sites")
    @classmethod
    def _n_sites_ok(cls, v: int):
        if v <= 0:
            raise ValueError("pwm.sampling.n_sites must be > 0")
        return v

    @field_validator("bgfile")
    @classmethod
    def _bgfile_ok(cls, v: Optional[str]):
        if v is None:
            return v
        if not str(v).strip():
            raise ValueError("pwm.sampling.bgfile must be a non-empty string when set")
        return str(v).strip()

    @field_validator("tier_fractions")
    @classmethod
    def _tier_fractions_ok(cls, v: Optional[List[float]]):
        if v is None:
            return v
        from ..core.score_tiers import normalize_tier_fractions

        normalize_tier_fractions(v)
        return [float(val) for val in v]

    @model_validator(mode="after")
    def _sampling_rules(self):
        length_fields = set(getattr(self.length, "model_fields_set", set()))
        if self.strategy == "consensus" and int(self.n_sites) != 1:
            raise ValueError("pwm.sampling.strategy=consensus requires n_sites=1")
        if self.length.policy == "exact":
            if "range" in length_fields and self.length.range is not None:
                raise ValueError("pwm.sampling.length.range is not allowed when policy=exact")
            self.length.range = None
        if self.length.policy == "range" and self.length.range is None:
            raise ValueError("pwm.sampling.length.range is required when policy=range")
        if not self.include_matched_sequence:
            raise ValueError("pwm.sampling.include_matched_sequence must be true for PWM sampling.")
        if self.selection.policy == "mmr" and self.uniqueness.key not in {"core", "sequence"}:
            raise ValueError("pwm.sampling.uniqueness.key must be 'core' or 'sequence'")
        return self


class PWMMemeInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    type: Literal["pwm_meme"]
    path: str
    motif_ids: Optional[List[str]] = None
    sampling: PWMSamplingConfig

    @field_validator("motif_ids")
    @classmethod
    def _motif_ids_ok(cls, v: Optional[List[str]]):
        if v is None:
            return v
        cleaned = []
        for m in v:
            if not isinstance(m, str) or not m.strip():
                raise ValueError("pwm.motif_ids must contain non-empty strings")
            cleaned.append(m.strip())
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("pwm.motif_ids must be unique")
        return cleaned


class PWMMemeSetInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    type: Literal["pwm_meme_set"]
    paths: List[str]
    motif_ids: Optional[List[str]] = None
    sampling: PWMSamplingConfig

    @field_validator("paths")
    @classmethod
    def _paths_ok(cls, v: List[str]):
        if not v:
            raise ValueError("pwm_meme_set.paths must be a non-empty list")
        cleaned = []
        for path in v:
            if not isinstance(path, str) or not path.strip():
                raise ValueError("pwm_meme_set.paths must contain non-empty strings")
            cleaned.append(path.strip())
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("pwm_meme_set.paths must be unique")
        return cleaned

    @field_validator("motif_ids")
    @classmethod
    def _motif_ids_ok(cls, v: Optional[List[str]]):
        if v is None:
            return v
        cleaned = []
        for m in v:
            if not isinstance(m, str) or not m.strip():
                raise ValueError("pwm_meme_set.motif_ids must contain non-empty strings")
            cleaned.append(m.strip())
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("pwm_meme_set.motif_ids must be unique")
        return cleaned


class PWMJasparInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    type: Literal["pwm_jaspar"]
    path: str
    motif_ids: Optional[List[str]] = None
    sampling: PWMSamplingConfig

    @field_validator("motif_ids")
    @classmethod
    def _motif_ids_ok(cls, v: Optional[List[str]]):
        if v is None:
            return v
        cleaned = []
        for m in v:
            if not isinstance(m, str) or not m.strip():
                raise ValueError("pwm.motif_ids must contain non-empty strings")
            cleaned.append(m.strip())
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("pwm.motif_ids must be unique")
        return cleaned


class PWMMatrixColumns(BaseModel):
    model_config = ConfigDict(extra="forbid")
    A: str = "A"
    C: str = "C"
    G: str = "G"
    T: str = "T"


class PWMMatrixCSVInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    type: Literal["pwm_matrix_csv"]
    path: str
    motif_id: str
    columns: PWMMatrixColumns = Field(default_factory=PWMMatrixColumns)
    sampling: PWMSamplingConfig

    @field_validator("motif_id")
    @classmethod
    def _motif_id_ok(cls, v: str):
        if not v or not str(v).strip():
            raise ValueError("pwm_matrix_csv.motif_id must be a non-empty string")
        return str(v).strip()


class USRSequencesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    type: Literal["usr_sequences"]
    dataset: str
    root: str
    limit: Optional[int] = None


class PWMArtifactInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    type: Literal["pwm_artifact"]
    path: str
    sampling: PWMSamplingConfig


class PWMArtifactSetInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    type: Literal["pwm_artifact_set"]
    paths: List[str]
    sampling: PWMSamplingConfig
    overrides_by_motif_id: Dict[str, PWMSamplingConfig] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _merge_partial_overrides(cls, data: object):
        if not isinstance(data, dict):
            return data
        overrides = data.get("overrides_by_motif_id")
        if not isinstance(overrides, dict) or not overrides:
            return data
        sampling = data.get("sampling")
        if sampling is None:
            return data
        if isinstance(sampling, BaseModel):
            base_sampling = sampling.model_dump()
        elif isinstance(sampling, dict):
            base_sampling = sampling
        else:
            return data
        merged: dict[str, object] = {}
        for key, value in overrides.items():
            if isinstance(value, dict):
                merged[key] = _deep_merge_dicts(base_sampling, value)
            else:
                merged[key] = value
        data["overrides_by_motif_id"] = merged
        return data

    @field_validator("paths")
    @classmethod
    def _paths_ok(cls, v: List[str]):
        if not v:
            raise ValueError("pwm_artifact_set.paths must be a non-empty list")
        cleaned = []
        for path in v:
            if not isinstance(path, str) or not path.strip():
                raise ValueError("pwm_artifact_set.paths must contain non-empty strings")
            cleaned.append(path.strip())
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("pwm_artifact_set.paths must be unique")
        return cleaned

    @field_validator("overrides_by_motif_id")
    @classmethod
    def _overrides_ok(cls, v: Dict[str, PWMSamplingConfig]):
        cleaned: Dict[str, PWMSamplingConfig] = {}
        for key, val in (v or {}).items():
            name = str(key).strip()
            if not name:
                raise ValueError("pwm_artifact_set.overrides_by_motif_id keys must be non-empty strings")
            if name in cleaned:
                raise ValueError("pwm_artifact_set.overrides_by_motif_id keys must be unique")
            cleaned[name] = val
        return cleaned


InputConfig = Annotated[
    Union[
        BindingSitesInput,
        SequenceLibraryInput,
        BackgroundPoolInput,
        PWMMemeInput,
        PWMMemeSetInput,
        PWMJasparInput,
        PWMMatrixCSVInput,
        PWMArtifactInput,
        PWMArtifactSetInput,
        USRSequencesInput,
    ],
    Field(discriminator="type"),
]
