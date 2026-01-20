"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/config/__init__.py

Strict, versioned config schema for DenseGen (breaking changes, no fallbacks).

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Dict, List, Optional, Union

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from typing_extensions import Literal

from ..core.pvalue_bins import CANONICAL_PVALUE_BINS


# ---- Strict YAML loader (duplicate keys fail) ----
class _StrictLoader(yaml.SafeLoader):
    pass


def _construct_mapping(loader, node, deep: bool = False):
    mapping: Dict[Any, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise KeyError(f"Duplicate key in YAML: {key!r}")
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


LATEST_SCHEMA_VERSION = "2.4"
SUPPORTED_SCHEMA_VERSIONS = {LATEST_SCHEMA_VERSION}

KNOWN_SOLVER_OPTION_KEYS = {
    "CBC": {
        "threads",
        "timelimit",
        "timelimitseconds",
        "maxseconds",
        "seconds",
        "ratiogap",
        "mipgap",
        "seed",
        "randomseed",
        "loglevel",
    },
    "GUROBI": {
        "threads",
        "timelimit",
        "mipgap",
        "seed",
        "logtoconsole",
        "logfile",
        "method",
        "presolve",
    },
}


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class LoadedConfig:
    path: Path
    root: "RootConfig"


def _expand_path(value: str | os.PathLike) -> Path:
    return Path(os.path.expanduser(os.path.expandvars(str(value))))


def resolve_relative_path(cfg_path: Path, value: str | os.PathLike) -> Path:
    p = _expand_path(value)
    if p.is_absolute():
        return p
    return (cfg_path.parent / p).resolve()


def _is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def resolve_run_root(cfg_path: Path, run_root: str | os.PathLike) -> Path:
    root = resolve_relative_path(cfg_path, run_root)
    if root.exists() and not root.is_dir():
        raise ConfigError(f"densegen.run.root must be a directory: {root}")
    if not root.exists():
        raise ConfigError(f"densegen.run.root does not exist: {root}")
    return root


def resolve_run_scoped_path(cfg_path: Path, run_root: Path, value: str | os.PathLike, *, label: str) -> Path:
    resolved = resolve_relative_path(cfg_path, value)
    if not _is_relative_to(resolved, run_root):
        raise ConfigError(f"{label} must be within densegen.run.root ({run_root}), got: {resolved}")
    return resolved


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    root: str

    @field_validator("id")
    @classmethod
    def _id_nonempty(cls, v: str):
        if not v or not str(v).strip():
            raise ValueError("run.id must be a non-empty string")
        return str(v).strip()

    @field_validator("root")
    @classmethod
    def _root_nonempty(cls, v: str):
        if not v or not str(v).strip():
            raise ValueError("run.root must be a non-empty string")
        return str(v).strip()


# ---- Input sources ----
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


class PWMMiningConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    batch_size: int = 100000
    max_batches: Optional[int] = None
    max_candidates: Optional[int] = None
    max_seconds: Optional[float] = 60.0
    retain_bin_ids: Optional[List[int]] = None
    log_every_batches: int = 1

    @field_validator("batch_size")
    @classmethod
    def _batch_size_ok(cls, v: int):
        if v <= 0:
            raise ValueError("pwm.sampling.mining.batch_size must be > 0")
        return v

    @field_validator("max_batches")
    @classmethod
    def _max_batches_ok(cls, v: Optional[int]):
        if v is not None and v <= 0:
            raise ValueError("pwm.sampling.mining.max_batches must be > 0 when set")
        return v

    @field_validator("max_candidates")
    @classmethod
    def _max_candidates_ok(cls, v: Optional[int]):
        if v is not None and v <= 0:
            raise ValueError("pwm.sampling.mining.max_candidates must be > 0 when set")
        return v

    @field_validator("max_seconds")
    @classmethod
    def _max_seconds_ok(cls, v: Optional[float]):
        if v is None:
            return v
        if not isinstance(v, (int, float)) or float(v) <= 0:
            raise ValueError("pwm.sampling.mining.max_seconds must be > 0 when set")
        return float(v)

    @field_validator("retain_bin_ids")
    @classmethod
    def _retain_bin_ids_ok(cls, v: Optional[List[int]]):
        if v is None:
            return v
        if not v:
            raise ValueError("pwm.sampling.mining.retain_bin_ids must be non-empty when set")
        ids = [int(x) for x in v]
        if any(idx < 0 for idx in ids):
            raise ValueError("pwm.sampling.mining.retain_bin_ids values must be >= 0")
        if len(set(ids)) != len(ids):
            raise ValueError("pwm.sampling.mining.retain_bin_ids must be unique")
        return ids

    @field_validator("log_every_batches")
    @classmethod
    def _log_every_batches_ok(cls, v: int):
        if v <= 0:
            raise ValueError("pwm.sampling.mining.log_every_batches must be > 0")
        return v


class PWMSamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy: Literal["consensus", "stochastic", "background"] = "stochastic"
    n_sites: int
    oversample_factor: int = 10
    max_candidates: Optional[int] = 100000
    max_seconds: Optional[float] = None
    score_threshold: Optional[float] = None
    score_percentile: Optional[float] = None
    scoring_backend: Literal["densegen", "fimo"] = "densegen"
    pvalue_threshold: Optional[float] = None
    pvalue_bins: Optional[List[float]] = None
    mining: Optional[PWMMiningConfig] = None
    bgfile: Optional[str] = None
    selection_policy: Literal["random_uniform", "top_n", "stratified"] = "random_uniform"
    keep_all_candidates_debug: bool = False
    include_matched_sequence: bool = False
    length_policy: Literal["exact", "range"] = "exact"
    length_range: Optional[tuple[int, int]] = None
    trim_window_length: Optional[int] = None
    trim_window_strategy: Literal["max_info"] = "max_info"

    @field_validator("n_sites")
    @classmethod
    def _n_sites_ok(cls, v: int):
        if v <= 0:
            raise ValueError("pwm.sampling.n_sites must be > 0")
        return v

    @field_validator("oversample_factor")
    @classmethod
    def _oversample_ok(cls, v: int):
        if v <= 0:
            raise ValueError("pwm.sampling.oversample_factor must be > 0")
        return v

    @field_validator("max_candidates")
    @classmethod
    def _max_candidates_ok(cls, v: Optional[int]):
        if v is not None and v <= 0:
            raise ValueError("pwm.sampling.max_candidates must be > 0 when set")
        return v

    @field_validator("max_seconds")
    @classmethod
    def _max_seconds_ok(cls, v: Optional[float]):
        if v is None:
            return v
        if not isinstance(v, (int, float)) or float(v) <= 0:
            raise ValueError("pwm.sampling.max_seconds must be > 0 when set")
        return float(v)

    @field_validator("length_range")
    @classmethod
    def _length_range_ok(cls, v: Optional[tuple[int, int]]):
        if v is None:
            return v
        if len(v) != 2:
            raise ValueError("pwm.sampling.length_range must be a 2-tuple (min, max)")
        lo, hi = v
        if lo <= 0 or hi <= 0:
            raise ValueError("pwm.sampling.length_range values must be > 0")
        if lo > hi:
            raise ValueError("pwm.sampling.length_range must be min <= max")
        return v

    @field_validator("trim_window_length")
    @classmethod
    def _trim_length_ok(cls, v: Optional[int]):
        if v is None:
            return v
        if not isinstance(v, int) or v <= 0:
            raise ValueError("pwm.sampling.trim_window_length must be a positive integer")
        return v

    @field_validator("bgfile")
    @classmethod
    def _bgfile_ok(cls, v: Optional[str]):
        if v is None:
            return v
        if not str(v).strip():
            raise ValueError("pwm.sampling.bgfile must be a non-empty string when set")
        return str(v).strip()

    @field_validator("pvalue_bins")
    @classmethod
    def _pvalue_bins_ok(cls, v: Optional[List[float]]):
        if v is None:
            return v
        if not v:
            raise ValueError("pwm.sampling.pvalue_bins must be non-empty when set")
        bins = [float(x) for x in v]
        prev = 0.0
        for val in bins:
            if not (0.0 < val <= 1.0):
                raise ValueError("pwm.sampling.pvalue_bins values must be in (0, 1]")
            if val <= prev:
                raise ValueError("pwm.sampling.pvalue_bins must be strictly increasing")
            prev = val
        if abs(bins[-1] - 1.0) > 1e-12:
            raise ValueError("pwm.sampling.pvalue_bins must end with 1.0")
        return bins

    @model_validator(mode="after")
    def _score_mode(self):
        has_thresh = self.score_threshold is not None
        has_pct = self.score_percentile is not None
        if self.scoring_backend == "densegen":
            if has_thresh == has_pct:
                raise ValueError("pwm.sampling must set exactly one of score_threshold or score_percentile")
            if self.pvalue_threshold is not None:
                raise ValueError("pwm.sampling.pvalue_threshold is only valid when scoring_backend='fimo'")
            if self.pvalue_bins is not None:
                raise ValueError("pwm.sampling.pvalue_bins is only valid when scoring_backend='fimo'")
            if self.mining is not None:
                raise ValueError("pwm.sampling.mining is only valid when scoring_backend='fimo'")
            if self.include_matched_sequence:
                raise ValueError("pwm.sampling.include_matched_sequence is only valid when scoring_backend='fimo'")
        else:
            if self.pvalue_threshold is None:
                raise ValueError("pwm.sampling.pvalue_threshold is required when scoring_backend='fimo'")
            if not (0.0 < float(self.pvalue_threshold) <= 1.0):
                raise ValueError("pwm.sampling.pvalue_threshold must be between 0 and 1")
            if "max_candidates" in self.model_fields_set and self.max_candidates is not None:
                raise ValueError(
                    "pwm.sampling.max_candidates is not used with scoring_backend='fimo'. "
                    "Use pwm.sampling.mining.max_candidates instead."
                )
            if "max_seconds" in self.model_fields_set and self.max_seconds is not None:
                raise ValueError(
                    "pwm.sampling.max_seconds is not used with scoring_backend='fimo'. "
                    "Use pwm.sampling.mining.max_seconds instead."
                )
            if "max_candidates" not in self.model_fields_set:
                self.max_candidates = None
            if "max_seconds" not in self.model_fields_set:
                self.max_seconds = None
            if self.mining is None:
                self.mining = PWMMiningConfig()
            if self.pvalue_bins is None:
                self.pvalue_bins = list(CANONICAL_PVALUE_BINS)
            if self.mining is not None and self.mining.max_candidates is not None:
                if int(self.mining.max_candidates) < int(self.n_sites):
                    raise ValueError("pwm.sampling.mining.max_candidates must be >= n_sites")
            if self.mining is not None and self.mining.retain_bin_ids is not None:
                bins = list(self.pvalue_bins) if self.pvalue_bins is not None else list(CANONICAL_PVALUE_BINS)
                max_idx = len(bins) - 1
                if any(idx > max_idx for idx in self.mining.retain_bin_ids):
                    raise ValueError("pwm.sampling.mining.retain_bin_ids contains an index outside the available bins")
        if self.strategy == "consensus" and int(self.n_sites) != 1:
            raise ValueError("pwm.sampling.strategy=consensus requires n_sites=1")
        if self.scoring_backend == "densegen" and self.score_percentile is not None:
            if not (0.0 < float(self.score_percentile) < 100.0):
                raise ValueError("pwm.sampling.score_percentile must be between 0 and 100")
        if self.length_policy == "exact" and self.length_range is not None:
            raise ValueError("pwm.sampling.length_range is not allowed when length_policy=exact")
        if self.length_policy == "range" and self.length_range is None:
            raise ValueError("pwm.sampling.length_range is required when length_policy=range")
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


# ---- Generation / constraints ----
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


class PlanItem(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    quota: Optional[int] = None
    fraction: Optional[float] = None
    fixed_elements: FixedElements = Field(default_factory=FixedElements)
    required_regulators: List[str] = Field(default_factory=list)
    min_required_regulators: Optional[int] = None
    min_count_by_regulator: Dict[str, int] = Field(default_factory=dict)

    @field_validator("required_regulators")
    @classmethod
    def _required_regs_ok(cls, v: List[str]):
        regs = []
        for r in v:
            if not isinstance(r, str) or not r.strip():
                raise ValueError("required_regulators must contain non-empty strings")
            regs.append(r.strip())
        if len(set(regs)) != len(regs):
            raise ValueError("required_regulators must be unique")
        return regs

    @field_validator("min_required_regulators")
    @classmethod
    def _min_required_regs_ok(cls, v: Optional[int]):
        if v is None:
            return v
        if int(v) <= 0:
            raise ValueError("min_required_regulators must be > 0 when set")
        return int(v)

    @field_validator("min_count_by_regulator")
    @classmethod
    def _min_counts_ok(cls, v: Dict[str, int]):
        cleaned: Dict[str, int] = {}
        for key, val in (v or {}).items():
            name = str(key).strip()
            if not name:
                raise ValueError("min_count_by_regulator keys must be non-empty strings")
            if name in cleaned:
                raise ValueError("min_count_by_regulator keys must be unique after trimming")
            count = int(val)
            if count <= 0:
                raise ValueError(f"min_count_by_regulator[{name!r}] must be > 0")
            cleaned[name] = count
        return cleaned

    @model_validator(mode="after")
    def _quota_or_fraction(self):
        if (self.quota is None) == (self.fraction is None):
            raise ValueError("Plan item must define exactly one of: quota or fraction")
        if self.quota is not None and self.quota <= 0:
            raise ValueError("Plan item quota must be > 0")
        if self.fraction is not None and self.fraction <= 0:
            raise ValueError("Plan item fraction must be > 0")
        return self

    @model_validator(mode="after")
    def _required_regulator_k_ok(self):
        if self.min_required_regulators is not None and self.required_regulators:
            if int(self.min_required_regulators) > len(self.required_regulators):
                raise ValueError(
                    "min_required_regulators cannot exceed required_regulators size "
                    f"({self.min_required_regulators} > {len(self.required_regulators)})."
                )
        return self


class SamplingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    pool_strategy: Literal["full", "subsample", "iterative_subsample"] = "subsample"
    library_size: int = 16
    subsample_over_length_budget_by: int = 30
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
    cover_all_regulators: bool = True
    unique_binding_sites: bool = True
    max_sites_per_regulator: Optional[int] = None
    relax_on_exhaustion: bool = False
    allow_incomplete_coverage: bool = False
    iterative_max_libraries: int = 50
    iterative_min_new_solutions: int = 1

    @field_validator("subsample_over_length_budget_by", "library_size", "iterative_max_libraries")
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

    @field_validator("iterative_min_new_solutions")
    @classmethod
    def _iter_min_new_ok(cls, v: int):
        if v < 0:
            raise ValueError("iterative_min_new_solutions must be >= 0")
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
        return self


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sequence_length: int
    quota: int
    sampling: SamplingConfig = Field(default_factory=SamplingConfig)
    plan: List[PlanItem]

    @field_validator("sequence_length", "quota")
    @classmethod
    def _positive(cls, v: int):
        if v <= 0:
            raise ValueError("Value must be > 0")
        return v

    @model_validator(mode="after")
    def _plan_mode(self):
        if not self.plan:
            raise ValueError("generation.plan must contain at least one item")
        has_quota = all(p.quota is not None for p in self.plan)
        has_fraction = all(p.fraction is not None for p in self.plan)
        if not (has_quota or has_fraction):
            raise ValueError("Plan items must all use quota or all use fraction (no mixing)")
        if has_fraction:
            total = sum(float(p.fraction) for p in self.plan)
            if abs(total - 1.0) > 1e-6:
                raise ValueError("Plan fractions must sum to 1.0")
        return self

    def resolve_plan(self) -> List["ResolvedPlanItem"]:
        if all(p.quota is not None for p in self.plan):
            return [
                ResolvedPlanItem(
                    name=p.name,
                    quota=int(p.quota),
                    fixed_elements=p.fixed_elements,
                    required_regulators=p.required_regulators,
                    min_required_regulators=p.min_required_regulators,
                    min_count_by_regulator=p.min_count_by_regulator,
                )
                for p in self.plan
            ]

        # Fractions mode
        remaining = self.quota
        resolved: List[ResolvedPlanItem] = []
        for i, p in enumerate(self.plan):
            if i == len(self.plan) - 1:
                q = remaining
            else:
                q = int(round(self.quota * float(p.fraction)))
                q = min(q, remaining)
            if q <= 0:
                raise ValueError("Resolved plan quota must be > 0")
            resolved.append(
                ResolvedPlanItem(
                    name=p.name,
                    quota=q,
                    fixed_elements=p.fixed_elements,
                    required_regulators=p.required_regulators,
                    min_required_regulators=p.min_required_regulators,
                    min_count_by_regulator=p.min_count_by_regulator,
                )
            )
            remaining -= q
        if remaining != 0:
            raise ValueError("Resolved plan quotas do not sum to generation.quota")
        return resolved


@dataclass(frozen=True)
class ResolvedPlanItem:
    name: str
    quota: int
    fixed_elements: FixedElements
    required_regulators: List[str]
    min_required_regulators: Optional[int]
    min_count_by_regulator: Dict[str, int]


# ---- Output ----
class OutputSchemaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bio_type: str = "dna"
    alphabet: str = "dna_4"

    @field_validator("bio_type", "alphabet")
    @classmethod
    def _nonempty(cls, v: str, info):
        if not v or not str(v).strip():
            raise ValueError(f"output.schema.{info.field_name} must be a non-empty string")
        return v


class OutputUSRConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset: str
    root: str
    chunk_size: int = 128
    allow_overwrite: bool = False


class OutputParquetConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    deduplicate: bool = True
    chunk_size: int = 2048

    @field_validator("path")
    @classmethod
    def _path_is_file(cls, v: str):
        if not v or not str(v).strip():
            raise ValueError("output.parquet.path must be a non-empty string")
        if not str(v).endswith(".parquet"):
            raise ValueError("output.parquet.path must point to a .parquet file")
        return v

    @field_validator("chunk_size")
    @classmethod
    def _chunk_size_ok(cls, v: int):
        if v <= 0:
            raise ValueError("chunk_size must be > 0")
        return v


class OutputConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    targets: List[Literal["usr", "parquet"]]
    output_schema: OutputSchemaConfig = Field(alias="schema")
    usr: Optional[OutputUSRConfig] = None
    parquet: Optional[OutputParquetConfig] = None

    @model_validator(mode="after")
    def _require_targets(self):
        if not self.targets:
            raise ValueError("output.targets must contain at least one sink.")
        if len(set(self.targets)) != len(self.targets):
            raise ValueError("output.targets must not contain duplicates.")
        if "usr" in self.targets and self.usr is None:
            raise ValueError("output.usr is required when output.targets includes 'usr'")
        if "parquet" in self.targets and self.parquet is None:
            raise ValueError("output.parquet is required when output.targets includes 'parquet'")
        return self


# ---- Solver ----
class SolverConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    backend: Optional[str] = None
    strategy: Literal["iterate", "diverse", "optimal", "approximate"]
    options: List[str] = Field(default_factory=list)
    strands: Literal["single", "double"] = "double"
    fallback_to_cbc: bool = False
    allow_unknown_options: bool = False

    @field_validator("backend")
    @classmethod
    def _backend_nonempty(cls, v: str | None):
        if v is None:
            return v
        if not str(v).strip():
            raise ValueError("solver.backend must be a non-empty string")
        return v

    @model_validator(mode="after")
    def _strategy_backend_consistency(self):
        if self.strategy != "approximate" and not self.backend:
            raise ValueError("solver.backend is required unless strategy=approximate")
        if self.strategy == "approximate" and self.options:
            raise ValueError("solver.options must be empty when strategy=approximate")
        if self.options:
            cleaned: list[str] = []
            for opt in self.options:
                if not isinstance(opt, str) or not opt.strip():
                    raise ValueError("solver.options entries must be non-empty strings")
                cleaned.append(opt.strip())
            self.options = cleaned
            if not self.allow_unknown_options:
                backend = (self.backend or "").strip().upper()
                allowed = KNOWN_SOLVER_OPTION_KEYS.get(backend)
                if allowed is None:
                    raise ValueError(
                        f"solver.options provided but backend '{backend}' has no known option list. "
                        "Set solver.allow_unknown_options: true to bypass validation."
                    )
                unknown: list[str] = []
                for opt in self.options:
                    key = opt.split("=", 1)[0].split()[0].strip().lower()
                    if key not in allowed:
                        unknown.append(opt)
                if unknown:
                    preview = ", ".join(unknown[:5])
                    raise ValueError(
                        f"Unknown solver.options for backend '{backend}': {preview}. "
                        "Set solver.allow_unknown_options: true to bypass validation."
                    )
        return self


# ---- Runtime ----
class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    round_robin: bool = False
    arrays_generated_before_resample: int = 1
    min_count_per_tf: int = 0
    max_duplicate_solutions: int = 3
    stall_seconds_before_resample: int = 30
    stall_warning_every_seconds: int = 15
    max_resample_attempts: int = 50
    max_total_resamples: int = 500
    max_seconds_per_plan: int = 0
    max_failed_solutions: int = 0
    leaderboard_every: int = 50
    checkpoint_every: int = 50
    random_seed: int = 1337

    @field_validator(
        "arrays_generated_before_resample",
        "min_count_per_tf",
        "max_duplicate_solutions",
        "stall_seconds_before_resample",
        "stall_warning_every_seconds",
        "max_resample_attempts",
        "max_total_resamples",
        "max_seconds_per_plan",
        "max_failed_solutions",
        "leaderboard_every",
        "checkpoint_every",
    )
    @classmethod
    def _non_negative(cls, v: int, info):
        if v < 0:
            raise ValueError(f"{info.field_name} must be >= 0")
        return v

    @field_validator("arrays_generated_before_resample")
    @classmethod
    def _arrays_positive(cls, v: int):
        if v <= 0:
            raise ValueError("arrays_generated_before_resample must be > 0")
        return v


# ---- Postprocess ----
class GapFillConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["off", "strict", "adaptive"] = "adaptive"
    end: Literal["5prime", "3prime"] = "5prime"
    gc_min: float = 0.40
    gc_max: float = 0.60
    max_tries: int = 2000

    @field_validator("gc_min", "gc_max")
    @classmethod
    def _gc_ok(cls, v: float):
        if not (0.0 <= v <= 1.0):
            raise ValueError("GC fraction must be between 0 and 1")
        return v

    @model_validator(mode="after")
    def _gc_bounds(self):
        if self.gc_min > self.gc_max:
            raise ValueError("gc_min must be <= gc_max")
        return self


class PostprocessConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gap_fill: GapFillConfig = Field(default_factory=GapFillConfig)


# ---- Logging ----
class LoggingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    log_dir: str
    level: str = "INFO"
    suppress_solver_stderr: bool = True
    print_visual: bool = True
    progress_style: Literal["stream", "summary", "screen"] = "stream"
    progress_every: int = 1
    progress_refresh_seconds: float = 1.0

    @field_validator("log_dir")
    @classmethod
    def _log_dir_nonempty(cls, v: str):
        if not v or not str(v).strip():
            raise ValueError("logging.log_dir must be a non-empty string")
        return v

    @field_validator("level")
    @classmethod
    def _level_ok(cls, v: str):
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        lv = (v or "").upper()
        if lv not in allowed:
            raise ValueError(f"logging.level must be one of {sorted(allowed)}")
        return lv

    @field_validator("progress_every")
    @classmethod
    def _progress_every_ok(cls, v: int):
        if v < 0:
            raise ValueError("logging.progress_every must be >= 0")
        return int(v)

    @field_validator("progress_refresh_seconds")
    @classmethod
    def _progress_refresh_ok(cls, v: float):
        if not isinstance(v, (int, float)) or float(v) <= 0:
            raise ValueError("logging.progress_refresh_seconds must be > 0")
        return float(v)


# ---- Plots ----
class PlotConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    out_dir: str = "outputs"
    format: Literal["png", "pdf", "svg"] = "png"
    source: Optional[Literal["usr", "parquet"]] = None
    default: List[str] = Field(default_factory=list)
    options: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    style: Dict[str, Any] = Field(default_factory=dict)
    sample_rows: Optional[int] = None

    @field_validator("sample_rows")
    @classmethod
    def _sample_rows_ok(cls, v: Optional[int]):
        if v is None:
            return v
        if int(v) <= 0:
            raise ValueError("plots.sample_rows must be > 0")
        return int(v)


# ---- Root ----
class DenseGenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: str
    run: RunConfig
    inputs: List[InputConfig]
    output: OutputConfig
    generation: GenerationConfig
    solver: SolverConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    postprocess: PostprocessConfig = Field(default_factory=PostprocessConfig)
    logging: LoggingConfig

    @field_validator("schema_version")
    @classmethod
    def _schema_version_supported(cls, v: str):
        if not v or not str(v).strip():
            raise ValueError("densegen.schema_version must be a non-empty string")
        if v not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"Unsupported densegen.schema_version: {v!r}. Supported versions: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )
        return v

    @model_validator(mode="after")
    def _inputs_nonempty(self):
        if not self.inputs:
            raise ValueError("At least one input is required")
        names = [i.name for i in self.inputs]
        if len(set(names)) != len(names):
            raise ValueError("Input names must be unique")
        return self


class RootConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    densegen: DenseGenConfig
    plots: Optional[PlotConfig] = None

    @model_validator(mode="after")
    def _plots_source_required(self):
        targets = self.densegen.output.targets
        if len(targets) > 1:
            if self.plots is None or self.plots.source is None:
                raise ValueError("plots.source must be set when output.targets has multiple sinks")
            if self.plots.source not in targets:
                raise ValueError("plots.source must be one of output.targets")
        return self


def _validate_run_scoped_paths(cfg_path: Path, root_cfg: RootConfig) -> None:
    run_cfg = root_cfg.densegen.run
    run_root = resolve_run_root(cfg_path, run_cfg.root)
    if not _is_relative_to(cfg_path, run_root):
        raise ConfigError(f"Config file must live inside densegen.run.root ({run_root}), got: {cfg_path}")

    out_cfg = root_cfg.densegen.output
    if out_cfg.parquet is not None:
        resolve_run_scoped_path(
            cfg_path,
            run_root,
            out_cfg.parquet.path,
            label="output.parquet.path",
        )
    if out_cfg.usr is not None:
        resolve_run_scoped_path(
            cfg_path,
            run_root,
            out_cfg.usr.root,
            label="output.usr.root",
        )

    log_dir = root_cfg.densegen.logging.log_dir
    resolve_run_scoped_path(cfg_path, run_root, log_dir, label="logging.log_dir")

    if root_cfg.plots is not None:
        resolve_run_scoped_path(
            cfg_path,
            run_root,
            root_cfg.plots.out_dir,
            label="plots.out_dir",
        )


def load_config(path: Path | str) -> LoadedConfig:
    cfg_path = Path(path).resolve()
    raw = yaml.load(cfg_path.read_text(), Loader=_StrictLoader)
    try:
        root = RootConfig.model_validate(raw)
    except ValidationError as e:
        raise ConfigError(f"Invalid DenseGen config: {e}")
    _validate_run_scoped_paths(cfg_path, root)
    return LoadedConfig(path=cfg_path, root=root)
