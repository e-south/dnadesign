"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/config/schema_v3.py

Defines the Cruncher v3 configuration schema.

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dnadesign.cruncher.config.schema_v2 import (
    CampaignConfig,
    CampaignMetadataConfig,
    HttpRetryConfig,
    LocalMotifSourceConfig,
    LocalSiteSourceConfig,
    RegulonDBConfig,
)


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


logger = logging.getLogger(__name__)


class ParserConfig(StrictBaseModel):
    extra_modules: List[str] = Field(
        default_factory=list,
        description="Additional modules to import for parser registration.",
    )

    @field_validator("extra_modules")
    @classmethod
    def _check_extra_modules(cls, v: List[str]) -> List[str]:
        cleaned = []
        for mod in v:
            name = str(mod).strip()
            if not name:
                raise ValueError("io.parsers.extra_modules entries must be non-empty strings")
            cleaned.append(name)
        return cleaned


class IOConfig(StrictBaseModel):
    parsers: ParserConfig = ParserConfig()


class WorkspaceConfig(StrictBaseModel):
    out_dir: Path
    regulator_sets: List[List[str]]
    regulator_categories: Dict[str, List[str]] = Field(default_factory=dict)

    @field_validator("out_dir")
    @classmethod
    def _check_out_dir(cls, v: Path) -> Path:
        if v.is_absolute():
            raise ValueError("workspace.out_dir must be a relative path")
        normalized = Path(v)
        if any(part == ".." for part in normalized.parts):
            raise ValueError("workspace.out_dir must not traverse outside the workspace")
        return normalized

    @field_validator("regulator_sets")
    @classmethod
    def _check_regulator_sets(cls, v: List[List[str]]) -> List[List[str]]:
        if not v:
            raise ValueError("workspace.regulator_sets must be a non-empty list")
        cleaned: list[list[str]] = []
        for idx, group in enumerate(v, start=1):
            if not group:
                raise ValueError(f"workspace.regulator_sets[{idx}] must be a non-empty list")
            seen: set[str] = set()
            tfs: list[str] = []
            for tf in group:
                name = str(tf).strip()
                if not name:
                    raise ValueError(f"workspace.regulator_sets[{idx}] must contain non-empty TF names")
                if name in seen:
                    continue
                seen.add(name)
                tfs.append(name)
            cleaned.append(tfs)
        return cleaned

    @field_validator("regulator_categories")
    @classmethod
    def _check_regulator_categories(cls, v: Dict[str, List[str]]) -> Dict[str, List[str]]:
        cleaned: dict[str, list[str]] = {}
        for raw_name, raw_tfs in v.items():
            name = str(raw_name).strip()
            if not name:
                raise ValueError("workspace.regulator_categories keys must be non-empty strings")
            if not raw_tfs:
                raise ValueError(f"workspace.regulator_categories['{name}'] must be a non-empty list")
            tfs: list[str] = []
            seen: set[str] = set()
            for tf in raw_tfs:
                tf_name = str(tf).strip()
                if not tf_name:
                    raise ValueError(f"workspace.regulator_categories['{name}'] must contain non-empty TF names")
                if tf_name in seen:
                    continue
                seen.add(tf_name)
                tfs.append(tf_name)
            cleaned[name] = tfs
        return cleaned


class CatalogConfig(StrictBaseModel):
    root: Path = Path(".cruncher")
    source_preference: List[str] = Field(default_factory=list)
    pwm_source: Literal["matrix", "sites"] = "matrix"
    allow_ambiguous: bool = False
    combine_sites: bool = False
    pseudocounts: float = Field(
        0.5,
        description="Pseudocounts for PWM construction from sites (Biopython).",
    )
    dataset_preference: List[str] = Field(
        default_factory=list,
        description="Preferred dataset IDs to resolve HT ambiguity (first match wins).",
    )
    dataset_map: Dict[str, str] = Field(
        default_factory=dict,
        description="Explicit TF->dataset_id map for HT data selection.",
    )
    site_kinds: Optional[List[str]] = None
    site_window_lengths: Dict[str, int] = Field(default_factory=dict)
    site_window_center: Literal["midpoint", "summit"] = "midpoint"
    pwm_window_lengths: Dict[str, int] = Field(default_factory=dict)
    pwm_window_strategy: Literal["max_info"] = "max_info"
    min_sites_for_pwm: int = 2
    allow_low_sites: bool = False

    @field_validator("root")
    @classmethod
    def _check_root(cls, v: Path) -> Path:
        normalized = Path(v)
        if not str(normalized).strip():
            raise ValueError("catalog.root must be a non-empty path")
        if not normalized.is_absolute() and any(part == ".." for part in normalized.parts):
            raise ValueError("catalog.root must not traverse outside the cruncher root")
        return normalized

    @field_validator("site_window_lengths")
    @classmethod
    def _check_window_lengths(cls, v: Dict[str, int]) -> Dict[str, int]:
        for key, value in v.items():
            if value <= 0:
                raise ValueError(f"catalog.site_window_lengths['{key}'] must be > 0")
        return v

    @field_validator("pwm_window_lengths")
    @classmethod
    def _check_pwm_window_lengths(cls, v: Dict[str, int]) -> Dict[str, int]:
        for key, value in v.items():
            if value <= 0:
                raise ValueError(f"catalog.pwm_window_lengths['{key}'] must be > 0")
        return v

    @field_validator("min_sites_for_pwm")
    @classmethod
    def _check_min_sites(cls, v: int) -> int:
        if v < 1:
            raise ValueError("catalog.min_sites_for_pwm must be >= 1")
        return v

    @field_validator("pseudocounts")
    @classmethod
    def _check_pseudocounts(cls, v: float) -> float:
        if v < 0:
            raise ValueError("catalog.pseudocounts must be >= 0")
        return float(v)

    @property
    def catalog_root(self) -> Path:
        return self.root


class DiscoverConfig(StrictBaseModel):
    enabled: bool = False
    tool: Literal["auto", "streme", "meme"] = "auto"
    source_id: str = "meme_suite"
    tool_path: Optional[Path] = None
    window_sites: bool = False
    minw: Optional[int] = None
    maxw: Optional[int] = None
    nmotifs: int = 1
    meme_mod: Optional[Literal["oops", "zoops", "anr"]] = None
    meme_prior: Optional[Literal["dirichlet", "dmix", "mega", "megap", "addone"]] = None
    min_sequences_for_streme: int = 50
    replace_existing: bool = True

    @field_validator("minw", "maxw")
    @classmethod
    def _check_optional_positive_ints(cls, v: Optional[int], info) -> Optional[int]:
        if v is None:
            return v
        if v < 1:
            raise ValueError(f"discover.{info.field_name} must be >= 1")
        return int(v)

    @field_validator("nmotifs", "min_sequences_for_streme")
    @classmethod
    def _check_positive_ints(cls, v: int, info) -> int:
        if v < 1:
            raise ValueError(f"discover.{info.field_name} must be >= 1")
        return v

    @model_validator(mode="after")
    def _check_widths(self) -> "DiscoverConfig":
        if self.minw is not None and self.maxw is not None and self.maxw < self.minw:
            raise ValueError("discover.maxw must be >= discover.minw")
        return self


class IngestConfig(StrictBaseModel):
    genome_source: Literal["ncbi", "fasta", "none"] = "ncbi"
    genome_fasta: Optional[Path] = None
    genome_cache: Path = Path(".cruncher/genomes")
    genome_assembly: Optional[str] = None
    contig_aliases: Dict[str, str] = Field(default_factory=dict)
    ncbi_email: Optional[str] = None
    ncbi_tool: str = "cruncher"
    ncbi_api_key: Optional[str] = None
    ncbi_timeout_seconds: int = 30
    http: HttpRetryConfig = HttpRetryConfig()
    regulondb: RegulonDBConfig = RegulonDBConfig()
    local_sources: List[LocalMotifSourceConfig] = Field(
        default_factory=list,
        description="Local filesystem motif sources.",
    )
    site_sources: List[LocalSiteSourceConfig] = Field(
        default_factory=list,
        description="Local FASTA binding-site sources.",
    )

    @field_validator("genome_cache")
    @classmethod
    def _check_genome_cache(cls, v: Path) -> Path:
        if v.is_absolute():
            raise ValueError("ingest.genome_cache must be a relative path")
        normalized = Path(v)
        if any(part == ".." for part in normalized.parts):
            raise ValueError("ingest.genome_cache must not traverse outside the workspace")
        return normalized

    @field_validator("ncbi_timeout_seconds")
    @classmethod
    def _check_ncbi_timeout(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("ingest.ncbi_timeout_seconds must be > 0")
        return v


class SampleBudgetConfig(StrictBaseModel):
    tune: int
    draws: int

    @field_validator("tune", "draws")
    @classmethod
    def _check_positive_ints(cls, v: int, info) -> int:
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"sample.budget.{info.field_name} must be >= 0")
        return v

    @model_validator(mode="after")
    def _check_budget(self) -> "SampleBudgetConfig":
        if self.draws < 1:
            raise ValueError("sample.budget.draws must be >= 1")
        return self


class SampleObjectiveSoftminConfig(StrictBaseModel):
    enabled: bool = True
    schedule: Literal["fixed", "linear"] = "linear"
    beta_start: float = 0.5
    beta_end: float = 10.0

    @field_validator("beta_start", "beta_end")
    @classmethod
    def _check_beta(cls, v: float, info) -> float:
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError(f"objective.softmin.{info.field_name} must be > 0")
        return float(v)

    @model_validator(mode="after")
    def _check_beta_range(self) -> "SampleObjectiveSoftminConfig":
        if self.schedule == "linear" and self.beta_end < self.beta_start:
            raise ValueError("objective.softmin.beta_end must be >= beta_start for linear schedule")
        return self


class SampleObjectiveScoringConfig(StrictBaseModel):
    pwm_pseudocounts: float = 0.10
    log_odds_clip: float | None = None

    @field_validator("pwm_pseudocounts")
    @classmethod
    def _check_pseudocounts(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or v < 0:
            raise ValueError("objective.scoring.pwm_pseudocounts must be >= 0")
        return float(v)

    @field_validator("log_odds_clip")
    @classmethod
    def _check_clip(cls, v: float | None) -> float | None:
        if v is None:
            return None
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError("objective.scoring.log_odds_clip must be > 0")
        return float(v)


class SampleObjectiveConfig(StrictBaseModel):
    bidirectional: bool = True
    score_scale: Literal["normalized-llr", "llr", "logp", "z", "consensus-neglop-sum"] = "normalized-llr"
    combine: Literal["min", "sum"] = "min"
    softmin: SampleObjectiveSoftminConfig = SampleObjectiveSoftminConfig()
    scoring: SampleObjectiveScoringConfig = SampleObjectiveScoringConfig()


class MoveConfig(StrictBaseModel):
    block_len_range: Tuple[int, int] = (3, 12)
    multi_k_range: Tuple[int, int] = (2, 6)
    slide_max_shift: int = 2
    swap_len_range: Tuple[int, int] = (6, 12)
    move_probs: Dict[Literal["S", "B", "M", "L", "W", "I"], float] = {
        "S": 0.80,
        "B": 0.10,
        "M": 0.10,
        "L": 0.00,
        "W": 0.00,
        "I": 0.00,
    }
    move_schedule: Optional["MoveScheduleConfig"] = None
    target_worst_tf_prob: float = 0.0
    target_window_pad: int = 0
    insertion_consensus_prob: float = 0.50

    @field_validator("block_len_range", "multi_k_range", "swap_len_range", mode="before")
    @classmethod
    def _list_to_tuple(cls, v):
        if isinstance(v, (list, tuple)):
            return tuple(int(x) for x in v)
        return v

    @staticmethod
    def _normalize_move_probs(v: Dict[str, float], *, label: str) -> Dict[str, float]:
        expected_keys = {"S", "B", "M", "L", "W", "I"}
        got_keys = set(v.keys())
        if not got_keys.issubset(expected_keys):
            extra = sorted(got_keys - expected_keys)
            raise ValueError(f"{label} keys must be a subset of {sorted(expected_keys)}, but got extra {extra}")
        total = 0.0
        out = {}
        for k in ("S", "B", "M", "L", "W", "I"):
            fv = float(v.get(k, 0.0))
            if fv < 0:
                raise ValueError(f"{label}['{k}'] must be \u2265 0, but got {fv}")
            out[k] = fv
            total += fv
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"{label} values must sum to 1.0; got sum={total:.6f}")
        return out

    @field_validator("move_probs")
    @classmethod
    def _check_move_probs_keys_and_values(cls, v):
        return cls._normalize_move_probs(v, label="move_probs")

    @field_validator("target_worst_tf_prob")
    @classmethod
    def _check_target_worst_tf_prob(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or v < 0 or v > 1:
            raise ValueError("target_worst_tf_prob must be between 0 and 1")
        return float(v)

    @field_validator("target_window_pad")
    @classmethod
    def _check_target_window_pad(cls, v: int) -> int:
        if not isinstance(v, int) or v < 0:
            raise ValueError("target_window_pad must be a non-negative integer")
        return v

    @field_validator("insertion_consensus_prob")
    @classmethod
    def _check_insertion_consensus_prob(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or v < 0 or v > 1:
            raise ValueError("insertion_consensus_prob must be between 0 and 1")
        return float(v)


class MoveScheduleConfig(StrictBaseModel):
    enabled: bool = False
    kind: Literal["linear"] = "linear"
    end: Optional[Dict[Literal["S", "B", "M", "L", "W", "I"], float]] = None

    @field_validator("end")
    @classmethod
    def _check_end_probs(cls, v):
        if v is None:
            return v
        return MoveConfig._normalize_move_probs(v, label="move_schedule.end")

    @model_validator(mode="after")
    def _validate_schedule(self) -> "MoveScheduleConfig":
        if self.enabled and self.end is None:
            raise ValueError("move_schedule.enabled=true requires move_schedule.end")
        return self


class MoveOverridesConfig(StrictBaseModel):
    block_len_range: Optional[Tuple[int, int]] = None
    multi_k_range: Optional[Tuple[int, int]] = None
    slide_max_shift: Optional[int] = None
    swap_len_range: Optional[Tuple[int, int]] = None
    move_probs: Optional[Dict[Literal["S", "B", "M", "L", "W", "I"], float]] = None
    move_schedule: Optional[MoveScheduleConfig] = None
    target_worst_tf_prob: Optional[float] = None
    target_window_pad: Optional[int] = None
    insertion_consensus_prob: Optional[float] = None

    @field_validator("block_len_range", "multi_k_range", "swap_len_range", mode="before")
    @classmethod
    def _list_to_tuple(cls, v):
        if isinstance(v, (list, tuple)):
            return tuple(int(x) for x in v)
        return v

    @field_validator("target_worst_tf_prob")
    @classmethod
    def _check_target_worst_tf_prob(cls, v: float | None) -> float | None:
        if v is None:
            return None
        if not isinstance(v, (int, float)) or v < 0 or v > 1:
            raise ValueError("moves.overrides.target_worst_tf_prob must be between 0 and 1")
        return float(v)

    @field_validator("insertion_consensus_prob")
    @classmethod
    def _check_insertion_consensus_prob(cls, v: float | None) -> float | None:
        if v is None:
            return v
        if not isinstance(v, (int, float)) or v < 0 or v > 1:
            raise ValueError("moves.overrides.insertion_consensus_prob must be between 0 and 1")
        return float(v)


class SampleMovesConfig(StrictBaseModel):
    profile: Literal["balanced", "local", "global", "aggressive"] = "balanced"
    overrides: Optional[MoveOverridesConfig] = None


class SamplePtAdaptConfig(StrictBaseModel):
    enabled: bool = True
    target_swap: float = 0.25
    window: int = 50
    k: float = 0.5
    min_scale: float = 0.25
    max_scale: float = 4.0
    stop_after_tune: bool = True

    @field_validator("target_swap")
    @classmethod
    def _check_target_swap(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or v <= 0 or v >= 1:
            raise ValueError("pt.adapt.target_swap must be between 0 and 1")
        return float(v)

    @field_validator("window")
    @classmethod
    def _check_window(cls, v: int) -> int:
        if v < 1:
            raise ValueError("pt.adapt.window must be >= 1")
        return int(v)

    @field_validator("k", "min_scale", "max_scale")
    @classmethod
    def _check_scales(cls, v: float, info) -> float:
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError(f"pt.adapt.{info.field_name} must be > 0")
        return float(v)

    @model_validator(mode="after")
    def _check_scale_bounds(self) -> "SamplePtAdaptConfig":
        if self.max_scale < self.min_scale:
            raise ValueError("pt.adapt.max_scale must be >= pt.adapt.min_scale")
        return self


class SamplePtConfig(StrictBaseModel):
    n_temps: int = 6
    temp_max: float = 20.0
    swap_stride: int = 1
    adapt: SamplePtAdaptConfig = SamplePtAdaptConfig()

    @field_validator("n_temps")
    @classmethod
    def _check_n_temps(cls, v: int) -> int:
        if v < 1:
            raise ValueError("sample.pt.n_temps must be >= 1")
        return int(v)

    @field_validator("temp_max")
    @classmethod
    def _check_temp_max(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or v < 1.0:
            raise ValueError("sample.pt.temp_max must be >= 1.0")
        return float(v)

    @field_validator("swap_stride")
    @classmethod
    def _check_swap_stride(cls, v: int) -> int:
        if v < 1:
            raise ValueError("sample.pt.swap_stride must be >= 1")
        return int(v)


class SampleEliteFilterConfig(StrictBaseModel):
    min_per_tf_norm: Union[Literal["auto"], float, None] = "auto"
    require_all_tfs: bool = True
    pwm_sum_min: float = 0.0

    @field_validator("min_per_tf_norm")
    @classmethod
    def _check_min_per_tf_norm(cls, v):
        if v is None or v == "auto":
            return v
        if not isinstance(v, (int, float)) or v < 0 or v > 1:
            raise ValueError("elites.filter.min_per_tf_norm must be 'auto', null, or between 0 and 1")
        return float(v)

    @field_validator("pwm_sum_min")
    @classmethod
    def _check_pwm_sum_min(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or v < 0:
            raise ValueError("elites.filter.pwm_sum_min must be >= 0")
        return float(v)


class SampleEliteSelectConfig(StrictBaseModel):
    policy: Literal["mmr"] = "mmr"
    alpha: float = 0.85
    pool_size: Union[Literal["auto"], int] = "auto"
    diversity_metric: Literal["tfbs_core_weighted_hamming"] = "tfbs_core_weighted_hamming"

    @field_validator("alpha")
    @classmethod
    def _check_alpha(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or v <= 0 or v > 1:
            raise ValueError("elites.select.alpha must be in (0, 1]")
        return float(v)

    @field_validator("pool_size")
    @classmethod
    def _check_pool_size(cls, v):
        if v == "auto":
            return v
        if not isinstance(v, int) or v < 1:
            raise ValueError("elites.select.pool_size must be 'auto' or >= 1")
        return v


class SampleElitesConfig(StrictBaseModel):
    k: int = 10
    filter: SampleEliteFilterConfig = SampleEliteFilterConfig()
    select: SampleEliteSelectConfig = SampleEliteSelectConfig()

    @field_validator("k")
    @classmethod
    def _check_k(cls, v: int) -> int:
        if v < 1:
            raise ValueError("sample.elites.k must be >= 1")
        return int(v)


class SampleOutputConfig(StrictBaseModel):
    save_sequences: bool = True
    save_trace: bool = True
    include_tune_in_sequences: bool = False
    live_metrics: bool = True


class SampleConfig(StrictBaseModel):
    seed: int
    sequence_length: int
    budget: SampleBudgetConfig
    objective: SampleObjectiveConfig = SampleObjectiveConfig()
    moves: SampleMovesConfig = SampleMovesConfig()
    pt: SamplePtConfig = SamplePtConfig()
    elites: SampleElitesConfig = SampleElitesConfig()
    output: SampleOutputConfig = SampleOutputConfig()

    @field_validator("seed")
    @classmethod
    def _check_seed(cls, v: int) -> int:
        if not isinstance(v, int) or v < 0:
            raise ValueError("sample.seed must be a non-negative integer")
        return v

    @field_validator("sequence_length")
    @classmethod
    def _check_sequence_length(cls, v: int) -> int:
        if not isinstance(v, int) or v < 1:
            raise ValueError("sample.sequence_length must be >= 1")
        return v


class AnalysisConfig(StrictBaseModel):
    enabled: bool = True
    run_selector: Literal["latest", "explicit"] = "latest"
    runs: List[str] = Field(default_factory=list)
    pairwise: Union[Literal["off", "auto"], List[str]] = "auto"
    plot_format: Literal["png", "pdf", "svg"] = "png"
    plot_dpi: int = 150
    table_format: Literal["parquet", "csv"] = "parquet"
    archive: bool = False
    max_points: int = 5000

    @field_validator("plot_dpi", "max_points")
    @classmethod
    def _check_positive_ints(cls, v: int, info) -> int:
        if not isinstance(v, int) or v < 1:
            raise ValueError(f"analysis.{info.field_name} must be >= 1")
        return v

    @field_validator("runs")
    @classmethod
    def _check_runs(cls, v: List[str]) -> List[str]:
        cleaned: list[str] = []
        for raw in v:
            name = str(raw).strip()
            if not name:
                raise ValueError("analysis.runs entries must be non-empty strings")
            cleaned.append(name)
        return cleaned

    @field_validator("pairwise")
    @classmethod
    def _check_pairwise(cls, v):
        if v in ("off", "auto"):
            return v
        if isinstance(v, list):
            if len(v) != 2:
                raise ValueError("analysis.pairwise must be 'off', 'auto', or a list of two TF names")
            tf_a = str(v[0]).strip()
            tf_b = str(v[1]).strip()
            if not tf_a or not tf_b:
                raise ValueError("analysis.pairwise TF names must be non-empty")
            return [tf_a, tf_b]
        raise ValueError("analysis.pairwise must be 'off', 'auto', or a list of two TF names")


class CruncherConfig(StrictBaseModel):
    schema_version: int = Field(3, description="Schema version (must be 3).")
    workspace: WorkspaceConfig
    io: IOConfig = IOConfig()
    catalog: CatalogConfig = CatalogConfig()
    discover: DiscoverConfig = DiscoverConfig()
    ingest: IngestConfig = IngestConfig()
    sample: Optional[SampleConfig] = None
    analysis: Optional[AnalysisConfig] = None
    campaigns: List[CampaignConfig] = Field(default_factory=list)
    campaign: Optional[CampaignMetadataConfig] = None

    @model_validator(mode="after")
    def _check_schema_version(self) -> "CruncherConfig":
        if self.schema_version != 3:
            raise ValueError("Config schema v3 required (schema_version: 3)")
        return self

    @property
    def out_dir(self) -> Path:
        return self.workspace.out_dir

    @property
    def regulator_sets(self) -> List[List[str]]:
        return self.workspace.regulator_sets

    @property
    def regulator_categories(self) -> Dict[str, List[str]]:
        return self.workspace.regulator_categories

    @property
    def motif_store(self) -> CatalogConfig:
        return self.catalog

    @property
    def motif_discovery(self) -> DiscoverConfig:
        return self.discover


class CruncherRoot(StrictBaseModel):
    cruncher: CruncherConfig
