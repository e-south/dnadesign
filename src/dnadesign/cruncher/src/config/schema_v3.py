"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/config/schema_v3.py

Defines the Cruncher v3 configuration schema.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


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


class OrganismConfig(StrictBaseModel):
    taxon: Optional[int] = None
    name: Optional[str] = None
    strain: Optional[str] = None
    assembly: Optional[str] = None


class CampaignSelectorsConfig(StrictBaseModel):
    min_info_bits: Optional[float] = None
    min_site_count: Optional[int] = None
    min_pwm_length: Optional[int] = None
    max_pwm_length: Optional[int] = None
    source_preference: List[str] = Field(default_factory=list)
    dataset_preference: List[str] = Field(default_factory=list)

    @field_validator("min_info_bits")
    @classmethod
    def _check_min_info_bits(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return None
        if not isinstance(v, (int, float)) or v < 0:
            raise ValueError("selectors.min_info_bits must be a non-negative number")
        return float(v)

    @field_validator("min_site_count", "min_pwm_length", "max_pwm_length")
    @classmethod
    def _check_non_negative_ints(cls, v: Optional[int], info) -> Optional[int]:
        if v is None:
            return None
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"selectors.{info.field_name} must be a non-negative integer")
        return v

    @field_validator("source_preference", "dataset_preference")
    @classmethod
    def _check_text_list(cls, v: List[str], info) -> List[str]:
        cleaned: list[str] = []
        for item in v:
            name = str(item).strip()
            if not name:
                raise ValueError(f"selectors.{info.field_name} entries must be non-empty strings")
            cleaned.append(name)
        return cleaned

    @model_validator(mode="after")
    def _check_pwm_length_bounds(self) -> "CampaignSelectorsConfig":
        if self.min_pwm_length is not None and self.max_pwm_length is not None:
            if self.max_pwm_length < self.min_pwm_length:
                raise ValueError("selectors.max_pwm_length must be >= selectors.min_pwm_length")
        return self

    def requires_catalog(self) -> bool:
        return any(
            [
                self.min_info_bits is not None,
                self.min_site_count is not None,
                self.min_pwm_length is not None,
                self.max_pwm_length is not None,
                bool(self.source_preference),
                bool(self.dataset_preference),
            ]
        )


class CampaignWithinCategoryConfig(StrictBaseModel):
    sizes: List[int] = Field(default_factory=list)

    @field_validator("sizes")
    @classmethod
    def _check_sizes(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("within_category.sizes must be a non-empty list")
        cleaned: list[int] = []
        for size in v:
            if not isinstance(size, int) or size < 1:
                raise ValueError("within_category.sizes must be positive integers")
            cleaned.append(size)
        return sorted(set(cleaned))


class CampaignAcrossCategoriesConfig(StrictBaseModel):
    sizes: List[int] = Field(default_factory=list)
    max_per_category: Optional[int] = None

    @field_validator("sizes")
    @classmethod
    def _check_sizes(cls, v: List[int]) -> List[int]:
        if not v:
            raise ValueError("across_categories.sizes must be a non-empty list")
        cleaned: list[int] = []
        for size in v:
            if not isinstance(size, int) or size < 2:
                raise ValueError("across_categories.sizes must be integers >= 2")
            cleaned.append(size)
        return sorted(set(cleaned))

    @field_validator("max_per_category")
    @classmethod
    def _check_max_per_category(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        if not isinstance(v, int) or v < 1:
            raise ValueError("across_categories.max_per_category must be a positive integer")
        return v


class CampaignConfig(StrictBaseModel):
    name: str
    categories: List[str]
    within_category: Optional[CampaignWithinCategoryConfig] = None
    across_categories: Optional[CampaignAcrossCategoriesConfig] = None
    allow_overlap: bool = True
    distinct_across_categories: bool = True
    dedupe_sets: bool = True
    selectors: CampaignSelectorsConfig = CampaignSelectorsConfig()
    tags: Dict[str, str] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _check_name(cls, v: str) -> str:
        name = str(v).strip()
        if not name:
            raise ValueError("campaign.name must be a non-empty string")
        return name

    @field_validator("categories")
    @classmethod
    def _check_categories(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("campaign.categories must be a non-empty list")
        cleaned: list[str] = []
        for item in v:
            name = str(item).strip()
            if not name:
                raise ValueError("campaign.categories entries must be non-empty strings")
            cleaned.append(name)
        if len(set(cleaned)) != len(cleaned):
            raise ValueError("campaign.categories must be unique")
        return cleaned

    @field_validator("tags")
    @classmethod
    def _check_tags(cls, v: Dict[str, str]) -> Dict[str, str]:
        cleaned: dict[str, str] = {}
        for key, value in v.items():
            key_clean = str(key).strip()
            if not key_clean:
                raise ValueError("campaign.tags keys must be non-empty strings")
            cleaned[key_clean] = str(value)
        return cleaned

    @model_validator(mode="after")
    def _check_rules(self) -> "CampaignConfig":
        if self.within_category is None and self.across_categories is None:
            raise ValueError("campaign must define within_category or across_categories rules")
        return self


class CampaignMetadataConfig(StrictBaseModel):
    name: str
    campaign_id: str
    manifest_path: Optional[Path] = None
    generated_at: Optional[str] = None

    @field_validator("name", "campaign_id")
    @classmethod
    def _check_required_text(cls, v: str, info) -> str:
        text = str(v).strip()
        if not text:
            raise ValueError(f"campaign.{info.field_name} must be a non-empty string")
        return text


class LocalMotifSourceConfig(StrictBaseModel):
    source_id: str = Field(..., description="Unique identifier for the local motif source.")
    description: Optional[str] = Field(None, description="Human-readable description for the source list.")
    root: Path = Field(..., description="Root directory containing motif files.")
    patterns: List[str] = Field(
        default_factory=lambda: ["*.txt"],
        description="Glob patterns to select motif files under root.",
    )
    recursive: bool = Field(False, description="Recursively search subdirectories when true.")
    format_map: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from file extension to parser format (e.g., .txt -> MEME).",
    )
    default_format: Optional[str] = Field(
        None,
        description="Default parser format when an extension is not listed in format_map.",
    )
    tf_name_strategy: Literal["stem", "filename"] = Field(
        "stem",
        description="Strategy for deriving TF names from files.",
    )
    matrix_semantics: Literal["probabilities", "weights"] = Field(
        "probabilities",
        description="Matrix semantics to store in the catalog.",
    )
    organism: Optional[OrganismConfig] = None
    citation: Optional[str] = None
    license: Optional[str] = None
    source_url: Optional[str] = None
    source_version: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)
    extract_sites: bool = Field(
        False,
        description="Enable binding-site extraction from MEME BLOCKS sections.",
    )
    meme_motif_selector: Optional[Union[str, int]] = Field(
        None,
        description="Select a motif from multi-motif MEME files (name_match, MEME-1, index, or label).",
    )

    @field_validator("source_id")
    @classmethod
    def _check_source_id(cls, v: str) -> str:
        source_id = str(v).strip()
        if not source_id:
            raise ValueError("local source_id must be a non-empty string")
        return source_id

    @field_validator("patterns")
    @classmethod
    def _check_patterns(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("local source patterns must be a non-empty list")
        cleaned = []
        for pat in v:
            pat = str(pat).strip()
            if not pat:
                raise ValueError("local source patterns must be non-empty strings")
            cleaned.append(pat)
        return cleaned

    @model_validator(mode="after")
    def _check_format_config(self) -> "LocalMotifSourceConfig":
        if not self.format_map and not self.default_format:
            raise ValueError("local source must set format_map or default_format")
        return self


class LocalSiteSourceConfig(StrictBaseModel):
    source_id: str = Field(..., description="Unique identifier for the local site source.")
    description: Optional[str] = Field(None, description="Human-readable description for the source list.")
    path: Path = Field(..., description="Path to a FASTA file containing binding sites.")
    tf_name: Optional[str] = Field(
        None,
        description="Optional TF name override (defaults to the FASTA header prefix).",
    )
    record_kind: Optional[str] = Field(
        None,
        description="Optional site-kind label stored in provenance tags (e.g., chip_exo).",
    )
    organism: Optional[OrganismConfig] = None
    citation: Optional[str] = None
    license: Optional[str] = None
    source_url: Optional[str] = None
    source_version: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)

    @field_validator("source_id")
    @classmethod
    def _check_source_id(cls, v: str) -> str:
        source_id = str(v).strip()
        if not source_id:
            raise ValueError("site source_id must be a non-empty string")
        return source_id

    @field_validator("tf_name", "record_kind")
    @classmethod
    def _check_optional_text(cls, v: Optional[str], info) -> Optional[str]:
        if v is None:
            return None
        value = str(v).strip()
        if not value:
            raise ValueError(f"{info.field_name} must be a non-empty string")
        return value


class RegulonDBConfig(StrictBaseModel):
    base_url: str = "https://regulondb.ccg.unam.mx/graphql"
    verify_ssl: bool = True
    ca_bundle: Optional[Path] = None
    timeout_seconds: int = 30
    motif_matrix_source: Literal["alignment", "sites"] = "alignment"
    alignment_matrix_semantics: Literal["probabilities", "counts"] = "probabilities"
    min_sites_for_pwm: int = 2
    pseudocounts: float = Field(
        0.5,
        description="Pseudocounts for PWM construction from curated sites (Biopython).",
    )
    allow_low_sites: bool = False
    curated_sites: bool = True
    ht_sites: bool = False
    ht_dataset_sources: Optional[List[str]] = None
    ht_dataset_type: Literal["TFBINDING"] = "TFBINDING"
    ht_binding_mode: Literal["tfbinding", "peaks"] = "tfbinding"
    uppercase_binding_site_only: bool = True

    @field_validator("min_sites_for_pwm")
    @classmethod
    def _check_min_sites(cls, v: int) -> int:
        if v < 1:
            raise ValueError("min_sites_for_pwm must be >= 1")
        return v

    @field_validator("pseudocounts")
    @classmethod
    def _check_pseudocounts(cls, v: float) -> float:
        if v < 0:
            raise ValueError("pseudocounts must be >= 0")
        return float(v)


class HttpRetryConfig(StrictBaseModel):
    retries: int = Field(3, description="Number of retry attempts for transient network failures.")
    backoff_seconds: float = Field(0.5, description="Base backoff (seconds) between retries.")
    max_backoff_seconds: float = Field(8.0, description="Maximum backoff between retries.")
    retry_statuses: List[int] = Field(
        default_factory=lambda: [429, 500, 502, 503, 504],
        description="HTTP status codes that trigger retry.",
    )
    respect_retry_after: bool = Field(True, description="Respect Retry-After header when provided.")

    @field_validator("retries")
    @classmethod
    def _check_retries(cls, v: int) -> int:
        if v < 0:
            raise ValueError("retries must be >= 0")
        return v

    @field_validator("backoff_seconds", "max_backoff_seconds")
    @classmethod
    def _check_backoff(cls, v: float) -> float:
        if v < 0:
            raise ValueError("backoff_seconds must be >= 0")
        return float(v)


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
                    raise ValueError(f"workspace.regulator_sets[{idx}] contains duplicate TF '{name}'")
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
                    raise ValueError(f"workspace.regulator_categories['{name}'] contains duplicate TF '{tf_name}'")
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

    @property
    def pwm_window_lengths(self) -> Dict[str, int]:
        return {}

    @property
    def pwm_window_strategy(self) -> str:
        return "max_info"


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
    schedule: Literal["fixed", "linear"] = "fixed"
    beta_start: float = 0.5
    beta_end: float = 6.0

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


class MoveAdaptiveWeightsConfig(StrictBaseModel):
    enabled: bool = False
    window: int = 250
    k: float = 0.5
    min_prob: float = 0.01
    max_prob: float = 0.95
    targets: Dict[Literal["S", "B", "M", "L", "W", "I"], float] = {
        "S": 0.95,
        "B": 0.40,
        "M": 0.35,
        "I": 0.35,
    }
    kinds: List[Literal["S", "B", "M", "L", "W", "I"]] = Field(default_factory=lambda: ["S", "B", "M", "I"])

    @field_validator("window")
    @classmethod
    def _check_window(cls, v: int) -> int:
        if v < 1:
            raise ValueError("moves.adaptive_weights.window must be >= 1")
        return int(v)

    @field_validator("k")
    @classmethod
    def _check_k(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError("moves.adaptive_weights.k must be > 0")
        return float(v)

    @field_validator("min_prob", "max_prob")
    @classmethod
    def _check_prob_bounds(cls, v: float, info) -> float:
        if not isinstance(v, (int, float)) or v <= 0 or v > 1:
            raise ValueError(f"moves.adaptive_weights.{info.field_name} must be in (0, 1]")
        return float(v)

    @field_validator("targets")
    @classmethod
    def _check_targets(cls, v: Dict[str, float]) -> Dict[str, float]:
        out: dict[str, float] = {}
        for key, value in v.items():
            fv = float(value)
            if fv <= 0 or fv >= 1:
                raise ValueError(f"moves.adaptive_weights.targets['{key}'] must be in (0, 1)")
            out[str(key)] = fv
        return out

    @field_validator("kinds")
    @classmethod
    def _check_kinds(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("moves.adaptive_weights.kinds must be non-empty")
        return [str(kind) for kind in v]

    @model_validator(mode="after")
    def _check_min_max(self) -> "MoveAdaptiveWeightsConfig":
        if self.max_prob < self.min_prob:
            raise ValueError("moves.adaptive_weights.max_prob must be >= min_prob")
        return self


class MoveProposalAdaptConfig(StrictBaseModel):
    enabled: bool = False
    window: int = 250
    step: float = 0.10
    min_scale: float = 0.50
    max_scale: float = 2.0
    target_low: float = 0.25
    target_high: float = 0.75

    @field_validator("window")
    @classmethod
    def _check_window(cls, v: int) -> int:
        if v < 1:
            raise ValueError("moves.proposal_adapt.window must be >= 1")
        return int(v)

    @field_validator("step", "min_scale", "max_scale")
    @classmethod
    def _check_positive(cls, v: float, info) -> float:
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError(f"moves.proposal_adapt.{info.field_name} must be > 0")
        return float(v)

    @field_validator("target_low", "target_high")
    @classmethod
    def _check_target_bounds(cls, v: float, info) -> float:
        if not isinstance(v, (int, float)) or v < 0 or v > 1:
            raise ValueError(f"moves.proposal_adapt.{info.field_name} must be in [0, 1]")
        return float(v)

    @model_validator(mode="after")
    def _check_ranges(self) -> "MoveProposalAdaptConfig":
        if self.max_scale < self.min_scale:
            raise ValueError("moves.proposal_adapt.max_scale must be >= min_scale")
        if self.target_high <= self.target_low:
            raise ValueError("moves.proposal_adapt.target_high must be > target_low")
        return self


class MoveConfig(StrictBaseModel):
    block_len_range: Tuple[int, int] = (2, 6)
    multi_k_range: Tuple[int, int] = (2, 3)
    slide_max_shift: int = 2
    swap_len_range: Tuple[int, int] = (6, 12)
    move_probs: Dict[Literal["S", "B", "M", "L", "W", "I"], float] = {
        "S": 0.85,
        "B": 0.07,
        "M": 0.04,
        "L": 0.00,
        "W": 0.00,
        "I": 0.04,
    }
    move_schedule: Optional["MoveScheduleConfig"] = None
    target_worst_tf_prob: float = 0.0
    target_window_pad: int = 0
    insertion_consensus_prob: float = 0.35
    adaptive_weights: MoveAdaptiveWeightsConfig = MoveAdaptiveWeightsConfig()
    proposal_adapt: MoveProposalAdaptConfig = MoveProposalAdaptConfig()

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
    adaptive_weights: Optional[MoveAdaptiveWeightsConfig] = None
    proposal_adapt: Optional[MoveProposalAdaptConfig] = None

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


class SampleOptimizerCoolingStageConfig(StrictBaseModel):
    sweeps: int
    beta: float

    @field_validator("sweeps")
    @classmethod
    def _check_sweeps(cls, v: int) -> int:
        if not isinstance(v, int) or v < 1:
            raise ValueError("sample.optimizer.cooling.stages[].sweeps must be >= 1")
        return int(v)

    @field_validator("beta")
    @classmethod
    def _check_beta(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError("sample.optimizer.cooling.stages[].beta must be > 0")
        return float(v)


class SampleOptimizerCoolingFixedConfig(StrictBaseModel):
    kind: Literal["fixed"] = "fixed"
    beta: float = 1.0

    @field_validator("beta")
    @classmethod
    def _check_beta(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError("sample.optimizer.cooling.beta must be > 0")
        return float(v)


class SampleOptimizerCoolingLinearConfig(StrictBaseModel):
    kind: Literal["linear"] = "linear"
    beta_start: float = 0.20
    beta_end: float = 4.0

    @field_validator("beta_start", "beta_end")
    @classmethod
    def _check_positive_betas(cls, v: float, info) -> float:
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError(f"sample.optimizer.cooling.{info.field_name} must be > 0")
        return float(v)

    @model_validator(mode="after")
    def _check_beta_order(self) -> "SampleOptimizerCoolingLinearConfig":
        if self.beta_end < self.beta_start:
            raise ValueError("sample.optimizer.cooling.beta_end must be >= beta_start when kind='linear'")
        return self


class SampleOptimizerCoolingPiecewiseConfig(StrictBaseModel):
    kind: Literal["piecewise"] = "piecewise"
    stages: List[SampleOptimizerCoolingStageConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_piecewise_stages(self) -> "SampleOptimizerCoolingPiecewiseConfig":
        if not self.stages:
            raise ValueError("sample.optimizer.cooling.stages must be non-empty when kind='piecewise'")
        sweeps = [stage.sweeps for stage in self.stages]
        if any(curr <= prev for prev, curr in zip(sweeps, sweeps[1:])):
            raise ValueError("sample.optimizer.cooling.stages[].sweeps must be strictly increasing")
        return self


SampleOptimizerCoolingConfig = Annotated[
    Union[
        SampleOptimizerCoolingFixedConfig,
        SampleOptimizerCoolingLinearConfig,
        SampleOptimizerCoolingPiecewiseConfig,
    ],
    Field(discriminator="kind"),
]


class SampleOptimizerEarlyStopConfig(StrictBaseModel):
    enabled: bool = False
    patience: int = 0
    min_delta: float = 0.0
    require_min_unique: bool = False
    min_unique: int = 0
    success_min_per_tf_norm: float = 0.0

    @field_validator("patience", "min_unique")
    @classmethod
    def _check_non_negative_ints(cls, v: int, info) -> int:
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"sample.optimizer.early_stop.{info.field_name} must be >= 0")
        return int(v)

    @field_validator("min_delta")
    @classmethod
    def _check_min_delta(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or v < 0:
            raise ValueError("sample.optimizer.early_stop.min_delta must be >= 0")
        return float(v)

    @field_validator("success_min_per_tf_norm")
    @classmethod
    def _check_success_threshold(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or v < 0 or v > 1:
            raise ValueError("sample.optimizer.early_stop.success_min_per_tf_norm must be between 0 and 1")
        return float(v)

    @model_validator(mode="after")
    def _check_consistency(self) -> "SampleOptimizerEarlyStopConfig":
        if self.enabled and self.patience < 1:
            raise ValueError("sample.optimizer.early_stop.patience must be >= 1 when enabled=true")
        if self.require_min_unique and self.min_unique < 1:
            raise ValueError("sample.optimizer.early_stop.min_unique must be >= 1 when require_min_unique=true")
        return self


class SampleOptimizerConfig(StrictBaseModel):
    kind: Literal["gibbs_anneal"] = "gibbs_anneal"
    chains: int = 1
    cooling: SampleOptimizerCoolingConfig = Field(default_factory=SampleOptimizerCoolingLinearConfig)
    early_stop: SampleOptimizerEarlyStopConfig = SampleOptimizerEarlyStopConfig()

    @field_validator("chains")
    @classmethod
    def _check_chains(cls, v: int) -> int:
        if not isinstance(v, int) or v < 1:
            raise ValueError("sample.optimizer.chains must be >= 1")
        return int(v)


class SampleEliteFilterConfig(StrictBaseModel):
    min_per_tf_norm: float | None = None
    require_all_tfs: bool = True
    pwm_sum_min: float = 0.0

    @field_validator("min_per_tf_norm")
    @classmethod
    def _check_min_per_tf_norm(cls, v):
        if v is None:
            return v
        if not isinstance(v, (int, float)) or v < 0 or v > 1:
            raise ValueError("elites.filter.min_per_tf_norm must be null or between 0 and 1")
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

    @model_validator(mode="after")
    def _check_sequences_required(self) -> "SampleOutputConfig":
        if not self.save_sequences:
            raise ValueError("sample.output.save_sequences must be true because analyze requires sequences.parquet")
        return self


class SampleMotifWidthConfig(StrictBaseModel):
    minw: Optional[int] = None
    maxw: Optional[int] = None
    strategy: Literal["max_info"] = "max_info"

    @field_validator("minw", "maxw")
    @classmethod
    def _check_optional_positive_ints(cls, v: Optional[int], info) -> Optional[int]:
        if v is None:
            return v
        if v < 1:
            raise ValueError(f"sample.motif_width.{info.field_name} must be >= 1")
        return int(v)

    @model_validator(mode="after")
    def _check_widths(self) -> "SampleMotifWidthConfig":
        if self.minw is not None and self.maxw is not None and self.maxw < self.minw:
            raise ValueError("sample.motif_width.maxw must be >= sample.motif_width.minw")
        return self


class SampleConfig(StrictBaseModel):
    seed: int
    sequence_length: int
    budget: SampleBudgetConfig
    objective: SampleObjectiveConfig = SampleObjectiveConfig()
    moves: SampleMovesConfig = SampleMovesConfig()
    optimizer: SampleOptimizerConfig = SampleOptimizerConfig()
    elites: SampleElitesConfig = SampleElitesConfig()
    output: SampleOutputConfig = SampleOutputConfig()
    motif_width: SampleMotifWidthConfig = SampleMotifWidthConfig()

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

    @model_validator(mode="after")
    def _check_motif_width_bounds(self) -> "SampleConfig":
        minw = self.motif_width.minw
        maxw = self.motif_width.maxw
        if minw is not None and minw > self.sequence_length:
            raise ValueError("sample.motif_width.minw must be <= sample.sequence_length")
        if maxw is not None and maxw > self.sequence_length:
            raise ValueError("sample.motif_width.maxw must be <= sample.sequence_length")
        return self


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
    trajectory_stride: int = 5
    trajectory_scatter_scale: Literal["normalized-llr", "llr"] = "llr"
    trajectory_sweep_y_column: Literal["raw_llr_objective", "objective_scalar", "norm_llr_objective"] = (
        "raw_llr_objective"
    )
    trajectory_particle_alpha_min: float = 0.15
    trajectory_particle_alpha_max: float = 0.45
    trajectory_chain_overlay: bool = False

    @field_validator(
        "plot_dpi",
        "max_points",
        "trajectory_stride",
    )
    @classmethod
    def _check_positive_ints(cls, v: int, info) -> int:
        if not isinstance(v, int) or v < 1:
            raise ValueError(f"analysis.{info.field_name} must be >= 1")
        return v

    @field_validator("trajectory_particle_alpha_min", "trajectory_particle_alpha_max")
    @classmethod
    def _check_alpha_bounds(cls, v: float, info) -> float:
        if not isinstance(v, (int, float)):
            raise ValueError(f"analysis.{info.field_name} must be numeric")
        value = float(v)
        if value < 0 or value > 1:
            raise ValueError(f"analysis.{info.field_name} must be between 0 and 1")
        return value

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

    @model_validator(mode="after")
    def _check_particle_alpha_order(self) -> "AnalysisConfig":
        if self.trajectory_particle_alpha_max < self.trajectory_particle_alpha_min:
            raise ValueError("analysis.trajectory_particle_alpha_max must be >= analysis.trajectory_particle_alpha_min")
        return self


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
        if not self.workspace.regulator_sets and not self.campaigns:
            raise ValueError("Config must define workspace.regulator_sets or campaigns.")
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


class CruncherRoot(StrictBaseModel):
    cruncher: CruncherConfig
