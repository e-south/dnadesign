"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/config/schema_v2.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class PlotConfig(BaseModel):
    logo: bool
    bits_mode: Literal["information", "probability"]
    dpi: int


class ParseConfig(BaseModel):
    plot: PlotConfig


class ParserConfig(BaseModel):
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


class IOConfig(BaseModel):
    parsers: ParserConfig = ParserConfig()


class OrganismConfig(BaseModel):
    taxon: Optional[int] = None
    name: Optional[str] = None
    strain: Optional[str] = None
    assembly: Optional[str] = None


class MoveConfig(BaseModel):
    block_len_range: Tuple[int, int] = (3, 12)
    multi_k_range: Tuple[int, int] = (2, 6)
    slide_max_shift: int = 2
    swap_len_range: Tuple[int, int] = (6, 12)
    move_probs: Dict[Literal["S", "B", "M"], float] = {"S": 0.50, "B": 0.30, "M": 0.20}

    @field_validator("block_len_range", "multi_k_range", "swap_len_range", mode="before")
    @classmethod
    def _list_to_tuple(cls, v):
        if isinstance(v, (list, tuple)):
            return tuple(int(x) for x in v)
        return v

    @field_validator("move_probs")
    @classmethod
    def _check_move_probs_keys_and_values(cls, v):
        expected_keys = {"S", "B", "M"}
        got_keys = set(v.keys())
        if got_keys != expected_keys:
            raise ValueError(f"move_probs keys must be exactly {expected_keys}, but got {got_keys}")
        total = 0.0
        out = {}
        for k in ("S", "B", "M"):
            fv = float(v[k])
            if fv < 0:
                raise ValueError(f"move_probs['{k}'] must be ≥ 0, but got {fv}")
            out[k] = fv
            total += fv
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"move_probs values must sum to 1.0; got sum={total:.6f}")
        return out


class CoolingFixed(BaseModel):
    kind: Literal["fixed"] = "fixed"
    beta: float = 1.0

    @field_validator("beta")
    @classmethod
    def _check_positive_beta(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Fixed cooling beta must be > 0")
        return v


class CoolingLinear(BaseModel):
    kind: Literal["linear"] = "linear"
    beta: Tuple[float, float]

    @field_validator("beta")
    @classmethod
    def _two_positive(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        if len(v) != 2:
            raise ValueError("Linear cooling.beta must be length-2 [beta_start, beta_end]")
        if v[0] <= 0 or v[1] <= 0:
            raise ValueError("Both β_start and β_end must be > 0")
        return v


class CoolingGeometric(BaseModel):
    kind: Literal["geometric"] = "geometric"
    beta: List[float]

    @field_validator("beta")
    @classmethod
    def _check_list_positive(cls, v: List[float]) -> List[float]:
        if not isinstance(v, list) or len(v) < 2:
            raise ValueError("Geometric cooling.beta must be a list of at least two positive floats")
        if any(x <= 0 for x in v):
            raise ValueError("All entries in geometric β list must be > 0")
        return v


CoolingConfig = Union[CoolingFixed, CoolingLinear, CoolingGeometric]


class OptimiserConfig(BaseModel):
    kind: Literal["gibbs", "pt"]
    scorer_scale: Literal["llr", "z", "logp", "consensus-neglop-sum"]
    cooling: CoolingConfig
    swap_prob: float = 0.10


class InitConfig(BaseModel):
    kind: Literal["random", "consensus", "consensus_mix"]
    length: int
    regulator: Optional[str] = None
    pad_with: Optional[Literal["background", "A", "C", "G", "T"]] = "background"

    @field_validator("length")
    @classmethod
    def _check_length_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("init.length must be >= 1")
        return v

    @model_validator(mode="after")
    def _check_fields_for_modes(self) -> "InitConfig":
        if self.kind == "consensus" and not self.regulator:
            raise ValueError("When init.kind=='consensus', you must supply init.regulator=<PWM_name>")
        return self


class SampleConfig(BaseModel):
    bidirectional: bool = True
    seed: int = Field(42, description="Random seed for reproducible sampling.")
    record_tune: bool = Field(
        False,
        description="Whether to store burn-in states in sequences.parquet and optimizer buffers.",
    )
    progress_bar: bool = Field(
        True,
        description="Show progress bars during optimization.",
    )
    progress_every: int = Field(
        1000,
        description="Log a progress summary every N iterations per chain (0 disables logging).",
    )
    save_trace: bool = Field(
        True,
        description="Write trace.nc for analyze/report. Disable to skip NetCDF output.",
    )

    init: InitConfig
    draws: int
    tune: int
    chains: int
    min_dist: int
    top_k: int

    moves: MoveConfig = MoveConfig()
    optimiser: OptimiserConfig
    save_sequences: bool = True
    include_consensus_in_elites: bool = Field(
        False,
        description="Include PWM consensus strings in elite metadata (adds per-TF consensus to elites JSON).",
    )

    pwm_sum_threshold: float = Field(
        0.0,
        description="If >0, only sequences with sum(per-TF scaled_score) ≥ this are written to elites.json",
    )

    @field_validator("draws", "tune", "chains", "min_dist", "top_k")
    @classmethod
    def _check_positive_ints(cls, v: int) -> int:
        if not isinstance(v, int) or v < 0:
            raise ValueError("must be a non-negative integer")
        return v

    @field_validator("seed")
    @classmethod
    def _check_seed(cls, v: int) -> int:
        if not isinstance(v, int) or v < 0:
            raise ValueError("seed must be a non-negative integer")
        return v

    @field_validator("progress_every")
    @classmethod
    def _check_progress_every(cls, v: int) -> int:
        if not isinstance(v, int) or v < 0:
            raise ValueError("progress_every must be a non-negative integer")
        return v


class AnalysisPlotConfig(BaseModel):
    trace: bool = False
    autocorr: bool = False
    convergence: bool = False
    scatter_pwm: bool = False
    pair_pwm: bool = False
    parallel_pwm: bool = False
    pairgrid: bool = False
    score_hist: bool = False
    score_box: bool = False
    correlation_heatmap: bool = False
    parallel_coords: bool = False


class AnalysisConfig(BaseModel):
    runs: Optional[List[str]]
    plots: AnalysisPlotConfig = AnalysisPlotConfig()
    scatter_scale: Literal["llr", "z", "logp", "consensus-neglop-sum"]
    subsampling_epsilon: float
    scatter_style: Literal["edges", "thresholds"] = "edges"
    tf_pair: Optional[List[str]] = None
    archive: bool = Field(
        False,
        description=(
            "If true, archive previous analysis outputs under analysis/_archive/<analysis_id> "
            "before writing the new analysis."
        ),
    )

    @field_validator("subsampling_epsilon")
    @classmethod
    def _check_positive_epsilon(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or v <= 0.0:
            raise ValueError("subsampling_epsilon must be a positive number (float or int)")
        return float(v)

    @field_validator("tf_pair")
    @classmethod
    def _check_tf_pair(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return None
        if len(v) != 2:
            raise ValueError("analysis.tf_pair must contain exactly two TF names.")
        return v

    @model_validator(mode="after")
    def _check_scatter_style(self) -> "AnalysisConfig":
        if self.scatter_style == "thresholds" and self.scatter_scale != "llr":
            raise ValueError("scatter_style='thresholds' requires scatter_scale='llr'")
        return self


class CampaignSelectorsConfig(BaseModel):
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


class CampaignWithinCategoryConfig(BaseModel):
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


class CampaignAcrossCategoriesConfig(BaseModel):
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


class CampaignConfig(BaseModel):
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


class CampaignMetadataConfig(BaseModel):
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


class MotifStoreConfig(BaseModel):
    catalog_root: Path = Path(".cruncher")
    source_preference: List[str] = []
    allow_ambiguous: bool = False
    pwm_source: Literal["matrix", "sites"] = "matrix"
    site_kinds: Optional[List[str]] = None
    combine_sites: bool = False
    dataset_preference: List[str] = Field(
        default_factory=list,
        description="Preferred dataset IDs to resolve HT ambiguity (first match wins).",
    )
    dataset_map: Dict[str, str] = Field(
        default_factory=dict,
        description="Explicit TF->dataset_id map for HT data selection.",
    )
    site_window_lengths: Dict[str, int] = Field(
        default_factory=dict,
        description="Window length overrides for HT sites keyed by TF name or dataset:<id>.",
    )
    site_window_center: Literal["midpoint", "summit"] = "midpoint"
    min_sites_for_pwm: int = 2
    allow_low_sites: bool = False

    @field_validator("site_window_lengths")
    @classmethod
    def _check_window_lengths(cls, v: Dict[str, int]) -> Dict[str, int]:
        for key, value in v.items():
            if value <= 0:
                raise ValueError(f"site_window_lengths['{key}'] must be > 0")
        return v

    @field_validator("min_sites_for_pwm")
    @classmethod
    def _check_min_sites(cls, v: int) -> int:
        if v < 1:
            raise ValueError("min_sites_for_pwm must be >= 1")
        return v


class LocalMotifSourceConfig(BaseModel):
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


class RegulonDBConfig(BaseModel):
    base_url: str = "https://regulondb.ccg.unam.mx/graphql"
    verify_ssl: bool = True
    ca_bundle: Optional[Path] = None
    timeout_seconds: int = 30
    motif_matrix_source: Literal["alignment", "sites"] = "alignment"
    alignment_matrix_semantics: Literal["probabilities", "counts"] = "probabilities"
    min_sites_for_pwm: int = 2
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


class HttpRetryConfig(BaseModel):
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
            raise ValueError("backoff seconds must be >= 0")
        return float(v)


class IngestConfig(BaseModel):
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

    @field_validator("ncbi_timeout_seconds")
    @classmethod
    def _check_ncbi_timeout(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("ncbi_timeout_seconds must be > 0")
        return v


class CruncherConfig(BaseModel):
    out_dir: Path
    regulator_sets: List[List[str]]
    regulator_categories: Dict[str, List[str]] = Field(default_factory=dict)
    campaigns: List[CampaignConfig] = Field(default_factory=list)
    campaign: Optional[CampaignMetadataConfig] = None
    io: IOConfig = IOConfig()
    motif_store: MotifStoreConfig = MotifStoreConfig()
    ingest: IngestConfig = IngestConfig()

    parse: ParseConfig
    sample: Optional[SampleConfig] = None
    analysis: Optional[AnalysisConfig] = None

    @field_validator("regulator_categories")
    @classmethod
    def _check_regulator_categories(cls, v: Dict[str, List[str]]) -> Dict[str, List[str]]:
        cleaned: dict[str, list[str]] = {}
        for raw_name, raw_tfs in v.items():
            name = str(raw_name).strip()
            if not name:
                raise ValueError("regulator_categories keys must be non-empty strings")
            if not raw_tfs:
                raise ValueError(f"regulator_categories['{name}'] must be a non-empty list")
            tfs: list[str] = []
            for tf in raw_tfs:
                tf_name = str(tf).strip()
                if not tf_name:
                    raise ValueError(f"regulator_categories['{name}'] entries must be non-empty strings")
                tfs.append(tf_name)
            if len(set(tfs)) != len(tfs):
                raise ValueError(f"regulator_categories['{name}'] contains duplicate TF names")
            cleaned[name] = tfs
        return cleaned

    @model_validator(mode="after")
    def _check_campaigns(self) -> "CruncherConfig":
        if not self.campaigns:
            return self
        if not self.regulator_categories:
            raise ValueError("campaigns require regulator_categories to be defined")
        seen: set[str] = set()
        category_names = set(self.regulator_categories.keys())
        for campaign in self.campaigns:
            if campaign.name in seen:
                raise ValueError(f"campaign name '{campaign.name}' is duplicated")
            seen.add(campaign.name)
            missing = [name for name in campaign.categories if name not in category_names]
            if missing:
                missing_list = ", ".join(missing)
                raise ValueError(f"campaign '{campaign.name}' references missing categories: {missing_list}")
            if not campaign.allow_overlap:
                overlaps: set[str] = set()
                seen_tfs: set[str] = set()
                for category in campaign.categories:
                    for tf in self.regulator_categories.get(category, []):
                        if tf in seen_tfs:
                            overlaps.add(tf)
                        seen_tfs.add(tf)
                if overlaps:
                    overlaps_list = ", ".join(sorted(overlaps))
                    raise ValueError(
                        f"campaign '{campaign.name}' forbids overlaps, but TFs appear in multiple categories: "
                        f"{overlaps_list}"
                    )
        return self


class CruncherRoot(BaseModel):
    cruncher: CruncherConfig
