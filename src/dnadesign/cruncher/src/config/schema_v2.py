"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/config/schema_v2.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


logger = logging.getLogger(__name__)


class PlotConfig(StrictBaseModel):
    logo: bool
    bits_mode: Literal["information", "probability"]
    dpi: int


class ParseConfig(StrictBaseModel):
    plot: PlotConfig


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
                raise ValueError(f"{label}['{k}'] must be ≥ 0, but got {fv}")
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


class CoolingStage(StrictBaseModel):
    sweeps: int
    beta: float

    @field_validator("sweeps")
    @classmethod
    def _check_sweeps(cls, v: int) -> int:
        if not isinstance(v, int) or v <= 0:
            raise ValueError("softmin.stages.sweeps must be a positive integer")
        return v

    @field_validator("beta")
    @classmethod
    def _check_beta(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError("softmin.stages.beta must be > 0")
        return float(v)


class SoftminConfig(StrictBaseModel):
    enabled: bool = True
    kind: Literal["fixed", "linear", "piecewise"] = "linear"
    beta: Optional[Union[float, Tuple[float, float]]] = None
    stages: Optional[List[CoolingStage]] = None

    @model_validator(mode="before")
    @classmethod
    def _set_defaults(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        kind = data.get("kind", "linear")
        if "beta" not in data and "stages" not in data:
            if kind == "fixed":
                data["beta"] = 1.0
            elif kind == "linear":
                data["beta"] = (0.5, 10.0)
        return data

    @model_validator(mode="after")
    def _validate_softmin(self) -> "SoftminConfig":
        if self.kind == "fixed":
            if self.stages is not None:
                raise ValueError("softmin.stages is not allowed for kind='fixed'")
            if not isinstance(self.beta, (int, float)):
                raise ValueError("softmin.beta must be a positive float for kind='fixed'")
            if float(self.beta) <= 0:
                raise ValueError("softmin.beta must be > 0")
            self.beta = float(self.beta)
            return self
        if self.kind == "linear":
            if self.stages is not None:
                raise ValueError("softmin.stages is not allowed for kind='linear'")
            if not isinstance(self.beta, (list, tuple)) or len(self.beta) != 2:
                raise ValueError("softmin.beta must be length-2 [beta_start, beta_end] for kind='linear'")
            b0, b1 = float(self.beta[0]), float(self.beta[1])
            if b0 <= 0 or b1 <= 0:
                raise ValueError("softmin beta values must be > 0")
            self.beta = (b0, b1)
            return self
        if self.kind == "piecewise":
            if self.beta is not None:
                raise ValueError("softmin.beta is not allowed for kind='piecewise'")
            if not self.stages:
                raise ValueError("softmin.stages must be provided for kind='piecewise'")
            return self
        raise ValueError(f"Unknown softmin.kind '{self.kind}'")


class AdaptiveBetaConfig(StrictBaseModel):
    enabled: bool = False
    target_acceptance: float = 0.40
    window: int = 100
    k: float = 0.50
    min_beta: float = 1.0e-3
    max_beta: float = 100.0
    moves: List[Literal["S", "B", "M", "L", "W", "I"]] = Field(default_factory=lambda: ["B", "M"])
    stop_after_tune: bool = True

    @field_validator("target_acceptance")
    @classmethod
    def _check_target_acceptance(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or v <= 0 or v >= 1:
            raise ValueError("adaptive_beta.target_acceptance must be between 0 and 1")
        return float(v)

    @field_validator("window")
    @classmethod
    def _check_window(cls, v: int) -> int:
        if not isinstance(v, int) or v < 1:
            raise ValueError("adaptive_beta.window must be >= 1")
        return v

    @field_validator("k")
    @classmethod
    def _check_k(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError("adaptive_beta.k must be > 0")
        return float(v)

    @model_validator(mode="after")
    def _check_beta_bounds(self) -> "AdaptiveBetaConfig":
        if self.min_beta <= 0 or self.max_beta <= 0:
            raise ValueError("adaptive_beta min/max beta must be > 0")
        if self.min_beta > self.max_beta:
            raise ValueError("adaptive_beta.min_beta must be <= adaptive_beta.max_beta")
        return self


class AdaptiveSwapConfig(StrictBaseModel):
    enabled: bool = False
    target_swap: float = 0.25
    window: int = 50
    k: float = 0.50
    min_scale: float = 0.25
    max_scale: float = 50.0
    stop_after_tune: bool = True

    @field_validator("target_swap")
    @classmethod
    def _check_target_swap(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or v <= 0 or v >= 1:
            raise ValueError("adaptive_swap.target_swap must be between 0 and 1")
        return float(v)

    @field_validator("window")
    @classmethod
    def _check_swap_window(cls, v: int) -> int:
        if not isinstance(v, int) or v < 1:
            raise ValueError("adaptive_swap.window must be >= 1")
        return v

    @field_validator("k")
    @classmethod
    def _check_swap_k(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError("adaptive_swap.k must be > 0")
        return float(v)

    @model_validator(mode="after")
    def _check_swap_bounds(self) -> "AdaptiveSwapConfig":
        if self.min_scale <= 0 or self.max_scale <= 0:
            raise ValueError("adaptive_swap min/max scale must be > 0")
        if self.min_scale > self.max_scale:
            raise ValueError("adaptive_swap.min_scale must be <= adaptive_swap.max_scale")
        return self


class BetaLadderFixed(StrictBaseModel):
    kind: Literal["fixed"] = "fixed"
    beta: float = 1.0

    @field_validator("beta")
    @classmethod
    def _check_beta(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError("beta_ladder.beta must be > 0")
        return float(v)


class BetaLadderGeometric(StrictBaseModel):
    kind: Literal["geometric"] = "geometric"
    betas: Optional[List[float]] = None
    beta_min: Optional[float] = None
    beta_max: Optional[float] = None
    n_temps: Optional[int] = None

    @field_validator("betas")
    @classmethod
    def _check_betas(cls, v: Optional[List[float]]) -> Optional[List[float]]:
        if v is None:
            return v
        if len(v) < 2:
            raise ValueError("beta_ladder.betas must contain at least 2 values")
        if any(beta <= 0 for beta in v):
            raise ValueError("beta_ladder.betas values must be > 0")
        return v

    @field_validator("beta_min", "beta_max")
    @classmethod
    def _check_beta_bounds(cls, v: Optional[float], info) -> Optional[float]:
        if v is None:
            return v
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError(f"beta_ladder.{info.field_name} must be > 0")
        return float(v)

    @field_validator("n_temps")
    @classmethod
    def _check_n_temps(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return v
        if not isinstance(v, int) or v < 2:
            raise ValueError("beta_ladder.n_temps must be >= 2")
        return v

    @model_validator(mode="after")
    def _validate_inputs(self) -> "BetaLadderGeometric":
        if self.betas is not None:
            if self.beta_min is not None or self.beta_max is not None or self.n_temps is not None:
                raise ValueError("beta_ladder: provide either betas or (beta_min,beta_max,n_temps), not both.")
            return self
        if self.beta_min is None or self.beta_max is None or self.n_temps is None:
            raise ValueError("beta_ladder requires betas or beta_min/beta_max/n_temps")
        if self.beta_min > self.beta_max:
            raise ValueError("beta_ladder.beta_min must be <= beta_ladder.beta_max")
        return self


BetaLadderConfig = Union[BetaLadderFixed, BetaLadderGeometric]


class PTOptimizerConfig(StrictBaseModel):
    beta_ladder: BetaLadderConfig = Field(default_factory=lambda: BetaLadderGeometric(betas=[0.2, 1.0, 5.0, 25.0]))
    swap_prob: float = 0.10
    ladder_adapt: AdaptiveSwapConfig = AdaptiveSwapConfig()

    @field_validator("swap_prob")
    @classmethod
    def _check_swap_prob(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or v < 0 or v > 1:
            raise ValueError("optimizers.pt.swap_prob must be between 0 and 1")
        return float(v)


class OptimizersConfig(StrictBaseModel):
    pt: PTOptimizerConfig = PTOptimizerConfig()


class OptimizerSelectionConfig(StrictBaseModel):
    name: Literal["auto", "pt"] = "auto"


class AutoOptScorecardConfig(StrictBaseModel):
    k: int = 20
    alpha: float = 0.85

    @field_validator("alpha")
    @classmethod
    def _check_alpha(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or v <= 0 or v > 1:
            raise ValueError("auto_opt.scorecard.alpha must be in (0, 1]")
        return float(v)

    @field_validator("k")
    @classmethod
    def _check_k(cls, v: int) -> int:
        if not isinstance(v, int) or v < 1:
            raise ValueError("auto_opt.scorecard.k must be >= 1")
        return v


class AutoOptPolicyConfig(StrictBaseModel):
    allow_warn: bool = False
    scorecard: AutoOptScorecardConfig = Field(default_factory=AutoOptScorecardConfig)
    diversity_weight: float = 0.25

    @field_validator("diversity_weight")
    @classmethod
    def _check_diversity_weight(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or v < 0:
            raise ValueError("auto_opt.policy.diversity_weight must be >= 0")
        return float(v)


class AutoOptConfig(StrictBaseModel):
    enabled: bool = True
    budget_levels: List[int] = Field(default_factory=lambda: [200, 800])
    replicates: int = 1
    keep_pilots: Literal["all", "ok", "best"] = "ok"
    cooling_boosts: List[float] = Field(default_factory=lambda: [1.0, 2.0])
    pt_swap_probs: List[float] = Field(default_factory=list)
    move_profiles: List[Literal["balanced", "local", "global", "aggressive"]] = Field(
        default_factory=lambda: ["balanced", "aggressive"]
    )
    policy: AutoOptPolicyConfig = AutoOptPolicyConfig()

    @field_validator("budget_levels")
    @classmethod
    def _check_budget_levels(cls, v: List[int]) -> List[int]:
        if not isinstance(v, list) or not v:
            raise ValueError("auto_opt.budget_levels must be a non-empty list of integers")
        cleaned: list[int] = []
        for item in v:
            if not isinstance(item, int) or item < 4:
                raise ValueError("auto_opt.budget_levels entries must be >= 4")
            cleaned.append(item)
        return cleaned

    @field_validator("cooling_boosts")
    @classmethod
    def _check_cooling_boosts(cls, v: List[float]) -> List[float]:
        if not isinstance(v, list) or not v:
            raise ValueError("auto_opt.cooling_boosts must be a non-empty list of numbers")
        cleaned: list[float] = []
        for item in v:
            if not isinstance(item, (int, float)) or float(item) <= 0:
                raise ValueError("auto_opt.cooling_boosts entries must be > 0")
            cleaned.append(float(item))
        return cleaned

    @field_validator("pt_swap_probs")
    @classmethod
    def _check_pt_swap_probs(cls, v: List[float]) -> List[float]:
        if not isinstance(v, list):
            raise ValueError("auto_opt.pt_swap_probs must be a list of numbers")
        cleaned: list[float] = []
        for item in v:
            if not isinstance(item, (int, float)) or float(item) < 0 or float(item) > 1:
                raise ValueError("auto_opt.pt_swap_probs entries must be between 0 and 1")
            cleaned.append(float(item))
        return cleaned

    @field_validator("move_profiles")
    @classmethod
    def _check_move_profiles(cls, v: List[str]) -> List[str]:
        if not isinstance(v, list) or not v:
            raise ValueError("auto_opt.move_profiles must be a non-empty list")
        return v

    @field_validator("replicates")
    @classmethod
    def _check_positive_ints(cls, v: int, info) -> int:
        if not isinstance(v, int) or v < 1:
            raise ValueError(f"auto_opt.{info.field_name} must be >= 1")
        return v


class InitConfig(StrictBaseModel):
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


class ScoringConfig(StrictBaseModel):
    pwm_pseudocounts: float = 0.10
    log_odds_clip: Optional[float] = None

    @field_validator("pwm_pseudocounts")
    @classmethod
    def _check_pwm_pseudocounts(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or v < 0:
            raise ValueError("scoring.pwm_pseudocounts must be >= 0")
        return float(v)

    @field_validator("log_odds_clip")
    @classmethod
    def _check_log_odds_clip(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return v
        if not isinstance(v, (int, float)) or v <= 0:
            raise ValueError("scoring.log_odds_clip must be a positive number or null")
        return float(v)


class SampleRngConfig(StrictBaseModel):
    seed: int = Field(42, description="Random seed for reproducible sampling.")
    deterministic: bool = Field(
        True,
        description="If true, auto-opt pilots derive deterministic seeds from config + locks.",
    )

    @field_validator("seed")
    @classmethod
    def _check_seed(cls, v: int) -> int:
        if not isinstance(v, int) or v < 0:
            raise ValueError("sample.rng.seed must be a non-negative integer")
        return v


class SampleBudgetConfig(StrictBaseModel):
    tune: int
    draws: int
    restarts: int = 1

    @field_validator("tune", "draws")
    @classmethod
    def _check_non_negative(cls, v: int, info) -> int:
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"sample.budget.{info.field_name} must be a non-negative integer")
        return v

    @field_validator("restarts")
    @classmethod
    def _check_restarts(cls, v: int) -> int:
        if not isinstance(v, int) or v < 1:
            raise ValueError("sample.budget.restarts must be >= 1")
        return v


class SampleEarlyStopConfig(StrictBaseModel):
    enabled: bool = True
    patience: int = 500
    min_delta: float = 0.5
    require_min_unique: bool = False
    min_unique: int = 20
    success_min_per_tf_norm: float = 0.80

    @field_validator("patience")
    @classmethod
    def _check_patience(cls, v: int) -> int:
        if not isinstance(v, int) or v < 0:
            raise ValueError("sample.early_stop.patience must be a non-negative integer")
        return v

    @field_validator("min_delta")
    @classmethod
    def _check_min_delta(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or v < 0:
            raise ValueError("sample.early_stop.min_delta must be >= 0")
        return float(v)

    @field_validator("min_unique")
    @classmethod
    def _check_min_unique(cls, v: int) -> int:
        if not isinstance(v, int) or v < 0:
            raise ValueError("sample.early_stop.min_unique must be >= 0")
        return v

    @field_validator("success_min_per_tf_norm")
    @classmethod
    def _check_success_min_per_tf_norm(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or v < 0 or v > 1:
            raise ValueError("sample.early_stop.success_min_per_tf_norm must be between 0 and 1")
        return float(v)

    @model_validator(mode="after")
    def _check_require_min_unique(self) -> "SampleEarlyStopConfig":
        if self.require_min_unique and self.min_unique < 1:
            raise ValueError("sample.early_stop.min_unique must be >= 1 when require_min_unique=true")
        return self


class SampleObjectiveConfig(StrictBaseModel):
    bidirectional: bool = True
    score_scale: Literal["llr", "z", "logp", "consensus-neglop-sum", "normalized-llr"] = "llr"
    combine: Literal["min", "sum"] | None = Field(
        None,
        description="How to combine per-TF scores. Defaults to min (or sum for consensus-neglop-sum).",
    )
    scoring: ScoringConfig = ScoringConfig()
    softmin: SoftminConfig = SoftminConfig()
    length_penalty_lambda: float = 0.0
    allow_unscaled_llr: bool = False

    @field_validator("length_penalty_lambda")
    @classmethod
    def _check_length_penalty_lambda(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or v < 0:
            raise ValueError("objective.length_penalty_lambda must be >= 0")
        return float(v)


class EliteFiltersConfig(StrictBaseModel):
    pwm_sum_min: float = 0.0
    min_per_tf_norm: float | None = None
    require_all_tfs_over_min_norm: bool = True

    @field_validator("pwm_sum_min")
    @classmethod
    def _check_pwm_sum_min(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or v < 0:
            raise ValueError("sample.elites.filters.pwm_sum_min must be >= 0")
        return float(v)

    @field_validator("min_per_tf_norm")
    @classmethod
    def _check_min_per_tf_norm(cls, v: float | None) -> float | None:
        if v is None:
            return v
        if not isinstance(v, (int, float)) or v < 0:
            raise ValueError("sample.elites.filters.min_per_tf_norm must be >= 0")
        return float(v)


class SampleElitesSelectionConfig(StrictBaseModel):
    policy: Literal["mmr"] = "mmr"
    pool_size: int = 1000
    alpha: float = 0.85
    relevance: Literal["min_per_tf_norm", "combined_score_final"] = "min_per_tf_norm"
    min_distance: float | None = None

    @field_validator("pool_size")
    @classmethod
    def _check_pool_size(cls, v: int) -> int:
        if not isinstance(v, int) or v < 1:
            raise ValueError("sample.elites.selection.pool_size must be >= 1")
        return v

    @field_validator("alpha")
    @classmethod
    def _check_alpha(cls, v: float) -> float:
        if not isinstance(v, (int, float)) or v <= 0 or v > 1:
            raise ValueError("sample.elites.selection.alpha must be in (0, 1]")
        return float(v)

    @field_validator("min_distance")
    @classmethod
    def _check_min_distance(cls, v: float | None) -> float | None:
        if v is None:
            return v
        if not isinstance(v, (int, float)) or v < 0 or v > 1:
            raise ValueError("sample.elites.selection.min_distance must be between 0 and 1 or null")
        return float(v)


class SampleElitesConfig(StrictBaseModel):
    k: int = 10
    filters: EliteFiltersConfig = EliteFiltersConfig()
    selection: SampleElitesSelectionConfig = SampleElitesSelectionConfig()

    @field_validator("k")
    @classmethod
    def _check_elite_ints(cls, v: int, info) -> int:
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"sample.elites.{info.field_name} must be a non-negative integer")
        return v

    @model_validator(mode="after")
    def _warn_pool_size(self) -> "SampleElitesConfig":
        if self.selection.policy == "mmr" and self.selection.pool_size < self.k:
            logger.warning(
                "sample.elites.selection.pool_size=%d < sample.elites.k=%d; MMR pool will be clamped.",
                self.selection.pool_size,
                self.k,
            )
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
        if v is None:
            return v
        if isinstance(v, (list, tuple)):
            return tuple(int(x) for x in v)
        return v

    @field_validator("move_probs")
    @classmethod
    def _check_move_probs(cls, v):
        if v is None:
            return v
        return MoveConfig._normalize_move_probs(v, label="moves.overrides.move_probs")

    @field_validator("slide_max_shift")
    @classmethod
    def _check_slide_max_shift(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return v
        if not isinstance(v, int) or v < 0:
            raise ValueError("moves.overrides.slide_max_shift must be a non-negative integer")
        return v

    @field_validator("target_worst_tf_prob")
    @classmethod
    def _check_target_worst_tf_prob(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return v
        if not isinstance(v, (int, float)) or v < 0 or v > 1:
            raise ValueError("moves.overrides.target_worst_tf_prob must be between 0 and 1")
        return float(v)

    @field_validator("target_window_pad")
    @classmethod
    def _check_target_window_pad(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return v
        if not isinstance(v, int) or v < 0:
            raise ValueError("moves.overrides.target_window_pad must be a non-negative integer")
        return v

    @field_validator("insertion_consensus_prob")
    @classmethod
    def _check_insertion_consensus_prob(cls, v: Optional[float]) -> Optional[float]:
        if v is None:
            return v
        if not isinstance(v, (int, float)) or v < 0 or v > 1:
            raise ValueError("moves.overrides.insertion_consensus_prob must be between 0 and 1")
        return float(v)


class SampleMovesConfig(StrictBaseModel):
    profile: Literal["balanced", "local", "global", "aggressive"] = "balanced"
    overrides: MoveOverridesConfig = MoveOverridesConfig()


class SampleOutputTraceConfig(StrictBaseModel):
    save: bool = Field(True, description="Write trace.nc for analyze/report.")
    include_tune: bool = Field(
        False,
        description="Include tune phase samples in sequences.parquet (trace.nc is draws-only).",
    )


class SampleOutputConfig(StrictBaseModel):
    save_sequences: bool = True
    include_consensus_in_elites: bool = False
    live_metrics: bool = True
    trace: SampleOutputTraceConfig = SampleOutputTraceConfig()


class SampleUiConfig(StrictBaseModel):
    progress_bar: bool = True
    progress_every: int = 0

    @field_validator("progress_every")
    @classmethod
    def _check_progress_every(cls, v: int) -> int:
        if not isinstance(v, int) or v < 0:
            raise ValueError("sample.ui.progress_every must be a non-negative integer")
        return v


class SampleConfig(StrictBaseModel):
    mode: Literal["optimize", "sample"] = "optimize"
    rng: SampleRngConfig = SampleRngConfig()
    budget: SampleBudgetConfig
    early_stop: SampleEarlyStopConfig = SampleEarlyStopConfig()
    init: InitConfig
    objective: SampleObjectiveConfig = SampleObjectiveConfig()
    elites: SampleElitesConfig = SampleElitesConfig()
    moves: SampleMovesConfig = SampleMovesConfig()
    optimizer: OptimizerSelectionConfig = OptimizerSelectionConfig()
    optimizers: OptimizersConfig = OptimizersConfig()
    auto_opt: AutoOptConfig | None = AutoOptConfig()
    output: SampleOutputConfig = SampleOutputConfig()
    ui: SampleUiConfig = SampleUiConfig()

    @model_validator(mode="after")
    def _validate_optimizer_settings(self) -> "SampleConfig":
        if self.optimizer.name == "auto":
            if self.auto_opt is None or not self.auto_opt.enabled:
                raise ValueError("sample.optimizer.name='auto' requires auto_opt.enabled=true")
        else:
            if self.auto_opt is not None and self.auto_opt.enabled:
                raise ValueError("auto_opt.enabled must be false when optimizer.name is not 'auto'")

        if self.optimizer.name == "pt" and self.budget.restarts != 1:
            raise ValueError("PT does not support budget.restarts > 1; set sample.budget.restarts=1.")
        if self.early_stop.enabled and self.objective.score_scale == "normalized-llr":
            if self.early_stop.min_delta > 0.1:
                raise ValueError(
                    "sample.early_stop.min_delta must be <= 0.1 when objective.score_scale='normalized-llr'."
                )
        return self


class AnalysisPlotConfig(StrictBaseModel):
    dashboard: bool = True
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
    worst_tf_trace: bool = True
    worst_tf_identity: bool = True
    elite_filter_waterfall: bool = True
    overlap_heatmap: bool = True
    overlap_bp_distribution: bool = True
    overlap_strand_combos: bool = False
    motif_offset_rug: bool = False
    pt_swap_by_pair: bool = False
    move_acceptance_time: bool = False
    move_usage_time: bool = False


class AnalysisConfig(StrictBaseModel):
    runs: Optional[List[str]]
    extra_plots: bool = False
    extra_tables: bool = False
    mcmc_diagnostics: bool = False
    dashboard_only: bool = True
    plots: AnalysisPlotConfig = AnalysisPlotConfig()
    table_format: Literal["parquet", "csv"] = "parquet"
    plot_format: Literal["png", "pdf", "svg"] = "png"
    scatter_scale: Literal["llr", "z", "logp", "consensus-neglop-sum", "normalized-llr"]
    subsampling_epsilon: float
    plot_dpi: int = 150
    png_compress_level: int = 9
    scatter_style: Literal["edges", "thresholds"] = "edges"
    scatter_background: bool = True
    scatter_background_samples: Optional[int] = None
    scatter_background_seed: int = 0
    tf_pair: Optional[List[str]] = None
    include_sequences_in_tables: bool = False
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

    @field_validator("plot_dpi")
    @classmethod
    def _check_plot_dpi(cls, v: int) -> int:
        if not isinstance(v, int) or v <= 0:
            raise ValueError("analysis.plot_dpi must be a positive integer")
        return v

    @field_validator("png_compress_level")
    @classmethod
    def _check_png_compress_level(cls, v: int) -> int:
        if not isinstance(v, int) or v < 0 or v > 9:
            raise ValueError("analysis.png_compress_level must be an integer in [0, 9]")
        return v

    @field_validator("tf_pair")
    @classmethod
    def _check_tf_pair(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return None
        if len(v) != 2:
            raise ValueError("analysis.tf_pair must contain exactly two TF names.")
        return v

    @field_validator("scatter_background_samples")
    @classmethod
    def _check_scatter_background_samples(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        if not isinstance(v, int) or v < 0:
            raise ValueError("analysis.scatter_background_samples must be a non-negative integer or null.")
        return v

    @field_validator("scatter_background_seed")
    @classmethod
    def _check_scatter_background_seed(cls, v: int) -> int:
        if not isinstance(v, int) or v < 0:
            raise ValueError("analysis.scatter_background_seed must be a non-negative integer")
        return v

    @model_validator(mode="after")
    def _check_scatter_style(self) -> "AnalysisConfig":
        if self.scatter_style == "thresholds" and self.scatter_scale != "llr":
            raise ValueError("scatter_style='thresholds' requires scatter_scale='llr'")
        return self


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


class MotifStoreConfig(StrictBaseModel):
    catalog_root: Path = Path(".cruncher")
    source_preference: List[str] = Field(default_factory=list)
    allow_ambiguous: bool = False
    pwm_source: Literal["matrix", "sites"] = "matrix"
    site_kinds: Optional[List[str]] = None
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
    site_window_lengths: Dict[str, int] = Field(
        default_factory=dict,
        description="Window length overrides for HT sites keyed by TF name or dataset:<id>.",
    )
    site_window_center: Literal["midpoint", "summit"] = "midpoint"
    pwm_window_lengths: Dict[str, int] = Field(
        default_factory=dict,
        description="Window length overrides for PWM trimming keyed by TF name or dataset:<id>.",
    )
    pwm_window_strategy: Literal["max_info"] = "max_info"
    min_sites_for_pwm: int = 2
    allow_low_sites: bool = False

    @field_validator("catalog_root")
    @classmethod
    def _check_catalog_root(cls, v: Path) -> Path:
        normalized = Path(v)
        if not str(normalized).strip():
            raise ValueError("motif_store.catalog_root must be a non-empty path")
        if not normalized.is_absolute() and any(part == ".." for part in normalized.parts):
            raise ValueError("motif_store.catalog_root must not traverse outside the cruncher root")
        return normalized

    @field_validator("site_window_lengths")
    @classmethod
    def _check_window_lengths(cls, v: Dict[str, int]) -> Dict[str, int]:
        for key, value in v.items():
            if value <= 0:
                raise ValueError(f"site_window_lengths['{key}'] must be > 0")
        return v

    @field_validator("pwm_window_lengths")
    @classmethod
    def _check_pwm_window_lengths(cls, v: Dict[str, int]) -> Dict[str, int]:
        for key, value in v.items():
            if value <= 0:
                raise ValueError(f"pwm_window_lengths['{key}'] must be > 0")
        return v

    @field_validator("pseudocounts")
    @classmethod
    def _check_pseudocounts(cls, v: float) -> float:
        if v < 0:
            raise ValueError("pseudocounts must be >= 0")
        return float(v)

    @field_validator("min_sites_for_pwm")
    @classmethod
    def _check_min_sites(cls, v: int) -> int:
        if v < 1:
            raise ValueError("min_sites_for_pwm must be >= 1")
        return v


class MotifDiscoveryConfig(StrictBaseModel):
    tool: Literal["auto", "streme", "meme"] = Field(
        "auto",
        description="Motif discovery tool: auto selects STREME for larger sets, MEME for small sets.",
    )
    tool_path: Optional[Path] = Field(
        None,
        description="Optional path to MEME Suite executable or bin directory (overrides PATH).",
    )
    window_sites: bool = Field(
        False,
        description="Pre-window binding sites using motif_store.site_window_lengths before discovery.",
    )
    minw: Optional[int] = Field(
        None,
        description="Minimum motif width for discovery (auto from site lengths when unset).",
    )
    maxw: Optional[int] = Field(
        None,
        description="Maximum motif width for discovery (auto from site lengths when unset).",
    )
    nmotifs: int = Field(1, description="Number of motifs to report per TF.")
    meme_mod: Optional[Literal["oops", "zoops", "anr"]] = Field(
        None,
        description="Optional MEME -mod setting (oops/zoops/anr). When unset, MEME defaults apply.",
    )
    meme_prior: Optional[Literal["dirichlet", "dmix", "mega", "megap", "addone"]] = Field(
        None,
        description="Optional MEME -prior setting (dirichlet/dmix/mega/megap/addone).",
    )
    min_sequences_for_streme: int = Field(50, description="Threshold for auto tool selection.")
    source_id: str = Field("meme_suite", description="Catalog source_id for discovered motifs.")
    replace_existing: bool = Field(
        True,
        description="Replace existing discovered motifs for the same TF/source to avoid cache bloat.",
    )

    @field_validator("minw", "maxw")
    @classmethod
    def _check_optional_positive_ints(cls, v: Optional[int], info) -> Optional[int]:
        if v is None:
            return v
        if v < 1:
            raise ValueError(f"{info.field_name} must be >= 1")
        return int(v)

    @field_validator("nmotifs", "min_sequences_for_streme")
    @classmethod
    def _check_positive_ints(cls, v: int, info) -> int:
        if v < 1:
            raise ValueError(f"{info.field_name} must be >= 1")
        return v

    @model_validator(mode="after")
    def _check_widths(self) -> "MotifDiscoveryConfig":
        if self.minw is not None and self.maxw is not None and self.maxw < self.minw:
            raise ValueError("motif_discovery.maxw must be >= motif_discovery.minw")
        return self


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
            raise ValueError("backoff seconds must be >= 0")
        return float(v)


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
            raise ValueError("ncbi_timeout_seconds must be > 0")
        return v


class CruncherConfig(StrictBaseModel):
    out_dir: Path
    regulator_sets: List[List[str]]
    regulator_categories: Dict[str, List[str]] = Field(default_factory=dict)
    campaigns: List[CampaignConfig] = Field(default_factory=list)
    campaign: Optional[CampaignMetadataConfig] = None
    io: IOConfig = IOConfig()
    motif_store: MotifStoreConfig = MotifStoreConfig()
    motif_discovery: MotifDiscoveryConfig = MotifDiscoveryConfig()
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

    @field_validator("out_dir")
    @classmethod
    def _check_out_dir(cls, v: Path) -> Path:
        if v.is_absolute():
            raise ValueError("out_dir must be a relative path")
        normalized = Path(v)
        if any(part == ".." for part in normalized.parts):
            raise ValueError("out_dir must not traverse outside the workspace")
        return normalized

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

    @model_validator(mode="after")
    def _check_score_scale_llr(self) -> "CruncherConfig":
        if self.sample is None:
            return self
        objective = self.sample.objective
        if objective.score_scale == "llr" and not objective.allow_unscaled_llr:
            if any(len(regs) > 1 for regs in self.regulator_sets):
                raise ValueError(
                    "score_scale='llr' is not comparable across PWMs in multi-TF runs. "
                    "Use a normalized scale (e.g. normalized-llr or logp) or set "
                    "sample.objective.allow_unscaled_llr=true to proceed."
                )
        return self


class CruncherRoot(StrictBaseModel):
    cruncher: CruncherConfig
