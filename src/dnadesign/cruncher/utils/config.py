"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/cruncher/utils/config.py

Central config-loading and validation via Pydantic.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import yaml
from pydantic import BaseModel, Field, root_validator, validator


# PARSE MODE SECTION
class PlotConfig(BaseModel):
    """
    Settings needed to draw PWM logos (parse mode).
    """

    logo: bool
    bits_mode: Literal["information", "probability"]
    dpi: int


class ParseConfig(BaseModel):
    """
    Settings needed to load and parse PWMs.
    """

    formats: Dict[str, str]  # e.g. {".txt": "MEME", ".pfm": "JASPAR"}
    plot: PlotConfig


# SAMPLE MODE SECTION
class MoveConfig(BaseModel):
    """
    Shared move-kernel parameters for MCMC.

    In addition to block_len_range / multi_k_range / slide_max_shift / swap_len_range,
    we now allow the user to specify the relative probability of choosing each move kind:
      • "S" = single-nucleotide flip
      • "B" = contiguous block replacement
      • "M" = k disjoint flips
    """

    block_len_range: Tuple[int, int] = (3, 12)
    multi_k_range: Tuple[int, int] = (2, 6)
    slide_max_shift: int = 2
    swap_len_range: Tuple[int, int] = (6, 12)

    # Probability mass for each move kind (must sum to 1.0)
    move_probs: Dict[Literal["S", "B", "M"], float] = {
        "S": 0.50,
        "B": 0.30,
        "M": 0.20,
    }

    @validator("block_len_range", "multi_k_range", "swap_len_range", pre=True)
    def _list_to_tuple(cls, v):
        if isinstance(v, (list, tuple)):
            return tuple(int(x) for x in v)
        return v

    @validator("move_probs", pre=True)
    def _check_move_probs_keys_and_values(cls, v):
        """
        Ensure that:
          • Exactly keys "S","B","M" are present
          • Each value is a float ≥ 0
          • The three values sum to (approximately) 1.0
        """
        if not isinstance(v, dict):
            raise ValueError("move_probs must be a mapping {S: float, B: float, M: float}")

        # Check exactly the expected keys
        expected_keys = {"S", "B", "M"}
        got_keys = set(v.keys())
        if got_keys != expected_keys:
            raise ValueError(f"move_probs keys must be exactly {expected_keys}, but got {got_keys}")

        # Check each value is non‐negative float
        for k, val in v.items():
            try:
                fv = float(val)
            except (TypeError, ValueError):
                raise ValueError(f"move_probs['{k}'] must be a float, got {val!r}")
            if fv < 0:
                raise ValueError(f"move_probs['{k}'] must be ≥ 0, but got {fv}")
        # Check they sum to 1.0 (allow slight numerical tolerance)
        total = sum(float(v[k]) for k in expected_keys)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"move_probs values must sum to 1.0; got sum={total:.6f}")

        # Return as floats
        return {k: float(v[k]) for k in expected_keys}


class CoolingFixed(BaseModel):
    """
    A single-value fixed β (e.g., always β = 1.0).
    """

    kind: Literal["fixed"] = "fixed"
    beta: float = 1.0

    @validator("beta")
    def _check_positive_beta(cls, v):
        if v <= 0:
            raise ValueError("Fixed cooling beta must be > 0")
        return v


class CoolingLinear(BaseModel):
    """
    Linear ramp from β_start → β_end over the entire run (tune + draws).
    """

    kind: Literal["linear"] = "linear"
    beta: Tuple[float, float]

    @validator("beta")
    def _two_positive(cls, v):
        if len(v) != 2:
            raise ValueError("Linear cooling.beta must be length-2 [beta_start, beta_end]")
        if v[0] <= 0 or v[1] <= 0:
            raise ValueError("Both β_start and β_end must be > 0")
        return v


class CoolingGeometric(BaseModel):
    """
    Explicit list of β values (a “ladder”) used in Parallel Tempering.
    """

    kind: Literal["geometric"] = "geometric"
    beta: List[float]

    @validator("beta")
    def _check_list_positive(cls, v):
        if not isinstance(v, list) or len(v) < 2:
            raise ValueError("Geometric cooling.beta must be a list of at least two positive floats")
        if any(x <= 0 for x in v):
            raise ValueError("All entries in geometric β list must be > 0")
        return v


CoolingConfig = Union[CoolingFixed, CoolingLinear, CoolingGeometric]


class OptimiserConfig(BaseModel):
    """
    Common “optimiser” block for both Gibbs and PT.

    - kind: “gibbs” or “pt”
    - scorer_scale: “llr” | “z” | “p” | “logp” | “logp_norm” | “consensus-neglop-sum”
    - cooling: one of fixed | linear | geometric
    - swap_prob: intra-chain swap (Gibbs) or inter-chain swap (PT)
    - softmax_beta: only required if kind == “pt”
    """

    kind: Literal["gibbs", "pt"]
    scorer_scale: Literal["llr", "z", "p", "logp", "logp_norm", "consensus-neglop-sum"]
    cooling: CoolingConfig
    swap_prob: float = 0.10
    softmax_beta: Optional[float] = None

    @root_validator
    def _check_pt_needs_softmax(cls, values):
        kind = values.get("kind")
        sb = values.get("softmax_beta")
        if kind == "pt":
            if sb is None:
                raise ValueError("softmax_beta must be supplied when optimiser.kind == 'pt'")
            if sb <= 0:
                raise ValueError("softmax_beta must be > 0")
        return values


class InitConfig(BaseModel):
    """
    Settings for sequence initialization in sample mode.
    """

    kind: Literal["random", "consensus", "consensus_mix"]
    length: int
    regulator: Optional[str] = None
    pad_with: Optional[Literal["background", "A", "C", "G", "T"]] = "background"

    @validator("length")
    def _check_length_positive(cls, v):
        if v < 1:
            raise ValueError("init.length must be >= 1")
        return v

    @root_validator
    def _check_fields_for_modes(cls, values):
        kind = values.get("kind")
        regulator = values.get("regulator")
        if kind == "consensus" and not regulator:
            raise ValueError("When init.kind=='consensus', you must supply init.regulator=<PWM_name>")
        return values


class SampleConfig(BaseModel):
    """
    Top-level “sample” section: MCMC settings.
    """

    bidirectional: bool = True
    penalties: Dict[str, float] = {}

    init: InitConfig
    draws: int
    tune: int
    chains: int
    min_dist: int
    top_k: int

    moves: MoveConfig = MoveConfig()
    optimiser: OptimiserConfig
    save_sequences: bool = True

    pwm_sum_threshold: float = Field(
        0.0, description="If >0, only sequences with sum(per-TF scaled_score) ≥ this are written to elites.json"
    )

    @validator("draws", "tune", "chains", "min_dist", "top_k")
    def _check_positive_ints(cls, v, field):
        if not isinstance(v, int) or v < 0:
            raise ValueError(f"{field.name} must be a non-negative integer")
        return v


# ANALYSIS MODE SECTION
class AnalysisConfig(BaseModel):
    """
    Top-level “analysis” section.

    runs:         list of batch-name strings to re-analyse
    plots:        which plots to generate (trace, autocorr, convergence, scatter_pwm)
    scatter_scale: which scale to use for scatter_pwm (llr, z, p, logp, logp_norm)
    subsampling_epsilon: minimum Euclidean distance Δ in per-TF-score space to keep a new draw
    """

    runs: Optional[List[str]]
    plots: Dict[Literal["trace", "autocorr", "convergence", "scatter_pwm"], bool]
    scatter_scale: Literal["llr", "z", "p", "logp", "logp_norm"]
    subsampling_epsilon: float
    scatter_style: Literal["edges", "thresholds"] = "edges"

    @validator("subsampling_epsilon")
    def _check_positive_epsilon(cls, v):
        if not isinstance(v, (int, float)) or v <= 0.0:
            raise ValueError("subsampling_epsilon must be a positive number (float or int)")
        return float(v)


class CruncherConfig(BaseModel):
    mode: Literal["parse", "sample", "analyse", "analyze", "sample-analyse"]
    out_dir: Path
    regulator_sets: List[List[str]]

    parse: ParseConfig
    sample: Optional[SampleConfig]
    analysis: Optional[AnalysisConfig]

    @root_validator
    def _check_mode_sections(cls, values):
        mode = values.get("mode")
        has_sample = values.get("sample") is not None
        has_analysis = values.get("analysis") is not None

        if mode == "sample" and not has_sample:
            raise ValueError("When mode='sample', a [sample:] section is required.")
        if mode in ("analyse", "analyze") and not has_analysis:
            raise ValueError("When mode='analyse', an [analysis:] section is required.")
        return values


def load_config(path: Path) -> CruncherConfig:
    raw = yaml.safe_load(path.read_text())["cruncher"]
    return CruncherConfig.parse_obj(raw)
