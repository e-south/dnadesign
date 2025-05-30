"""
--------------------------------------------------------------------------------
<dnadesign project>
cruncher/utils/config.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Union

import yaml
from pydantic import BaseModel, root_validator, validator


# Low-level plot & parse objects
class PlotConfig(BaseModel):
    logo: bool
    bits_mode: Literal["information", "probability"]
    dpi: int


class ParseConfig(BaseModel):
    formats: Dict[str, str]
    plot: PlotConfig


# Cooling & move mixer configuration
class CoolingStage(BaseModel):
    sweeps: int
    beta: float

    @validator("sweeps", "beta")
    def _non_negative(cls, v, field):
        assert v >= 0, f"{field.name} must be ≥ 0"
        return v


class CoolingConfig(BaseModel):
    kind: Literal["piecewise", "fixed"] = "fixed"
    stages: Optional[List[CoolingStage]] = None
    beta: Optional[float] = None  # may be back-filled

    @validator("beta", always=True)
    def _validate_beta(cls, v, values):
        if values.get("kind") == "fixed" and v is not None:
            assert v > 0, "beta must be > 0"
        return v


class MoveConfig(BaseModel):
    block_len_range: Sequence[int] = (3, 12)
    multi_k_range: Sequence[int] = (2, 6)
    slide_max_shift: int = 2
    swap_len_range: Sequence[int] = (6, 12)

    # *Only* coerce the ranges into tuples — leave slide_max_shift as-is.
    @validator("block_len_range", "multi_k_range", "swap_len_range", pre=True)
    def _list_to_tuple(cls, v):
        if isinstance(v, (list, tuple)):
            return tuple(int(x) for x in v)
        return v


class GibbsConfig(BaseModel):
    draws: int
    tune: int
    chains: int
    cores: int
    min_dist: int
    cooling: CoolingConfig = CoolingConfig()
    moves: MoveConfig = MoveConfig()

    # legacy keys
    beta: Optional[float] = 1.0
    block_size: Optional[int] = None
    swap_prob: Optional[float] = None

    @validator("cooling", always=True)
    def _backward_fill_cooling(cls, v, values):
        if values.get("beta") is not None and v.kind == "fixed" and v.beta is None:
            v.beta = values["beta"]
        return v


class OptimiserConfig(BaseModel):
    kind: Literal["gibbs"]
    gibbs: GibbsConfig


class InitConfig(BaseModel):
    # allow legacy 'length' → 'kind'
    kind: Union[int, Literal["random", "consensus_shortest", "consensus_longest"]]
    pad_with: Literal["background", "background_pwm", "A", "C", "G", "T"] = "background"

    @root_validator(pre=True)
    def alias_length_to_kind(cls, values):
        if "length" in values and "kind" not in values:
            values["kind"] = values.pop("length")
        return values


class SampleConfig(BaseModel):
    init: InitConfig
    optimiser: OptimiserConfig
    top_k: int
    bidirectional: bool = True
    plots: Dict[Literal["trace", "autocorr", "convergence", "scatter_pwm"], bool] = {
        "trace": True,
        "autocorr": True,
        "convergence": True,
        "scatter_pwm": False,
    }


class AnalysisConfig(BaseModel):
    runs: Optional[List[str]]
    plots: List[str]


class CruncherConfig(BaseModel):
    mode: Literal["parse", "sample", "analyse", "analyze"]
    out_dir: Path
    regulator_sets: List[List[str]]
    parse: ParseConfig
    sample: Optional[SampleConfig]
    analysis: Optional[AnalysisConfig]


def load_config(path: Path) -> CruncherConfig:
    raw = yaml.safe_load(path.read_text())["cruncher"]
    return CruncherConfig.parse_obj(raw)
