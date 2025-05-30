"""
--------------------------------------------------------------------------------
<dnadesign project>
cruncher/config.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Optional, Union

import yaml
from pydantic import BaseModel


class PlotConfig(BaseModel):
    logo: bool
    bits_mode: Literal["information", "probability"]
    dpi: int


class MotifConfig(BaseModel):
    formats: Dict[str, str]
    plot: PlotConfig


class GibbsConfig(BaseModel):
    draws: int
    tune: int
    beta: float
    chains: int
    cores: int
    min_dist: int
    block_size: int = 1
    swap_prob: float = 0.0


class OptimiserConfig(BaseModel):
    kind: Literal["gibbs"]
    gibbs: Optional[GibbsConfig]


class InitConfig(BaseModel):
    kind: Union[int, Literal["random", "consensus_shortest", "consensus_longest"]]
    pad_with: Literal[
        "background", "background_pwm", "A", "C", "G", "T"  # uniform random  # sample from PWM‐derived base frequencies
    ] = "background"


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
    motif: MotifConfig
    sample: Optional[SampleConfig]
    analysis: Optional[AnalysisConfig]


def load_config(path: Path) -> CruncherConfig:
    raw = yaml.safe_load(path.read_text())["cruncher"]
    return CruncherConfig.parse_obj(raw)
