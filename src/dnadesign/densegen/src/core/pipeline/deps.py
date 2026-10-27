"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/src/core/pipeline/deps.py

Pipeline dependency wiring helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from ...adapters.optimizer import DenseArraysAdapter, OptimizerAdapter
from ...adapters.outputs import SinkBase, build_sinks
from ...adapters.sources import data_source_factory
from ...adapters.sources.base import BaseDataSource
from ...config import DenseGenConfig
from ..postprocess import generate_pad


@dataclass
class PipelineDeps:
    source_factory: Callable[[object, Path], BaseDataSource]
    sink_factory: Callable[[DenseGenConfig, Path], Iterable[SinkBase]]
    optimizer: OptimizerAdapter
    pad: Callable[..., tuple[str, dict] | str]


def default_deps() -> PipelineDeps:
    return PipelineDeps(
        source_factory=data_source_factory,
        sink_factory=build_sinks,
        optimizer=DenseArraysAdapter(),
        pad=generate_pad,
    )
