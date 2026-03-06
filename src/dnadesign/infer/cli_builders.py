"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/infer/cli_builders.py

Shared builders for infer CLI model configuration and progress-managed execution.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional, TypeVar

from ._console import RichProgressManager
from .config import ModelConfig
from .progress import ProgressFactory

_RunnerResult = TypeVar("_RunnerResult")


def build_model_config(
    *,
    model_id: Optional[str],
    device: Optional[str],
    precision: Optional[str],
    alphabet: Optional[str],
    batch_size: Optional[int],
    preset_model: Optional[Dict[str, Any]] = None,
) -> ModelConfig:
    preset_model = preset_model or {}
    return ModelConfig(
        id=model_id or preset_model.get("id") or "evo2_7b",
        device=device or "cpu",
        precision=precision or preset_model.get("precision", "fp32"),
        alphabet=alphabet or preset_model.get("alphabet", "dna"),
        batch_size=batch_size,
    )


def run_with_progress(
    *,
    progress: bool,
    runner: Callable[[ProgressFactory], _RunnerResult],
) -> _RunnerResult:
    if not progress:
        os.environ["DNADESIGN_PROGRESS"] = "0"
    pm = RichProgressManager(enabled=progress)
    with pm:
        progress_factory = pm.factory if progress else None
        return runner(progress_factory)
