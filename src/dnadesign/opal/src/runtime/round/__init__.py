"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/runtime/round/__init__.py

Round-level orchestration helpers and contracts for OPAL runs. Exposes round
stage contracts and reusable stage utilities.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from .contracts import (
    ArtifactBundle,
    RoundInputs,
    RunRoundRequest,
    RunRoundResult,
    ScoreBundle,
    TrainingBundle,
    XBundle,
)

__all__ = [
    "ArtifactBundle",
    "RoundInputs",
    "RunRoundRequest",
    "RunRoundResult",
    "ScoreBundle",
    "TrainingBundle",
    "XBundle",
]
