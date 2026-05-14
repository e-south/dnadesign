"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/contracts/folding/__init__.py

Neutral folding-contract exports.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .secondary_structure_prediction_v1 import SecondaryStructurePredictionRequestV1, SecondaryStructurePredictionV1

__all__ = ["SecondaryStructurePredictionRequestV1", "SecondaryStructurePredictionV1"]
