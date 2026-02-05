"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/config/moves.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Dict

from dnadesign.cruncher.config.schema_v3 import MoveConfig

MOVE_PROFILE_OVERRIDES: Dict[str, Dict[str, object]] = {
    "balanced": {"move_probs": {"S": 0.80, "B": 0.10, "M": 0.10}},
    "local": {"move_probs": {"S": 0.90, "B": 0.10, "M": 0.00}},
    "global": {"move_probs": {"S": 0.20, "B": 0.40, "M": 0.30, "L": 0.05, "W": 0.05}},
    "aggressive": {"move_probs": {"S": 0.10, "B": 0.40, "M": 0.30, "L": 0.10, "W": 0.10}},
}


def resolve_move_config(moves_cfg) -> MoveConfig:
    base = MoveConfig()
    merged = base.model_dump()
    merged.update(MOVE_PROFILE_OVERRIDES.get(moves_cfg.profile, {}))
    overrides = moves_cfg.overrides.model_dump(exclude_none=True) if moves_cfg.overrides else {}
    merged.update(overrides)
    return MoveConfig.model_validate(merged)
