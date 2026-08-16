"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/storage/history_contracts.py

Validates continuity requirements for append-only OPAL campaign histories.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from ..core.utils import ExitCodes, OpalError
from .state import CampaignState


def require_completed_predecessors(state: CampaignState, *, requested_round: int) -> None:
    round_index = int(requested_round)
    if round_index <= 0:
        return
    completed = {int(entry.round_index) for entry in state.rounds if entry.status == "completed"}
    required = set(range(round_index))
    missing = sorted(required - completed)
    if missing:
        raise OpalError(
            f"Round {round_index} requires completed predecessor rounds {missing} in the same campaign history. "
            "Relocate the prior history with `opal history import` before continuing.",
            ExitCodes.BAD_ARGS,
        )
