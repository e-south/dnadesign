"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/contracts/sensitivity.py

Coverage-only, non-selectable sensitivity-evidence contract.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ._values import MetastudyContractError, _digest, _nonnegative


@dataclass(frozen=True, slots=True)
class SensitivityEvaluation:
    """Digest-bound coverage receipt, never an effect or rank comparison."""

    kind: Literal["dose", "endpoint", "centered_window"]
    value: float
    profile_count: int
    evidence_digest: str
    selectable: Literal[False] = False

    def __post_init__(self) -> None:
        if self.kind not in {"dose", "endpoint", "centered_window"}:
            raise MetastudyContractError("sensitivity kind is undeclared")
        _nonnegative(self.value, label="sensitivity value")
        if isinstance(self.profile_count, bool) or not isinstance(self.profile_count, int) or self.profile_count < 1:
            raise MetastudyContractError("sensitivity profile_count must be positive")
        _digest(self.evidence_digest, label="sensitivity evidence_digest")
        if self.selectable is not False:
            raise MetastudyContractError("sensitivity evaluations are never selectable")


__all__ = ["SensitivityEvaluation"]
