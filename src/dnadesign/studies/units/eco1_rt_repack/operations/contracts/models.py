"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/contracts/models.py

Shared contract validation report types.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContractIssue:
    """A single actionable contract-validation failure."""

    check_id: str
    message: str
    path: str = ""
    severity: str = "error"

    def as_dict(self) -> dict[str, str]:
        return {
            "check_id": self.check_id,
            "message": self.message,
            "path": self.path,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class ContractReport:
    """Structured result for one validation pass."""

    phase: str
    issues: tuple[ContractIssue, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.issues

    @property
    def issue_count(self) -> int:
        return len(self.issues)

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "passed": self.passed,
            "issue_count": self.issue_count,
            "issues": [issue.as_dict() for issue in self.issues],
        }
