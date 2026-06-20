"""Report model for Eco1 source-sequence sufficiency validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.contracts.models import ContractIssue


@dataclass(frozen=True)
class SourceSequenceBundleSufficiencyReport:
    """Validation result for one source-sequence bundle sufficiency gate."""

    issues: tuple[ContractIssue, ...] = ()

    @property
    def passed(self) -> bool:
        return not self.issues

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "issue_count": len(self.issues),
            "issues": [issue.as_dict() for issue in self.issues],
        }


def dedupe_issues(issues: tuple[ContractIssue, ...]) -> tuple[ContractIssue, ...]:
    """Deduplicate identical check failures while preserving first occurrence."""

    observed: set[tuple[str, str, str]] = set()
    deduped: list[ContractIssue] = []
    for issue in issues:
        key = (issue.check_id, issue.path, issue.message)
        if key not in observed:
            observed.add(key)
            deduped.append(issue)
    return tuple(deduped)
