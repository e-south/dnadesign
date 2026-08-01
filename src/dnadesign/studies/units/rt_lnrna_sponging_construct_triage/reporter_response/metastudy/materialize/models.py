"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/materialize/models.py

Materialization readiness value owned by the reporter-response meta-study.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ..contracts.materialization import MaterializationAttemptReceipt, MaterializationOmission
from ..contracts.profile import ProfileEvidence
from ..sensitivity_coverage import SensitivityCoverageLedger


@dataclass(frozen=True, slots=True)
class MaterializationReadiness:
    """Complete, partial, or blocked materialization with explicit issue scope."""

    status: Literal["complete", "partial", "blocked"]
    attempt: MaterializationAttemptReceipt
    candidate_evidence: tuple[ProfileEvidence, ...] = ()
    endpoint_evidence: tuple[ProfileEvidence, ...] = ()
    centered_window_evidence: tuple[ProfileEvidence, ...] = ()
    sensitivity_coverage: SensitivityCoverageLedger | None = None

    def __post_init__(self) -> None:
        evidence = self.candidate_evidence + self.endpoint_evidence + self.centered_window_evidence
        if self.status != self.attempt.status:
            raise ValueError("materialization status must equal its attempt receipt")
        if self.status == "blocked" and (
            not (self.blockers or self.omissions) or evidence or self.sensitivity_coverage is not None
        ):
            raise ValueError("blocked materialization requires issues and no evidence")
        if self.status in {"complete", "partial"} and (
            self.blockers
            or not self.candidate_evidence
            or self.sensitivity_coverage is None
            or self.attempt.attempt_digest != self.sensitivity_coverage.materialization_attempt_digest
        ):
            raise ValueError("usable materialization requires candidate evidence and exact sensitivity coverage")
        if self.status == "complete" and self.omissions:
            raise ValueError("complete materialization cannot contain omissions")
        if self.status == "partial" and not self.omissions:
            raise ValueError("partial materialization requires coordinate omissions")

    @property
    def blockers(self) -> tuple[str, ...]:
        return tuple(row.code for row in self.attempt.blockers)

    @property
    def omissions(self) -> tuple[MaterializationOmission, ...]:
        return self.attempt.candidate_omissions


__all__ = ["MaterializationReadiness"]
