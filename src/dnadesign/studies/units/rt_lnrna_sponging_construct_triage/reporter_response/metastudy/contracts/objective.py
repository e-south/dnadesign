"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/contracts/objective.py

Independent objective-readiness contract and parser.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from ._values import MetastudyContractError


@dataclass(frozen=True, slots=True)
class ObjectiveReadiness:
    """Independent readiness of a descriptive reduction for optimization use."""

    contract_id: Literal["rt_lnrna_reporter_response_objective_readiness.v3"]
    status: Literal["ready", "blocked"]
    objective_id: str | None
    blockers: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.contract_id != "rt_lnrna_reporter_response_objective_readiness.v3":
            raise MetastudyContractError("objective-readiness contract_id changed")
        if (
            not isinstance(self.blockers, tuple)
            or len(self.blockers) != len(set(self.blockers))
            or any(not isinstance(value, str) or not value.strip() or value != value.strip() for value in self.blockers)
        ):
            raise MetastudyContractError("objective-readiness blockers must be unique trimmed strings")
        if self.status == "ready":
            if not isinstance(self.objective_id, str) or not self.objective_id.strip():
                raise MetastudyContractError("ready objective readiness requires an objective_id")
            if self.blockers:
                raise MetastudyContractError("ready objective readiness cannot contain blockers")
        elif self.status == "blocked":
            if self.objective_id is not None or not self.blockers:
                raise MetastudyContractError("blocked objective readiness requires no objective and explicit blockers")
        else:
            raise MetastudyContractError("objective-readiness status is invalid")


DEFAULT_OBJECTIVE_READINESS = ObjectiveReadiness(
    contract_id="rt_lnrna_reporter_response_objective_readiness.v3",
    status="blocked",
    objective_id=None,
    blockers=(
        "constrained_objective_not_defined",
        "biological_replicate_uncertainty_not_estimable",
        "od_linearity_not_validated",
    ),
)


def objective_readiness_from_payload(value: object) -> ObjectiveReadiness:
    """Parse one exact JSON/YAML objective-readiness projection."""

    expected = {"contract_id", "status", "objective_id", "blockers"}
    if not isinstance(value, Mapping) or set(value) != expected:
        raise MetastudyContractError("objective-readiness fields do not match the exact contract")
    blockers = value["blockers"]
    if not isinstance(blockers, (list, tuple)):
        raise MetastudyContractError("objective-readiness blockers must be an array")
    return ObjectiveReadiness(
        contract_id=value["contract_id"],
        status=value["status"],
        objective_id=value["objective_id"],
        blockers=tuple(blockers),
    )


__all__ = [
    "DEFAULT_OBJECTIVE_READINESS",
    "ObjectiveReadiness",
    "objective_readiness_from_payload",
]
