"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/policy.py

Canonical study-owned reporter-response observation policy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Literal

from ._contract_values import ReporterResponseContractError
from ._contract_values import json_value as _json_value
from ._contract_values import required_text as _required_text
from .profile import (
    PairingKind,
    UncertaintyPolicy,
    WithinAcquisitionReductionStatistic,
)

OBSERVATION_POLICY_CONTRACT_ID = "rt_lnrna_reporter_response_observation_policy.v3"
NORMALIZED_REPORTER_FORMULA = "(Z_d-Z_0)/(Z_positive-Z_0)"
RELATIVE_OD_FORMULA = "OD600_d/OD600_0"


@dataclass(frozen=True, slots=True)
class ReporterResponseObservationPolicy:
    """Typed non-window semantics shared by comparable response profiles."""

    policy_id: str
    pairing_kind: PairingKind
    within_acquisition_reduction_statistic: WithinAcquisitionReductionStatistic
    biological_replicate_uncertainty_policy: UncertaintyPolicy
    contract_id: Literal["rt_lnrna_reporter_response_observation_policy.v3"] = field(
        default=OBSERVATION_POLICY_CONTRACT_ID,
        init=False,
    )
    normalized_reporter_formula: Literal["(Z_d-Z_0)/(Z_positive-Z_0)"] = field(
        default=NORMALIZED_REPORTER_FORMULA,
        init=False,
    )
    relative_od_formula: Literal["OD600_d/OD600_0"] = field(
        default=RELATIVE_OD_FORMULA,
        init=False,
    )
    clipping_policy: Literal["forbidden"] = field(default="forbidden", init=False)
    digest: str = field(default="", init=False)

    def __post_init__(self) -> None:
        _required_text(self.policy_id, field_name="observation_policy.policy_id")
        if self.pairing_kind not in {"paired_by_design", "pooled_controls_by_design"}:
            raise ReporterResponseContractError(
                "observation_policy.pairing_kind must be paired_by_design or pooled_controls_by_design"
            )
        if self.within_acquisition_reduction_statistic != "median":
            raise ReporterResponseContractError(
                "observation_policy.within_acquisition_reduction_statistic must equal median"
            )
        if not isinstance(self.biological_replicate_uncertainty_policy, UncertaintyPolicy):
            raise ReporterResponseContractError(
                "observation_policy.biological_replicate_uncertainty_policy must be UncertaintyPolicy"
            )
        object.__setattr__(self, "digest", _policy_digest(self))


def _policy_digest(policy: ReporterResponseObservationPolicy) -> str:
    payload = asdict(policy)
    payload.pop("digest", None)
    encoded = json.dumps(
        _json_value(payload),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


__all__ = [
    "NORMALIZED_REPORTER_FORMULA",
    "OBSERVATION_POLICY_CONTRACT_ID",
    "RELATIVE_OD_FORMULA",
    "ReporterResponseObservationPolicy",
]
