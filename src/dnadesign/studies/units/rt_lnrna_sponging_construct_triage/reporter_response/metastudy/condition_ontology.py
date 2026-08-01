"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/condition_ontology.py

Exact study-owned condition identities for reporter-response materialization.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Literal

from ..profile import ConditionRole
from .contracts._values import MetastudyContractError

CONDITION_ONTOLOGY_CONTRACT_ID = "rt_lnrna_reporter_response_condition_ontology.v1"


@dataclass(frozen=True, slots=True)
class ConditionDefinition:
    """One exact Reader treatment label and its study role."""

    condition_id: str
    treatment_label: str
    role: ConditionRole
    dose_uM: float | None

    def __post_init__(self) -> None:
        for name in ("condition_id", "treatment_label"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise MetastudyContractError(f"condition.{name} must be non-empty trimmed text")
        if self.role not in {"baseline", "positive_control", "dose"}:
            raise MetastudyContractError("condition.role must be baseline, positive_control, or dose")
        if self.role == "dose":
            if (
                isinstance(self.dose_uM, bool)
                or not isinstance(self.dose_uM, (int, float))
                or not math.isfinite(float(self.dose_uM))
                or float(self.dose_uM) <= 0.0
            ):
                raise MetastudyContractError("dose conditions require a positive finite dose_uM")
        elif self.dose_uM is not None:
            raise MetastudyContractError("control conditions require dose_uM=null")


@dataclass(frozen=True, slots=True)
class ReporterResponseConditionOntology:
    """Exact labels and channels; no treatment string is parsed or normalized."""

    ontology_id: str
    conditions: tuple[ConditionDefinition, ...]
    sample_type_value: str
    reporter_channel: str
    normalizer_channel: str
    ratio_channel: str
    contract_id: Literal["rt_lnrna_reporter_response_condition_ontology.v1"] = field(
        default=CONDITION_ONTOLOGY_CONTRACT_ID,
        init=False,
    )
    digest: str = field(default="", init=False)

    def __post_init__(self) -> None:
        for name in (
            "ontology_id",
            "sample_type_value",
            "reporter_channel",
            "normalizer_channel",
            "ratio_channel",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip() or value != value.strip():
                raise MetastudyContractError(f"condition ontology {name} must be non-empty trimmed text")
        if not self.conditions or not all(isinstance(row, ConditionDefinition) for row in self.conditions):
            raise MetastudyContractError("condition ontology requires typed condition definitions")
        ids = [row.condition_id for row in self.conditions]
        labels = [row.treatment_label for row in self.conditions]
        if len(ids) != len(set(ids)) or len(labels) != len(set(labels)):
            raise MetastudyContractError("condition ids and treatment labels must be unique")
        roles = [row.role for row in self.conditions]
        if roles.count("baseline") != 1 or roles.count("positive_control") > 1 or "dose" not in roles:
            raise MetastudyContractError(
                "condition ontology requires one baseline, zero or one positive control, and doses"
            )
        doses = [float(row.dose_uM) for row in self.conditions if row.role == "dose"]
        if len(doses) != len(set(doses)):
            raise MetastudyContractError("condition ontology dose values must be unique")
        channels = (self.reporter_channel, self.normalizer_channel, self.ratio_channel)
        if len(set(channels)) != 3:
            raise MetastudyContractError("reporter, normalizer, and ratio channels must be distinct")
        payload = asdict(self)
        payload.pop("digest", None)
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        object.__setattr__(self, "digest", "sha256:" + hashlib.sha256(encoded).hexdigest())

    @property
    def by_treatment_label(self) -> dict[str, ConditionDefinition]:
        return {row.treatment_label: row for row in self.conditions}

    def condition_for_dose(self, dose_uM: float) -> ConditionDefinition:
        matches = [row for row in self.conditions if row.role == "dose" and row.dose_uM == dose_uM]
        if len(matches) != 1:
            raise MetastudyContractError(f"condition ontology does not declare dose {dose_uM:g} uM exactly once")
        return matches[0]

    @property
    def positive_control(self) -> ConditionDefinition | None:
        """Return the explicitly declared positive control, if one exists."""

        return next((row for row in self.conditions if row.role == "positive_control"), None)


DEFAULT_CONDITION_ONTOLOGY = ReporterResponseConditionOntology(
    ontology_id="rt_lnrna_reporter_response_conditions.v1",
    conditions=(
        ConditionDefinition("baseline", "0 nm aTc; 0 uM IPTG", "baseline", None),
        ConditionDefinition("positive_control", "200 nm aTc; 0 uM IPTG", "positive_control", None),
        ConditionDefinition("dose_5_uM", "0 nm aTc; 5 uM IPTG", "dose", 5.0),
        ConditionDefinition("dose_50_uM", "0 nm aTc; 50 uM IPTG", "dose", 50.0),
        ConditionDefinition("dose_500_uM", "0 nm aTc; 500 uM IPTG", "dose", 500.0),
    ),
    sample_type_value="SAMPLE",
    reporter_channel="RFP",
    normalizer_channel="OD600",
    ratio_channel="RFP/OD600",
)


__all__ = [
    "CONDITION_ONTOLOGY_CONTRACT_ID",
    "ConditionDefinition",
    "DEFAULT_CONDITION_ONTOLOGY",
    "ReporterResponseConditionOntology",
]
