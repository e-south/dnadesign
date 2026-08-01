"""Control pairing and derived dose-response contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from .._contract_values import ReporterResponseContractError
from .._contract_values import explicit_id_set as _explicit_id_set
from .._contract_values import finite_number as _finite_number
from .._contract_values import required_text as _required_text

PairingKind: TypeAlias = Literal["paired_by_design", "pooled_controls_by_design"]


@dataclass(frozen=True, slots=True)
class ControlAssignment:
    """Explicit control observations used for one dose observation."""

    dose_observation_id: str
    baseline_observation_ids: tuple[str, ...]
    positive_control_observation_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _required_text(self.dose_observation_id, field_name="dose_observation_id")
        _explicit_id_set(self.baseline_observation_ids, field_name="baseline_observation_ids")
        _explicit_id_set(self.positive_control_observation_ids, field_name="positive_control_observation_ids")
        if self.dose_observation_id in {*self.baseline_observation_ids, *self.positive_control_observation_ids}:
            raise ReporterResponseContractError("a dose observation cannot also be its own control")


@dataclass(frozen=True, slots=True)
class PairingPolicy:
    """Design-declared pairing; no identifier similarity is interpreted as pairing."""

    kind: PairingKind
    assignments: tuple[ControlAssignment, ...]

    def __post_init__(self) -> None:
        if self.kind not in {"paired_by_design", "pooled_controls_by_design"}:
            raise ReporterResponseContractError(
                "pairing_policy.kind must be paired_by_design or pooled_controls_by_design"
            )
        if not self.assignments:
            raise ReporterResponseContractError("pairing policy requires an explicit control assignment")
        dose_ids = [assignment.dose_observation_id for assignment in self.assignments]
        if len(dose_ids) != len(set(dose_ids)):
            raise ReporterResponseContractError("each dose observation requires exactly one control assignment")
        if self.kind == "paired_by_design":
            for assignment in self.assignments:
                if (
                    len(assignment.baseline_observation_ids) != 1
                    or len(assignment.positive_control_observation_ids) != 1
                ):
                    raise ReporterResponseContractError(
                        "paired_by_design assignments require exactly one baseline and one positive control"
                    )


@dataclass(frozen=True, slots=True)
class DoseResponse:
    """One scoped biological replicate's response, or one descriptive acquisition summary."""

    dose_uM: float
    dose_observation_id: str
    biological_replicate_id: str | None
    acquisition_id: str
    baseline_observation_ids: tuple[str, ...]
    positive_control_observation_ids: tuple[str, ...]
    normalized_reporter_response: float
    relative_od: float

    def __post_init__(self) -> None:
        dose = _finite_number(self.dose_uM, field_name="dose_response.dose_uM")
        if dose <= 0.0:
            raise ReporterResponseContractError("dose_response.dose_uM must be positive")
        for name in ("dose_observation_id", "acquisition_id"):
            _required_text(getattr(self, name), field_name=f"dose_response.{name}")
        if self.biological_replicate_id is not None:
            _required_text(self.biological_replicate_id, field_name="dose_response.biological_replicate_id")
        _explicit_id_set(self.baseline_observation_ids, field_name="dose_response.baseline_observation_ids")
        _explicit_id_set(
            self.positive_control_observation_ids,
            field_name="dose_response.positive_control_observation_ids",
        )
        _finite_number(
            self.normalized_reporter_response,
            field_name="dose_response.normalized_reporter_response",
        )
        relative_od = _finite_number(self.relative_od, field_name="dose_response.relative_od")
        if relative_od < 0.0:
            raise ReporterResponseContractError("dose_response.relative_od must be non-negative")
