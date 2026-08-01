"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/profile.py

Study-owned descriptive reporter-response profile contract.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeAlias

from ._contract_values import (
    ReporterResponseContractError,
)
from ._contract_values import (
    explicit_id_set as _explicit_id_set,
)
from ._contract_values import (
    finite_number as _finite_number,
)
from ._contract_values import (
    nonnegative_number as _nonnegative_number,
)
from ._contract_values import (
    positive_integer as _positive_integer,
)
from ._contract_values import (
    required_text as _required_text,
)
from ._contract_values import (
    sha256_digest as _sha256_digest,
)
from .temporal import (
    TemporalPolicyProjection,
    endpoint_temporal_policy_projection,
    window_temporal_policy_projection,
)

if TYPE_CHECKING:
    from ..reader_evidence import ReaderEvidenceBindingSet
    from .policy import ReporterResponseObservationPolicy

CONTRACT_ID = "rt_lnrna_reporter_response_profile.v3"
STUDY_ID = "rt_lnrna_sponging_construct_triage"

ConditionRole: TypeAlias = Literal["baseline", "positive_control", "dose"]
PairingKind: TypeAlias = Literal["paired_by_design", "pooled_controls_by_design"]
WithinAcquisitionReductionStatistic: TypeAlias = Literal["median"]
BiologicalReplicateReductionStatistic: TypeAlias = Literal["median", "mean"]
TimeSummaryStatistic: TypeAlias = Literal["median"]
RatioReductionOrder: TypeAlias = Literal[
    "ratio_then_reduce",
    "reduce_channels_then_ratio",
]
ResamplingUnit: TypeAlias = Literal[
    "biological_replicate",
    "paired_biological_replicate",
]
NotEstimableReason: TypeAlias = Literal[
    "biological_replicate_identity_unknown",
    "below_minimum_biological_replicates",
    "insufficient_valid_resamples",
]


@dataclass(frozen=True, slots=True)
class ReaderEvidenceProvenance:
    """Exact Reader record and evidence-binding artifact identities."""

    raw_design_id: str | None
    raw_assay_subject_id: str | None
    reader_experiment_id: str
    reader_protocol_id: str
    reader_record_id: str
    reader_record_kind: str
    reader_record_revision: int
    reader_record_revision_digest: str
    reader_record_content_digest: str
    reader_record_schema_version: int
    reader_record_contract_id: str
    reader_record_path: str
    evidence_binding_artifact_id: str
    evidence_binding_artifact_digest: str
    _bound_subject_id: str | None = field(default=None, init=False, repr=False, compare=False)
    _source_closed: bool = field(default=False, init=False, repr=False, compare=False)
    _declared_biological_replicate_scopes: tuple[tuple[str, str], ...] = field(
        default=(), init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.raw_design_id is None and self.raw_assay_subject_id is None:
            raise ReporterResponseContractError("provenance requires at least one raw Reader identity")
        for name in ("raw_design_id", "raw_assay_subject_id"):
            value = getattr(self, name)
            if value is not None:
                _required_text(value, field_name=name)
        for name in (
            "reader_experiment_id",
            "reader_protocol_id",
            "reader_record_id",
            "reader_record_kind",
            "reader_record_contract_id",
            "reader_record_path",
            "evidence_binding_artifact_id",
        ):
            _required_text(getattr(self, name), field_name=name)
        _positive_integer(self.reader_record_revision, field_name="reader_record_revision")
        if self.reader_record_schema_version != 6:
            raise ReporterResponseContractError("reader_record_schema_version must equal 6")
        if self.reader_record_kind != "dataframe_artifact":
            raise ReporterResponseContractError("reader_record_kind must equal dataframe_artifact")
        for name in (
            "reader_record_revision_digest",
            "reader_record_content_digest",
            "evidence_binding_artifact_digest",
        ):
            _sha256_digest(getattr(self, name), field_name=name)
        record_path = Path(self.reader_record_path)
        if record_path.is_absolute() or ".." in record_path.parts:
            raise ReporterResponseContractError("reader_record_path must be outputs-relative")

    @classmethod
    def _from_source_closed_bindings(
        cls,
        *,
        evidence_bindings: ReaderEvidenceBindingSet,
        subject_id: str,
        raw_design_id: str | None,
        raw_assay_subject_id: str | None,
    ) -> ReaderEvidenceProvenance:
        from ..reader_evidence import ReaderEvidenceBindingSet

        if not isinstance(evidence_bindings, ReaderEvidenceBindingSet) or not evidence_bindings.is_source_closed:
            raise ReporterResponseContractError(
                "reporter-response provenance requires a source-closed Reader evidence-binding set"
            )
        matches = tuple(
            row
            for row in evidence_bindings.rows
            if row.binding_state == "bound"
            and row.subject_id == subject_id
            and row.raw_design_id == raw_design_id
            and row.raw_assay_subject_id == raw_assay_subject_id
        )
        if len(matches) != 1:
            raise ReporterResponseContractError(
                f"subject {subject_id!r} and Reader identity {(raw_design_id, raw_assay_subject_id)!r} "
                "require exactly one bound Reader evidence-binding row; "
                f"observed {len(matches)}"
            )
        row = matches[0]
        provenance = cls(
            raw_design_id=row.raw_design_id,
            raw_assay_subject_id=row.raw_assay_subject_id,
            reader_experiment_id=row.reader_experiment_id,
            reader_protocol_id=row.reader_protocol_id,
            reader_record_id=row.reader_record_id,
            reader_record_kind=row.reader_record_kind,
            reader_record_revision=row.reader_record_revision,
            reader_record_revision_digest=row.reader_record_revision_digest,
            reader_record_content_digest=row.reader_record_content_digest,
            reader_record_schema_version=row.reader_record_schema_version,
            reader_record_contract_id=row.reader_record_contract_id,
            reader_record_path=row.reader_record_path,
            evidence_binding_artifact_id=evidence_bindings.artifact_id,
            evidence_binding_artifact_digest=evidence_bindings.artifact_digest,
        )
        object.__setattr__(provenance, "_bound_subject_id", subject_id)
        object.__setattr__(provenance, "_source_closed", True)
        object.__setattr__(
            provenance,
            "_declared_biological_replicate_scopes",
            tuple(
                sorted(
                    (scope.condition_value, scope.biological_replicate_id)
                    for scope in row.biological_replicate_identity_scopes
                )
            ),
        )
        return provenance

    @property
    def is_source_closed(self) -> bool:
        return self._source_closed and self._bound_subject_id is not None

    def require_bound_subject(self, subject_id: str) -> None:
        if not self.is_source_closed or self._bound_subject_id != subject_id:
            raise ReporterResponseContractError(
                "profile subject must equal the exact subject bound by source-closed Reader evidence"
            )

    def require_biological_replicate_scopes(
        self,
        values: tuple[tuple[str, str | None], ...],
    ) -> None:
        """Require condition-scoped profile identities to equal the source declaration."""

        observed = tuple(sorted({(condition, value) for condition, value in values if value is not None}))
        contains_unknown = any(value is None for _, value in values)
        expected = self._declared_biological_replicate_scopes
        if expected:
            if contains_unknown or observed != expected:
                raise ReporterResponseContractError(
                    "profile condition-scoped biological-replicate identities must equal the source-closed "
                    "Reader binding"
                )
        elif observed:
            raise ReporterResponseContractError(
                "profile cannot invent biological-replicate identities when the Reader binding declares none"
            )


@dataclass(frozen=True, slots=True)
class EndpointReduction:
    """One recorded endpoint with an optional, separately estimated elapsed time."""

    recorded_time_h: float
    estimated_elapsed_time_h: float | None = None
    ratio_reduction_order: Literal["reduce_channels_then_ratio"] = "reduce_channels_then_ratio"
    expected_cadence_h: float = 1.0 / 6.0
    temporal_policy: TemporalPolicyProjection | None = None
    kind: Literal["endpoint"] = field(default="endpoint", init=False)

    def __post_init__(self) -> None:
        _nonnegative_number(self.recorded_time_h, field_name="recorded_time_h")
        if self.estimated_elapsed_time_h is not None:
            _nonnegative_number(self.estimated_elapsed_time_h, field_name="estimated_elapsed_time_h")
        if self.ratio_reduction_order != "reduce_channels_then_ratio":
            raise ReporterResponseContractError("endpoint ratio_reduction_order must equal reduce_channels_then_ratio")
        cadence = _finite_number(self.expected_cadence_h, field_name="expected_cadence_h")
        if cadence <= 0.0:
            raise ReporterResponseContractError("expected_cadence_h must be positive")
        expected = endpoint_temporal_policy_projection(time_h=self.recorded_time_h)
        if self.temporal_policy is None:
            object.__setattr__(self, "temporal_policy", expected)
        elif not isinstance(self.temporal_policy, TemporalPolicyProjection):
            raise ReporterResponseContractError("endpoint temporal_policy must be a typed projection")
        assert self.temporal_policy is not None
        if self.temporal_policy != expected:
            raise ReporterResponseContractError("endpoint temporal_policy is incompatible with endpoint reduction")


@dataclass(frozen=True, slots=True)
class TimeWindowReduction:
    """One recorded time window with explicit summary and ratio-reduction semantics."""

    recorded_start_time_h: float
    recorded_end_time_h: float
    summary_statistic: TimeSummaryStatistic
    ratio_reduction_order: RatioReductionOrder
    estimated_elapsed_start_h: float | None = None
    estimated_elapsed_end_h: float | None = None
    expected_cadence_h: float = 1.0 / 6.0
    temporal_policy: TemporalPolicyProjection | None = None
    kind: Literal["time_window"] = field(default="time_window", init=False)

    def __post_init__(self) -> None:
        recorded_start = _nonnegative_number(self.recorded_start_time_h, field_name="recorded_start_time_h")
        recorded_end = _nonnegative_number(self.recorded_end_time_h, field_name="recorded_end_time_h")
        if recorded_start >= recorded_end:
            raise ReporterResponseContractError("recorded window start must be less than its end")
        if self.summary_statistic != "median":
            raise ReporterResponseContractError("time-window summary_statistic must equal median")
        if self.ratio_reduction_order != "ratio_then_reduce":
            raise ReporterResponseContractError("time-window ratio_reduction_order must equal ratio_then_reduce")
        cadence = _finite_number(self.expected_cadence_h, field_name="expected_cadence_h")
        if cadence <= 0.0:
            raise ReporterResponseContractError("expected_cadence_h must be positive")
        estimated = (self.estimated_elapsed_start_h, self.estimated_elapsed_end_h)
        if (estimated[0] is None) != (estimated[1] is None):
            raise ReporterResponseContractError("estimated elapsed window bounds must be provided together")
        if estimated[0] is not None and estimated[1] is not None:
            estimated_start = _nonnegative_number(estimated[0], field_name="estimated_elapsed_start_h")
            estimated_end = _nonnegative_number(estimated[1], field_name="estimated_elapsed_end_h")
            if estimated_start >= estimated_end:
                raise ReporterResponseContractError("estimated elapsed window start must be less than its end")
        expected = window_temporal_policy_projection(
            start_h=recorded_start,
            end_h=recorded_end,
            expected_cadence_h=cadence,
        )
        if self.temporal_policy is None:
            object.__setattr__(self, "temporal_policy", expected)
        elif not isinstance(self.temporal_policy, TemporalPolicyProjection):
            raise ReporterResponseContractError("time-window temporal_policy must be a typed projection")
        assert self.temporal_policy is not None
        if self.temporal_policy != expected:
            raise ReporterResponseContractError("time-window temporal_policy is incompatible with window reduction")


Reduction: TypeAlias = EndpointReduction | TimeWindowReduction


@dataclass(frozen=True, slots=True)
class ConditionMeasurement:
    """One condition summary from an acquisition and optional biological replicate."""

    observation_id: str
    condition_id: str
    source_condition_value: str
    role: ConditionRole
    dose_uM: float | None
    biological_replicate_id: str | None
    acquisition_id: str
    within_acquisition_observation_count: int
    within_acquisition_reduction_statistic: WithinAcquisitionReductionStatistic
    rfp: float
    od600: float
    rfp_over_od600: float

    def __post_init__(self) -> None:
        for name in (
            "observation_id",
            "condition_id",
            "source_condition_value",
            "acquisition_id",
        ):
            _required_text(getattr(self, name), field_name=name)
        if self.biological_replicate_id is not None:
            _required_text(self.biological_replicate_id, field_name="biological_replicate_id")
        if self.role not in {"baseline", "positive_control", "dose"}:
            raise ReporterResponseContractError("role must be baseline, positive_control, or dose")
        if self.role == "dose":
            dose = _finite_number(self.dose_uM, field_name="dose_uM")
            if dose <= 0.0:
                raise ReporterResponseContractError("dose observations require a positive dose_uM")
        elif self.dose_uM is not None:
            raise ReporterResponseContractError("baseline and positive_control observations require dose_uM=null")
        _positive_integer(
            self.within_acquisition_observation_count,
            field_name="within_acquisition_observation_count",
        )
        if self.within_acquisition_reduction_statistic != "median":
            raise ReporterResponseContractError("within_acquisition_reduction_statistic must equal median")
        _finite_number(self.rfp, field_name="rfp")
        od600 = _finite_number(self.od600, field_name="od600")
        if od600 <= 0.0:
            raise ReporterResponseContractError("od600 must be positive when rfp_over_od600 is declared")
        _finite_number(self.rfp_over_od600, field_name="rfp_over_od600")


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


@dataclass(frozen=True, slots=True)
class UncertaintyPolicy:
    """Declared biological-replicate support required before uncertainty is estimable."""

    minimum_biological_replicates: int
    biological_replicate_reduction_statistic: BiologicalReplicateReductionStatistic

    def __post_init__(self) -> None:
        minimum_units = _positive_integer(
            self.minimum_biological_replicates,
            field_name="minimum_biological_replicates",
        )
        if minimum_units < 2:
            raise ReporterResponseContractError("minimum_biological_replicates must be at least 2")
        if self.biological_replicate_reduction_statistic not in {"median", "mean"}:
            raise ReporterResponseContractError("biological_replicate_reduction_statistic must be median or mean")


@dataclass(frozen=True, slots=True)
class EstimatedMetricUncertainty:
    """Caller-supplied interval metadata for one descriptive metric."""

    estimate: float
    interval_lower: float
    interval_upper: float
    confidence_level: float
    method: str
    resampling_unit: ResamplingUnit
    draws: int
    status: Literal["estimated"] = field(default="estimated", init=False)

    def __post_init__(self) -> None:
        estimate = _finite_number(self.estimate, field_name="uncertainty.estimate")
        lower = _finite_number(self.interval_lower, field_name="uncertainty.interval_lower")
        upper = _finite_number(self.interval_upper, field_name="uncertainty.interval_upper")
        if lower > upper or not lower <= estimate <= upper:
            raise ReporterResponseContractError("uncertainty interval must contain the estimate")
        confidence = _finite_number(self.confidence_level, field_name="uncertainty.confidence_level")
        if not 0.0 < confidence < 1.0:
            raise ReporterResponseContractError("uncertainty.confidence_level must be between zero and one")
        _required_text(self.method, field_name="uncertainty.method")
        if self.resampling_unit not in {
            "biological_replicate",
            "paired_biological_replicate",
        }:
            raise ReporterResponseContractError("uncertainty.resampling_unit must be a biological replicate")
        _positive_integer(self.draws, field_name="uncertainty.draws")


@dataclass(frozen=True, slots=True)
class NotEstimableMetricUncertainty:
    """Descriptive estimate with a typed reason that uncertainty is unavailable."""

    estimate: float
    reason: NotEstimableReason
    status: Literal["not_estimable"] = field(default="not_estimable", init=False)

    def __post_init__(self) -> None:
        _finite_number(self.estimate, field_name="uncertainty.estimate")
        if self.reason not in {
            "biological_replicate_identity_unknown",
            "below_minimum_biological_replicates",
            "insufficient_valid_resamples",
        }:
            raise ReporterResponseContractError("uncertainty.reason must be a declared not-estimable reason")


MetricUncertainty: TypeAlias = EstimatedMetricUncertainty | NotEstimableMetricUncertainty


@dataclass(frozen=True, slots=True)
class DoseUncertainty:
    """Supplied estimability state for both descriptive values at one dose."""

    dose_uM: float
    biological_replicate_count: int
    normalized_reporter_response: MetricUncertainty
    relative_od: MetricUncertainty

    def __post_init__(self) -> None:
        dose = _finite_number(self.dose_uM, field_name="dose_uncertainty.dose_uM")
        if dose <= 0.0:
            raise ReporterResponseContractError("dose_uncertainty.dose_uM must be positive")
        if type(self.biological_replicate_count) is not int or self.biological_replicate_count < 0:
            raise ReporterResponseContractError("dose_uncertainty.biological_replicate_count must be non-negative")
        for name in ("normalized_reporter_response", "relative_od"):
            if not isinstance(
                getattr(self, name),
                (EstimatedMetricUncertainty, NotEstimableMetricUncertainty),
            ):
                raise ReporterResponseContractError(
                    f"dose_uncertainty.{name} must declare estimated or not_estimable uncertainty"
                )
        if self.relative_od.estimate < 0.0:
            raise ReporterResponseContractError("dose_uncertainty.relative_od estimate must be non-negative")


@dataclass(frozen=True, slots=True)
class ProfileEligibility:
    """Profiles are descriptive evidence and not optimization objectives."""

    evidence_use: Literal["descriptive"]
    optimization_status: Literal["ineligible"]
    reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.evidence_use != "descriptive":
            raise ReporterResponseContractError("evidence_use must equal descriptive")
        if self.optimization_status != "ineligible":
            raise ReporterResponseContractError("optimization_status must equal ineligible")
        _explicit_id_set(self.reasons, field_name="eligibility.reasons")


@dataclass(frozen=True, slots=True)
class ReporterResponseProfile:
    """A descriptive dose profile with exact evidence and comparability identity."""

    contract_id: str
    study_id: str
    profile_id: str
    subject_id: str
    provenance: ReaderEvidenceProvenance
    observation_policy: ReporterResponseObservationPolicy
    reduction: Reduction
    dose_grid_uM: tuple[float, ...]
    measurements: tuple[ConditionMeasurement, ...]
    pairing_policy: PairingPolicy
    dose_uncertainties: tuple[DoseUncertainty, ...]
    eligibility: ProfileEligibility
    dose_responses: tuple[DoseResponse, ...] = field(init=False)
    comparability_key: str = field(init=False)

    def __post_init__(self) -> None:
        from .canonical import comparability_key, derive_profile_rows
        from .policy import ReporterResponseObservationPolicy

        if self.contract_id != CONTRACT_ID:
            raise ReporterResponseContractError(f"contract_id must equal {CONTRACT_ID!r}")
        if self.study_id != STUDY_ID:
            raise ReporterResponseContractError(f"study_id must equal {STUDY_ID!r}")
        _required_text(self.profile_id, field_name="profile_id")
        _required_text(self.subject_id, field_name="subject_id")
        if not isinstance(self.provenance, ReaderEvidenceProvenance) or not self.provenance.is_source_closed:
            raise ReporterResponseContractError(
                "provenance must be derived from a source-closed Reader evidence-binding set"
            )
        self.provenance.require_bound_subject(self.subject_id)
        if not isinstance(self.observation_policy, ReporterResponseObservationPolicy):
            raise ReporterResponseContractError("observation_policy must be ReporterResponseObservationPolicy")
        if not isinstance(self.eligibility, ProfileEligibility):
            raise ReporterResponseContractError("eligibility must be ProfileEligibility")
        self.provenance.require_biological_replicate_scopes(
            tuple((row.source_condition_value, row.biological_replicate_id) for row in self.measurements)
        )
        dose_grid, measurement_rows, expected_responses, uncertainty_rows = derive_profile_rows(
            reduction=self.reduction,
            dose_grid_uM=self.dose_grid_uM,
            measurements=self.measurements,
            pairing_policy=self.pairing_policy,
            observation_policy=self.observation_policy,
            dose_uncertainties=self.dose_uncertainties,
        )
        if self.dose_grid_uM != dose_grid:
            raise ReporterResponseContractError("dose_grid_uM must be stored as its canonical tuple")
        if self.measurements != measurement_rows:
            raise ReporterResponseContractError("measurements must be stored as their canonical tuple")
        if self.dose_uncertainties != uncertainty_rows:
            raise ReporterResponseContractError("dose_uncertainties must be stored as their canonical tuple")
        object.__setattr__(self, "dose_responses", expected_responses)
        object.__setattr__(
            self,
            "comparability_key",
            comparability_key(
                observation_policy_digest=self.observation_policy.digest,
                reduction=self.reduction,
                dose_grid_uM=self.dose_grid_uM,
                dose_uncertainties=self.dose_uncertainties,
            ),
        )


__all__ = [
    "CONTRACT_ID",
    "STUDY_ID",
    "ConditionMeasurement",
    "ControlAssignment",
    "DoseResponse",
    "DoseUncertainty",
    "EndpointReduction",
    "EstimatedMetricUncertainty",
    "NotEstimableMetricUncertainty",
    "PairingPolicy",
    "ProfileEligibility",
    "ReporterResponseContractError",
    "ReporterResponseProfile",
    "TimeWindowReduction",
    "TemporalPolicyProjection",
    "UncertaintyPolicy",
]
