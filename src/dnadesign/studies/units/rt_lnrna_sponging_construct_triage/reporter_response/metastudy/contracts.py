"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/contracts.py

Typed protocol, evidence, evaluation, and decision contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from dataclasses import fields as dataclass_fields
from typing import Literal

from ..profile import ReporterResponseProfile

PROTOCOL_ID = "rt_lnrna_reporter_response_metastudy.v3"
DECISION_CONTRACT_ID = "rt_lnrna_reporter_response_metastudy_decision.v4"
DecisionStatus = Literal["selected", "blocked"]
Window = tuple[float, float]
CANONICAL_CONDITION_ONTOLOGY_DIGEST = "sha256:d40f953d8415a515b79565078f59e47a3eb726d5f6569bdf84d039a017b34b28"
CANONICAL_OBSERVATION_POLICY_DIGEST = "sha256:5d1ebf8a2e4c0cac751fcd80e378ca32d0986799db7ed53b5c6913298d293dec"
_RECEIPT_CLOSURE_TOKEN = object()
_OWNER_BRIDGE_CLOSURE_TOKEN = object()
_AUDIT_DERIVATION_TOKEN = object()
_SELECTION_CLOSURE_TOKEN = object()


class MetastudyContractError(ValueError):
    """Raised when meta-study evidence or a decision violates the protocol."""


@dataclass(frozen=True, slots=True)
class MetastudyProtocol:
    """Predeclared study policy; endpoints and alternate widths are sensitivity only."""

    protocol_id: str
    primary_dose_uM: float
    sensitivity_doses_uM: tuple[float, ...]
    candidate_windows_h: tuple[Window, ...]
    endpoint_sensitivity_h: tuple[float, ...]
    centered_window_sensitivity_widths_h: tuple[float, ...]
    time_summary_statistic: Literal["median"]
    within_acquisition_observation_reduction: Literal["median"]
    ratio_reduction_order: Literal["ratio_then_reduce"]
    window_boundary: Literal["inclusive"]
    channel_time_alignment: Literal["exact"]
    expected_sampling_interval_h: float
    minimum_aligned_timepoints_per_4h_window: int
    minimum_within_acquisition_observations_per_stratum: int
    growth_phase_slope_window_h: float
    growth_phase_scale_quantile: float
    growth_phase_minimum_slope_points: int
    growth_phase_start_minimum: float
    growth_phase_end_minimum: float
    growth_phase_end_maximum: float
    within_acquisition_range_method: Literal["within_acquisition_observation_range"]
    within_acquisition_range_reference: Literal["endpoint_10h"]
    minimum_kinetic_experiments: int
    planned_kinetic_experiments: int
    planned_kinetic_experiment_ids: tuple[str, ...]
    excluded_snapshot_experiment_ids: tuple[str, ...]
    anchor_subject_order: tuple[str, ...]
    planned_anchor_experiment_ids: tuple[str, ...]
    reference_panel_target_ordered_acquisitions: int
    planned_anchor_acquisitions: int
    loo_same_or_adjacent_target_fraction: float
    clipping_or_capping: Literal["forbidden"]
    selection_order: tuple[str, ...]
    condition_ontology_digest: str
    observation_policy_digest: str

    def __post_init__(self) -> None:
        if self.protocol_id != PROTOCOL_ID:
            raise MetastudyContractError(f"protocol_id must equal {PROTOCOL_ID!r}")
        if self.primary_dose_uM != 500.0 or self.sensitivity_doses_uM != (5.0, 50.0):
            raise MetastudyContractError(
                "dose cohorts must remain the predeclared 500 uM primary and 5/50 uM sensitivity"
            )
        if self.candidate_windows_h != (
            (4.0, 8.0),
            (6.0, 10.0),
            (8.0, 12.0),
            (10.0, 14.0),
            (12.0, 16.0),
        ):
            raise MetastudyContractError("candidate windows must remain the five predeclared equal-width windows")
        if self.endpoint_sensitivity_h != (8.0, 10.0, 12.0, 14.0, 16.0):
            raise MetastudyContractError("endpoint sensitivity set changed")
        if self.centered_window_sensitivity_widths_h != (2.0, 6.0):
            raise MetastudyContractError("centered-window sensitivity widths changed")
        if (
            self.time_summary_statistic != "median"
            or self.within_acquisition_observation_reduction != "median"
            or self.ratio_reduction_order != "ratio_then_reduce"
        ):
            raise MetastudyContractError("reduction semantics changed")
        if self.window_boundary != "inclusive" or self.channel_time_alignment != "exact":
            raise MetastudyContractError("window boundaries and channel-time alignment changed")
        if self.expected_sampling_interval_h != 1.0 / 6.0:
            raise MetastudyContractError("expected sampling interval must remain ten minutes")
        if self.minimum_aligned_timepoints_per_4h_window != 25:
            raise MetastudyContractError("four-hour windows require 25 aligned inclusive timepoints")
        if self.minimum_within_acquisition_observations_per_stratum != 3:
            raise MetastudyContractError("condition strata require at least three within-acquisition observations")
        if self.growth_phase_slope_window_h != 1.0:
            raise MetastudyContractError("growth-phase slopes must use one-hour log-normalizer windows")
        if self.growth_phase_scale_quantile != 0.9:
            raise MetastudyContractError("growth-phase slopes must use the positive-slope 90th percentile scale")
        if self.growth_phase_minimum_slope_points != 4:
            raise MetastudyContractError("growth-phase slopes require at least four observations")
        if (
            self.growth_phase_start_minimum,
            self.growth_phase_end_minimum,
            self.growth_phase_end_maximum,
        ) != (0.5, 0.1, 0.6):
            raise MetastudyContractError("growth-phase thresholds changed")
        if (
            self.within_acquisition_range_method != "within_acquisition_observation_range"
            or self.within_acquisition_range_reference != "endpoint_10h"
        ):
            raise MetastudyContractError("within-acquisition range method or reference changed")
        if (self.minimum_kinetic_experiments, self.planned_kinetic_experiments) != (7, 8):
            raise MetastudyContractError("kinetic experiment gate must remain at least 7 of 8")
        if self.planned_kinetic_experiment_ids != (
            "20250622_retron_Eco1_26_43_benchmark",
            "20250707_retron_Eco1_26_43_45_46_benchmark",
            "20250718_retron_Eco1_26_45_47_48_benchmark",
            "20260418_retron_Eco1_26_43_170_171_benchmark",
            "20260507_retron_Eco1_26_43_172_173_174_175_176_benchmark",
            "20260529_retron_Eco1_26_43_177_186_benchmark",
            "20260705_retron_Eco1_26_195_196_180_199_200_197_198_benchmark",
            "20260720_retron_Eco1_26_180_201_202_203_204_benchmark",
        ):
            raise MetastudyContractError("planned kinetic experiment identities changed")
        if self.excluded_snapshot_experiment_ids != ("20251105_retron_Eco1_RT_variants",):
            raise MetastudyContractError("excluded snapshot experiment identity changed")
        if self.anchor_subject_order != (
            "rt_lnrna_pair__eco1_wt_rt__retron43_lnrna__tetO",
            "rt_lnrna_pair__eco1_wt_rt__retron26_lnrna__tetO",
        ):
            raise MetastudyContractError("anchor subject ordering must remain failed-anchor to working-anchor")
        if self.planned_anchor_experiment_ids != (
            "20250622_retron_Eco1_26_43_benchmark",
            "20250707_retron_Eco1_26_43_45_46_benchmark",
            "20260418_retron_Eco1_26_43_170_171_benchmark",
            "20260507_retron_Eco1_26_43_172_173_174_175_176_benchmark",
            "20260529_retron_Eco1_26_43_177_186_benchmark",
        ):
            raise MetastudyContractError("planned anchor co-measurement experiment identities changed")
        if not set(self.planned_anchor_experiment_ids) <= set(self.planned_kinetic_experiment_ids):
            raise MetastudyContractError("planned anchor experiments must be planned kinetic experiments")
        if (self.reference_panel_target_ordered_acquisitions, self.planned_anchor_acquisitions) != (4, 5):
            raise MetastudyContractError("reference-panel support target must remain 4 of 5 acquisitions")
        if self.loo_same_or_adjacent_target_fraction != 0.75:
            raise MetastudyContractError("leave-one-out stability target must remain 0.75")
        if self.clipping_or_capping != "forbidden":
            raise MetastudyContractError("clipping and capping are forbidden")
        if self.selection_order != (
            "require_active_to_decelerating_growth_phase",
            "maximize_worst_experiment_control_separation",
            "minimize_repeated_anchor_drift",
            "minimize_within_acquisition_observation_range",
            "earlier_end_tie_break",
        ):
            raise MetastudyContractError("selection must use the predeclared lexicographic order")
        if self.condition_ontology_digest != CANONICAL_CONDITION_ONTOLOGY_DIGEST:
            raise MetastudyContractError("condition ontology digest changed")
        if self.observation_policy_digest != CANONICAL_OBSERVATION_POLICY_DIGEST:
            raise MetastudyContractError("observation policy digest changed")


DEFAULT_PROTOCOL = MetastudyProtocol(
    protocol_id=PROTOCOL_ID,
    primary_dose_uM=500.0,
    sensitivity_doses_uM=(5.0, 50.0),
    candidate_windows_h=((4.0, 8.0), (6.0, 10.0), (8.0, 12.0), (10.0, 14.0), (12.0, 16.0)),
    endpoint_sensitivity_h=(8.0, 10.0, 12.0, 14.0, 16.0),
    centered_window_sensitivity_widths_h=(2.0, 6.0),
    time_summary_statistic="median",
    within_acquisition_observation_reduction="median",
    ratio_reduction_order="ratio_then_reduce",
    window_boundary="inclusive",
    channel_time_alignment="exact",
    expected_sampling_interval_h=1.0 / 6.0,
    minimum_aligned_timepoints_per_4h_window=25,
    minimum_within_acquisition_observations_per_stratum=3,
    growth_phase_slope_window_h=1.0,
    growth_phase_scale_quantile=0.9,
    growth_phase_minimum_slope_points=4,
    growth_phase_start_minimum=0.5,
    growth_phase_end_minimum=0.1,
    growth_phase_end_maximum=0.6,
    within_acquisition_range_method="within_acquisition_observation_range",
    within_acquisition_range_reference="endpoint_10h",
    minimum_kinetic_experiments=7,
    planned_kinetic_experiments=8,
    planned_kinetic_experiment_ids=(
        "20250622_retron_Eco1_26_43_benchmark",
        "20250707_retron_Eco1_26_43_45_46_benchmark",
        "20250718_retron_Eco1_26_45_47_48_benchmark",
        "20260418_retron_Eco1_26_43_170_171_benchmark",
        "20260507_retron_Eco1_26_43_172_173_174_175_176_benchmark",
        "20260529_retron_Eco1_26_43_177_186_benchmark",
        "20260705_retron_Eco1_26_195_196_180_199_200_197_198_benchmark",
        "20260720_retron_Eco1_26_180_201_202_203_204_benchmark",
    ),
    excluded_snapshot_experiment_ids=("20251105_retron_Eco1_RT_variants",),
    anchor_subject_order=(
        "rt_lnrna_pair__eco1_wt_rt__retron43_lnrna__tetO",
        "rt_lnrna_pair__eco1_wt_rt__retron26_lnrna__tetO",
    ),
    planned_anchor_experiment_ids=(
        "20250622_retron_Eco1_26_43_benchmark",
        "20250707_retron_Eco1_26_43_45_46_benchmark",
        "20260418_retron_Eco1_26_43_170_171_benchmark",
        "20260507_retron_Eco1_26_43_172_173_174_175_176_benchmark",
        "20260529_retron_Eco1_26_43_177_186_benchmark",
    ),
    reference_panel_target_ordered_acquisitions=4,
    planned_anchor_acquisitions=5,
    loo_same_or_adjacent_target_fraction=0.75,
    clipping_or_capping="forbidden",
    selection_order=(
        "require_active_to_decelerating_growth_phase",
        "maximize_worst_experiment_control_separation",
        "minimize_repeated_anchor_drift",
        "minimize_within_acquisition_observation_range",
        "earlier_end_tie_break",
    ),
    condition_ontology_digest=CANONICAL_CONDITION_ONTOLOGY_DIGEST,
    observation_policy_digest=CANONICAL_OBSERVATION_POLICY_DIGEST,
)


@dataclass(frozen=True, slots=True)
class EvidenceReadiness:
    """Read-only summary of exact selected evidence readiness."""

    selected_experiment_count: int
    ready_experiment_count: int
    ready_experiment_ids: tuple[str, ...]
    blocked_experiment_ids: tuple[str, ...]
    receipt_digest: str
    _receipt_closure: object | None = field(default=None, init=False, repr=False, compare=False)
    _owner_bridge_closure: object | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.selected_experiment_count, bool) or self.selected_experiment_count < 1:
            raise MetastudyContractError("selected_experiment_count must be positive")
        if (
            isinstance(self.ready_experiment_count, bool)
            or not 0 <= self.ready_experiment_count <= self.selected_experiment_count
        ):
            raise MetastudyContractError("ready_experiment_count must be between zero and selected_experiment_count")
        _unique_text(self.ready_experiment_ids, label="ready_experiment_ids", allow_empty=True)
        _unique_text(self.blocked_experiment_ids, label="blocked_experiment_ids", allow_empty=True)
        if len(self.ready_experiment_ids) != self.ready_experiment_count:
            raise MetastudyContractError("ready experiment identities must match ready_experiment_count")
        if len(self.blocked_experiment_ids) != self.selected_experiment_count - self.ready_experiment_count:
            raise MetastudyContractError(
                "blocked experiment identities must account for every selected experiment not ready"
            )
        if set(self.ready_experiment_ids) & set(self.blocked_experiment_ids):
            raise MetastudyContractError("ready and blocked experiment identities must not overlap")
        _digest(self.receipt_digest, label="receipt_digest")

    @classmethod
    def _from_validated_receipt(cls, **values: object) -> EvidenceReadiness:
        readiness = cls(**values)
        object.__setattr__(readiness, "_receipt_closure", _RECEIPT_CLOSURE_TOKEN)
        return readiness

    @property
    def is_receipt_validated(self) -> bool:
        return self._receipt_closure is _RECEIPT_CLOSURE_TOKEN

    @classmethod
    def _from_owner_bridge_receipt(cls, **values: object) -> EvidenceReadiness:
        readiness = cls._from_validated_receipt(**values)
        object.__setattr__(readiness, "_owner_bridge_closure", _OWNER_BRIDGE_CLOSURE_TOKEN)
        return readiness

    @property
    def is_selection_authorized(self) -> bool:
        return self._owner_bridge_closure is _OWNER_BRIDGE_CLOSURE_TOKEN


@dataclass(frozen=True, slots=True)
class ReaderRecordIdentity:
    """Exact public Reader record identity preserved by one materialization attempt."""

    reader_experiment_id: str
    reader_protocol_id: str
    reader_record_id: str
    reader_record_kind: str
    reader_record_schema_version: int
    reader_record_revision: int
    reader_record_revision_digest: str
    reader_record_contract_id: str
    reader_record_content_digest: str
    reader_record_path: str

    def __post_init__(self) -> None:
        for name in (
            "reader_experiment_id",
            "reader_protocol_id",
            "reader_record_id",
            "reader_record_kind",
            "reader_record_contract_id",
            "reader_record_path",
        ):
            _required_text(getattr(self, name), label=name)
        if self.reader_record_schema_version != 6:
            raise MetastudyContractError("attempt Reader record schema version must equal 6")
        if type(self.reader_record_revision) is not int or self.reader_record_revision < 1:
            raise MetastudyContractError("attempt Reader record revision must be positive")
        _digest(self.reader_record_revision_digest, label="attempt Reader revision digest")
        _digest(self.reader_record_content_digest, label="attempt Reader content digest")


@dataclass(frozen=True, slots=True)
class MaterializationBlocker:
    """One source- or experiment-level failure that prevents materialization."""

    code: str

    def __post_init__(self) -> None:
        _required_text(self.code, label="materialization blocker code")


@dataclass(frozen=True, slots=True)
class MaterializationOmission:
    """One unusable subject/window coordinate within an otherwise usable record."""

    code: str
    subject_id: str
    reduction_id: str

    def __post_init__(self) -> None:
        _required_text(self.code, label="materialization omission code")
        _required_text(self.subject_id, label="materialization omission subject_id")
        _required_text(self.reduction_id, label="materialization omission reduction_id")


@dataclass(frozen=True, slots=True)
class MaterializationAttemptReceipt:
    """Digest-bound result of attempting one selected Reader experiment."""

    contract_id: Literal["rt_lnrna_reporter_response_materialization_attempt.v4"]
    experiment_id: str
    reader_record_identity: ReaderRecordIdentity | None
    evidence_binding_artifact_id: str | None
    evidence_binding_artifact_digest: str | None
    expected_subject_ids: tuple[str, ...]
    status: Literal["complete", "partial", "blocked"]
    candidate_profile_count: int
    candidate_profile_digests: tuple[str, ...]
    candidate_omissions: tuple[MaterializationOmission, ...]
    blockers: tuple[MaterializationBlocker, ...]
    attempt_digest: str = field(default="", init=False)

    def __post_init__(self) -> None:
        if self.contract_id != "rt_lnrna_reporter_response_materialization_attempt.v4":
            raise MetastudyContractError("materialization attempt contract_id changed")
        _required_text(self.experiment_id, label="materialization attempt experiment_id")
        if self.reader_record_identity is not None and not isinstance(
            self.reader_record_identity, ReaderRecordIdentity
        ):
            raise MetastudyContractError("materialization attempt Reader record identity must be typed or null")
        if (
            self.reader_record_identity is not None
            and self.reader_record_identity.reader_experiment_id != self.experiment_id
        ):
            raise MetastudyContractError("materialization attempt experiment identity mismatch")
        if (self.evidence_binding_artifact_id is None) != (self.evidence_binding_artifact_digest is None):
            raise MetastudyContractError("materialization attempt binding identity and digest must be paired")
        if self.evidence_binding_artifact_id is not None:
            _required_text(self.evidence_binding_artifact_id, label="materialization binding artifact id")
            _digest(self.evidence_binding_artifact_digest, label="materialization binding artifact digest")
        _unique_text(self.expected_subject_ids, label="expected_subject_ids", allow_empty=self.status == "blocked")
        if self.expected_subject_ids != tuple(sorted(self.expected_subject_ids)):
            raise MetastudyContractError("expected_subject_ids must use canonical subject order")
        if type(self.candidate_profile_count) is not int or self.candidate_profile_count < 0:
            raise MetastudyContractError("candidate_profile_count must be non-negative")
        for digest in self.candidate_profile_digests:
            _digest(digest, label="candidate profile digest")
        if len(self.candidate_profile_digests) != self.candidate_profile_count:
            raise MetastudyContractError("candidate profile count and digests differ")
        if len(set(self.candidate_profile_digests)) != len(self.candidate_profile_digests):
            raise MetastudyContractError("candidate profile digests must be unique")
        if self.candidate_profile_digests != tuple(sorted(self.candidate_profile_digests)):
            raise MetastudyContractError("candidate profile digests must use canonical digest order")
        if not all(isinstance(row, MaterializationBlocker) for row in self.blockers):
            raise MetastudyContractError("materialization attempt blockers must be typed")
        if not all(isinstance(row, MaterializationOmission) for row in self.candidate_omissions):
            raise MetastudyContractError("materialization attempt omissions must be typed")
        omission_order = tuple(
            sorted(
                self.candidate_omissions,
                key=lambda row: (row.subject_id, row.reduction_id, row.code),
            )
        )
        if self.candidate_omissions != omission_order or len(set(self.candidate_omissions)) != len(
            self.candidate_omissions
        ):
            raise MetastudyContractError("materialization attempt omissions must be unique and canonical")
        if self.status == "complete":
            if (
                self.reader_record_identity is None
                or self.evidence_binding_artifact_id is None
                or not self.expected_subject_ids
                or self.blockers
                or self.candidate_omissions
                or self.candidate_profile_count < 1
            ):
                raise MetastudyContractError(
                    "complete materialization requires source identities, profiles, and no issues"
                )
        elif self.status == "partial":
            if (
                self.reader_record_identity is None
                or self.evidence_binding_artifact_id is None
                or not self.expected_subject_ids
                or self.blockers
                or not self.candidate_omissions
                or self.candidate_profile_count < 1
            ):
                raise MetastudyContractError(
                    "partial materialization requires profiles and coordinate omissions without fatal blockers"
                )
        elif self.status == "blocked":
            if (
                not (self.blockers or self.candidate_omissions)
                or self.candidate_profile_count
                or self.candidate_profile_digests
            ):
                raise MetastudyContractError("blocked materialization requires issues and no profiles")
            if self.reader_record_identity is None and self.blockers != (
                MaterializationBlocker("reader_records_not_ready"),
            ):
                raise MetastudyContractError(
                    "a blocked attempt without a Reader record must report reader_records_not_ready"
                )
            if not self.blockers:
                expected_coordinates = {
                    (subject_id, f"window-{start:g}-{end:g}h")
                    for subject_id in self.expected_subject_ids
                    for start, end in DEFAULT_PROTOCOL.candidate_windows_h
                }
                omission_coordinates = {(row.subject_id, row.reduction_id) for row in self.candidate_omissions}
                if (
                    not expected_coordinates
                    or len(omission_coordinates) != len(self.candidate_omissions)
                    or omission_coordinates != expected_coordinates
                ):
                    raise MetastudyContractError(
                        "omission-only blocked materialization requires complete expected coordinate closure"
                    )
        else:
            raise MetastudyContractError("materialization attempt status is invalid")
        object.__setattr__(self, "attempt_digest", canonical_digest(materialization_attempt_payload(self, False)))


@dataclass(frozen=True, slots=True)
class GrowthPhaseStratum:
    """Normalized one-hour log-normalizer slopes for one study condition."""

    condition_id: str
    normalized_start_slope: float
    normalized_end_slope: float

    def __post_init__(self) -> None:
        _required_text(self.condition_id, label="growth-phase condition_id")
        for name in ("normalized_start_slope", "normalized_end_slope"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise MetastudyContractError(f"{name} must be finite")


@dataclass(frozen=True, slots=True)
class ProfileAuditArtifact:
    """One canonical audit artifact bound to an exact profile source identity."""

    contract_id: Literal["rt_lnrna_reporter_response_profile_audit.v3"]
    method_id: Literal["synthetic_profile_audit_v1", "canonical_profile_observation_audit_v1"]
    profile_source_digest: str
    profile_digest: str
    condition_ontology_digest: str
    within_acquisition_observation_range: float
    reference_within_acquisition_observation_range: float
    required_observation_count: int
    overflow_observation_count: int
    clipped_observation_count: int
    growth_phase_strata: tuple[GrowthPhaseStratum, ...]
    artifact_digest: str
    _derivation_closure: object | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.contract_id != "rt_lnrna_reporter_response_profile_audit.v3":
            raise MetastudyContractError("profile audit contract_id changed")
        if self.method_id not in {"synthetic_profile_audit_v1", "canonical_profile_observation_audit_v1"}:
            raise MetastudyContractError("profile audit method_id is not enumerated")
        _digest(self.profile_source_digest, label="profile audit profile_source_digest")
        _digest(self.profile_digest, label="profile audit profile_digest")
        _digest(self.condition_ontology_digest, label="profile audit condition_ontology_digest")
        width = _nonnegative(
            self.within_acquisition_observation_range,
            label="profile audit within_acquisition_observation_range",
        )
        reference = _nonnegative(
            self.reference_within_acquisition_observation_range,
            label="profile audit reference_within_acquisition_observation_range",
        )
        if reference == 0.0 and width != 0.0:
            raise MetastudyContractError("a zero reference observation range cannot support a nonzero range")
        for name in (
            "required_observation_count",
            "overflow_observation_count",
            "clipped_observation_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise MetastudyContractError(f"{name} must be a non-negative integer")
        if not isinstance(self.growth_phase_strata, tuple) or not all(
            isinstance(row, GrowthPhaseStratum) for row in self.growth_phase_strata
        ):
            raise MetastudyContractError("profile audit growth_phase_strata must be a typed tuple")
        condition_ids = tuple(row.condition_id for row in self.growth_phase_strata)
        if condition_ids != tuple(sorted(condition_ids)) or len(condition_ids) != len(set(condition_ids)):
            raise MetastudyContractError("profile audit growth-phase strata must use unique canonical condition order")
        _digest(self.artifact_digest, label="profile audit artifact_digest")

    @classmethod
    def _from_canonical_derivation(cls, **values: object) -> ProfileAuditArtifact:
        audit = cls(**values)
        object.__setattr__(audit, "_derivation_closure", _AUDIT_DERIVATION_TOKEN)
        return audit

    @property
    def is_derivation_closed(self) -> bool:
        return self._derivation_closure is _AUDIT_DERIVATION_TOKEN


@dataclass(frozen=True, slots=True)
class ProfileEvidence:
    """One canonical profile plus digest-bound within-acquisition range evidence."""

    profile: ReporterResponseProfile
    audit: ProfileAuditArtifact

    def __post_init__(self) -> None:
        from .audits import profile_audit_payload, profile_digest, profile_source_identity_payload

        if not isinstance(self.profile, ReporterResponseProfile):
            raise MetastudyContractError("profile evidence must contain ReporterResponseProfile")
        if not isinstance(self.audit, ProfileAuditArtifact):
            raise MetastudyContractError("profile evidence requires ProfileAuditArtifact")
        expected_source = canonical_digest(profile_source_identity_payload(self.profile))
        if self.audit.profile_source_digest != expected_source:
            raise MetastudyContractError("profile audit source identity digest mismatch")
        if self.audit.profile_digest != profile_digest(self.profile):
            raise MetastudyContractError("profile audit full profile digest mismatch")
        if self.audit.artifact_digest != canonical_digest(profile_audit_payload(self.audit, include_digest=False)):
            raise MetastudyContractError("profile audit artifact digest mismatch")
        from ..profile import EndpointReduction, TimeWindowReduction

        if isinstance(self.profile.reduction, TimeWindowReduction) and not self.audit.growth_phase_strata:
            raise MetastudyContractError("time-window profile evidence requires growth-phase strata")
        if isinstance(self.profile.reduction, EndpointReduction) and self.audit.growth_phase_strata:
            raise MetastudyContractError("endpoint profile evidence cannot contain growth-phase strata")


@dataclass(frozen=True, slots=True)
class SensitivityEvaluation:
    """Digest-bound non-selectable sensitivity evidence summary."""

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


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    """Derived metrics for one selectable primary-cohort window."""

    reduction: Window
    eligible_experiment_count: int
    worst_experiment_control_separation: float
    repeated_anchor_drift: float
    within_acquisition_observation_range: float
    growth_phase_start: float
    growth_phase_end: float
    anchor_ordered_acquisition_count: int
    co_measured_anchor_acquisition_count: int
    loo_same_or_adjacent_fraction: float
    eligible: bool
    blockers: tuple[str, ...]
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.reduction not in DEFAULT_PROTOCOL.candidate_windows_h:
            raise MetastudyContractError("candidate evaluation reduction is undeclared")
        for name in (
            "eligible_experiment_count",
            "anchor_ordered_acquisition_count",
            "co_measured_anchor_acquisition_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise MetastudyContractError(f"{name} must be a non-negative integer")
        for name in (
            "worst_experiment_control_separation",
            "repeated_anchor_drift",
            "within_acquisition_observation_range",
            "growth_phase_start",
            "growth_phase_end",
            "loo_same_or_adjacent_fraction",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise MetastudyContractError(f"{name} must be finite")
        if not 0.0 <= self.loo_same_or_adjacent_fraction <= 1.0:
            raise MetastudyContractError("loo_same_or_adjacent_fraction must be between zero and one")
        _unique_text(self.blockers, label="candidate blockers", allow_empty=self.eligible)
        if self.eligible and self.blockers:
            raise MetastudyContractError("eligible candidate cannot contain blockers")
        if not self.eligible and not self.blockers:
            raise MetastudyContractError("ineligible candidate requires blockers")
        _unique_text(self.limitations, label="candidate limitations", allow_empty=True)


@dataclass(frozen=True, slots=True)
class MetastudyDecision:
    """Typed selected-or-blocked result with nullable selected reduction."""

    contract_id: str
    protocol_id: str
    status: DecisionStatus
    selection_use: Literal["descriptive_comparison"]
    evidence_grade: Literal["provisional_descriptive", "none"]
    selected_reduction: Window | None
    blockers: tuple[str, ...]
    limitations: tuple[str, ...]
    policy_digest: str
    evidence_digest: str
    readiness: EvidenceReadiness
    evaluations: tuple[CandidateEvaluation, ...]
    materialization_attempts: tuple[MaterializationAttemptReceipt, ...] = ()
    _selection_closure: object | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.contract_id != DECISION_CONTRACT_ID or self.protocol_id != PROTOCOL_ID:
            raise MetastudyContractError("decision contract or protocol identity changed")
        if self.status not in {"selected", "blocked"}:
            raise MetastudyContractError("decision status must be selected or blocked")
        if self.selection_use != "descriptive_comparison":
            raise MetastudyContractError("meta-study selection use must remain descriptive_comparison")
        _digest(self.policy_digest, label="policy_digest")
        _digest(self.evidence_digest, label="evidence_digest")
        if self.policy_digest != protocol_digest():
            raise MetastudyContractError("decision policy_digest does not match the predeclared protocol")
        _unique_text(self.blockers, label="blockers", allow_empty=self.status == "selected")
        _unique_text(self.limitations, label="limitations", allow_empty=True)
        if not isinstance(self.materialization_attempts, tuple) or not all(
            isinstance(row, MaterializationAttemptReceipt) for row in self.materialization_attempts
        ):
            raise MetastudyContractError("decision materialization_attempts must be a typed tuple")
        evidence_bearing = decision_is_evidence_bearing(self)
        if evidence_bearing:
            _validate_evaluated_decision_order(
                evaluations=self.evaluations,
                attempts=self.materialization_attempts,
            )
        elif self.status == "selected":
            raise MetastudyContractError("selected decisions must be evidence-bearing")
        if self.status == "selected":
            if self._selection_closure is not _SELECTION_CLOSURE_TOKEN:
                raise MetastudyContractError("selected decisions must be returned by canonical evaluation")
            if self.selected_reduction not in DEFAULT_PROTOCOL.candidate_windows_h or self.blockers:
                raise MetastudyContractError("selected decision requires one declared reduction and no blockers")
            if self.evidence_grade != "provisional_descriptive":
                raise MetastudyContractError("selected decisions are provisional descriptive recommendations")
            _validate_selected_decision(self)
        elif self.selected_reduction is not None or not self.blockers or self.evidence_grade != "none":
            raise MetastudyContractError("blocked decision requires no reduction, no evidence grade, and blockers")

    @classmethod
    def _from_canonical_evaluation(cls, **values: object) -> MetastudyDecision:
        decision = cls.__new__(cls)
        for name, value in values.items():
            object.__setattr__(decision, name, value)
        object.__setattr__(decision, "_selection_closure", _SELECTION_CLOSURE_TOKEN)
        decision.__post_init__()
        return decision


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


def protocol_digest(protocol: MetastudyProtocol = DEFAULT_PROTOCOL) -> str:
    """Return the canonical digest of the complete protocol policy."""

    return _canonical_digest(asdict(protocol))


def decision_to_dict(decision: MetastudyDecision) -> dict[str, object]:
    """Serialize and revalidate a decision as strict JSON data."""

    if not isinstance(decision, MetastudyDecision):
        raise MetastudyContractError("decision must be MetastudyDecision")
    payload = asdict(decision)
    payload.pop("_selection_closure", None)
    readiness = payload.get("readiness")
    if isinstance(readiness, dict):
        readiness.pop("_receipt_closure", None)
        readiness.pop("_owner_bridge_closure", None)
    validate_decision_payload(payload)
    return payload


def validate_decision_payload(payload: Mapping[str, object]) -> None:
    """Fail closed on unknown, missing, or internally inconsistent decision fields."""

    expected = {
        "contract_id",
        "protocol_id",
        "status",
        "selection_use",
        "evidence_grade",
        "selected_reduction",
        "blockers",
        "limitations",
        "policy_digest",
        "evidence_digest",
        "readiness",
        "evaluations",
        "materialization_attempts",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise MetastudyContractError("decision payload fields do not match the exact contract")
    if payload["contract_id"] != DECISION_CONTRACT_ID or payload["protocol_id"] != PROTOCOL_ID:
        raise MetastudyContractError("decision payload identity changed")
    status = payload["status"]
    blockers = payload["blockers"]
    limitations = payload["limitations"]
    if not isinstance(blockers, (list, tuple)) or not isinstance(limitations, (list, tuple)):
        raise MetastudyContractError("decision blockers and limitations must be arrays")
    if payload["selection_use"] != "descriptive_comparison":
        raise MetastudyContractError("decision selection_use changed")
    if status == "blocked":
        if payload["selected_reduction"] is not None or not blockers or payload["evidence_grade"] != "none":
            raise MetastudyContractError("blocked decision requires no reduction, no grade, and explicit blockers")
    elif status == "selected":
        if (
            not isinstance(payload["selected_reduction"], (list, tuple))
            or blockers
            or payload["evidence_grade"] != "provisional_descriptive"
        ):
            raise MetastudyContractError("selected decision requires one provisional descriptive reduction")
    else:
        raise MetastudyContractError("decision status must be selected or blocked")
    _digest(payload["policy_digest"], label="policy_digest")
    _digest(payload["evidence_digest"], label="evidence_digest")
    if payload["policy_digest"] != protocol_digest():
        raise MetastudyContractError("decision policy_digest does not match the predeclared protocol")
    readiness = payload["readiness"]
    evaluations = payload["evaluations"]
    attempts = payload["materialization_attempts"]
    if (
        not isinstance(readiness, Mapping)
        or not isinstance(evaluations, (list, tuple))
        or not isinstance(attempts, (list, tuple))
    ):
        raise MetastudyContractError("decision readiness, evaluations, and attempts must be structured")
    readiness_fields = {
        "selected_experiment_count",
        "ready_experiment_count",
        "ready_experiment_ids",
        "blocked_experiment_ids",
        "receipt_digest",
    }
    if (
        set(readiness) != readiness_fields
        or not isinstance(readiness["ready_experiment_ids"], (list, tuple))
        or not isinstance(readiness["blocked_experiment_ids"], (list, tuple))
    ):
        raise MetastudyContractError("decision readiness fields do not match the exact contract")
    parsed_readiness = EvidenceReadiness(
        selected_experiment_count=readiness["selected_experiment_count"],
        ready_experiment_count=readiness["ready_experiment_count"],
        ready_experiment_ids=tuple(readiness["ready_experiment_ids"]),
        blocked_experiment_ids=tuple(readiness["blocked_experiment_ids"]),
        receipt_digest=readiness["receipt_digest"],
    )
    evaluation_fields = {
        "reduction",
        "eligible_experiment_count",
        "worst_experiment_control_separation",
        "repeated_anchor_drift",
        "within_acquisition_observation_range",
        "growth_phase_start",
        "growth_phase_end",
        "anchor_ordered_acquisition_count",
        "co_measured_anchor_acquisition_count",
        "loo_same_or_adjacent_fraction",
        "eligible",
        "blockers",
        "limitations",
    }
    parsed_evaluations: list[CandidateEvaluation] = []
    for index, row in enumerate(evaluations):
        if not isinstance(row, Mapping) or set(row) != evaluation_fields:
            raise MetastudyContractError(f"evaluations[{index}] fields do not match the exact contract")
        reduction = row["reduction"]
        row_blockers = row["blockers"]
        row_limitations = row["limitations"]
        if (
            not isinstance(reduction, (list, tuple))
            or not isinstance(row_blockers, (list, tuple))
            or not isinstance(row_limitations, (list, tuple))
        ):
            raise MetastudyContractError(f"evaluations[{index}] array fields are malformed")
        parsed_evaluations.append(
            CandidateEvaluation(
                reduction=tuple(reduction),
                eligible_experiment_count=row["eligible_experiment_count"],
                worst_experiment_control_separation=row["worst_experiment_control_separation"],
                repeated_anchor_drift=row["repeated_anchor_drift"],
                within_acquisition_observation_range=row["within_acquisition_observation_range"],
                growth_phase_start=row["growth_phase_start"],
                growth_phase_end=row["growth_phase_end"],
                anchor_ordered_acquisition_count=row["anchor_ordered_acquisition_count"],
                co_measured_anchor_acquisition_count=row["co_measured_anchor_acquisition_count"],
                loo_same_or_adjacent_fraction=row["loo_same_or_adjacent_fraction"],
                eligible=row["eligible"],
                blockers=tuple(row_blockers),
                limitations=tuple(row_limitations),
            )
        )
    reduction_payload = payload["selected_reduction"]
    selected_reduction = tuple(reduction_payload) if isinstance(reduction_payload, (list, tuple)) else None
    parsed_attempts = tuple(
        materialization_attempt_from_payload(row, index=index) for index, row in enumerate(attempts)
    )
    evidence_bearing = decision_is_evidence_bearing(
        {
            "evaluations": parsed_evaluations,
            "materialization_attempts": parsed_attempts,
        }
    )
    if evidence_bearing:
        _validate_evaluated_decision_order(
            evaluations=tuple(parsed_evaluations),
            attempts=parsed_attempts,
        )
    elif status == "selected":
        raise MetastudyContractError("selected decisions must be evidence-bearing")
    if status == "selected":
        assert selected_reduction is not None
        _validate_selected_projection(
            readiness=parsed_readiness,
            evaluations=tuple(parsed_evaluations),
            selected_reduction=selected_reduction,
        )
    else:
        MetastudyDecision(
            contract_id=payload["contract_id"],
            protocol_id=payload["protocol_id"],
            status=status,
            selection_use=payload["selection_use"],
            evidence_grade=payload["evidence_grade"],
            selected_reduction=selected_reduction,
            blockers=tuple(blockers),
            limitations=tuple(limitations),
            policy_digest=payload["policy_digest"],
            evidence_digest=payload["evidence_digest"],
            readiness=parsed_readiness,
            evaluations=tuple(parsed_evaluations),
            materialization_attempts=parsed_attempts,
        )


def materialization_attempt_from_payload(value: object, *, index: int) -> MaterializationAttemptReceipt:
    """Strictly parse one attempt receipt without granting source authority."""

    if not isinstance(value, Mapping):
        raise MetastudyContractError(f"materialization_attempts[{index}] must be an object")
    expected = {
        "contract_id",
        "experiment_id",
        "reader_record_identity",
        "evidence_binding_artifact_id",
        "evidence_binding_artifact_digest",
        "expected_subject_ids",
        "status",
        "candidate_profile_count",
        "candidate_profile_digests",
        "candidate_omissions",
        "blockers",
        "attempt_digest",
    }
    if set(value) != expected:
        raise MetastudyContractError(f"materialization_attempts[{index}] fields do not match the exact contract")
    identity = value["reader_record_identity"]
    if identity is not None:
        if not isinstance(identity, Mapping):
            raise MetastudyContractError(
                f"materialization_attempts[{index}].reader_record_identity must be an object or null"
            )
        identity_fields = {item.name for item in dataclass_fields(ReaderRecordIdentity)}
        if set(identity) != identity_fields:
            raise MetastudyContractError(f"materialization_attempts[{index}] Reader identity fields changed")
    blockers = value["blockers"]
    omissions = value["candidate_omissions"]
    digests = value["candidate_profile_digests"]
    subjects = value["expected_subject_ids"]
    if not all(isinstance(rows, (list, tuple)) for rows in (blockers, omissions, digests, subjects)):
        raise MetastudyContractError(f"materialization_attempts[{index}] array fields are malformed")
    parsed_blockers = _materialization_blockers_from_payload(blockers, index=index, field="blockers")
    parsed_omissions = _materialization_omissions_from_payload(omissions, index=index)
    attempt = MaterializationAttemptReceipt(
        contract_id=value["contract_id"],
        experiment_id=value["experiment_id"],
        reader_record_identity=ReaderRecordIdentity(**identity) if identity is not None else None,
        evidence_binding_artifact_id=value["evidence_binding_artifact_id"],
        evidence_binding_artifact_digest=value["evidence_binding_artifact_digest"],
        expected_subject_ids=tuple(subjects),
        status=value["status"],
        candidate_profile_count=value["candidate_profile_count"],
        candidate_profile_digests=tuple(digests),
        candidate_omissions=parsed_omissions,
        blockers=tuple(parsed_blockers),
    )
    if value["attempt_digest"] != attempt.attempt_digest:
        raise MetastudyContractError(f"materialization_attempts[{index}] digest mismatch")
    return attempt


def _materialization_blockers_from_payload(
    rows: object,
    *,
    index: int,
    field: str,
) -> tuple[MaterializationBlocker, ...]:
    assert isinstance(rows, (list, tuple))
    parsed: list[MaterializationBlocker] = []
    for blocker_index, blocker in enumerate(rows):
        if not isinstance(blocker, Mapping) or set(blocker) != {"code"}:
            raise MetastudyContractError(f"materialization_attempts[{index}].{field}[{blocker_index}] fields changed")
        parsed.append(MaterializationBlocker(**blocker))
    return tuple(parsed)


def _materialization_omissions_from_payload(
    rows: object,
    *,
    index: int,
) -> tuple[MaterializationOmission, ...]:
    assert isinstance(rows, (list, tuple))
    parsed: list[MaterializationOmission] = []
    for omission_index, omission in enumerate(rows):
        if not isinstance(omission, Mapping) or set(omission) != {"code", "subject_id", "reduction_id"}:
            raise MetastudyContractError(
                f"materialization_attempts[{index}].candidate_omissions[{omission_index}] fields changed"
            )
        parsed.append(MaterializationOmission(**omission))
    return tuple(parsed)


def _validate_selected_decision(decision: MetastudyDecision) -> None:
    _validate_selected_projection(
        readiness=decision.readiness,
        evaluations=decision.evaluations,
        selected_reduction=decision.selected_reduction,
    )


def _validate_selected_projection(
    *,
    readiness: EvidenceReadiness,
    evaluations: tuple[CandidateEvaluation, ...],
    selected_reduction: Window | None,
) -> None:
    """Validate a selected projection without minting canonical-evaluation authority."""

    ready_kinetic = set(readiness.ready_experiment_ids) & set(DEFAULT_PROTOCOL.planned_kinetic_experiment_ids)
    if len(ready_kinetic) < DEFAULT_PROTOCOL.minimum_kinetic_experiments:
        raise MetastudyContractError("selected decision requires at least 7 verified kinetic experiment identities")
    reductions = tuple(row.reduction for row in evaluations)
    if len(reductions) != len(DEFAULT_PROTOCOL.candidate_windows_h) or set(reductions) != set(
        DEFAULT_PROTOCOL.candidate_windows_h
    ):
        raise MetastudyContractError("selected decision requires exactly one evaluation per declared candidate window")
    if any(row.eligible_experiment_count == 0 for row in evaluations):
        raise MetastudyContractError("selected decision cannot contain zero experiment support")
    if selected_reduction not in DEFAULT_PROTOCOL.candidate_windows_h:
        raise MetastudyContractError("selected reduction must be one declared candidate window")
    selected = next(row for row in evaluations if row.reduction == selected_reduction)
    if (
        not selected.eligible
        or selected.eligible_experiment_count < DEFAULT_PROTOCOL.minimum_kinetic_experiments
        or selected.worst_experiment_control_separation <= 0.0
        or selected.growth_phase_start < DEFAULT_PROTOCOL.growth_phase_start_minimum
        or not DEFAULT_PROTOCOL.growth_phase_end_minimum
        <= selected.growth_phase_end
        <= DEFAULT_PROTOCOL.growth_phase_end_maximum
    ):
        raise MetastudyContractError("selected evaluation does not satisfy descriptive support and phase gates")
    eligible = tuple(row for row in evaluations if row.eligible)
    expected = min(
        eligible,
        key=lambda row: (
            -row.worst_experiment_control_separation,
            (
                float("inf")
                if "repeated_reference_drift_not_estimable" in row.limitations
                else row.repeated_anchor_drift
            ),
            row.within_acquisition_observation_range,
            row.reduction[1],
        ),
    )
    if expected.reduction != selected_reduction:
        raise MetastudyContractError("selected reduction does not match the lexicographic evaluation winner")


def materialization_attempt_payload(
    attempt: MaterializationAttemptReceipt,
    include_digest: bool = True,
) -> dict[str, object]:
    """Serialize one typed attempt receipt without trusting caller-authored fields."""

    if not isinstance(attempt, MaterializationAttemptReceipt):
        raise MetastudyContractError("attempt must be MaterializationAttemptReceipt")
    payload = asdict(attempt)
    if not include_digest:
        payload.pop("attempt_digest", None)
    return payload


def decision_is_evidence_bearing(decision: MetastudyDecision | Mapping[str, object]) -> bool:
    """Return whether a decision contains a complete primary-evidence evaluation."""

    if isinstance(decision, MetastudyDecision):
        evaluations = decision.evaluations
        attempts = decision.materialization_attempts
    elif isinstance(decision, Mapping):
        evaluations = decision.get("evaluations")
        attempts = decision.get("materialization_attempts")
        if not isinstance(evaluations, (list, tuple)) or not isinstance(attempts, (list, tuple)):
            raise MetastudyContractError("decision evaluations and attempts must be arrays")
    else:
        raise MetastudyContractError("decision must be typed or structured")
    if bool(evaluations) != bool(attempts):
        raise MetastudyContractError("decision evaluations and attempts must be jointly empty or complete")
    return bool(evaluations)


def _validate_evaluated_decision_order(
    *,
    evaluations: tuple[CandidateEvaluation, ...],
    attempts: tuple[MaterializationAttemptReceipt, ...],
) -> None:
    if tuple(row.reduction for row in evaluations) != DEFAULT_PROTOCOL.candidate_windows_h:
        raise MetastudyContractError("evaluated decisions must use canonical candidate-window order")
    if tuple(row.experiment_id for row in attempts) != DEFAULT_PROTOCOL.planned_kinetic_experiment_ids:
        raise MetastudyContractError("evaluated decisions must use canonical materialization-attempt order")


def canonical_digest(value: object) -> str:
    """Digest JSON-compatible evidence deterministically."""

    return _canonical_digest(value)


def _canonical_digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _digest(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 71 or not value.startswith("sha256:"):
        raise MetastudyContractError(f"{label} must be a lowercase sha256 digest")
    if any(character not in "0123456789abcdef" for character in value[7:]):
        raise MetastudyContractError(f"{label} must be a lowercase sha256 digest")
    return value


def _nonnegative(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise MetastudyContractError(f"{label} must be a finite non-negative number")
    result = float(value)
    if result < 0.0:
        raise MetastudyContractError(f"{label} must be a finite non-negative number")
    return result


def _unique_text(values: tuple[str, ...], *, label: str, allow_empty: bool) -> None:
    if not isinstance(values, tuple) or (not values and not allow_empty):
        raise MetastudyContractError(f"{label} must be a {'possibly empty' if allow_empty else 'non-empty'} tuple")
    if any(not isinstance(value, str) or not value.strip() or value != value.strip() for value in values):
        raise MetastudyContractError(f"{label} must contain non-empty trimmed strings")
    if len(values) != len(set(values)):
        raise MetastudyContractError(f"{label} must not contain duplicates")


def _required_text(value: object, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise MetastudyContractError(f"{label} must be non-empty trimmed text")
    return value


__all__ = [
    "DECISION_CONTRACT_ID",
    "DEFAULT_OBJECTIVE_READINESS",
    "DEFAULT_PROTOCOL",
    "PROTOCOL_ID",
    "CandidateEvaluation",
    "EvidenceReadiness",
    "GrowthPhaseStratum",
    "MaterializationAttemptReceipt",
    "MaterializationBlocker",
    "MaterializationOmission",
    "MetastudyContractError",
    "MetastudyDecision",
    "MetastudyProtocol",
    "ObjectiveReadiness",
    "ProfileAuditArtifact",
    "ProfileEvidence",
    "ReaderRecordIdentity",
    "SensitivityEvaluation",
    "canonical_digest",
    "decision_to_dict",
    "materialization_attempt_payload",
    "objective_readiness_from_payload",
    "protocol_digest",
    "validate_decision_payload",
]
