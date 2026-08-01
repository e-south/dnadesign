"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/rt_lnrna_sponging_construct_triage/reporter_response/metastudy/evaluation.py

Fail-closed profile evaluation and lexicographic window selection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import replace
from pathlib import Path

from .. import profile_to_dict
from ..profile import TimeWindowReduction
from ..temporal import window_temporal_policy_projection
from .audits import profile_audit_payload, profile_source_identity_payload
from .contracts._values import MetastudyContractError, canonical_digest
from .contracts.decision import (
    DECISION_CONTRACT_ID,
    CandidateEvaluation,
    MetastudyDecision,
)
from .contracts.materialization import (
    EvidenceReadiness,
    MaterializationAttemptReceipt,
    materialization_attempt_payload,
)
from .contracts.profile import ProfileEvidence
from .contracts.protocol import (
    DEFAULT_PROTOCOL,
    MetastudyProtocol,
    Window,
    protocol_digest,
)
from .evidence_projection import (
    ProfileEvidenceProjection,
    profile_source_identity_projection,
)

METASTUDY_ROUTE_ID = "rt_lnrna_reporter_response_metastudy"


def decision_from_readiness(
    readiness: EvidenceReadiness,
    *,
    protocol: MetastudyProtocol = DEFAULT_PROTOCOL,
) -> MetastudyDecision:
    """Create the evidence-free blocked decision required by an unready route."""

    if not isinstance(readiness, EvidenceReadiness) or not readiness.is_receipt_validated:
        raise MetastudyContractError("readiness must come from readiness_from_receipt")
    ready_kinetic_ids = set(readiness.ready_experiment_ids) & set(protocol.planned_kinetic_experiment_ids)
    if len(ready_kinetic_ids) >= protocol.minimum_kinetic_experiments:
        raise MetastudyContractError("ready evidence requires profile evaluation, not a readiness-only decision")
    blocker = f"reader_evidence_ready_{readiness.ready_experiment_count}_of_{readiness.selected_experiment_count}"
    return MetastudyDecision(
        contract_id=DECISION_CONTRACT_ID,
        protocol_id=protocol.protocol_id,
        status="blocked",
        selection_use="descriptive_comparison",
        evidence_grade="none",
        selected_reduction=None,
        blockers=(blocker, "minimum_7_of_8_kinetic_experiments_not_met"),
        limitations=(),
        policy_digest=protocol_digest(protocol),
        evidence_digest=canonical_digest(
            {
                "receipt_digest": readiness.receipt_digest,
                "selected_experiment_count": readiness.selected_experiment_count,
                "ready_experiment_count": readiness.ready_experiment_count,
                "ready_experiment_ids": readiness.ready_experiment_ids,
                "blocked_experiment_ids": readiness.blocked_experiment_ids,
            }
        ),
        readiness=readiness,
        evaluations=(),
        materialization_attempts=(),
    )


def readiness_from_receipt(payload: Mapping[str, object]) -> EvidenceReadiness:
    """Adapt one public read-only readiness receipt without importing its producer."""

    if not isinstance(payload, Mapping):
        raise MetastudyContractError("readiness receipt must be an object")
    expected_top_level = {
        "available_protocols",
        "contract_errors",
        "experiments",
        "ok",
        "reader_command",
        "route_id",
        "selected_blockers",
        "summary",
    }
    if set(payload) != expected_top_level:
        raise MetastudyContractError("readiness receipt top-level fields do not match the exact contract")
    if payload["route_id"] != METASTUDY_ROUTE_ID:
        raise MetastudyContractError(f"readiness receipt route_id must equal {METASTUDY_ROUTE_ID}")
    summary = payload["summary"]
    blockers = payload["selected_blockers"]
    experiments = payload["experiments"]
    contract_errors = payload["contract_errors"]
    if (
        not isinstance(summary, Mapping)
        or not isinstance(blockers, list)
        or not isinstance(experiments, list)
        or not isinstance(contract_errors, list)
    ):
        raise MetastudyContractError("readiness receipt requires summary and selected_blockers")
    expected_summary = {
        "contract_error_count",
        "experiment_count",
        "membership_count",
        "related_membership_count",
        "selected_blocker_count",
        "selected_membership_count",
        "selected_ready_count",
    }
    if set(summary) != expected_summary:
        raise MetastudyContractError("readiness receipt summary fields do not match the exact contract")
    for field in expected_summary:
        value = summary[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise MetastudyContractError(f"readiness receipt summary.{field} must be a non-negative integer")
    if len(contract_errors) != summary["contract_error_count"]:
        raise MetastudyContractError("readiness contract_error_count does not match contract_errors")
    if contract_errors:
        raise MetastudyContractError("readiness receipt contains contract_errors")
    selected = summary.get("selected_membership_count")
    ready = summary.get("selected_ready_count")
    if isinstance(selected, bool) or not isinstance(selected, int):
        raise MetastudyContractError("selected_membership_count must be an integer")
    if isinstance(ready, bool) or not isinstance(ready, int):
        raise MetastudyContractError("selected_ready_count must be an integer")
    blocked_ids: list[str] = []
    for index, blocker in enumerate(blockers):
        if not isinstance(blocker, Mapping) or set(blocker) != {"experiment_id", "route_id"}:
            raise MetastudyContractError(f"selected_blockers[{index}] must be an object")
        if blocker["route_id"] != payload["route_id"]:
            raise MetastudyContractError(f"selected_blockers[{index}].route_id changed")
        experiment_id = blocker.get("experiment_id")
        if not isinstance(experiment_id, str) or not experiment_id.strip():
            raise MetastudyContractError(f"selected_blockers[{index}].experiment_id must be text")
        blocked_ids.append(experiment_id)
    if len(blocked_ids) != summary.get("selected_blocker_count"):
        raise MetastudyContractError("selected blocker count does not match blocker identities")
    ready_ids: list[str] = []
    selected_ids: list[str] = []
    related_ids: list[str] = []
    membership_count = 0
    for index, experiment in enumerate(experiments):
        if not isinstance(experiment, Mapping):
            raise MetastudyContractError(f"experiments[{index}] must be an object")
        experiment_id = experiment.get("experiment_id")
        memberships = experiment.get("memberships")
        if not isinstance(experiment_id, str) or not experiment_id.strip() or not isinstance(memberships, list):
            raise MetastudyContractError(f"experiments[{index}] identity or memberships are malformed")
        for membership_index, membership in enumerate(memberships):
            if not isinstance(membership, Mapping) or set(membership) != {
                "membership",
                "ready",
                "required_reader_state",
                "route_id",
            }:
                raise MetastudyContractError(f"experiments[{index}].memberships[{membership_index}] fields changed")
            membership_count += 1
            if membership["route_id"] != payload["route_id"]:
                continue
            if membership["required_reader_state"] != "records_ready" or not isinstance(membership["ready"], bool):
                raise MetastudyContractError("meta-study readiness membership semantics changed")
            if membership["membership"] == "selected":
                selected_ids.append(experiment_id)
                if membership["ready"]:
                    ready_ids.append(experiment_id)
            elif membership["membership"] == "related":
                related_ids.append(experiment_id)
            else:
                raise MetastudyContractError("meta-study membership must be selected or related")
    if len(experiments) != summary["experiment_count"] or membership_count != summary["membership_count"]:
        raise MetastudyContractError("readiness experiment or membership counts changed")
    if len(selected_ids) != selected or len(ready_ids) != ready:
        raise MetastudyContractError("selected readiness identities do not match summary counts")
    if len(related_ids) != summary["related_membership_count"]:
        raise MetastudyContractError("related readiness identities do not match summary count")
    expected_selected_ids = set(DEFAULT_PROTOCOL.planned_kinetic_experiment_ids)
    if set(selected_ids) != expected_selected_ids:
        raise MetastudyContractError("selected readiness identity set does not match the predeclared route cohort")
    if set(related_ids) != set(DEFAULT_PROTOCOL.excluded_snapshot_experiment_ids):
        raise MetastudyContractError("related readiness identity set must equal the excluded snapshot context")
    if set(blocked_ids) != set(selected_ids) - set(ready_ids):
        raise MetastudyContractError("selected blocker identities do not close the selected experiment set")
    complete = not blocked_ids and ready == selected and not contract_errors
    if not isinstance(payload["ok"], bool) or payload["ok"] is not complete:
        raise MetastudyContractError("readiness receipt ok does not match complete selected readiness")
    return EvidenceReadiness._from_validated_receipt(
        selected_experiment_count=selected,
        ready_experiment_count=ready,
        ready_experiment_ids=tuple(ready_ids),
        blocked_experiment_ids=tuple(blocked_ids),
        receipt_digest=canonical_digest({key: value for key, value in payload.items() if key != "reader_command"}),
    )


def readiness_from_live_bridge(*, phd_root: Path) -> EvidenceReadiness:
    """Run the exact bridge-owned route checker and authorize its typed receipt."""

    root = Path(phd_root).expanduser().resolve()
    skill_root = (root / ".agents/skills/retron-assay-study-bridge").resolve()
    registry = (skill_root / "references/reader-experiment-routes.json").resolve()
    checker = (skill_root / "scripts/check_reader_experiment_readiness.py").resolve()
    if not registry.is_file() or not checker.is_file():
        raise MetastudyContractError("canonical bridge registry or live-readiness checker is missing")
    command = [
        sys.executable,
        str(checker),
        "--registry",
        str(registry),
        "--phd-root",
        str(root),
        "--route-id",
        METASTUDY_ROUTE_ID,
    ]
    completed = subprocess.run(command, cwd=root, check=False, capture_output=True, text=True)
    raw = completed.stdout.strip() or completed.stderr.strip()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise MetastudyContractError("live bridge checker returned invalid JSON") from exc
    structural = readiness_from_receipt(payload)
    return EvidenceReadiness._from_owner_bridge_receipt(
        selected_experiment_count=structural.selected_experiment_count,
        ready_experiment_count=structural.ready_experiment_count,
        ready_experiment_ids=structural.ready_experiment_ids,
        blocked_experiment_ids=structural.blocked_experiment_ids,
        receipt_digest=structural.receipt_digest,
    )


def evaluate_metastudy(
    evidence: Iterable[ProfileEvidence],
    *,
    readiness: EvidenceReadiness,
    attempts: Iterable[MaterializationAttemptReceipt],
    protocol: MetastudyProtocol = DEFAULT_PROTOCOL,
) -> MetastudyDecision:
    """Evaluate the predeclared primary cohort and select without a weighted score."""

    if not isinstance(readiness, EvidenceReadiness) or not readiness.is_receipt_validated:
        raise MetastudyContractError("readiness must come from readiness_from_receipt")
    ready_kinetic_ids = set(readiness.ready_experiment_ids) & set(protocol.planned_kinetic_experiment_ids)
    if len(ready_kinetic_ids) < protocol.minimum_kinetic_experiments:
        return decision_from_readiness(readiness, protocol=protocol)
    if not readiness.is_selection_authorized:
        raise MetastudyContractError("selection requires readiness returned by the owner-bound live bridge runner")
    rows = tuple(evidence)
    attempt_rows = tuple(attempts)
    return _evaluate_canonical_evidence(
        rows,
        readiness=readiness,
        attempts=attempt_rows,
        protocol=protocol,
        evidence_digest=_evidence_digest(rows, readiness, attempt_rows),
        require_source_closure=True,
    )


def reevaluate_evidence_projection(
    evidence: Iterable[ProfileEvidenceProjection],
    *,
    readiness: EvidenceReadiness,
    attempts: Iterable[MaterializationAttemptReceipt],
    evidence_digest: str,
    protocol: MetastudyProtocol = DEFAULT_PROTOCOL,
) -> MetastudyDecision:
    """Repeat decision math from bundled content without claiming source authenticity."""

    if not isinstance(readiness, EvidenceReadiness):
        raise MetastudyContractError("publication readiness must be a typed structural projection")
    rows = tuple(evidence)
    attempt_rows = tuple(attempts)
    return _evaluate_canonical_evidence(
        rows,
        readiness=readiness,
        attempts=attempt_rows,
        protocol=protocol,
        evidence_digest=evidence_digest,
        require_source_closure=False,
    )


def _evaluate_canonical_evidence(
    rows: tuple[ProfileEvidence | ProfileEvidenceProjection, ...],
    *,
    readiness: EvidenceReadiness,
    attempts: tuple[MaterializationAttemptReceipt, ...],
    protocol: MetastudyProtocol,
    evidence_digest: str,
    require_source_closure: bool,
) -> MetastudyDecision:
    """Shared deterministic evaluation after live or publication-specific parsing."""

    anchor_subject_order = protocol.anchor_subject_order
    _require_attempt_ledger(attempts, rows=rows, protocol=protocol)
    if not rows:
        raise MetastudyContractError("profile evidence must not be empty")
    grouped: dict[Window, list[ProfileEvidence | ProfileEvidenceProjection]] = defaultdict(list)
    for row in rows:
        expected_type = ProfileEvidence if require_source_closure else ProfileEvidenceProjection
        if not isinstance(row, expected_type):
            raise MetastudyContractError("evidence rows do not match the selected evaluation boundary")
        if require_source_closure and (
            row.audit.method_id != "canonical_profile_observation_audit_v1" or not row.audit.is_derivation_closed
        ):
            raise MetastudyContractError("selection requires derivation-closed canonical profile audits")
        if not require_source_closure and row.audit.method_id != "canonical_profile_observation_audit_v1":
            raise MetastudyContractError("publication evaluation requires canonical profile-observation audits")
        if row.audit.condition_ontology_digest != protocol.condition_ontology_digest:
            raise MetastudyContractError("profile evidence condition ontology does not match the protocol")
        if row.profile.observation_policy.digest != protocol.observation_policy_digest:
            raise MetastudyContractError("profile observation policy does not match the protocol")
        reduction = row.profile.reduction
        if not isinstance(reduction, TimeWindowReduction):
            raise MetastudyContractError("primary selection accepts only TimeWindowReduction profiles")
        window = (reduction.recorded_start_time_h, reduction.recorded_end_time_h)
        if window not in protocol.candidate_windows_h:
            raise MetastudyContractError(f"profile uses undeclared candidate window {window!r}")
        if (
            reduction.summary_statistic != protocol.time_summary_statistic
            or reduction.ratio_reduction_order != protocol.ratio_reduction_order
        ):
            raise MetastudyContractError("profile reduction semantics do not match the protocol")
        expected_temporal_policy = window_temporal_policy_projection(
            start_h=reduction.recorded_start_time_h,
            end_h=reduction.recorded_end_time_h,
            expected_cadence_h=reduction.expected_cadence_h,
        )
        if reduction.temporal_policy != expected_temporal_policy:
            raise MetastudyContractError("profile temporal policy projection does not match the canonical operator")
        if protocol.primary_dose_uM not in row.profile.dose_grid_uM:
            raise MetastudyContractError("primary selection profiles must contain the 500 uM cohort")
        if any(
            measurement.within_acquisition_reduction_statistic != protocol.within_acquisition_observation_reduction
            for measurement in row.profile.measurements
        ):
            raise MetastudyContractError("profiles must use median within-acquisition observation reduction")
        grouped[window].append(row)
    if set(grouped) != set(protocol.candidate_windows_h):
        raise MetastudyContractError("evidence must cover every predeclared candidate window exactly")
    common_coordinates = _require_cross_window_identity(grouped, readiness=readiness, protocol=protocol)
    grouped = {
        window: [
            row
            for row in candidate_rows
            if (row.profile.provenance.reader_experiment_id, row.profile.subject_id) in common_coordinates
        ]
        for window, candidate_rows in grouped.items()
    }
    for candidate_rows in grouped.values():
        comparability_keys = {row.profile.comparability_key for row in candidate_rows}
        if len(candidate_rows) < 2 or len(comparability_keys) != 1:
            raise MetastudyContractError("candidate profiles fail exact comparability")

    evaluations = tuple(
        _evaluate_candidate(tuple(grouped[window]), window=window, anchors=anchor_subject_order, protocol=protocol)
        for window in protocol.candidate_windows_h
    )
    quality_blockers = tuple(
        dict.fromkeys(
            blocker
            for evaluation in evaluations
            for blocker in evaluation.blockers
            if blocker
            in {
                "required_observation_count_zero",
                "observation_overflow_detected",
                "observation_clipping_detected",
            }
        )
    )
    if quality_blockers:
        return _blocked(
            readiness,
            protocol,
            attempts,
            evaluations,
            evidence_digest=evidence_digest,
            blockers=quality_blockers,
        )
    eligible = tuple(row for row in evaluations if row.eligible)
    if not eligible:
        return _blocked(
            readiness,
            protocol,
            attempts,
            evaluations,
            evidence_digest=evidence_digest,
            blockers=tuple(
                f"window_{row.reduction[0]:g}_{row.reduction[1]:g}:" + ",".join(row.blockers) for row in evaluations
            ),
        )
    selected = min(eligible, key=_selection_key)
    experiment_ids = sorted(
        {row.profile.provenance.reader_experiment_id for candidate_rows in grouped.values() for row in candidate_rows}
    )
    stable = 0
    failed_fold = False
    selected_index = protocol.candidate_windows_h.index(selected.reduction)
    for omitted in experiment_ids:
        loo_evaluations = tuple(
            _evaluate_candidate(
                tuple(row for row in grouped[window] if row.profile.provenance.reader_experiment_id != omitted),
                window=window,
                anchors=anchor_subject_order,
                protocol=protocol,
                minimum_experiments=protocol.minimum_kinetic_experiments - 1,
            )
            for window in protocol.candidate_windows_h
        )
        loo_eligible = tuple(row for row in loo_evaluations if row.eligible)
        if not loo_eligible:
            failed_fold = True
            continue
        loo_selected = min(loo_eligible, key=_selection_key)
        loo_index = protocol.candidate_windows_h.index(loo_selected.reduction)
        stable += int(abs(loo_index - selected_index) <= 1)
    loo_fraction = stable / len(experiment_ids) if experiment_ids else 0.0
    selected_limitations = list(selected.limitations)
    if failed_fold:
        selected_limitations.append("loo_fold_without_eligible_candidate")
    if loo_fraction < protocol.loo_same_or_adjacent_target_fraction:
        selected_limitations.append("loo_choice_same_or_adjacent_below_75_percent")
    evaluations = tuple(
        replace(
            row,
            loo_same_or_adjacent_fraction=loo_fraction,
            limitations=tuple(dict.fromkeys(selected_limitations)),
        )
        if row.reduction == selected.reduction
        else row
        for row in evaluations
    )
    selected = next(row for row in evaluations if row.reduction == selected.reduction)
    decision_limitations = [
        "retrospective_calibration_cohort",
        "growth_phase_rule_requires_external_replication",
        "acquisition_projection_is_descriptive_only",
        "study_side_blank_correction_not_claimed",
        *selected.limitations,
    ]
    if any(attempt.candidate_omissions for attempt in attempts):
        decision_limitations.append("subject_window_omissions_present")
    return MetastudyDecision._from_canonical_evaluation(
        contract_id=DECISION_CONTRACT_ID,
        protocol_id=protocol.protocol_id,
        status="selected",
        selection_use="descriptive_comparison",
        evidence_grade="provisional_descriptive",
        selected_reduction=selected.reduction,
        blockers=(),
        limitations=tuple(dict.fromkeys(decision_limitations)),
        policy_digest=protocol_digest(protocol),
        evidence_digest=evidence_digest,
        readiness=readiness,
        evaluations=evaluations,
        materialization_attempts=attempts,
    )


def _evaluate_candidate(
    rows: tuple[ProfileEvidence | ProfileEvidenceProjection, ...],
    *,
    window: Window,
    anchors: tuple[str, ...],
    protocol: MetastudyProtocol,
    minimum_experiments: int | None = None,
) -> CandidateEvaluation:
    by_identity: dict[tuple[str, str], ProfileEvidence | ProfileEvidenceProjection] = {}
    experiment_subjects: dict[str, set[str]] = defaultdict(set)
    blockers: list[str] = []
    limitations: list[str] = []
    growth_phase_start_by_experiment: dict[str, list[float]] = defaultdict(list)
    growth_phase_end_by_experiment: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        experiment_id = row.profile.provenance.reader_experiment_id
        identity = (experiment_id, row.profile.subject_id)
        if identity in by_identity:
            raise MetastudyContractError(f"duplicate candidate profile identity {identity!r}")
        by_identity[identity] = row
        experiment_subjects[experiment_id].add(row.profile.subject_id)
        audit = row.audit
        quality = row.audit
        if audit.within_acquisition_observation_range > audit.reference_within_acquisition_observation_range:
            limitations.append("within_acquisition_observation_range_exceeds_endpoint_reference")
        if quality.required_observation_count == 0:
            blockers.append("required_observation_count_zero")
        if quality.overflow_observation_count:
            blockers.append("observation_overflow_detected")
        if quality.clipped_observation_count:
            blockers.append("observation_clipping_detected")
        growth_phase_start_by_experiment[experiment_id].extend(
            value.normalized_start_slope for value in audit.growth_phase_strata
        )
        growth_phase_end_by_experiment[experiment_id].extend(
            value.normalized_end_slope for value in audit.growth_phase_strata
        )
    anchor_set = set(anchors)
    planned_anchor_experiments = set(protocol.planned_anchor_experiment_ids)
    eligible_experiments = tuple(sorted(experiment_subjects))
    required_experiments = minimum_experiments or protocol.minimum_kinetic_experiments
    if len(eligible_experiments) < required_experiments:
        blockers.append("minimum_7_of_8_kinetic_experiments_not_met")
    growth_phase_start_values = tuple(
        statistics.median(growth_phase_start_by_experiment[experiment_id])
        for experiment_id in sorted(growth_phase_start_by_experiment)
        if growth_phase_start_by_experiment[experiment_id]
    )
    growth_phase_end_values = tuple(
        statistics.median(growth_phase_end_by_experiment[experiment_id])
        for experiment_id in sorted(growth_phase_end_by_experiment)
        if growth_phase_end_by_experiment[experiment_id]
    )
    growth_phase_start = statistics.median(growth_phase_start_values) if growth_phase_start_values else 0.0
    growth_phase_end = statistics.median(growth_phase_end_values) if growth_phase_end_values else 0.0
    if not growth_phase_start_values or not growth_phase_end_values:
        blockers.append("growth_phase_not_estimable")
    else:
        if growth_phase_start < protocol.growth_phase_start_minimum:
            blockers.append("growth_phase_start_below_active_threshold")
        if growth_phase_end < protocol.growth_phase_end_minimum:
            blockers.append("growth_phase_end_near_plateau")
        if growth_phase_end > protocol.growth_phase_end_maximum:
            blockers.append("growth_phase_end_before_deceleration")

    separation_by_experiment: dict[str, list[float]] = defaultdict(list)
    anchor_values_by_acquisition: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        profile = row.profile
        experiment_id = profile.provenance.reader_experiment_id
        if experiment_id not in eligible_experiments:
            continue
        observations = {measurement.observation_id: measurement for measurement in profile.measurements}
        response_by_observation = {response.dose_observation_id: response for response in profile.dose_responses}
        for assignment in profile.pairing_policy.assignments:
            dose = observations[assignment.dose_observation_id]
            if dose.dose_uM != protocol.primary_dose_uM:
                continue
            baseline = statistics.median(
                observations[value].rfp_over_od600 for value in assignment.baseline_observation_ids
            )
            positive = statistics.median(
                observations[value].rfp_over_od600 for value in assignment.positive_control_observation_ids
            )
            separation_by_experiment[experiment_id].append(positive - baseline)
            if experiment_id in planned_anchor_experiments and profile.subject_id in anchor_set:
                acquisition_key = (experiment_id, dose.acquisition_id)
                anchor_values_by_acquisition[acquisition_key][profile.subject_id].append(
                    response_by_observation[dose.observation_id].normalized_reporter_response
                )
    if any(not values or min(values) <= 0.0 for values in separation_by_experiment.values()):
        blockers.append("positive_control_separation_failed")
    worst_separation = min((min(values) for values in separation_by_experiment.values()), default=0.0)

    co_measured = 0
    ordered = 0
    plate_anchor_medians: dict[str, list[float]] = defaultdict(list)
    for subjects in anchor_values_by_acquisition.values():
        if set(subjects) != anchor_set:
            continue
        co_measured += 1
        medians = [statistics.median(subjects[subject]) for subject in anchors]
        ordered += int(all(left < right for left, right in zip(medians, medians[1:], strict=False)))
        for subject, value in zip(anchors, medians, strict=True):
            plate_anchor_medians[subject].append(value)
    if co_measured < protocol.planned_anchor_acquisitions:
        limitations.append("reference_panel_incomplete")
    if (
        co_measured < protocol.reference_panel_target_ordered_acquisitions
        or ordered < protocol.reference_panel_target_ordered_acquisitions
    ):
        limitations.append("reference_panel_support_below_target")
    drift_values = [max(values) - min(values) for values in plate_anchor_medians.values() if len(values) >= 2]
    if not drift_values:
        limitations.append("repeated_reference_drift_not_estimable")
    drift = max(drift_values, default=0.0)
    ranges = [row.audit.within_acquisition_observation_range for row in rows]
    within_acquisition_range = statistics.median(ranges) if ranges else 0.0
    unique_blockers = tuple(dict.fromkeys(blockers))
    return CandidateEvaluation(
        reduction=window,
        eligible_experiment_count=len(eligible_experiments),
        worst_experiment_control_separation=worst_separation,
        repeated_anchor_drift=drift,
        within_acquisition_observation_range=within_acquisition_range,
        growth_phase_start=growth_phase_start,
        growth_phase_end=growth_phase_end,
        anchor_ordered_acquisition_count=ordered,
        co_measured_anchor_acquisition_count=co_measured,
        loo_same_or_adjacent_fraction=0.0,
        eligible=not unique_blockers,
        blockers=unique_blockers,
        limitations=tuple(dict.fromkeys(limitations)),
    )


def _selection_key(row: CandidateEvaluation) -> tuple[float, float, float, float]:
    return (
        -row.worst_experiment_control_separation,
        (float("inf") if "repeated_reference_drift_not_estimable" in row.limitations else row.repeated_anchor_drift),
        row.within_acquisition_observation_range,
        row.reduction[1],
    )


def _blocked(
    readiness: EvidenceReadiness,
    protocol: MetastudyProtocol,
    attempts: tuple[MaterializationAttemptReceipt, ...],
    evaluations: tuple[CandidateEvaluation, ...],
    *,
    evidence_digest: str,
    blockers: tuple[str, ...],
) -> MetastudyDecision:
    attempt_blockers = tuple(
        f"materialization:{attempt.experiment_id}:source:{blocker.code}"
        for attempt in attempts
        if attempt.status == "blocked"
        for blocker in attempt.blockers
    )
    return MetastudyDecision(
        contract_id=DECISION_CONTRACT_ID,
        protocol_id=protocol.protocol_id,
        status="blocked",
        selection_use="descriptive_comparison",
        evidence_grade="none",
        selected_reduction=None,
        blockers=tuple(dict.fromkeys((*attempt_blockers, *blockers))),
        limitations=(),
        policy_digest=protocol_digest(protocol),
        evidence_digest=evidence_digest,
        readiness=readiness,
        evaluations=evaluations,
        materialization_attempts=attempts,
    )


def _evidence_digest(
    rows: tuple[ProfileEvidence, ...],
    readiness: EvidenceReadiness,
    attempts: tuple[MaterializationAttemptReceipt, ...],
) -> str:
    profiles = [
        {
            "profile": profile_to_dict(row.profile),
            "audit": profile_audit_payload(row.audit),
        }
        for row in sorted(rows, key=lambda item: item.profile.profile_id)
    ]
    return canonical_digest(_evidence_payload(profiles=profiles, readiness=readiness, attempts=attempts))


def decision_evidence_payload(
    evidence: Iterable[ProfileEvidence],
    *,
    decision: MetastudyDecision,
) -> dict[str, object]:
    """Build the canonical evidence-bearing payload for one evaluated decision."""

    rows = tuple(evidence)
    _require_attempt_ledger(decision.materialization_attempts, rows=rows, protocol=DEFAULT_PROTOCOL)
    profiles = [
        {"profile": profile_to_dict(row.profile), "audit": profile_audit_payload(row.audit)}
        for row in sorted(rows, key=lambda item: item.profile.profile_id)
    ]
    payload = _evidence_payload(
        profiles=profiles,
        readiness=decision.readiness,
        attempts=decision.materialization_attempts,
    )
    if canonical_digest(payload) != decision.evidence_digest:
        raise MetastudyContractError("publication evidence does not match the evaluated decision digest")
    return payload


def _evidence_payload(
    *,
    profiles: list[dict[str, object]],
    readiness: EvidenceReadiness,
    attempts: tuple[MaterializationAttemptReceipt, ...],
) -> dict[str, object]:
    return {
        "readiness_receipt_digest": readiness.receipt_digest,
        "materialization_attempts": [materialization_attempt_payload(row) for row in attempts],
        "profiles": profiles,
    }


def _require_attempt_ledger(
    attempts: tuple[MaterializationAttemptReceipt, ...],
    *,
    rows: tuple[ProfileEvidence | ProfileEvidenceProjection, ...],
    protocol: MetastudyProtocol,
) -> None:
    if not attempts or not all(isinstance(row, MaterializationAttemptReceipt) for row in attempts):
        raise MetastudyContractError("selection requires typed materialization attempts")
    if tuple(row.experiment_id for row in attempts) != protocol.planned_kinetic_experiment_ids:
        raise MetastudyContractError("materialization attempts must use canonical selected-experiment order")
    attempt_by_id = {row.experiment_id: row for row in attempts}
    evidence_by_experiment: dict[str, list[ProfileEvidence | ProfileEvidenceProjection]] = defaultdict(list)
    for row in rows:
        evidence_by_experiment[row.profile.provenance.reader_experiment_id].append(row)
    for experiment_id, attempt in attempt_by_id.items():
        experiment_rows = evidence_by_experiment.get(experiment_id, [])
        observed_digests = tuple(sorted(row.audit.profile_digest for row in experiment_rows))
        if attempt.status in {"complete", "partial"}:
            if observed_digests != attempt.candidate_profile_digests:
                raise MetastudyContractError(f"materialization attempt profile digests differ for {experiment_id}")
            assert attempt.reader_record_identity is not None
            expected_identity = attempt.reader_record_identity
            for row in experiment_rows:
                provenance = row.profile.provenance
                if (
                    provenance.reader_experiment_id,
                    provenance.reader_protocol_id,
                    provenance.reader_record_id,
                    provenance.reader_record_kind,
                    provenance.reader_record_schema_version,
                    provenance.reader_record_revision,
                    provenance.reader_record_revision_digest,
                    provenance.reader_record_contract_id,
                    provenance.reader_record_content_digest,
                    provenance.reader_record_path,
                ) != (
                    expected_identity.reader_experiment_id,
                    expected_identity.reader_protocol_id,
                    expected_identity.reader_record_id,
                    expected_identity.reader_record_kind,
                    expected_identity.reader_record_schema_version,
                    expected_identity.reader_record_revision,
                    expected_identity.reader_record_revision_digest,
                    expected_identity.reader_record_contract_id,
                    expected_identity.reader_record_content_digest,
                    expected_identity.reader_record_path,
                ):
                    raise MetastudyContractError(
                        f"materialization attempt Reader identity differs from profile provenance for {experiment_id}"
                    )
                if (
                    provenance.evidence_binding_artifact_id != attempt.evidence_binding_artifact_id
                    or provenance.evidence_binding_artifact_digest != attempt.evidence_binding_artifact_digest
                ):
                    raise MetastudyContractError(
                        f"materialization attempt binding identity differs from profile provenance for {experiment_id}"
                    )
            profile_coordinates = {(row.profile.subject_id, _profile_reduction_id(row)) for row in experiment_rows}
            omission_coordinates = {(row.subject_id, row.reduction_id) for row in attempt.candidate_omissions}
            expected_coordinates = {
                (subject_id, f"window-{start:g}-{end:g}h")
                for subject_id in attempt.expected_subject_ids
                for start, end in protocol.candidate_windows_h
            }
            if (
                profile_coordinates & omission_coordinates
                or profile_coordinates | omission_coordinates != expected_coordinates
            ):
                raise MetastudyContractError(
                    f"materialization attempt candidate coordinate closure differs for {experiment_id}"
                )
        if attempt.status == "blocked" and experiment_rows:
            raise MetastudyContractError(f"blocked materialization attempt cannot contribute profiles: {experiment_id}")


def _profile_reduction_id(row: ProfileEvidence | ProfileEvidenceProjection) -> str:
    reduction = row.profile.reduction
    if not isinstance(reduction, TimeWindowReduction):
        raise MetastudyContractError("primary materialization attempts accept only time-window profiles")
    return f"window-{reduction.recorded_start_time_h:g}-{reduction.recorded_end_time_h:g}h"


def _require_cross_window_identity(
    grouped: Mapping[Window, list[ProfileEvidence | ProfileEvidenceProjection]],
    *,
    readiness: EvidenceReadiness,
    protocol: MetastudyProtocol,
) -> set[tuple[str, str]]:
    rosters: list[set[tuple[str, str]]] = []
    expected_provenance: dict[tuple[str, str], tuple[object, ...]] = {}
    ready_ids = set(readiness.ready_experiment_ids)
    planned_ids = set(protocol.planned_kinetic_experiment_ids)
    for window in protocol.candidate_windows_h:
        roster: set[tuple[str, str]] = set()
        for row in grouped[window]:
            profile = row.profile
            identity = (profile.provenance.reader_experiment_id, profile.subject_id)
            roster.add(identity)
            if identity[0] not in ready_ids or identity[0] not in planned_ids:
                raise MetastudyContractError("profile experiment identity is not a verified planned kinetic experiment")
            source_identity_payload = (
                profile_source_identity_payload(profile)
                if isinstance(row, ProfileEvidence)
                else profile_source_identity_projection(profile)
            )
            source_identity = tuple(source_identity_payload.items())
            prior = expected_provenance.setdefault(identity, source_identity)
            if prior != source_identity:
                raise MetastudyContractError("cross-window Reader provenance identity changed")
        rosters.append(roster)
    common = set.intersection(*rosters) if rosters else set()
    if not common:
        raise MetastudyContractError("candidate windows have no common experiment-subject coordinates")
    return common


__all__ = [
    "decision_evidence_payload",
    "decision_from_readiness",
    "evaluate_metastudy",
    "reevaluate_evidence_projection",
    "readiness_from_live_bridge",
    "readiness_from_receipt",
]
