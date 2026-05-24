"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/studies/rt_lnrna_sponging_construct_triage/reader_spop_plan.py

Dry-run Reader SPOP planning for RT-lnRNA sponging assay labels.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import pyarrow.parquet as pq
import yaml

from .variant_genbank_catalog import build_variant_genbank_catalog

DEFAULT_READER_EXPERIMENT_IDS: tuple[str, ...] = (
    "20250622_retron_Eco1_26_43_benchmark",
    "20250707_retron_Eco1_26_43_45_46_benchmark",
    "20250718_retron_Eco1_26_45_47_48_benchmark",
    "20251105_retron_Eco1_RT_variants",
    "20260418_retron_Eco1_26_43_170_171_benchmark",
    "20260507_retron_Eco1_26_43_172_173_174_175_176_benchmark",
)

SPOP_METRIC_ID = "reader_spop_endpoint_auc_v1"
REPORTER_PLASMID_ID = "pBbS2c-RFP"
REPORTER_DESIGN_ID = "pBbS2c-rfp"
REPORTER_READOUT = "RFP/OD600"
VIABILITY_READOUT = "OD600"
READER_RATIO_RECORD_ID = "ratio_reporter_normalizer/df"
DEFAULT_LAMBDA = 0.5
_EPS_POSITIVE = 1e-8
_RETRON_DESIGN_RE = re.compile(r"pES-retron-(?P<number>\d+);\s*pBbS2c-rfp", flags=re.IGNORECASE)
_TREATMENT_RE = re.compile(
    r"^\s*(?P<atc>[0-9]+(?:\.[0-9]+)?)\s*nm\s*aTc;\s*"
    r"(?P<iptg>[0-9]+(?:\.[0-9]+)?)\s*[uµ]M\s*IPTG\s*$",
    flags=re.IGNORECASE,
)
_BASE_RESOLVED_CONSTRUCT_CANDIDATES = {
    "retron26": "rt_lnrna_pair__eco1_wt_rt__retron26_lnrna__tetO",
    "retron43": "rt_lnrna_pair__eco1_wt_rt__retron43_lnrna__tetO",
}
_REQUIRED_RATIO_COLUMNS = ("time", "position", "channel", "value", "design_id", "treatment")
_SINGLE_POINT_ENDPOINT_OVERRIDES = {
    "20251105_retron_Eco1_RT_variants": {
        "reader_artifact_time_h": 0.0,
        "elapsed_time_h": 10.0,
        "basis": "single_point_mid_log_elapsed_time_override",
        "caveat": (
            "Reader artifact stores a single-point mid-log plate read at time 0; "
            "elapsed time is approximately 10 h after seeding."
        ),
    }
}
_EXCLUDED_ASSAY_SUBJECTS = {
    ("20260507_retron_Eco1_26_43_172_173_174_175_176_benchmark", "retron176"): (
        "plate map carried retron176, but no actual strain was present in those wells"
    )
}


class ReaderSpopContractError(ValueError):
    """Raised when Reader evidence cannot satisfy the SPOP dry-run contract."""


@dataclass(frozen=True, slots=True)
class ReaderSpopIssue:
    experiment_id: str
    code: str
    message: str
    severity: str = "error"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReaderArtifactRef:
    experiment_id: str
    record_id: str
    path: Path
    manifest_path: Path
    content_digest: str

    @property
    def ref(self) -> str:
        return f"{self.experiment_id}:{self.record_id}"


@dataclass(frozen=True, slots=True)
class _EndpointSelection:
    row_time_h: float
    report_time_h: float
    endpoint_time_h: float
    assay_metadata: Mapping[str, object]
    qc_flags: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ReaderSpopObservation:
    observation_id: str
    candidate_key: str
    assay_subject_key: str
    proposed_candidate_id: str
    construct_candidate_id: str | None
    candidate_bridge_status: str
    reader_design_id: str
    reader_experiment_id: str
    reader_artifact_ref: str
    reader_artifact_record_id: str
    reader_artifact_content_digest: str
    experiment_id: str
    assay_id: str
    payload_program_id: str
    batch_id: str
    metric_id: str
    reporter_plasmid_id: str
    readout_kind: str
    report_time_h: float
    endpoint_time_h: float
    iptg_doses_uM: tuple[float, ...]
    y_derepression_by_dose: tuple[float, ...]
    viability_by_dose: tuple[float, ...]
    replicate_count: int
    replicate_count_min: int
    uncertainty: object | None
    assay_metadata: dict[str, object]
    rfp_over_od600_baseline: float
    rfp_over_od600_positive: float
    positive_control_atc_nM: float
    spop_potency: float
    spop_viability: float
    spop_score: float
    spop_score_raw: float
    spop_score_calibrated: float | None
    raw_value: float
    normalized_value: float
    normalization_basis: str
    construct_promotable: bool
    qc_flags: tuple[str, ...]
    status: str = "ok"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReaderSpopCandidateSummary:
    candidate_key: str
    assay_subject_key: str
    proposed_candidate_id: str
    construct_candidate_id: str | None
    construct_promotable: bool
    candidate_bridge_status: str
    observation_count: int
    experiment_ids: tuple[str, ...]
    spop_score_median: float
    spop_score_min: float
    spop_score_max: float
    dose_counts: tuple[int, ...]
    qc_flags: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ReaderSpopPlan:
    reader_root: str
    metric_id: str
    observations: tuple[ReaderSpopObservation, ...]
    candidate_summaries: tuple[ReaderSpopCandidateSummary, ...]
    issues: tuple[ReaderSpopIssue, ...]

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, object]:
        return {
            "reader_root": self.reader_root,
            "metric_id": self.metric_id,
            "ok": self.ok,
            "observations": [row.to_dict() for row in self.observations],
            "candidate_summaries": [row.to_dict() for row in self.candidate_summaries],
            "issues": [issue.to_dict() for issue in self.issues],
        }


def build_reader_spop_plan(
    *,
    reader_root: Path | None = None,
    experiment_ids: Sequence[str] = DEFAULT_READER_EXPERIMENT_IDS,
    lambda_viability: float = DEFAULT_LAMBDA,
    strict: bool = False,
) -> ReaderSpopPlan:
    """Build a dry-run SPOP label plan from sibling Reader retron experiments."""
    if not math.isfinite(lambda_viability) or not 0.0 <= lambda_viability <= 1.0:
        raise ReaderSpopContractError("lambda_viability must be finite and in [0, 1].")
    resolved_reader_root = _resolve_reader_root(reader_root)
    observations: list[ReaderSpopObservation] = []
    issues: list[ReaderSpopIssue] = []
    for experiment_id in experiment_ids:
        experiment_dir = _find_experiment_dir(resolved_reader_root, experiment_id)
        if experiment_dir is None:
            issues.append(
                ReaderSpopIssue(
                    experiment_id=experiment_id,
                    code="reader_experiment_missing",
                    message=f"{experiment_id}: Reader experiment directory is absent under {resolved_reader_root}",
                )
            )
            continue
        try:
            experiment_observations, experiment_issues = _read_experiment_observations(
                experiment_dir=experiment_dir,
                lambda_viability=lambda_viability,
            )
        except ReaderSpopContractError:
            raise
        observations.extend(experiment_observations)
        issues.extend(experiment_issues)
    ordered_observations = tuple(sorted(observations, key=lambda row: (row.experiment_id, row.candidate_key)))
    plan = ReaderSpopPlan(
        reader_root=str(resolved_reader_root),
        metric_id=SPOP_METRIC_ID,
        observations=ordered_observations,
        candidate_summaries=_candidate_summaries(ordered_observations),
        issues=tuple(issues),
    )
    if strict and not plan.ok:
        codes = ", ".join(sorted({issue.code for issue in plan.issues if issue.severity == "error"}))
        raise ReaderSpopContractError(f"Reader SPOP plan has blocking issue(s): {codes}")
    return plan


def _read_experiment_observations(
    *,
    experiment_dir: Path,
    lambda_viability: float,
) -> tuple[list[ReaderSpopObservation], list[ReaderSpopIssue]]:
    config = _load_experiment_config(experiment_dir)
    experiment_id = str(_mapping(config.get("experiment"), label="experiment").get("id") or experiment_dir.name)
    report_time, time_tolerance = _validate_reader_config(config, experiment_id=experiment_id)
    ratio_artifact = _resolve_reader_ratio_artifact(experiment_dir, experiment_id=experiment_id)
    rows = _read_ratio_rows(ratio_artifact.path, experiment_id=experiment_id)
    retron_rows = [row for row in rows if _candidate_key_for_design(row.get("design_id")) is not None]
    _validate_treatments(retron_rows, experiment_id=experiment_id)
    if not retron_rows:
        return [], [
            ReaderSpopIssue(
                experiment_id=experiment_id,
                code="no_retron_reporter_rows",
                message=f"{experiment_id}: no pES-retron plus {REPORTER_PLASMID_ID} rows were found.",
            )
        ]
    endpoint = _select_endpoint(
        retron_rows,
        experiment_id=experiment_id,
        report_time=report_time,
        tolerance=time_tolerance,
    )
    if endpoint is None:
        return [], [
            ReaderSpopIssue(
                experiment_id=experiment_id,
                code="configured_endpoint_time_absent",
                message=(
                    f"{experiment_id}: no retron reporter rows are within {time_tolerance:g} h of configured "
                    f"report_time={report_time:g}."
                ),
            )
        ]

    endpoint_rows = [row for row in retron_rows if _float(row.get("time"), context="time") == endpoint.row_time_h]
    grouped = _group_endpoint_values(endpoint_rows, experiment_id=experiment_id)
    observations: list[ReaderSpopObservation] = []
    issues: list[ReaderSpopIssue] = []
    for reader_design_id in sorted(grouped):
        candidate_key = _candidate_key_for_design(reader_design_id)
        excluded_reason = _excluded_assay_subject_reason(experiment_id=experiment_id, candidate_key=candidate_key)
        if excluded_reason is not None:
            issues.append(
                ReaderSpopIssue(
                    experiment_id=experiment_id,
                    code="assay_subject_excluded_no_strain",
                    message=f"{experiment_id}/{reader_design_id}: omitted from SPOP labels because {excluded_reason}.",
                    severity="warning",
                )
            )
            continue
        try:
            observations.append(
                _score_design(
                    experiment_id=experiment_id,
                    reader_design_id=reader_design_id,
                    ratio_artifact=ratio_artifact,
                    report_time=endpoint.report_time_h,
                    endpoint_time=endpoint.endpoint_time_h,
                    values=grouped[reader_design_id],
                    lambda_viability=lambda_viability,
                    assay_metadata=endpoint.assay_metadata,
                    qc_flags=endpoint.qc_flags,
                )
            )
        except _CandidateCannotScore as exc:
            issues.append(
                ReaderSpopIssue(
                    experiment_id=experiment_id,
                    code=exc.code,
                    message=f"{experiment_id}/{reader_design_id}: {exc}",
                )
            )
    return observations, issues


def _score_design(
    *,
    experiment_id: str,
    reader_design_id: str,
    ratio_artifact: ReaderArtifactRef,
    report_time: float,
    endpoint_time: float,
    values: Mapping[tuple[float, float, str], tuple[float, int]],
    lambda_viability: float,
    assay_metadata: Mapping[str, object] | None = None,
    qc_flags: tuple[str, ...] = (),
) -> ReaderSpopObservation:
    candidate_key = _candidate_key_for_design(reader_design_id)
    if candidate_key is None:
        raise _CandidateCannotScore("non_retron_candidate", "reader design id is not a retron reporter row")
    zero_key = (0.0, 0.0, REPORTER_READOUT)
    zero_od_key = (0.0, 0.0, VIABILITY_READOUT)
    if zero_key not in values:
        raise _CandidateCannotScore("zero_inducer_baseline_missing", "missing 0 aTc / 0 IPTG RFP/OD600 baseline")
    if zero_od_key not in values:
        raise _CandidateCannotScore("zero_inducer_od_missing", "missing 0 aTc / 0 IPTG OD600 baseline")
    positive_keys = sorted(key for key in values if key[0] > 0.0 and key[1] == 0.0 and key[2] == REPORTER_READOUT)
    if not positive_keys:
        raise _CandidateCannotScore("positive_control_missing", "missing aTc positive-control RFP/OD600 row")
    positive_atc = positive_keys[-1][0]
    dose_keys = sorted(key for key in values if key[0] == 0.0 and key[1] > 0.0 and key[2] == REPORTER_READOUT)
    if not dose_keys:
        raise _CandidateCannotScore("iptg_dose_rows_missing", "missing nonzero IPTG RFP/OD600 rows")

    baseline_z, baseline_n = values[zero_key]
    baseline_od, _baseline_od_n = values[zero_od_key]
    positive_z, positive_n = values[(positive_atc, 0.0, REPORTER_READOUT)]
    positive_denominator = positive_z - baseline_z
    if positive_denominator <= _EPS_POSITIVE:
        raise _CandidateCannotScore(
            "positive_control_not_above_baseline",
            "aTc positive-control RFP/OD600 is not above zero-inducer baseline",
        )
    if baseline_od <= 0.0:
        raise _CandidateCannotScore("baseline_od_not_positive", "zero-inducer OD600 baseline must be positive")

    y_values: list[float] = []
    viability_values: list[float] = []
    doses: list[float] = []
    replicate_counts = [baseline_n, positive_n]
    resolved_qc_flags = set(qc_flags)
    for atc_n_m, iptg_u_m, _channel in dose_keys:
        dose_z, dose_n = values[(atc_n_m, iptg_u_m, REPORTER_READOUT)]
        od_key = (atc_n_m, iptg_u_m, VIABILITY_READOUT)
        if od_key not in values:
            raise _CandidateCannotScore("dose_od_missing", f"missing OD600 for IPTG dose {iptg_u_m:g} uM")
        dose_od, dose_od_n = values[od_key]
        y = (dose_z - baseline_z) / positive_denominator
        viability = min(1.0, dose_od / baseline_od)
        if y > 1.0:
            resolved_qc_flags.add("derepression_exceeds_atc_positive")
        if y < 0.0:
            resolved_qc_flags.add("derepression_below_zero_inducer")
        if viability < 0.8:
            resolved_qc_flags.add("induction_growth_penalty")
        doses.append(iptg_u_m)
        y_values.append(y)
        viability_values.append(viability)
        replicate_counts.extend([dose_n, dose_od_n])
    if len(doses) == 1:
        resolved_qc_flags.add("single_dose_endpoint")

    raw_potency = statistics.fmean(y_values)
    potency = statistics.fmean(max(0.0, value) for value in y_values)
    viability_mean = statistics.fmean(viability_values)
    raw_score = raw_potency * ((1.0 - lambda_viability) + (lambda_viability * viability_mean))
    score = potency * ((1.0 - lambda_viability) + (lambda_viability * viability_mean))
    resolved_construct_candidates = _resolved_construct_candidates()
    proposed_candidate_id = _proposed_candidate_id_for_key(candidate_key)
    construct_candidate_id = resolved_construct_candidates.get(candidate_key)
    construct_promotable = construct_candidate_id is not None
    candidate_bridge_status = (
        "resolved_construct_sequence_authority" if construct_promotable else "missing_construct_sequence_authority"
    )
    if not construct_promotable:
        resolved_qc_flags.add("construct_sequence_authority_missing")
    resolved_assay_metadata = {
        "baseline_condition": "0 nm aTc; 0 uM IPTG",
        "positive_control_condition": f"{positive_atc:g} nm aTc; 0 uM IPTG",
        "lambda_viability": lambda_viability,
    }
    resolved_assay_metadata.update(dict(assay_metadata or {}))
    return ReaderSpopObservation(
        observation_id=f"reader:{experiment_id}:{candidate_key}:{SPOP_METRIC_ID}",
        candidate_key=candidate_key,
        assay_subject_key=candidate_key,
        proposed_candidate_id=proposed_candidate_id,
        construct_candidate_id=construct_candidate_id,
        candidate_bridge_status=candidate_bridge_status,
        reader_design_id=reader_design_id,
        reader_experiment_id=experiment_id,
        reader_artifact_ref=ratio_artifact.ref,
        reader_artifact_record_id=ratio_artifact.record_id,
        reader_artifact_content_digest=ratio_artifact.content_digest,
        experiment_id=experiment_id,
        assay_id=f"{experiment_id}::{REPORTER_PLASMID_ID}::{REPORTER_READOUT}",
        payload_program_id="rt_lnrna_sponging",
        batch_id=experiment_id,
        metric_id=SPOP_METRIC_ID,
        reporter_plasmid_id=REPORTER_PLASMID_ID,
        readout_kind=SPOP_METRIC_ID,
        report_time_h=report_time,
        endpoint_time_h=endpoint_time,
        iptg_doses_uM=tuple(doses),
        y_derepression_by_dose=tuple(y_values),
        viability_by_dose=tuple(viability_values),
        replicate_count=min(replicate_counts),
        replicate_count_min=min(replicate_counts),
        uncertainty=None,
        assay_metadata=resolved_assay_metadata,
        rfp_over_od600_baseline=baseline_z,
        rfp_over_od600_positive=positive_z,
        positive_control_atc_nM=positive_atc,
        spop_potency=potency,
        spop_viability=viability_mean,
        spop_score=score,
        spop_score_raw=raw_score,
        spop_score_calibrated=None,
        raw_value=raw_score,
        normalized_value=score,
        normalization_basis="rfp_od600_derepression_fraction_relative_to_atc_positive_control",
        construct_promotable=construct_promotable,
        qc_flags=tuple(sorted(resolved_qc_flags)),
    )


def _candidate_summaries(observations: Iterable[ReaderSpopObservation]) -> tuple[ReaderSpopCandidateSummary, ...]:
    grouped: dict[str, list[ReaderSpopObservation]] = {}
    for observation in observations:
        if observation.status == "ok":
            grouped.setdefault(observation.candidate_key, []).append(observation)
    summaries: list[ReaderSpopCandidateSummary] = []
    for candidate_key, rows in grouped.items():
        scores = [row.spop_score for row in rows]
        flags = sorted({flag for row in rows for flag in row.qc_flags})
        dose_counts = tuple(sorted({len(row.iptg_doses_uM) for row in rows}))
        first = rows[0]
        summaries.append(
            ReaderSpopCandidateSummary(
                candidate_key=candidate_key,
                assay_subject_key=first.assay_subject_key,
                proposed_candidate_id=first.proposed_candidate_id,
                construct_candidate_id=first.construct_candidate_id,
                construct_promotable=first.construct_promotable,
                candidate_bridge_status=first.candidate_bridge_status,
                observation_count=len(rows),
                experiment_ids=tuple(sorted(row.experiment_id for row in rows)),
                spop_score_median=float(statistics.median(scores)),
                spop_score_min=float(min(scores)),
                spop_score_max=float(max(scores)),
                dose_counts=dose_counts,
                qc_flags=tuple(flags),
            )
        )
    return tuple(sorted(summaries, key=lambda row: row.candidate_key))


class _CandidateCannotScore(ValueError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _resolve_reader_ratio_artifact(experiment_dir: Path, *, experiment_id: str) -> ReaderArtifactRef:
    manifest_path = experiment_dir / "outputs" / "manifests" / "records.json"
    if not manifest_path.exists():
        raise ReaderSpopContractError(f"{experiment_id}: Reader outputs/manifests/records.json is required")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = _mapping(payload, label=f"{experiment_id}.records.json")
    latest = _mapping(manifest.get("latest"), label=f"{experiment_id}.records.json.latest")
    record = _mapping(latest.get(READER_RATIO_RECORD_ID), label=f"{experiment_id}.latest.{READER_RATIO_RECORD_ID}")
    record_id = str(record.get("record_id") or "").strip()
    if record_id != READER_RATIO_RECORD_ID:
        raise ReaderSpopContractError(
            f"{experiment_id}: latest {READER_RATIO_RECORD_ID} record_id must be {READER_RATIO_RECORD_ID!r}"
        )
    relative_path = str(record.get("path") or "").strip()
    if not relative_path:
        raise ReaderSpopContractError(f"{experiment_id}: {READER_RATIO_RECORD_ID} missing relative artifact path")
    if Path(relative_path).is_absolute():
        raise ReaderSpopContractError(f"{experiment_id}: {READER_RATIO_RECORD_ID} path must be outputs-relative")
    artifact_path = experiment_dir / "outputs" / relative_path
    if not artifact_path.exists():
        raise ReaderSpopContractError(f"{experiment_id}: missing Reader ratio artifact {artifact_path}")
    content_digest = str(record.get("content_digest") or "").strip()
    if not content_digest:
        raise ReaderSpopContractError(f"{experiment_id}: {READER_RATIO_RECORD_ID} missing content_digest")
    return ReaderArtifactRef(
        experiment_id=experiment_id,
        record_id=record_id,
        path=artifact_path,
        manifest_path=manifest_path,
        content_digest=content_digest,
    )


def _read_ratio_rows(path: Path, *, experiment_id: str) -> list[dict[str, object]]:
    table = pq.read_table(path)
    missing = [column for column in _REQUIRED_RATIO_COLUMNS if column not in table.schema.names]
    if missing:
        raise ReaderSpopContractError(f"{experiment_id}: Reader ratio artifact missing required column(s): {missing}")
    return table.to_pylist()


def _load_experiment_config(experiment_dir: Path) -> dict[str, object]:
    config_path = experiment_dir / "config.yaml"
    if not config_path.exists():
        raise ReaderSpopContractError(f"{experiment_dir.name}: missing Reader config.yaml")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ReaderSpopContractError(f"{experiment_dir.name}: Reader config.yaml must be a mapping")
    return payload


def _validate_reader_config(config: Mapping[str, object], *, experiment_id: str) -> tuple[float, float]:
    protocol = _mapping(config.get("protocol"), label=f"{experiment_id}.protocol")
    if str(protocol.get("id") or "").strip() != "plate_reader/single_reporter_screen":
        raise ReaderSpopContractError(f"{experiment_id}: protocol.id must be plate_reader/single_reporter_screen")
    analysis = _mapping(protocol.get("analysis"), label=f"{experiment_id}.protocol.analysis")
    if str(analysis.get("reporter_channel") or "").strip() != "RFP":
        raise ReaderSpopContractError(f"{experiment_id}: protocol.analysis.reporter_channel must be RFP")
    inputs = _mapping(protocol.get("inputs"), label=f"{experiment_id}.protocol.inputs")
    fold_change = _mapping(inputs.get("fold_change"), label=f"{experiment_id}.protocol.inputs.fold_change")
    report_times = fold_change.get("report_times")
    if not isinstance(report_times, list) or len(report_times) != 1:
        raise ReaderSpopContractError(f"{experiment_id}: fold_change.report_times must contain exactly one endpoint")
    report_time = _float(report_times[0], context=f"{experiment_id}.report_times[0]")
    time_tolerance = _float(fold_change.get("time_tolerance", 0.51), context=f"{experiment_id}.time_tolerance")
    if time_tolerance < 0.0:
        raise ReaderSpopContractError(f"{experiment_id}: time_tolerance must be non-negative")
    if str(fold_change.get("treatment_column") or "").strip() != "treatment":
        raise ReaderSpopContractError(f"{experiment_id}: fold_change.treatment_column must be treatment")
    if fold_change.get("global_baseline_value") != "0 nm aTc; 0 uM IPTG":
        raise ReaderSpopContractError(f"{experiment_id}: global_baseline_value must be '0 nm aTc; 0 uM IPTG'")
    return report_time, time_tolerance


def _validate_treatments(rows: Iterable[Mapping[str, object]], *, experiment_id: str) -> None:
    for row in rows:
        treatment = str(row.get("treatment") or "").strip()
        if _parse_treatment(treatment) is None:
            design_id = str(row.get("design_id") or "").strip()
            raise ReaderSpopContractError(
                f"{experiment_id}: malformed treatment {treatment!r} for retron reporter row {design_id!r}"
            )


def _group_endpoint_values(
    rows: Iterable[Mapping[str, object]],
    *,
    experiment_id: str,
) -> dict[str, dict[tuple[float, float, str], tuple[float, int]]]:
    grouped_values: dict[str, dict[tuple[float, float, str], list[float]]] = {}
    grouped_positions: dict[str, dict[tuple[float, float, str], set[str]]] = {}
    for row in rows:
        design_id = str(row.get("design_id") or "").strip()
        treatment = str(row.get("treatment") or "").strip()
        parsed = _parse_treatment(treatment)
        if parsed is None:
            raise ReaderSpopContractError(f"{experiment_id}: malformed treatment {treatment!r}")
        channel = str(row.get("channel") or "").strip()
        if channel not in {REPORTER_READOUT, VIABILITY_READOUT}:
            continue
        value = _float(row.get("value"), context=f"{experiment_id}/{design_id}/{treatment}/{channel}")
        if value < 0.0:
            raise ReaderSpopContractError(f"{experiment_id}: {channel} value must be non-negative for {design_id!r}")
        key = (parsed[0], parsed[1], channel)
        grouped_values.setdefault(design_id, {}).setdefault(key, []).append(value)
        grouped_positions.setdefault(design_id, {}).setdefault(key, set()).add(str(row.get("position") or ""))
    out: dict[str, dict[tuple[float, float, str], tuple[float, int]]] = {}
    for design_id, design_values in grouped_values.items():
        out[design_id] = {
            key: (float(statistics.median(values)), len(grouped_positions[design_id][key]))
            for key, values in design_values.items()
        }
    return out


def _select_endpoint_time(
    rows: Sequence[Mapping[str, object]],
    *,
    report_time: float,
    tolerance: float,
) -> float | None:
    times = sorted({_float(row.get("time"), context="time") for row in rows})
    if not times:
        return None
    nearest = min(times, key=lambda value: (abs(value - report_time), value))
    if abs(nearest - report_time) > tolerance:
        return None
    return nearest


def _select_endpoint(
    rows: Sequence[Mapping[str, object]],
    *,
    experiment_id: str,
    report_time: float,
    tolerance: float,
) -> _EndpointSelection | None:
    endpoint_time = _select_endpoint_time(rows, report_time=report_time, tolerance=tolerance)
    if endpoint_time is not None:
        return _EndpointSelection(
            row_time_h=endpoint_time,
            report_time_h=report_time,
            endpoint_time_h=endpoint_time,
            assay_metadata={"endpoint_time_basis": "configured_report_time"},
        )
    override = _SINGLE_POINT_ENDPOINT_OVERRIDES.get(experiment_id)
    if override is None:
        return None
    artifact_time = _float(override["reader_artifact_time_h"], context=f"{experiment_id}.reader_artifact_time_h")
    times = sorted({_float(row.get("time"), context="time") for row in rows})
    if artifact_time not in times:
        return None
    elapsed_time = _float(override["elapsed_time_h"], context=f"{experiment_id}.elapsed_time_h")
    return _EndpointSelection(
        row_time_h=artifact_time,
        report_time_h=elapsed_time,
        endpoint_time_h=elapsed_time,
        assay_metadata={
            "endpoint_time_basis": str(override["basis"]),
            "endpoint_time_caveat": str(override["caveat"]),
            "reader_artifact_time_h": artifact_time,
            "configured_report_time_h": report_time,
        },
        qc_flags=("single_point_endpoint_time_override",),
    )


def _excluded_assay_subject_reason(*, experiment_id: str, candidate_key: str | None) -> str | None:
    if candidate_key is None:
        return None
    return _EXCLUDED_ASSAY_SUBJECTS.get((experiment_id, candidate_key))


def _parse_treatment(value: str) -> tuple[float, float] | None:
    match = _TREATMENT_RE.match(value)
    if match is None:
        return None
    return float(match.group("atc")), float(match.group("iptg"))


def _candidate_key_for_design(value: object) -> str | None:
    text = str(value or "").strip()
    match = _RETRON_DESIGN_RE.search(text)
    if match is None:
        return None
    return f"retron{int(match.group('number'))}"


@cache
def _resolved_construct_candidates() -> dict[str, str]:
    catalog = build_variant_genbank_catalog()
    if not catalog.ok:
        joined = "; ".join(catalog.errors)
        raise ReaderSpopContractError(f"Variant GenBank catalog is invalid: {joined}")
    resolved = dict(_BASE_RESOLVED_CONSTRUCT_CANDIDATES)
    resolved.update(
        {
            record.variant_id: record.construct_candidate_id
            for record in catalog.records
            if record.variant_id.startswith("retron") and record.construct_projection_status == "representable"
        }
    )
    return resolved


def _proposed_candidate_id_for_key(candidate_key: str) -> str:
    resolved_construct_candidates = _resolved_construct_candidates()
    if candidate_key in resolved_construct_candidates:
        return resolved_construct_candidates[candidate_key]
    suffix = candidate_key.removeprefix("retron")
    return f"rt_lnrna_pair__unresolved_rt__retron{suffix}_lnrna__tetO"


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ReaderSpopContractError(f"{label} must be a mapping")
    return value


def _float(value: object, *, context: str) -> float:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ReaderSpopContractError(f"{context} must be numeric; got {value!r}") from exc
    if not math.isfinite(numeric):
        raise ReaderSpopContractError(f"{context} must be finite; got {value!r}")
    return numeric


def _resolve_reader_root(reader_root: Path | None) -> Path:
    if reader_root is not None:
        return Path(reader_root).expanduser().resolve()
    return _resolve_repo_root().parent / "reader"


def _resolve_repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def _find_experiment_dir(reader_root: Path, experiment_id: str) -> Path | None:
    experiments_root = reader_root / "experiments"
    candidates = sorted(experiments_root.glob(f"*/*{experiment_id}*"))
    for candidate in candidates:
        if candidate.is_dir() and candidate.name == experiment_id:
            return candidate
    return None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dry-run Reader SPOP label planning for RT-lnRNA candidates.")
    parser.add_argument("--reader-root", type=Path, default=None)
    parser.add_argument("--experiment-id", action="append", dest="experiment_ids")
    parser.add_argument("--lambda-viability", type=float, default=DEFAULT_LAMBDA)
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true", help="Emit JSON. Plain text is the default.")
    args = parser.parse_args(argv)
    plan = build_reader_spop_plan(
        reader_root=args.reader_root,
        experiment_ids=tuple(args.experiment_ids or DEFAULT_READER_EXPERIMENT_IDS),
        lambda_viability=args.lambda_viability,
        strict=bool(args.strict),
    )
    if args.json:
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    else:
        print(
            f"Reader SPOP plan: ok={plan.ok} observations={len(plan.observations)} "
            f"summaries={len(plan.candidate_summaries)} issues={len(plan.issues)}"
        )
        for issue in plan.issues:
            print(f"- issue {issue.code}: {issue.message}")
    return 0 if plan.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
