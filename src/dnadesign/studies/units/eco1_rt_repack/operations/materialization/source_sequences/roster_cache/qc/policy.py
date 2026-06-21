"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/source_sequences/roster_cache/qc/policy.py

Sequence-QC policy for Mestre-derived Eco1 conservation source records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .metrics import pairwise_target_metrics, range_status
from .motifs import call_motif_markers

_QC_METHOD_ID = "eco1_roster_cache_sequence_qc_v1"


@dataclass(frozen=True)
class SequenceQcResult:
    """Sequence-derived QC metadata for one provider-backed roster row."""

    method_id: str
    target_sequence_hash: str
    sequence_length_aa: int
    query_coverage: float
    pairwise_identity_to_target: float
    identity_range_status: str
    length_status: str
    query_coverage_status: str
    motif_qc_markers: Mapping[str, str]
    hard_reject_filters_triggered: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.hard_reject_filters_triggered

    @property
    def exclusion_reason(self) -> str:
        return "failed_sequence_qc:" + ",".join(self.hard_reject_filters_triggered)

    def to_yaml_row(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "target_sequence_hash": self.target_sequence_hash,
            "sequence_length_aa": self.sequence_length_aa,
            "query_coverage": round(self.query_coverage, 6),
            "pairwise_identity_to_target": round(self.pairwise_identity_to_target, 6),
            "identity_range_status": self.identity_range_status,
            "length_status": self.length_status,
            "query_coverage_status": self.query_coverage_status,
            "motif_qc_markers": dict(self.motif_qc_markers),
            "hard_reject_filters_triggered": list(self.hard_reject_filters_triggered),
        }


def evaluate_sequence_qc(
    *,
    sequence: str,
    target_sequence: str,
    target_sequence_hash: str,
    source_group: Mapping[str, Any],
) -> SequenceQcResult:
    """Evaluate one provider sequence against the selected source-group policy."""

    selection_rule = _require_mapping(source_group.get("selection_rule"), "selection_rule")
    coverage_minimum = _require_float(selection_rule.get("query_coverage_minimum"), "query_coverage_minimum")
    identity_min, identity_max = _two_float_range(selection_rule.get("identity_range"), "identity_range")
    length_min, length_max = _two_float_range(selection_rule.get("length_range_aa"), "length_range_aa")
    hard_reject_filters = set(_string_list(selection_rule.get("hard_reject_filters"), "hard_reject_filters"))
    motif_markers = _string_list(selection_rule.get("motif_qc_markers"), "motif_qc_markers")

    normalized_sequence = sequence.upper()
    metrics = pairwise_target_metrics(sequence=normalized_sequence, target_sequence=target_sequence)
    coverage = metrics.query_coverage
    identity = metrics.identity_to_target
    identity_status = range_status(identity, identity_min, identity_max)
    length = len(normalized_sequence)
    length_status = range_status(float(length), length_min, length_max)
    coverage_status = "below_declared_minimum" if coverage < coverage_minimum else "meets_declared_minimum"
    motif_calls = call_motif_markers(normalized_sequence, motif_markers)

    triggered = _hard_rejects(
        coverage=coverage,
        coverage_minimum=coverage_minimum,
        identity_status=identity_status,
        length_status=length_status,
        motif_calls=motif_calls,
        hard_reject_filters=hard_reject_filters,
    )
    return SequenceQcResult(
        method_id=_QC_METHOD_ID,
        target_sequence_hash=target_sequence_hash,
        sequence_length_aa=length,
        query_coverage=coverage,
        pairwise_identity_to_target=identity,
        identity_range_status=identity_status,
        length_status=length_status,
        query_coverage_status=coverage_status,
        motif_qc_markers=motif_calls,
        hard_reject_filters_triggered=tuple(triggered),
    )


def _hard_rejects(
    *,
    coverage: float,
    coverage_minimum: float,
    identity_status: str,
    length_status: str,
    motif_calls: Mapping[str, str],
    hard_reject_filters: set[str],
) -> list[str]:
    triggered: list[str] = []
    if coverage < coverage_minimum and "below_query_coverage_minimum" in hard_reject_filters:
        triggered.append("below_query_coverage_minimum")
    if identity_status != "within_declared_range" and "outside_identity_range" in hard_reject_filters:
        triggered.append("outside_identity_range")
    if length_status != "within_declared_range" and "outside_length_range" in hard_reject_filters:
        triggered.append("outside_length_range")
    if length_status == "below_declared_range" and "obvious_fragment" in hard_reject_filters:
        triggered.append("obvious_fragment")
    if length_status == "above_declared_range" and "unresolved_long_fusion" in hard_reject_filters:
        triggered.append("unresolved_long_fusion")
    if (
        motif_calls.get("rt_catalytic_dd_or_yadd_like_region") == "absent"
        and "missing_catalytic_rt_core" in hard_reject_filters
    ):
        triggered.append("missing_catalytic_rt_core")
    return triggered


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"{name} must be a non-empty list of strings")
    return list(value)


def _require_float(value: Any, name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _two_float_range(value: Any, name: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must be a two-value range")
    lower = _require_float(value[0], f"{name}[0]")
    upper = _require_float(value[1], f"{name}[1]")
    if lower > upper:
        raise ValueError(f"{name} lower bound must be <= upper bound")
    return lower, upper
