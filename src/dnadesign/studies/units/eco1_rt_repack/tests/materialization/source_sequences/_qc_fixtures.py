"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/source_sequences/_qc_fixtures.py

Sequence-QC fixture helpers for Eco1 source-sequence tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import Any


def included_source_row(
    *,
    profile_id: str,
    record_id: str,
    provider_id: str,
    accession: str,
    target: str,
    sequence: str,
    omit_sequence_qc: bool,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "profile_id": profile_id,
        "record_id": record_id,
        "provider_id": provider_id,
        "accession": accession,
        "status": "included",
    }
    if not omit_sequence_qc:
        row["sequence_qc"] = sequence_qc(target=target, sequence=sequence)
    return row


def sequence_qc(*, target: str, sequence: str) -> dict[str, Any]:
    identity = sum(left == right for left, right in zip(target, sequence, strict=True)) / len(target)
    return {
        "method_id": "eco1_roster_cache_sequence_qc_v1",
        "target_sequence_hash": "fixture-target-hash",
        "sequence_length_aa": len(sequence),
        "query_coverage": 1.0,
        "pairwise_identity_to_target": round(identity, 6),
        "identity_range_status": "within_declared_range",
        "length_status": "within_declared_range",
        "query_coverage_status": "meets_declared_minimum",
        "motif_qc_markers": {
            "rt_catalytic_dd_or_yadd_like_region": "present",
            "retron_x_naxxH_like_motif": "present",
            "retron_y_vtg_like_motif": "present",
        },
        "hard_reject_filters_triggered": [],
    }
