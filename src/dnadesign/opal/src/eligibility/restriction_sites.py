"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/eligibility/restriction_sites.py

Restriction-site exclusion for assembled DNA candidates.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from ..core.utils import OpalError
from ..registries.eligibility import register_candidate_eligibility
from .contracts import (
    CandidateEligibilityRuleResult,
    RestrictionSiteHit,
    RestrictionSiteScanReport,
    RestrictionSiteSpec,
    _require_dna,
    params_sha256,
)


def _restriction_sites_from_params(params: Mapping[str, Any]) -> tuple[RestrictionSiteSpec, ...]:
    raw_sites = params.get("forbidden_sites")
    if not isinstance(raw_sites, Sequence) or isinstance(raw_sites, str | bytes):
        raise OpalError("restriction_site_exclusion.params.forbidden_sites must be a non-empty list")
    sites = tuple(RestrictionSiteSpec.from_mapping(site) for site in raw_sites)
    if not sites:
        raise OpalError("restriction_site_exclusion.params.forbidden_sites must be a non-empty list")
    return sites


def _region_for_span(start: int, end: int, *, left_len: int, core_len: int, final_len: int) -> str:
    core_start = int(left_len)
    core_end = core_start + int(core_len)
    if start >= 0 and end <= core_start:
        return "left_flank"
    if start >= core_start and end <= core_end:
        return "core"
    if start >= core_end and end <= final_len:
        return "right_flank"
    if start < core_start and end > core_start:
        return "left_flank_core_junction"
    if start < core_end and end > core_end:
        return "core_right_flank_junction"
    return "unknown_boundary"


def _iter_motif_starts(sequence: str, motif: str) -> list[int]:
    starts: list[int] = []
    start = sequence.find(motif)
    while start != -1:
        starts.append(int(start))
        start = sequence.find(motif, start + 1)
    return starts


def scan_restriction_sites(
    *,
    candidate_id: str,
    core_sequence: str,
    left_flank: str,
    right_flank: str,
    expected_core_length: int,
    forbidden_sites: Sequence[RestrictionSiteSpec | Mapping[str, Any]],
) -> RestrictionSiteScanReport:
    """Scan one assembled insert and report unexpected restriction-site hits."""

    candidate = str(candidate_id).strip()
    if not candidate:
        raise OpalError("restriction-site scan requires a non-empty candidate_id")
    core = _require_dna(core_sequence, field=f"{candidate}.core_sequence")
    left = _require_dna(left_flank, field="left_flank", case="lower").upper()
    right = _require_dna(right_flank, field="right_flank", case="lower").upper()
    expected = int(expected_core_length)
    if expected <= 0:
        raise OpalError("restriction-site scan expected_core_length must be positive")
    if len(core) != expected:
        raise OpalError(f"restriction-site scan candidate {candidate} expected core length {expected}, got {len(core)}")

    specs = tuple(
        site if isinstance(site, RestrictionSiteSpec) else RestrictionSiteSpec.from_mapping(site)
        for site in forbidden_sites
    )
    if not specs:
        raise OpalError("restriction-site scan requires at least one forbidden site")

    final = f"{left}{core}{right}"
    left_len = len(left)
    final_len = len(final)
    hits: list[RestrictionSiteHit] = []
    for site in specs:
        motif = site.motif
        for start in _iter_motif_starts(final, motif):
            end = start + len(motif)
            region = _region_for_span(start, end, left_len=left_len, core_len=len(core), final_len=final_len)
            allowed = region in site.allowed_regions
            hits.append(
                RestrictionSiteHit(
                    enzyme=site.enzyme,
                    motif=site.motif,
                    start_0=start,
                    end_0=end,
                    region=region,
                    allowed=allowed,
                )
            )
    return RestrictionSiteScanReport(candidate_id=candidate, final_length=final_len, hits=tuple(hits))


def _unexpected_summary(report: RestrictionSiteScanReport) -> str:
    return ";".join(
        f"{hit.enzyme}:{hit.motif}@{hit.start_0}-{hit.end_0}:{hit.region}" for hit in report.unexpected_hits
    )


@register_candidate_eligibility("restriction_site_exclusion")
def restriction_site_exclusion(*, frame: pd.DataFrame, params: Mapping[str, Any]) -> CandidateEligibilityRuleResult:
    """Exclude rows whose assembled insert contains non-designated restriction sites."""

    if not isinstance(frame, pd.DataFrame):
        raise OpalError("restriction_site_exclusion expects a pandas DataFrame")
    if frame.empty:
        raise OpalError("restriction_site_exclusion received an empty candidate frame")
    on_violation = str(params.get("on_violation", "exclude")).strip()
    if on_violation != "exclude":
        raise OpalError("restriction_site_exclusion.params.on_violation currently supports only 'exclude'")
    scan_space = str(params.get("scan_space", "final_assembled_insert")).strip()
    if scan_space != "final_assembled_insert":
        raise OpalError("restriction_site_exclusion.params.scan_space must be 'final_assembled_insert'")
    sequence_column = str(params.get("sequence_column", "sequence")).strip()
    if not sequence_column:
        raise OpalError("restriction_site_exclusion.params.sequence_column must be non-empty")
    if sequence_column not in frame.columns:
        raise OpalError(f"restriction_site_exclusion missing sequence column {sequence_column!r}")
    if "id" not in frame.columns:
        raise OpalError("restriction_site_exclusion requires candidate frame column 'id'")

    left_flank = str(params.get("left_flank", ""))
    right_flank = str(params.get("right_flank", ""))
    expected_core_length = int(params.get("expected_core_length", 0))
    sites = _restriction_sites_from_params(params)

    keep_mask: list[bool] = []
    violation_rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        candidate_id = str(row["id"])
        report = scan_restriction_sites(
            candidate_id=candidate_id,
            core_sequence=str(row[sequence_column]),
            left_flank=left_flank,
            right_flank=right_flank,
            expected_core_length=expected_core_length,
            forbidden_sites=sites,
        )
        unexpected = report.unexpected_hits
        keep_mask.append(not unexpected)
        if unexpected:
            violation_rows.append(
                {
                    "id": candidate_id,
                    "unexpected_site_count": int(len(unexpected)),
                    "unexpected_sites": _unexpected_summary(report),
                }
            )

    filtered = frame.loc[keep_mask].copy().reset_index(drop=True)
    min_remaining_raw = params.get("min_remaining_candidates")
    if min_remaining_raw is not None:
        min_remaining = int(min_remaining_raw)
        if len(filtered) < min_remaining:
            raise OpalError(
                "restriction_site_exclusion produced too few eligible candidates: "
                f"remaining={len(filtered)} min_remaining_candidates={min_remaining}"
            )
    else:
        min_remaining = None
    if filtered.empty:
        raise OpalError("restriction_site_exclusion excluded every candidate")

    report = {
        "rule": "restriction_site_exclusion",
        "assembly_strategy_ref": str(params.get("assembly_strategy_ref", "")),
        "params_sha256": params_sha256(params),
        "input_rows": int(len(frame)),
        "output_rows": int(len(filtered)),
        "excluded_rows": int(len(frame) - len(filtered)),
        "min_remaining_candidates": min_remaining,
        "violation_preview": violation_rows[:10],
    }
    return CandidateEligibilityRuleResult(frame=filtered, report=report)
