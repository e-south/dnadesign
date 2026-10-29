"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/eligibility/candidate_ids.py

Generic candidate-ID exclusion before model scoring and selection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from ..core.utils import OpalError
from ..registries.eligibility import register_candidate_eligibility
from .contracts import CandidateEligibilityRuleResult, params_sha256


def _required_candidate_columns(params: Mapping[str, Any]) -> tuple[str, ...]:
    _ = params
    return ("id",)


@register_candidate_eligibility("candidate_id_exclusion", required_columns=_required_candidate_columns)
def candidate_id_exclusion(*, frame: pd.DataFrame, params: Mapping[str, Any]) -> CandidateEligibilityRuleResult:
    """Exclude an exact, reasoned candidate-ID set and fail on stale identities."""

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise OpalError("candidate_id_exclusion expects a non-empty pandas DataFrame")
    if "id" not in frame.columns:
        raise OpalError("candidate_id_exclusion requires candidate frame column 'id'")
    frame_ids = frame["id"].astype(str)
    if frame_ids.str.strip().eq("").any() or frame_ids.duplicated().any():
        raise OpalError("candidate_id_exclusion requires unique, non-empty candidate IDs")
    exclusion_set_id = str(params.get("exclusion_set_id", "")).strip()
    if not exclusion_set_id:
        raise OpalError("candidate_id_exclusion.params.exclusion_set_id must be non-empty")
    entries = _entries(params.get("entries"))
    configured_ids = {candidate_id for candidate_id, _ in entries}
    unknown = sorted(configured_ids - set(frame_ids))
    if unknown:
        raise OpalError(f"candidate_id_exclusion contains unknown candidate IDs: {unknown[:10]}")
    excluded = frame_ids.isin(configured_ids)
    filtered = frame.loc[~excluded].copy().reset_index(drop=True)
    minimum = _minimum_remaining(params.get("min_remaining_candidates"))
    if len(filtered) < minimum:
        raise OpalError(
            "candidate_id_exclusion produced too few eligible candidates: "
            f"remaining={len(filtered)} min_remaining_candidates={minimum}"
        )
    reasons = Counter(reason for _, reason in entries)
    return CandidateEligibilityRuleResult(
        frame=filtered,
        report={
            "rule": "candidate_id_exclusion",
            "exclusion_set_id": exclusion_set_id,
            "params_sha256": params_sha256(params),
            "input_rows": int(len(frame)),
            "output_rows": int(len(filtered)),
            "excluded_rows": int(excluded.sum()),
            "min_remaining_candidates": minimum,
            "reason_counts": dict(sorted(reasons.items())),
            "exclusion_preview": [
                {"candidate_id": candidate_id, "reason": reason} for candidate_id, reason in entries[:10]
            ],
        },
    )


def _entries(value: object) -> list[tuple[str, str]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes) or not value:
        raise OpalError("candidate_id_exclusion.params.entries must contain at least one entry")
    rows: list[tuple[str, str]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != {"candidate_id", "reason"}:
            raise OpalError(f"candidate_id_exclusion.params.entries[{index}] must contain candidate_id and reason")
        candidate_id = str(raw["candidate_id"]).strip()
        reason = str(raw["reason"]).strip()
        if not candidate_id or not reason:
            raise OpalError("candidate_id_exclusion entry candidate_id and reason must be non-empty")
        rows.append((candidate_id, reason))
    ids = [candidate_id for candidate_id, _ in rows]
    if len(ids) != len(set(ids)):
        raise OpalError("candidate_id_exclusion entries contain duplicate candidate IDs")
    return rows


def _minimum_remaining(value: object) -> int:
    if isinstance(value, bool):
        raise OpalError("candidate_id_exclusion.params.min_remaining_candidates must be positive")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise OpalError("candidate_id_exclusion.params.min_remaining_candidates must be positive") from exc
    if result < 1:
        raise OpalError("candidate_id_exclusion.params.min_remaining_candidates must be positive")
    return result


__all__ = ["candidate_id_exclusion"]
