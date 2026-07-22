"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_window_label_promotion/exclusions.py

Bind study-owned observation exclusions to OPAL campaign eligibility.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .contracts import ResponseWindowLabelPromotionError

CANDIDATE_EXCLUSION_SET_ID = "stress_response_window_observation_dispositions_v1"
_AUTHORITY = "study_observation_bundle"
_DERIVATION = "contribution_candidates_absent_from_observations"
_SOURCE_RECORD = "contributions"
_PROVENANCE_FIELDS = {
    "authority",
    "derivation",
    "entry_count",
    "entries",
    "exclusion_set_id",
    "source_record",
}


def derive_candidate_selection_exclusions(
    observations: pd.DataFrame,
    contributions: pd.DataFrame,
) -> list[dict[str, str]]:
    """Derive one reasoned exclusion for every measured candidate without a label."""

    required = {
        "candidate_id",
        "selected_as_label_source",
        "included_in_label",
        "label_exclusion_reason",
    }
    if missing := sorted(required - set(contributions.columns)):
        raise ResponseWindowLabelPromotionError(f"observation contributions lack candidate exclusion fields: {missing}")
    if "candidate_id" not in observations.columns:
        raise ResponseWindowLabelPromotionError("candidate observations lack candidate_id.")
    observation_ids = _unique_ids(observations["candidate_id"], label="candidate observations")
    contribution_ids = _id_set(contributions["candidate_id"], label="observation contributions")
    if missing := sorted(observation_ids - contribution_ids):
        raise ResponseWindowLabelPromotionError(
            f"candidate observations lack measured contribution candidates: {missing[:10]}"
        )
    selected = _strict_boolean(contributions["selected_as_label_source"], field="selected_as_label_source")
    included = _strict_boolean(contributions["included_in_label"], field="included_in_label")
    included_ids = set(contributions.loc[included, "candidate_id"].astype(str))
    if included_ids != observation_ids:
        raise ResponseWindowLabelPromotionError(
            "promoted observations disagree with included study contribution candidates."
        )

    entries: list[dict[str, str]] = []
    excluded_ids = sorted(contribution_ids - observation_ids)
    for candidate_id in excluded_ids:
        rows = contributions.loc[contributions["candidate_id"].astype(str).eq(candidate_id)]
        selected_rows = rows.loc[selected.loc[rows.index]]
        if len(selected_rows) > 1:
            raise ResponseWindowLabelPromotionError(
                f"{candidate_id}: observation contributions select multiple label sources."
            )
        reason_rows = selected_rows if len(selected_rows) == 1 else rows
        reasons = _nonempty_text_values(reason_rows["label_exclusion_reason"])
        if len(reasons) != 1:
            raise ResponseWindowLabelPromotionError(
                f"{candidate_id}: measured candidate exclusion requires one authoritative reason."
            )
        entries.append({"candidate_id": candidate_id, "reason": reasons[0]})
    return entries


def require_exclusion_candidates_in_records(
    entries: Sequence[Mapping[str, object]],
    *,
    records: pd.DataFrame,
) -> None:
    """Reject exclusions whose candidate identity has gone stale in OPAL records."""

    if "id" not in records.columns:
        raise ResponseWindowLabelPromotionError("OPAL candidate records require an id column.")
    record_ids = _unique_ids(records["id"], label="OPAL candidate records")
    excluded_ids = {entry["candidate_id"] for entry in _normalize_entries(entries)}
    if missing := sorted(excluded_ids - record_ids):
        raise ResponseWindowLabelPromotionError(
            f"candidate exclusions reference IDs absent from OPAL candidate records: {missing[:10]}"
        )


def build_candidate_selection_exclusion_provenance(
    entries: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    normalized = _normalize_entries(entries)
    return {
        "authority": _AUTHORITY,
        "derivation": _DERIVATION,
        "entry_count": len(normalized),
        "entries": normalized,
        "exclusion_set_id": CANDIDATE_EXCLUSION_SET_ID,
        "source_record": _SOURCE_RECORD,
    }


def validate_candidate_selection_exclusion_provenance(value: object) -> list[dict[str, str]]:
    if not isinstance(value, Mapping) or set(value) != _PROVENANCE_FIELDS:
        raise ResponseWindowLabelPromotionError("published candidate-selection exclusion provenance is malformed.")
    if (
        value["authority"] != _AUTHORITY
        or value["derivation"] != _DERIVATION
        or value["exclusion_set_id"] != CANDIDATE_EXCLUSION_SET_ID
        or value["source_record"] != _SOURCE_RECORD
    ):
        raise ResponseWindowLabelPromotionError("published candidate-selection exclusion semantics disagree.")
    entries = _normalize_entries(value["entries"])
    if value["entries"] != entries or value["entry_count"] != len(entries):
        raise ResponseWindowLabelPromotionError("published candidate-selection exclusion inventory disagrees.")
    return entries


def require_campaign_candidate_exclusion_parity(
    config: Any,
    *,
    authoritative_entries: Sequence[Mapping[str, object]],
) -> None:
    """Require the campaign projection to exactly match study-owned provenance."""

    expected = _normalize_entries(authoritative_entries)
    matching_rules = [
        rule
        for rule in config.candidate_eligibility.rules
        if str(rule.name) == "candidate_id_exclusion"
        and str(rule.params.get("exclusion_set_id", "")) == CANDIDATE_EXCLUSION_SET_ID
    ]
    if not matching_rules and not expected:
        return
    if not matching_rules:
        raise ResponseWindowLabelPromotionError(
            f"campaign is missing candidate_id_exclusion set {CANDIDATE_EXCLUSION_SET_ID!r}."
        )
    if len(matching_rules) != 1:
        raise ResponseWindowLabelPromotionError(
            f"campaign must declare candidate_id_exclusion set {CANDIDATE_EXCLUSION_SET_ID!r} exactly once."
        )
    actual = _normalize_entries(matching_rules[0].params.get("entries"))
    expected_by_id = {entry["candidate_id"]: entry["reason"] for entry in expected}
    actual_by_id = {entry["candidate_id"]: entry["reason"] for entry in actual}
    if missing := sorted(set(expected_by_id) - set(actual_by_id)):
        raise ResponseWindowLabelPromotionError(
            f"campaign candidate exclusion parity failed: missing candidate IDs={missing[:10]}."
        )
    if extra := sorted(set(actual_by_id) - set(expected_by_id)):
        raise ResponseWindowLabelPromotionError(
            f"campaign candidate exclusion parity failed: extra or stale candidate IDs={extra[:10]}."
        )
    mismatched = sorted(
        candidate_id for candidate_id in expected_by_id if actual_by_id[candidate_id] != expected_by_id[candidate_id]
    )
    if mismatched:
        raise ResponseWindowLabelPromotionError(
            f"campaign candidate exclusion reason mismatch for candidate IDs={mismatched[:10]}."
        )


def _normalize_entries(value: object) -> list[dict[str, str]]:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise ResponseWindowLabelPromotionError("candidate exclusion entries must be a list.")
    entries: list[dict[str, str]] = []
    for index, raw in enumerate(value):
        if not isinstance(raw, Mapping) or set(raw) != {"candidate_id", "reason"}:
            raise ResponseWindowLabelPromotionError(
                f"candidate exclusion entry {index} must contain candidate_id and reason."
            )
        candidate_id = str(raw["candidate_id"]).strip()
        reason = str(raw["reason"]).strip()
        if not candidate_id or not reason:
            raise ResponseWindowLabelPromotionError("candidate exclusion identity and reason must be non-empty.")
        entries.append({"candidate_id": candidate_id, "reason": reason})
    ids = [entry["candidate_id"] for entry in entries]
    if len(ids) != len(set(ids)):
        raise ResponseWindowLabelPromotionError("candidate exclusion entries contain duplicate candidate IDs.")
    return sorted(entries, key=lambda entry: entry["candidate_id"])


def _strict_boolean(values: pd.Series, *, field: str) -> pd.Series:
    if not values.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ResponseWindowLabelPromotionError(f"observation contribution field {field!r} must be boolean.")
    return values.astype(bool)


def _unique_ids(values: pd.Series, *, label: str) -> set[str]:
    ids = _canonical_ids(values, label=label)
    if len(ids) != len(set(ids)):
        raise ResponseWindowLabelPromotionError(f"{label} require unique non-empty candidate IDs.")
    return set(ids)


def _id_set(values: pd.Series, *, label: str) -> set[str]:
    ids = set(_canonical_ids(values, label=label))
    if not ids:
        raise ResponseWindowLabelPromotionError(f"{label} require non-empty candidate IDs.")
    return ids


def _canonical_ids(values: pd.Series, *, label: str) -> list[str]:
    ids = values.astype(str).tolist()
    if values.isna().any() or any(not value or value != value.strip() for value in ids):
        raise ResponseWindowLabelPromotionError(f"{label} require canonical non-empty candidate IDs.")
    return ids


def _nonempty_text_values(values: pd.Series) -> list[str]:
    text = [str(value).strip() for value in values.dropna()]
    return sorted({value for value in text if value})
