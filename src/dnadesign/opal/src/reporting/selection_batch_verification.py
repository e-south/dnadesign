"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/reporting/selection_batch_verification.py

Run-artifact verification for persisted OPAL selection batches.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..core.utils import OpalError, file_sha256
from .verify_outputs import read_selection_table

_SELECTION_REQUIRED_COLUMNS = frozenset(
    {
        "run_id",
        "as_of_round",
        "campaign_slug",
        "selection_view_id",
        "id",
        "selection_batch_key",
        "deduplicate_by",
        "rank_competition",
        "rank_ordinal",
        "score",
        "selection_score",
        "score_ref",
        "allocation_slot",
        "selection_origin",
    }
)
_ALLOCATION_TRACE_REQUIRED_COLUMNS = frozenset(
    {
        "run_id",
        "as_of_round",
        "campaign_slug",
        "selection_view_id",
        "id",
        "selection_batch_key",
        "deduplicate_by",
        "selection_origin",
    }
)


def verify_file_digest(path: Path, *, expected_sha256: str, artifact_key: str) -> str:
    """Require one artifact to match its run-ledger SHA-256 digest."""

    expected = str(expected_sha256).strip().lower()
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise OpalError(f"Run ledger artifact {artifact_key!r} is missing a valid SHA-256 digest.")
    try:
        actual = file_sha256(path)
    except (OSError, ValueError) as exc:
        raise OpalError(f"Cannot read run artifact {artifact_key!r}: {path}.") from exc
    if actual != expected:
        raise OpalError(f"Run artifact {artifact_key!r} SHA-256 mismatch: expected={expected}, actual={actual}.")
    return actual


def verify_selection_batch_memberships(
    rows: Sequence[Mapping[str, Any]],
    *,
    selections_path: Path,
    allocation_trace_path: Path | None,
    campaign_slug: str,
    run_id: str,
    as_of_round: int,
    deduplicate_by: str,
    eps: float = 1e-6,
) -> dict[str, Any]:
    """Reconcile batch claims to digest-bound run selection evidence."""

    selections = read_selection_table(selections_path)
    missing = sorted(_SELECTION_REQUIRED_COLUMNS - set(selections.columns))
    if missing:
        raise OpalError(f"Selection artifact is missing batch-verification columns: {missing}.")
    _validate_selection_scope(
        selections,
        campaign_slug=campaign_slug,
        run_id=run_id,
        as_of_round=as_of_round,
    )
    persisted = _persisted_memberships(selections)
    declared = _declared_memberships(rows)
    if set(persisted) != set(declared):
        missing_keys = sorted(set(declared) - set(persisted))
        extra_keys = sorted(set(persisted) - set(declared))
        raise OpalError(
            "Selection batch memberships do not match the run selection artifact; "
            f"missing={missing_keys[:10]}, unexpected={extra_keys[:10]}."
        )
    mismatches: list[str] = []
    for key in sorted(declared):
        expected = persisted[key]
        observed = declared[key]
        for field in ("rank_competition", "rank_ordinal", "score_ref", "allocation_slot", "selection_origin"):
            if observed[field] != expected[field]:
                mismatches.append(f"{key[0]}/{key[1]}:{field}")
        for field in ("score", "selection_score"):
            if not np.isclose(float(observed[field]), float(expected[field]), rtol=0.0, atol=float(eps)):
                mismatches.append(f"{key[0]}/{key[1]}:{field}")
    if mismatches:
        raise OpalError(
            f"Selection batch membership provenance disagrees with the run selection artifact: {mismatches[:10]}."
        )
    _verify_batch_keys(rows, selections=selections, deduplicate_by=deduplicate_by)
    preference_evidence = selections
    preference_source = "selection artifact"
    if allocation_trace_path is not None:
        preference_evidence = read_selection_table(allocation_trace_path)
        trace_missing = sorted(_ALLOCATION_TRACE_REQUIRED_COLUMNS - set(preference_evidence.columns))
        if trace_missing:
            raise OpalError(f"Allocation trace is missing batch-verification columns: {trace_missing}.")
        _validate_selection_scope(
            preference_evidence,
            campaign_slug=campaign_slug,
            run_id=run_id,
            as_of_round=as_of_round,
        )
        preference_source = "allocation trace"
    _verify_preferred_views(
        rows,
        evidence=preference_evidence,
        deduplicate_by=deduplicate_by,
        evidence_label=preference_source,
    )
    return {
        "status": "pass",
        "selection_path": str(selections_path),
        "preference_source": preference_source,
        "membership_count": len(declared),
        "mismatch_count": 0,
    }


def _verify_batch_keys(
    rows: Sequence[Mapping[str, Any]],
    *,
    selections: pd.DataFrame,
    deduplicate_by: str,
) -> None:
    key_column = _required_text(deduplicate_by, field="deduplicate_by")
    observed_dedup = selections["deduplicate_by"].map(lambda value: "" if pd.isna(value) else str(value).strip())
    if observed_dedup.eq("").any() or not observed_dedup.eq(key_column).all():
        raise OpalError(
            f"Selection artifact contains mixed or unexpected deduplicate_by values; expected {key_column!r}."
        )
    key_by_id: dict[str, str] = {}
    for raw in selections.to_dict(orient="records"):
        candidate_id = _required_text(raw.get("id"), field="id")
        batch_key = _required_text(raw.get("selection_batch_key"), field="selection_batch_key")
        prior = key_by_id.setdefault(candidate_id, batch_key)
        if prior != batch_key:
            raise OpalError(
                f"Selection artifact candidate {candidate_id!r} has conflicting selection_batch_key values."
            )
    mismatches: list[str] = []
    for row in rows:
        candidate_id = _required_text(row.get("id"), field="id")
        batch_key = _required_text(row.get("selection_batch_key"), field="selection_batch_key")
        if key_by_id.get(candidate_id) != batch_key:
            mismatches.append(candidate_id)
    if mismatches:
        raise OpalError(
            f"Selection batch keys disagree with the run selection artifact for candidate ids: {mismatches[:10]}."
        )


def _verify_preferred_views(
    rows: Sequence[Mapping[str, Any]],
    *,
    evidence: pd.DataFrame,
    deduplicate_by: str,
    evidence_label: str,
) -> None:
    observed_dedup = evidence["deduplicate_by"].map(lambda value: "" if pd.isna(value) else str(value).strip())
    if observed_dedup.eq("").any() or not observed_dedup.eq(deduplicate_by).all():
        raise OpalError(
            f"{evidence_label.title()} contains mixed or unexpected deduplicate_by values; expected {deduplicate_by!r}."
        )
    preferred_by_key: dict[str, set[str]] = {}
    for raw in evidence.to_dict(orient="records"):
        origin = _required_text(raw.get("selection_origin"), field="selection_origin")
        if origin != "preferred_top_k":
            continue
        batch_key = _required_text(raw.get("selection_batch_key"), field="selection_batch_key")
        view_id = _required_text(raw.get("selection_view_id"), field="selection_view_id")
        preferred_by_key.setdefault(batch_key, set()).add(view_id)
    mismatches: list[str] = []
    for row in rows:
        batch_key = _required_text(row.get("selection_batch_key"), field="selection_batch_key")
        declared = {
            _required_text(value, field="preferred_view_ids") for value in (row.get("preferred_view_ids") or [])
        }
        if declared != preferred_by_key.get(batch_key, set()):
            mismatches.append(batch_key)
    if mismatches:
        raise OpalError(
            f"Selection batch preferred_view_ids disagree with the run {evidence_label}: {mismatches[:10]}."
        )


def _validate_selection_scope(
    selections: pd.DataFrame,
    *,
    campaign_slug: str,
    run_id: str,
    as_of_round: int,
) -> None:
    expected_text = {"campaign_slug": campaign_slug, "run_id": run_id}
    for field, expected in expected_text.items():
        values = selections[field].map(lambda value: "" if pd.isna(value) else str(value).strip())
        if values.eq("").any() or not values.eq(expected).all():
            raise OpalError(f"Selection artifact contains mixed or unexpected {field}; expected {expected!r}.")
    rounds = pd.to_numeric(selections["as_of_round"], errors="coerce")
    if rounds.isna().any() or not (rounds.astype(int) == int(as_of_round)).all():
        raise OpalError(f"Selection artifact contains mixed or unexpected as_of_round; expected {int(as_of_round)}.")


def _persisted_memberships(selections: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for raw in selections.to_dict(orient="records"):
        candidate_id = _required_text(raw.get("id"), field="id")
        view_id = _required_text(raw.get("selection_view_id"), field="selection_view_id")
        key = (candidate_id, view_id)
        if key in result:
            raise OpalError(f"Selection artifact contains duplicate candidate/view membership {key!r}.")
        result[key] = _normalized_membership(raw, candidate_id=candidate_id)
    return result


def _declared_memberships(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        candidate_id = _required_text(row.get("id"), field="id")
        for raw in row.get("selection_memberships") or []:
            if not isinstance(raw, Mapping):
                raise OpalError(f"Selection batch membership for candidate {candidate_id!r} must be a mapping.")
            view_id = _required_text(raw.get("selection_view_id"), field="selection_view_id")
            key = (candidate_id, view_id)
            if key in result:
                raise OpalError(f"Selection batch contains duplicate candidate/view membership {key!r}.")
            result[key] = _normalized_membership(raw, candidate_id=candidate_id)
    return result


def _normalized_membership(raw: Mapping[str, Any], *, candidate_id: str) -> dict[str, Any]:
    return {
        "rank_competition": _integer(raw.get("rank", raw.get("rank_competition")), field="rank"),
        "rank_ordinal": _integer(raw.get("rank_ordinal"), field="rank_ordinal"),
        "score": _finite_float(raw.get("score"), field="score", candidate_id=candidate_id),
        "selection_score": _finite_float(
            raw.get("selection_score"),
            field="selection_score",
            candidate_id=candidate_id,
        ),
        "score_ref": _required_text(raw.get("score_ref"), field="score_ref"),
        "allocation_slot": _optional_integer(raw.get("allocation_slot"), field="allocation_slot"),
        "selection_origin": _required_text(raw.get("selection_origin"), field="selection_origin"),
    }


def _required_text(value: Any, *, field: str) -> str:
    if value is None or (not isinstance(value, (list, dict, tuple, np.ndarray)) and bool(pd.isna(value))):
        raise OpalError(f"Selection artifact {field} must be non-empty.")
    text = str(value).strip()
    if not text:
        raise OpalError(f"Selection artifact {field} must be non-empty.")
    return text


def _integer(value: Any, *, field: str) -> int:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise OpalError(f"Selection artifact {field} must be an integer.") from exc
    if not np.isfinite(number) or not number.is_integer():
        raise OpalError(f"Selection artifact {field} must be an integer.")
    return int(number)


def _optional_integer(value: Any, *, field: str) -> int | None:
    if value is None or (not isinstance(value, (list, dict, tuple, np.ndarray)) and bool(pd.isna(value))):
        return None
    return _integer(value, field=field)


def _finite_float(value: Any, *, field: str, candidate_id: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise OpalError(
            f"Selection artifact {field} for candidate {candidate_id!r} must be finite numeric data."
        ) from exc
    if not np.isfinite(number):
        raise OpalError(f"Selection artifact {field} for candidate {candidate_id!r} must be finite numeric data.")
    return number


__all__ = ["verify_file_digest", "verify_selection_batch_memberships"]
