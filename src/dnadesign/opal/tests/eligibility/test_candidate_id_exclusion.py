"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/eligibility/test_candidate_id_exclusion.py

Candidate-ID eligibility exclusion contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd
import pytest

from dnadesign.opal.src.config.plugin_schemas import validate_params
from dnadesign.opal.src.core.utils import OpalError
from dnadesign.opal.src.eligibility.candidate_ids import candidate_id_exclusion


def test_candidate_id_exclusion_filters_declared_ids_with_reasoned_report() -> None:
    result = candidate_id_exclusion(
        frame=pd.DataFrame({"id": ["a", "b", "c"], "sequence": ["A", "C", "G"]}),
        params={
            "exclusion_set_id": "study-observation-dispositions-v1",
            "entries": [
                {"candidate_id": "b", "reason": "nonexact_primary_component"},
                {"candidate_id": "c", "reason": "repeat_excluded_noncomparable"},
            ],
            "min_remaining_candidates": 1,
        },
    )

    assert result.frame["id"].tolist() == ["a"]
    assert result.report["rule"] == "candidate_id_exclusion"
    assert result.report["exclusion_set_id"] == "study-observation-dispositions-v1"
    assert result.report["excluded_rows"] == 2
    assert result.report["reason_counts"] == {
        "nonexact_primary_component": 1,
        "repeat_excluded_noncomparable": 1,
    }


@pytest.mark.parametrize(
    ("entries", "message"),
    [
        ([], "at least one"),
        (
            [
                {"candidate_id": "b", "reason": "one"},
                {"candidate_id": "b", "reason": "two"},
            ],
            "duplicate candidate IDs",
        ),
        ([{"candidate_id": "missing", "reason": "not present"}], "unknown candidate IDs"),
    ],
)
def test_candidate_id_exclusion_rejects_incomplete_or_stale_entries(entries, message: str) -> None:
    with pytest.raises(OpalError, match=message):
        candidate_id_exclusion(
            frame=pd.DataFrame({"id": ["a", "b"]}),
            params={
                "exclusion_set_id": "study-observation-dispositions-v1",
                "entries": entries,
                "min_remaining_candidates": 1,
            },
        )


def test_candidate_id_exclusion_fails_when_too_few_candidates_remain() -> None:
    with pytest.raises(OpalError, match="min_remaining_candidates=2"):
        candidate_id_exclusion(
            frame=pd.DataFrame({"id": ["a", "b"]}),
            params={
                "exclusion_set_id": "study-observation-dispositions-v1",
                "entries": [{"candidate_id": "b", "reason": "excluded"}],
                "min_remaining_candidates": 2,
            },
        )


def test_candidate_id_exclusion_config_rejects_blank_and_duplicate_entries() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        validate_params(
            "candidate_eligibility",
            "candidate_id_exclusion",
            {
                "exclusion_set_id": "",
                "entries": [{"candidate_id": "b", "reason": "excluded"}],
                "min_remaining_candidates": 1,
            },
        )

    with pytest.raises(ValueError, match="duplicate candidate IDs"):
        validate_params(
            "candidate_eligibility",
            "candidate_id_exclusion",
            {
                "exclusion_set_id": "set-v1",
                "entries": [
                    {"candidate_id": "b", "reason": "one"},
                    {"candidate_id": "b", "reason": "two"},
                ],
                "min_remaining_candidates": 1,
            },
        )
