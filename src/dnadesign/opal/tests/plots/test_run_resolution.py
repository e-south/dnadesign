"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/plots/test_run_resolution.py

Tests for metric-neutral single-run plot resolution.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import polars as pl
import pytest

from dnadesign.opal.src.core.utils import OpalError
from dnadesign.opal.src.plots._run_resolution import (
    parse_run_view_definitions,
    resolve_run_id,
    resolve_single_round,
)


def _runs() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "as_of_round": [0, 1],
            "run_id": ["run-zero", "run-one"],
        }
    )


def test_single_round_resolution_is_objective_neutral_and_fail_fast() -> None:
    runs = _runs()

    assert resolve_single_round(runs, round_selector="latest") == 1
    assert resolve_single_round(runs, round_selector=[0]) == 0
    with pytest.raises(OpalError, match="single round"):
        resolve_single_round(runs, round_selector="all")


def test_implicit_run_resolution_rejects_ambiguous_rounds() -> None:
    runs = pl.DataFrame(
        {
            "as_of_round": [0, 0],
            "run_id": ["run-a", "run-b"],
        }
    )

    with pytest.raises(OpalError, match="Multiple run_ids"):
        resolve_run_id(runs, round_k=0, run_id=None)
    assert resolve_run_id(runs, round_k=0, run_id="run-a") == "run-a"


@pytest.mark.parametrize("raw", ['{"selection_view_id": "a"}', '["not-a-mapping"]'])
def test_run_view_definitions_reject_valid_json_with_the_wrong_shape(raw: str) -> None:
    with pytest.raises(OpalError, match="must be a JSON list of mappings"):
        parse_run_view_definitions(raw, field_label="Run objective definitions")


@pytest.mark.parametrize("raw", ["{", "not-json"])
def test_run_view_definitions_reject_malformed_json(raw: str) -> None:
    with pytest.raises(OpalError, match="Run objective definitions is invalid JSON"):
        parse_run_view_definitions(raw, field_label="Run objective definitions")
