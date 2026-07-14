"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_response_examples.py

Tests for response-only measured example selection.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd
import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.evaluation import (
    response_examples,
)


def test_response_examples_preserve_reader_values_without_sfxi_fields() -> None:
    response = pd.DataFrame([_response_row("ethanol"), _response_row("ciprofloxacin"), _response_row("and")])

    result = response_examples.build_response_example_rows(
        response,
        examples={"pDual-10-spyp": "SpyP measured ethanol-response example"},
        selection_view_ids=("ethanol", "ciprofloxacin", "and"),
    )

    assert len(result) == 3
    assert set(result["example_label"]) == {"SpyP measured ethanol-response example"}
    assert set(result["off_suppression"]) == {-0.25}
    assert result.loc[0, "r10"] == response.loc[0, "r10"]
    assert not any(column.startswith("v") or column.endswith("_star") for column in result.columns)


def test_response_examples_reject_missing_declared_view() -> None:
    response = pd.DataFrame([_response_row("ethanol")])

    with pytest.raises(ValueError, match="lack design and selection-view pairs"):
        response_examples.build_response_example_rows(
            response,
            examples={"pDual-10-spyp": "SpyP"},
            selection_view_ids=("ethanol", "and"),
        )


def test_response_examples_reject_incomplete_design_by_view_grid() -> None:
    first = [_response_row(view) for view in ("ethanol", "ciprofloxacin", "and")]
    second = _response_row("ethanol")
    second.update(id="second", design_id="pDual-10-sulap")

    with pytest.raises(ValueError, match="lack design and selection-view pairs"):
        response_examples.build_response_example_rows(
            pd.DataFrame([*first, second]),
            examples={"pDual-10-spyp": "SpyP", "pDual-10-sulap": "sulAp"},
            selection_view_ids=("ethanol", "ciprofloxacin", "and"),
        )


def _response_row(selection_view_id: str) -> dict[str, object]:
    return {
        "id": "candidate",
        "design_id": "pDual-10-spyp",
        "reader_experiment_id": "experiment",
        "selection_view_id": selection_view_id,
        "response_separation": 2.0,
        "on_magnitude_floor": 1.5,
        "off_magnitude_ceiling": 0.25,
        "passes_all_zero_constraints": False,
        **{
            f"{prefix}{state}": float(index)
            for prefix in ("r", "b")
            for index, state in enumerate(("00", "10", "01", "11"))
        },
    }
