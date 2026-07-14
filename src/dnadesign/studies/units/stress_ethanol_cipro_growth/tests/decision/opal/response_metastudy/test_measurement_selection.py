"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_measurement_selection.py

Response-owned measurement-selection contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
import yaml

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.runtime import (
    measurement_selection,
)

CONFIG = Path(
    "src/dnadesign/studies/units/stress_ethanol_cipro_growth/decision/opal/response_metastudy/config/"
    "response_model_screen_selection.yaml"
)


def _payload() -> dict[str, object]:
    return {
        "schema_id": "stress_ethanol_cipro_growth.response_measurement_selection.v1",
        "schema_version": "1",
        "study_id": "stress_ethanol_cipro_growth",
        "selection_id": "response_metastudy_model_screen_v1",
        "scope": "model_screen_only",
        "promotion_aggregation": "not_defined",
        "measurements": [
            {"reader_experiment_id": "experiment-a", "design_id": "design-a"},
        ],
    }


def _reader_designs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "experiment_id": ["experiment-a", "experiment-a", "experiment-a"],
            "design_id": ["design-a", "design-a", "reference"],
            "reduction_id": ["primary", "sensitivity", "primary"],
            "is_reference": [False, False, True],
        }
    )


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def test_response_measurement_selection_resolves_exact_reader_rows(tmp_path: Path) -> None:
    result = measurement_selection.load_response_measurement_selection(
        _write(tmp_path / "selection.yaml", _payload()),
        reader_designs=_reader_designs(),
        primary_reduction_id="primary",
    )

    assert result.rows.to_dict(orient="records") == [{"reader_experiment_id": "experiment-a", "design_id": "design-a"}]
    assert result.scope == "model_screen_only"
    assert result.promotion_aggregation == "not_defined"


def test_response_measurement_selection_rejects_candidate_identity_fields(tmp_path: Path) -> None:
    payload = deepcopy(_payload())
    payload["measurements"][0]["candidate_id"] = "candidate-1"  # type: ignore[index]

    with pytest.raises(measurement_selection.ResponseMeasurementSelectionError, match="fields must be exactly"):
        measurement_selection.load_response_measurement_selection(
            _write(tmp_path / "selection.yaml", payload),
            reader_designs=_reader_designs(),
            primary_reduction_id="primary",
        )


def test_response_measurement_selection_rejects_missing_reader_pair(tmp_path: Path) -> None:
    payload = deepcopy(_payload())
    payload["measurements"][0]["design_id"] = "design-missing"  # type: ignore[index]

    with pytest.raises(
        measurement_selection.ResponseMeasurementSelectionError,
        match="absent from the Reader primary reduction",
    ):
        measurement_selection.load_response_measurement_selection(
            _write(tmp_path / "selection.yaml", payload),
            reader_designs=_reader_designs(),
            primary_reduction_id="primary",
        )


def test_repository_response_screen_selection_is_explicit_and_identity_free() -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    measurements = payload["measurements"]
    reader_designs = pd.DataFrame.from_records(measurements).rename(columns={"reader_experiment_id": "experiment_id"})
    reader_designs["reduction_id"] = "primary"
    reader_designs["is_reference"] = False

    result = measurement_selection.load_response_measurement_selection(
        CONFIG,
        reader_designs=reader_designs,
        primary_reduction_id="primary",
    )

    assert len(result.rows) == 35
    assert all(set(row) == {"reader_experiment_id", "design_id"} for row in measurements)
