"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/response_metastudy/test_candidate_identity.py

Exact study-binding contracts for response-metastudy candidate identity.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.response_metastudy.runtime import (
    candidate_identity,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.promoter_candidate_bindings import (
    PromoterCandidateBindingsError,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.tests.decision.opal.reader_promoter_evidence._fixtures import (
    write_candidate_bindings,
)


def _measurement_selection(*, design_id: str = "design-a") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "design_id": [design_id],
            "reader_experiment_id": ["experiment-a"],
        }
    )


def _excluded_designs(*, design_id: str | None = None) -> pd.DataFrame:
    rows = [] if design_id is None else [{"design_id": design_id, "reason": "absent_from_study_candidate_bindings"}]
    return pd.DataFrame.from_records(rows, columns=["design_id", "reason"])


def test_load_response_candidate_identity_bindings_uses_exact_study_alias(tmp_path: Path) -> None:
    bundle = write_candidate_bindings(
        tmp_path / "bindings",
        [("candidate-1", "design-a", "densegen_tfbs")],
    )

    resolved = candidate_identity.load_response_candidate_identity_bindings(
        measurement_selection=_measurement_selection(),
        excluded_designs=_excluded_designs(),
        bundle_root=bundle,
    )

    assert resolved.rows.to_dict(orient="records") == [
        {
            "id": "candidate-1",
            "design_id": "design-a",
            "reader_experiment_id": "experiment-a",
        }
    ]
    assert resolved.manifest_path == (bundle / "manifest.json").resolve()
    assert resolved.records_path == (bundle / "bindings.parquet").resolve()


def test_load_response_candidate_identity_bindings_rejects_missing_reader_alias(tmp_path: Path) -> None:
    bundle = write_candidate_bindings(
        tmp_path / "bindings",
        [("candidate-1", "design-a", "densegen_tfbs")],
    )

    with pytest.raises(PromoterCandidateBindingsError, match="unresolved Reader design aliases"):
        candidate_identity.load_response_candidate_identity_bindings(
            measurement_selection=_measurement_selection(design_id="design-missing"),
            excluded_designs=_excluded_designs(),
            bundle_root=bundle,
        )


def test_load_response_candidate_identity_bindings_rejects_duplicate_candidate_resolution(tmp_path: Path) -> None:
    bundle = write_candidate_bindings(
        tmp_path / "bindings",
        [("candidate-1", "design-a", "densegen_tfbs")],
    )
    measurements = pd.DataFrame(
        {
            "reader_experiment_id": ["experiment-a", "experiment-b"],
            "design_id": ["design-a", "design-a"],
        }
    )

    with pytest.raises(PromoterCandidateBindingsError, match="duplicate candidate IDs"):
        candidate_identity.load_response_candidate_identity_bindings(
            measurement_selection=measurements,
            excluded_designs=_excluded_designs(),
            bundle_root=bundle,
        )


def test_load_response_candidate_identity_bindings_rejects_exclusion_that_now_resolves(tmp_path: Path) -> None:
    bundle = write_candidate_bindings(
        tmp_path / "bindings",
        [("candidate-1", "design-a", "densegen_tfbs"), ("candidate-2", "design-unbound", "densegen_tfbs")],
    )

    with pytest.raises(PromoterCandidateBindingsError, match="declared absent.*but resolve"):
        candidate_identity.load_response_candidate_identity_bindings(
            measurement_selection=_measurement_selection(),
            excluded_designs=_excluded_designs(design_id="design-unbound"),
            bundle_root=bundle,
        )
