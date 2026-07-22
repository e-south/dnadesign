"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/response_window_observations/test_sources.py

Contract tests for exact Reader-to-candidate observation routing.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations import sources
from dnadesign.studies.units.stress_ethanol_cipro_growth.response_window_observations.reader_bundle import (
    ReaderResponseBundle,
)


def test_exact_reader_aliases_resolve_and_preserve_scientific_evidence() -> None:
    bundle = _bundle()

    result = sources.resolve_reader_candidate_evidence(
        bundle,
        binding_rows=_bindings(),
        unbound_reader_designs=pd.DataFrame(
            [{"design_id": "unbound", "reason": "absent_from_study_candidate_bindings"}]
        ),
    )

    assert set(result.measurements["candidate_id"]) == {"candidate-a"}
    assert set(result.measurements["reader_experiment_id"]) == {"experiment-a", "experiment-b"}
    assert result.measurements["n00"].tolist() == [3, 4]
    assert set(result.measurements["sequence_sha256"]) == {"a" * 64}
    assert set(result.bootstrap_draws["candidate_id"]) == {"candidate-a"}
    assert set(result.excluded_reader_designs["design_id"]) == {"unbound"}


def test_resolution_rejects_missing_stale_or_bound_exclusion_declarations() -> None:
    bundle = _bundle()

    with pytest.raises(sources.ResponseWindowObservationSourceError, match="unbound design accounting"):
        sources.resolve_reader_candidate_evidence(
            bundle,
            binding_rows=_bindings(),
            unbound_reader_designs=pd.DataFrame(columns=["design_id", "reason"]),
        )

    with pytest.raises(sources.ResponseWindowObservationSourceError, match="unbound design accounting"):
        sources.resolve_reader_candidate_evidence(
            bundle,
            binding_rows=_bindings(),
            unbound_reader_designs=pd.DataFrame(
                [
                    {"design_id": "unbound", "reason": "absent_from_study_candidate_bindings"},
                    {"design_id": "stale", "reason": "absent_from_study_candidate_bindings"},
                ]
            ),
        )

    bindings = pd.concat(
        [
            _bindings(),
            pd.DataFrame(
                [
                    {
                        "alias_namespace": "reader.design_id",
                        "alias": "unbound",
                        "candidate_id": "candidate-new",
                        "sequence_sha256": "b" * 64,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(sources.ResponseWindowObservationSourceError, match="unbound design accounting"):
        sources.resolve_reader_candidate_evidence(
            bundle,
            binding_rows=bindings,
            unbound_reader_designs=pd.DataFrame(
                [{"design_id": "unbound", "reason": "absent_from_study_candidate_bindings"}]
            ),
        )


def test_resolution_rejects_non_exact_binding_rows() -> None:
    bindings = _bindings()
    bindings.loc[0, "alias_namespace"] = "source.alias"

    with pytest.raises(sources.ResponseWindowObservationSourceError, match="unbound design accounting"):
        sources.resolve_reader_candidate_evidence(
            _bundle(),
            binding_rows=bindings,
            unbound_reader_designs=pd.DataFrame(
                [{"design_id": "unbound", "reason": "absent_from_study_candidate_bindings"}]
            ),
        )


def _bundle() -> ReaderResponseBundle:
    design_rows: list[dict[str, object]] = []
    draw_rows: list[dict[str, object]] = []
    for experiment_id, support in (("experiment-a", 3), ("experiment-b", 4)):
        for design_id, is_reference in (("design-a", False), ("unbound", False), ("reference", True)):
            design_rows.append(
                {
                    "experiment_id": experiment_id,
                    "design_id": design_id,
                    "reduction_id": "primary",
                    "reduction_role": "primary",
                    "is_reference": is_reference,
                    "n00": support,
                    **_values(1.0),
                    **{f"{column}_event_half_range": 0.1 for column in sources.VALUE_COLUMNS},
                    **_exact_censor_provenance(),
                }
            )
            for draw_index in range(2):
                draw_rows.append(
                    {
                        "experiment_id": experiment_id,
                        "design_id": design_id,
                        "reduction_id": "primary",
                        "draw_index": draw_index,
                        "is_reference": is_reference,
                        **_values(1.0 + draw_index),
                    }
                )
    return ReaderResponseBundle(
        root=Path("/reader"),
        manifest_path=Path("/reader/manifest.json"),
        manifest={"primary_reduction_id": "primary"},
        designs=pd.DataFrame.from_records(design_rows),
        bootstrap_draws=pd.DataFrame.from_records(draw_rows),
        wells=pd.DataFrame(),
        traces=pd.DataFrame(),
        events=pd.DataFrame(),
    )


def _bindings() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "alias_namespace": "reader.design_id",
                "alias": "design-a",
                "candidate_id": "candidate-a",
                "sequence_sha256": "a" * 64,
            }
        ]
    )


def _values(value: float) -> dict[str, float]:
    return {column: value for column in sources.VALUE_COLUMNS}


def _exact_censor_provenance() -> dict[str, object]:
    return {
        f"{component}_{suffix}": False if suffix != "bound_kind" else "exact"
        for component in sources.VALUE_COLUMNS
        for suffix in ("has_policy_clipping", "has_instrument_overflow", "bound_kind")
    }
