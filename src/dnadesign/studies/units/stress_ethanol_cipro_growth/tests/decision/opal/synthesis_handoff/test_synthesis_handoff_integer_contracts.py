"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/synthesis_handoff/test_synthesis_handoff_integer_contracts.py

Strict integer parsing tests for synthesis-handoff lifecycle fields.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff.contracts import (
    CloningStrategy,
    SelectedCandidate,
    SelectionMembership,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff.records import (
    ExpectedHandoffArtifact,
    ExpectedSelectionView,
    SynthesisHandoffRecord,
)
from dnadesign.studies.units.stress_ethanol_cipro_growth.decision.opal.synthesis_handoff.strategy import (
    load_cloning_strategy,
)


def _artifact(*, expected_rows: object = 1) -> ExpectedHandoffArtifact:
    return ExpectedHandoffArtifact(
        campaign_slug="secg_msrb_greedy",
        expected_rows=expected_rows,  # type: ignore[arg-type]
        manifest_path="out/manifest.csv",
        vendor_workbook_path="out/order.xlsx",
        genbank_dir_path="out/genbank",
        genbank_feature_table_path="out/features.csv",
    )


@pytest.mark.parametrize("bad_value", [True, 1.5, float("nan"), "1.5"])
def test_expected_row_counts_reject_lossy_integer_coercion(bad_value: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _artifact(expected_rows=bad_value)
    with pytest.raises(ValueError, match="positive integer"):
        ExpectedSelectionView(
            selection_view_id="ethanol",
            expected_rows=bad_value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("field", ["assay_batch_index", "model_as_of_round"])
@pytest.mark.parametrize("bad_value", [True, 1.5, float("nan"), "1.5", -1])
def test_lifecycle_indices_reject_non_integer_or_negative_values(field: str, bad_value: object) -> None:
    values: dict[str, object] = {
        "assay_batch_index": 1,
        "model_as_of_round": 0,
    }
    values[field] = bad_value
    with pytest.raises(ValueError, match=f"{field} must be a non-negative integer"):
        SynthesisHandoffRecord(
            handoff_id="stress-opal-r0-msrb-v1",
            lifecycle_status="authorized_for_materialization",
            source_authority="opal_selection_batch",
            selection_epoch="opal_model_round",
            assay_batch_index=values["assay_batch_index"],  # type: ignore[arg-type]
            model_as_of_round=values["model_as_of_round"],  # type: ignore[arg-type]
            run_id="run-1",
            strategy_id="stress_promoter_insert:v1",
            campaign_slug="secg_msrb_greedy",
            expected_selection_views=(ExpectedSelectionView(selection_view_id="ethanol", expected_rows=1),),
            expected_study_aliases=("SECG-019",),
            expected_artifact=_artifact(),
        )


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("as_of_round", 1.5, "as_of_round must be a non-negative integer"),
        ("as_of_round", True, "as_of_round must be a non-negative integer"),
        ("selection_rank", 1.5, "selection_rank must be a positive integer"),
        ("selection_rank", True, "selection_rank must be a positive integer"),
    ],
)
def test_selected_candidate_rejects_lossy_integer_coercion(
    field: str,
    bad_value: object,
    message: str,
) -> None:
    values: dict[str, object] = {"as_of_round": 0, "selection_rank": 1}
    values[field] = bad_value
    with pytest.raises(ValueError, match=message):
        SelectedCandidate(
            campaign_slug="secg_msrb_greedy",
            selection_memberships=(SelectionMembership(selection_view_id="ethanol", rank=1),),
            as_of_round=values["as_of_round"],  # type: ignore[arg-type]
            run_id="run-1",
            selection_rank=values["selection_rank"],  # type: ignore[arg-type]
            id="candidate-1",
            sequence="A" * 60,
            synthesis_name="SECG-019",
            selection_epoch="opal_model_round",
        )


@pytest.mark.parametrize("bad_value", [True, 60.5, "60.5"])
def test_cloning_strategy_rejects_lossy_core_length_coercion(bad_value: object) -> None:
    with pytest.raises(ValueError, match="expected_core_length must be a positive integer"):
        CloningStrategy(
            name="stress_promoter_insert",
            version="1",
            left_flank="acgt",
            right_flank="tgca",
            expected_core_length=bad_value,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("bad_yaml_value", ["true", "60.5"])
def test_cloning_strategy_loader_preserves_strict_core_length_contract(
    tmp_path: Path,
    bad_yaml_value: str,
) -> None:
    strategy_path = tmp_path / "strategy.yaml"
    strategy_path.write_text(
        "\n".join(
            [
                "name: stress_promoter_insert",
                "version: '1'",
                "left_flank: acgt",
                "right_flank: tgca",
                f"expected_core_length: {bad_yaml_value}",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="expected_core_length must be a positive integer"):
        load_cloning_strategy(strategy_path)
