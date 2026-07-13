"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/eligibility/test_restriction_site_eligibility.py

Restriction-site candidate eligibility contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pandas as pd

from dnadesign.opal import CandidateEligibilityBlock
from dnadesign.opal.src.config.types import (
    CampaignBlock,
    DataBlock,
    IngestBlock,
    LabelsBlock,
    LocationLocal,
    OwnershipBlock,
    PluginRef,
    RootConfig,
    SafetyBlock,
    ScoringBlock,
    SelectionBatchBlock,
    SelectionView,
    TrainingBlock,
)
from dnadesign.opal.src.eligibility.restriction_sites import restriction_site_exclusion
from dnadesign.opal.src.runtime.round_plan import plan_round
from dnadesign.opal.src.storage.data_access import RecordsStore

LEFT_FLANK = "accgggatcctgcag"
RIGHT_FLANK = "tgagggaattcgcga"


def _eligibility_rule(*, min_remaining: int = 1) -> PluginRef:
    return PluginRef(
        name="restriction_site_exclusion",
        params={
            "sequence_column": "sequence",
            "scan_space": "final_assembled_insert",
            "assembly_strategy_ref": "sfxi_promoter_insert:v1",
            "left_flank": LEFT_FLANK,
            "right_flank": RIGHT_FLANK,
            "expected_core_length": 60,
            "min_remaining_candidates": min_remaining,
            "forbidden_sites": [
                {"enzyme": "BamHI", "motif": "GGATCC", "allowed_regions": ["left_flank"]},
                {"enzyme": "EcoRI", "motif": "GAATTC", "allowed_regions": ["right_flank"]},
            ],
        },
    )


def _cfg(tmp_path, *, min_remaining: int = 1) -> RootConfig:
    return RootConfig(
        schema_version="opal.campaign.v3",
        campaign=CampaignBlock(name="demo", slug="demo", workdir=str(tmp_path / "campaign")),
        ownership=OwnershipBlock(owner_scope="opal_demo"),
        data=DataBlock(
            location=LocationLocal(kind="local", path=str(tmp_path / "records.parquet")),
            x_column_name="X",
            y_column_name="Y",
            transforms_x=PluginRef(name="identity", params={}),
            transforms_y=PluginRef(name="scalar_from_table_v1", params={}),
            y_expected_length=1,
        ),
        candidate_eligibility=CandidateEligibilityBlock(rules=[_eligibility_rule(min_remaining=min_remaining)]),
        labels=LabelsBlock(),
        model=PluginRef(name="random_forest", params={"n_estimators": 5, "random_state": 0}),
        selection_views=[
            SelectionView(
                id="primary",
                objective=PluginRef(name="scalar_identity_v1", params={}),
                selection=PluginRef(
                    name="top_n",
                    params={
                        "top_k": 1,
                        "score_ref": "scalar",
                        "objective_mode": "maximize",
                        "tie_handling": "competition_rank",
                    },
                ),
            )
        ],
        selection_batch=SelectionBatchBlock(),
        training=TrainingBlock(policy={"cumulative_training": True}),
        ingest=IngestBlock(duplicate_policy="error"),
        scoring=ScoringBlock(score_batch_size=1000),
        safety=SafetyBlock(),
    )


class _NoLabels:
    kind = "test_no_labels"

    def training_labels(self, df, as_of_round, *, cumulative_training, dedup_policy):
        _unused = (df, as_of_round, cumulative_training, dedup_policy)
        return pd.DataFrame({"id": [], "y": [], "r": []})

    def labeled_id_set_leq_round(self, df, as_of_round):
        _unused = (df, as_of_round)
        return set()

    def labeled_id_set_any_round(self, df):
        _unused = df
        return set()


def test_round_plan_filters_unexpected_restriction_sites_before_selection(tmp_path) -> None:
    records_path = tmp_path / "records.parquet"
    df = pd.DataFrame(
        {
            "id": ["ok", "core_bamhi", "left_junction_ecori"],
            "sequence": [
                "A" * 60,
                "T" * 10 + "GGATCC" + "A" * 44,
                "AATTC" + "A" * 55,
            ],
            "bio_type": ["dna", "dna", "dna"],
            "alphabet": ["dna_4", "dna_4", "dna_4"],
            "X": [[0.1], [0.2], [0.3]],
        }
    )
    df.to_parquet(records_path, index=False)
    cfg = _cfg(tmp_path)
    store = RecordsStore(
        kind="local",
        records_path=records_path,
        campaign_slug="demo",
        x_col="X",
        y_col="Y",
        x_transform_name="identity",
        x_transform_params={},
    )

    plan = plan_round(store, df, cfg, 0, label_source=_NoLabels())

    assert plan.candidate_df["id"].astype(str).tolist() == ["ok"]
    assert plan.candidate_total_before_eligibility == 3
    assert plan.candidate_eligibility_filtered_out == 2
    assert plan.candidate_total_before_filter == 1
    assert plan.candidate_eligibility_reports[0]["rule"] == "restriction_site_exclusion"
    assert plan.candidate_eligibility_reports[0]["excluded_rows"] == 2


def test_restriction_site_exclusion_can_pre_exclude_non_synthesis_controls() -> None:
    frame = pd.DataFrame(
        {
            "id": ["candidate-a", "control-a"],
            "sequence": ["A" * 60, "G" * 165],
            "opal_candidate__design_family": ["ethanol", "control"],
        }
    )
    params = {
        **_eligibility_rule().params,
        "exclude_rows_where": [{"column": "id", "equals": "control-a"}],
    }

    result = restriction_site_exclusion(frame=frame, params=params)

    assert result.frame["id"].tolist() == ["candidate-a"]
    assert result.report["pre_excluded_rows"] == 1
    assert result.report["scanned_rows"] == 1
    assert result.report["restriction_site_excluded_rows"] == 0


def test_round_plan_fails_fast_when_eligibility_leaves_too_few_candidates(tmp_path) -> None:
    records_path = tmp_path / "records.parquet"
    df = pd.DataFrame(
        {
            "id": ["ok", "left_junction_ecori"],
            "sequence": ["A" * 60, "AATTC" + "A" * 55],
            "bio_type": ["dna", "dna"],
            "alphabet": ["dna_4", "dna_4"],
            "X": [[0.1], [0.2]],
        }
    )
    df.to_parquet(records_path, index=False)
    cfg = _cfg(tmp_path, min_remaining=2)
    store = RecordsStore(
        kind="local",
        records_path=records_path,
        campaign_slug="demo",
        x_col="X",
        y_col="Y",
        x_transform_name="identity",
        x_transform_params={},
    )

    try:
        plan_round(store, df, cfg, 0, label_source=_NoLabels())
    except Exception as exc:
        message = str(exc)
    else:
        raise AssertionError("expected restriction-site eligibility to fail fast")

    assert "restriction_site_exclusion" in message
    assert "min_remaining_candidates=2" in message
