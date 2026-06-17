from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from .helpers import _dark_edge_pixel_count
from .probe_modules import probe_module

contracts = probe_module("tfbs.stage_b.learning_loop_baselines.contracts")
frames = probe_module("tfbs.stage_b.learning_loop_baselines.frames")
plots = probe_module("tfbs.stage_b.learning_loop_baselines.plots.renderers")
replay = probe_module("tfbs.stage_b.learning_loop_baselines.replay")


def test_frozen_rank_chunks_are_deterministic_and_exclude_seed_ids() -> None:
    scores = pd.DataFrame(
        {
            "id": ["seed", "b", "a", "c", "d"],
            "score": [0.99, 0.8, 0.8, 0.6, 0.4],
        }
    )

    chunks = replay.frozen_rank_chunks(
        scores,
        selection_k=2,
        rounds=2,
        excluded_ids={"seed"},
    )

    assert chunks[["round", "id", "frozen_rank", "score"]].to_dict(orient="records") == [
        {"round": 0, "id": "b", "frozen_rank": 1, "score": 0.8},
        {"round": 0, "id": "a", "frozen_rank": 2, "score": 0.8},
        {"round": 1, "id": "c", "frozen_rank": 3, "score": 0.6},
        {"round": 1, "id": "d", "frozen_rank": 4, "score": 0.4},
    ]


def test_frozen_rank_chunks_fail_fast_on_duplicate_or_insufficient_ids() -> None:
    duplicate_scores = pd.DataFrame({"id": ["a", "a"], "score": [0.3, 0.2]})

    with pytest.raises(ValueError, match="duplicate"):
        replay.frozen_rank_chunks(duplicate_scores, selection_k=1, rounds=1, excluded_ids=set())

    short_scores = pd.DataFrame({"id": ["a"], "score": [0.3]})
    with pytest.raises(ValueError, match="insufficient"):
        replay.frozen_rank_chunks(short_scores, selection_k=2, rounds=1, excluded_ids=set())


def test_top_budget_chunks_are_deterministic_and_exclude_seed_ids() -> None:
    labels = pd.DataFrame(
        {
            "id": ["seed", "b", "a", "c", "d"],
            "target": [1.0, 0.8, 0.8, 0.6, 0.4],
        }
    )

    chunks = replay.top_budget_chunks(
        labels,
        label_name="target",
        selection_k=2,
        rounds=2,
        excluded_ids={"seed"},
    )

    assert chunks[["round", "id", "top_budget_rank", "label_value"]].to_dict(orient="records") == [
        {"round": 0, "id": "b", "top_budget_rank": 1, "label_value": 0.8},
        {"round": 0, "id": "a", "top_budget_rank": 2, "label_value": 0.8},
        {"round": 1, "id": "c", "top_budget_rank": 3, "label_value": 0.6},
        {"round": 1, "id": "d", "top_budget_rank": 4, "label_value": 0.4},
    ]
    assert chunks["selection_source"].unique().tolist() == ["top_budget_ceiling"]


def test_cumulative_lift_trajectory_uses_acquired_budget_not_round_rows() -> None:
    selections = pd.DataFrame(
        {
            "round": [0, 0, 1, 1],
            "id": ["a", "b", "c", "d"],
            "selection_source": ["frozen_round0"] * 4,
        }
    )
    labels = pd.DataFrame(
        {
            "id": ["a", "b", "c", "d", "e"],
            "target": [1.0, 0.0, 1.0, 1.0, 0.0],
        }
    )

    trajectory = frames.cumulative_lift_trajectory(
        selections,
        labels,
        label_name="target",
        pool_baseline=0.4,
        campaign_key="campaign",
        oracle_role="positive",
        seed=7,
        selection_k=2,
    )

    assert trajectory[["round", "cumulative_selected_count", "cumulative_label_sum"]].to_dict(orient="records") == [
        {"round": 0, "cumulative_selected_count": 2, "cumulative_label_sum": 1.0},
        {"round": 1, "cumulative_selected_count": 4, "cumulative_label_sum": 3.0},
    ]
    assert trajectory["scientific_control_role"].tolist() == ["", ""]
    assert trajectory["cumulative_label_mean"].tolist() == pytest.approx([0.5, 0.75])
    assert trajectory["cumulative_lift_ratio"].tolist() == pytest.approx([1.25, 1.875])


def test_claim_interpretation_reports_top_budget_signal_recovery() -> None:
    rows = []
    sources = {
        "active_retraining": 2.5,
        "frozen_round0": 1.5,
        "top_budget_ceiling": 3.0,
    }
    for campaign_key, oracle_role in [("positive", "positive"), ("control", "matched_null")]:
        for source, lift in sources.items():
            rows.append(
                {
                    "campaign_key": campaign_key,
                    "label_name": "target",
                    "oracle_role": oracle_role,
                    "seed": 7,
                    "selection_source": source,
                    "round": 1,
                    "cumulative_selected_count": 12,
                    "cumulative_label_mean": 0.5,
                    "pool_baseline": 0.2,
                    "cumulative_lift_ratio": lift,
                }
            )
    endpoint = frames.endpoint_summary_frame(
        pd.DataFrame(rows),
        pairs=[
            {
                "label_name": "target",
                "seed": 7,
                "positive_campaign_key": "positive",
                "null_campaign_key": "control",
            }
        ],
    )

    interpretation = frames.claim_interpretation_frame(endpoint)

    row = interpretation.iloc[0]
    assert row["top_budget_final_cumulative_lift_mean"] == pytest.approx(3.0)
    assert row["active_fraction_of_top_budget_final_lift_mean"] == pytest.approx(2.5 / 3.0)
    assert row["active_fraction_of_top_budget_gain_recovered_mean"] == pytest.approx((2.5 - 1.0) / (3.0 - 1.0))


def test_learning_loop_plots_use_square_review_panels_without_edge_clipping(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "trajectory.csv"
    endpoint_summary_path = tmp_path / "endpoint_summary.csv"
    claim_path = tmp_path / "claim_interpretation.csv"

    trajectory_rows = []
    for label_name in ("baeR_count_fraction", "cpxR_count_fraction", "lexA_count_fraction"):
        for seed in (7, 17, 29):
            for selection_source, oracle_role, lift in (
                ("active_retraining", "positive", 3.0 + seed / 100),
                ("frozen_round0", "positive", 1.5 + seed / 100),
                ("top_budget_ceiling", "positive", 5.0),
                ("active_retraining", "matched_null", 1.2),
                ("frozen_round0", "matched_null", 1.0),
            ):
                for round_index, acquired in enumerate((6, 12, 18)):
                    trajectory_rows.append(
                        {
                            "campaign_key": f"{label_name}_{oracle_role}_{seed}",
                            "label_name": label_name,
                            "oracle_role": oracle_role,
                            "scientific_control_role": (
                                "count_fixed_shuffled_slot_negative_control" if oracle_role == "matched_null" else ""
                            ),
                            "seed": seed,
                            "selection_source": selection_source,
                            "round": round_index,
                            "cumulative_selected_count": acquired,
                            "cumulative_label_mean": 0.2,
                            "pool_baseline": 0.1,
                            "cumulative_lift_ratio": lift + round_index / 10,
                        }
                    )
    pd.DataFrame(trajectory_rows).to_csv(trajectory_path, index=False)
    assert plots._control_roles(pd.DataFrame(trajectory_rows)) == (  # noqa: SLF001
        "matched_null",
        "count_fixed_shuffled_slot_negative_control",
    )
    pd.DataFrame({"source": ["nonempty"]}).to_csv(endpoint_summary_path, index=False)
    pd.DataFrame(
        {
            "label_name": ["baeR_count_fraction", "cpxR_count_fraction", "lexA_count_fraction"],
            "active_minus_frozen_final_cumulative_lift_mean": [2.3, 1.8, 2.1],
            "active_minus_frozen_final_cumulative_lift_sample_sd": [0.3, 0.2, 0.25],
            "active_fraction_of_top_budget_gain_recovered_mean": [0.49, 0.36, 0.81],
            "active_fraction_of_top_budget_gain_recovered_sample_sd": [0.17, 0.06, 0.07],
        }
    ).to_csv(claim_path, index=False)

    manifest_path = plots.materialize_frozen_replay_plots(
        trajectory_csv_path=trajectory_path,
        endpoint_summary_csv_path=endpoint_summary_path,
        claim_interpretation_csv_path=claim_path,
        out_dir=tmp_path / "plots",
        spec=contracts.COUNT_FRACTION_LEARNING_LOOP_SPEC,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["style_contract"]["subplot_layout"] == "single_row_square_panels_for_label_trajectories"
    assert manifest["style_contract"]["trajectory_reference_series"] == [
        "pool_baseline",
        "same_budget_top_label_reference",
    ]
    by_kind = {plot["kind"]: plot for plot in manifest["plots"]}
    cumulative_text = " ".join(
        str(by_kind["frozen_round0_cumulative_enrichment"][field]) for field in ("title", "caption", "alt_text")
    )
    assert "best possible same-budget ceiling" in cumulative_text
    assert "theoretical maximum" not in cumulative_text
    assert by_kind["top_budget_signal_recovery"]["title"] == "Fraction of achievable enrichment recovered"

    cumulative = Image.open(by_kind["frozen_round0_cumulative_enrichment"]["path"]).convert("RGB")
    assert cumulative.size[0] > cumulative.size[1] * 1.75
    assert _dark_edge_pixel_count(cumulative) == 0

    endpoint = Image.open(by_kind["frozen_round0_endpoint_adaptive_gain"]["path"]).convert("RGB")
    ceiling = Image.open(by_kind["top_budget_signal_recovery"]["path"]).convert("RGB")
    assert endpoint.size == (1152, 1152)
    assert ceiling.size == (1152, 1152)
    assert _dark_edge_pixel_count(endpoint) == 0
    assert _dark_edge_pixel_count(ceiling) == 0
