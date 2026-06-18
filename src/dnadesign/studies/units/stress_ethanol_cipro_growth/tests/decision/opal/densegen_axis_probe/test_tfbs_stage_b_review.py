"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/densegen_axis_probe/test_tfbs_stage_b_review.py

Regression tests for TFBS stage b review studies units stress ethanol.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from .helpers import _write_stage_b_review_fixture
from .probe_modules import probe_module

build_tfbs_stage_b_realized_label_review = probe_module(
    "tfbs.stage_b.review.materialization"
).build_tfbs_stage_b_realized_label_review
main = probe_module("cli").main
review_plot_text = probe_module("tfbs.stage_b.review.plots.display_text")
materialize_tfbs_stage_b_realized_review_plots = probe_module(
    "tfbs.stage_b.review.plots.materialization"
).materialize_tfbs_stage_b_realized_review_plots


def test_stage_b_realized_review_reports_true_label_lift_and_pair_deltas(tmp_path: Path) -> None:
    manifest_path = _write_stage_b_review_fixture(tmp_path)

    result = build_tfbs_stage_b_realized_label_review(manifest_path)

    trajectories = pd.read_csv(result.trajectory_csv_path)
    pairs = pd.read_csv(result.pair_summary_csv_path)
    summary = json.loads(result.summary_json_path.read_text(encoding="utf-8"))

    positive_final = trajectories.query("label_name == 'lexA_present' and oracle_role == 'positive' and round == 1")
    null_final = trajectories.query("label_name == 'lexA_present' and oracle_role == 'matched_null' and round == 1")
    assert positive_final["selected_true_mean"].iloc[0] == pytest.approx(1.0)
    assert positive_final["pool_baseline"].iloc[0] == pytest.approx(0.5)
    assert positive_final["selected_true_lift_ratio"].iloc[0] == pytest.approx(2.0)
    assert positive_final["same_batch_top_label_mean"].iloc[0] == pytest.approx(1.0)
    assert positive_final["same_batch_top_lift_ratio"].iloc[0] == pytest.approx(2.0)
    assert positive_final["seed_label_count"].iloc[0] == 2
    assert positive_final["seed_true_mean"].iloc[0] == pytest.approx(0.5)
    assert positive_final["round_zero_semantics"].iloc[0] == "first_model_selected_batch_after_seed_labels"
    assert null_final["selected_true_mean"].iloc[0] == pytest.approx(0.5)
    assert null_final["selected_true_lift_ratio"].iloc[0] == pytest.approx(1.0)

    pair = pairs.iloc[0].to_dict()
    assert pair["label_name"] == "lexA_present"
    assert pair["positive_final_lift_ratio"] == pytest.approx(2.0)
    assert pair["null_final_lift_ratio"] == pytest.approx(1.0)
    assert pair["final_positive_minus_null_lift_ratio"] == pytest.approx(1.0)
    assert pair["positive_final_selected_true_sum"] == pytest.approx(2.0)
    assert pair["positive_final_selected_count"] == 2
    assert pair["positive_mean_round_lift_ratio"] > pair["null_mean_round_lift_ratio"]
    assert pair["positive_trapezoid_auc_lift_ratio"] == pytest.approx(1.5)
    assert pair["trapezoid_auc_positive_minus_null_lift_ratio"] == pytest.approx(1.0)
    assert pair["peer_review_claim_status"] == "positive_exceeds_null"
    assert summary["status"] == "PASS"
    assert summary["campaign_count"] == 2
    assert summary["pair_count"] == 1
    assert summary["target_profile"]["profile_id"] == "custom_tfbs_learnability_label_set"
    assert summary["target_profile"]["canonical"] is False
    assert Path(summary["plot_manifest_json_path"]).exists()
    assert Path(summary["claim_assessment_csv_path"]).exists()
    claims = pd.read_csv(summary["claim_assessment_csv_path"])
    assert claims["claim_readiness_status"].tolist() == ["READY_AS_VALID_NULL_LEARNABILITY_SIGNAL"]
    assert summary["claim_readiness"]["ready_claim_count"] == 1
    assert summary["claim_readiness"]["blocked_or_limited_claim_count"] == 0
    plot_manifest = json.loads(Path(summary["plot_manifest_json_path"]).read_text(encoding="utf-8"))
    assert plot_manifest["plot_count"] == 2
    assert plot_manifest["style_contract"] == {
        "axis_style": "stress_ethanol_cipro_growth.tfbs_review_axis.v1",
        "axes_facecolor": "white",
        "grid": "light_gray_background_grid_lines",
        "visible_spines": ["left", "bottom"],
        "tick_style": "styled_outward_ticks",
        "font_scale": "unified_review_body_font_for_ticks_axes_subtitle_legend",
        "title_anchor": "axes_center",
        "square_axes": "where_data_shape_supports_it",
        "trajectory_axes": "square",
        "trajectory_reference_lines": ["pool_average", "best_possible_single_batch_reference"],
    }
    assert plot_manifest["text_contract"] == {
        "baseline": "No enrichment: selected mean equals the same label-table pool mean",
        "count_fraction_label": (
            "count_fraction label = target TFBS count / 3 per sequence; plotted values are enrichment ratios, "
            "not raw counts"
        ),
        "initial_batch": (
            "diamond markers are the same initial seed-batch IDs scored by each label table before round 0"
        ),
        "interval": "mean plus/minus sample SD across seed runs; n is recorded; not an inferential CI",
        "legend_layout": "legend below the plot; wrap when needed to avoid clipping",
        "pairing": (
            "sequence-matched metadata and control campaigns share initial selected IDs; only the label table differs"
        ),
        "role_labels": "sequence-matched metadata versus profile-appropriate matched control",
        "selected_label_values": (
            "selected_true_* artifact columns are selected values from that campaign's label table; for shuffled "
            "controls this is a control-label value, not post hoc sequence-matched metadata truth"
        ),
        "trajectory_semantics": (
            "line points are per-round top-k selected batches; round 0 is the first acquired batch after the "
            "initial seed-batch IDs, not the initial seed batch itself"
        ),
        "same_batch_top_k_reference": (
            "Best possible single batch: mean of the top selection_k label values in the same label table divided by "
            "the same label-table pool mean. This is a full-pool reference, not an observed campaign and not the "
            "multi-round same-budget known-label ranking."
        ),
        "subtitle_layout": "centered single-line subtitle",
        "title_alignment": "title centered over the axes frame; title may wrap, subtitle must not wrap",
        "type_scale": "axis labels, tick labels, subtitle, and legend use the same review body size",
    }
    assert [plot["kind"] for plot in plot_manifest["plots"]] == [
        "realized_label_lift_trajectory",
        "positive_null_lift_summary",
    ]
    assert plot_manifest["plots"][0]["title"] == "Active selection enriches LexA presence over row-shuffled control"
    assert "selected-batch enrichment vs pool" in plot_manifest["plots"][0]["alt_text"]
    assert "Round 0 follows the initial seed batch" in plot_manifest["plots"][0]["alt_text"]
    assert "best possible single batch" in plot_manifest["plots"][0]["alt_text"]
    assert "active-learning rounds" not in plot_manifest["plots"][0]["alt_text"]
    assert "LexA presence" in plot_manifest["plots"][0]["alt_text"]
    assert all("each sentinel label" not in plot["alt_text"] for plot in plot_manifest["plots"])
    visible_text = " ".join(str(plot[field]) for plot in plot_manifest["plots"] for field in ("title", "alt_text"))
    assert "oracle" not in visible_text.lower()
    assert "pool baseline" not in visible_text.lower()
    assert "densegen label" not in visible_text.lower()
    assert "same-batch top-k" not in visible_text.lower()


def test_stage_b_realized_review_fails_fast_on_selected_ids_missing_from_label_table(tmp_path: Path) -> None:
    manifest_path = _write_stage_b_review_fixture(tmp_path, include_missing_selection_id=True)

    with pytest.raises(ValueError, match="selected id"):
        build_tfbs_stage_b_realized_label_review(manifest_path)


def test_stage_b_review_cli_writes_realized_label_artifacts(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    manifest_path = _write_stage_b_review_fixture(tmp_path)

    assert main(["tfbs-stage-b-review", "--config-manifest", str(manifest_path), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "PASS"
    assert Path(payload["trajectory_csv_path"]).exists()
    assert Path(payload["pair_summary_csv_path"]).exists()
    assert Path(payload["plot_manifest_json_path"]).exists()
    assert payload["notebook_visual_registration"]["realized_label_review"]["status"] == "SKIPPED_INDEX_NOT_FOUND"
    assert payload["notebook_visual_registration"]["slot_count_diagnostics"]["status"] == "SKIPPED_NO_SLOT_LABELS"


def test_stage_b_realized_review_registers_notebook_collection_visuals(tmp_path: Path) -> None:
    stage_b_root = tmp_path / "stage_b"
    manifest_path = _write_stage_b_review_fixture(stage_b_root / "manifests")
    visual_index_path = stage_b_root / "notebooks" / "collection_visuals" / "collection_visual_manifest.json"
    visual_index_path.parent.mkdir(parents=True)
    visual_index_path.write_text(
        json.dumps(
            {
                "schema_version": "opal.collection_visual_manifest_index.v1",
                "generated_at": "2026-06-01T00:00:00+00:00",
                "collection_id": "fixture",
                "output_dir": str(visual_index_path.parent),
                "comparison_set_count": 0,
                "comparison_sets": [],
                "visual_count": 0,
                "visuals": [],
            }
        ),
        encoding="utf-8",
    )

    result = build_tfbs_stage_b_realized_label_review(manifest_path)

    summary = json.loads(result.summary_json_path.read_text(encoding="utf-8"))
    refreshed = json.loads(visual_index_path.read_text(encoding="utf-8"))
    assert summary["notebook_visual_registration"]["realized_label_review"]["status"] == "REGISTERED"
    assert summary["notebook_visual_registration"]["slot_count_diagnostics"]["status"] == "SKIPPED_NO_SLOT_LABELS"
    assert refreshed["comparison_set_count"] == 1
    assert refreshed["visual_count"] == 2
    assert refreshed["comparison_sets"][0]["key"] == "stage_b_realized_label_review__lexA_present"
    assert refreshed["comparison_sets"][0]["label"] == (
        "LexA presence: Sequence-matched metadata vs row-shuffled control"
    )
    assert {visual["surface_kind"] for visual in refreshed["visuals"]} == {"study_realized_label_review"}
    assert {visual["target_label"] for visual in refreshed["visuals"]} == {"LexA presence"}
    assert all("each sentinel label" not in visual["alt_text"] for visual in refreshed["visuals"])
    assert {visual["metric"] for visual in refreshed["visuals"]} == {
        "selected_label_lift_ratio",
        "positive_minus_null_lift_ratio",
    }
    assert all(Path(visual["path"]).exists() for visual in refreshed["visuals"])

    build_tfbs_stage_b_realized_label_review(manifest_path)
    rerun = json.loads(visual_index_path.read_text(encoding="utf-8"))
    assert rerun["visual_count"] == 2
    assert len(rerun["visuals"]) == 2


def test_stage_b_review_plot_display_text_is_manuscript_safe() -> None:
    assert (
        review_plot_text.role_display_label("positive", label_name="baeR_count_fraction") == "Sequence-matched metadata"
    )
    assert (
        review_plot_text.role_display_label("matched_null", label_name="baeR_count_fraction") == "Row-shuffled control"
    )
    assert (
        review_plot_text.role_display_label(
            "matched_null",
            label_name="lexA_in_slot0",
            control_role="count_fixed_shuffled_slot_negative_control",
        )
        == "Slot-shuffled control"
    )
    assert review_plot_text.trajectory_plot_title("baeR_count_fraction", replicate_count=3) == (
        "Active selection enriches BaeR count-fraction over row-shuffled control"
    )
    assert review_plot_text.label_definition("baeR_count_fraction") == (
        "BaeR count-fraction = BaeR count / 3 per sequence"
    )
    assert review_plot_text.trajectory_y_axis_label("baeR_count_fraction") == (
        r"Enrichment vs pool ($\bar{y}_{sel}/\bar{y}_{pool}$)"
    )
    assert review_plot_text.enrichment_formula_text("baeR_count_fraction") == (
        "y = selected mean fraction / same label-table pool mean fraction"
    )
    assert review_plot_text.trajectory_plot_subtitle("baeR_count_fraction", replicate_count=3) == ""
    assert (
        review_plot_text.trajectory_plot_subtitle(
            "lexA_in_slot0",
            replicate_count=3,
            control_role="count_fixed_shuffled_slot_negative_control",
        )
        == ""
    )
    assert "\n" not in review_plot_text.trajectory_plot_subtitle("baeR_count_fraction", replicate_count=3)
    assert review_plot_text.seed_run_sample_sd_label(replicate_count=3) == "Mean +/- SD (n=3)"
    assert review_plot_text.seed_pair_sample_sd_label(replicate_count=3) == "Mean +/- SD (n=3)"
    assert review_plot_text.positive_null_summary_title("baeR_count_fraction", replicate_count=3) == (
        "Sequence-matched metadata beats row-shuffled control for BaeR count-fraction"
    )
    assert review_plot_text.positive_null_summary_subtitle(replicate_count=3) == ""
    assert "\n" not in review_plot_text.positive_null_summary_subtitle(replicate_count=3)
    trajectory_alt = review_plot_text.plot_manifest_alt_text(
        "realized_label_lift_trajectory",
        label_name="baeR_count_fraction",
        replicate_count=3,
        control_role="matched_label_permutation_negative_control",
    )
    summary_alt = review_plot_text.plot_manifest_alt_text(
        "positive_null_lift_summary",
        label_name="baeR_count_fraction",
        replicate_count=3,
    )
    assert "selected-batch enrichment vs pool" in trajectory_alt
    assert "Round 0 follows the initial seed batch" in trajectory_alt
    assert "Lines show sequence-matched metadata and row-shuffled control" in trajectory_alt
    assert "best possible single batch" in trajectory_alt
    assert "not CI" in trajectory_alt
    assert len(trajectory_alt) < 240
    assert "final round and trajectory AUC" in summary_alt
    assert len(summary_alt) < 170
    assert review_plot_text.TRAILING_TRAJECTORY_NOTE.startswith("Faint lines = seed runs; bold line = mean")
    assert "diamond = initial seed batch" in review_plot_text.TRAILING_TRAJECTORY_NOTE
    assert review_plot_text.NO_ENRICHMENT_BASELINE_LABEL == "Pool average"
    assert review_plot_text.SAME_BATCH_TOP_K_REFERENCE_LABEL == "Best possible batch"


def test_stage_b_review_plots_record_sample_sd_interval_when_seed_replicates_exist(tmp_path: Path) -> None:
    trajectory_path = tmp_path / "trajectory.csv"
    pair_summary_path = tmp_path / "pair_summary.csv"
    pd.DataFrame(
        {
            "campaign_key": [
                "positive_s1",
                "positive_s2",
                "positive_s1",
                "positive_s2",
                "matched_null_s1",
                "matched_null_s2",
                "matched_null_s1",
                "matched_null_s2",
            ],
            "label_name": ["lexA_present"] * 8,
            "oracle_role": ["positive", "positive", "positive", "positive"]
            + ["matched_null", "matched_null", "matched_null", "matched_null"],
            "seed": [1, 2, 1, 2, 1, 2, 1, 2],
            "round": [0, 0, 1, 1, 0, 0, 1, 1],
            "same_batch_top_lift_ratio": [2.5] * 8,
            "selected_true_lift_ratio": [1.2, 1.6, 2.0, 2.4, 1.0, 1.1, 1.2, 1.4],
            "seed_true_lift_ratio": [0.9, 1.1, 0.9, 1.1, 1.0, 1.2, 1.0, 1.2],
        }
    ).to_csv(trajectory_path, index=False)
    pd.DataFrame(
        {
            "label_name": ["lexA_present", "lexA_present"],
            "final_positive_minus_null_lift_ratio": [0.8, 1.0],
            "trapezoid_auc_positive_minus_null_lift_ratio": [0.6, 0.9],
        }
    ).to_csv(pair_summary_path, index=False)

    manifest_path = materialize_tfbs_stage_b_realized_review_plots(
        trajectory_csv_path=trajectory_path,
        pair_summary_csv_path=pair_summary_path,
        out_dir=tmp_path / "plots",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_kind = {plot["kind"]: plot for plot in manifest["plots"]}
    trajectory_interval = by_kind["realized_label_lift_trajectory"]["interval"]
    summary_interval = by_kind["positive_null_lift_summary"]["interval"]
    assert trajectory_interval == {
        "applies_to": "selected label lift ratio by label source and round",
        "estimator": "mean_plus_minus_sample_standard_deviation",
        "is_confidence_interval": False,
        "kind": "sample_sd",
        "replicate_count": 2,
        "status": "available",
        "unit": "seed replicate",
    }
    assert summary_interval == {
        "applies_to": "sequence-matched-minus-control lift summary",
        "estimator": "mean_plus_minus_sample_standard_deviation",
        "is_confidence_interval": False,
        "kind": "sample_sd",
        "replicate_count": 2,
        "status": "available",
        "unit": "sequence-matched/control seed pair",
    }


def test_stage_b_realized_review_fails_on_invalid_notebook_visual_index_schema(tmp_path: Path) -> None:
    manifest_path = _write_stage_b_review_fixture(tmp_path)
    visual_index_path = tmp_path / "bad_collection_visual_manifest.json"
    visual_index_path.write_text(
        json.dumps(
            {
                "schema_version": "opal.collection_visual_manifest_index.v0",
                "comparison_sets": [],
                "visuals": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Unsupported OPAL collection visual index schema"):
        build_tfbs_stage_b_realized_label_review(
            manifest_path,
            collection_visual_index_path=visual_index_path,
        )
