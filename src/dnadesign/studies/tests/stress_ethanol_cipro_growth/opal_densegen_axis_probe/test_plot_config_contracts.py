from __future__ import annotations

from .helpers import ORACLE_ID, Path, RunSpec, _write_campaign_plot_config, yaml


def test_scratch_campaign_plot_config_declares_round_dogfood_primitives(tmp_path: Path) -> None:
    config_path = tmp_path / "campaign" / "configs" / "campaign.yaml"
    config_path.parent.mkdir(parents=True)
    run = RunSpec(
        campaign_key="cipro",
        oracle_id=ORACLE_ID,
        split_id="random_id",
        run_key="cipro_positive_random_id",
        target_class="cipro_only",
        workdir=tmp_path / "campaign",
        config_path=config_path,
        label_input_path=tmp_path / "labels.parquet",
        sidecar_path=tmp_path / "observed_labels.parquet",
        selection_k=9,
    )

    _write_campaign_plot_config(run)

    payload = yaml.safe_load((config_path.parent / "plots.yaml").read_text(encoding="utf-8"))
    plots_by_name = {plot["name"]: plot for plot in payload["plots"]}
    plot_names = {plot["name"] for plot in payload["plots"]}
    assert payload["plot_defaults"]["output"]["save_data"] is True
    assert payload["plot_defaults"]["output"]["format"] == "png"
    assert plot_names == {
        "score_selected_over_rounds",
        "score_vs_rank_over_rounds",
        "score_threshold_over_rounds",
        "feature_importance_heatmap",
        "feature_importance_bars",
        "selected_target_vector_summary",
    }
    assert {plot["kind"] for plot in payload["plots"]} == {
        "metric_over_rounds",
        "scatter_score_vs_rank",
        "percent_high_activity_over_rounds",
        "feature_importance_heatmap",
        "feature_importance_bars",
        "vector_summary_heatmap",
    }
    assert plots_by_name["score_selected_over_rounds"]["params"]["cohort"] == "selected"
    assert plots_by_name["score_selected_over_rounds"]["params"]["summaries"] == ["median", "q25", "q75", "count"]
    assert plots_by_name["score_selected_over_rounds"]["params"]["band"] == "iqr"
    assert "highlight_round" not in plots_by_name["score_selected_over_rounds"]["params"]
    assert plots_by_name["score_vs_rank_over_rounds"]["round_selector"] == "all"
    assert plots_by_name["score_vs_rank_over_rounds"]["params"]["rank_mode"] == "competition"
    assert plots_by_name["score_threshold_over_rounds"]["params"]["metric"] == "pred__score_selected"
    assert plots_by_name["score_threshold_over_rounds"]["params"]["threshold_quantile"] == 0.9
    assert plots_by_name["score_threshold_over_rounds"]["params"]["mode"] == "line"
    assert "highlight_round" not in plots_by_name["score_threshold_over_rounds"]["params"]
    heatmap_params = plots_by_name["feature_importance_heatmap"]["params"]
    assert heatmap_params["order_policy"] == "sort_index"
    assert heatmap_params["rasterized"] is True
    assert heatmap_params["cmap"] == "opal_importance"
    assert heatmap_params["colorbar_label"] == "rf_feature_importance"
    assert heatmap_params["figsize_in"] == [14.0, 4.4]
    assert heatmap_params["max_xticks"] == 16
    assert heatmap_params["contrast_gamma"] == 0.55
    assert "top_n" not in heatmap_params
    assert "sort" not in heatmap_params
    assert plots_by_name["feature_importance_bars"]["params"]["order_policy"] == "sort_index"
    assert plots_by_name["feature_importance_bars"]["params"]["figsize_in"] == [14.0, 4.4]
    assert plots_by_name["feature_importance_bars"]["params"]["cmap"] == "round_progression"
    assert plots_by_name["selected_target_vector_summary"]["params"]["reference_vector"] == [0, 0, 1, 1]
    assert plots_by_name["selected_target_vector_summary"]["params"]["reference_label"] == "Target vector"
    assert plots_by_name["selected_target_vector_summary"]["params"]["reference_mse_panel"] is True
    assert plots_by_name["selected_target_vector_summary"]["params"]["cmap"] == "opal_seafoam"
    assert plots_by_name["selected_target_vector_summary"]["params"]["channel_labels"] == [
        "v00",
        "v10",
        "v01",
        "v11",
    ]
    for name in plot_names:
        if name.endswith("_latest"):
            assert plots_by_name[name]["round_selector"] == "latest"
            assert plots_by_name[name]["round_variants"] == "each"
            assert "by round" in plots_by_name[name]["params"]["title"].lower()
        else:
            assert plots_by_name[name]["round_selector"] == "all"
            assert "round_variants" not in plots_by_name[name]
