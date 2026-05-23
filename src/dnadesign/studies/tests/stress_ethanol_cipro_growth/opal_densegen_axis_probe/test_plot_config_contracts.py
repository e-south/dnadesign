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
        "score_vs_rank_by_round",
        "score_threshold_over_rounds",
        "feature_importance_heatmap",
        "feature_importance_bars",
        "selected_vec8_summary",
        "fold_change_vs_logic_fidelity_latest",
        "sfxi_logic_fidelity_closeness_latest",
        "sfxi_factorial_effects_latest",
        "sfxi_setpoint_sweep_latest",
        "sfxi_support_diagnostics_latest",
        "sfxi_uncertainty_latest",
        "sfxi_intensity_scaling_latest",
    }
    assert {plot["kind"] for plot in payload["plots"]} == {
        "metric_over_rounds",
        "scatter_score_vs_rank",
        "percent_high_activity_over_rounds",
        "feature_importance_heatmap",
        "feature_importance_bars",
        "vector_summary_heatmap",
        "fold_change_vs_logic_fidelity",
        "sfxi_logic_fidelity_closeness",
        "sfxi_factorial_effects",
        "sfxi_setpoint_sweep",
        "sfxi_support_diagnostics",
        "sfxi_uncertainty",
        "sfxi_intensity_scaling",
    }
    assert plots_by_name["score_selected_over_rounds"]["params"]["top_k"] == 9
    assert plots_by_name["score_selected_over_rounds"]["params"]["highlight_round"] == "latest"
    assert plots_by_name["score_vs_rank_by_round"]["params"]["rank_mode"] == "competition"
    assert plots_by_name["score_threshold_over_rounds"]["params"]["metric"] == "pred__score_selected"
    assert plots_by_name["score_threshold_over_rounds"]["params"]["hue"] == "logic_fidelity"
    assert plots_by_name["score_threshold_over_rounds"]["params"]["highlight_round"] == "latest"
    heatmap_params = plots_by_name["feature_importance_heatmap"]["params"]
    assert heatmap_params["order_policy"] == "sort_index"
    assert heatmap_params["rasterized"] is True
    assert "top_n" not in heatmap_params
    assert "sort" not in heatmap_params
    assert plots_by_name["feature_importance_bars"]["params"]["order_policy"] == "sort_index"
    assert plots_by_name["selected_vec8_summary"]["params"]["reference_vector"] == [0, 0, 1, 1, 0, 0, 1, 1]
    assert plots_by_name["selected_vec8_summary"]["params"]["reference_label"] == "target vec8"
    assert plots_by_name["selected_vec8_summary"]["params"]["channel_labels"] == [
        "v00",
        "v10",
        "v01",
        "v11",
        "y00_star",
        "y10_star",
        "y01_star",
        "y11_star",
    ]
    for name in plot_names:
        if name.endswith("_latest"):
            assert plots_by_name[name]["round_selector"] == "latest"
            assert plots_by_name[name]["round_variants"] == ["latest", "each"]
        else:
            assert plots_by_name[name]["round_variants"] == ["all", "each"]
