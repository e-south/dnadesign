import dnadesign.opal as opal


def test_package_root_does_not_export_dashboard_helpers() -> None:
    prohibited = {
        "campaign_label_from_path",
        "diagnostics_to_lines",
        "find_repo_root",
        "list_campaign_paths",
        "load_campaign_selection",
        "load_parquet_cached",
    }

    assert prohibited.isdisjoint(set(opal.__all__))
    for name in prohibited:
        assert not hasattr(opal, name)


def test_package_root_does_not_export_generated_notebook_components() -> None:
    prohibited = {
        "build_notebook_artifact_garden_rows",
        "build_notebook_at_a_glance_rows",
        "build_notebook_baserender_contract",
        "build_notebook_plot_card_rows",
        "build_notebook_plot_method_rows",
        "build_notebook_visual_surface_model",
        "render_notebook_baserender_record",
        "resolve_notebook_round_default",
    }

    assert prohibited.isdisjoint(set(opal.__all__))
    for name in prohibited:
        assert not hasattr(opal, name)
