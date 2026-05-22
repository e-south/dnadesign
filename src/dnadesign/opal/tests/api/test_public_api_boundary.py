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
