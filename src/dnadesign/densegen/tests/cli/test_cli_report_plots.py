"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/cli/test_cli_report_plots.py

CLI guardrails for DenseGen notebook command surfaces.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import textwrap
from base64 import b64decode
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dnadesign.densegen.src.cli.main import app
from dnadesign.densegen.tests.config_fixtures import write_minimal_config


@pytest.fixture(autouse=True)
def _clear_codex_sandbox_env(monkeypatch) -> None:
    monkeypatch.delenv("CODEX_SANDBOX", raising=False)


@pytest.fixture(autouse=True)
def _bypass_notebook_plot_completeness_for_server_plumbing_tests(request, monkeypatch) -> None:
    if not request.node.name.startswith("test_notebook_run_"):
        return
    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(
        notebook_commands,
        "_assert_notebook_plot_inventory_is_complete",
        lambda **_kwargs: None,
    )


_TINY_PNG = b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+tm6kAAAAASUVORK5CYII=")


def _write_plot(path: Path, *, content: bytes | str = _TINY_PNG) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(content, bytes):
        path.write_bytes(content)
    else:
        path.write_text(content)


def _write_notebook_surface_inventory(plot_root: Path, *, suffix: str = ".png") -> None:
    plot_root.mkdir(parents=True, exist_ok=True)
    entries = [
        ("source_cohort_concentration", f"dataset/source_cohort_concentration{suffix}"),
        ("stage_a_sampling_yield", f"stage_a/stage_a_sampling_yield{suffix}"),
        ("stage_a_pool_diversity", f"stage_a/stage_a_pool_diversity{suffix}"),
        ("plan_regulator_deployment_heatmap", f"stage_b_summary/plan_regulator_deployment_heatmap{suffix}"),
        ("placement_occupancy_map", f"stage_b/demo_plan/demo_input/placement_occupancy_map{suffix}"),
        ("retained_pool_coverage_by_regulator", f"stage_b_summary/retained_pool_coverage_by_regulator{suffix}"),
        ("attempt_outcome_timeline", f"run_health/attempt_outcome_timeline{suffix}"),
        ("solve_pressure_and_progress", f"run_health/solve_pressure_and_progress{suffix}"),
        ("background_sequence_logo", f"stage_a/demo_input__background_sequence_logo{suffix}"),
        ("compression_ratio_by_plan", f"run_health/compression_ratio_by_plan{suffix}"),
        ("dense_array_showcase_video", "stage_b/all_plans/showcase.mp4"),
        (
            "retained_vs_deployed_length_mix_by_regulator",
            f"stage_b_summary/retained_vs_deployed_length_mix_by_regulator{suffix}",
        ),
        (
            "retained_vs_deployed_tier_mix_by_regulator",
            f"stage_b_summary/retained_vs_deployed_tier_mix_by_regulator{suffix}",
        ),
        (
            "score_strata_and_deployed_length_bridge",
            f"stage_b_summary/score_strata_and_deployed_length_bridge{suffix}",
        ),
        ("source_plan_input_heatmap", f"dataset/source_plan_input_heatmap{suffix}"),
        ("stage_a_pool_score_strata", f"stage_a/stage_a_pool_score_strata{suffix}"),
        ("tfbs_concentration_profile", f"stage_b/demo_plan/demo_input/tfbs_concentration_profile{suffix}"),
        (
            "upstream_motif_supply_and_pwm_strength",
            f"stage_b_summary/upstream_motif_supply_and_pwm_strength{suffix}",
        ),
    ]
    for _plot_id, _rel_path in entries:
        _write_plot(plot_root / _rel_path)
    (plot_root / "current_inventory.json").write_text(
        json.dumps(
            {
                "schema_version": "densegen.current_inventory.v2",
                "plots": [{"plot_id": _plot_id, "path": _rel_path} for _plot_id, _rel_path in entries],
            }
        )
    )


def test_report_command_removed() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["report", "--help"])
    assert result.exit_code != 0
    assert "No such command 'report'" in result.output


def test_notebook_generate_writes_workspace_notebook(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    assert notebook_path.exists()
    content = notebook_path.read_text()
    expected_literals = [
        'app = marimo.App(width="medium")',
        "workspace_heading = ",
        "workspace_run_details_payload = ",
        "from io import BytesIO",
        "import shutil",
        "import subprocess",
        "import tempfile",
        "from dnadesign.baserender import adapt_records",
        "from dnadesign.baserender import render_record_figure",
        "load_current_inventory_strict",
        "def consume_click(click_count: int, last_handled: int) -> tuple[bool, int]:",
        "def load_preview_records(",
        'preview_strategy = str(preview_payload["preview_strategy"])',
        'record_plan_filter = mo.ui.dropdown(options=_plan_options, value=_default_plan_value, label="Record plan")',
        "Record plan filter applies to preview rows only within this head-window.",
        "get_records_export_handled_click",
        "get_baserender_export_handled_click",
        "get_plot_export_handled_click",
        "active_baserender_display_payload = {",
        "def build_baserender_request(*, record, preview_row):",
        "def build_baserender_figure(request: dict[str, object]):",
        "_buffer = BytesIO()",
        "@lru_cache(maxsize=64)",
        "def render_baserender_preview_image(",
        "mo.image(",
        '"max-height": "560px"',
        '_status_text = str(get_baserender_export_status() or "")',
        'payload_entries = list(_payload.get("plots", []))',
        "plot_inventory_load_error = (",
        "Plot gallery requires a current `outputs/plots/current_inventory.json` ",
        "with the full notebook-visible surface. ",
        "plot_export_target = mo.ui.dropdown(",
        "def export_plot_artifact(source_path: Path, destination_path: Path, fmt: str) -> None:",
        'plot_export_button = mo.ui.run_button(label="Export", kind="neutral")',
        "Saved `",
        '_figure.patch.set_facecolor("white")',
        "def resolve_plot_display_media(source_path: Path) -> tuple[str, object]:",
    ]
    for literal in expected_literals:
        assert literal in content

    absent_literals = [
        "__RUN_ROOT__",
        "__CFG_PATH__",
        "__RECORDS_PATH__",
        "__OUTPUT_SOURCE__",
        "__USR_ROOT__",
        "__USR_DATASET__",
        "__WORKSPACE_HEADING__",
        "__WORKSPACE_PLAN_NAMES__",
        "__WORKSPACE_RUN_DETAILS_PAYLOAD__",
        "records_with_overlays.parquet",
        "prepare_usr_preview_records",
        "load_records_from_parquet",
        "render_baserender_preview_path",
        "get_baserender_display_payload, set_baserender_display_payload = mo.state({})",
        "prefetch_indices = (active_row_index, active_row_index - 1, active_row_index + 1)",
        'preview_cache_dir = run_root / "outputs" / "notebooks" / ".baserender_preview_cache"',
        'preview_dir = plot_inventory_path.parent / ".preview_png"',
        "resolve_plot_preview_image",
        "plot_inventory_recovery_notice",
        'plot_inventory_source = "plot_manifest"',
        "filesystem_scan",
        "__GENERATED_WITH__",
    ]
    for literal in absent_literals:
        assert literal not in content


def test_notebook_generate_uses_repo_relative_defaults_for_export_path_boxes(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    content = notebook_path.read_text()

    assert "repo_root = _find_repo_root(run_root)" in content
    assert "default_export_path_text = to_repo_relative_path(default_export_path)" in content
    assert "default_baserender_export_path_text = to_repo_relative_path(default_baserender_export_path)" in content
    assert "default_plot_export_dir_text = to_repo_relative_path(default_plot_export_dir)" in content
    assert "- Path behavior: relative export paths resolve from the repository root." in content
    assert "repo_root=repo_root," in content


def test_notebook_generate_includes_slider_and_numeric_jump_for_record_navigation(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    content = notebook_path.read_text()

    assert "record_index_slider = mo.ui.slider(" in content
    assert 'label="Record"' in content
    assert "record_index_jump = mo.ui.number(" in content
    assert 'label="Jump to"' in content
    assert "_slider_row = mo.hstack(" in content
    assert "[record_index_slider]," in content
    assert "_nav_row = mo.hstack(" in content


def test_notebook_generate_derives_baserender_preview_from_active_record_and_keeps_export_widgets_stable(
    tmp_path: Path,
) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    content = notebook_path.read_text()

    assert (
        "def _(\n    active_baserender_request,\n    active_record_id,\n    render_baserender_preview_image,\n):"
    ) in content
    assert "active_baserender_display_payload = {" in content
    assert (
        '"BaseRender preview payload drifted away from the active record selection. Regenerate the notebook and rerun."'
        in content
    )
    assert (
        "def _(mo, run_root, to_repo_relative_path):\n"
        '    baserender_export_format = mo.ui.dropdown(options=["png", "pdf", "svg"], value="png", label="")'
    ) in content
    assert "import matplotlib.pyplot as _plt" in content
    assert content.count("close(_figure)") == 2


def test_notebook_generate_keeps_navigation_state_out_of_preview_loading_cell(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    content = notebook_path.read_text()

    assert (
        "def _(\n"
        "    build_records_preview_table,\n"
        "    df_window,\n"
        "    has_plan_column,\n"
        "    preview_strategy,\n"
        "    preview_total_rows,\n"
        "    preview_window_limit,\n"
        "    record_id_column,\n"
        "    record_plan_filter,\n"
        "    require,\n"
        "):"
    ) in content
    assert "def _(get_active_record_index, record_count, set_active_record_index):" in content
    assert "active_record_index_default = max(0, min(record_count - 1, _raw_active_index))" in content
    assert (
        "def _(\n    active_record_index_default,\n    mo,\n    record_count,\n    set_active_record_index,\n):"
    ) in content


def test_notebook_generate_handles_empty_plot_filter_intersection_without_error(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    content = notebook_path.read_text()
    assert 'require(not _filtered_entries, f"No plots found for plan `{selected_plot_plan}`.")' not in content
    assert 'plot_filter_message = ""' in content
    assert "if not _filtered_entries:" in content
    assert "active_plot_entry = None" in content
    assert "No plots found for plot type `" in content
    assert "Configured plot types in selected scope:" not in content


def test_notebook_generate_shows_plot_gallery_notice_when_plots_are_missing(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    content = notebook_path.read_text()

    assert 'plot_gallery_notice = ""' in content
    assert "Run `uv run dense plot` to generate plot artifacts for this run." in content
    assert "require(\n        not plot_entries," not in content


def test_notebook_generate_includes_available_plot_ids_even_when_not_generated(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    content = notebook_path.read_text()

    assert "known_plot_ids = notebook_visible_plot_ids()" in content
    assert "generated_plot_ids = sorted(" in content
    assert "_count = int(_generated_counts.get(_plot_id, _generated_base_counts.get(_plot_id, 0)))" in content
    assert "if _count <= 0:" in content
    assert "known_plot_ids=known_plot_ids," not in content
    assert "No generated plots for plot type `" in content
    assert "Run `uv run dense plot --only " in content
    assert "Available but not generated: " not in content
    assert "Required artifacts:" in content
    assert "Availability: `" in content
    assert "resolve_plot_record(" in content


def test_notebook_generate_formats_plot_availability_in_gallery_filters(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    content = notebook_path.read_text()

    assert "plot_id_label_to_id = {}" in content
    assert "describe_visual_plot_type(_plot_id)" in content
    assert '_label = f"{describe_visual_plot_type(_plot_id)} [{_status}]"' in content
    assert "plot_availability_rows = []" in content
    assert "plot_availability_table = pd.DataFrame(" in content
    assert "resolve_plot_availability(" in content
    assert "generated_plot_ids=generated_plot_ids," in content
    assert '"Plot type"' in content
    assert '"Status"' in content
    assert '"Generated files"' in content
    assert '"Gallery filters": mo.vstack(' not in content
    assert '"Plot availability": mo.ui.table(plot_availability_table),' in content
    assert "_selected_plot_type = str(plot_id_label_to_id.get(" in content


def test_notebook_generate_uses_shared_plot_inventory_helpers_in_gallery_filters(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    content = notebook_path.read_text()

    assert "resolve_plot_record(" in content
    assert "resolve_plot_availability(" in content
    assert "def _entry_matches_selected_plot_id(_entry: dict[str, object]) -> bool:" in content
    assert "return base_plot_id(_visual_plot_type) == selected_plot_id" in content


def test_notebook_generate_constrains_plot_preview_image_size(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    content = notebook_path.read_text()

    assert "mo.image(" in content
    assert "alt=_alt_text," in content
    assert "caption=_caption or None," in content
    assert '"max-width": "860px",' in content
    assert '"max-height": "560px",' in content
    assert '"margin": "0 auto",' in content
    assert '"width": "100%",' in content


def test_notebook_generate_embeds_local_plot_assets_without_dead_absolute_urls(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    content = notebook_path.read_text()

    assert "resolve_plot_display_media(_plot_path)" in content
    assert 'return ("image_bytes", preview_path.read_bytes())' in content
    assert "mo.image(\n                    _display_payload," in content
    assert "mo.pdf(_plot_path)" in content
    assert "mo.video(\n                    _plot_path.read_bytes()," in content
    assert "mo.pdf(str(_plot_path))" not in content
    assert "mo.video(\n                    str(_plot_path)," not in content


def test_notebook_generate_offers_artifact_export_alongside_converted_formats(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    content = notebook_path.read_text()

    assert 'options=["artifact", "png", "pdf", "svg"]' in content
    assert "Format `artifact`: copy each plot in its generated source format." in content
    assert '_selected_format not in {"artifact", "pdf", "png", "svg"}' in content
    assert '_destination_suffix = _source_path.suffix if _selected_format == "artifact"' in content


def test_notebook_generate_uses_real_newlines_in_plot_metadata_and_export_help(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    content = notebook_path.read_text()

    selected_metadata_tail = content.split('"Selected plot metadata": mo.md(', 1)[1]
    selected_metadata_block = selected_metadata_tail.split(")", 1)[0]
    assert '"\\n".join(' in selected_metadata_block
    assert '"\\\\n".join(' not in selected_metadata_block

    export_behavior_tail = content.split('"Export behavior": mo.md(', 1)[1]
    export_behavior_block = export_behavior_tail.split(")", 1)[0]
    assert '"\\n".join(' in export_behavior_block
    assert '"\\\\n".join(' not in export_behavior_block

    dataset_export_tail = content.split('"Dataset export details": mo.md(', 1)[1]
    dataset_export_block = dataset_export_tail.split(")", 1)[0]
    assert '"\\n".join(' in dataset_export_block
    assert '"\\\\n".join(' not in dataset_export_block


def test_notebook_generate_skips_hidden_plot_cache_directories(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    content = notebook_path.read_text()

    assert "load_current_inventory_strict(plot_root, config_path=config_path)" in content
    assert "plot_manifest.json" not in content
    assert "filesystem_scan" not in content


def test_notebook_generate_treats_variant_plots_as_generated_for_base_plot_types(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    content = notebook_path.read_text()

    assert "_generated_base_plot_ids = {" in content
    assert "and selected_plot_id not in _generated_base_plot_ids" in content


def test_notebook_generate_baserender_preview_adds_title_and_legend_clearance(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    content = notebook_path.read_text()

    assert '"padding_y": 12.0,' in content
    assert '_legend_pad_px = float(_contract_style_overrides.get("legend_pad_px", 20.0))' in content
    assert '_legend_vertical_align = float(_contract_style_overrides.get("legend_vertical_align", 0.5))' in content
    assert '"legend_pad_px": _legend_pad_px,' in content
    assert "_figure.text(" in content
    assert '_record_id = str(getattr(record, "id", "") or "unknown")' in content
    assert '_workspace_title = str(workspace_heading or "").strip()' in content
    assert (
        '_header_subtitle = densegen_video_subtitle_text(record_id=_record_id, plan_name=str(plan_summary or ""))'
        in content
    )
    assert "_header_title_wrapped = textwrap.fill(" in content
    assert "_header_subtitle_wrapped = textwrap.fill(" in content
    assert "_title_font_size = _base_typography_size" in content
    assert '_header_text = f"{workspace_name} | sequence {_record_display_id}"' not in content
    assert "_buffer = BytesIO()" in content
    assert "@lru_cache(maxsize=64)" in content
    assert "def render_baserender_preview_image(" in content
    assert "render_baserender_preview_path" not in content


def test_notebook_generate_streamlines_summary_and_adds_plot_export_controls(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    content = notebook_path.read_text()

    assert '{"Field": "Records path", "Value": _records_path_display}' not in content
    assert "plan_quota_table, run_summary_table = build_run_summary_tables(" not in content
    assert "run_manifest=run_manifest," not in content
    assert "run_items=run_items," not in content
    assert 'mo.md("### Run summary")' not in content
    assert "mo.ui.table(run_summary_table)" not in content
    assert 'mo.md("### Workspace context")' not in content
    assert 'mo.md("### Plan quota breakdown")' not in content
    assert "### Records summary" not in content
    assert '{"Field": "Generated total", "Value": str(run_manifest.get("total_generated", "-"))}' not in content
    assert '{"Field": "Quota total", "Value": str(run_manifest.get("total_quota", "-"))}' not in content
    assert '{"Field": "Quota progress", "Value": str(run_manifest.get("quota_progress_pct", "-"))}' not in content
    assert 'mo.md("### Plot gallery")' in content
    assert 'label="Scope"' not in content
    assert (
        'plot_id_filter = mo.ui.dropdown(options=plot_id_options, value=plot_id_options[0], label="Plot type")'
        in content
    )
    assert 'plot_selector = mo.ui.dropdown(options=plot_options, value=plot_options[0], label="Artifact")' in content
    assert "build_plot_ids_by_scope(" not in content
    assert "plot_export_target = mo.ui.dropdown(" in content
    assert 'label="",' in content
    assert (
        'plot_export_format = mo.ui.dropdown(options=["artifact", "png", "pdf", "svg"], value="png", label="")'
    ) in content
    assert 'plot_export_button = mo.ui.run_button(label="Export", kind="neutral")' in content
    assert "plot(s) to `" in content
    assert '"No plots found for plot type `' in content

    assert (
        "mo.ui.table(df_window_filtered.loc[:, list(df_window_filtered.columns)]),\n"
        '            mo.md("Dataset export path"),\n'
        "            export_controls,\n"
        "            export_details,"
    ) in content
    assert (
        'mo.md("### Plot export"),\n'
        "            mo.hstack(\n"
        "                [\n"
        "                    plot_export_target,\n"
        "                    plot_export_format,\n"
        "                    plot_export_path,\n"
        "                    plot_export_button,\n"
        "                ],\n"
        '                justify="start",\n'
        '                align="end",\n'
        "                gap=0.2,\n"
        "                widths=[1.0, 1.0, 8.0, 0.9],\n"
        "                wrap=False,\n"
        "            ),\n"
        "            plot_export_details,"
    ) in content


def test_notebook_generate_uses_configured_parquet_path(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    cfg_path.write_text(
        cfg_path.read_text().replace("outputs/tables/records.parquet", "outputs/tables/custom_records.parquet")
    )
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    content = notebook_path.read_text()
    assert "custom_records.parquet" in content
    assert "No `records.parquet` artifact was found for this workspace" not in content
    assert "Ensure `records.parquet` has unique `id` values" not in content


def test_notebook_generate_supports_usr_output_target(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            densegen:
              schema_version: "2.9"
              run:
                id: demo
                root: "."
              inputs:
                - name: demo_input
                  type: binding_sites
                  path: inputs.csv
              output:
                targets: [usr]
                schema:
                  bio_type: dna
                  alphabet: dna_4
                usr:
                  dataset: densegen_demo
                  root: ../usr_root
              generation:
                sequence_length: 10
                plan:
                  - name: demo_plan
                    sequences: 1
                    sampling:
                      include_inputs: [demo_input]
                    regulator_constraints:
                      groups: []
              solver:
                backend: CBC
                strategy: iterate
              logging:
                log_dir: outputs/logs
            """
        ).strip()
        + "\n"
    )
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    content = notebook_path.read_text()
    assert "__OUTPUT_SOURCE__" not in content
    assert "__USR_ROOT__" not in content
    assert "__USR_DATASET__" not in content
    assert 'output_source = "usr"' in content
    assert 'workspace_plan_names = ["demo_plan"]' in content
    assert "include_overlays=True" in content
    assert "records_with_overlays.parquet" not in content


def test_notebook_generate_uses_plots_source_when_output_targets_are_both(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            densegen:
              schema_version: "2.9"
              run:
                id: demo
                root: "."
              inputs:
                - name: demo_input
                  type: binding_sites
                  path: inputs.csv
              output:
                targets: [parquet, usr]
                schema:
                  bio_type: dna
                  alphabet: dna_4
                parquet:
                  path: outputs/tables/records.parquet
                usr:
                  dataset: densegen_demo
                  root: ../usr_root
              generation:
                sequence_length: 10
                plan:
                  - name: demo_plan
                    sequences: 1
                    sampling:
                      include_inputs: [demo_input]
                    regulator_constraints:
                      groups: []
              solver:
                backend: CBC
                strategy: iterate
              logging:
                log_dir: outputs/logs
            plots:
              source: usr
              out_dir: outputs/plots
            """
        ).strip()
        + "\n"
    )
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    content = notebook_path.read_text()
    assert 'output_source = "usr"' in content
    assert "include_overlays=True" in content
    assert "records_with_overlays.parquet" not in content


def test_notebook_generate_custom_out_suggests_run_with_path(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")
    out_path = tmp_path / "custom notebook.py"

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path), "--out", str(out_path)])
    assert result.exit_code == 0, result.output
    assert "dense notebook run --path 'custom notebook.py'" in result.output


def test_notebook_generate_under_pixi_suggests_explicit_config_flag(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")
    monkeypatch.setenv("PIXI_PROJECT_MANIFEST", "/tmp/pixi.toml")

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])

    assert result.exit_code == 0, result.output
    assert "pixi run dense notebook run -c" in result.output
    assert str(cfg_path) in result.output


def test_notebook_run_requires_existing_notebook(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "run", "-c", str(cfg_path)])
    assert result.exit_code == 1
    assert "No notebook found" in result.output
    assert "outputs/notebooks/densegen_run_overview.py" in result.output


def test_notebook_generate_passes_marimo_check(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    assert notebook_path.exists()

    check_result = subprocess.run(
        [sys.executable, "-m", "marimo", "check", str(notebook_path)],
        capture_output=True,
        text=True,
    )
    assert check_result.returncode == 0, check_result.stdout + check_result.stderr
    assert "warning[" not in check_result.stdout


def test_notebook_generate_exports_html_without_cell_execution_errors(tmp_path: Path) -> None:
    import pandas as pd

    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")
    tables_dir = tmp_path / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "id": "row1",
                "sequence": "AAAAAA",
                "densegen__plan": "demo_plan",
                "densegen__used_tfbs_detail": [
                    {"regulator": "lexA", "orientation": "fwd", "sequence": "AAA", "offset": 0}
                ],
            }
        ]
    ).to_parquet(tables_dir / "records.parquet", index=False)
    plots_dir = tmp_path / "outputs" / "plots"
    _write_notebook_surface_inventory(plots_dir)

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    assert notebook_path.exists()

    html_out = tmp_path / "densegen_run_overview.html"
    export_result = subprocess.run(
        [sys.executable, "-m", "marimo", "export", "html", str(notebook_path), "-o", str(html_out)],
        capture_output=True,
        text=True,
    )
    output_text = export_result.stdout + export_result.stderr
    assert export_result.returncode == 0, output_text
    assert "MarimoExceptionRaisedError" not in output_text
    assert "cells failed to execute" not in output_text.lower()
    assert html_out.exists()
    assert html_out.stat().st_size > 0


def test_notebook_generate_exports_html_without_plots_and_without_cell_errors(tmp_path: Path) -> None:
    import pandas as pd

    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")
    tables_dir = tmp_path / "outputs" / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "id": "row1",
                "sequence": "AAAAAA",
                "densegen__plan": "demo_plan",
                "densegen__used_tfbs_detail": [
                    {"regulator": "lexA", "orientation": "fwd", "sequence": "AAA", "offset": 0}
                ],
            }
        ]
    ).to_parquet(tables_dir / "records.parquet", index=False)

    runner = CliRunner()
    result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert result.exit_code == 0, result.output
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    assert notebook_path.exists()

    html_out = tmp_path / "densegen_run_overview_no_plots.html"
    export_result = subprocess.run(
        [sys.executable, "-m", "marimo", "export", "html", str(notebook_path), "-o", str(html_out)],
        capture_output=True,
        text=True,
    )
    output_text = export_result.stdout + export_result.stderr
    assert export_result.returncode != 0
    assert "current_inventory.json" in output_text
    assert "full notebook-visible surface" in output_text


def test_notebook_launch_rejects_incomplete_plot_inventory(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    plot_root = tmp_path / "outputs" / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    (plot_root / "current_inventory.json").write_text(
        json.dumps(
            {
                "schema_version": "densegen.current_inventory.v2",
                "plots": [
                    {
                        "plot_id": "source_cohort_concentration",
                        "path": "dataset/source_cohort_concentration.png",
                    },
                    {
                        "plot_id": "stage_a_sampling_yield",
                        "path": "stage_a/stage_a_sampling_yield.png",
                    },
                ],
            }
        )
    )

    run_result = runner.invoke(app, ["notebook", "run", "-c", str(cfg_path)])

    assert run_result.exit_code == 1
    assert "missing required notebook plot ids" in run_result.output
    assert "dense plot" in run_result.output


def test_notebook_run_uses_marimo_run_mode_by_default(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)
    monkeypatch.setattr(notebook_commands, "_port_is_available", lambda host, port: True)
    captured: dict[str, object] = {}

    def _fake_run_marimo_command(
        *,
        command,
        env,
        browser_url=None,
        open_timeout_seconds=12.0,
        on_browser_open_failure=None,
    ):
        captured["command"] = command
        captured["env"] = env
        captured["browser_url"] = browser_url
        captured["open_timeout_seconds"] = open_timeout_seconds
        captured["on_browser_open_failure"] = on_browser_open_failure
        return True

    monkeypatch.setattr(notebook_commands, "_run_marimo_command", _fake_run_marimo_command)
    run_result = runner.invoke(app, ["notebook", "run", "--reuse-server", "-c", str(cfg_path)])
    assert run_result.exit_code == 0, run_result.output
    assert captured["command"][:2] == ["marimo", "run"]
    assert "--headless" not in captured["command"]
    env = captured["env"]
    assert isinstance(env, dict)
    assert env.get("MARIMO_SKIP_UPDATE_CHECK") == "1"
    assert captured["browser_url"] is None
    assert "Notebook URL" in run_result.output
    assert "press Ctrl+C to stop the notebook server" in run_result.output


def test_notebook_run_auto_opens_when_codex_sandbox_is_set(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setenv("CODEX_SANDBOX", "seatbelt")
    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)
    monkeypatch.setattr(notebook_commands, "_port_is_available", lambda host, port: True)
    captured: dict[str, object] = {}

    def _fake_run_marimo_command(
        *,
        command,
        env,
        browser_url=None,
        open_timeout_seconds=12.0,
        on_browser_open_failure=None,
    ):
        captured["command"] = command
        captured["browser_url"] = browser_url
        return True

    monkeypatch.setattr(notebook_commands, "_run_marimo_command", _fake_run_marimo_command)
    run_result = runner.invoke(app, ["notebook", "run", "-c", str(cfg_path)])

    assert run_result.exit_code == 0, run_result.output
    assert "--headless" not in captured["command"]
    assert captured["browser_url"] is None
    assert "Notebook URL" in run_result.output


def test_notebook_run_rejects_stale_default_notebook_template(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    notebook_path.write_text("import marimo\napp = marimo.App()\nif __name__ == '__main__':\n    app.run()\n")

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(
        notebook_commands,
        "_run_marimo_command",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("marimo should not launch for a stale notebook")),
    )
    run_result = runner.invoke(app, ["notebook", "run", "--reuse-server", "-c", str(cfg_path)])

    assert run_result.exit_code == 1, run_result.output
    assert "generated notebook is stale" in run_result.output
    assert "dense notebook generate --force" in run_result.output
    assert notebook_path.read_text() == "import marimo\napp = marimo.App()\nif __name__ == '__main__':\n    app.run()\n"


def test_fresh_generated_notebook_template_is_not_false_positive_stale(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands
    from dnadesign.densegen.src.cli.main import _run_root_for, cli_context
    from dnadesign.densegen.src.config import load_config

    loaded = load_config(cfg_path)
    run_root = _run_root_for(loaded)
    notebook_path = run_root / "outputs" / "notebooks" / "densegen_run_overview.py"

    assert (
        notebook_commands._sync_default_notebook_template(
            loaded=loaded,
            run_root=run_root,
            cfg_path=loaded.path,
            notebook_path=notebook_path,
            context=cli_context,
            write=False,
        )
        is False
    )


def test_notebook_run_wsl_open_uses_marimo_native_open_flow(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)
    monkeypatch.setattr(notebook_commands, "_is_wsl", lambda: True)
    monkeypatch.setattr(notebook_commands, "_port_is_available", lambda host, port: True)
    captured: dict[str, object] = {}

    def _fake_run_marimo_command(
        *,
        command,
        env,
        browser_url=None,
        open_timeout_seconds=12.0,
        on_browser_open_failure=None,
    ):
        captured["command"] = command
        captured["env"] = env
        captured["browser_url"] = browser_url
        captured["open_timeout_seconds"] = open_timeout_seconds
        captured["on_browser_open_failure"] = on_browser_open_failure
        return True

    monkeypatch.setattr(notebook_commands, "_run_marimo_command", _fake_run_marimo_command)
    run_result = runner.invoke(app, ["notebook", "run", "--reuse-server", "-c", str(cfg_path)])

    assert run_result.exit_code == 0, run_result.output
    assert captured["command"][:2] == ["marimo", "run"]
    assert "--headless" not in captured["command"]
    assert captured["browser_url"] is None
    assert "Notebook URL" in run_result.output


def test_notebook_run_wsl_open_warns_when_browser_does_not_open(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)
    monkeypatch.setattr(notebook_commands, "_is_wsl", lambda: True)
    monkeypatch.setattr(notebook_commands, "_port_is_available", lambda host, port: True)

    observed: dict[str, object] = {}

    def _fake_run_marimo_command(**kwargs):
        observed["on_browser_open_failure"] = kwargs.get("on_browser_open_failure")
        return False

    monkeypatch.setattr(notebook_commands, "_run_marimo_command", _fake_run_marimo_command)
    run_result = runner.invoke(app, ["notebook", "run", "--reuse-server", "-c", str(cfg_path)])

    assert run_result.exit_code == 0, run_result.output
    assert observed["on_browser_open_failure"] is None
    assert "Browser did not open automatically" not in run_result.output


def test_notebook_run_wsl_reuses_running_server_and_opens_browser(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)
    monkeypatch.setattr(notebook_commands, "_is_wsl", lambda: True)
    monkeypatch.setattr(notebook_commands, "_port_is_available", lambda host, port: False)
    monkeypatch.setattr(notebook_commands, "_url_is_reachable", lambda url: True)
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    monkeypatch.setattr(notebook_commands, "_running_notebook_filename", lambda url: str(notebook_path))
    opened: dict[str, str] = {}

    def _fake_open_browser_tab(url: str) -> bool:
        opened["url"] = url
        return True

    monkeypatch.setattr(notebook_commands, "_open_browser_tab", _fake_open_browser_tab)
    run_result = runner.invoke(app, ["notebook", "run", "--reuse-server", "-c", str(cfg_path)])

    assert run_result.exit_code == 0, run_result.output
    assert "already serving this notebook" in run_result.output
    assert opened["url"] == "http://127.0.0.1:2718"


def test_notebook_run_explicit_open_keeps_marimo_auto_open_behavior(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)
    monkeypatch.setattr(notebook_commands, "_port_is_available", lambda host, port: True)
    captured: dict[str, object] = {}

    def _fake_run_marimo_command(
        *,
        command,
        env,
        browser_url=None,
        open_timeout_seconds=12.0,
        on_browser_open_failure=None,
    ):
        captured["command"] = command
        captured["env"] = env
        captured["browser_url"] = browser_url
        return True

    monkeypatch.setattr(notebook_commands, "_run_marimo_command", _fake_run_marimo_command)
    run_result = runner.invoke(app, ["notebook", "run", "--mode", "run", "--open", "-c", str(cfg_path)])

    assert run_result.exit_code == 0, run_result.output
    assert captured["command"][:2] == ["marimo", "run"]
    assert "--headless" not in captured["command"]
    env = captured.get("env")
    assert isinstance(env, dict)
    assert captured["browser_url"] is None
    assert "Notebook URL" in run_result.output


def test_notebook_run_no_open_passes_headless_to_marimo(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)
    monkeypatch.setattr(notebook_commands, "_port_is_available", lambda host, port: True)
    captured: dict[str, object] = {}

    def _fake_run_marimo_command(
        *,
        command,
        env,
        browser_url=None,
        open_timeout_seconds=12.0,
        on_browser_open_failure=None,
    ):
        captured["command"] = command
        captured["env"] = env
        captured["browser_url"] = browser_url
        return True

    monkeypatch.setattr(notebook_commands, "_run_marimo_command", _fake_run_marimo_command)
    run_result = runner.invoke(app, ["notebook", "run", "--mode", "run", "--no-open", "-c", str(cfg_path)])

    assert run_result.exit_code == 0, run_result.output
    assert captured["command"][:2] == ["marimo", "run"]
    assert "--headless" in captured["command"]
    env = captured.get("env")
    assert isinstance(env, dict)
    assert captured["browser_url"] is None
    assert "Notebook URL" in run_result.output


def test_notebook_run_edit_mode_failure_suggests_run_mode(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)
    monkeypatch.setattr(
        notebook_commands,
        "_run_marimo_command",
        lambda **kwargs: (_ for _ in ()).throw(subprocess.CalledProcessError(returncode=1, cmd=["marimo"])),
    )
    run_result = runner.invoke(app, ["notebook", "run", "--mode", "edit", "-c", str(cfg_path)])
    assert run_result.exit_code == 1
    assert "dense notebook run --mode run" in run_result.output


def test_notebook_run_headless_passes_flag_to_marimo(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)
    monkeypatch.setattr(notebook_commands, "_port_is_available", lambda host, port: True)
    captured: dict[str, object] = {}

    def _fake_run_marimo_command(
        *,
        command,
        env,
        browser_url=None,
        open_timeout_seconds=12.0,
        on_browser_open_failure=None,
    ):
        captured["command"] = command
        captured["browser_url"] = browser_url
        return True

    monkeypatch.setattr(notebook_commands, "_run_marimo_command", _fake_run_marimo_command)
    run_result = runner.invoke(app, ["notebook", "run", "--mode", "run", "--headless", "-c", str(cfg_path)])
    assert run_result.exit_code == 0, run_result.output
    assert captured["command"][:2] == ["marimo", "run"]
    assert "--headless" in captured["command"]
    assert captured["browser_url"] is None


def test_notebook_run_rejects_headless_edit_mode(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)
    run_result = runner.invoke(app, ["notebook", "run", "--mode", "edit", "--headless", "-c", str(cfg_path)])
    assert run_result.exit_code == 1
    assert "--headless is only supported with --mode run" in run_result.output


def test_notebook_run_rejects_no_open_edit_mode(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)
    run_result = runner.invoke(app, ["notebook", "run", "--mode", "edit", "--no-open", "-c", str(cfg_path)])
    assert run_result.exit_code == 1
    assert "--open/--no-open is only supported with --mode run" in run_result.output


def test_notebook_run_rejects_invalid_host(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)
    run_result = runner.invoke(app, ["notebook", "run", "--host", "   ", "-c", str(cfg_path)])
    assert run_result.exit_code == 1
    assert "--host must be a non-empty value" in run_result.output


def test_notebook_run_rejects_invalid_port(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)
    run_result = runner.invoke(app, ["notebook", "run", "--port", "70000", "-c", str(cfg_path)])
    assert run_result.exit_code == 1
    assert "--port must be within 1-65535" in run_result.output


def test_notebook_run_rejects_nonpositive_open_timeout(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)
    run_result = runner.invoke(app, ["notebook", "run", "--open-timeout", "0", "-c", str(cfg_path)])
    assert run_result.exit_code == 1
    assert "--open-timeout must be > 0 seconds" in run_result.output


def test_notebook_run_rejects_open_timeout_without_open(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)
    run_result = runner.invoke(
        app,
        ["notebook", "run", "--mode", "run", "--no-open", "--open-timeout", "5", "-c", str(cfg_path)],
    )
    assert run_result.exit_code == 1
    assert "--open-timeout requires --mode run with --open" in run_result.output


def test_notebook_run_passes_open_timeout_override(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)
    monkeypatch.setattr(notebook_commands, "_port_is_available", lambda host, port: True)
    captured: dict[str, object] = {}

    def _fake_run_marimo_command(
        *,
        command,
        env,
        browser_url=None,
        open_timeout_seconds=12.0,
        on_browser_open_failure=None,
    ):
        captured["command"] = command
        captured["browser_url"] = browser_url
        captured["open_timeout_seconds"] = open_timeout_seconds
        return True

    monkeypatch.setattr(notebook_commands, "_run_marimo_command", _fake_run_marimo_command)
    run_result = runner.invoke(app, ["notebook", "run", "--open-timeout", "5.5", "-c", str(cfg_path)])

    assert run_result.exit_code == 0, run_result.output
    assert captured["browser_url"] is None
    assert captured["open_timeout_seconds"] == 5.5


def test_notebook_run_default_open_uses_marimo_native_auto_open(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)
    monkeypatch.setattr(notebook_commands, "_port_is_available", lambda host, port: True)
    captured: dict[str, object] = {}

    def _fake_run_marimo_command(
        *,
        command,
        env,
        browser_url=None,
        open_timeout_seconds=12.0,
        on_browser_open_failure=None,
        on_process_start=None,
    ):
        captured["command"] = command
        captured["browser_url"] = browser_url
        captured["on_browser_open_failure"] = on_browser_open_failure
        captured["on_process_start"] = on_process_start
        return True

    monkeypatch.setattr(notebook_commands, "_run_marimo_command", _fake_run_marimo_command)
    run_result = runner.invoke(app, ["notebook", "run", "-c", str(cfg_path)])

    assert run_result.exit_code == 0, run_result.output
    assert captured["command"][:2] == ["marimo", "run"]
    assert "--headless" not in captured["command"]
    assert captured["browser_url"] is None
    assert captured["on_browser_open_failure"] is None
    assert captured["on_process_start"] is None
    assert "Notebook URL" in run_result.output


def test_notebook_run_sets_browser_launcher_on_macos_for_auto_open(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)
    monkeypatch.setattr(notebook_commands, "_port_is_available", lambda host, port: True)
    monkeypatch.setattr(notebook_commands.sys, "platform", "darwin")
    captured: dict[str, object] = {}

    def _fake_run_marimo_command(
        *,
        command,
        env,
        browser_url=None,
        open_timeout_seconds=12.0,
        on_browser_open_failure=None,
        on_process_start=None,
    ):
        captured["env"] = env
        return True

    monkeypatch.setattr(notebook_commands, "_run_marimo_command", _fake_run_marimo_command)
    run_result = runner.invoke(app, ["notebook", "run", "-c", str(cfg_path)])

    assert run_result.exit_code == 0, run_result.output
    env = captured.get("env")
    assert isinstance(env, dict)
    assert env.get("BROWSER") == "open"


def test_notebook_run_keeps_existing_browser_launcher_on_macos(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)
    monkeypatch.setattr(notebook_commands, "_port_is_available", lambda host, port: True)
    monkeypatch.setattr(notebook_commands.sys, "platform", "darwin")
    monkeypatch.setenv("BROWSER", "custom-browser")
    captured: dict[str, object] = {}

    def _fake_run_marimo_command(
        *,
        command,
        env,
        browser_url=None,
        open_timeout_seconds=12.0,
        on_browser_open_failure=None,
        on_process_start=None,
    ):
        captured["env"] = env
        return True

    monkeypatch.setattr(notebook_commands, "_run_marimo_command", _fake_run_marimo_command)
    run_result = runner.invoke(app, ["notebook", "run", "-c", str(cfg_path)])

    assert run_result.exit_code == 0, run_result.output
    env = captured.get("env")
    assert isinstance(env, dict)
    assert env.get("BROWSER") == "custom-browser"


def test_notebook_run_reuses_running_server_when_requested_port_is_in_use_for_same_notebook(
    tmp_path: Path, monkeypatch
) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)
    monkeypatch.setattr(notebook_commands, "_port_is_available", lambda host, port: False)
    monkeypatch.setattr(notebook_commands, "_url_is_reachable", lambda url: True)
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    monkeypatch.setattr(notebook_commands, "_running_notebook_filename", lambda url: str(notebook_path))
    opened: dict[str, object] = {}

    def _fake_open_browser(url: str) -> bool:
        opened["url"] = url
        return True

    monkeypatch.setattr(notebook_commands, "_open_browser_tab", _fake_open_browser)

    def _unexpected_run(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called when reusing an existing server")

    monkeypatch.setattr(subprocess, "run", _unexpected_run)
    run_result = runner.invoke(app, ["notebook", "run", "--reuse-server", "-c", str(cfg_path)])

    assert run_result.exit_code == 0
    assert "Notebook URL" in run_result.output
    assert "already serving this notebook" in run_result.output
    assert opened["url"] == "http://127.0.0.1:2718"


def test_notebook_run_does_not_reuse_mismatched_notebook_when_reuse_server_enabled(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)
    monkeypatch.setattr(notebook_commands, "_port_is_available", lambda host, port: False if port == 2718 else True)
    monkeypatch.setattr(notebook_commands, "_url_is_reachable", lambda url: True)
    monkeypatch.setattr(notebook_commands, "_running_notebook_filename", lambda url: "/tmp/other-notebook.py")
    monkeypatch.setattr(notebook_commands, "_find_available_port", lambda host: 3031)

    captured: dict[str, object] = {}

    def _fake_run_marimo_command(
        *,
        command,
        env,
        browser_url=None,
        open_timeout_seconds=12.0,
        on_browser_open_failure=None,
    ):
        captured["command"] = command
        captured["browser_url"] = browser_url
        return True

    monkeypatch.setattr(notebook_commands, "_run_marimo_command", _fake_run_marimo_command)
    run_result = runner.invoke(app, ["notebook", "run", "--reuse-server", "-c", str(cfg_path)])

    assert run_result.exit_code == 0, run_result.output
    assert "serves a different notebook" in run_result.output
    assert "launching a fresh server on a free port" in run_result.output
    assert captured["command"][:2] == ["marimo", "run"]
    assert "--port" in captured["command"]
    assert captured["command"][captured["command"].index("--port") + 1] == "3031"


def test_notebook_run_blocks_before_server_reuse_checks_when_default_notebook_is_stale(
    tmp_path: Path, monkeypatch
) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    notebook_path.write_text("import marimo\napp = marimo.App()\nif __name__ == '__main__':\n    app.run()\n")

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    def _fail_port_check(*_args, **_kwargs):
        raise AssertionError("port checks should not run for stale notebook")

    monkeypatch.setattr(
        notebook_commands,
        "_port_is_available",
        _fail_port_check,
    )
    monkeypatch.setattr(
        notebook_commands,
        "_run_marimo_command",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("marimo should not launch for a stale notebook")),
    )
    run_result = runner.invoke(app, ["notebook", "run", "--reuse-server", "-c", str(cfg_path)])

    assert run_result.exit_code == 1, run_result.output
    assert "generated notebook is stale" in run_result.output
    assert "dense notebook generate --force" in run_result.output


def test_notebook_run_switches_to_free_port_when_requested_port_is_in_use(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)
    monkeypatch.setattr(notebook_commands, "_port_is_available", lambda host, port: False if port == 2718 else True)
    monkeypatch.setattr(notebook_commands, "_url_is_reachable", lambda url: False)
    monkeypatch.setattr(notebook_commands, "_find_available_port", lambda host: 3031)
    captured: dict[str, object] = {}

    def _fake_run_marimo_command(
        *,
        command,
        env,
        browser_url=None,
        open_timeout_seconds=12.0,
        on_browser_open_failure=None,
    ):
        captured["command"] = command
        captured["browser_url"] = browser_url
        return True

    monkeypatch.setattr(notebook_commands, "_run_marimo_command", _fake_run_marimo_command)
    run_result = runner.invoke(app, ["notebook", "run", "-c", str(cfg_path)])

    assert run_result.exit_code == 0, run_result.output
    assert "switching to 3031" in run_result.output
    assert captured["command"][:2] == ["marimo", "run"]
    assert "--headless" not in captured["command"]
    assert "--port" in captured["command"]
    port_index = captured["command"].index("--port")
    assert captured["command"][port_index + 1] == "3031"
    assert captured["browser_url"] is None


def test_notebook_run_prefers_fresh_server_by_default_when_port_is_in_use(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)
    monkeypatch.setattr(notebook_commands, "_port_is_available", lambda host, port: False if port == 2718 else True)
    monkeypatch.setattr(notebook_commands, "_url_is_reachable", lambda url: True)
    monkeypatch.setattr(notebook_commands, "_running_notebook_filename", lambda url: "/tmp/other-notebook.py")
    monkeypatch.setattr(notebook_commands, "_find_available_port", lambda host: 3031)
    captured: dict[str, object] = {}

    def _fake_run_marimo_command(
        *,
        command,
        env,
        browser_url=None,
        open_timeout_seconds=12.0,
        on_browser_open_failure=None,
    ):
        captured["command"] = command
        captured["browser_url"] = browser_url
        return True

    monkeypatch.setattr(notebook_commands, "_run_marimo_command", _fake_run_marimo_command)
    run_result = runner.invoke(app, ["notebook", "run", "-c", str(cfg_path)])

    assert run_result.exit_code == 0, run_result.output
    assert "launching a fresh server on a free port" in run_result.output
    assert "currently serves `/tmp/other-notebook.py`" in run_result.output
    assert captured["command"][:2] == ["marimo", "run"]
    assert "--headless" not in captured["command"]
    assert "--port" in captured["command"]
    port_index = captured["command"].index("--port")
    assert captured["command"][port_index + 1] == "3031"
    assert captured["browser_url"] is None


def test_notebook_run_starts_fresh_server_by_default_when_same_notebook_already_serving(
    tmp_path: Path, monkeypatch
) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)
    monkeypatch.setattr(notebook_commands, "_port_is_available", lambda host, port: False if port == 2718 else True)
    monkeypatch.setattr(notebook_commands, "_url_is_reachable", lambda url: True)
    notebook_path = tmp_path / "outputs" / "notebooks" / "densegen_run_overview.py"
    monkeypatch.setattr(notebook_commands, "_running_notebook_filename", lambda url: str(notebook_path))
    monkeypatch.setattr(notebook_commands, "_find_available_port", lambda host: 3031)
    captured: dict[str, object] = {}

    def _fake_run_marimo_command(
        *,
        command,
        env,
        browser_url=None,
        open_timeout_seconds=12.0,
        on_browser_open_failure=None,
    ):
        captured["command"] = command
        captured["browser_url"] = browser_url
        return True

    monkeypatch.setattr(notebook_commands, "_run_marimo_command", _fake_run_marimo_command)
    run_result = runner.invoke(app, ["notebook", "run", "-c", str(cfg_path)])

    assert run_result.exit_code == 0, run_result.output
    assert "launching a fresh server on a free port because --no-reuse-server is active" in run_result.output
    assert captured["command"][:2] == ["marimo", "run"]
    assert "--port" in captured["command"]
    assert captured["command"][captured["command"].index("--port") + 1] == "3031"


def test_notebook_run_fails_when_no_available_port_can_be_found(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)
    monkeypatch.setattr(notebook_commands, "_port_is_available", lambda host, port: False)
    monkeypatch.setattr(notebook_commands, "_url_is_reachable", lambda url: False)
    monkeypatch.setattr(notebook_commands, "_find_available_port", lambda host: None)

    def _unexpected_run(*args, **kwargs):
        raise AssertionError("subprocess.run should not be called when no replacement port is available")

    monkeypatch.setattr(subprocess, "run", _unexpected_run)
    run_result = runner.invoke(app, ["notebook", "run", "-c", str(cfg_path)])

    assert run_result.exit_code == 1
    assert "No available port found" in run_result.output


def test_notebook_run_strips_host_before_launch(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)
    monkeypatch.setattr(notebook_commands, "_port_is_available", lambda host, port: True)
    captured: dict[str, object] = {}

    def _fake_run_marimo_command(
        *,
        command,
        env,
        browser_url=None,
        open_timeout_seconds=12.0,
        on_browser_open_failure=None,
    ):
        captured["command"] = command
        captured["browser_url"] = browser_url
        return True

    monkeypatch.setattr(notebook_commands, "_run_marimo_command", _fake_run_marimo_command)
    run_result = runner.invoke(app, ["notebook", "run", "--host", " 127.0.0.1 ", "-c", str(cfg_path)])

    assert run_result.exit_code == 0, run_result.output
    assert captured["command"][:2] == ["marimo", "run"]
    assert "--headless" not in captured["command"]
    assert "--host" in captured["command"]
    host_index = captured["command"].index("--host")
    assert captured["command"][host_index + 1] == "127.0.0.1"
    assert captured["browser_url"] is None
    assert "http://127.0.0.1:2718" in run_result.output


def test_notebook_run_formats_ipv6_browser_url(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)
    monkeypatch.setattr(notebook_commands, "_port_is_available", lambda host, port: True)
    captured: dict[str, object] = {}

    def _fake_run_marimo_command(
        *,
        command,
        env,
        browser_url=None,
        open_timeout_seconds=12.0,
        on_browser_open_failure=None,
    ):
        captured["command"] = command
        captured["browser_url"] = browser_url
        return True

    monkeypatch.setattr(notebook_commands, "_run_marimo_command", _fake_run_marimo_command)
    run_result = runner.invoke(app, ["notebook", "run", "--host", "::1", "-c", str(cfg_path)])

    assert run_result.exit_code == 0, run_result.output
    assert captured["browser_url"] is None
    assert "http://[::1]:2718" in run_result.output


def test_notebook_run_maps_wildcard_host_to_localhost_browser_url(tmp_path: Path, monkeypatch) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    gen_result = runner.invoke(app, ["notebook", "generate", "-c", str(cfg_path)])
    assert gen_result.exit_code == 0, gen_result.output

    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_ensure_marimo_installed", lambda: None)
    monkeypatch.setattr(notebook_commands, "_port_is_available", lambda host, port: True)
    captured: dict[str, object] = {}

    def _fake_run_marimo_command(
        *,
        command,
        env,
        browser_url=None,
        open_timeout_seconds=12.0,
        on_browser_open_failure=None,
    ):
        captured["command"] = command
        captured["browser_url"] = browser_url
        return True

    monkeypatch.setattr(notebook_commands, "_run_marimo_command", _fake_run_marimo_command)
    run_result = runner.invoke(app, ["notebook", "run", "--host", "0.0.0.0", "-c", str(cfg_path)])

    assert run_result.exit_code == 0, run_result.output
    assert "--host" in captured["command"]
    host_index = captured["command"].index("--host")
    assert captured["command"][host_index + 1] == "0.0.0.0"
    assert captured["browser_url"] is None
    assert "http://localhost:2718" in run_result.output


def test_url_is_reachable_requires_marimo_markers(monkeypatch) -> None:
    import dnadesign.densegen.src.cli.notebook as notebook_commands

    class _Response:
        def __init__(self, body: str, content_type: str = "text/html", status: int = 200) -> None:
            self._body = body.encode("utf-8")
            self.headers = {"content-type": content_type}
            self.status = status

        def read(self, _n: int = -1) -> bytes:
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        notebook_commands.urllib.request,
        "urlopen",
        lambda request, timeout=0.5: _Response("<html><body>hello</body></html>"),
    )
    assert notebook_commands._url_is_reachable("http://127.0.0.1:1") is False

    marimo_body = (
        "<html><head><meta name='description' content='a marimo app'></head>"
        "<body>marimo <link rel='icon' href='./favicon.ico'></body></html>"
    )
    monkeypatch.setattr(
        notebook_commands.urllib.request,
        "urlopen",
        lambda request, timeout=0.5: _Response(marimo_body),
    )
    assert notebook_commands._url_is_reachable("http://127.0.0.1:1") is True

    marimo_data_attr_body = '<html><body><script data-marimo="true"></script></body></html>'
    monkeypatch.setattr(
        notebook_commands.urllib.request,
        "urlopen",
        lambda request, timeout=0.5: _Response(marimo_data_attr_body),
    )
    assert notebook_commands._url_is_reachable("http://127.0.0.1:1") is True


def test_running_notebook_filename_extracts_tag_value(monkeypatch) -> None:
    import dnadesign.densegen.src.cli.notebook as notebook_commands

    class _Response:
        def __init__(self, body: str, content_type: str = "text/html", status: int = 200) -> None:
            self._body = body.encode("utf-8")
            self.headers = {"content-type": content_type}
            self.status = status

        def read(self, _n: int = -1) -> bytes:
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    body = (
        "<html><body><marimo-filename hidden>"
        "/tmp/a&amp;b/outputs/notebooks/densegen_run_overview.py"
        "</marimo-filename></body></html>"
    )
    monkeypatch.setattr(
        notebook_commands.urllib.request,
        "urlopen",
        lambda request, timeout=0.5: _Response(body),
    )
    assert notebook_commands._running_notebook_filename("http://127.0.0.1:1") == (
        "/tmp/a&b/outputs/notebooks/densegen_run_overview.py"
    )


def test_running_notebook_filename_returns_none_without_tag(monkeypatch) -> None:
    import dnadesign.densegen.src.cli.notebook as notebook_commands

    class _Response:
        def __init__(self, body: str, content_type: str = "text/html", status: int = 200) -> None:
            self._body = body.encode("utf-8")
            self.headers = {"content-type": content_type}
            self.status = status

        def read(self, _n: int = -1) -> bytes:
            return self._body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(
        notebook_commands.urllib.request,
        "urlopen",
        lambda request, timeout=0.5: _Response("<html><body>hello</body></html>"),
    )
    assert notebook_commands._running_notebook_filename("http://127.0.0.1:1") is None


def test_port_is_available_checks_all_resolved_host_addresses(monkeypatch) -> None:
    import dnadesign.densegen.src.cli.notebook as notebook_commands

    attempts: list[tuple[int, object]] = []

    def _fake_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        assert host == "localhost"
        assert port == 2718
        return [
            (socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("::1", 2718, 0, 0)),
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", 2718)),
        ]

    class _FakeSocket:
        def __init__(self, family, *args):
            self.family = family

        def setsockopt(self, *args) -> None:
            return None

        def bind(self, sockaddr) -> None:
            attempts.append((self.family, sockaddr))
            if self.family == socket.AF_INET6:
                raise OSError("already in use")

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(notebook_commands.socket, "getaddrinfo", _fake_getaddrinfo)
    monkeypatch.setattr(notebook_commands.socket, "socket", _FakeSocket)

    assert notebook_commands._port_is_available("localhost", 2718) is False
    assert attempts == [(socket.AF_INET6, ("::1", 2718, 0, 0))]


def test_find_available_port_retries_until_port_is_usable(monkeypatch) -> None:
    import dnadesign.densegen.src.cli.notebook as notebook_commands

    ephemeral_ports = iter([33001, 33002])
    availability_checks: list[int] = []

    def _fake_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        assert host == "localhost"
        return [(socket.AF_INET6, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("::1", port, 0, 0))]

    class _FakeSocket:
        def __init__(self, family, *args):
            self.family = family
            self._sockaddr = None

        def setsockopt(self, *args) -> None:
            return None

        def bind(self, sockaddr) -> None:
            self._sockaddr = sockaddr

        def getsockname(self):
            return ("::1", next(ephemeral_ports), 0, 0)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    def _fake_port_is_available(host: str, port: int) -> bool:
        assert host == "localhost"
        availability_checks.append(port)
        return port == 33002

    monkeypatch.setattr(notebook_commands.socket, "getaddrinfo", _fake_getaddrinfo)
    monkeypatch.setattr(notebook_commands.socket, "socket", _FakeSocket)
    monkeypatch.setattr(notebook_commands, "_port_is_available", _fake_port_is_available)

    assert notebook_commands._find_available_port("localhost") == 33002
    assert availability_checks == [33001, 33002]


def test_open_browser_tab_does_not_use_webbrowser_fallback(monkeypatch) -> None:
    import dnadesign.densegen.src.cli.notebook as notebook_commands

    monkeypatch.setattr(notebook_commands, "_is_wsl", lambda: False)
    monkeypatch.setattr(notebook_commands.shutil, "which", lambda name: None)
    run_calls: list[list[str]] = []
    monkeypatch.setattr(
        notebook_commands.subprocess,
        "run",
        lambda command, check, stdout, stderr: run_calls.append(command) or subprocess.CompletedProcess(command, 0),
    )

    assert notebook_commands._open_browser_tab("http://127.0.0.1:2718") is False
    assert run_calls == []


def test_run_marimo_command_starts_new_session(monkeypatch) -> None:
    import dnadesign.densegen.src.cli.notebook as notebook_commands

    popen_kwargs: dict[str, object] = {}

    class _FakeProcess:
        def poll(self):
            return None

        def wait(self, timeout: float | None = None):
            return 0

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            return None

    def _fake_popen(*args, **kwargs):
        popen_kwargs.update(kwargs)
        return _FakeProcess()

    monkeypatch.setattr(notebook_commands.subprocess, "Popen", _fake_popen)
    opened = notebook_commands._run_marimo_command(command=["marimo", "run", "app.py"], env={}, browser_url=None)

    assert opened is False
    assert popen_kwargs.get("start_new_session") == (os.name == "posix")


def test_run_marimo_command_does_not_open_unreachable_url(monkeypatch) -> None:
    import dnadesign.densegen.src.cli.notebook as notebook_commands

    class _FakeProcess:
        def poll(self):
            return None

        def wait(self, timeout: float | None = None):
            return 0

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            return None

    monkeypatch.setattr(notebook_commands.subprocess, "Popen", lambda *args, **kwargs: _FakeProcess())
    monkeypatch.setattr(notebook_commands, "_url_is_reachable", lambda url: False)
    browser_attempts: list[str] = []
    monkeypatch.setattr(notebook_commands, "_open_browser_tab", lambda url: browser_attempts.append(url) or False)
    warnings: list[str] = []
    opened = notebook_commands._run_marimo_command(
        command=["marimo", "run", "app.py"],
        env={},
        browser_url="http://127.0.0.1:2718",
        open_timeout_seconds=0.0,
        on_browser_open_failure=lambda reason: warnings.append(reason),
    )

    assert opened is False
    assert browser_attempts == []
    assert warnings == ["notebook-not-reachable"]


def test_run_marimo_command_reachable_open_failure_warns_once(monkeypatch) -> None:
    import dnadesign.densegen.src.cli.notebook as notebook_commands

    class _FakeProcess:
        def poll(self):
            return None

        def wait(self, timeout: float | None = None):
            return 0

        def terminate(self) -> None:
            return None

        def kill(self) -> None:
            return None

    monkeypatch.setattr(notebook_commands.subprocess, "Popen", lambda *args, **kwargs: _FakeProcess())
    monkeypatch.setattr(notebook_commands, "_url_is_reachable", lambda url: True)
    browser_attempts: list[str] = []
    monkeypatch.setattr(notebook_commands, "_open_browser_tab", lambda url: browser_attempts.append(url) or False)
    warnings: list[str] = []
    opened = notebook_commands._run_marimo_command(
        command=["marimo", "run", "app.py"],
        env={},
        browser_url="http://127.0.0.1:2718",
        open_timeout_seconds=5.0,
        on_browser_open_failure=lambda reason: warnings.append(reason),
    )

    assert opened is False
    assert browser_attempts == ["http://127.0.0.1:2718"]
    assert warnings == ["browser-open-failed"]


def test_release_workspace_notebook_port_clears_corrupt_state_file(tmp_path: Path) -> None:
    import dnadesign.densegen.src.cli.notebook as notebook_commands

    run_root = tmp_path
    notebook_path = run_root / "outputs" / "notebooks" / "densegen_run_overview.py"
    state_path = notebook_commands._notebook_server_state_path(run_root)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text("{broken-json")

    released = notebook_commands._release_workspace_notebook_port(
        run_root=run_root,
        host="127.0.0.1",
        port=2718,
        notebook_path=notebook_path,
    )

    assert released is False
    assert state_path.exists() is False


def test_release_workspace_notebook_port_clears_state_when_pid_is_dead(tmp_path: Path, monkeypatch) -> None:
    import dnadesign.densegen.src.cli.notebook as notebook_commands

    run_root = tmp_path
    notebook_path = run_root / "outputs" / "notebooks" / "densegen_run_overview.py"
    state_path = notebook_commands._notebook_server_state_path(run_root)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "pid": 731,
                "host": "127.0.0.1",
                "port": 2718,
                "notebook_path": str(notebook_path.resolve()),
            }
        )
    )
    monkeypatch.setattr(notebook_commands, "_process_is_running", lambda pid: False)

    released = notebook_commands._release_workspace_notebook_port(
        run_root=run_root,
        host="127.0.0.1",
        port=2718,
        notebook_path=notebook_path,
    )

    assert released is False
    assert state_path.exists() is False


def test_release_workspace_notebook_port_keeps_state_when_terminate_fails(tmp_path: Path, monkeypatch) -> None:
    import dnadesign.densegen.src.cli.notebook as notebook_commands

    run_root = tmp_path
    notebook_path = run_root / "outputs" / "notebooks" / "densegen_run_overview.py"
    state_path = notebook_commands._notebook_server_state_path(run_root)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "pid": 944,
                "host": "127.0.0.1",
                "port": 2718,
                "notebook_path": str(notebook_path.resolve()),
            }
        )
    )
    monkeypatch.setattr(notebook_commands, "_process_is_running", lambda pid: True)
    monkeypatch.setattr(notebook_commands, "_terminate_process_tree", lambda pid: False)

    released = notebook_commands._release_workspace_notebook_port(
        run_root=run_root,
        host="127.0.0.1",
        port=2718,
        notebook_path=notebook_path,
    )

    assert released is False
    assert state_path.exists() is True


def test_release_tracked_notebook_server_if_stale_clears_state_and_stops_process(tmp_path: Path, monkeypatch) -> None:
    import dnadesign.densegen.src.cli.notebook as notebook_commands

    run_root = tmp_path
    notebook_path = run_root / "outputs" / "notebooks" / "densegen_run_overview.py"
    notebook_path.parent.mkdir(parents=True, exist_ok=True)
    notebook_path.write_text("print('fresh notebook')\n")
    state_path = notebook_commands._notebook_server_state_path(run_root)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(
            {
                "pid": 1234,
                "host": "127.0.0.1",
                "port": 2718,
                "notebook_path": str(notebook_path.resolve()),
                "notebook_digest": "stale-digest",
            }
        )
    )
    monkeypatch.setattr(notebook_commands, "_process_is_running", lambda pid: True)
    monkeypatch.setattr(notebook_commands, "_terminate_process_tree", lambda pid: True)

    released = notebook_commands._release_tracked_notebook_server_if_stale(
        run_root=run_root,
        notebook_path=notebook_path,
        notebook_digest=notebook_commands._notebook_content_digest(notebook_path),
    )

    assert released is True
    assert state_path.exists() is False


def test_assert_plot_artifacts_are_fresh_rejects_stale_current_inventory_outputs(tmp_path: Path, monkeypatch) -> None:
    import dnadesign.densegen.src.cli.notebook as notebook_commands

    run_root = tmp_path
    plot_root = run_root / "outputs" / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    artifact_path = plot_root / "stage_a" / "sampling_vs_length_ridgeline.pdf"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("pdf")
    inventory_path = plot_root / "current_inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "plots": [
                    {
                        "name": "stage_a_summary",
                        "path": "stage_a/sampling_vs_length_ridgeline.pdf",
                    }
                ]
            }
        )
    )
    render_source = tmp_path / "render_source.py"
    render_source.write_text("print('new renderer')\n")
    os.utime(artifact_path, (1_000, 1_000))
    os.utime(render_source, (2_000, 2_000))
    monkeypatch.setattr(notebook_commands, "_notebook_render_source_files", lambda: [render_source])

    with pytest.raises(RuntimeError, match="Run `dense plot` before launching a fresh notebook"):
        notebook_commands._assert_plot_artifacts_are_fresh(run_root=run_root)


def test_assert_notebook_plot_inventory_is_complete_requires_current_inventory(tmp_path: Path) -> None:
    import dnadesign.densegen.src.cli.notebook as notebook_commands

    with pytest.raises(RuntimeError, match="current_inventory.json is missing"):
        notebook_commands._assert_notebook_plot_inventory_is_complete(run_root=tmp_path)


def test_assert_notebook_plot_inventory_is_complete_rejects_missing_visible_plot_ids(tmp_path: Path) -> None:
    import dnadesign.densegen.src.cli.notebook as notebook_commands

    run_root = tmp_path
    plot_root = run_root / "outputs" / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    inventory_path = plot_root / "current_inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "schema_version": "densegen.current_inventory.v2",
                "plots": [
                    {
                        "plot_id": "source_cohort_concentration",
                        "path": "dataset/source_cohort_concentration.pdf",
                    },
                    {
                        "plot_id": "stage_a_sampling_yield",
                        "path": "stage_a/stage_a_sampling_yield.pdf",
                    },
                ],
            }
        )
    )

    with pytest.raises(RuntimeError, match="missing required notebook plot ids"):
        notebook_commands._assert_notebook_plot_inventory_is_complete(run_root=run_root)


def test_assert_notebook_plot_inventory_is_complete_rejects_core_only_surface_without_notebook_drilldowns(
    tmp_path: Path,
) -> None:
    import dnadesign.densegen.src.cli.notebook as notebook_commands

    plot_root = tmp_path / "outputs" / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    (plot_root / "current_inventory.json").write_text(
        json.dumps(
            {
                "schema_version": "densegen.current_inventory.v2",
                "plots": [
                    {"plot_id": "source_cohort_concentration", "path": "dataset/source_cohort_concentration.pdf"},
                    {"plot_id": "stage_a_sampling_yield", "path": "stage_a/stage_a_sampling_yield.pdf"},
                    {"plot_id": "stage_a_pool_diversity", "path": "stage_a/stage_a_pool_diversity.pdf"},
                    {
                        "plot_id": "plan_regulator_deployment_heatmap",
                        "path": "stage_b_summary/plan_regulator_deployment_heatmap.pdf",
                    },
                    {
                        "plot_id": "placement_occupancy_map",
                        "path": "stage_b/demo_plan/demo_input/placement_occupancy_map.pdf",
                    },
                    {
                        "plot_id": "retained_pool_coverage_by_regulator",
                        "path": "stage_b_summary/retained_pool_coverage_by_regulator.pdf",
                    },
                    {"plot_id": "attempt_outcome_timeline", "path": "run_health/attempt_outcome_timeline.pdf"},
                    {"plot_id": "solve_pressure_and_progress", "path": "run_health/solve_pressure_and_progress.pdf"},
                ],
            }
        )
    )

    with pytest.raises(RuntimeError, match="source_plan_input_heatmap"):
        notebook_commands._assert_notebook_plot_inventory_is_complete(run_root=tmp_path)


def test_assert_notebook_plot_inventory_is_complete_requires_grouped_stage_b_core_scopes(tmp_path: Path) -> None:
    import dnadesign.densegen.src.cli.notebook as notebook_commands

    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            densegen:
              schema_version: "2.9"
              run:
                id: demo
                root: "."
              inputs:
                - name: demo_input
                  type: binding_sites
                  path: inputs.csv
              output:
                targets: [parquet]
                schema:
                  bio_type: dna
                  alphabet: dna_4
                parquet:
                  path: outputs/tables/records.parquet
              generation:
                sequence_length: 10
                plan:
                  - name: background_only__sig35=a
                    sequences: 1
                    sampling:
                      include_inputs: [demo_input]
                    regulator_constraints:
                      groups: []
                  - name: ethanol__sig35=a
                    sequences: 1
                    sampling:
                      include_inputs: [demo_input]
                    regulator_constraints:
                      groups: []
                  - name: background_only__sig35=b
                    sequences: 1
                    sampling:
                      include_inputs: [demo_input]
                    regulator_constraints:
                      groups: []
                  - name: ethanol__sig35=b
                    sequences: 1
                    sampling:
                      include_inputs: [demo_input]
                    regulator_constraints:
                      groups: []
              solver:
                backend: CBC
                strategy: iterate
              logging:
                log_dir: outputs/logs
            plots:
              out_dir: outputs/plots
              format: pdf
              default:
                [
                  source_cohort_concentration,
                  stage_a_sampling_yield,
                  stage_a_pool_diversity,
                  plan_regulator_deployment_heatmap,
                  placement_occupancy_map,
                  retained_pool_coverage_by_regulator,
                  attempt_outcome_timeline,
                  solve_pressure_and_progress
                ]
              options:
                placement_occupancy_map:
                  scope: auto
                  max_plans: 2
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    plot_root = tmp_path / "outputs" / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    (plot_root / "current_inventory.json").write_text(
        json.dumps(
            {
                "schema_version": "densegen.current_inventory.v2",
                "plots": [
                    {"plot_id": "source_cohort_concentration", "path": "dataset/source_cohort_concentration.pdf"},
                    {"plot_id": "stage_a_sampling_yield", "path": "stage_a/stage_a_sampling_yield.pdf"},
                    {"plot_id": "stage_a_pool_diversity", "path": "stage_a/stage_a_pool_diversity.pdf"},
                    {
                        "plot_id": "plan_regulator_deployment_heatmap",
                        "path": "stage_b_summary/plan_regulator_deployment_heatmap.pdf",
                    },
                    {
                        "plot_id": "placement_occupancy_map",
                        "path": "stage_b/background_only/placement_occupancy_map.pdf",
                    },
                    {
                        "plot_id": "retained_pool_coverage_by_regulator",
                        "path": "stage_b_summary/retained_pool_coverage_by_regulator.pdf",
                    },
                    {"plot_id": "attempt_outcome_timeline", "path": "run_health/attempt_outcome_timeline.pdf"},
                    {"plot_id": "solve_pressure_and_progress", "path": "run_health/solve_pressure_and_progress.pdf"},
                ],
            }
        )
    )

    with pytest.raises(RuntimeError, match=r"placement_occupancy_map\[ethanol\]"):
        notebook_commands._assert_notebook_plot_inventory_is_complete(
            run_root=tmp_path,
            config_path=cfg_path,
        )


def test_assert_notebook_plot_inventory_is_complete_rejects_legacy_taxonomy(tmp_path: Path) -> None:
    import dnadesign.densegen.src.cli.notebook as notebook_commands

    plot_root = tmp_path / "outputs" / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    (plot_root / "current_inventory.json").write_text(
        json.dumps(
            {
                "schema_version": "densegen.current_inventory.v1",
                "plots": [
                    {"plot_id": "stage_a_summary", "path": "stage_a/yield_bias.pdf"},
                ],
            }
        )
    )

    with pytest.raises(RuntimeError, match="legacy|unsupported inventory taxonomy"):
        notebook_commands._assert_notebook_plot_inventory_is_complete(run_root=tmp_path)


def test_assert_plot_artifacts_are_fresh_ignores_hidden_current_inventory_outputs(tmp_path: Path, monkeypatch) -> None:
    import dnadesign.densegen.src.cli.notebook as notebook_commands

    run_root = tmp_path
    plot_root = run_root / "outputs" / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    artifact_path = plot_root / "support" / "summary.csv"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text("pdf")
    inventory_path = plot_root / "current_inventory.json"
    inventory_path.write_text(
        json.dumps(
            {
                "schema_version": "densegen.current_inventory.v2",
                "plots": [
                    {
                        "plot_id": "support_summary_csv",
                        "visual_plot_type": "support_summary_csv",
                        "path": "support/summary.csv",
                    }
                ],
            }
        )
    )
    render_source = tmp_path / "render_source.py"
    render_source.write_text("print('new renderer')\n")
    os.utime(artifact_path, (1_000, 1_000))
    os.utime(render_source, (2_000, 2_000))
    monkeypatch.setattr(notebook_commands, "_notebook_render_source_files", lambda: [render_source])

    notebook_commands._assert_plot_artifacts_are_fresh(run_root=run_root)


def test_plot_missing_records_reports_actionable_error(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["plot", "-c", str(cfg_path)])

    assert result.exit_code == 1
    assert "Plot generation failed" in result.output
    assert "Parquet output not found" in result.output
    assert "dense run --fresh --no-plot" in result.output


def test_plot_missing_records_with_dual_sinks_reports_plots_source_hint(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        textwrap.dedent(
            """
            densegen:
              schema_version: "2.9"
              run:
                id: demo
                root: "."
              inputs:
                - name: demo_input
                  type: binding_sites
                  path: inputs.csv
              output:
                targets: [parquet, usr]
                schema:
                  bio_type: dna
                  alphabet: dna_4
                parquet:
                  path: outputs/tables/records.parquet
                usr:
                  root: ../usr_root
                  dataset: densegen_demo
              generation:
                sequence_length: 10
                plan:
                  - name: demo_plan
                    sequences: 1
                    sampling:
                      include_inputs: [demo_input]
                    regulator_constraints:
                      groups: []
              solver:
                backend: CBC
                strategy: iterate
              logging:
                log_dir: outputs/logs
            plots:
              source: parquet
              out_dir: outputs/plots
              default: [placement_occupancy_map]
            """
        ).strip()
        + "\n"
    )
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["plot", "-c", str(cfg_path)])

    assert result.exit_code == 1
    assert "Plot generation failed" in result.output
    assert "Parquet output not found" in result.output


def test_plot_invalid_only_name_reports_ls_plots_recovery(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    write_minimal_config(cfg_path)
    (tmp_path / "inputs.csv").write_text("tf,tfbs\n")

    runner = CliRunner()
    result = runner.invoke(app, ["plot", "--only", "definitely_missing", "-c", str(cfg_path)])

    assert result.exit_code == 1
    assert "Unknown plot name requested: definitely_missing" in result.output
    assert "dense ls-plots" in result.output
    assert "dense run --fresh --no-plot" not in result.output
    assert "dense run --resume --no-plot" not in result.output
