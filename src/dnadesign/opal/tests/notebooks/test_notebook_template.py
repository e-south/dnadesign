import ast
from pathlib import Path

from dnadesign.opal.src.analysis.notebook_components import (
    build_notebook_artifact_garden_lines,
    build_notebook_artifact_garden_rows,
    build_notebook_at_a_glance_rows,
    build_notebook_baserender_contract,
    build_notebook_baserender_contract_rows,
    build_notebook_change_lines,
    build_notebook_change_rows,
    build_notebook_evidence_rows,
    build_notebook_metric_definition_rows,
    build_notebook_plot_card_rows,
    build_notebook_plot_method_rows,
    build_notebook_run_summary_lines,
    build_notebook_validity_lines,
    build_notebook_visual_surface_model,
    render_notebook_baserender_record,
    render_visual_surface_cells,
)
from dnadesign.opal.src.analysis.notebook_set_template import render_campaign_set_notebook
from dnadesign.opal.src.analysis.notebook_template import render_campaign_notebook


def test_notebook_template_data_source_options() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert "predictions (selected run)" in text
    assert "labels (all rounds)" in text


def test_notebook_template_uses_medium_width() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert 'marimo.App(width="medium")' in text


def test_notebook_template_removes_extra_tables() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert "mo.ui.dataframe(summary_df)" not in text
    assert "mo.ui.dataframe(labels_df)" not in text
    assert "mo.ui.data_explorer(filtered_df)" not in text


def test_notebook_template_has_visual_surface() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert "Visual surface" in text
    assert "plots_dir" in text
    assert "build_notebook_view_model" in text
    assert "Select one operative visual surface" in text
    assert '"Plot deliverables": plot_panel' not in text


def test_notebook_template_has_opal_schema_sentinel() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")

    assert '__opal_notebook_template_schema__ = "opal.generated_campaign_notebook.v1"' in text


def test_notebook_template_does_not_read_widget_values_in_definition_cells() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    for cell in text.split("@app.cell"):
        if " = mo.ui.dropdown(" in cell:
            assert ".value" not in cell


def test_notebook_template_uses_visual_surface_component() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    surface_text = render_visual_surface_cells()

    assert render_visual_surface_cells.__module__.endswith(".notebook_components.visual_surface")
    assert "def render_visual_surface_cells" not in text
    assert surface_text in text
    assert "visual_surface_note" in surface_text
    assert "manifest-backed plot outputs" in surface_text
    assert 'label="Visual"' in surface_text
    assert "Record render" in surface_text
    assert "thumbnail_gallery" not in surface_text


def test_notebook_template_does_not_hide_generic_plots_for_sfxi_campaigns() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")

    assert "SFXI plots only" not in text
    assert "Non-SFXI plots only" not in text
    assert "plot_entries_filtered" not in text


def test_notebook_template_is_campaign_specific_accordion_surface() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")

    assert "# OPAL Campaign Notebook" in text
    assert "Campaign analysis command surface" in text
    assert "mo.accordion(" in text
    for section in [
        "Campaign contract",
        "Round and run",
        "Ledger readiness",
        "Records and active record",
        "Labels and predictions",
        "Validity",
        "Changes",
        "X provenance and limitations",
        "Metric definitions",
        "Artifacts",
    ]:
        assert section in text


def test_notebook_template_uses_public_opal_helpers() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")

    assert "from dnadesign.opal.notebooks.api import" in text
    assert "dnadesign.opal.src" not in text
    assert "assess_records_contract_for_schema" in text
    assert "build_ledger_status_table" in text
    assert "build_records_preview" in text
    assert "cli_handoff_lines" in text
    assert "compact_notebook_path" in text
    assert "read_optional_table" in text
    assert "records_status_lines" in text


def test_notebook_template_degrades_without_runs() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")

    assert "build_notebook_no_run_lines" in text
    assert "mo.stop(len(rounds) == 0" not in text
    assert 'default_source = "records"' in text


def test_notebook_template_can_pin_initial_run_scope() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="0", run_id="run-1")

    assert "default_round = '0'" in text
    assert "default_run_id = 'run-1'" in text
    assert "run_default = default_run_id if default_run_id in run_options" in text
    summary = "\n".join(
        build_notebook_run_summary_lines(
            "run-1",
            {"as_of_round": 0, "selection__name": "top_n", "model__name": "rf"},
            "sfxi",
            selected_round=0,
            default_round=0,
            run_options=["run-0", "run-1"],
        )
    )
    assert "Run scope: selected round `0`, selected run `run-1`." in summary


def test_notebook_template_keeps_lateral_tools_out() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")

    assert "dnadesign.baserender" not in text
    assert "densegen__visual" not in text
    assert "cluster__ldn_v1__umap_x" not in text
    assert "cluster__ldn_v1__umap_y" not in text
    assert "obj__logic_fidelity" not in text
    assert "obj__effect_raw" not in text
    assert "obj__effect_scaled" not in text


def test_notebook_template_omits_altair_import() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert "import altair as alt" not in text


def test_notebook_template_uses_schema_pruned_records_loading() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")

    assert "store.load()" not in text
    assert "store.schema_columns()" in text
    assert "store.load_columns(records_loaded_columns)" in text
    assert "records_df = pl.from_pandas" in text


def test_notebook_component_primitives_build_shared_evidence_models() -> None:
    view_model = {
        "campaign": {
            "slug": "campaign_a",
            "config_path": "campaign.yaml",
            "workdir": "workdir",
            "x_column": "x_vec",
            "label_source": "usr_sidecar",
        },
        "status": {
            "progress_status": "attention",
            "round_selector": "latest",
            "round_count": 3,
            "latest_run_id": "run-3",
        },
        "progress": {
            "schema_version": "opal.campaign_progress.v1",
            "state": {"exists": True, "path": "workdir/state.json"},
            "round_selector": "latest",
            "event_contract": {
                "schema_version": "opal.progress_event_rollup.v1",
                "command_events": 1,
                "preflight_events": 2,
                "run_events": 6,
                "finalize_events": 1,
                "abort_events": 0,
                "attempt_ids": ["attempt-3"],
                "aborted_rounds": [],
                "ambiguous_rounds": [],
            },
            "rounds": [
                {
                    "round_index": 3,
                    "status": "done",
                    "last_stage": "round_done",
                    "elapsed_sec": 12.5,
                    "events": 10,
                    "predict": {"batch": 4, "of": 4, "rows": 157},
                    "summary": {
                        "aborted": False,
                        "run_scope": {
                            "resolved_run_id": "run-3",
                            "run_ids": ["run-3"],
                            "attempt_ids": ["attempt-3"],
                            "ambiguous_run_scope": False,
                        },
                    },
                    "path": "workdir/outputs/rounds/round_3/logs/round.log.jsonl",
                }
            ],
        },
        "review_manifest": {
            "schema_version": "opal.campaign_review.v1",
            "selection": {"selected_count": 6},
        },
        "plot_manifests": [
            {
                "name": "score",
                "kind": "metric_over_rounds",
                "status": "written",
                "generated_at": "2026-05-21T00:00:00Z",
                "run_id": "run-3",
                "rounds": [3],
                "tidy_csv": "plots/score.csv",
                "params": {"metric": "pred__score_selected"},
                "freshness": {"status": "fresh"},
                "metadata": {
                    "data_shape": "scalar over rounds",
                    "tidy_schema": ["round", "cohort", "metric", "summary", "value"],
                    "failure_modes": ["missing metric column", "metric is not numeric"],
                },
                "inputs": [{"path": "outputs/ledger/predictions", "role": "run_pred"}],
                "outputs": [{"role": "media", "path": "plots/score.png", "exists": True}],
            }
        ],
        "warnings": [
            {
                "category": "ReviewManifestWarning",
                "severity": "warning",
                "message": "Review manifest not found",
            }
        ],
        "stale_artifacts": [
            {
                "category": "StaleArtifactWarning",
                "severity": "warning",
                "path": "plots/old.png",
                "message": "old plot",
            }
        ],
        "artifact_garden": {
            "schema_version": "opal.artifact_garden.v1",
            "root": "workdir",
            "local_only": True,
            "artifact_roots": [
                {"name": "outputs", "path": "workdir/outputs", "exists": True, "file_count": 7, "size_bytes": 2048}
            ],
            "active_manifests": [{"kind": "plot_index", "status": "loaded"}],
            "stale_artifacts": [
                {"scope": "configured_plots", "path": "plots/old.png", "size_bytes": 12, "reason": "manifest absent"}
            ],
            "bytes": {"artifact_roots": 2048, "stale_artifacts": 12},
            "prune_plan": {"item_count": 1, "bytes_to_delete": 12, "requires_apply": True},
        },
    }

    glance_rows = build_notebook_at_a_glance_rows(view_model)
    assert {"field": "campaign", "value": "campaign_a"} in glance_rows
    assert {"field": "X column", "value": "x_vec"} in glance_rows
    assert {"field": "stale artifacts", "value": 1} in glance_rows

    visual_surface = build_notebook_visual_surface_model(view_model)
    assert visual_surface["missing_outputs"] == []
    assert visual_surface["stale_artifacts"] == view_model["stale_artifacts"]
    assert visual_surface["choices"][0]["label"] == "Score"
    assert visual_surface["choices"][0]["path_label"] == "plots/score.png"

    card_rows = build_notebook_plot_card_rows(visual_surface["choices"][0])
    assert {"field": "media", "value": "plots/score.png"} in card_rows
    assert {"field": "freshness", "value": "fresh"} in card_rows
    assert {"field": "tidy data", "value": "plots/score.csv"} in card_rows
    assert any(row["field"] == "source data" for row in card_rows)
    method_rows = build_notebook_plot_method_rows(visual_surface["choices"][0])
    assert any(row["section"] == "math" and "mean = sum(x) / n" in row["detail"] for row in method_rows)

    evidence = build_notebook_evidence_rows(view_model)
    assert [row["source"] for row in evidence] == ["path", "path", "warning", "stale_artifact"]
    assert evidence[0]["path"] == "campaign.yaml"

    validity_lines = "\n".join(build_notebook_validity_lines(view_model))
    assert "Review manifest: `present`" in validity_lines
    assert "Plot manifests: `1`" in validity_lines
    assert "Missing plot outputs: `0`" in validity_lines
    assert "Artifact garden: `opal.artifact_garden.v1`" in validity_lines

    change_lines = "\n".join(build_notebook_change_lines(view_model))
    assert "Latest run ID: `run-3`" in change_lines
    assert "Event phases: command=`1`, preflight=`2`, run=`6`, finalize=`1`" in change_lines
    change_rows = build_notebook_change_rows(view_model)
    assert change_rows == [
        {
            "round": 3,
            "status": "done",
            "last_stage": "round_done",
            "run_id": "run-3",
            "attempts": 1,
            "events": 10,
            "elapsed_sec": 12.5,
            "predict": "4/4 batches, 157 rows",
            "aborted": False,
            "ambiguous_run_scope": False,
            "log": "outputs/rounds/round_3/logs/round.log.jsonl",
        }
    ]

    metric_rows = build_notebook_metric_definition_rows(view_model)
    assert metric_rows == [
        {
            "plot": "score",
            "kind": "metric_over_rounds",
            "data_shape": "scalar over rounds",
            "tidy_schema": "round, cohort, metric, summary, value",
            "failure_modes": "missing metric column; metric is not numeric",
            "freshness": "fresh",
            "purpose": "not recorded",
        }
    ]

    artifact_lines = "\n".join(build_notebook_artifact_garden_lines(view_model))
    assert "local-only" in artifact_lines
    assert "Stale artifacts: `1`" in artifact_lines
    artifact_rows = build_notebook_artifact_garden_rows(view_model)
    assert [row["source"] for row in artifact_rows] == ["artifact_root", "stale_artifact", "prune_plan"]


def test_notebook_template_is_valid_python() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    ast.parse(text)


def test_campaign_set_notebook_has_campaign_and_plot_dropdowns() -> None:
    text = render_campaign_set_notebook(
        [Path("campaign_a.yaml"), Path("campaign_b.yaml")],
        round_selector="latest",
    )

    assert "# OPAL Campaign Set Notebook" in text
    assert "from dnadesign.opal.notebooks.api import (" in text
    assert "build_campaign_set_notebook_view_model" in text
    assert "dnadesign.opal.src" not in text
    assert "__generated_with" in text
    assert 'generated_with = "' in text
    assert "Generated with marimo: `{__generated_with}`" not in text
    assert 'label="Campaign"' in text
    assert 'label="Visual"' in text
    assert "## Visual surface" in text
    assert "Validity" in text
    assert "Changes" in text
    assert "build_notebook_artifact_garden_rows" in text
    assert "build_notebook_change_rows" in text
    assert "build_notebook_metric_definition_rows" in text
    assert "build_notebook_visual_surface_model" in text
    assert "build_notebook_plot_card_rows" in text
    assert "build_notebook_plot_method_rows" in text
    assert "build_notebook_validity_lines" in text
    assert "campaign_set_view_model" in text
    assert "LatentDNA" not in text
    assert "UMAP" not in text
    ast.parse(text)


def test_notebook_templates_stay_bounded_wiring_surfaces() -> None:
    single = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    campaign_set = render_campaign_set_notebook(
        [Path("campaign_a.yaml"), Path("campaign_b.yaml")],
        round_selector="latest",
    )

    assert len(single.splitlines()) <= 1000
    assert len(campaign_set.splitlines()) <= 340


def test_notebook_baserender_contract_detects_schema_without_generated_import() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert "render_notebook_baserender_record" in text
    assert "from dnadesign.baserender import" not in text

    unavailable = build_notebook_baserender_contract(["id", "sequence"], records_path="records.parquet")
    assert unavailable["available"] is False
    assert build_notebook_baserender_contract_rows(unavailable)[0] == {"field": "available", "value": "false"}

    contract = build_notebook_baserender_contract(
        ["id", "sequence", "densegen__used_tfbs_detail"],
        records_path="records.parquet",
    )
    assert contract["available"] is True
    assert contract["adapter_kind"] == "densegen_tfbs"
    assert contract["adapter_columns"]["annotations"] == "densegen__used_tfbs_detail"
    assert callable(render_notebook_baserender_record)
