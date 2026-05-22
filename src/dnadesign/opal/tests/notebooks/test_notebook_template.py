import ast
from pathlib import Path

from dnadesign.opal.src.analysis.notebook_components import (
    build_notebook_artifact_garden_lines,
    build_notebook_artifact_garden_rows,
    build_notebook_at_a_glance_lines,
    build_notebook_change_lines,
    build_notebook_change_rows,
    build_notebook_evidence_rows,
    build_notebook_metric_definition_rows,
    build_notebook_plot_card_lines,
    build_notebook_plot_gallery_model,
    build_notebook_validity_lines,
    render_plot_gallery_cells,
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


def test_notebook_template_has_plot_gallery() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert "Plot artifacts" in text
    assert "outputs/plots" in text
    assert "build_notebook_view_model" in text
    assert "All configured plots with written manifests." in text


def test_notebook_template_has_opal_schema_sentinel() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")

    assert '__opal_notebook_template_schema__ = "opal.generated_campaign_notebook.v1"' in text


def test_notebook_template_does_not_read_widget_values_in_definition_cells() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    for cell in text.split("@app.cell"):
        if " = mo.ui.dropdown(" in cell:
            assert ".value" not in cell


def test_notebook_template_uses_plot_gallery_component() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    gallery_text = render_plot_gallery_cells()

    assert render_plot_gallery_cells.__module__.endswith(".notebook_components.plots")
    assert "def render_plot_gallery_cells" not in text
    assert gallery_text in text
    assert "plot_gallery_note" in gallery_text
    assert "manifest-backed plot outputs" in gallery_text


def test_notebook_template_does_not_hide_generic_plots_for_sfxi_campaigns() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")

    assert "SFXI plots only" not in text
    assert "Non-SFXI plots only" not in text
    assert "plot_entries_filtered" not in text


def test_notebook_template_is_campaign_specific_accordion_surface() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")

    assert "# OPAL Campaign Notebook" in text
    assert "Campaign-specific artifact viewer" in text
    assert "mo.accordion(" in text
    for section in [
        "Campaign contract",
        "Round and run",
        "Ledger readiness",
        "Records and active record",
        "Labels and predictions",
        "Plot deliverables",
        "Validity",
        "Changes",
        "X provenance and limitations",
        "Metric definitions",
        "Artifacts",
    ]:
        assert section in text


def test_notebook_template_uses_public_opal_helpers() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")

    assert "from dnadesign.opal import" in text
    assert "dnadesign.opal.src" not in text
    assert "assess_records_contract_for_schema" in text
    assert "build_ledger_status_table" in text
    assert "build_records_preview" in text
    assert "cli_handoff_lines" in text
    assert "read_optional_table" in text
    assert "records_status_lines" in text


def test_notebook_template_degrades_without_runs() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")

    assert "build_notebook_no_run_lines" in text
    assert "mo.stop(len(rounds) == 0" not in text
    assert 'default_source = "records"' in text


def test_notebook_template_keeps_lateral_tools_out() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")

    assert "dnadesign.baserender" not in text
    assert "densegen__visual" not in text
    assert "cluster__ldn_v1__umap_x" not in text
    assert "cluster__ldn_v1__umap_y" not in text


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

    glance = "\n".join(build_notebook_at_a_glance_lines(view_model))
    assert "campaign_a" in glance
    assert "x_vec" in glance
    assert "Stale artifacts" in glance

    gallery = build_notebook_plot_gallery_model(view_model)
    assert gallery["missing_outputs"] == []
    assert gallery["stale_artifacts"] == view_model["stale_artifacts"]
    assert gallery["choices"][0]["label"] == "score (score.png)"

    card = "\n".join(build_notebook_plot_card_lines(gallery["choices"][0]))
    assert "Source data" in card
    assert "Freshness: `fresh`" in card
    assert "Tidy CSV: `plots/score.csv`" in card

    evidence = build_notebook_evidence_rows(view_model)
    assert [row["source"] for row in evidence] == ["warning", "stale_artifact"]

    validity_lines = "\n".join(build_notebook_validity_lines(view_model))
    assert "### Validity" in validity_lines
    assert "Review manifest: `present`" in validity_lines
    assert "Plot manifests: `1`" in validity_lines
    assert "Missing plot outputs: `0`" in validity_lines
    assert "Artifact garden: `opal.artifact_garden.v1`" in validity_lines

    change_lines = "\n".join(build_notebook_change_lines(view_model))
    assert "### Changes" in change_lines
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
            "log_path": "workdir/outputs/rounds/round_3/logs/round.log.jsonl",
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
    assert "from dnadesign.opal import (" in text
    assert "build_campaign_set_notebook_view_model" in text
    assert "dnadesign.opal.src" not in text
    assert 'label="Campaign"' in text
    assert 'label="Plot"' in text
    assert "Validity" in text
    assert "Changes" in text
    assert "build_notebook_artifact_garden_rows" in text
    assert "build_notebook_change_rows" in text
    assert "build_notebook_metric_definition_rows" in text
    assert "build_notebook_plot_gallery_model" in text
    assert "build_notebook_plot_card_lines" in text
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

    assert len(single.splitlines()) <= 810
    assert len(campaign_set.splitlines()) <= 290
