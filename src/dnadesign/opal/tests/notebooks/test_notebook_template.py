import ast
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dnadesign.opal.src.analysis.notebook_components import (
    build_notebook_artifact_garden_lines,
    build_notebook_artifact_garden_rows,
    build_notebook_at_a_glance_rows,
    build_notebook_baserender_contract,
    build_notebook_baserender_contract_rows,
    build_notebook_baserender_record_options,
    build_notebook_campaign_header_lines,
    build_notebook_campaign_set_group_options,
    build_notebook_campaign_set_metric_comparison_rows,
    build_notebook_change_lines,
    build_notebook_change_rows,
    build_notebook_evidence_rows,
    build_notebook_metric_definition_rows,
    build_notebook_plot_card_rows,
    build_notebook_plot_inventory_rows,
    build_notebook_plot_method_rows,
    build_notebook_plot_method_sections,
    build_notebook_plot_scope_options,
    build_notebook_run_summary_lines,
    build_notebook_validity_lines,
    build_notebook_visual_surface_model,
    load_notebook_baserender_record_row,
    render_notebook_baserender_record,
    render_notebook_campaign_set_metric_comparison_image,
    render_visual_surface_cells,
    select_notebook_plot_scope,
)
from dnadesign.opal.src.analysis.notebook_components.plot_text import plot_alt_text
from dnadesign.opal.src.analysis.notebook_set_template import render_campaign_set_notebook
from dnadesign.opal.src.analysis.notebook_template import render_campaign_notebook
from dnadesign.opal.src.registries.plots import describe_plot_kind, get_plot, list_plot_kinds


def test_notebook_template_data_source_options() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert 'label="Campaign"' in text
    assert 'label="Round"' in text
    assert "predictions (selected run)" not in text
    assert "labels (all rounds)" not in text


def test_notebook_template_uses_full_width() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert 'marimo.App(width="full")' in text


def test_notebook_template_removes_extra_tables() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert "mo.ui.dataframe(summary_df)" not in text
    assert "mo.ui.dataframe(labels_df)" not in text
    assert "mo.ui.data_explorer(filtered_df)" not in text


def test_notebook_template_has_visual_surface() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert 'label="Visual surface"' in text
    assert 'label="Compare by"' in text
    assert "build_campaign_set_notebook_view_model" in text
    assert "Select one operative visual surface" not in text
    assert '"Plot deliverables": plot_panel' not in text


def test_notebook_template_has_opal_schema_sentinel() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")

    assert '__opal_notebook_template_schema__ = "opal.generated_campaign_notebook.v2"' in text


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
    assert surface_text not in text
    assert "manifest-backed plot media are available" in text
    assert 'label="Visual surface"' in text
    assert '"label": plot_choice["title"]' not in text
    assert "max-height" in text
    assert "Plot:" not in text
    assert "build_notebook_plot_method_sections" in text
    assert "thumbnail_gallery" not in text
    assert "plot_scope_controls" not in text
    assert "selected_plot_choice" in text


def test_notebook_template_does_not_hide_generic_plots_for_sfxi_campaigns() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")

    assert "SFXI plots only" not in text
    assert "Non-SFXI plots only" not in text
    assert "plot_entries_filtered" not in text


def test_notebook_template_is_campaign_specific_accordion_surface() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")

    assert "build_notebook_campaign_header_lines" in text
    assert "`{cfg.campaign.slug}` uses" not in text
    assert "Campaign analysis command surface" not in text
    assert "mo.accordion(" in text
    for section in [
        "Campaigns at a glance",
        "Selected campaign",
        "Validity",
        "Changes",
        "Metric definitions",
        "Artifacts",
        "Warnings and stale artifacts",
    ]:
        assert section in text


def test_notebook_template_uses_public_opal_helpers() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")

    assert "from dnadesign.opal.notebooks.api.generated import" in text
    assert "from dnadesign.opal.notebooks.api import" not in text
    assert "dnadesign.opal.src" not in text
    assert "build_campaign_set_notebook_view_model" in text
    assert "build_campaign_set_round_options" in text
    assert "build_notebook_campaign_summary_row" in text
    assert "build_notebook_visual_surface_model" in text
    assert "build_notebook_campaign_set_metric_comparison_rows" in text


def test_notebook_template_degrades_without_runs() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")

    assert "No written manifest-backed plot media are available" in text
    assert "mo.stop(len(rounds) == 0" not in text


def test_notebook_template_can_pin_initial_run_scope() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="0", run_id="run-1")

    assert "round_default = '0'" in text
    assert "run_id='run-1'" in text
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


def test_campaign_set_metric_comparison_uses_campaign_metadata(tmp_path: Path) -> None:
    def _campaign(slug: str, group: str, values: list[float]) -> dict:
        workdir = tmp_path / slug
        plots_dir = workdir / "outputs" / "plots"
        plots_dir.mkdir(parents=True)
        tidy_path = plots_dir / "score_selected_over_rounds_rall.csv"
        tidy_path.write_text(
            "round,cohort,metric,summary,value\n"
            + "\n".join(
                f"{round_index},selected,pred__score_selected,median,{value}"
                for round_index, value in enumerate(values)
            )
            + "\n"
            + "\n".join(
                f"{round_index},selected,pred__score_selected,count,6" for round_index, _value in enumerate(values)
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "campaign": {
                "slug": slug,
                "workdir": str(workdir),
                "metadata": {
                    "probe_oracle_kind": group,
                    "probe_oracle_id": f"{group}_id",
                    "probe_split_id": "random_id",
                },
            },
            "plot_manifests": [
                {
                    "name": "score_selected_over_rounds",
                    "kind": "metric_over_rounds",
                    "status": "written",
                    "rounds": "all",
                    "tidy_csv": str(tidy_path),
                    "outputs": [{"role": "tidy_csv", "path": str(tidy_path), "exists": True}],
                }
            ],
        }

    campaigns = [
        _campaign("cipro_positive_random_id", "positive", [0.2, 0.5]),
        _campaign("cipro_null_random_id", "null", [0.1, 0.15]),
    ]

    options = build_notebook_campaign_set_group_options(campaigns)
    assert options == ["probe_oracle_kind"]
    rows = build_notebook_campaign_set_metric_comparison_rows(
        campaigns,
        plot_name="score_selected_over_rounds",
        group_key="probe_oracle_kind",
    )
    assert {row["group"] for row in rows} == {"positive", "null"}
    payload = render_notebook_campaign_set_metric_comparison_image(
        rows,
        title="Selected score over rounds",
        group_key="probe_oracle_kind",
    )
    assert payload is not None
    assert payload["image_bytes"].startswith(b"\x89PNG")
    assert "probe_oracle_kind" in payload["alt_text"]
    assert "Selected n=6" in payload["alt_text"]
    mixed_rows = [
        {**row, "cohort": "all_pool" if row["campaign"] == "cipro_null_random_id" else row["cohort"]} for row in rows
    ]
    with pytest.raises(ValueError, match="one metric/cohort pair"):
        render_notebook_campaign_set_metric_comparison_image(
            mixed_rows,
            title="Selected score over rounds",
            group_key="probe_oracle_kind",
        )


def test_notebook_template_omits_altair_import() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert "import altair as alt" not in text


def test_notebook_template_uses_schema_pruned_records_loading() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")

    assert "store.load()" not in text
    assert "store.load_columns(records_loaded_columns)" not in text
    assert "records_df = pl.from_pandas" not in text


def test_notebook_component_primitives_build_shared_evidence_models() -> None:
    view_model = {
        "campaign": {
            "slug": "campaign_a",
            "config_path": "campaign.yaml",
            "workdir": "workdir",
            "x_column": "x_vec",
            "label_source": "usr_sidecar",
            "model": "random_forest",
            "selection": "top_n",
            "objectives": ["sfxi_v1"],
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
    assert {
        "field": "description",
        "value": (
            "Campaign ID `campaign_a`. Campaign A scores the configured OPAL records table with `random_forest` "
            "and selects candidates by `top_n` against `sfxi_v1`. The active X contract is `x_vec`."
        ),
    } in glance_rows
    assert {"field": "description source", "value": "derived"} in glance_rows
    assert {"field": "X column", "value": "x_vec"} in glance_rows
    assert {"field": "stale artifacts", "value": 1} in glance_rows
    header_lines = build_notebook_campaign_header_lines(view_model)
    assert header_lines[0] == "# Campaign A"
    assert "Campaign ID `campaign_a`." in header_lines[2]
    assert "scores the configured OPAL records table with `random_forest`" in header_lines[2]
    assert "The active X contract is `x_vec`." in header_lines[2]
    assert build_notebook_campaign_header_lines(view_model, heading_level=2)[0] == "## Campaign A"
    named_header = build_notebook_campaign_header_lines(
        {
            "campaign": {
                "name": "Stress ethanol/ciprofloxacin cipro factor RF + SFXI + top_n [cipro_positive_random_id]",
                "slug": "opal_axis_probe_v0_cipro_positive_random_id",
                "model": "random_forest",
                "selection": "top_n",
                "objectives": ["sfxi_v1"],
            }
        }
    )
    assert named_header[0] == "# Stress ethanol/ciprofloxacin cipro factor RF + SFXI + top N"
    assert "Opal Axis Probe" not in named_header[2]

    visual_surface = build_notebook_visual_surface_model(view_model)
    assert visual_surface["missing_outputs"] == []
    assert visual_surface["stale_artifacts"] == view_model["stale_artifacts"]
    assert visual_surface["inventory_status_counts"] == {
        "generated_current": 1,
        "stale_unmanifested": 1,
    }
    assert visual_surface["choices"][0]["label"] == "Score"
    assert visual_surface["choices"][0]["path_label"] == "plots/score.png"
    assert visual_surface["choices"][0]["capability"]["objective_family"] == "generic"
    assert "Scope: round 3" in visual_surface["choices"][0]["alt_text"]
    assert "freshness fresh" in visual_surface["choices"][0]["alt_text"]

    scope_view_model = {
        **view_model,
        "plot_manifests": [
            {
                **view_model["plot_manifests"][0],
                "run_id": None,
                "rounds": "all",
                "tidy_csv": "plots/score_rall.csv",
                "outputs": [{"role": "media", "path": "plots/score_rall.png", "exists": True}],
            },
            {
                **view_model["plot_manifests"][0],
                "rounds": [3],
                "tidy_csv": "plots/score_r3.csv",
                "outputs": [{"role": "media", "path": "plots/score_r3.png", "exists": True}],
            },
        ],
    }
    scope_surface = build_notebook_visual_surface_model(scope_view_model)
    scope_choice = scope_surface["choices"][0]
    assert scope_choice["scope_count"] == 2
    scope_options = build_notebook_plot_scope_options(scope_choice)
    assert [option["label"] for option in scope_options] == ["all rounds", "round 3; run run-3"]
    assert select_notebook_plot_scope(scope_choice, "round 3; run run-3")["path_label"] == "plots/score_r3.png"

    inventory_rows = build_notebook_plot_inventory_rows(visual_surface)
    assert {
        "plot": "score",
        "kind": "metric_over_rounds",
        "status": "generated_current",
        "rounds": "round 3",
        "objective": "generic",
        "data": "predictions",
        "round behavior": "round_history",
        "labels": "none",
        "model artifact": False,
        "tidy": True,
        "path": "plots/score.png",
    } in inventory_rows
    assert any(row["plot"] == "old" and row["status"] == "stale_unmanifested" for row in inventory_rows)

    configured_surface = build_notebook_visual_surface_model(
        view_model,
        plot_entries=[
            {"name": "score", "kind": "metric_over_rounds"},
            {"name": "missing_plot", "kind": "scatter_score_vs_rank", "round_selector": "latest"},
        ],
    )
    assert configured_surface["missing_outputs"] == ["missing_plot"]
    configured_inventory = build_notebook_plot_inventory_rows(configured_surface)
    assert any(
        row["plot"] == "missing_plot"
        and row["status"] == "configured_missing_output"
        and row["round behavior"] == "single_or_round_history"
        for row in configured_inventory
    )
    with pytest.raises(KeyError, match="not_registered"):
        build_notebook_visual_surface_model(
            view_model,
            plot_entries=[{"name": "bad_plot", "kind": "not_registered"}],
        )

    card_rows = build_notebook_plot_card_rows(visual_surface["choices"][0])
    assert {"field": "media", "value": "plots/score.png"} in card_rows
    assert {"field": "freshness", "value": "fresh"} in card_rows
    assert any(row["field"] == "capability" and "objective_family=generic" in row["value"] for row in card_rows)
    assert {"field": "tidy data", "value": "plots/score.csv"} in card_rows
    assert any(row["field"] == "source data" for row in card_rows)
    per_round_card_rows = build_notebook_plot_card_rows(select_notebook_plot_scope(scope_choice, "round 3; run run-3"))
    assert {"field": "rounds", "value": "round 3"} in per_round_card_rows
    assert {"field": "warnings", "value": "0"} in per_round_card_rows
    method_rows = build_notebook_plot_method_rows(visual_surface["choices"][0])
    assert any(row["section"] == "math" and "mean = sum(x) / n" in row["detail"] for row in method_rows)
    method_sections = build_notebook_plot_method_sections(visual_surface["choices"][0])
    assert "Read" in method_sections
    assert "mean = sum(x) / n" in method_sections["Math"]
    assert "Freshness: `fresh`" in method_sections["Data contract"]

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
    assert "Event phases: `command=1, preflight=2, run=6, finalize=1`" in change_lines
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


def test_registered_plot_kinds_have_explicit_math_disclosure() -> None:
    fallback = "See the plot kind metadata"

    builtin_kinds = [
        kind for kind in list_plot_kinds() if get_plot(kind).__module__.startswith("dnadesign.opal.src.plots.")
    ]
    assert builtin_kinds

    for kind in builtin_kinds:
        meta = describe_plot_kind(kind)
        choice = {
            "kind": kind,
            "name": kind,
            "rounds": [2],
            "freshness": "fresh",
            "manifest": {
                "kind": kind,
                "rounds": [2],
                "run_id": "run-2",
                "generated_at": "2026-05-21T00:00:00Z",
                "manifest_path": "outputs/plots/example.manifest.json",
                "metadata": meta,
                "inputs": [{"role": "input", "path": "outputs/ledger/predictions.parquet"}],
                "params": {"metric": "pred__score_selected", "sample_n": 50, "min_n": 5, "top_k": 10},
            },
        }
        method_rows = build_notebook_plot_method_rows(choice)
        math_rows = [row for row in method_rows if row["section"] == "math"]
        method_sections = build_notebook_plot_method_sections(choice)

        assert math_rows, kind
        assert math_rows[0]["detail"], kind
        assert fallback not in math_rows[0]["detail"], kind
        assert fallback not in method_sections["Math"], kind
        assert "Input data layer:" in method_sections["Data contract"], kind
        assert "Provenance:" in method_sections["Data contract"], kind
        assert "Counts and replicates:" in method_sections["Data contract"], kind
        assert "manifest=outputs/plots/example.manifest.json" in method_sections["Data contract"], kind


def test_registered_plot_alt_text_exposes_primary_visual_encoding() -> None:
    builtin_kinds = [
        kind for kind in list_plot_kinds() if get_plot(kind).__module__.startswith("dnadesign.opal.src.plots.")
    ]
    assert builtin_kinds

    for kind in builtin_kinds:
        meta = describe_plot_kind(kind)
        alt_text = plot_alt_text(
            title=kind,
            kind=kind,
            summary=meta["summary"],
            params={
                "metric": "pred__score_selected",
                "score_field": "pred__score_selected",
                "y_axis": "score",
                "hue": "logic_fidelity",
                "size_by": "obj__effect_scaled",
                "vector_field": "pred__y_hat_model",
            },
            metadata=meta,
            rounds=[3],
            run_id=None,
            freshness="fresh",
            warning_count=0,
        )

        assert "Encoded fields:" in alt_text, kind
        assert any(token in alt_text for token in ("x=", "left panel x=", "panels=")), kind
        assert "Scope: round 3" in alt_text, kind


def test_notebook_template_is_valid_python() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    ast.parse(text)


def test_campaign_set_notebook_has_campaign_and_plot_dropdowns() -> None:
    text = render_campaign_set_notebook(
        [Path("campaign_a.yaml"), Path("campaign_b.yaml")],
        round_selector="latest",
    )

    assert "# Campaigns" in text
    assert "from dnadesign.opal.notebooks.api.generated import (" in text
    assert "from dnadesign.opal.notebooks.api import (" not in text
    assert "build_campaign_set_notebook_view_model" in text
    assert "build_campaign_set_round_options" in text
    assert "build_notebook_campaign_header_lines" in text
    assert "dnadesign.opal.src" not in text
    assert "__generated_with" in text
    assert 'generated_with = "' in text
    assert "Generated with marimo: `{__generated_with}`" not in text
    assert 'label="Round"' in text
    assert "selected_round_selector = str(round_ui.value)" in text
    assert 'label="Campaign"' in text
    assert "campaign_labels = [f\"{index + 1}. {row['label']}\"" in text
    assert "selected_index = campaign_labels.index(selected_label)" in text
    assert 'label="Visual surface"' in text
    assert "Plot:" not in text
    assert "Validity" in text
    assert "Changes" in text
    assert "build_notebook_artifact_garden_rows" in text
    assert "build_notebook_change_rows" in text
    assert "build_notebook_metric_definition_rows" in text
    assert "build_notebook_visual_surface_model" in text
    assert "build_notebook_plot_card_rows" in text
    assert "build_notebook_plot_method_sections" in text
    assert "build_notebook_validity_rows" in text
    assert "build_notebook_validity_lines" not in text
    assert "build_notebook_artifact_garden_lines" not in text
    assert "build_notebook_change_lines" not in text
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

    assert '_scope_control_label = str(plot_scope_options[0].get("control_label") or "Plot scope")' in single
    assert "label=_scope_control_label" in single
    assert '_scope_control_label = str(plot_scope_options[0].get("control_label") or "Plot scope")' in campaign_set
    assert "label=_scope_control_label" in campaign_set
    assert len(single.splitlines()) <= 1050
    assert len(campaign_set.splitlines()) <= 430


def test_notebook_baserender_contract_detects_schema_without_generated_import() -> None:
    text = render_campaign_notebook(Path("campaign.yaml"), round_selector="latest")
    assert "render_notebook_baserender_record" not in text
    assert "from dnadesign.baserender import" not in text

    unavailable = build_notebook_baserender_contract(["id", "sequence"], records_path="records.parquet")
    assert unavailable["available"] is False
    unavailable_rows = {
        str(row["field"]): str(row["value"]) for row in build_notebook_baserender_contract_rows(unavailable)
    }
    assert unavailable_rows["available"] == "false"
    assert unavailable_rows["contract"] == "dnadesign.baserender.record_render_contract.v1"

    contract = build_notebook_baserender_contract(
        ["id", "sequence", "densegen__used_tfbs_detail"],
        records_path="records.parquet",
    )
    assert contract["available"] is True
    assert contract["adapter_kind"] == "densegen_tfbs"
    assert contract["adapter_columns"]["annotations"] == "densegen__used_tfbs_detail"
    assert callable(render_notebook_baserender_record)
    assert (
        "densegen__used_tfbs_detail"
        not in Path("src/dnadesign/opal/src/analysis/notebook_components/baserender.py").read_text()
    )

    generic = build_notebook_baserender_contract(
        ["id", "sequence", "opal__baserender_features", "densegen__used_tfbs_detail"],
        records_path="records.parquet",
    )
    assert generic["adapter_kind"] == "generic_features"


def test_notebook_baserender_options_fail_fast_for_bad_available_contract(tmp_path: Path) -> None:
    contract = build_notebook_baserender_contract(
        ["id", "sequence", "opal__baserender_features"],
        records_path=str(tmp_path / "missing.parquet"),
    )

    with pytest.raises(Exception, match="missing.parquet|No such file|not found"):
        build_notebook_baserender_record_options(tmp_path / "missing.parquet", contract)


def test_notebook_baserender_record_options_skip_empty_feature_rows(tmp_path: Path) -> None:
    feature_type = pa.list_(
        pa.struct(
            [
                ("regulator", pa.string()),
                ("sequence", pa.string()),
                ("orientation", pa.string()),
                ("offset", pa.int64()),
            ]
        )
    )
    records_path = tmp_path / "records.parquet"
    table = pa.table(
        {
            "id": pa.array(["empty", "good"]),
            "sequence": pa.array(["TTGACATATAAT", "TTGACATATAAT"]),
            "densegen__used_tfbs_detail": pa.array(
                [
                    [],
                    [{"regulator": "lexA", "sequence": "TTGACA", "orientation": "fwd", "offset": 0}],
                ],
                type=feature_type,
            ),
        }
    )
    pq.write_table(table, records_path)
    contract = build_notebook_baserender_contract(table.column_names, records_path=str(records_path))

    assert build_notebook_baserender_record_options(records_path, contract) == ["good"]
    assert load_notebook_baserender_record_row(records_path, "empty", contract) is None
    assert load_notebook_baserender_record_row(records_path, "good", contract)["id"] == "good"
