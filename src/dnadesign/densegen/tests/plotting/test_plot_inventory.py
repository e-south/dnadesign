"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/plotting/test_plot_inventory.py

Coverage for shared plot inventory helpers used by plotting and notebook.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from dnadesign.densegen.src.viz.plot_inventory import (
    build_plot_ids_by_scope,
    build_plot_text_contract,
    compact_plan_label,
    describe_visual_plot_type,
    load_current_inventory_strict,
    missing_notebook_visible_plot_ids,
    notebook_visible_plot_ids,
    plot_missing_hint,
    plot_required_artifacts,
    resolve_plot_availability,
    resolve_plot_record,
    stage_b_scope_seed_plot_ids,
)


def test_build_plot_ids_by_scope_adds_stage_b_recoverable_types_for_known_plans() -> None:
    plot_ids_by_scope, generated_plot_ids_by_scope = build_plot_ids_by_scope(
        [],
        stage_b_scope_names=["demo_plan"],
    )

    assert plot_ids_by_scope["all"] == []
    assert plot_ids_by_scope["demo_plan"] == ["placement_occupancy_map", "tfbs_concentration_profile"]
    assert generated_plot_ids_by_scope["demo_plan"] == []


def test_build_plot_ids_by_scope_seeds_all_scope_with_known_plot_ids() -> None:
    plot_ids_by_scope, generated_plot_ids_by_scope = build_plot_ids_by_scope(
        [],
        stage_b_scope_names=["demo_plan"],
        known_plot_ids=[
            "dense_array_showcase_video",
            "placement_occupancy_map",
            "tfbs_concentration_profile",
            "solve_pressure_and_progress",
            "stage_a_sampling_yield",
        ],
    )

    assert plot_ids_by_scope["all"] == [
        "dense_array_showcase_video",
        "placement_occupancy_map",
        "tfbs_concentration_profile",
        "solve_pressure_and_progress",
        "stage_a_sampling_yield",
    ]
    assert generated_plot_ids_by_scope["all"] == []
    assert plot_ids_by_scope["demo_plan"] == ["placement_occupancy_map", "tfbs_concentration_profile"]


def test_resolve_plot_availability_uses_explicit_status_contract() -> None:
    generated_ids = ["placement_occupancy_map"]

    assert resolve_plot_availability("placement_occupancy_map", generated_plot_ids=generated_ids) == "generated"
    assert resolve_plot_availability("dense_array_showcase_video", generated_plot_ids=[]) == "recoverable_read_only"
    assert resolve_plot_availability("tfbs_concentration_profile", generated_plot_ids=[]) == "recoverable_read_only"
    assert resolve_plot_availability("solve_pressure_and_progress", generated_plot_ids=[]) == "requires_local_artifacts"


def test_plot_inventory_exposes_registry_backed_required_artifact_contract() -> None:
    assert "outputs/tables/attempts.parquet or attempts_part-*.parquet" in plot_required_artifacts(
        "solve_pressure_and_progress"
    )
    assert "plots.video configuration" in plot_required_artifacts("dense_array_showcase_video")
    assert stage_b_scope_seed_plot_ids() == ["placement_occupancy_map", "tfbs_concentration_profile"]
    assert "workspace-local Stage-A pool artifacts" in plot_missing_hint("stage_a_sampling_yield")


def test_resolve_plot_record_infers_stage_b_scope_and_variant_from_path() -> None:
    plot_root = Path("/tmp/run/outputs/plots")
    plot_path = plot_root / "stage_b" / "demo_plan" / "demo_input" / "placement_occupancy_map.png"

    record = resolve_plot_record(
        plot_root=plot_root,
        plot_path=plot_path,
    )

    assert record["plot_id"] == "placement_occupancy_map"
    assert record["plan_name"] == "demo_plan"
    assert record["input_name"] == "demo_input"
    assert record["variant"] == "placement_occupancy_map"
    assert record["visual_plot_type"] == "placement_occupancy_map"
    assert record["title"] == "Placement occupancy map"
    assert "Stage-B positional occupancy map for accepted arrays" in str(record["caption"])
    assert "Placement occupancy map." in str(record["alt_text"])


def test_resolve_plot_record_infers_showcase_video_without_manifest() -> None:
    plot_root = Path("/tmp/run/outputs/plots")
    plot_path = plot_root / "stage_b" / "all_plans" / "showcase.mp4"

    record = resolve_plot_record(
        plot_root=plot_root,
        plot_path=plot_path,
    )

    assert record["plot_id"] == "dense_array_showcase_video"
    assert record["plan_name"] == "all_plans"
    assert record["variant"] == "showcase"
    assert record["visual_plot_type"] == "dense_array_showcase_video"
    assert record["title"] == "Dense array showcase video"
    assert "sampled accepted outputs" in str(record["caption"])
    assert "drawn across all plans" in str(record["caption"])


def test_resolve_plot_record_exposes_stage_a_yield_plot_for_gallery_filters() -> None:
    plot_root = Path("/tmp/run/outputs/plots")
    plot_path = plot_root / "stage_a" / "stage_a_sampling_yield.png"

    record = resolve_plot_record(
        plot_root=plot_root,
        plot_path=plot_path,
    )

    assert record["plot_id"] == "stage_a_sampling_yield"
    assert record["plan_name"] == "stage_a"
    assert record["variant"] == "stage_a_sampling_yield"
    assert record["visual_plot_type"] == "stage_a_sampling_yield"
    assert record["title"] == "Stage A sampling yield"
    assert "retained yield" in str(record["caption"])


def test_resolve_plot_record_infers_stage_b_summary_plot_from_path() -> None:
    plot_root = Path("/tmp/run/outputs/plots")
    plot_path = plot_root / "stage_b_summary" / "retained_pool_coverage_by_regulator.png"

    record = resolve_plot_record(
        plot_root=plot_root,
        plot_path=plot_path,
    )

    assert record["plot_id"] == "retained_pool_coverage_by_regulator"
    assert record["plan_name"] == "unscoped"
    assert record["variant"] == "retained_pool_coverage_by_regulator"
    assert record["visual_plot_type"] == "retained_pool_coverage_by_regulator"
    assert record["title"] == "Retained pool coverage by regulator"


def test_resolve_plot_record_does_not_treat_stage_b_filename_as_input_scope() -> None:
    plot_root = Path("/tmp/run/outputs/plots")
    plot_path = plot_root / "stage_b" / "ethanol" / "placement_occupancy_map.png"

    record = resolve_plot_record(
        plot_root=plot_root,
        plot_path=plot_path,
    )

    assert record["plot_id"] == "placement_occupancy_map"
    assert record["plan_name"] == "ethanol"
    assert record["input_name"] == ""


def test_resolve_plot_record_keeps_source_cohort_concentration_as_single_gallery_type() -> None:
    plot_root = Path("/tmp/run/outputs/plots")
    plot_path = plot_root / "dataset" / "source_cohort_concentration.png"

    record = resolve_plot_record(
        plot_root=plot_root,
        plot_path=plot_path,
    )

    assert record["plot_id"] == "source_cohort_concentration"
    assert record["variant"] == "source_cohort_concentration"
    assert record["visual_plot_type"] == "source_cohort_concentration"
    assert record["title"] == "Source cohort concentration"


def test_build_plot_ids_by_scope_keeps_optional_surface_visible_when_generated() -> None:
    plot_ids_by_scope, generated_plot_ids_by_scope = build_plot_ids_by_scope(
        [
            {
                "visual_plot_type": "source_cohort_concentration",
                "plan_name": "unscoped",
            },
            {
                "visual_plot_type": "source_plan_input_heatmap",
                "plan_name": "unscoped",
            },
        ],
        known_plot_ids=["source_cohort_concentration", "source_plan_input_heatmap", "solve_pressure_and_progress"],
    )

    assert plot_ids_by_scope["all"] == [
        "source_cohort_concentration",
        "source_plan_input_heatmap",
        "solve_pressure_and_progress",
    ]
    assert generated_plot_ids_by_scope["all"] == ["source_cohort_concentration", "source_plan_input_heatmap"]


def test_build_plot_ids_by_scope_collapses_stage_b_scope_outputs_into_single_gallery_type() -> None:
    plot_ids_by_scope, generated_plot_ids_by_scope = build_plot_ids_by_scope(
        [
            {
                "visual_plot_type": "placement_occupancy_map",
                "plan_name": "background_only",
            }
        ],
        stage_b_scope_names=["background_only"],
        known_plot_ids=["placement_occupancy_map", "tfbs_concentration_profile"],
    )

    assert plot_ids_by_scope["all"] == ["placement_occupancy_map", "tfbs_concentration_profile"]
    assert generated_plot_ids_by_scope["all"] == ["placement_occupancy_map"]
    assert plot_ids_by_scope["background_only"] == ["placement_occupancy_map", "tfbs_concentration_profile"]


def test_notebook_visible_plot_ids_hide_internal_gallery_types() -> None:
    visible_ids = notebook_visible_plot_ids()

    assert "source_plan_input_heatmap" in visible_ids
    assert "retained_pool_coverage_by_regulator" in visible_ids
    assert "solve_pressure_and_progress" in visible_ids
    assert "accepted_arrays_by_plan" not in visible_ids
    assert "stage_a_summary" not in visible_ids
    assert "run_health" not in visible_ids


def test_missing_notebook_visible_plot_ids_reports_missing_base_plot_ids() -> None:
    missing_ids = missing_notebook_visible_plot_ids(
        [
            {"plot_id": "stage_a_sampling_yield", "visual_plot_type": "stage_a_sampling_yield"},
            {"plot_id": "placement_occupancy_map", "visual_plot_type": "placement_occupancy_map"},
            {"plot_id": "attempt_outcome_timeline", "visual_plot_type": "attempt_outcome_timeline"},
        ]
    )

    assert "stage_a_sampling_yield" not in missing_ids
    assert "placement_occupancy_map" not in missing_ids
    assert "attempt_outcome_timeline" not in missing_ids
    assert "stage_a_pool_diversity" in missing_ids


def test_visual_plot_type_and_scope_labels_are_human_readable() -> None:
    assert describe_visual_plot_type("attempt_outcome_timeline") == "Attempt outcome timeline"
    assert describe_visual_plot_type("stage_a_sampling_yield") == "Stage A sampling yield"
    assert compact_plan_label("ethanol_ciprofloxacin") == "EtOH + Cipro"
    assert compact_plan_label("all_plans") == "All plans"


def test_plot_text_contract_defines_upstream_counts_and_pwm_proxy_in_plain_language() -> None:
    contract = build_plot_text_contract("upstream_motif_supply_and_pwm_strength")

    assert "source hits" in contract["caption"].lower()
    assert "eligible unique" in contract["caption"].lower()
    assert "retained stage-a motifs" in contract["caption"].lower()
    assert "theoretical maximum" in contract["alt_text"].lower()
    assert "not averaged over deployed densegen arrays" in contract["alt_text"].lower()


def test_plot_text_contract_defines_retained_and_deployed_for_tier_mix() -> None:
    contract = build_plot_text_contract("retained_vs_deployed_tier_mix_by_regulator")

    assert "retained means the stage-a tfbs pool kept after sampling and selection" in contract["alt_text"].lower()
    assert (
        "deployed means tfbs annotations that actually appear in accepted densegen outputs"
        in contract["alt_text"].lower()
    )
    assert "tier 0" in contract["alt_text"].lower()


def test_plot_text_contract_defines_stage_a_to_stage_b_bridge_in_plain_language() -> None:
    contract = build_plot_text_contract("score_strata_and_deployed_length_bridge")

    assert "eligible-unique stage-a pwm matches" in contract["alt_text"].lower()
    assert "minimum retained score" in contract["alt_text"].lower()
    assert "how many unique tfbs sequences" in contract["alt_text"].lower()
    assert "core average pairwise hamming distance" in contract["alt_text"].lower()
    assert "unique deployed tfbs sequences" in contract["alt_text"].lower()
    assert "from longest to shortest" in contract["alt_text"].lower()


def test_load_current_inventory_strict_requires_all_grouped_core_stage_b_scopes(tmp_path: Path) -> None:
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
              default: [placement_occupancy_map]
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
                    {
                        "plot_id": "placement_occupancy_map",
                        "path": "stage_b/background_only/placement_occupancy_map.pdf",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"placement_occupancy_map\[ethanol\]"):
        load_current_inventory_strict(
            plot_root,
            required_plot_ids=["placement_occupancy_map"],
            config_path=cfg_path,
        )
