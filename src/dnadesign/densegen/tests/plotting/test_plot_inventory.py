"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/plotting/test_plot_inventory.py

Coverage for shared plot inventory helpers used by plotting and notebook
gallery discovery.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.densegen.src.viz.plot_inventory import (
    build_plot_ids_by_scope,
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
    assert plot_ids_by_scope["demo_plan"] == ["placement_map", "tfbs_usage"]
    assert generated_plot_ids_by_scope["demo_plan"] == []


def test_build_plot_ids_by_scope_seeds_all_scope_with_known_plot_ids() -> None:
    plot_ids_by_scope, generated_plot_ids_by_scope = build_plot_ids_by_scope(
        [],
        stage_b_scope_names=["demo_plan"],
        known_plot_ids=["dense_array_video_showcase", "placement_map", "tfbs_usage", "run_health", "stage_a_summary"],
    )

    assert plot_ids_by_scope["all"] == [
        "dense_array_video_showcase",
        "placement_map",
        "tfbs_usage",
        "run_health",
        "stage_a_summary",
    ]
    assert generated_plot_ids_by_scope["all"] == []
    assert plot_ids_by_scope["demo_plan"] == ["placement_map", "tfbs_usage"]


def test_resolve_plot_availability_uses_explicit_status_contract() -> None:
    generated_ids = ["placement_map/occupancy"]

    assert resolve_plot_availability("placement_map/occupancy", generated_plot_ids=generated_ids) == "generated"
    assert resolve_plot_availability("dense_array_video_showcase", generated_plot_ids=[]) == "recoverable_read_only"
    assert resolve_plot_availability("tfbs_usage", generated_plot_ids=[]) == "recoverable_read_only"
    assert resolve_plot_availability("run_health", generated_plot_ids=[]) == "requires_local_artifacts"


def test_plot_inventory_exposes_registry_backed_required_artifact_contract() -> None:
    assert "outputs/tables/attempts.parquet or attempts_part-*.parquet" in plot_required_artifacts("run_health")
    assert "plots.video configuration" in plot_required_artifacts("dense_array_video_showcase")
    assert stage_b_scope_seed_plot_ids() == ["placement_map", "tfbs_usage"]
    assert "workspace-local Stage-A pool artifacts" in plot_missing_hint("stage_a_summary")


def test_resolve_plot_record_infers_stage_b_scope_and_variant_from_path() -> None:
    plot_root = Path("/tmp/run/outputs/plots")
    plot_path = plot_root / "stage_b" / "demo_plan" / "demo_input" / "placement_map.png"

    record = resolve_plot_record(
        plot_root=plot_root,
        plot_path=plot_path,
    )

    assert record["plot_id"] == "placement_map"
    assert record["plan_name"] == "demo_plan"
    assert record["input_name"] == "demo_input"
    assert record["variant"] == "placement_map"
    assert record["visual_plot_type"] == "placement_map"


def test_resolve_plot_record_infers_showcase_video_without_manifest() -> None:
    plot_root = Path("/tmp/run/outputs/plots")
    plot_path = plot_root / "stage_b" / "all_plans" / "showcase.mp4"

    record = resolve_plot_record(
        plot_root=plot_root,
        plot_path=plot_path,
    )

    assert record["plot_id"] == "dense_array_video_showcase"
    assert record["plan_name"] == "all_plans"
    assert record["variant"] == "showcase"
    assert record["visual_plot_type"] == "dense_array_video_showcase"
