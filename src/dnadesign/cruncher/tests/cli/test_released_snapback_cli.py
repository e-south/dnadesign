"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_released_snapback_cli.py

CLI contract tests for released-product snapback commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from dnadesign.cruncher.cli.app import app
from dnadesign.cruncher.tests.cli_output import normalized_cli_output

runner = CliRunner()


def _write_workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    workspace = tmp_path / "workspaces" / "demo_released"
    spec_path = workspace / "configs" / "snapback" / "demo.released.snapback.yaml"
    nick_catalog_path = workspace / "inputs" / "nickases" / "local.nickases.yaml"
    release_catalog_path = workspace / "inputs" / "release_enzymes" / "local.release.yaml"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    nick_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    release_catalog_path.parent.mkdir(parents=True, exist_ok=True)
    nick_catalog_path.write_text(
        yaml.safe_dump(
            {
                "nickases": {
                    "schema_version": 1,
                    "entries": [
                        {
                            "id": "Nx.Exact7",
                            "specificity_id": "Nx.Exact7",
                            "motif_top_5to3": "AACGTTG",
                            "top_cut_offset": 0,
                        },
                        {
                            "id": "Nx.Near7",
                            "specificity_id": "Nx.Near7",
                            "motif_top_5to3": "TAACGTT",
                            "top_cut_offset": 1,
                        },
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    release_catalog_path.write_text(
        yaml.safe_dump(
            {
                "release_enzymes": {
                    "schema_version": 1,
                    "entries": [
                        {
                            "variant_id": "Re.Exact",
                            "display_name": "Re.Exact",
                            "recognition_sequence": "CCAA",
                            "top_cut_offset": 1,
                            "bottom_cut_offset": 0,
                            "class_label": "other_ds_re",
                            "commercial_confidence": "primary_vendor_current",
                            "source_catalog_id": "local_release",
                        }
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    spec_path.write_text(
        yaml.safe_dump(
            {
                "released_snapback": {
                    "schema_version": 1,
                    "kind": "single_nick_released_snapback_v1",
                    "name": "demo_released",
                },
                "input": {"precursor_top_strand": "AACGTTGTTCCAA"},
                "nick_stage": {
                    "nickase_variant_id": "Nx.Exact7",
                    "catalog": {"additional_paths": ["inputs/nickases/local.nickases.yaml"]},
                },
                "release_stage": {
                    "release_variant_id": "Re.Exact",
                    "catalog": {"additional_paths": ["inputs/release_enzymes/local.release.yaml"]},
                    "retained_side": "upstream",
                    "stage_order": "nick_then_release",
                },
                "final_target": {"nick_boundary_from_left": 0, "paired_bp": 3, "cap_nt": 3},
                "constraints": {
                    "allow_post_release_loss_of_nickase_site": True,
                    "allow_post_release_loss_of_release_site": True,
                    "require_release_site_downstream_of_nick": True,
                    "require_complete_downstream_fragment_separation": True,
                },
                "output": {"run_dir": "outputs/released_design"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return workspace, spec_path, release_catalog_path


def test_snapback_help_includes_released_product_commands() -> None:
    result = runner.invoke(app, ["snapback", "--help"], color=False)

    assert result.exit_code == 0
    assert "screen" in result.output
    assert "released-design" in result.output
    assert "released-target-search" in result.output
    assert "released-solve" in result.output
    assert "released-show" in result.output


def test_released_design_and_show_round_trip(tmp_path: Path) -> None:
    _workspace, spec_path, _release_catalog_path = _write_workspace(tmp_path)

    design_result = runner.invoke(
        app,
        ["snapback", "released-design", "--spec", str(spec_path)],
        color=False,
    )

    assert design_result.exit_code == 0
    run_dir = spec_path.parent.parent.parent / "outputs" / "released_design"
    show_result = runner.invoke(
        app,
        ["snapback", "released-show", "--run", str(run_dir), "--json"],
        color=False,
    )

    assert show_result.exit_code == 0
    payload = json.loads(show_result.output)
    assert payload["status"] == "satisfied"
    assert payload["kind"] == "released_explicit"


def test_released_target_search_json_reports_exact_and_near_hits(tmp_path: Path) -> None:
    workspace, _spec_path, _release_catalog_path = _write_workspace(tmp_path)

    result = runner.invoke(
        app,
        [
            "snapback",
            "released-target-search",
            "--workspace-root",
            str(workspace),
            "--nick-additional-path",
            "inputs/nickases/local.nickases.yaml",
            "--release-additional-path",
            "inputs/release_enzymes/local.release.yaml",
            "--max-results",
            "2",
            "--json",
        ],
        color=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["status"] == "exact_hits_found"
    assert payload["exact_hits"][0]["nick_boundary_from_left"] == 0
    assert payload["exact_hits"][0]["release_variant_id"] == "Re.Exact"
    assert payload["near_hits"]
    assert all(hit["nick_boundary_from_left"] > 0 for hit in payload["near_hits"])


def test_released_target_search_json_reports_route_policy_when_top_active_routes_are_enabled(tmp_path: Path) -> None:
    workspace, _spec_path, _release_catalog_path = _write_workspace(tmp_path)

    result = runner.invoke(
        app,
        [
            "snapback",
            "released-target-search",
            "--workspace-root",
            str(workspace),
            "--nick-additional-path",
            "inputs/nickases/local.nickases.yaml",
            "--release-additional-path",
            "inputs/release_enzymes/local.release.yaml",
            "--allow-top-active-routes",
            "--allow-precut-footprint-outside-active-product",
            "--json",
        ],
        color=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "final_geometry_source" not in payload["metadata"]
    assert payload["metadata"]["route_policy_final_geometry_source"] == "retained_active_strand"
    assert payload["metadata"]["allowed_active_strands"] == ["top", "bottom"]
    assert payload["metadata"]["allowed_route_families"] == [
        "bottom_active_from_top_nick",
        "top_active_from_bottom_nick",
    ]


def test_snapback_screen_json_reports_study_semantics(tmp_path: Path) -> None:
    workspace = tmp_path / "workspaces" / "de033"
    workspace.mkdir(parents=True, exist_ok=True)

    result = runner.invoke(
        app,
        [
            "snapback",
            "screen",
            "--workspace-root",
            str(workspace),
            "--max-results",
            "16",
            "--json",
        ],
        color=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["kind"] == "snapback_screen_report_v1"
    assert payload["status"] == "exact_hits_found"
    assert payload["target_topology"]["logical_origin"] == 0
    assert payload["target_topology"]["stem_bp"] == 3
    assert payload["target_topology"]["cap_nt"] == 3
    assert payload["target_topology"]["retained_product_strands"] == ["top", "bottom"]
    assert payload["search_report"]["metadata"]["allowed_release_variant_ids"] == ["BspQI"]
    ledger_ids = {entry["nickase_variant_id"] for entry in payload["mechanism_ledger"]}
    assert ledger_ids == {"Nt.BstNBI", "Nt.AlwI", "Nt.BsmAI", "Nb.BsrDI", "Nb.BtsI"}
    assert {entry["release_variant_id"] for entry in payload["mechanism_ledger"]} == {"BspQI"}


def test_released_target_search_requires_explicit_sources(tmp_path: Path) -> None:
    workspace, _spec_path, _release_catalog_path = _write_workspace(tmp_path)

    result = runner.invoke(
        app,
        [
            "snapback",
            "released-target-search",
            "--workspace-root",
            str(workspace),
            "--json",
        ],
        color=False,
    )

    assert result.exit_code == 1
    assert "requires at least one explicit nickase source" in result.output


def test_released_target_search_help_mentions_active_route_flags() -> None:
    result = runner.invoke(app, ["snapback", "released-target-search", "--help"], color=False)

    assert result.exit_code == 0
    output = normalized_cli_output(result.output)
    assert "--allow-top-active-routes" in output
    assert "--allow-precut-footprint-out" in output
    assert "retained-active audits" in output
    assert "vendor nickase" in output
    assert "top strand" in output


def test_released_target_search_excludes_demo_only_entries_unless_opted_in(tmp_path: Path) -> None:
    workspace, _spec_path, _release_catalog_path = _write_workspace(tmp_path)
    nick_catalog_path = workspace / "inputs" / "nickases" / "local.nickases.yaml"
    release_catalog_path = workspace / "inputs" / "release_enzymes" / "local.release.yaml"

    nick_payload = yaml.safe_load(nick_catalog_path.read_text(encoding="utf-8"))
    nick_entries = nick_payload["nickases"]["entries"]
    nick_entries[0]["metadata"] = {"demo_only": True}
    nick_catalog_path.write_text(yaml.safe_dump(nick_payload, sort_keys=False), encoding="utf-8")

    release_payload = yaml.safe_load(release_catalog_path.read_text(encoding="utf-8"))
    release_payload["release_enzymes"]["entries"][0]["metadata"] = {"demo_only": True}
    release_catalog_path.write_text(yaml.safe_dump(release_payload, sort_keys=False), encoding="utf-8")

    blocked = runner.invoke(
        app,
        [
            "snapback",
            "released-target-search",
            "--workspace-root",
            str(workspace),
            "--nick-additional-path",
            "inputs/nickases/local.nickases.yaml",
            "--release-additional-path",
            "inputs/release_enzymes/local.release.yaml",
            "--json",
        ],
        color=False,
    )

    assert blocked.exit_code == 1
    blocked_payload = json.loads(blocked.output)
    assert blocked_payload["status"] == "no_hits"
    assert blocked_payload["metadata"]["evaluated_pair_count"] == 0
    assert blocked_payload["metadata"]["blocker_counts"]["DEMO_ONLY_PAIR_SUPPRESSED"] >= 1

    allowed = runner.invoke(
        app,
        [
            "snapback",
            "released-target-search",
            "--workspace-root",
            str(workspace),
            "--nick-additional-path",
            "inputs/nickases/local.nickases.yaml",
            "--release-additional-path",
            "inputs/release_enzymes/local.release.yaml",
            "--allow-demo-hits",
            "--json",
        ],
        color=False,
    )

    assert allowed.exit_code == 0
    allowed_payload = json.loads(allowed.output)
    assert allowed_payload["status"] == "exact_hits_found"
    assert allowed_payload["exact_hits"][0]["release_variant_id"] == "Re.Exact"


def test_released_target_search_excludes_frequent_cutter_nickases_unless_opted_in(tmp_path: Path) -> None:
    workspace, _spec_path, _release_catalog_path = _write_workspace(tmp_path)
    nick_catalog_path = workspace / "inputs" / "nickases" / "local.nickases.yaml"

    nick_payload = yaml.safe_load(nick_catalog_path.read_text(encoding="utf-8"))
    nick_payload["nickases"]["entries"][0]["selection"] = {"warning_codes": ["FREQUENT_CUTTER"]}
    nick_catalog_path.write_text(yaml.safe_dump(nick_payload, sort_keys=False), encoding="utf-8")

    blocked = runner.invoke(
        app,
        [
            "snapback",
            "released-target-search",
            "--workspace-root",
            str(workspace),
            "--nick-additional-path",
            "inputs/nickases/local.nickases.yaml",
            "--release-additional-path",
            "inputs/release_enzymes/local.release.yaml",
            "--json",
        ],
        color=False,
    )

    assert blocked.exit_code == 0
    blocked_payload = json.loads(blocked.output)
    assert blocked_payload["status"] == "near_hits_only"
    assert blocked_payload["metadata"]["blocker_counts"]["DISALLOWED_NICKASE_WARNING_CODE"] >= 1
    assert all(hit["nickase_variant_id"] != "Nx.Exact7" for hit in blocked_payload["near_hits"])

    allowed = runner.invoke(
        app,
        [
            "snapback",
            "released-target-search",
            "--workspace-root",
            str(workspace),
            "--nick-additional-path",
            "inputs/nickases/local.nickases.yaml",
            "--release-additional-path",
            "inputs/release_enzymes/local.release.yaml",
            "--allow-frequent-cutter-nickases",
            "--json",
        ],
        color=False,
    )

    assert allowed.exit_code == 0
    allowed_payload = json.loads(allowed.output)
    assert allowed_payload["status"] == "exact_hits_found"
    assert allowed_payload["exact_hits"][0]["nickase_variant_id"] == "Nx.Exact7"


def test_released_solve_json_reports_route_policy_when_top_active_routes_are_enabled(tmp_path: Path) -> None:
    workspace, _spec_path, _release_catalog_path = _write_workspace(tmp_path)

    result = runner.invoke(
        app,
        [
            "snapback",
            "released-solve",
            "--workspace-root",
            str(workspace),
            "--nick-additional-path",
            "inputs/nickases/local.nickases.yaml",
            "--release-additional-path",
            "inputs/release_enzymes/local.release.yaml",
            "--allow-top-active-routes",
            "--allow-precut-footprint-outside-active-product",
            "--materialize-top-k",
            "1",
            "--force-overwrite",
            "--json",
        ],
        color=False,
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert "final_geometry_source" not in payload["metadata"]
    assert payload["metadata"]["route_policy_final_geometry_source"] == "retained_active_strand"
    assert payload["metadata"]["allowed_active_strands"] == ["top", "bottom"]
    assert payload["metadata"]["allowed_route_families"] == [
        "bottom_active_from_top_nick",
        "top_active_from_bottom_nick",
    ]


def test_released_target_search_text_distinguishes_policy_and_hit_geometry(tmp_path: Path) -> None:
    workspace, _spec_path, _release_catalog_path = _write_workspace(tmp_path)

    result = runner.invoke(
        app,
        [
            "snapback",
            "released-target-search",
            "--workspace-root",
            str(workspace),
            "--nick-additional-path",
            "inputs/nickases/local.nickases.yaml",
            "--release-additional-path",
            "inputs/release_enzymes/local.release.yaml",
            "--allow-top-active-routes",
            "--allow-precut-footprint-outside-active-product",
        ],
        color=False,
    )

    assert result.exit_code == 0
    assert "Route policy -> policy_final_geometry=retained_active_strand" in result.output
    assert "hit_final_geometry=" in result.output


def test_released_solve_excludes_demo_only_entries_unless_opted_in(tmp_path: Path) -> None:
    workspace, _spec_path, _release_catalog_path = _write_workspace(tmp_path)
    nick_catalog_path = workspace / "inputs" / "nickases" / "local.nickases.yaml"
    release_catalog_path = workspace / "inputs" / "release_enzymes" / "local.release.yaml"

    nick_payload = yaml.safe_load(nick_catalog_path.read_text(encoding="utf-8"))
    nick_payload["nickases"]["entries"][0]["metadata"] = {"demo_only": True}
    nick_catalog_path.write_text(yaml.safe_dump(nick_payload, sort_keys=False), encoding="utf-8")

    release_payload = yaml.safe_load(release_catalog_path.read_text(encoding="utf-8"))
    release_payload["release_enzymes"]["entries"][0]["metadata"] = {"demo_only": True}
    release_catalog_path.write_text(yaml.safe_dump(release_payload, sort_keys=False), encoding="utf-8")

    blocked = runner.invoke(
        app,
        [
            "snapback",
            "released-solve",
            "--workspace-root",
            str(workspace),
            "--nick-additional-path",
            "inputs/nickases/local.nickases.yaml",
            "--release-additional-path",
            "inputs/release_enzymes/local.release.yaml",
            "--materialize-top-k",
            "1",
            "--force-overwrite",
            "--json",
        ],
        color=False,
    )

    assert blocked.exit_code == 1
    blocked_payload = json.loads(blocked.output)
    assert blocked_payload["status"] == "no_hits"
    assert blocked_payload["metadata"]["evaluated_pair_count"] == 0
    assert blocked_payload["metadata"]["blocker_counts"]["DEMO_ONLY_PAIR_SUPPRESSED"] >= 1

    allowed = runner.invoke(
        app,
        [
            "snapback",
            "released-solve",
            "--workspace-root",
            str(workspace),
            "--nick-additional-path",
            "inputs/nickases/local.nickases.yaml",
            "--release-additional-path",
            "inputs/release_enzymes/local.release.yaml",
            "--allow-demo-hits",
            "--materialize-top-k",
            "1",
            "--force-overwrite",
            "--json",
        ],
        color=False,
    )

    assert allowed.exit_code == 0
    allowed_payload = json.loads(allowed.output)
    assert allowed_payload["status"] == "exact_hits_materialized"
    assert allowed_payload["metadata"]["materialized_hit_count"] == 1
