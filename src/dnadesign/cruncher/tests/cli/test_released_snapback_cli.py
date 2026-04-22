"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/cli/test_released_snapback_cli.py

CLI contract tests for released-product snapback commands.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from dnadesign.cruncher.cli.app import app

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
                            "top_cut_offset": 0,
                            "bottom_cut_offset": 1,
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
                    "normalized_to_top_strand_nick": True,
                    "require_site_sequence_preserved_pre_nick": True,
                },
                "release_stage": {
                    "release_variant_id": "Re.Exact",
                    "catalog": {"additional_paths": ["inputs/release_enzymes/local.release.yaml"]},
                    "retained_side": "upstream",
                    "stage_order": "nick_then_release",
                    "require_site_sequence_preserved_pre_release": True,
                },
                "final_target": {"nick_boundary_from_left": 0, "paired_bp": 3, "cap_nt": 3},
                "constraints": {
                    "allow_post_release_loss_of_nickase_site": True,
                    "allow_post_release_loss_of_release_site": True,
                    "require_nick_survives_in_retained_product": True,
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
    assert "released-design" in result.output
    assert "released-target-search" in result.output
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
    assert any(
        hit["nickase_variant_id"] == "Nx.Near7" and hit["nick_boundary_from_left"] == 1 for hit in payload["near_hits"]
    )


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
