"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/portfolio/test_portfolio_load_and_layout.py

Validate Portfolio spec loading and output layout contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.cruncher.portfolio.layout import (
    portfolio_manifest_path,
    portfolio_plot_path,
    portfolio_status_path,
    portfolio_table_path,
    resolve_portfolio_run_dir,
)
from dnadesign.cruncher.portfolio.load import load_portfolio_spec


def test_load_resolves_workspace_and_run_paths(tmp_path: Path) -> None:
    spec_path = tmp_path / "workspaces" / "portfolio" / "handoff.portfolio.yaml"
    source_workspace = tmp_path / "workspaces" / "pairwise_cpxr_baer"
    source_run = source_workspace / "outputs" / "set1_cpxr_baer"
    source_run.mkdir(parents=True, exist_ok=True)

    spec_payload = {
        "portfolio": {
            "schema_version": 3,
            "name": "pairwise_handoff",
            "execution": {"mode": "aggregate_only"},
            "sources": [
                {
                    "id": "pairwise_cpxr_baer",
                    "workspace": "../pairwise_cpxr_baer",
                    "run_dir": "outputs/set1_cpxr_baer",
                }
            ],
        }
    }
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(yaml.safe_dump(spec_payload))

    spec = load_portfolio_spec(spec_path)
    source = spec.sources[0]
    assert source.workspace == source_workspace.resolve()
    assert source.run_dir == source_run.resolve()


def test_load_schema_v3_resolves_prepare_runbook_inside_source_workspace(tmp_path: Path) -> None:
    spec_path = tmp_path / "workspaces" / "portfolio" / "configs" / "handoff.portfolio.yaml"
    source_workspace = tmp_path / "workspaces" / "pairwise_cpxr_baer"
    source_run = source_workspace / "outputs" / "set1_cpxr_baer"
    source_run.mkdir(parents=True, exist_ok=True)
    source_runbook = source_workspace / "configs" / "runbook.yaml"
    source_runbook.parent.mkdir(parents=True, exist_ok=True)
    source_runbook.write_text(
        yaml.safe_dump(
            {
                "runbook": {
                    "schema_version": 1,
                    "name": "pairwise_cpxr_baer",
                    "steps": [{"id": "analyze_summary", "run": ["analyze", "--summary", "-c", "configs/config.yaml"]}],
                }
            }
        )
    )
    study_spec = source_workspace / "configs" / "studies" / "diversity_vs_score.study.yaml"
    study_spec.parent.mkdir(parents=True, exist_ok=True)
    study_spec.write_text(
        yaml.safe_dump(
            {
                "study": {
                    "schema_version": 3,
                    "name": "diversity_vs_score",
                    "base_config": "config.yaml",
                    "target": {"kind": "regulator_set", "set_index": 1},
                    "replicates": {"seed_path": "sample.seed", "seeds": [1]},
                    "trials": [{"id": "BASE", "factors": {}}],
                }
            }
        )
    )
    (source_workspace / "configs" / "config.yaml").write_text(
        "cruncher: {schema_version: 3, workspace: {out_dir: outputs, regulator_sets: [[cpxR, baeR]]}}\n"
    )

    spec_payload = {
        "portfolio": {
            "schema_version": 3,
            "name": "pairwise_handoff",
            "execution": {"mode": "prepare_then_aggregate"},
            "sources": [
                {
                    "id": "pairwise_cpxr_baer",
                    "workspace": "../pairwise_cpxr_baer",
                    "run_dir": "outputs/set1_cpxr_baer",
                    "prepare": {
                        "runbook": "configs/runbook.yaml",
                        "step_ids": ["analyze_summary"],
                    },
                    "study_spec": "configs/studies/diversity_vs_score.study.yaml",
                }
            ],
        }
    }
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(yaml.safe_dump(spec_payload))

    spec = load_portfolio_spec(spec_path)
    source = spec.sources[0]
    assert source.workspace == source_workspace.resolve()
    assert source.run_dir == source_run.resolve()
    assert source.prepare is not None
    assert source.prepare.runbook == source_runbook.resolve()
    assert source.prepare.step_ids == ["analyze_summary"]
    assert source.study_spec == study_spec.resolve()


def test_load_schema_v3_prepare_mode_allows_run_dir_to_be_created_by_prepare(tmp_path: Path) -> None:
    spec_path = tmp_path / "workspaces" / "portfolio" / "configs" / "handoff.portfolio.yaml"
    source_workspace = tmp_path / "workspaces" / "pairwise_cpxr_baer"
    source_workspace.mkdir(parents=True, exist_ok=True)
    source_runbook = source_workspace / "configs" / "runbook.yaml"
    source_runbook.parent.mkdir(parents=True, exist_ok=True)
    source_runbook.write_text(
        yaml.safe_dump(
            {
                "runbook": {
                    "schema_version": 1,
                    "name": "pairwise_cpxr_baer",
                    "steps": [
                        {
                            "id": "sample_run",
                            "run": ["sample", "--force-overwrite", "-c", "configs/config.yaml"],
                        }
                    ],
                }
            }
        )
    )

    spec_payload = {
        "portfolio": {
            "schema_version": 3,
            "name": "pairwise_handoff",
            "execution": {"mode": "prepare_then_aggregate"},
            "sources": [
                {
                    "id": "pairwise_cpxr_baer",
                    "workspace": "../pairwise_cpxr_baer",
                    "run_dir": "outputs/set1_cpxr_baer",
                    "prepare": {
                        "runbook": "configs/runbook.yaml",
                        "step_ids": ["sample_run"],
                    },
                }
            ],
        }
    }
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(yaml.safe_dump(spec_payload))

    spec = load_portfolio_spec(spec_path)
    source = spec.sources[0]
    assert source.run_dir == (source_workspace / "outputs" / "set1_cpxr_baer").resolve()


def test_load_rejects_missing_workspace(tmp_path: Path) -> None:
    spec_path = tmp_path / "handoff.portfolio.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "portfolio": {
                    "schema_version": 3,
                    "name": "pairwise_handoff",
                    "execution": {"mode": "aggregate_only"},
                    "sources": [
                        {
                            "id": "pairwise_cpxr_baer",
                            "workspace": "missing_workspace",
                            "run_dir": "outputs/set1",
                        }
                    ],
                }
            }
        )
    )

    with pytest.raises(FileNotFoundError, match="workspace does not exist"):
        load_portfolio_spec(spec_path)


def test_load_rejects_runbook_yaml_with_actionable_hint(tmp_path: Path) -> None:
    spec_path = tmp_path / "runbook.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "runbook": {
                    "schema_version": 1,
                    "name": "pairwise_cpxr_baer",
                    "steps": [
                        {
                            "id": "sample_run",
                            "run": ["sample", "--force-overwrite", "-c", "configs/config.yaml"],
                        }
                    ],
                }
            }
        )
    )

    with pytest.raises(ValueError, match="missing root key: portfolio") as exc_info:
        load_portfolio_spec(spec_path)
    message = str(exc_info.value)
    assert "looks like a workspace runbook" in message
    assert "cruncher workspaces run --runbook" in message


def test_load_reports_multiple_missing_workspaces_in_one_error(tmp_path: Path) -> None:
    spec_path = tmp_path / "handoff.portfolio.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "portfolio": {
                    "schema_version": 3,
                    "name": "pairwise_handoff",
                    "execution": {"mode": "aggregate_only"},
                    "sources": [
                        {
                            "id": "pairwise_cpxr_baer",
                            "workspace": "missing_a",
                            "run_dir": "outputs/set1",
                        },
                        {
                            "id": "pairwise_cpxr_lexa",
                            "workspace": "missing_b",
                            "run_dir": "outputs/set1",
                        },
                    ],
                }
            }
        )
    )

    with pytest.raises(ValueError, match="invalid source paths or artifacts") as exc_info:
        load_portfolio_spec(spec_path)
    message = str(exc_info.value)
    assert "pairwise_cpxr_baer" in message
    assert "pairwise_cpxr_lexa" in message


def test_load_formats_schema_validation_errors_without_pydantic_noise(tmp_path: Path) -> None:
    spec_path = tmp_path / "bad_schema.portfolio.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "portfolio": {
                    "schema_version": 2,
                    "name": "pairwise_handoff",
                    "execution": {"mode": "aggregate_only"},
                    "sources": [],
                    "extra_field": True,
                }
            }
        )
    )

    with pytest.raises(ValueError, match="Portfolio schema validation failed") as exc_info:
        load_portfolio_spec(spec_path)
    message = str(exc_info.value)
    assert "- portfolio.schema_version:" in message
    assert "- portfolio.sources:" in message
    assert "- portfolio.extra_field:" in message
    assert "https://errors.pydantic.dev" not in message


def test_load_rejects_run_dir_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace_a"
    workspace.mkdir(parents=True)
    outside_run = tmp_path / "workspace_b" / "outputs" / "set1"
    outside_run.mkdir(parents=True)

    spec_path = tmp_path / "handoff.portfolio.yaml"
    spec_path.write_text(
        yaml.safe_dump(
            {
                "portfolio": {
                    "schema_version": 3,
                    "name": "pairwise_handoff",
                    "execution": {"mode": "aggregate_only"},
                    "sources": [
                        {
                            "id": "pairwise_cpxr_baer",
                            "workspace": "workspace_a",
                            "run_dir": "../workspace_b/outputs/set1",
                        }
                    ],
                }
            }
        )
    )

    with pytest.raises(ValueError, match="run_dir must be inside workspace"):
        load_portfolio_spec(spec_path)


def test_layout_builds_expected_paths(tmp_path: Path) -> None:
    workspace_root = (tmp_path / "portfolio_workspace").resolve()
    run_dir = resolve_portfolio_run_dir(
        workspace_root,
        portfolio_name="pairwise_handoff",
        portfolio_id="abc123",
    )

    assert run_dir == workspace_root / "outputs" / "pairwise_handoff" / "abc123"
    assert portfolio_manifest_path(run_dir) == run_dir / "meta" / "manifest.json"
    assert portfolio_status_path(run_dir) == run_dir / "meta" / "status.json"
    assert (
        portfolio_table_path(
            run_dir,
            "handoff_windows_long",
            "parquet",
        )
        == run_dir / "tables" / "table__handoff_windows_long.parquet"
    )
    assert portfolio_plot_path(run_dir, "source_tradeoff", "pdf") == (run_dir / "plots" / "plot__source_tradeoff.pdf")
