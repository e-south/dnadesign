"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/baserender/tests/test_workspace_scope.py

Tests for workspace-scoped baserender job scaffolding and workspace-aware CLI flow.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from dnadesign.baserender.src.cli import app
from dnadesign.baserender.src.config import load_cruncher_showcase_job
from dnadesign.baserender.src.core import SchemaError
from dnadesign.baserender.src.public import run_cruncher_showcase_job
from dnadesign.baserender.src.workspaces import discover_workspaces, init_workspace, resolve_workspace_job_path

from .conftest import write_job, write_parquet


def _workspace_job_payload() -> dict:
    return {
        "version": 3,
        "contract": {"kind": "sequence_rows_render_v3"},
        "input": {
            "kind": "parquet",
            "path": "inputs/input.parquet",
            "adapter": {
                "kind": "generic_features",
                "columns": {
                    "sequence": "sequence",
                    "features": "features",
                    "effects": "effects",
                    "display": "display",
                    "id": "id",
                },
                "policies": {},
            },
            "alphabet": "DNA",
        },
        "render": {"renderer": "sequence_rows", "style": {"preset": None, "overrides": {}}},
        "outputs": [{"kind": "images", "fmt": "png"}],
    }


def test_workspace_init_scaffolds_standard_layout(tmp_path: Path) -> None:
    workspace = init_workspace("demo_workspace", root=tmp_path)

    assert workspace.name == "demo_workspace"
    assert workspace.root == (tmp_path / "demo_workspace").resolve()
    assert workspace.job_path == workspace.root / "job.yaml"
    assert (workspace.root / ".baserender-workspace").exists()
    assert (workspace.root / "README.md").exists()
    assert (workspace.root / "inputs").exists()
    assert (workspace.root / "inputs" / "README.md").exists()
    assert (workspace.root / "outputs").exists()
    assert (workspace.root / "outputs" / "README.md").exists()
    assert not (workspace.root / "reports").exists()
    assert "inputs/input.parquet" in (workspace.root / "README.md").read_text(encoding="utf-8")
    assert "outputs/plots/" in (workspace.root / "outputs" / "README.md").read_text(encoding="utf-8")
    assert "contract:\n  kind: sequence_rows_render_v3" in workspace.job_path.read_text(encoding="utf-8")


def test_workspace_job_uses_workspace_outputs_by_default(tmp_path: Path) -> None:
    workspace = init_workspace("demo_workspace", root=tmp_path)
    write_parquet(
        workspace.root / "inputs" / "input.parquet",
        [
            {
                "id": "r1",
                "sequence": "ACGT",
                "features": [
                    {
                        "id": "k1",
                        "kind": "kmer",
                        "span": {"start": 0, "end": 4, "strand": "fwd"},
                        "label": "ACGT",
                        "tags": ["demo"],
                    }
                ],
                "effects": [],
                "display": {"overlay_text": None, "tag_labels": {"demo": "demo"}},
            }
        ],
    )
    write_job(workspace.job_path, _workspace_job_payload())

    parsed = load_cruncher_showcase_job(workspace.job_path)
    assert parsed.results_root == (workspace.root / "outputs").resolve()


def test_workspace_selector_resolves_in_cli_and_validate_passes(tmp_path: Path) -> None:
    workspace = init_workspace("demo_workspace", root=tmp_path)
    write_parquet(
        workspace.root / "inputs" / "input.parquet",
        [
            {
                "id": "r1",
                "sequence": "ACGT",
                "features": [
                    {
                        "id": "k1",
                        "kind": "kmer",
                        "span": {"start": 0, "end": 4, "strand": "fwd"},
                        "label": "ACGT",
                        "tags": ["demo"],
                    }
                ],
                "effects": [],
                "display": {"overlay_text": None, "tag_labels": {"demo": "demo"}},
            }
        ],
    )
    write_job(workspace.job_path, _workspace_job_payload())

    found = discover_workspaces(root=tmp_path)
    assert [ws.name for ws in found] == ["demo_workspace"]
    assert resolve_workspace_job_path("demo_workspace", root=tmp_path) == workspace.job_path

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "job",
            "validate",
            "--workspace",
            "demo_workspace",
            "--workspace-root",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0
    assert "OK:" in result.output


def test_workspace_discovery_ignores_unmarked_job_directories(tmp_path: Path) -> None:
    candidate = tmp_path / "accidental_workspace"
    (candidate / "inputs").mkdir(parents=True)
    (candidate / "outputs").mkdir()
    write_job(candidate / "job.yaml", _workspace_job_payload())

    assert discover_workspaces(root=tmp_path) == ()


def test_workspace_selector_rejects_unmarked_job_directory(tmp_path: Path) -> None:
    candidate = tmp_path / "accidental_workspace"
    (candidate / "inputs").mkdir(parents=True)
    (candidate / "outputs").mkdir()
    write_job(candidate / "job.yaml", _workspace_job_payload())

    with pytest.raises(SchemaError, match=".baserender-workspace"):
        resolve_workspace_job_path("accidental_workspace", root=tmp_path)

    result = CliRunner().invoke(
        app,
        [
            "job",
            "validate",
            "--workspace",
            "accidental_workspace",
            "--workspace-root",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 2
    assert ".baserender-workspace" in result.output


def test_unmarked_job_yaml_with_inputs_outputs_uses_job_local_results(tmp_path: Path) -> None:
    candidate = tmp_path / "accidental_workspace"
    (candidate / "inputs").mkdir(parents=True)
    (candidate / "outputs").mkdir()
    write_parquet(
        candidate / "inputs" / "input.parquet",
        [
            {
                "id": "r1",
                "sequence": "ACGT",
                "features": [],
                "effects": [],
                "display": {"overlay_text": None},
            }
        ],
    )
    write_job(candidate / "job.yaml", _workspace_job_payload())

    parsed = load_cruncher_showcase_job(candidate / "job.yaml")

    assert parsed.results_root == (candidate / "results").resolve()


def test_job_validate_requires_exactly_one_job_source() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["job", "validate"])
    assert result.exit_code == 2
    assert "Provide exactly one of <job> or --workspace" in result.output


def test_workspace_init_rejects_path_like_name_with_actionable_error() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["workspace", "init", "/tmp/path_like_name"])
    assert result.exit_code == 2
    assert "use --root <dir>" in result.output


def test_workspace_init_cli_prints_next_step_hint(tmp_path: Path) -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["workspace", "init", "demo_workspace", "--root", str(tmp_path)])
    assert result.exit_code == 0
    assert "inputs/input.parquet" in result.output


def test_workspace_run_defaults_outputs_to_workspace_outputs_root(tmp_path: Path) -> None:
    workspace = init_workspace("demo_workspace", root=tmp_path)
    write_parquet(
        workspace.root / "inputs" / "input.parquet",
        [
            {
                "id": "r1",
                "sequence": "ACGT",
                "features": [
                    {
                        "id": "k1",
                        "kind": "kmer",
                        "span": {"start": 0, "end": 4, "strand": "fwd"},
                        "label": "ACGT",
                        "tags": ["demo"],
                    }
                ],
                "effects": [],
                "display": {"overlay_text": "demo", "tag_labels": {"demo": "demo"}},
            }
        ],
    )
    write_job(workspace.job_path, _workspace_job_payload())

    report = run_cruncher_showcase_job(str(workspace.job_path))
    assert Path(report.outputs["images_dir"]) == (workspace.root / "outputs" / "plots").resolve()
    assert "report_path" not in report.outputs
