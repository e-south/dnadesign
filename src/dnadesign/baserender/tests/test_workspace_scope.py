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

from typer.testing import CliRunner

from dnadesign.baserender.src.api import run_cruncher_showcase_job
from dnadesign.baserender.src.cli import app
from dnadesign.baserender.src.config import load_cruncher_showcase_job
from dnadesign.baserender.src.workspace import discover_workspaces, init_workspace, resolve_workspace_job_path

from .conftest import write_job, write_parquet


def _workspace_job_payload() -> dict:
    return {
        "version": 3,
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
    assert (workspace.root / "inputs").exists()
    assert (workspace.root / "outputs").exists()
    assert (workspace.root / "reports").exists()


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


def test_job_validate_requires_exactly_one_job_source() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["job", "validate"])
    assert result.exit_code == 2
    assert "Provide exactly one of <job> or --workspace" in result.output


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
    assert Path(report.outputs["report_path"]) == (workspace.root / "outputs" / "run_report.json").resolve()
