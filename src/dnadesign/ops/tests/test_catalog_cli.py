"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/tests/test_catalog_cli.py

Contract tests for the read-only ops catalog discovery surface.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from dnadesign.ops.catalog import CatalogQuery, filter_runbook_catalog, load_runbook_catalog
from dnadesign.ops.cli import app


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def test_load_runbook_catalog_reads_shared_registry() -> None:
    catalog = load_runbook_catalog(repo_root=_repo_root())

    assert catalog.procedures
    assert catalog.tool_sources
    assert catalog.find_procedure("ops.control-plane.orchestration") is not None
    assert catalog.find_procedure("cluster.downstream.exploratory-clustering") is not None


def test_catalog_query_filters_procedures_without_touching_registry_ownership() -> None:
    catalog = load_runbook_catalog(repo_root=_repo_root())

    procedures, tool_sources = filter_runbook_catalog(
        catalog,
        query=CatalogQuery(plane="data-plane", query="infer"),
    )

    assert procedures
    assert all(entry.plane == "data-plane" for entry in procedures)
    assert any(entry.registry_id == "usr.data-plane.promoter-feature-matrix" for entry in procedures)
    assert tool_sources == ()


def test_cli_catalog_list_emits_grouped_text_inventory() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["catalog", "list", "--repo-root", str(_repo_root())])

    assert result.exit_code == 0
    assert "Catalog inventory" in result.output
    assert "Counts:" in result.output
    assert "Cross-tool procedures" in result.output
    assert "ops.control-plane.orchestration" in result.output
    assert "cluster.downstream.exploratory-clustering" in result.output
    assert "Tool-local runbook sources" in result.output
    assert "densegen: DenseGen docs" in result.output


def test_cli_catalog_list_supports_json_and_section_filter() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "catalog",
            "list",
            "--repo-root",
            str(_repo_root()),
            "--section",
            "procedures",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["section"] == "procedures"
    assert payload["filters"] == {}
    assert payload["counts"]["procedures"] >= 1
    assert "procedures" in payload
    assert "tool_sources" not in payload
    assert any(entry["registry_id"] == "opal.downstream.usr-infer-x-active-learning" for entry in payload["procedures"])


def test_cli_catalog_list_supports_queryable_filters() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "catalog",
            "list",
            "--repo-root",
            str(_repo_root()),
            "--plane",
            "downstream-tool",
            "--query",
            "active learning",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["filters"] == {"plane": "downstream-tool", "query": "active learning"}
    assert payload["counts"] == {"procedures": 1, "tool_sources": 0}
    assert [entry["registry_id"] for entry in payload["procedures"]] == ["opal.downstream.usr-infer-x-active-learning"]
    assert payload["tool_sources"] == []


def test_cli_catalog_show_emits_registered_entry() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "catalog",
            "show",
            "usr.data-plane.promoter-feature-matrix",
            "--repo-root",
            str(_repo_root()),
        ],
    )

    assert result.exit_code == 0
    assert "Registry id: usr.data-plane.promoter-feature-matrix" in result.output
    assert "Progress kind: usr-dataset-state" in result.output


def test_cli_catalog_show_rejects_unknown_registry_id() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "catalog",
            "show",
            "missing.registry.id",
            "--repo-root",
            str(_repo_root()),
        ],
    )

    assert result.exit_code == 2
    assert "unknown registry id: missing.registry.id" in result.output


def test_cli_catalog_show_suggests_close_registry_ids() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "catalog",
            "show",
            "usr.data-plane.promoter-feature",
            "--repo-root",
            str(_repo_root()),
        ],
    )

    assert result.exit_code == 2
    assert "Did you mean:" in result.output
    assert "usr.data-plane.promoter-feature-matrix" in result.output


def test_cli_catalog_uses_module_checkout_when_cwd_is_outside_repo(tmp_path: Path, monkeypatch) -> None:
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(outside_dir)
    runner = CliRunner()

    result = runner.invoke(app, ["catalog", "list"])

    assert result.exit_code == 0
    assert "Cross-tool procedures" in result.output


def test_cli_catalog_rejects_bad_explicit_repo_root(tmp_path: Path) -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["catalog", "list", "--repo-root", str(tmp_path)])

    assert result.exit_code == 2
    assert "docs/runbooks/README.md" in result.output
