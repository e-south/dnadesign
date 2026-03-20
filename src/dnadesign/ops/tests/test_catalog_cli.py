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
    assert catalog.find_tool_source("usr") is not None


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
    assert "densegen: DenseGen documentation" in result.output
    assert "usr: USR docs" in result.output
    assert "Suggested next steps" in result.output
    assert "uv run ops catalog list --query <term>" in result.output
    assert "uv run ops catalog list --simple" in result.output
    assert "uv run ops catalog show ops.control-plane.orchestration" in result.output
    assert "uv run ops progress explain ops.control-plane.orchestration" in result.output
    assert "uv run ops progress scaffold ops.control-plane.orchestration" in result.output


def test_cli_help_points_new_users_to_task_first_catalog_list() -> None:
    runner = CliRunner()

    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    normalized_output = " ".join(result.output.split())
    assert "Start with `uv run ops catalog list --simple` to browse routes from the terminal." in normalized_output


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
    assert payload["view"] == "full"
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


def test_cli_catalog_list_matches_procedure_keywords_for_promoter_alias_queries() -> None:
    runner = CliRunner()

    for query in ("wildtype promoter", "evo2 promoter"):
        result = runner.invoke(
            app,
            [
                "catalog",
                "list",
                "--repo-root",
                str(_repo_root()),
                "--section",
                "procedures",
                "--query",
                query,
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert any(
            entry["registry_id"] == "usr.data-plane.promoter-feature-matrix" for entry in payload["procedures"]
        ), query


def test_cli_catalog_list_supports_simple_text_view() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "catalog",
            "list",
            "--repo-root",
            str(_repo_root()),
            "--simple",
        ],
    )

    assert result.exit_code == 0
    assert "Task-first procedures" in result.output
    assert "Tool docs" in result.output
    assert "Registry id: ops.control-plane.orchestration" in result.output
    assert "Inspect: uv run ops catalog show ops.control-plane.orchestration" in result.output
    assert "[runbook | control-plane | executable | ops-audit-json]" not in result.output


def test_cli_catalog_list_supports_tool_source_queries_for_promoter_path() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "catalog",
            "list",
            "--repo-root",
            str(_repo_root()),
            "--section",
            "tool-sources",
            "--query",
            "promoter feature matrix",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["filters"] == {"query": "promoter feature matrix"}
    tools = {entry["tool"] for entry in payload["tool_sources"]}
    assert {"usr", "infer", "opal"}.issubset(tools)
    assert "procedures" not in payload


def test_cli_catalog_list_supports_related_tool_sources() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "catalog",
            "list",
            "--repo-root",
            str(_repo_root()),
            "--section",
            "tool-sources",
            "--related-to",
            "usr.data-plane.promoter-feature-matrix",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["filters"] == {"related_to": "usr.data-plane.promoter-feature-matrix"}
    assert payload["counts"] == {"tool_sources": 5}
    assert [entry["tool"] for entry in payload["tool_sources"]] == [
        "densegen",
        "construct",
        "infer",
        "cluster",
        "opal",
    ]
    assert "procedures" not in payload


def test_cli_catalog_list_supports_related_to_filter() -> None:
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
            "--related-to",
            "usr.data-plane.promoter-feature-matrix",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["filters"] == {"related_to": "usr.data-plane.promoter-feature-matrix"}
    assert payload["counts"] == {"procedures": 4}
    assert [entry["registry_id"] for entry in payload["procedures"]] == [
        "usr.data-plane.multi-source-source-of-truth",
        "usr.data-plane.construct-infer-source-of-truth",
        "cluster.downstream.exploratory-clustering",
        "opal.downstream.usr-infer-x-active-learning",
    ]
    assert "tool_sources" not in payload


def test_cli_catalog_list_tool_sources_suggests_tool_source_specific_next_steps() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "catalog",
            "list",
            "--repo-root",
            str(_repo_root()),
            "--section",
            "tool-sources",
        ],
    )

    assert result.exit_code == 0
    assert "Narrow the docs by topic" in result.output
    assert "uv run ops catalog list --section tool-sources --query <term>" in result.output
    assert "Browse all registered procedures" in result.output
    assert "Inspect the first matching procedure" not in result.output


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
    assert "Owner boundary: usr" in result.output
    assert "Entry artifact: one or more USR-backed promoter datasets" in result.output
    assert "Exit artifact: infer-annotated USR feature matrix ready for cluster or OPAL" in result.output
    assert "Progress kind: usr-dataset-state" in result.output
    assert "Owner docs:" in result.output
    assert "- usr: USR docs" in result.output
    assert "Related tool docs:" in result.output
    assert "- densegen: DenseGen documentation" in result.output
    assert "- construct: Construct docs" in result.output
    assert "- infer: infer docs" in result.output
    assert "- cluster: Cluster docs" in result.output
    assert "- opal: OPAL Documentation" in result.output
    assert "Related deep docs:" in result.output
    assert "- construct/template-contexts: Construct Template Contexts" in result.output
    assert "- infer/architecture: Infer Architecture" in result.output
    assert "- infer/evo2-provider: Evo2 Provider Reference" in result.output
    assert "- infer/evo2-promoter-features: Evo2 Promoter Feature Runbook" in result.output
    assert "- cluster/exploratory-clustering: Exploratory clustering workflow" in result.output
    assert "- opal/usr-infer-x-active-learning: USR Dataset With Infer-Derived X -> OPAL Active Learning" in (
        result.output
    )
    assert "Required progress inputs:" in result.output
    assert "--usr-root <usr-root>" in result.output
    assert "--dataset <dataset>" in result.output
    assert "Related procedures:" in result.output
    assert "depends-on: usr.data-plane.multi-source-source-of-truth" in result.output
    assert "depends-on: usr.data-plane.construct-infer-source-of-truth" in result.output
    assert "handoff-to: cluster.downstream.exploratory-clustering" in result.output
    assert "handoff-to: opal.downstream.usr-infer-x-active-learning" in result.output
    assert "Next commands:" in result.output
    assert "uv run ops progress explain usr.data-plane.promoter-feature-matrix" in result.output
    assert (
        "uv run ops progress show usr.data-plane.promoter-feature-matrix --usr-root <usr-root> --dataset <dataset>"
    ) in result.output
    assert "uv run ops progress scaffold usr.data-plane.promoter-feature-matrix" in result.output
    assert "uv run ops catalog list --section tool-sources --tool usr" in result.output
    assert "uv run ops catalog list --section tool-sources --related-to usr.data-plane.promoter-feature-matrix" in (
        result.output
    )
    assert (
        "uv run ops catalog list --section procedures --related-to usr.data-plane.promoter-feature-matrix"
    ) in result.output
    assert "uv run ops progress scaffold --related-to usr.data-plane.promoter-feature-matrix" in result.output


def test_cli_catalog_show_json_includes_related_procedures() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "catalog",
            "show",
            "usr.data-plane.promoter-feature-matrix",
            "--repo-root",
            str(_repo_root()),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["owner_boundary"] == "usr"
    assert payload["entry_artifact"].startswith("one or more USR-backed promoter datasets")
    assert payload["exit_artifact"] == "infer-annotated USR feature matrix ready for cluster or OPAL"
    assert payload["owner_tool_source"]["tool"] == "usr"
    assert payload["owner_tool_source"]["doc_path"] == "src/dnadesign/usr/docs/README.md"
    assert [entry["tool"] for entry in payload["related_tool_sources"]] == [
        "densegen",
        "construct",
        "infer",
        "cluster",
        "opal",
    ]
    assert [(entry["tool"], entry["route_id"]) for entry in payload["related_tool_routes"]] == [
        ("construct", "template-contexts"),
        ("infer", "architecture"),
        ("infer", "evo2-provider"),
        ("infer", "evo2-promoter-features"),
        ("cluster", "exploratory-clustering"),
        ("opal", "usr-infer-x-active-learning"),
    ]
    assert payload["related_tool_routes"][0]["doc_path"] == (
        "src/dnadesign/construct/docs/reference/template-contexts.md"
    )
    assert payload["related_tool_routes"][1]["doc_path"] == "src/dnadesign/infer/docs/architecture/README.md"
    assert payload["progress_required_inputs"] == [
        {
            "cli_flag": "--usr-root",
            "manifest_key": "usr_root",
            "placeholder": "<usr-root>",
            "summary": "USR root containing the target dataset directory.",
        },
        {
            "cli_flag": "--dataset",
            "manifest_key": "dataset",
            "placeholder": "<dataset>",
            "summary": "USR dataset id to summarize.",
        },
    ]
    assert payload["next_commands"]["progress_explain"] == (
        "uv run ops progress explain usr.data-plane.promoter-feature-matrix"
    )
    assert payload["next_commands"]["progress_show"] == (
        "uv run ops progress show usr.data-plane.promoter-feature-matrix --usr-root <usr-root> --dataset <dataset>"
    )
    assert payload["next_commands"]["progress_scaffold"] == (
        "uv run ops progress scaffold usr.data-plane.promoter-feature-matrix"
    )
    assert payload["next_commands"]["catalog_owner_tool_source"] == (
        "uv run ops catalog list --section tool-sources --tool usr"
    )
    assert payload["next_commands"]["catalog_related_tool_sources"] == (
        "uv run ops catalog list --section tool-sources --related-to usr.data-plane.promoter-feature-matrix"
    )
    assert payload["next_commands"]["catalog_related"] == (
        "uv run ops catalog list --section procedures --related-to usr.data-plane.promoter-feature-matrix"
    )
    assert payload["next_commands"]["progress_scaffold_related"] == (
        "uv run ops progress scaffold --related-to usr.data-plane.promoter-feature-matrix"
    )
    assert [(entry["relation_type"], entry["registry_id"]) for entry in payload["related_procedures"]] == [
        ("depends-on", "usr.data-plane.multi-source-source-of-truth"),
        ("depends-on", "usr.data-plane.construct-infer-source-of-truth"),
        ("handoff-to", "cluster.downstream.exploratory-clustering"),
        ("handoff-to", "opal.downstream.usr-infer-x-active-learning"),
    ]


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


def test_cli_catalog_list_rejects_unknown_related_to_registry_id() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "catalog",
            "list",
            "--repo-root",
            str(_repo_root()),
            "--related-to",
            "usr.data-plane.promoter-feature",
        ],
    )

    assert result.exit_code == 2
    assert "unknown --related-to registry id: usr.data-plane.promoter-feature" in result.output
    assert "usr.data-plane.promoter-feature-matrix" in result.output


def test_cli_catalog_list_emits_recovery_steps_for_empty_results() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "catalog",
            "list",
            "--repo-root",
            str(_repo_root()),
            "--query",
            "no-such-catalog-entry",
        ],
    )

    assert result.exit_code == 0
    assert "No matching catalog entries. Try:" in result.output
    assert "uv run ops catalog list" in result.output
    assert "uv run ops catalog list --section tool-sources" in result.output


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
