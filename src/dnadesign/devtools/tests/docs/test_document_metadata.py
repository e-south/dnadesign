"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/docs/test_document_metadata.py

Tests for documentation document metadata.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from dnadesign.devtools.docs.checks import (
    _find_cross_tool_doc_metadata_issues,
    _find_landing_readme_frontmatter_issues,
    _find_repo_local_skill_frontmatter_issues,
    _find_runbook_catalog_issues,
    _find_shared_usr_dataset_layout_issues,
    _find_tool_docs_metadata_issues,
    main,
)
from dnadesign.devtools.tests.docs.check_test_support import (
    _empty_tool_source_section,
    _write,
    _write_changed_files,
    _write_generated_runbook_catalog_readme,
    _write_registry_metadata,
    _write_runbook_catalog_readme,
    _write_tool_source_metadata,
)
from dnadesign.ops.catalog import (
    load_runbook_catalog,
    render_catalog_procedure_section,
)


def test_find_repo_local_skill_frontmatter_issues_rejects_overlong_description(tmp_path: Path) -> None:
    _write(
        tmp_path / ".agents" / "skills" / "demo-skill" / "SKILL.md",
        "\n".join(
            [
                "---",
                "name: demo-skill",
                f"description: {'x' * 221}",
                "metadata:",
                "  version: 0.1.0",
                "  category: workflow-automation",
                "---",
                "",
                "# Demo Skill",
            ]
        )
        + "\n",
    )

    issues = _find_repo_local_skill_frontmatter_issues(tmp_path)

    assert any("frontmatter description length 221/220" in issue for issue in issues)


def test_shared_usr_dataset_layout_check_flags_nested_dataset_roots(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "usr" / "datasets" / "densegen" / "demo_sampling_baseline" / "records.parquet",
        "placeholder\n",
    )

    issues = _find_shared_usr_dataset_layout_issues(tmp_path)

    assert any("densegen/demo_sampling_baseline" in issue for issue in issues)
    assert all("Shared repo USR dataset roots must be flat" in issue for issue in issues)
    assert all("archived/ is the only special top-level bucket" in issue for issue in issues)


def test_shared_usr_dataset_layout_check_allows_flat_roots_and_archived_bucket(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "usr" / "datasets" / "densegen_demo_sampling_baseline" / "records.parquet",
        "placeholder\n",
    )
    _write(
        tmp_path
        / "src"
        / "dnadesign"
        / "usr"
        / "datasets"
        / "archived"
        / "densegen"
        / "demo_sampling_baseline"
        / "records.parquet",
        "placeholder\n",
    )

    assert _find_shared_usr_dataset_layout_issues(tmp_path) == []


def test_landing_readme_frontmatter_check_rejects_root_and_tool_metadata_blocks(tmp_path: Path) -> None:
    _write(
        tmp_path / "README.md",
        "---\ndoc_id: repository\n---\n\n![dnadesign banner](assets/banner.svg)\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "README.md",
        "\n".join(
            [
                "---",
                "doc_id: alpha-package",
                "owner: maintainers",
                f"last_verified: {dt.date.today().isoformat()}",
                "---",
                "",
                "![Alpha banner](assets/alpha-banner.svg)",
                "",
                "Short narrative overview.",
                "",
                "## Documentation",
                "",
                "See [docs index](docs/README.md) for workflows and references.",
                "",
            ]
        ),
    )
    issues = _find_landing_readme_frontmatter_issues(tmp_path)

    assert len(issues) == 2
    assert any(str(tmp_path / "README.md") in issue for issue in issues)
    assert any(str(tmp_path / "src" / "dnadesign" / "alpha" / "README.md") in issue for issue in issues)


def test_main_fails_when_root_sor_doc_missing_required_metadata(tmp_path: Path) -> None:
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(tmp_path / "ARCHITECTURE.md", "# ARCHITECTURE\n")

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_root_sor_doc_missing_type_metadata(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_root_sor_verification_predates_document_change(tmp_path: Path) -> None:
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        "# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** 2020-01-01\n",
    )

    changed_files = _write_changed_files(tmp_path, "ARCHITECTURE.md")
    rc = main(["--repo-root", str(tmp_path), "--changed-files-file", str(changed_files)])
    assert rc == 1


def test_main_fails_when_docs_index_missing_required_metadata(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "README.md", "## Documentation Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_passes_when_docs_index_has_required_metadata(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(
        tmp_path / "docs" / "README.md",
        f"## Documentation Index\n\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 0


def test_main_fails_when_docs_index_verification_predates_document_change(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(
        tmp_path / "docs" / "README.md",
        "## Documentation Index\n\n**Owner:** maintainers\n**Last verified:** 2020-01-01\n",
    )
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )

    changed_files = _write_changed_files(tmp_path, "docs/README.md")
    rc = main(["--repo-root", str(tmp_path), "--changed-files-file", str(changed_files)])
    assert rc == 1


def test_main_fails_when_selected_runbook_missing_required_metadata(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(
        tmp_path / "docs" / "setup" / "installation.md",
        "## Installation\n\nRun setup.\n",
    )
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_selected_runbook_verification_predates_document_change(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(
        tmp_path / "docs" / "setup" / "installation.md",
        "## Installation\n\n**Owner:** maintainers\n**Last verified:** 2020-01-01\n",
    )
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )

    changed_files = _write_changed_files(tmp_path, "docs/setup/installation.md")
    rc = main(["--repo-root", str(tmp_path), "--changed-files-file", str(changed_files)])
    assert rc == 1


def test_main_passes_when_selected_runbook_has_required_metadata(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(
        tmp_path / "docs" / "setup" / "installation.md",
        f"## Installation\n\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 0


def test_main_fails_when_operations_runbook_docs_missing_required_metadata(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(
        tmp_path / "docs" / "operations" / "README.md",
        "## Ops orchestration index\n\nMissing metadata.\n",
    )
    _write(
        tmp_path / "docs" / "operations" / "orchestration/runbooks.md",
        "## Orchestration runbooks\n\nMissing metadata.\n",
    )
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_exec_plan_missing_required_metadata(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "docs" / "exec-plans" / "active" / "example.md", "# Exec plan\n")

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_exec_plan_missing_required_living_sections(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(
        tmp_path / "docs" / "exec-plans" / "active" / "example.md",
        "\n".join(
            [
                "# Exec plan",
                "",
                "**Status:** active",
                "**Owner:** maintainers",
                "**Created:** 2026-02-18",
                "",
                "## Purpose / Big Picture",
                "Purpose.",
                "",
                "## Progress",
                "- [ ] (2026-02-18 10:00Z) pending",
                "",
                "[proposal](https://example.com/proposal)",
            ]
        )
        + "\n",
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_passes_for_exec_plan_with_required_metadata(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(
        tmp_path / "docs" / "exec-plans" / "active" / "example.md",
        "\n".join(
            [
                "## Exec plan",
                "",
                "**Status:** active",
                "**Owner:** maintainers",
                "**Created:** 2026-02-18",
                "",
                "### Purpose / Big Picture",
                "Purpose.",
                "",
                "### Progress",
                "- [ ] (2026-02-18 10:00Z) pending",
                "",
                "### Surprises & Discoveries",
                "- Observation: none",
                "  Evidence: none",
                "",
                "### Decision Log",
                "- Decision: none",
                "  Rationale: none",
                "  Date/Author: 2026-02-18 / maintainers",
                "",
                "### Outcomes & Retrospective",
                "Pending.",
                "",
                "### Context and Orientation",
                "Context.",
                "",
                "### Plan of Work",
                "Plan.",
                "",
                "### Concrete Steps",
                "Run command.",
                "",
                "### Validation and Acceptance",
                "Validate behavior.",
                "",
                "[proposal](https://example.com/proposal)",
            ]
        )
        + "\n",
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 0


def test_main_fails_when_exec_plan_progress_has_no_checklist_items(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(
        tmp_path / "docs" / "exec-plans" / "active" / "example.md",
        "\n".join(
            [
                "# Exec plan",
                "",
                "**Status:** active",
                "**Owner:** maintainers",
                "**Created:** 2026-02-18",
                "",
                "## Purpose / Big Picture",
                "Purpose.",
                "",
                "## Progress",
                "Progress narrative only.",
                "",
                "## Surprises & Discoveries",
                "- Observation: none",
                "  Evidence: none",
                "",
                "## Decision Log",
                "- Decision: none",
                "  Rationale: none",
                "  Date/Author: 2026-02-18 / maintainers",
                "",
                "## Outcomes & Retrospective",
                "Pending.",
                "",
                "## Context and Orientation",
                "Context.",
                "",
                "## Plan of Work",
                "Plan.",
                "",
                "## Concrete Steps",
                "Run command.",
                "",
                "## Validation and Acceptance",
                "Validate behavior.",
                "",
                "[proposal](https://example.com/proposal)",
            ]
        )
        + "\n",
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_exec_plan_has_checklist_outside_progress(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(
        tmp_path / "docs" / "exec-plans" / "active" / "example.md",
        "\n".join(
            [
                "# Exec plan",
                "",
                "**Status:** active",
                "**Owner:** maintainers",
                "**Created:** 2026-02-18",
                "",
                "## Purpose / Big Picture",
                "Purpose.",
                "",
                "## Progress",
                "- [ ] pending",
                "",
                "## Surprises & Discoveries",
                "- Observation: none",
                "  Evidence: none",
                "",
                "## Decision Log",
                "- Decision: none",
                "  Rationale: none",
                "  Date/Author: 2026-02-18 / maintainers",
                "",
                "## Outcomes & Retrospective",
                "Pending.",
                "",
                "## Context and Orientation",
                "Context.",
                "",
                "## Plan of Work",
                "Plan.",
                "",
                "## Concrete Steps",
                "Run command.",
                "",
                "## Validation and Acceptance",
                "- [ ] run tests",
                "",
                "[proposal](https://example.com/proposal)",
            ]
        )
        + "\n",
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_exec_plan_progress_checklist_lacks_timestamp(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(
        tmp_path / "docs" / "exec-plans" / "active" / "example.md",
        "\n".join(
            [
                "# Exec plan",
                "",
                "**Status:** active",
                "**Owner:** maintainers",
                "**Created:** 2026-02-18",
                "",
                "## Purpose / Big Picture",
                "Purpose.",
                "",
                "## Progress",
                "- [ ] pending",
                "",
                "## Surprises & Discoveries",
                "- Observation: none",
                "  Evidence: none",
                "",
                "## Decision Log",
                "- Decision: none",
                "  Rationale: none",
                "  Date/Author: 2026-02-18 / maintainers",
                "",
                "## Outcomes & Retrospective",
                "Pending.",
                "",
                "## Context and Orientation",
                "Context.",
                "",
                "## Plan of Work",
                "Plan.",
                "",
                "## Concrete Steps",
                "Run command.",
                "",
                "## Validation and Acceptance",
                "Validate behavior.",
                "",
                "[proposal](https://example.com/proposal)",
            ]
        )
        + "\n",
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_tool_docs_metadata_check_flags_missing_owner_and_last_verified(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "alpha" / "__init__.py", "")
    _write(tmp_path / "src" / "dnadesign" / "alpha" / "docs" / "README.md", "## Alpha docs\n")

    issues = _find_tool_docs_metadata_issues(tmp_path)

    assert any("missing '**Owner:**' metadata field." in issue for issue in issues)
    assert any("missing '**Last verified:**' metadata field." in issue for issue in issues)


def test_tool_docs_metadata_check_accepts_valid_owner_and_last_verified(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "src" / "dnadesign" / "alpha" / "__init__.py", "")
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "docs" / "README.md",
        f"## Alpha docs\n\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )

    issues = _find_tool_docs_metadata_issues(tmp_path)

    assert issues == []


def test_tool_docs_metadata_check_accepts_yaml_frontmatter(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "src" / "dnadesign" / "alpha" / "__init__.py", "")
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "docs" / "README.md",
        f"---\nowner: maintainers\nlast_verified: {today}\n---\n\n# Alpha docs\n",
    )

    assert _find_tool_docs_metadata_issues(tmp_path) == []


def test_tool_docs_metadata_check_does_not_expire_unchanged_docs(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "alpha" / "__init__.py", "")
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "docs" / "README.md",
        "## Alpha docs\n\n**Owner:** maintainers\n**Last verified:** 2020-01-01\n",
    )

    assert _find_tool_docs_metadata_issues(tmp_path) == []


def test_tool_docs_metadata_check_flags_verification_before_change(tmp_path: Path) -> None:
    relative_path = "src/dnadesign/alpha/docs/README.md"
    _write(tmp_path / "src" / "dnadesign" / "alpha" / "__init__.py", "")
    _write(
        tmp_path / relative_path,
        "## Alpha docs\n\n**Owner:** maintainers\n**Last verified:** 2020-01-01\n",
    )

    issues = _find_tool_docs_metadata_issues(
        tmp_path,
        changed_doc_dates={relative_path: dt.date(2026, 7, 12)},
    )

    assert len(issues) == 1
    assert "predates this document's 2026-07-12 change" in issues[0]


def test_cross_tool_doc_metadata_check_flags_missing_semantic_fields(tmp_path: Path) -> None:
    _write(
        tmp_path / "docs" / "operations" / "README.md",
        "\n".join(
            [
                "## Ops orchestration index",
                "",
                "**Type:** route",
                "**Owner:** maintainers",
                f"**Last verified:** {dt.date.today().isoformat()}",
                "",
                "Missing plane and artifact metadata.",
            ]
        )
        + "\n",
    )

    issues = _find_cross_tool_doc_metadata_issues(tmp_path)

    assert any("missing '**Plane:**'" in issue for issue in issues)
    assert any("missing '**Owner-boundary:**'" in issue for issue in issues)
    assert any("missing '**Entry artifact:**'" in issue for issue in issues)
    assert any("missing '**Exit artifact:**'" in issue for issue in issues)


def test_cross_tool_doc_metadata_check_covers_baserender_junction_route(tmp_path: Path) -> None:
    route_path = tmp_path / "src" / "dnadesign" / "baserender" / "docs" / "integrations" / "junction.md"
    _write(
        route_path,
        "\n".join(
            [
                "# junction review integration",
                "",
                "**Type:** route",
                "",
                "Missing the route ownership and artifact boundary.",
            ]
        )
        + "\n",
    )

    issues = _find_cross_tool_doc_metadata_issues(tmp_path)

    assert any(str(route_path) in issue and "missing '**Plane:**'" in issue for issue in issues)
    assert any(str(route_path) in issue and "missing '**Owner-boundary:**'" in issue for issue in issues)
    assert any(str(route_path) in issue and "missing '**Entry artifact:**'" in issue for issue in issues)
    assert any(str(route_path) in issue and "missing '**Exit artifact:**'" in issue for issue in issues)


def test_cross_tool_doc_metadata_check_flags_missing_registry_fields(tmp_path: Path) -> None:
    _write(
        tmp_path / "docs" / "operations" / "orchestration/runbooks.md",
        "\n".join(
            [
                "## Orchestration runbooks",
                "",
                "**Type:** runbook",
                "**Plane:** control-plane",
                "**Owner-boundary:** ops",
                "**Entry artifact:** ops runbook intent",
                "**Exit artifact:** audit output",
                "**Owner:** maintainers",
                f"**Last verified:** {dt.date.today().isoformat()}",
            ]
        )
        + "\n",
    )

    issues = _find_cross_tool_doc_metadata_issues(tmp_path)

    assert any("missing '**Registry-id:**'" in issue for issue in issues)
    assert any("missing '**Summary:**'" in issue for issue in issues)
    assert any("missing '**Execution-kind:**'" in issue for issue in issues)
    assert any("missing '**Status-kind:**'" in issue for issue in issues)


def test_cross_tool_doc_metadata_check_accepts_expected_contract_values(tmp_path: Path) -> None:
    _write(
        tmp_path / "docs" / "operations" / "README.md",
        "\n".join(
            [
                "## Ops orchestration index",
                "",
                "**Type:** route",
                "**Plane:** control-plane",
                "**Owner-boundary:** ops",
                "**Entry artifact:** batch orchestration intent",
                "**Exit artifact:** authoritative ops contract",
                "**Owner:** maintainers",
                f"**Last verified:** {dt.date.today().isoformat()}",
            ]
        )
        + "\n",
    )

    issues = _find_cross_tool_doc_metadata_issues(tmp_path)

    assert issues == []


def test_cross_tool_doc_metadata_check_accepts_registry_fields_for_runbook_docs(tmp_path: Path) -> None:
    _write(
        tmp_path / "docs" / "operations" / "orchestration/runbooks.md",
        "\n".join(
            [
                "## Orchestration runbooks",
                "",
                "**Type:** runbook",
                "**Plane:** control-plane",
                "**Owner-boundary:** ops",
                "**Entry artifact:** ops runbook intent",
                "**Exit artifact:** audit output",
                "**Registry-id:** ops.control-plane.orchestration",
                "**Summary:** Deterministic batch orchestration contract.",
                "**Execution-kind:** executable",
                "**Status-kind:** ops-audit-json",
                "**Owner:** maintainers",
                f"**Last verified:** {dt.date.today().isoformat()}",
            ]
        )
        + "\n",
    )

    issues = _find_cross_tool_doc_metadata_issues(tmp_path)

    assert issues == []


def test_runbook_catalog_check_flags_missing_registered_doc_entries(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    orchestration_doc = tmp_path / "docs" / "operations" / "orchestration/runbooks.md"
    _write(
        orchestration_doc,
        "\n".join(
            [
                "## Orchestration runbooks",
                "",
                "**Type:** runbook",
                "**Plane:** control-plane",
                "**Owner-boundary:** ops",
                "**Entry artifact:** ops runbook intent",
                "**Exit artifact:** audit output",
                "**Registry-id:** ops.control-plane.orchestration",
                "**Summary:** Deterministic batch orchestration contract.",
                "**Execution-kind:** executable",
                "**Status-kind:** ops-audit-json",
                "**Owner:** maintainers",
                f"**Last verified:** {today}",
            ]
        )
        + "\n",
    )
    _write_registry_metadata(
        orchestration_doc,
        catalog_order=1,
        registry_id="ops.control-plane.orchestration",
        entry_type="runbook",
        plane="control-plane",
        owner_boundary="ops",
        entry_artifact="ops runbook intent",
        exit_artifact="audit output",
        summary="Deterministic batch orchestration contract.",
        execution_kind="executable",
        status_kind="ops-audit-json",
    )
    hpc_sync_doc = tmp_path / "src" / "dnadesign" / "usr" / "docs" / "operations" / "sync" / "hpc-agent-flow.md"
    _write(
        hpc_sync_doc,
        "\n".join(
            [
                "## USR HPC Sync Flow",
                "",
                "**Type:** runbook",
                "**Plane:** data-plane",
                "**Owner-boundary:** usr",
                "**Entry artifact:** sync intent",
                "**Exit artifact:** synchronized dataset",
                "**Registry-id:** usr.data-plane.hpc-sync",
                "**Summary:** HPC and local sync flow.",
                "**Execution-kind:** iterative",
                "**Status-kind:** usr-sync-audit",
                "**Owner:** maintainers",
                f"**Last verified:** {today}",
            ]
        )
        + "\n",
    )
    _write_generated_runbook_catalog_readme(
        tmp_path,
        glossary_rows=["| `ops-audit-json` | Control-plane audit payload. | Inspect the audit JSON. |"],
    )

    issues = _find_runbook_catalog_issues(tmp_path)

    assert any("missing registry metadata sidecar" in issue for issue in issues)
    assert any("src/dnadesign/usr/docs/operations/sync/hpc-agent-flow.registry.yaml" in issue for issue in issues)


def test_runbook_catalog_check_flags_metadata_drift_against_owner_local_doc(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    orchestration_doc = tmp_path / "docs" / "operations" / "orchestration/runbooks.md"
    _write(
        orchestration_doc,
        "\n".join(
            [
                "## Orchestration runbooks",
                "",
                "**Type:** runbook",
                "**Plane:** control-plane",
                "**Owner-boundary:** ops",
                "**Entry artifact:** ops runbook intent",
                "**Exit artifact:** audit output",
                "**Registry-id:** ops.control-plane.orchestration",
                "**Summary:** Deterministic control-plane runbook contract.",
                "**Execution-kind:** executable",
                "**Status-kind:** ops-audit-json",
                "**Owner:** maintainers",
                f"**Last verified:** {today}",
            ]
        )
        + "\n",
    )
    _write_registry_metadata(
        orchestration_doc,
        catalog_order=1,
        registry_id="ops.control-plane.orchestration",
        entry_type="runbook",
        plane="control-plane",
        owner_boundary="ops",
        entry_artifact="ops runbook intent",
        exit_artifact="audit output",
        summary="Deterministic batch orchestration contract.",
        execution_kind="executable",
        status_kind="ops-audit-json",
    )
    _write_generated_runbook_catalog_readme(
        tmp_path,
        glossary_rows=["| `ops-audit-json` | Control-plane audit payload. | Inspect the audit JSON. |"],
    )

    issues = _find_runbook_catalog_issues(tmp_path)

    assert any(
        "Summary for docs/operations/orchestration/runbooks.md must match owner-local metadata" in issue
        for issue in issues
    )


def test_runbook_catalog_check_accepts_matching_owner_local_metadata(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    summary = "Deterministic control-plane runbook contract."
    orchestration_doc = tmp_path / "docs" / "operations" / "orchestration/runbooks.md"
    _write(
        orchestration_doc,
        "\n".join(
            [
                "## Orchestration runbooks",
                "",
                "**Type:** runbook",
                "**Plane:** control-plane",
                "**Owner-boundary:** ops",
                "**Entry artifact:** ops runbook intent",
                "**Exit artifact:** audit output",
                "**Registry-id:** ops.control-plane.orchestration",
                f"**Summary:** {summary}",
                "**Execution-kind:** executable",
                "**Status-kind:** ops-audit-json",
                "**Owner:** maintainers",
                f"**Last verified:** {today}",
            ]
        )
        + "\n",
    )
    _write_registry_metadata(
        orchestration_doc,
        catalog_order=1,
        registry_id="ops.control-plane.orchestration",
        entry_type="runbook",
        plane="control-plane",
        owner_boundary="ops",
        entry_artifact="ops runbook intent",
        exit_artifact="audit output",
        summary=summary,
        execution_kind="executable",
        status_kind="ops-audit-json",
    )
    _write_generated_runbook_catalog_readme(
        tmp_path,
        glossary_rows=["| `ops-audit-json` | Control-plane audit payload. | Inspect the audit JSON. |"],
    )

    issues = _find_runbook_catalog_issues(tmp_path)

    assert issues == []


def test_runbook_catalog_check_flags_stale_generated_procedure_section(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    orchestration_doc = tmp_path / "docs" / "operations" / "orchestration/runbooks.md"
    _write(
        orchestration_doc,
        "\n".join(
            [
                "## Orchestration runbooks",
                "",
                "**Type:** runbook",
                "**Plane:** control-plane",
                "**Owner-boundary:** ops",
                "**Entry artifact:** ops runbook intent",
                "**Exit artifact:** audit output",
                "**Registry-id:** ops.control-plane.orchestration",
                "**Summary:** Deterministic control-plane runbook contract.",
                "**Execution-kind:** executable",
                "**Status-kind:** ops-audit-json",
                "**Owner:** maintainers",
                f"**Last verified:** {today}",
            ]
        )
        + "\n",
    )
    _write_registry_metadata(
        orchestration_doc,
        catalog_order=1,
        registry_id="ops.control-plane.orchestration",
        entry_type="runbook",
        plane="control-plane",
        owner_boundary="ops",
        entry_artifact="ops runbook intent",
        exit_artifact="audit output",
        summary="Deterministic control-plane runbook contract.",
        execution_kind="executable",
        status_kind="ops-audit-json",
    )
    _write_runbook_catalog_readme(
        tmp_path,
        procedure_section="stale manually edited procedure section",
        tool_source_section=_empty_tool_source_section(),
        glossary_rows=["| `ops-audit-json` | Control-plane audit payload. | Inspect the audit JSON. |"],
    )

    issues = _find_runbook_catalog_issues(tmp_path)

    assert any("cross-tool procedures section is stale" in issue for issue in issues)


def test_runbook_catalog_check_flags_stale_generated_tool_source_section(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    orchestration_doc = tmp_path / "docs" / "operations" / "orchestration/runbooks.md"
    ops_docs = tmp_path / "src" / "dnadesign" / "ops" / "docs" / "README.md"
    _write(
        orchestration_doc,
        "\n".join(
            [
                "## Orchestration runbooks",
                "",
                "**Type:** runbook",
                "**Plane:** control-plane",
                "**Owner-boundary:** ops",
                "**Entry artifact:** ops runbook intent",
                "**Exit artifact:** audit output",
                "**Registry-id:** ops.control-plane.orchestration",
                "**Summary:** Deterministic control-plane runbook contract.",
                "**Execution-kind:** executable",
                "**Status-kind:** ops-audit-json",
                "**Owner:** maintainers",
                f"**Last verified:** {today}",
            ]
        )
        + "\n",
    )
    _write(
        ops_docs,
        "\n".join(
            [
                "## ops docs",
                "",
                "**Owner:** maintainers",
                f"**Last verified:** {today}",
            ]
        )
        + "\n",
    )
    _write_registry_metadata(
        orchestration_doc,
        catalog_order=1,
        registry_id="ops.control-plane.orchestration",
        entry_type="runbook",
        plane="control-plane",
        owner_boundary="ops",
        entry_artifact="ops runbook intent",
        exit_artifact="audit output",
        summary="Deterministic control-plane runbook contract.",
        execution_kind="executable",
        status_kind="ops-audit-json",
    )
    _write_tool_source_metadata(
        ops_docs,
        catalog_order=1,
        tool="ops",
        summary="Control-plane docs.",
        keywords=["control-plane", "runbooks"],
    )
    _write_runbook_catalog_readme(
        tmp_path,
        procedure_section="_placeholder_",
        tool_source_section="_placeholder_",
        glossary_rows=["| `ops-audit-json` | Control-plane audit payload. | Inspect the audit JSON. |"],
    )
    _write_runbook_catalog_readme(
        tmp_path,
        procedure_section=render_catalog_procedure_section(load_runbook_catalog(repo_root=tmp_path)),
        tool_source_section="stale manually edited tool-source section",
        glossary_rows=["| `ops-audit-json` | Control-plane audit payload. | Inspect the audit JSON. |"],
    )

    issues = _find_runbook_catalog_issues(tmp_path)

    assert any("tool docs section is stale" in issue for issue in issues)


def test_runbook_catalog_check_flags_missing_progress_surface_glossary_entry(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    orchestration_doc = tmp_path / "docs" / "operations" / "orchestration/runbooks.md"
    _write(
        orchestration_doc,
        "\n".join(
            [
                "## Orchestration runbooks",
                "",
                "**Type:** runbook",
                "**Plane:** control-plane",
                "**Owner-boundary:** ops",
                "**Entry artifact:** ops runbook intent",
                "**Exit artifact:** audit output",
                "**Registry-id:** ops.control-plane.orchestration",
                "**Summary:** Deterministic control-plane runbook contract.",
                "**Execution-kind:** executable",
                "**Status-kind:** ops-audit-json",
                "**Owner:** maintainers",
                f"**Last verified:** {today}",
            ]
        )
        + "\n",
    )
    _write_registry_metadata(
        orchestration_doc,
        catalog_order=1,
        registry_id="ops.control-plane.orchestration",
        entry_type="runbook",
        plane="control-plane",
        owner_boundary="ops",
        entry_artifact="ops runbook intent",
        exit_artifact="audit output",
        summary="Deterministic control-plane runbook contract.",
        execution_kind="executable",
        status_kind="ops-audit-json",
    )
    _write_generated_runbook_catalog_readme(
        tmp_path,
        glossary_rows=["| `usr-sync-audit` | Sync drift review. | Inspect the sync audit. |"],
    )

    issues = _find_runbook_catalog_issues(tmp_path)

    assert any("missing status surface glossary entry for 'ops-audit-json'" in issue for issue in issues)


def test_runbook_catalog_check_uses_status_registry_inventory_for_glossary(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    orchestration_doc = tmp_path / "docs" / "operations" / "orchestration/runbooks.md"
    _write(
        orchestration_doc,
        "\n".join(
            [
                "## Orchestration runbooks",
                "",
                "**Type:** runbook",
                "**Plane:** control-plane",
                "**Owner-boundary:** ops",
                "**Entry artifact:** ops runbook intent",
                "**Exit artifact:** audit output",
                "**Registry-id:** ops.control-plane.orchestration",
                "**Summary:** Deterministic control-plane runbook contract.",
                "**Execution-kind:** executable",
                "**Status-kind:** ops-audit-json",
                "**Owner:** maintainers",
                f"**Last verified:** {today}",
            ]
        )
        + "\n",
    )
    _write_registry_metadata(
        orchestration_doc,
        catalog_order=1,
        registry_id="ops.control-plane.orchestration",
        entry_type="runbook",
        plane="control-plane",
        owner_boundary="ops",
        entry_artifact="ops runbook intent",
        exit_artifact="audit output",
        summary="Deterministic control-plane runbook contract.",
        execution_kind="executable",
        status_kind="ops-audit-json",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "ops" / "providers" / "builtin" / "status.registry.yaml",
        "\n".join(
            [
                "version: 1",
                "provider_id: builtin.ops",
                "entries:",
                "  - status_kind: ops-audit-json",
                "    owner_boundary: ops",
                "    observes_plane: control",
                "    provider_ref: dnadesign.ops.providers.builtin.status_provider:provide_ops_audit_status",
                "    description: Read one orchestration audit JSON.",
                "    surface_type: orchestration_audit",
                "    cost_class: cheap",
                "    summary_scope: workspace",
                "",
            ]
        ),
    )
    _write(
        tmp_path / "src" / "dnadesign" / "latentdna" / "ops" / "status.registry.yaml",
        "\n".join(
            [
                "version: 1",
                "provider_id: latentdna.workspace-status",
                "entries:",
                "  - status_kind: latentdna-workspace-snapshot",
                "    owner_boundary: latentdna",
                "    observes_plane: data",
                "    provider_ref: dnadesign.latentdna.ops.status_providers:provide_snapshot",
                "    description: Read one LatentDNA workspace snapshot.",
                "    surface_type: artifact_catalog",
                "    cost_class: cheap",
                "    summary_scope: workspace",
                "",
            ]
        ),
    )
    _write_generated_runbook_catalog_readme(
        tmp_path,
        glossary_rows=["| `ops-audit-json` | Control-plane audit payload. | Inspect the audit JSON. |"],
    )

    issues = _find_runbook_catalog_issues(tmp_path)

    assert any("missing status surface glossary entry for 'latentdna-workspace-snapshot'" in issue for issue in issues)
