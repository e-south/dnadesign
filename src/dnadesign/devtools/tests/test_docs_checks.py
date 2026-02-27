"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/test_docs_checks.py

Tests for docs naming/link validation checks used in CI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from dnadesign.devtools.docs_checks import (
    _find_runbook_demo_snippet_issues,
    _find_tool_readme_banner_issues,
    main,
)


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_main_fails_when_docs_directory_is_missing(tmp_path: Path) -> None:
    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_for_non_kebab_docs_filename(tmp_path: Path) -> None:
    _write(tmp_path / "docs" / "bad_name.md", "# Bad\n")

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_for_broken_relative_link(tmp_path: Path) -> None:
    _write(tmp_path / "docs" / "index.md", "[missing](./nope.md)\n")

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_for_broken_relative_link_in_root_sor_doc(tmp_path: Path) -> None:
    _write(tmp_path / "docs" / "index.md", "# Index\n")
    _write(tmp_path / "ARCHITECTURE.md", "[broken](docs/missing.md)\n")

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_tool_readme_banner_check_flags_missing_or_non_svg_banners(tmp_path: Path) -> None:
    tool_root = tmp_path / "src" / "dnadesign"
    _write(tool_root / "alpha" / "README.md", "## Alpha\n\nNo banner.\n")
    _write(tool_root / "beta" / "README.md", "## Beta\n\n![Beta banner](images/beta-banner.png)\n")

    issues = _find_tool_readme_banner_issues(tmp_path)

    assert any("alpha/README.md" in issue and "missing top banner image" in issue for issue in issues)
    assert any("beta/README.md" in issue and "must target a local .svg asset" in issue for issue in issues)


def test_tool_readme_banner_check_accepts_existing_local_svg_banner(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "README.md",
        "## Alpha\n\n![Alpha banner](assets/alpha-banner.svg)\n\nCompact subtitle.\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "assets" / "alpha-banner.svg",
        "<svg xmlns='http://www.w3.org/2000/svg'></svg>\n",
    )

    issues = _find_tool_readme_banner_issues(tmp_path)

    assert issues == []


def test_runbook_demo_snippet_check_flags_missing_shell_and_yaml_comments(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "docs" / "tutorials" / "demo.md",
        "\n".join(
            [
                "## Demo",
                "",
                "```bash",
                "uv run alpha do-work",
                "```",
                "",
                "```yaml",
                "alpha:",
                "  enabled: true",
                "```",
                "",
            ]
        ),
    )

    issues = _find_runbook_demo_snippet_issues(tmp_path)

    assert any("command in shell block needs an explanatory comment" in issue for issue in issues)
    assert any("yaml key/value in runbook/demo snippets needs a right-side inline comment" in issue for issue in issues)


def test_runbook_demo_snippet_check_accepts_commented_shell_and_yaml_blocks(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "docs" / "tutorials" / "demo.md",
        "\n".join(
            [
                "## Demo",
                "",
                "```bash",
                "# Run the demo command.",
                "uv run alpha do-work",
                "```",
                "",
                "```yaml",
                "alpha:",
                "  enabled: true  # Toggle demo mode.",
                "```",
                "",
            ]
        ),
    )

    issues = _find_runbook_demo_snippet_issues(tmp_path)

    assert issues == []


def test_main_fails_when_root_sor_doc_missing_required_metadata(tmp_path: Path) -> None:
    _write(tmp_path / "docs" / "index.md", "# Index\n")
    _write(tmp_path / "ARCHITECTURE.md", "# ARCHITECTURE\n")

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_root_sor_doc_missing_type_metadata(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "# Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_root_sor_doc_last_verified_is_stale(tmp_path: Path) -> None:
    _write(tmp_path / "docs" / "index.md", "# Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        "# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** 2020-01-01\n",
    )

    rc = main(["--repo-root", str(tmp_path), "--max-sor-age-days", "30"])
    assert rc == 1


def test_main_fails_when_docs_index_missing_required_metadata(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "README.md", "# Documentation Index\n")
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
        f"# Documentation Index\n\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 0


def test_main_fails_when_docs_index_last_verified_is_stale(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(
        tmp_path / "docs" / "README.md",
        "# Documentation Index\n\n**Owner:** maintainers\n**Last verified:** 2020-01-01\n",
    )
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )

    rc = main(["--repo-root", str(tmp_path), "--max-sor-age-days", "30"])
    assert rc == 1


def test_main_fails_when_selected_runbook_missing_required_metadata(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(
        tmp_path / "docs" / "installation.md",
        "## Installation\n\nRun setup.\n",
    )
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_selected_runbook_last_verified_is_stale(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(
        tmp_path / "docs" / "installation.md",
        "## Installation\n\n**Owner:** maintainers\n**Last verified:** 2020-01-01\n",
    )
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )

    rc = main(["--repo-root", str(tmp_path), "--max-sor-age-days", "30"])
    assert rc == 1


def test_main_passes_when_selected_runbook_has_required_metadata(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(
        tmp_path / "docs" / "installation.md",
        f"## Installation\n\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 0


def test_main_fails_when_exec_plan_missing_required_metadata(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "# Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "docs" / "exec-plans" / "active" / "example.md", "# Exec plan\n")

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_exec_plan_missing_required_living_sections(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "# Index\n")
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
    _write(tmp_path / "docs" / "index.md", "# Index\n")
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
    assert rc == 0


def test_main_fails_when_exec_plan_progress_has_no_checklist_items(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "# Index\n")
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
    _write(tmp_path / "docs" / "index.md", "# Index\n")
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
    _write(tmp_path / "docs" / "index.md", "# Index\n")
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


def test_main_passes_for_valid_links(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "[guide](./guide.md)\n[#anchor](#x)\n[site](https://example.com)\n")
    _write(tmp_path / "docs" / "guide.md", "# Guide\n")
    _write(tmp_path / "README.md", "[docs](docs/index.md)\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n[docs](docs/guide.md)\n",
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 0


def test_main_fails_when_readme_tool_catalog_missing_repo_tool(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "# Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "aligner" / "__init__.py", "")
    _write(tmp_path / "src" / "dnadesign" / "notify" / "__init__.py", "")
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# dnadesign",
                "",
                "## Available tools",
                "",
                "| Tool | Description | Coverage |",
                "| --- | --- | --- |",
                "| [**aligner**](src/dnadesign/aligner) | alignment | "
                "[Codecov](https://codecov.io/gh/example/repo?component=aligner) |",
                "",
            ]
        ),
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_readme_tool_catalog_row_has_too_few_columns(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "# Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "aligner" / "__init__.py", "")
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# dnadesign",
                "",
                "## Available tools",
                "",
                "| Tool | Description | Coverage |",
                "| --- | --- | --- |",
                "| [**aligner**](src/dnadesign/aligner) |",
                "",
            ]
        ),
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_readme_tool_catalog_missing_coverage_column(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "# Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "aligner" / "__init__.py", "")
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# dnadesign",
                "",
                "## Available tools",
                "",
                "| Tool | Description |",
                "| --- | --- |",
                "| [**aligner**](src/dnadesign/aligner) | alignment |",
                "",
            ]
        ),
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_passes_when_readme_tool_catalog_matches_repo_tools(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "# Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "aligner" / "__init__.py", "")
    _write(tmp_path / "src" / "dnadesign" / "notify" / "__init__.py", "")
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# dnadesign",
                "",
                "## Available tools",
                "",
                "| Tool | Description | Coverage |",
                "| --- | --- | --- |",
                "| [**aligner**](src/dnadesign/aligner) | alignment | "
                "[Codecov](https://codecov.io/gh/example/repo?component=aligner) |",
                "| [**notify**](src/dnadesign/notify) | notifications | "
                "[Codecov](https://codecov.io/gh/example/repo?component=notify) |",
                "",
            ]
        ),
    )
    _write(
        tmp_path / "codecov.yml",
        "\n".join(
            [
                "component_management:",
                "  default_rules:",
                "    statuses:",
                "      - type: project",
                "        target: auto",
                "        threshold: 0.5%",
                "        if_ci_failed: error",
                "        if_not_found: failure",
                "  individual_components:",
                "    - component_id: aligner",
                "      name: aligner",
                "      paths:",
                "        - src/dnadesign/aligner/**",
                "    - component_id: notify",
                "      name: notify",
                "      paths:",
                "        - src/dnadesign/notify/**",
                "",
            ]
        ),
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 0


def test_main_fails_when_readme_tool_link_does_not_match_expected_path(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "# Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "aligner" / "__init__.py", "")
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# dnadesign",
                "",
                "## Available tools",
                "",
                "| Tool | Description | Coverage |",
                "| --- | --- | --- |",
                "| [**aligner**](src/dnadesign/aligner/docs) | alignment | "
                "[Codecov](https://codecov.io/gh/example/repo?component=aligner) |",
                "",
            ]
        ),
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_readme_tool_link_target_directory_is_missing(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "# Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# dnadesign",
                "",
                "## Available tools",
                "",
                "| Tool | Description | Coverage |",
                "| --- | --- | --- |",
                "| [**aligner**](src/dnadesign/aligner) | alignment | "
                "[Codecov](https://codecov.io/gh/example/repo?component=aligner) |",
                "",
            ]
        ),
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_readme_tool_coverage_link_component_mismatches_tool(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "# Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "aligner" / "__init__.py", "")
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# dnadesign",
                "",
                "## Available tools",
                "",
                "| Tool | Description | Coverage |",
                "| --- | --- | --- |",
                "| [**aligner**](src/dnadesign/aligner) | alignment | "
                "[Codecov](https://codecov.io/gh/example/repo?component=notify) |",
                "",
            ]
        ),
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_codecov_components_do_not_cover_repo_tools(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "# Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "aligner" / "__init__.py", "")
    _write(tmp_path / "src" / "dnadesign" / "notify" / "__init__.py", "")
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# dnadesign",
                "",
                "## Available tools",
                "",
                "| Tool | Description | Coverage |",
                "| --- | --- | --- |",
                "| [**aligner**](src/dnadesign/aligner) | alignment | "
                "[Codecov](https://codecov.io/gh/example/repo?component=aligner) |",
                "| [**notify**](src/dnadesign/notify) | notifications | "
                "[Codecov](https://codecov.io/gh/example/repo?component=notify) |",
                "",
            ]
        ),
    )
    _write(
        tmp_path / "codecov.yml",
        "\n".join(
            [
                "component_management:",
                "  default_rules:",
                "    statuses:",
                "      - type: project",
                "        target: auto",
                "        threshold: 0.5%",
                "        if_ci_failed: error",
                "        if_not_found: failure",
                "  individual_components:",
                "    - component_id: aligner",
                "      name: aligner",
                "      paths:",
                "        - src/dnadesign/aligner/**",
                "",
            ]
        ),
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_codecov_component_default_rules_are_missing(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "# Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "aligner" / "__init__.py", "")
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# dnadesign",
                "",
                "## Available tools",
                "",
                "| Tool | Description | Coverage |",
                "| --- | --- | --- |",
                "| [**aligner**](src/dnadesign/aligner) | alignment | "
                "[Codecov](https://codecov.io/gh/example/repo?component=aligner) |",
                "",
            ]
        ),
    )
    _write(
        tmp_path / "codecov.yml",
        "\n".join(
            [
                "component_management:",
                "  individual_components:",
                "    - component_id: aligner",
                "      name: aligner",
                "      paths:",
                "        - src/dnadesign/aligner/**",
                "",
            ]
        ),
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_public_interface_docs_use_absolute_paths(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "# Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "densegen" / "__init__.py", "")
    _write(
        tmp_path / "src" / "dnadesign" / "densegen" / "docs" / "tutorials" / "demo.md",
        "Use `uv run cruncher catalog export-densegen --densegen-workspace /tmp/demo`.\n",
    )
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# dnadesign",
                "",
                "## Available tools",
                "",
                "| Tool | Description | Coverage |",
                "| --- | --- | --- |",
                "| [**densegen**](src/dnadesign/densegen) | densegen tool | "
                "[Codecov](https://codecov.io/gh/example/repo?component=densegen) |",
                "",
            ]
        ),
    )
    _write(
        tmp_path / "codecov.yml",
        "\n".join(
            [
                "component_management:",
                "  default_rules:",
                "    statuses:",
                "      - type: project",
                "        target: auto",
                "        threshold: 0.5%",
                "        if_ci_failed: error",
                "        if_not_found: failure",
                "  individual_components:",
                "    - component_id: densegen",
                "      name: densegen",
                "      paths:",
                "        - src/dnadesign/densegen/**",
                "",
            ]
        ),
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_public_interface_docs_use_internal_source_inreach(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "# Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "densegen" / "__init__.py", "")
    _write(
        tmp_path / "src" / "dnadesign" / "densegen" / "docs" / "howto" / "handoff.md",
        "Call `python -m dnadesign.cruncher.src.cli.app` directly.\n",
    )
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# dnadesign",
                "",
                "## Available tools",
                "",
                "| Tool | Description | Coverage |",
                "| --- | --- | --- |",
                "| [**densegen**](src/dnadesign/densegen) | densegen tool | "
                "[Codecov](https://codecov.io/gh/example/repo?component=densegen) |",
                "",
            ]
        ),
    )
    _write(
        tmp_path / "codecov.yml",
        "\n".join(
            [
                "component_management:",
                "  default_rules:",
                "    statuses:",
                "      - type: project",
                "        target: auto",
                "        threshold: 0.5%",
                "        if_ci_failed: error",
                "        if_not_found: failure",
                "  individual_components:",
                "    - component_id: densegen",
                "      name: densegen",
                "      paths:",
                "        - src/dnadesign/densegen/**",
                "",
            ]
        ),
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_passes_when_codecov_components_match_repo_tools(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "# Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "aligner" / "__init__.py", "")
    _write(tmp_path / "src" / "dnadesign" / "notify" / "__init__.py", "")
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# dnadesign",
                "",
                "## Available tools",
                "",
                "| Tool | Description | Coverage |",
                "| --- | --- | --- |",
                "| [**aligner**](src/dnadesign/aligner) | alignment | "
                "[Codecov](https://codecov.io/gh/example/repo?component=aligner) |",
                "| [**notify**](src/dnadesign/notify) | notifications | "
                "[Codecov](https://codecov.io/gh/example/repo?component=notify) |",
                "",
            ]
        ),
    )
    _write(
        tmp_path / "codecov.yml",
        "\n".join(
            [
                "component_management:",
                "  default_rules:",
                "    statuses:",
                "      - type: project",
                "        target: auto",
                "        threshold: 0.5%",
                "        if_ci_failed: error",
                "        if_not_found: failure",
                "  individual_components:",
                "    - component_id: aligner",
                "      name: aligner",
                "      paths:",
                "        - src/dnadesign/aligner/**",
                "    - component_id: notify",
                "      name: notify",
                "      paths:",
                "        - src/dnadesign/notify/**",
                "",
            ]
        ),
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 0
