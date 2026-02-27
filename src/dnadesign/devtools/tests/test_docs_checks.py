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
    _find_broken_links,
    _find_deprecated_docs_entrypoint_issues,
    _find_docs_root_heading_style_issues,
    _find_entrypoint_local_path_literal_issues,
    _find_root_docs_entrypoint_issues,
    _find_runbook_demo_snippet_issues,
    _find_tool_docs_metadata_issues,
    _find_tool_readme_banner_issues,
    _find_tool_readme_structure_issues,
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
    _write(tmp_path / "docs" / "index.md", "## Index\n")
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


def test_tool_readme_structure_check_requires_banner_as_first_non_empty_line(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "README.md",
        "## Alpha\n\n![Alpha banner](assets/alpha-banner.svg)\n\nShort narrative.\n\n[Docs](../../../docs/README.md)\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "assets" / "alpha-banner.svg",
        "<svg xmlns='http://www.w3.org/2000/svg'></svg>\n",
    )

    issues = _find_tool_readme_structure_issues(tmp_path)

    assert any("first non-empty line must be the banner image line" in issue for issue in issues)


def test_tool_readme_structure_check_rejects_heading_immediately_after_banner(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "README.md",
        "![Alpha banner](assets/alpha-banner.svg)\n\n## Alpha\n\n[Docs](../../../docs/README.md)\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "assets" / "alpha-banner.svg",
        "<svg xmlns='http://www.w3.org/2000/svg'></svg>\n",
    )

    issues = _find_tool_readme_structure_issues(tmp_path)

    assert any("line after the banner must be narrative text" in issue for issue in issues)


def test_tool_readme_structure_check_requires_top_level_markdown_doc_link(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "README.md",
        "![Alpha banner](assets/alpha-banner.svg)\n\nShort narrative.\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "assets" / "alpha-banner.svg",
        "<svg xmlns='http://www.w3.org/2000/svg'></svg>\n",
    )

    issues = _find_tool_readme_structure_issues(tmp_path)

    assert any("top section must include a local markdown link" in issue for issue in issues)


def test_tool_readme_structure_check_accepts_banner_narrative_and_docs_link(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "README.md",
        "\n".join(
            [
                "![Alpha banner](assets/alpha-banner.svg)",
                "",
                "Short narrative overview.",
                "",
                "See [docs index](../../../docs/README.md) for workflows and references.",
                "",
                "## Usage",
                "",
                "Run alpha.",
                "",
            ]
        ),
    )
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "assets" / "alpha-banner.svg",
        "<svg xmlns='http://www.w3.org/2000/svg'></svg>\n",
    )

    issues = _find_tool_readme_structure_issues(tmp_path)

    assert issues == []


def test_root_docs_entrypoint_check_requires_docs_index_link(tmp_path: Path) -> None:
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "![dnadesign banner](assets/dnadesign-banner.svg)",
                "",
                "Use the docs index.",
                "",
            ]
        ),
    )

    issues = _find_root_docs_entrypoint_issues(tmp_path)

    assert any("must include a markdown link to docs/README.md" in issue for issue in issues)


def test_root_docs_entrypoint_check_accepts_banner_with_docs_index_link(tmp_path: Path) -> None:
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "![dnadesign banner](assets/dnadesign-banner.svg)",
                "",
                "[Docs index](docs/README.md)",
                "",
            ]
        ),
    )

    issues = _find_root_docs_entrypoint_issues(tmp_path)

    assert issues == []


def test_root_docs_entrypoint_check_rejects_plain_text_paths_without_links(tmp_path: Path) -> None:
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "![dnadesign banner](assets/dnadesign-banner.svg)",
                "",
                "Use docs/README.md as the docs entrypoint.",
                "",
            ]
        ),
    )

    issues = _find_root_docs_entrypoint_issues(tmp_path)

    assert any("must include a markdown link to docs/README.md" in issue for issue in issues)


def test_deprecated_docs_entrypoint_check_flags_start_here_file(tmp_path: Path) -> None:
    _write(tmp_path / "docs" / "start-here.md", "## Start Here\n")

    issues = _find_deprecated_docs_entrypoint_issues(tmp_path)

    assert any("docs/start-here.md" in issue and "deprecated" in issue for issue in issues)


def test_deprecated_docs_entrypoint_check_flags_start_here_links(tmp_path: Path) -> None:
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "![dnadesign banner](assets/dnadesign-banner.svg)",
                "",
                "[Start here](docs/start-here.md)",
                "",
            ]
        ),
    )

    issues = _find_deprecated_docs_entrypoint_issues(tmp_path)

    assert any("must not link to docs/start-here.md" in issue for issue in issues)


def test_entrypoint_local_path_link_check_flags_local_literal_paths(tmp_path: Path) -> None:
    _write(tmp_path / "docs" / "installation.md", "## Installation\n")
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "![dnadesign banner](assets/dnadesign-banner.svg)",
                "",
                "[Docs index](docs/README.md)",
                "",
                "Read `docs/installation.md` before running commands.",
                "",
            ]
        ),
    )

    issues = _find_entrypoint_local_path_literal_issues(tmp_path)

    assert any("local path literal" in issue and "docs/installation.md" in issue for issue in issues)


def test_entrypoint_local_path_link_check_allows_hyperlinked_local_paths(tmp_path: Path) -> None:
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "![dnadesign banner](assets/dnadesign-banner.svg)",
                "",
                "[Docs index](docs/README.md)",
                "",
                "Read [installation guide](docs/installation.md) before running commands.",
                "",
            ]
        ),
    )

    issues = _find_entrypoint_local_path_literal_issues(tmp_path)

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


def test_main_fails_when_root_sor_doc_last_verified_is_stale(tmp_path: Path) -> None:
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        "# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** 2020-01-01\n",
    )

    rc = main(["--repo-root", str(tmp_path), "--max-sor-age-days", "30"])
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


def test_main_fails_when_start_here_doc_is_present(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(
        tmp_path / "docs" / "README.md",
        f"## Documentation Index\n\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "docs" / "start-here.md", "## Start Here\n\nPick a path.\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_docs_index_last_verified_is_stale(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(
        tmp_path / "docs" / "README.md",
        "## Documentation Index\n\n**Owner:** maintainers\n**Last verified:** 2020-01-01\n",
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


def test_main_passes_for_valid_links(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "## x\n\n[guide](./guide.md)\n[#anchor](#x)\n[site](https://example.com)\n")
    _write(tmp_path / "docs" / "guide.md", "## Guide\n")
    _write(tmp_path / "README.md", "[docs](docs/index.md)\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n[docs](docs/guide.md)\n",
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 0


def test_main_fails_when_readme_tool_catalog_missing_repo_tool(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "aligner" / "__init__.py", "")
    _write(tmp_path / "src" / "dnadesign" / "notify" / "__init__.py", "")
    _write(
        tmp_path / "src" / "dnadesign" / "aligner" / "README.md",
        "![aligner banner](assets/aligner-banner.svg)\n\nAligner narrative.\n\n[Docs](../../../docs/README.md)\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "notify" / "README.md",
        "![notify banner](assets/notify-banner.svg)\n\nNotify narrative.\n\n[Docs](../../../docs/README.md)\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "aligner" / "assets" / "aligner-banner.svg",
        "<svg xmlns='http://www.w3.org/2000/svg'></svg>\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "notify" / "assets" / "notify-banner.svg",
        "<svg xmlns='http://www.w3.org/2000/svg'></svg>\n",
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
                "| [**aligner**](src/dnadesign/aligner/README.md) | alignment | "
                "[Codecov](https://codecov.io/gh/example/repo?component=aligner) |",
                "",
            ]
        ),
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_readme_tool_catalog_row_has_too_few_columns(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "## Index\n")
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
                "| [**aligner**](src/dnadesign/aligner/README.md) |",
                "",
            ]
        ),
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_readme_tool_catalog_missing_coverage_column(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "## Index\n")
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
                "| [**aligner**](src/dnadesign/aligner/README.md) | alignment |",
                "",
            ]
        ),
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_passes_when_readme_tool_catalog_matches_repo_tools(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "aligner" / "__init__.py", "")
    _write(tmp_path / "src" / "dnadesign" / "notify" / "__init__.py", "")
    _write(
        tmp_path / "src" / "dnadesign" / "aligner" / "README.md",
        "![aligner banner](assets/aligner-banner.svg)\n\nAligner narrative.\n\n[Docs](../../../docs/README.md)\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "notify" / "README.md",
        "![notify banner](assets/notify-banner.svg)\n\nNotify narrative.\n\n[Docs](../../../docs/README.md)\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "aligner" / "assets" / "aligner-banner.svg",
        "<svg xmlns='http://www.w3.org/2000/svg'></svg>\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "notify" / "assets" / "notify-banner.svg",
        "<svg xmlns='http://www.w3.org/2000/svg'></svg>\n",
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
                "| [**aligner**](src/dnadesign/aligner/README.md) | alignment | "
                "[Codecov](https://codecov.io/gh/example/repo?component=aligner) |",
                "| [**notify**](src/dnadesign/notify/README.md) | notifications | "
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
    _write(tmp_path / "docs" / "index.md", "## Index\n")
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
    _write(tmp_path / "docs" / "index.md", "## Index\n")
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
                "| [**aligner**](src/dnadesign/aligner/README.md) | alignment | "
                "[Codecov](https://codecov.io/gh/example/repo?component=aligner) |",
                "",
            ]
        ),
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_readme_tool_coverage_link_component_mismatches_tool(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "## Index\n")
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
                "| [**aligner**](src/dnadesign/aligner/README.md) | alignment | "
                "[Codecov](https://codecov.io/gh/example/repo?component=notify) |",
                "",
            ]
        ),
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_codecov_components_do_not_cover_repo_tools(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "aligner" / "__init__.py", "")
    _write(tmp_path / "src" / "dnadesign" / "notify" / "__init__.py", "")
    _write(
        tmp_path / "src" / "dnadesign" / "aligner" / "README.md",
        "![aligner banner](assets/aligner-banner.svg)\n\nAligner narrative.\n\n[Docs](../../../docs/README.md)\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "notify" / "README.md",
        "![notify banner](assets/notify-banner.svg)\n\nNotify narrative.\n\n[Docs](../../../docs/README.md)\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "aligner" / "assets" / "aligner-banner.svg",
        "<svg xmlns='http://www.w3.org/2000/svg'></svg>\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "notify" / "assets" / "notify-banner.svg",
        "<svg xmlns='http://www.w3.org/2000/svg'></svg>\n",
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
                "| [**aligner**](src/dnadesign/aligner/README.md) | alignment | "
                "[Codecov](https://codecov.io/gh/example/repo?component=aligner) |",
                "| [**notify**](src/dnadesign/notify/README.md) | notifications | "
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
    _write(tmp_path / "docs" / "index.md", "## Index\n")
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
                "| [**aligner**](src/dnadesign/aligner/README.md) | alignment | "
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
    _write(tmp_path / "docs" / "index.md", "## Index\n")
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
                "| [**densegen**](src/dnadesign/densegen/README.md) | densegen tool | "
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
    _write(tmp_path / "docs" / "index.md", "## Index\n")
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
                "| [**densegen**](src/dnadesign/densegen/README.md) | densegen tool | "
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
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "aligner" / "__init__.py", "")
    _write(tmp_path / "src" / "dnadesign" / "notify" / "__init__.py", "")
    _write(
        tmp_path / "src" / "dnadesign" / "aligner" / "README.md",
        "![aligner banner](assets/aligner-banner.svg)\n\nAligner narrative.\n\n[Docs](../../../docs/README.md)\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "notify" / "README.md",
        "![notify banner](assets/notify-banner.svg)\n\nNotify narrative.\n\n[Docs](../../../docs/README.md)\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "aligner" / "assets" / "aligner-banner.svg",
        "<svg xmlns='http://www.w3.org/2000/svg'></svg>\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "notify" / "assets" / "notify-banner.svg",
        "<svg xmlns='http://www.w3.org/2000/svg'></svg>\n",
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
                "| [**aligner**](src/dnadesign/aligner/README.md) | alignment | "
                "[Codecov](https://codecov.io/gh/example/repo?component=aligner) |",
                "| [**notify**](src/dnadesign/notify/README.md) | notifications | "
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


def test_docs_root_heading_style_check_flags_level_one_or_repeated_level_two_headings(tmp_path: Path) -> None:
    _write(
        tmp_path / "docs" / "README.md",
        "\n".join(
            [
                "# Documentation Index",
                "",
                "## One",
                "## Two",
                "",
            ]
        ),
    )

    issues = _find_docs_root_heading_style_issues(tmp_path)

    assert any("must start with '## '" in issue for issue in issues)
    assert any("use a single level-2 heading" in issue for issue in issues)


def test_docs_root_heading_style_check_accepts_level_two_title_and_lower_sections(tmp_path: Path) -> None:
    _write(
        tmp_path / "docs" / "README.md",
        "\n".join(
            [
                "## Documentation Index",
                "",
                "### Use this index",
                "",
                "Text.",
                "",
            ]
        ),
    )

    issues = _find_docs_root_heading_style_issues(tmp_path)

    assert issues == []


def test_broken_links_check_flags_missing_markdown_anchor(tmp_path: Path) -> None:
    source = tmp_path / "docs" / "source.md"
    target = tmp_path / "docs" / "target.md"
    _write(source, "[missing](./target.md#not-here)\n")
    _write(target, "## Present Heading\n")

    broken = _find_broken_links([source, target])

    assert any("anchor 'not-here'" in issue_link for _, issue_link in broken)


def test_tool_docs_metadata_check_flags_missing_owner_and_last_verified(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "alpha" / "__init__.py", "")
    _write(tmp_path / "src" / "dnadesign" / "alpha" / "docs" / "README.md", "## Alpha docs\n")

    issues = _find_tool_docs_metadata_issues(tmp_path, max_age_days=90)

    assert any("missing '**Owner:**' metadata field." in issue for issue in issues)
    assert any("missing '**Last verified:**' metadata field." in issue for issue in issues)


def test_tool_docs_metadata_check_accepts_valid_owner_and_last_verified(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "src" / "dnadesign" / "alpha" / "__init__.py", "")
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "docs" / "README.md",
        f"## Alpha docs\n\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )

    issues = _find_tool_docs_metadata_issues(tmp_path, max_age_days=90)

    assert issues == []
