"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/docs/test_public_surface_contracts.py

Tests for documentation public surface contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from dnadesign.devtools.docs.checks import (
    _find_agents_path_reference_issues,
    _find_densegen_disallowed_term_issues,
    _find_deprecated_docs_entrypoint_issues,
    _find_docs_root_heading_style_issues,
    _find_entrypoint_local_path_literal_issues,
    _find_public_interface_doc_contract_issues,
    _find_readme_tool_catalog_issues,
    _find_root_docs_entrypoint_issues,
    _find_tool_readme_structure_issues,
    main,
)
from dnadesign.devtools.tests.docs.check_test_support import (
    VALID_TOOL_BANNER_SVG,
    _write,
)


def test_tool_readme_structure_check_requires_top_level_markdown_doc_link(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "README.md",
        "![Alpha banner](assets/alpha-banner.svg)\n\nShort narrative.\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "assets" / "alpha-banner.svg",
        VALID_TOOL_BANNER_SVG,
    )

    issues = _find_tool_readme_structure_issues(tmp_path)

    assert any("top section must include a local markdown link" in issue for issue in issues)


def test_tool_readme_structure_check_rejects_multi_paragraph_intro(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "README.md",
        "\n".join(
            [
                "![Alpha banner](assets/alpha-banner.svg)",
                "",
                "First narrative paragraph.",
                "",
                "Second narrative paragraph belongs in deeper docs.",
                "",
                "## Documentation",
                "",
                "[Alpha docs](docs/README.md)",
                "",
            ]
        ),
    )
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "assets" / "alpha-banner.svg",
        VALID_TOOL_BANNER_SVG,
    )
    _write(tmp_path / "src" / "dnadesign" / "alpha" / "docs" / "README.md", "## Alpha docs\n")

    issues = _find_tool_readme_structure_issues(tmp_path)

    assert any("intro after the banner must be one paragraph" in issue for issue in issues)


def test_tool_readme_structure_check_rejects_self_referential_intro(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "README.md",
        "\n".join(
            [
                "![Alpha banner](assets/alpha-banner.svg)",
                "",
                "Alpha is the analysis package in `dnadesign`.",
                "",
                "## Documentation",
                "",
                "[Alpha docs](docs/README.md)",
                "",
            ]
        ),
    )
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "assets" / "alpha-banner.svg",
        VALID_TOOL_BANNER_SVG,
    )
    _write(tmp_path / "src" / "dnadesign" / "alpha" / "docs" / "README.md", "## Alpha docs\n")

    issues = _find_tool_readme_structure_issues(tmp_path)

    assert any("avoid self-referential package/layer-in-dnadesign wording" in issue for issue in issues)


def test_tool_readme_structure_check_requires_documentation_heading(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "README.md",
        "\n".join(
            [
                "![Alpha banner](assets/alpha-banner.svg)",
                "",
                "Alpha scores short sequence examples.",
                "",
                "## Start here",
                "",
                "[Alpha docs](docs/README.md)",
                "",
            ]
        ),
    )
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "assets" / "alpha-banner.svg",
        VALID_TOOL_BANNER_SVG,
    )
    _write(tmp_path / "src" / "dnadesign" / "alpha" / "docs" / "README.md", "## Alpha docs\n")

    issues = _find_tool_readme_structure_issues(tmp_path)

    assert any("first heading after the intro must be '## Documentation'" in issue for issue in issues)


def test_tool_readme_structure_check_rejects_overlong_tool_readmes(tmp_path: Path) -> None:
    body_lines = [
        "![Alpha banner](assets/alpha-banner.svg)",
        "",
        "Short narrative.",
        "",
        "## Documentation",
        "",
        "[Docs](docs/README.md)",
    ]
    body_lines.extend(f"Extra line {idx}." for idx in range(40))
    _write(tmp_path / "src" / "dnadesign" / "alpha" / "README.md", "\n".join(body_lines) + "\n")
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "assets" / "alpha-banner.svg",
        VALID_TOOL_BANNER_SVG,
    )
    _write(tmp_path / "src" / "dnadesign" / "alpha" / "docs" / "README.md", "## Alpha docs\n")

    issues = _find_tool_readme_structure_issues(tmp_path)

    assert any("top-level tool README has" in issue for issue in issues)


def test_tool_readme_structure_check_requires_docs_index_first_when_present(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "README.md",
        "\n".join(
            [
                "![Alpha banner](assets/alpha-banner.svg)",
                "",
                "Short narrative.",
                "",
                "## Documentation",
                "",
                "[Repository docs](../../../docs/README.md)",
                "[Alpha docs](docs/README.md)",
                "",
            ]
        ),
    )
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "assets" / "alpha-banner.svg",
        VALID_TOOL_BANNER_SVG,
    )
    _write(tmp_path / "src" / "dnadesign" / "alpha" / "docs" / "README.md", "## Alpha docs\n")

    monkeypatch.chdir(tmp_path)
    issues = _find_tool_readme_structure_issues(Path("."))

    assert any("first local markdown link must point to the tool docs index" in issue for issue in issues)


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
    _write(tmp_path / "docs" / "setup" / "installation.md", "## Installation\n")
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "![dnadesign banner](assets/dnadesign-banner.svg)",
                "",
                "[Docs index](docs/README.md)",
                "",
                "Read `docs/setup/installation.md` before running commands.",
                "",
            ]
        ),
    )

    issues = _find_entrypoint_local_path_literal_issues(tmp_path)

    assert any("local path literal" in issue and "docs/setup/installation.md" in issue for issue in issues)


def test_entrypoint_local_path_link_check_allows_hyperlinked_local_paths(tmp_path: Path) -> None:
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "![dnadesign banner](assets/dnadesign-banner.svg)",
                "",
                "[Docs index](docs/README.md)",
                "",
                "Read [installation guide](docs/setup/installation.md) before running commands.",
                "",
            ]
        ),
    )

    issues = _find_entrypoint_local_path_literal_issues(tmp_path)

    assert issues == []


def test_agents_path_reference_check_flags_missing_scoped_paths(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "cruncher" / "AGENTS.md",
        "- Default config: `src/dnadesign/cruncher/workspaces/missing/config.yaml`\n",
    )

    issues = _find_agents_path_reference_issues(tmp_path)

    assert any("src/dnadesign/cruncher/workspaces/missing/config.yaml" in issue for issue in issues)


def test_agents_path_reference_check_allows_existing_and_non_path_spans(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "cruncher" / "workspaces" / "demo" / "configs" / "config.yaml", "{}\n")
    _write(
        tmp_path / "src" / "dnadesign" / "cruncher" / "AGENTS.md",
        "\n".join(
            [
                "- Default config: `workspaces/demo/configs/config.yaml`",
                "- Repo-root config: `src/dnadesign/cruncher/workspaces/demo/configs/config.yaml`",
                "- This repo intentionally has no `./scripts/agent-verify`.",
                "- Use command `uv run cruncher --help`.",
                "- Template path: `workspaces/<id>/configs/config.yaml`.",
                "",
            ]
        ),
    )

    issues = _find_agents_path_reference_issues(tmp_path)

    assert issues == []


def test_agents_path_reference_check_ignores_local_worktrees(tmp_path: Path) -> None:
    _write(
        tmp_path / ".worktrees" / "feature" / "src" / "dnadesign" / "cruncher" / "AGENTS.md",
        "- Default config: `src/dnadesign/cruncher/workspaces/missing/config.yaml`\n",
    )

    issues = _find_agents_path_reference_issues(tmp_path)

    assert issues == []


def test_densegen_docs_language_check_flags_canonical_term(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "densegen" / "README.md", "This is the canonical densegen guide.\n")

    issues = _find_densegen_disallowed_term_issues(tmp_path)

    assert any("term 'canonical'" in issue for issue in issues)


def test_densegen_docs_language_check_accepts_plain_language(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "densegen" / "README.md", "DenseGen guide.\n")
    _write(
        tmp_path / "src" / "dnadesign" / "densegen" / "docs" / "tutorials" / "demo.md",
        "## Demo\n\nUse this tutorial to run the workflow.\n",
    )

    issues = _find_densegen_disallowed_term_issues(tmp_path)

    assert issues == []


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
        "![aligner banner](assets/aligner-banner.svg)\n\n"
        "Aligner narrative.\n\n"
        "## Documentation\n\n"
        "[Docs](../../../docs/README.md)\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "notify" / "README.md",
        "![notify banner](assets/notify-banner.svg)\n\n"
        "Notify narrative.\n\n"
        "## Documentation\n\n"
        "[Docs](../../../docs/README.md)\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "aligner" / "assets" / "aligner-banner.svg",
        VALID_TOOL_BANNER_SVG,
    )
    _write(
        tmp_path / "src" / "dnadesign" / "notify" / "assets" / "notify-banner.svg",
        VALID_TOOL_BANNER_SVG,
    )
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# dnadesign",
                "",
                "[Documentation](docs/README.md)",
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


def test_readme_tool_catalog_does_not_require_studies_row(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "aligner" / "__init__.py", "")
    _write(tmp_path / "src" / "dnadesign" / "studies" / "__init__.py", "")
    _write(tmp_path / "src" / "dnadesign" / "aligner" / "README.md", "# aligner\n")
    _write(tmp_path / "src" / "dnadesign" / "studies" / "README.md", "# studies\n")
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# dnadesign",
                "",
                "[Documentation](docs/README.md)",
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

    assert _find_readme_tool_catalog_issues(tmp_path) == []


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
                "[Documentation](docs/README.md)",
                "",
                "## Available tools",
                "",
                "| Tool | Description |",
                "| --- | --- |",
                "| [**aligner**](src/dnadesign/aligner/README.md) |",
                "",
            ]
        ),
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_main_fails_when_readme_tool_catalog_has_an_extra_column(tmp_path: Path) -> None:
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
                "[Documentation](docs/README.md)",
                "",
                "## Available tools",
                "",
                "| Tool | Description | Internal status |",
                "| --- | --- | --- |",
                "| [**aligner**](src/dnadesign/aligner/README.md) | alignment | covered |",
                "",
            ]
        ),
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_readme_tool_catalog_rejects_component_coverage_badges(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "aligner" / "__init__.py", "")
    _write(tmp_path / "src" / "dnadesign" / "aligner" / "README.md", "# aligner\n")
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "## Available tools",
                "",
                "| Tool | Description |",
                "| --- | --- |",
                "| [**aligner**](src/dnadesign/aligner/README.md) | "
                "[![coverage](https://codecov.io/gh/example/repo/graph/badge.svg?component=aligner)]"
                "(https://codecov.io/gh/example/repo?component=aligner) |",
                "",
            ]
        ),
    )

    assert _find_readme_tool_catalog_issues(tmp_path) == [
        f"{tmp_path / 'README.md'}: tool catalog must not repeat per-tool Codecov badges or component links."
    ]


def test_main_passes_when_readme_tool_catalog_matches_repo_tools(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "README.md", f"## Index\n\n**Owner:** maintainers\n**Last verified:** {today}\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "aligner" / "__init__.py", "")
    _write(tmp_path / "src" / "dnadesign" / "notify" / "__init__.py", "")
    _write(
        tmp_path / "src" / "dnadesign" / "aligner" / "README.md",
        "![aligner banner](assets/aligner-banner.svg)\n\n"
        "Aligner narrative.\n\n"
        "## Documentation\n\n"
        "[Docs](../../../docs/README.md)\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "notify" / "README.md",
        "![notify banner](assets/notify-banner.svg)\n\n"
        "Notify narrative.\n\n"
        "## Documentation\n\n"
        "[Docs](../../../docs/README.md)\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "aligner" / "assets" / "aligner-banner.svg",
        VALID_TOOL_BANNER_SVG,
    )
    _write(
        tmp_path / "src" / "dnadesign" / "notify" / "assets" / "notify-banner.svg",
        VALID_TOOL_BANNER_SVG,
    )
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# dnadesign",
                "",
                "[Documentation](docs/README.md)",
                "",
                "## Available tools",
                "",
                "| Tool | Description |",
                "| --- | --- |",
                "| [**aligner**](src/dnadesign/aligner/README.md) | alignment |",
                "| [**notify**](src/dnadesign/notify/README.md) | notifications |",
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
                "| Tool | Description |",
                "| --- | --- |",
                "| [**aligner**](src/dnadesign/aligner/docs) | alignment |",
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
                "| Tool | Description |",
                "| --- | --- |",
                "| [**aligner**](src/dnadesign/aligner/README.md) | alignment |",
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
        "![aligner banner](assets/aligner-banner.svg)\n\n"
        "Aligner narrative.\n\n"
        "## Documentation\n\n"
        "[Docs](../../../docs/README.md)\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "notify" / "README.md",
        "![notify banner](assets/notify-banner.svg)\n\n"
        "Notify narrative.\n\n"
        "## Documentation\n\n"
        "[Docs](../../../docs/README.md)\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "aligner" / "assets" / "aligner-banner.svg",
        VALID_TOOL_BANNER_SVG,
    )
    _write(
        tmp_path / "src" / "dnadesign" / "notify" / "assets" / "notify-banner.svg",
        VALID_TOOL_BANNER_SVG,
    )
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
                "| [**notify**](src/dnadesign/notify/README.md) | notifications |",
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
                "| Tool | Description |",
                "| --- | --- |",
                "| [**aligner**](src/dnadesign/aligner/README.md) | alignment |",
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
                "| Tool | Description |",
                "| --- | --- |",
                "| [**densegen**](src/dnadesign/densegen/README.md) | densegen tool |",
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
                "| Tool | Description |",
                "| --- | --- |",
                "| [**densegen**](src/dnadesign/densegen/README.md) | densegen tool |",
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


def test_public_interface_doc_contract_includes_maintainer_and_runbook_routers(tmp_path: Path) -> None:
    _write(tmp_path / "docs" / "dev" / "README.md", "Call `python -m dnadesign.cruncher.src.cli.app`.\n")
    _write(tmp_path / "docs" / "runbooks" / "README.md", "Use `/tmp/local-runbook.yaml` for scratch work.\n")

    issues = _find_public_interface_doc_contract_issues(tmp_path)

    assert any("docs/dev/README.md" in issue and "internal source inreach" in issue for issue in issues)
    assert any("docs/runbooks/README.md" in issue and "absolute filesystem path token" in issue for issue in issues)


def test_public_interface_doc_contract_includes_top_level_tool_readmes(tmp_path: Path) -> None:
    _write(tmp_path / "src" / "dnadesign" / "opal" / "__init__.py", "")
    _write(
        tmp_path / "src" / "dnadesign" / "opal" / "README.md",
        "Call `python -m dnadesign.cruncher.src.cli.app` from `/tmp/opal-demo`.\n",
    )

    issues = _find_public_interface_doc_contract_issues(tmp_path)

    assert any("src/dnadesign/opal/README.md" in issue and "internal source inreach" in issue for issue in issues)
    assert any(
        "src/dnadesign/opal/README.md" in issue and "absolute filesystem path token" in issue for issue in issues
    )


def test_broken_link_check_includes_top_level_tool_readmes(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "README.md", f"## Index\n\n**Owner:** maintainers\n**Last verified:** {today}\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "opal" / "__init__.py", "")
    _write(
        tmp_path / "src" / "dnadesign" / "opal" / "README.md",
        "![opal banner](assets/opal-banner.svg)\n\nOPAL narrative.\n\n## Documentation\n\n[Missing](docs/missing.md)\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "opal" / "assets" / "opal-banner.svg", VALID_TOOL_BANNER_SVG)
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# dnadesign",
                "",
                "[Documentation](docs/README.md)",
                "",
                "## Available tools",
                "",
                "| Tool | Description |",
                "| --- | --- |",
                "| [**opal**](src/dnadesign/opal/README.md) | opal tool |",
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
                "    - component_id: opal",
                "      name: opal",
                "      paths:",
                "        - src/dnadesign/opal/**",
                "",
            ]
        ),
    )

    rc = main(["--repo-root", str(tmp_path)])

    assert rc == 1


def test_main_passes_when_codecov_components_match_repo_tools(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(tmp_path / "docs" / "README.md", f"## Index\n\n**Owner:** maintainers\n**Last verified:** {today}\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(tmp_path / "src" / "dnadesign" / "aligner" / "__init__.py", "")
    _write(tmp_path / "src" / "dnadesign" / "notify" / "__init__.py", "")
    _write(
        tmp_path / "src" / "dnadesign" / "aligner" / "README.md",
        "![aligner banner](assets/aligner-banner.svg)\n\n"
        "Aligner narrative.\n\n"
        "## Documentation\n\n"
        "[Docs](../../../docs/README.md)\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "notify" / "README.md",
        "![notify banner](assets/notify-banner.svg)\n\n"
        "Notify narrative.\n\n"
        "## Documentation\n\n"
        "[Docs](../../../docs/README.md)\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "aligner" / "assets" / "aligner-banner.svg",
        VALID_TOOL_BANNER_SVG,
    )
    _write(
        tmp_path / "src" / "dnadesign" / "notify" / "assets" / "notify-banner.svg",
        VALID_TOOL_BANNER_SVG,
    )
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# dnadesign",
                "",
                "[Documentation](docs/README.md)",
                "",
                "## Available tools",
                "",
                "| Tool | Description |",
                "| --- | --- |",
                "| [**aligner**](src/dnadesign/aligner/README.md) | alignment |",
                "| [**notify**](src/dnadesign/notify/README.md) | notifications |",
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


def test_docs_root_heading_style_check_ignores_generated_output_markdown(tmp_path: Path) -> None:
    _write(tmp_path / "docs" / "README.md", "## Documentation Index\n")
    _write(
        tmp_path / "docs" / "studies" / "demo" / "workbench" / "outputs" / "bundle" / "README.md",
        "# Generated Bundle\n",
    )

    issues = _find_docs_root_heading_style_issues(tmp_path)

    assert issues == []
