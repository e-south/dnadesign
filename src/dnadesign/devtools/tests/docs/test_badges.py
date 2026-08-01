"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/docs/test_badges.py

Tests the restrained Markdown badge policy.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.devtools.docs.badges import ROOT_README_ALLOWED_BADGES, find_markdown_badge_policy_issues


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_badge_policy_allows_restrained_root_badges_and_text_coverage_link(tmp_path: Path) -> None:
    root_readme = tmp_path / "README.md"
    quality_doc = tmp_path / "docs" / "quality" / "README.md"
    tool_readme = tmp_path / "src" / "dnadesign" / "construct" / "README.md"
    figure_doc = tmp_path / "docs" / "workflow.md"
    example_doc = tmp_path / "docs" / "badge-example.md"
    _write(root_readme, "\n".join(sorted(ROOT_README_ALLOWED_BADGES)))
    _write(
        quality_doc,
        "[Coverage details](https://codecov.io/gh/e-south/dnadesign?component=aligner)\n",
    )
    _write(tool_readme, "![construct banner](docs/assets/construct-banner.svg)\n")
    _write(
        figure_doc,
        "[![Workflow diagram](diagram.png)](diagram-full.png)\n"
        "[![Security architecture](security-architecture.svg)](security-architecture-full.svg)\n"
        "[![Release sequence](release-sequence.svg)](release-sequence-full.svg)\n",
    )
    _write(
        example_doc,
        "Literal shortcut syntax: ![Coverage]\n\n"
        "Inline example: `[![Coverage](coverage.svg)](report)`\n\n"
        "<!-- [![Coverage](coverage.svg)](report) -->\n\n"
        "```markdown\n"
        "[![Coverage](https://example.test/coverage.svg)](https://example.test/report)\n"
        "```\n",
    )

    assert (
        find_markdown_badge_policy_issues(
            tmp_path,
            [example_doc, figure_doc, quality_doc, root_readme, tool_readme],
        )
        == []
    )


def test_badge_policy_rejects_component_badge_outside_root(tmp_path: Path) -> None:
    tool_readme = tmp_path / "src" / "dnadesign" / "aligner" / "README.md"
    _write(
        tool_readme,
        "[![coverage](https://codecov.io/gh/example/repo/graph/badge.svg?component=aligner)]"
        "(https://codecov.io/gh/example/repo?component=aligner)\n",
    )

    assert find_markdown_badge_policy_issues(tmp_path, [tool_readme]) == [
        f"{tool_readme}:1: badges belong only in the root README; use a plain text link instead."
    ]


@pytest.mark.parametrize(
    "content",
    [
        "[![Coverage](https://example.test/coverage.svg)](https://example.test/report)\n",
        "[![Coverage][coverage-image]][coverage-report]\n\n"
        "[coverage-image]: https://example.test/coverage.svg\n"
        "[coverage-report]: https://example.test/report\n",
        "[![Coverage]][coverage-report]\n\n"
        "[Coverage]: https://example.test/coverage.svg\n"
        "[coverage-report]: https://example.test/report\n",
        '<a href="https://example.test/report"><img alt="Coverage" src="https://example.test/status.svg"></a>\n',
    ],
)
def test_badge_policy_rejects_alternate_badge_syntax_outside_root(tmp_path: Path, content: str) -> None:
    tool_readme = tmp_path / "src" / "dnadesign" / "aligner" / "README.md"
    _write(tool_readme, content)

    assert find_markdown_badge_policy_issues(tmp_path, [tool_readme]) == [
        f"{tool_readme}:1: badges belong only in the root README; use a plain text link instead."
    ]


def test_badge_policy_rejects_unapproved_root_badge(tmp_path: Path) -> None:
    root_readme = tmp_path / "README.md"
    _write(
        root_readme,
        "[![docs](https://img.shields.io/badge/docs-online-blue.svg)](https://example.test/docs)\n",
    )

    assert find_markdown_badge_policy_issues(tmp_path, [root_readme]) == [
        f"{root_readme}:1: root README badge is not in the restrained CI, coverage, and license set."
    ]


def test_checked_in_root_readme_uses_the_restrained_badge_set() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    root_readme = repo_root / "README.md"
    badges = [line for line in root_readme.read_text(encoding="utf-8").splitlines() if line.startswith("[![")]

    assert len(badges) == 3
    assert set(badges) == ROOT_README_ALLOWED_BADGES
