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

from dnadesign.devtools.docs.badges import (
    ROOT_README_ALLOWED_BADGES,
    find_markdown_badge_policy_issues,
    rendered_markdown_badge_lines,
)


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
        "Escaped image example: \\![Coverage](badge.svg)\n\n"
        "Inline example: `[![Coverage](coverage.svg)](report)`\n\n"
        "<!-- [![Coverage](coverage.svg)](report) -->\n\n"
        "    [![Coverage](badge.svg)](report)\n\n"
        ">     [![Coverage](badge.svg)](report)\n\n"
        "[![quality][code-image]][code-report]\n\n"
        "    [code-image]: https://example.test/badge.svg\n"
        "    [code-report]: https://example.test/report\n\n"
        "<script>\n"
        '<img alt="Coverage" src="badge.svg">\n'
        "</script>\n\n"
        '<video><source src="status-badge.mp4"></video>\n\n'
        '<source srcset="https://img.shields.io/badge/build-passing.svg">\n\n'
        '<a href="outer">\n\n'
        "[inner](report) ![Coverage](status.svg)\n\n"
        "</a>\n\n"
        '[outer <a href="inner">inner</a> ![Coverage](status.svg)](report)\n\n'
        '<template><img src="badge.svg"></template>\n\n'
        "<template>\n\n"
        "![Coverage](badge.svg)\n\n"
        "</template>\n\n"
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
    ("content", "expected_line"),
    [
        ("[![Coverage](https://example.test/coverage.svg)](https://example.test/report)\n", 1),
        ("[ ![Coverage](status.svg) ](report)\n", 1),
        ("- item\n\n    [![Coverage](badge.svg)](report)\n", 3),
        (
            "[![Coverage][coverage-image]][coverage-report]\n\n"
            "[coverage-image]: https://example.test/coverage.svg\n"
            "[coverage-report]: https://example.test/report\n",
            1,
        ),
        (
            "[![quality][image]][report]\n\n"
            "   [image]: https://example.test/badge.svg\n"
            "   [report]: https://example.test/report\n",
            1,
        ),
        (
            "[![Coverage]][coverage-report]\n\n"
            "[Coverage]: https://example.test/coverage.svg\n"
            "[coverage-report]: https://example.test/report\n",
            1,
        ),
        (
            '<a href="https://example.test/report"><img alt="Coverage" src="https://example.test/status.svg"></a>\n',
            1,
        ),
        ('<img alt="quality > details" src="badge.svg">\n', 1),
        ('<a href="report">\n\n![Coverage](status.svg)\n\n</a>\n', 3),
        ('text <a href="report">\n<img alt="Coverage" src="status.svg">\n</a>\n', 2),
        (
            '<img alt="build" src="diagram.svg" srcset="https://img.shields.io/badge/build-passing.svg">\n',
            1,
        ),
        (
            '<picture><source srcset="https://img.shields.io/badge/build-passing.svg">'
            '<img alt="build" src="diagram.svg"></picture>\n',
            1,
        ),
        ('<a href="report" />\n\n<img alt="Coverage" src="status.svg">\n', 3),
        ('intro\n<img\n alt="Coverage"\n src="badge.svg">\noutro\n', 2),
        ('intro `code\nspan`\n<img\n alt="Coverage"\n src="badge.svg">\n', 3),
    ],
)
def test_badge_policy_rejects_alternate_badge_syntax_outside_root(
    tmp_path: Path,
    content: str,
    expected_line: int,
) -> None:
    tool_readme = tmp_path / "src" / "dnadesign" / "aligner" / "README.md"
    _write(tool_readme, content)

    assert find_markdown_badge_policy_issues(tmp_path, [tool_readme]) == [
        f"{tool_readme}:{expected_line}: badges belong only in the root README; use a plain text link instead."
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


@pytest.mark.parametrize(
    "content",
    [
        "```markdown\n" + "\n".join(sorted(ROOT_README_ALLOWED_BADGES)) + "\n```\n",
        "<!--\n" + "\n".join(sorted(ROOT_README_ALLOWED_BADGES)) + "\n-->\n",
    ],
)
def test_badge_policy_requires_root_badges_to_be_rendered(tmp_path: Path, content: str) -> None:
    root_readme = tmp_path / "README.md"
    _write(root_readme, content)

    assert rendered_markdown_badge_lines(root_readme.read_text(encoding="utf-8")) == ()


def test_checked_in_root_readme_uses_the_restrained_badge_set() -> None:
    repo_root = Path(__file__).resolve().parents[5]
    root_readme = repo_root / "README.md"
    badges = rendered_markdown_badge_lines(root_readme.read_text(encoding="utf-8"))

    assert len(badges) == 3
    assert set(badges) == ROOT_README_ALLOWED_BADGES
