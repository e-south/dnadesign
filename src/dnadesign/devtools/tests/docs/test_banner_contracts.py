"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/docs/test_banner_contracts.py

Tests for documentation banner contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pytest

from dnadesign.devtools.docs import checks as docs_checks
from dnadesign.devtools.docs.banners.catalog import BannerSpec
from dnadesign.devtools.docs.checks import (
    _find_root_docs_entrypoint_issues,
    _find_tool_readme_banner_issues,
    _find_tool_readme_structure_issues,
    main,
)
from dnadesign.devtools.tests.docs.check_test_support import (
    VALID_TOOL_BANNER_SVG,
    _write,
)


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
        VALID_TOOL_BANNER_SVG,
    )

    issues = _find_tool_readme_banner_issues(tmp_path)

    assert issues == []


@pytest.mark.parametrize(
    "banner_syntax",
    [
        "<!-- ![Alpha banner](assets/alpha-banner.svg) -->",
        r"\![Alpha banner](assets/alpha-banner.svg)",
        "`![Alpha banner](assets/alpha-banner.svg)`",
    ],
)
def test_tool_readme_banner_check_requires_a_rendered_image(tmp_path: Path, banner_syntax: str) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "README.md",
        f"{banner_syntax}\n\nCompact subtitle.\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "assets" / "alpha-banner.svg",
        VALID_TOOL_BANNER_SVG,
    )

    issues = _find_tool_readme_banner_issues(tmp_path)

    assert any("alpha/README.md" in issue and "missing top banner image" in issue for issue in issues)


def test_tool_readme_banner_check_rejects_uncatalogued_and_orphaned_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "README.md",
        "![Alpha banner](assets/alternate-banner.svg)\n\nCompact subtitle.\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "assets" / "alternate-banner.svg",
        VALID_TOOL_BANNER_SVG,
    )
    (tmp_path / "src" / "dnadesign" / "devtools" / "docs" / "banners").mkdir(parents=True)
    catalog_path = "src/dnadesign/alpha/assets/alpha-banner.svg"
    monkeypatch.setattr(
        docs_checks,
        "BANNERS",
        (
            BannerSpec(
                path=catalog_path,
                readme_path="src/dnadesign/alpha/README.md",
                name="alpha",
                capability="TEST ALPHA",
                description="Test alpha banners.",
                glyph="align",
            ),
        ),
        raising=False,
    )

    issues = _find_tool_readme_banner_issues(tmp_path)

    assert any("alternate-banner.svg" in issue and "not declared in the banner catalog" in issue for issue in issues)
    assert any(catalog_path in issue and "not referenced by a tool README" in issue for issue in issues)


def test_tool_readme_banner_check_rejects_swapped_catalog_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    alpha_banner = "src/dnadesign/alpha/assets/alpha-banner.svg"
    beta_banner = "src/dnadesign/beta/assets/beta-banner.svg"
    alpha_readme = "src/dnadesign/alpha/README.md"
    beta_readme = "src/dnadesign/beta/README.md"
    _write(
        tmp_path / alpha_readme,
        "![Alpha banner](../beta/assets/beta-banner.svg)\n\nAlpha narrative.\n",
    )
    _write(
        tmp_path / beta_readme,
        "![Beta banner](../alpha/assets/alpha-banner.svg)\n\nBeta narrative.\n",
    )
    _write(tmp_path / alpha_banner, VALID_TOOL_BANNER_SVG)
    _write(tmp_path / beta_banner, VALID_TOOL_BANNER_SVG)
    (tmp_path / "src" / "dnadesign" / "devtools" / "docs" / "banners").mkdir(parents=True)
    monkeypatch.setattr(
        docs_checks,
        "BANNERS",
        (
            BannerSpec(
                path=alpha_banner,
                readme_path=alpha_readme,
                name="alpha",
                capability="TEST ALPHA",
                description="Test alpha banners.",
                glyph="align",
            ),
            BannerSpec(
                path=beta_banner,
                readme_path=beta_readme,
                name="beta",
                capability="TEST BETA",
                description="Test beta banners.",
                glyph="align",
            ),
        ),
        raising=False,
    )

    issues = _find_tool_readme_banner_issues(tmp_path)

    assert any(alpha_readme in issue and alpha_banner in issue and beta_banner in issue for issue in issues)
    assert any(beta_readme in issue and beta_banner in issue and alpha_banner in issue for issue in issues)


def test_tool_readme_banner_check_rejects_nonstandard_banner_dimensions(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "README.md",
        "![Alpha banner](assets/alpha-banner.svg)\n\nShort narrative.\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "assets" / "alpha-banner.svg",
        '<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="420" viewBox="0 0 1600 420"></svg>\n',
    )

    issues = _find_tool_readme_banner_issues(tmp_path)

    assert any("1200x180 SVG contract" in issue for issue in issues)


def test_tool_readme_structure_check_requires_banner_as_first_non_empty_line(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "README.md",
        "## Alpha\n\n![Alpha banner](assets/alpha-banner.svg)\n\nShort narrative.\n\n[Docs](../../../docs/README.md)\n",
    )
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "assets" / "alpha-banner.svg",
        VALID_TOOL_BANNER_SVG,
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
        VALID_TOOL_BANNER_SVG,
    )

    issues = _find_tool_readme_structure_issues(tmp_path)

    assert any("line after the banner must be narrative text" in issue for issue in issues)


def test_tool_readme_structure_check_accepts_banner_narrative_and_docs_link(tmp_path: Path) -> None:
    _write(
        tmp_path / "src" / "dnadesign" / "alpha" / "README.md",
        "\n".join(
            [
                "![Alpha banner](assets/alpha-banner.svg)",
                "",
                "Short narrative overview.",
                "",
                "## Documentation",
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
        VALID_TOOL_BANNER_SVG,
    )

    issues = _find_tool_readme_structure_issues(tmp_path)

    assert issues == []


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


def test_root_docs_entrypoint_check_rejects_bannerless_readme_without_docs_link(tmp_path: Path) -> None:
    _write(
        tmp_path / "README.md",
        "\n".join(
            [
                "# dnadesign",
                "",
                "Use the docs index.",
                "",
            ]
        ),
    )

    issues = _find_root_docs_entrypoint_issues(tmp_path)

    assert any("must include a markdown link to docs/README.md" in issue for issue in issues)


def test_main_fails_when_generated_banner_is_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    today = dt.date.today().isoformat()
    _write(
        tmp_path / "docs" / "README.md",
        f"## Documentation Index\n\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"# ARCHITECTURE\n\n**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n",
    )
    (tmp_path / "src" / "dnadesign" / "devtools" / "docs" / "banners").mkdir(parents=True)
    stale_path = Path("assets/dnadesign-banner.svg")
    monkeypatch.setattr(docs_checks, "check_banners", lambda _repo_root: (stale_path,), raising=False)

    rc = main(["--repo-root", str(tmp_path)])

    captured = capsys.readouterr()
    assert rc == 1
    assert "Banner source drift check failed" in captured.out
    assert str(stale_path) in captured.out
