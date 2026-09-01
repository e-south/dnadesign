"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/devtools/tests/docs/test_markdown_inventory.py

Tests for documentation markdown inventory.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

from dnadesign.devtools.docs import checks as docs_checks
from dnadesign.devtools.docs.checks import (
    _find_broken_links,
    main,
)
from dnadesign.devtools.tests.docs.check_test_support import (
    _write,
)
from dnadesign.ops.runbooks import REPO_TRANSIENT_OPERATIONAL_DIR_NAMES


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


def test_broken_links_check_rejects_absolute_local_path_outside_repo(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.md"
    outside.write_text("# Outside\n", encoding="utf-8")
    source = tmp_path / "docs" / "index.md"
    _write(source, f"[outside]({outside})\n")

    broken = _find_broken_links([source], repo_root=tmp_path)

    assert broken == [(source, f"{outside} (local link escapes repository)")]


def test_broken_link_check_ignores_fenced_code_markdown_links(tmp_path: Path) -> None:
    index_path = tmp_path / "docs" / "index.md"
    _write(
        index_path,
        "\n".join(
            [
                "## Examples",
                "",
                "```md",
                "[illustrative missing link](./not-a-real-route.md)",
                "```",
                "",
            ]
        ),
    )

    broken = _find_broken_links([index_path])

    assert broken == []


def test_broken_link_check_still_flags_body_markdown_links(tmp_path: Path) -> None:
    index_path = tmp_path / "docs" / "index.md"
    _write(index_path, "[missing](./nope.md)\n")

    broken = _find_broken_links([index_path])

    assert broken == [(index_path, "./nope.md")]


def test_main_fails_for_broken_relative_link_in_root_sor_doc(tmp_path: Path) -> None:
    _write(tmp_path / "docs" / "index.md", "## Index\n")
    _write(tmp_path / "ARCHITECTURE.md", "[broken](docs/missing.md)\n")

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 1


def test_docs_checks_reuses_ops_path_policy_contract_constants() -> None:
    assert docs_checks.TRANSIENT_OPERATIONAL_ROOT_DIR_NAMES == REPO_TRANSIENT_OPERATIONAL_DIR_NAMES
    assert docs_checks.OPS_OPERATIONAL_RUNBOOK_ALLOWED_PREFIXES == (Path("docs/templates"),)


def test_main_passes_for_valid_links(tmp_path: Path) -> None:
    today = dt.date.today().isoformat()
    _write(
        tmp_path / "docs" / "README.md",
        f"## x\n\n**Owner:** maintainers\n**Last verified:** {today}\n\n"
        "[guide](./guide.md)\n[#anchor](#x)\n[site](https://example.com)\n",
    )
    _write(tmp_path / "docs" / "guide.md", "## Guide\n")
    _write(tmp_path / "README.md", "[docs](docs/README.md)\n")
    _write(
        tmp_path / "ARCHITECTURE.md",
        f"**Type:** system-of-record\n**Owner:** maintainers\n**Last verified:** {today}\n[docs](docs/guide.md)\n",
    )

    rc = main(["--repo-root", str(tmp_path)])
    assert rc == 0


def test_broken_links_check_flags_missing_markdown_anchor(tmp_path: Path) -> None:
    source = tmp_path / "docs" / "source.md"
    target = tmp_path / "docs" / "target.md"
    _write(source, "[missing](./target.md#not-here)\n")
    _write(target, "## Present Heading\n")

    broken = _find_broken_links([source, target])

    assert any("anchor 'not-here'" in issue_link for _, issue_link in broken)
