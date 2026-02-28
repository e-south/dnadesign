"""
--------------------------------------------------------------------------------
<densegen project>
src/dnadesign/densegen/tests/docs/test_densegen_docs_ia_contracts.py

Contract checks for DenseGen docs navigation and workspace runbook discoverability.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

from dnadesign.baserender import DENSEGEN_TFBS_REQUIRED_KEYS

ROOT = Path(__file__).resolve().parents[2]
DOCS_ROOT = ROOT / "docs"
WORKSPACES = ROOT / "workspaces"
LINK_PATTERN = re.compile(r"\[[^\]]+\]\((?P<link>[^)]+)\)")
TERM_PATTERN = re.compile(r"\b(switchboard|agent|human|canonical)\b", flags=re.IGNORECASE)
WORKSPACE_IDS = (
    "demo_tfbs_baseline",
    "demo_sampling_baseline",
    "study_constitutive_sigma_panel",
    "study_stress_ethanol_cipro",
)
SIGMA70_LITERAL_SOURCE_CITATION = (
    "Tuning the dynamic range of bacterial promoters regulated by ligand-inducible transcription factors"
)
SIGMA70_DOI = "10.1038/s41467-017-02473-5"
SIGMA70_NATURE_URL = "https://www.nature.com/articles/s41467-017-02473-5"


def _read(path: Path) -> str:
    assert path.exists(), f"Missing file: {path}"
    return path.read_text()


def _assert_token_order(text: str, tokens: list[str], *, label: str) -> None:
    cursor = -1
    for token in tokens:
        idx = text.find(token, cursor + 1)
        assert idx >= 0, f"{label}: missing token: {token!r}"
        assert idx > cursor, f"{label}: out-of-order token: {token!r}"
        cursor = idx


def _iter_local_markdown_targets(source: Path) -> list[Path]:
    targets: list[Path] = []
    for match in LINK_PATTERN.finditer(_read(source)):
        raw = match.group("link").strip().split()[0]
        if not raw or raw.startswith("#"):
            continue
        parsed = urlparse(raw)
        if parsed.scheme or raw.startswith("mailto:"):
            continue
        rel = raw.split("#", 1)[0].strip()
        if not rel:
            continue
        resolved = (source.parent / rel).resolve()
        if resolved.suffix != ".md":
            continue
        if resolved == DOCS_ROOT or DOCS_ROOT in resolved.parents:
            targets.append(resolved)
    return targets


def _collect_markdown_headings(path: Path) -> list[int]:
    levels: list[int] = []
    in_code = False
    for raw in _read(path).splitlines():
        if raw.startswith("    ") or raw.startswith("\t"):
            continue
        stripped = raw.strip()
        if stripped.startswith("```"):
            in_code = not in_code
            continue
        if in_code:
            continue
        if not stripped.startswith("#"):
            continue
        level = len(stripped) - len(stripped.lstrip("#"))
        if level > 0 and stripped[level : level + 1] == " ":
            levels.append(level)
    return levels


def _is_generated_markdown(path: Path) -> bool:
    return "outputs" in path.parts


def test_densegen_docs_entry_exists_with_workflow_sections() -> None:
    content = _read(DOCS_ROOT / "README.md")
    assert "## DenseGen documentation" in content
    assert "### Documentation by workflow" in content
    assert "### Workspace documentation" in content
    assert "### Documentation by type" in content


def test_densegen_docs_index_points_to_docs_entry() -> None:
    content = _read(DOCS_ROOT / "index.md")
    assert "docs/README.md" in content or "README.md" in content


def test_densegen_docs_entry_links_to_doc_index() -> None:
    content = _read(DOCS_ROOT / "README.md")
    assert "index.md" in content


def test_densegen_docs_entry_links_to_workspace_catalog_in_workspaces_dir() -> None:
    content = _read(DOCS_ROOT / "README.md")
    assert "../workspaces/catalog.md" in content
    assert "workspaces.md" not in content


def test_densegen_top_level_readme_points_to_docs_entry() -> None:
    content = _read(ROOT / "README.md")
    assert "docs/README.md" in content


def test_densegen_workspace_catalog_exists_in_workspaces_dir_and_covers_packaged_workspaces() -> None:
    catalog = _read(WORKSPACES / "catalog.md")
    for workspace_id in WORKSPACE_IDS:
        assert workspace_id in catalog


def test_densegen_top_level_readme_has_banner_and_ordered_documentation_map() -> None:
    content = _read(ROOT / "README.md")
    assert "![DenseGen banner]" in content
    _assert_token_order(
        content,
        [
            "docs/README.md",
            "workspaces/catalog.md",
            "docs/tutorials/demo_tfbs_baseline.md",
            "docs/tutorials/demo_sampling_baseline.md",
            "docs/tutorials/study_constitutive_sigma_panel.md",
            "docs/tutorials/study_stress_ethanol_cipro.md",
            "docs/tutorials/demo_usr_notify.md",
            "docs/concepts/quick-checklist.md",
            "docs/reference/cli.md",
            "docs/reference/config.md",
            "docs/reference/outputs.md",
            "docs/howto/hpc.md",
            "docs/howto/bu-scc.md",
            "docs/dev/architecture.md",
            "docs/dev/journal.md",
        ],
        label="densegen/README.md doc map",
    )


def test_packaged_workspaces_have_readme_entrypoint_linking_runbook() -> None:
    for workspace_id in WORKSPACE_IDS:
        readme = _read(WORKSPACES / workspace_id / "README.md")
        _read(WORKSPACES / workspace_id / "runbook.md")
        assert "runbook.md" in readme
        assert workspace_id in readme


def test_densegen_docs_are_reachable_from_docs_entry() -> None:
    docs_files = {path.resolve() for path in DOCS_ROOT.rglob("*.md")}
    queue = [DOCS_ROOT / "README.md"]
    seen: set[Path] = set()

    while queue:
        current = queue.pop()
        current = current.resolve()
        if current in seen:
            continue
        seen.add(current)
        for target in _iter_local_markdown_targets(current):
            if target not in seen:
                queue.append(target)

    missing = sorted(path.relative_to(DOCS_ROOT).as_posix() for path in docs_files - seen)
    assert not missing, f"DenseGen docs unreachable from docs/README.md: {missing}"


def test_densegen_agents_points_to_docs_entry() -> None:
    content = _read(ROOT / "AGENTS.md")
    assert "[Docs index by workflow](docs/README.md)" in content


def test_densegen_docs_use_plain_direct_language() -> None:
    paths = [ROOT / "README.md", ROOT / "AGENTS.md"]
    paths.extend(DOCS_ROOT.rglob("*.md"))
    paths.extend((WORKSPACES / workspace_id / "README.md" for workspace_id in WORKSPACE_IDS))
    for path in paths:
        content = _read(path)
        match = TERM_PATTERN.search(content)
        assert match is None, f"{path}: banned term '{match.group(1)}'"


def test_densegen_markdown_heading_levels_use_single_h2_root() -> None:
    markdown_files = sorted(path for path in ROOT.rglob("*.md") if not _is_generated_markdown(path))
    assert markdown_files, "No markdown files found under DenseGen root."
    for path in markdown_files:
        levels = _collect_markdown_headings(path)
        assert levels, f"{path}: missing markdown headings"
        assert levels[0] == 2, f"{path}: first heading must be level-2 (##)"
        assert levels.count(2) == 1, f"{path}: only one level-2 heading is allowed"
        assert all(level >= 2 for level in levels), f"{path}: level-1 headings are not allowed"


def test_sigma70_literal_docs_include_source_citation() -> None:
    targets = (
        WORKSPACES / "study_constitutive_sigma_panel" / "runbook.md",
        WORKSPACES / "study_stress_ethanol_cipro" / "runbook.md",
        DOCS_ROOT / "tutorials" / "study_constitutive_sigma_panel.md",
        DOCS_ROOT / "tutorials" / "study_stress_ethanol_cipro.md",
    )
    for path in targets:
        content = _read(path)
        assert SIGMA70_LITERAL_SOURCE_CITATION in content, f"{path}: missing sigma70 source title"
        assert SIGMA70_DOI in content, f"{path}: missing sigma70 source DOI"
        assert SIGMA70_NATURE_URL in content, f"{path}: missing sigma70 source URL"


def test_outputs_reference_documents_strict_notebook_render_contract() -> None:
    content = _read(DOCS_ROOT / "reference" / "outputs.md")
    assert "Adapter policy contract: `on_invalid_row=error`" in content
    assert "Required TFBS entry keys inside `densegen__used_tfbs_detail`" in content
    for key in DENSEGEN_TFBS_REQUIRED_KEYS:
        assert f"`{key}`" in content
