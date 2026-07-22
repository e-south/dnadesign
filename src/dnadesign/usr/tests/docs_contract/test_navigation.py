"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/usr/tests/docs_contract/test_navigation.py

Structural router and index contracts for USR docs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .helpers import assert_markdown_links_resolve, markdown_links, read_text


def test_repo_and_usr_indexes_have_resolving_markdown_links() -> None:
    for rel_path in (
        "README.md",
        "docs/README.md",
        "src/dnadesign/usr/README.md",
        "src/dnadesign/usr/docs/README.md",
        "src/dnadesign/usr/docs/reference/README.md",
        "src/dnadesign/usr/docs/operations/README.md",
    ):
        assert_markdown_links_resolve(rel_path)


def test_docs_indexes_route_to_expected_usr_surfaces() -> None:
    docs_index_links = markdown_links("docs/README.md")
    usr_docs_links = markdown_links("src/dnadesign/usr/docs/README.md")
    reference_links = markdown_links("src/dnadesign/usr/docs/reference/README.md")

    assert "../src/dnadesign/usr/docs/operations/routes/workflow-map.md" in docs_index_links
    assert "../src/dnadesign/usr/docs/operations/sync/hpc-agent-flow.md" in docs_index_links
    assert "../src/dnadesign/usr/docs/operations/sync/chained-densegen-infer-runbook.md" in docs_index_links
    assert "../src/dnadesign/usr/docs/operations/assembly/construct-infer-shared-dataset-runbook.md" in docs_index_links
    assert "studies/README.md" in docs_index_links

    assert "operations/routes/workflow-map.md" in usr_docs_links
    assert "operations/sync/README.md" in usr_docs_links
    assert "reference/python-api.md" in usr_docs_links
    assert "reference/dataset-layout-and-code-map.md" in usr_docs_links

    assert "dataset-layout-and-code-map.md" in reference_links
    assert "schema-contract.md" in reference_links
    assert "overlay-and-registry.md" in reference_links
    assert "event-log.md" in reference_links
    assert "python-api.md" in reference_links
    assert "maintenance.md" in reference_links


def test_docs_surface_excludes_start_here_and_meta_routing_jargon() -> None:
    readme = read_text("README.md")
    docs_index = read_text("docs/README.md")
    assert "docs/start-here.md" not in readme
    assert "start-here.md" not in docs_index

    for rel_path in ("README.md", "docs/README.md", "docs/runbooks/README.md"):
        text = read_text(rel_path).lower()
        assert "authoritative" not in text
        assert "canonical" not in text
        assert "progressive disclosure" not in text


def test_usr_docs_index_stays_decoupled_from_top_readme_anchor_fragments() -> None:
    usr_docs = read_text("src/dnadesign/usr/docs/README.md")
    assert "../README.md#" not in usr_docs
