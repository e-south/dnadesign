"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/docs/test_docs_path_contracts.py

Validate public docs against the current run-artifact path contract.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path


def _package_root() -> Path:
    return Path(__file__).resolve().parents[2]


def test_readme_uses_analysis_subdir_contract() -> None:
    readme = (_package_root() / "README.md").read_text()
    assert "outputs/analysis/" in readme
    assert "outputs/output/" not in readme


def test_campaign_demo_reset_is_shell_safe_without_nomatch_globs() -> None:
    demo_doc = (_package_root() / "docs" / "demos" / "demo_campaigns_multi_tf.md").read_text()
    assert "rm -f campaign_*.yaml campaign_*.campaign_manifest.json" not in demo_doc
    assert "find . -maxdepth 1 -type f" in demo_doc
