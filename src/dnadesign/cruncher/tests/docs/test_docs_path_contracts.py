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


def test_docs_define_baserender_public_api_boundary() -> None:
    architecture = (_package_root() / "docs" / "reference" / "architecture.md").read_text()
    assert "from dnadesign.baserender import" in architecture
    disallowed_path_marker = "dnadesign.baserender" + ".src.*"
    assert disallowed_path_marker in architecture

    analysis_guide = (_package_root() / "docs" / "guides" / "sampling_and_analysis.md").read_text()
    assert "minimal rendering primitives" in analysis_guide
    assert "dnadesign.baserender" in analysis_guide


def test_demo_docs_encode_merged_meme_oops_provenance_pattern() -> None:
    docs_root = _package_root() / "docs" / "demos"
    two_tf = (docs_root / "demo_basics_two_tf.md").read_text()
    campaign = (docs_root / "demo_campaigns_multi_tf.md").read_text()
    three_tf = (docs_root / "demo_densegen_prep_three_tf.md").read_text()

    assert "fetch sites --source demo_local_meme --tf lexA --tf cpxR" in two_tf
    assert "fetch sites --source regulondb      --tf lexA --tf cpxR" in two_tf
    assert "--tool meme --meme-mod oops --source-id demo_merged_meme_oops" in two_tf

    assert "fetch sites --source demo_local_meme --tf lexA --tf cpxR" in campaign
    assert "fetch sites --source regulondb      --tf lexA --tf cpxR" in campaign
    assert "fetch sites --source baer_chip_exo --tf baeR" in campaign
    assert "fetch sites --source regulondb    --tf baeR" in campaign
    assert "--tool meme --meme-mod oops --source-id demo_merged_meme_oops_campaign" in campaign
    assert "catalog export-densegen --set 1" not in campaign
    assert "catalog export-densegen --tf lexA --tf cpxR" in campaign

    assert "fetch sites --source demo_local_meme --tf lexA --tf cpxR" in three_tf
    assert "fetch sites --source regulondb      --tf lexA --tf cpxR" in three_tf
    assert "fetch sites --source baer_chip_exo --tf baeR" in three_tf
    assert "fetch sites --source regulondb      --tf baeR" in three_tf
    assert "--tool meme --meme-mod oops --source-id demo_merged_meme_oops_three_tf" in three_tf
