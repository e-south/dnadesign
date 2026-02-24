"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/docs/test_workspace_laci_arac_contracts.py

Docs/workspace contracts for the pairwise lacI+araC workspace.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]


def test_pairwise_laci_arac_workspace_files_exist() -> None:
    workspace = ROOT / "workspaces" / "pairwise_laci_arac"
    assert (workspace / "configs" / "config.yaml").exists()
    assert (workspace / "runbook.md").exists()
    assert (workspace / "inputs" / "local_motifs" / "lacI.txt").exists()
    readme = (ROOT / "workspaces" / "README.md").read_text(encoding="utf-8")
    assert "pairwise_laci_arac/" in readme


def test_pairwise_laci_arac_config_contract() -> None:
    config_path = ROOT / "workspaces" / "pairwise_laci_arac" / "configs" / "config.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cruncher = payload["cruncher"]

    assert cruncher["workspace"]["regulator_sets"] == [["lacI", "araC"]]
    assert cruncher["catalog"]["source_preference"] == ["pairwise_laci_arac_merged_meme_oops"]
    assert cruncher["discover"]["source_id"] == "pairwise_laci_arac_merged_meme_oops"
    assert cruncher["discover"]["tool"] == "meme"
    assert cruncher["discover"]["meme_mod"] == "oops"

    local_sources = cruncher["ingest"]["local_sources"]
    assert len(local_sources) >= 1
    assert local_sources[0]["source_id"] == "demo_local_meme"
    assert local_sources[0]["root"] == "inputs/local_motifs"

    regulondb = cruncher["ingest"]["regulondb"]
    assert regulondb["curated_sites"] is True
    assert regulondb["ht_sites"] is False


def test_pairwise_laci_arac_runbook_contract() -> None:
    runbook_path = ROOT / "workspaces" / "pairwise_laci_arac" / "runbook.md"
    runbook = runbook_path.read_text(encoding="utf-8")

    assert 'cruncher fetch sites --source demo_local_meme --tf lacI --update -c "$CONFIG"' in runbook
    assert 'cruncher fetch sites --source regulondb --tf lacI --tf araC --update -c "$CONFIG"' in runbook
    assert (
        "cruncher discover motifs --set 1 --tool meme --meme-mod oops --source-id "
        'pairwise_laci_arac_merged_meme_oops -c "$CONFIG"'
    ) in runbook
    assert 'cruncher catalog logos --source pairwise_laci_arac_merged_meme_oops --set 1 -c "$CONFIG"' in runbook
    assert (
        'cruncher catalog export-densegen --set 1 --densegen-workspace study_constitutive_sigma_panel -c "$CONFIG"'
    ) in runbook
