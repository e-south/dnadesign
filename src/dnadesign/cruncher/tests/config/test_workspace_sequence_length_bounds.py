"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/config/test_workspace_sequence_length_bounds.py

Contracts for workspace sequence-length defaults and study length ranges.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

WORKSPACES_ROOT = Path(__file__).resolve().parents[2] / "workspaces"

# Workspace default sequence length policy for one-off optimization.
EXPECTED_SEQUENCE_LENGTHS = {
    "demo_multitf": 18,
    "demo_pairwise": 18,
    "multitf_baer_lexa_soxr": 18,
    "multitf_baer_lexa_soxr_soxs": 18,
    "multitf_cpxr_baer_lexa": 18,
    "pairwise_baer_lexa": 18,
    "pairwise_baer_soxr": 18,
    "pairwise_cpxr_baer": 18,
    "pairwise_cpxr_lexa": 18,
    "pairwise_cpxr_soxr": 18,
    "pairwise_laci_arac": 19,
    "pairwise_soxr_soxs": 18,
    "project_tfs_lexa_cpxr_baer_rcda_lrp_fur_fnr_acrr_soxr_soxs": 18,
}


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_workspace_configs_use_policy_sequence_lengths() -> None:
    for workspace_name, expected_length in sorted(EXPECTED_SEQUENCE_LENGTHS.items()):
        config_path = WORKSPACES_ROOT / workspace_name / "configs" / "config.yaml"
        payload = _load_yaml(config_path)
        sample = payload["cruncher"]["sample"]
        assert sample["sequence_length"] == expected_length, (
            f"{workspace_name}: sample.sequence_length should match workspace policy bound={expected_length}"
        )


def test_demo_pairwise_length_study_spans_workspace_min_to_fifty() -> None:
    spec = _load_yaml(WORKSPACES_ROOT / "demo_pairwise" / "configs" / "studies" / "length_vs_score.study.yaml")
    grids = spec["study"]["trial_grids"]
    values = grids[0]["factors"]["sample.sequence_length"]
    assert values[0] == 15
    assert values[-1] == 49
    assert 18 in values
    expected_ladder = list(range(15, 50, 2))
    assert all(item in values for item in expected_ladder)
    assert set(values).issubset(set(expected_ladder) | {18})


def test_demo_pairwise_diversity_study_uses_workspace_min_sequence_length() -> None:
    spec = _load_yaml(WORKSPACES_ROOT / "demo_pairwise" / "configs" / "studies" / "diversity_vs_score.study.yaml")
    trial = spec["study"]["trials"][0]
    assert "sample.sequence_length" not in trial["factors"]
