"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/config/test_workspace_study_specs_contracts.py

Contract tests for workspace study specs used by portfolio orchestration.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
WORKSPACES_ROOT = ROOT / "workspaces"
LENGTH_STUDY_MIN_BY_WORKSPACE: dict[str, int] = {
    "pairwise_laci_arac": 19,
}


def _workspace_config_paths() -> list[Path]:
    return sorted(WORKSPACES_ROOT.glob("*/configs/config.yaml"))


def _load_yaml(path: Path) -> dict:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _collect_factor_keys(study_payload: dict) -> set[str]:
    keys: set[str] = set()
    trials = study_payload.get("trials")
    assert isinstance(trials, list)
    for trial in trials:
        assert isinstance(trial, dict)
        factors = trial.get("factors")
        assert isinstance(factors, dict)
        keys.update(str(key) for key in factors.keys())
    trial_grids = study_payload.get("trial_grids", [])
    assert isinstance(trial_grids, list)
    for grid in trial_grids:
        assert isinstance(grid, dict)
        factors = grid.get("factors")
        assert isinstance(factors, dict)
        keys.update(str(key) for key in factors.keys())
    return keys


def _collect_sequence_lengths(study_payload: dict) -> list[int]:
    values: list[int] = []
    trials = study_payload.get("trials")
    assert isinstance(trials, list)
    for trial in trials:
        assert isinstance(trial, dict)
        factors = trial.get("factors")
        assert isinstance(factors, dict)
        if "sample.sequence_length" in factors:
            values.append(int(factors["sample.sequence_length"]))
    trial_grids = study_payload.get("trial_grids", [])
    assert isinstance(trial_grids, list)
    for grid in trial_grids:
        assert isinstance(grid, dict)
        factors = grid.get("factors")
        assert isinstance(factors, dict)
        lengths = factors.get("sample.sequence_length")
        if lengths is None:
            continue
        assert isinstance(lengths, list)
        values.extend(int(item) for item in lengths)
    return sorted(set(values))


def test_all_workspaces_define_length_and_diversity_study_specs() -> None:
    for config_path in _workspace_config_paths():
        workspace = config_path.parent.parent
        length_spec = workspace / "configs" / "studies" / "length_vs_score.study.yaml"
        diversity_spec = workspace / "configs" / "studies" / "diversity_vs_score.study.yaml"
        assert length_spec.exists(), f"{workspace.name}: missing {length_spec.name}"
        assert diversity_spec.exists(), f"{workspace.name}: missing {diversity_spec.name}"


def test_length_vs_score_studies_follow_workspace_study_range_policy() -> None:
    for config_path in _workspace_config_paths():
        workspace = config_path.parent.parent
        config_payload = _load_yaml(config_path)
        base_sequence_length = int(config_payload["cruncher"]["sample"]["sequence_length"])
        length_payload = _load_yaml(workspace / "configs" / "studies" / "length_vs_score.study.yaml")
        study = length_payload["study"]
        lengths = _collect_sequence_lengths(study)
        expected_min_length = LENGTH_STUDY_MIN_BY_WORKSPACE.get(workspace.name, 15)

        assert study["name"] == "length_vs_score", f"{workspace.name}: study.name must be length_vs_score"
        assert study["schema_version"] == 3, f"{workspace.name}: study.schema_version must be 3"
        assert study["base_config"] == "config.yaml", f"{workspace.name}: base_config must be config.yaml"
        assert lengths, f"{workspace.name}: length_vs_score defines no sample.sequence_length factors"
        assert lengths[0] == expected_min_length, (
            f"{workspace.name}: length_vs_score minimum {lengths[0]} must match study minimum policy "
            f"{expected_min_length}"
        )
        assert lengths[-1] <= max(50, base_sequence_length), (
            f"{workspace.name}: length_vs_score maximum must be <= max(50, base sequence length={base_sequence_length})"
        )
        expected_ladder = list(range(expected_min_length, 51, 2))
        assert all(item in lengths for item in expected_ladder), (
            f"{workspace.name}: length_vs_score must include every step-2 length from {expected_min_length} to 50"
        )
        assert base_sequence_length in lengths, (
            f"{workspace.name}: length_vs_score must include the base config sequence length {base_sequence_length}"
        )
        assert set(lengths).issubset(set(expected_ladder) | {base_sequence_length}), (
            f"{workspace.name}: length_vs_score must only include step-2 ladder values plus base config anchor"
        )


def test_diversity_vs_score_studies_inherit_workspace_sequence_length() -> None:
    for config_path in _workspace_config_paths():
        workspace = config_path.parent.parent
        payload = _load_yaml(workspace / "configs" / "studies" / "diversity_vs_score.study.yaml")
        study = payload["study"]
        factor_keys = _collect_factor_keys(study)

        assert study["name"] == "diversity_vs_score", f"{workspace.name}: study.name must be diversity_vs_score"
        assert study["schema_version"] == 3, f"{workspace.name}: study.schema_version must be 3"
        assert study["base_config"] == "config.yaml", f"{workspace.name}: base_config must be config.yaml"
        assert "sample.sequence_length" not in factor_keys, (
            f"{workspace.name}: diversity_vs_score should inherit sample.sequence_length from configs/config.yaml"
        )
