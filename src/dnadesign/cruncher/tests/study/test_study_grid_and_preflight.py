"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/study/test_study_grid_and_preflight.py

Validate Study trial-grid expansion and parse-readiness preflight contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import dnadesign.cruncher.app.study_workflow as study_workflow
from dnadesign.cruncher.app.study_workflow import run_study
from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.config.schema_v3 import CruncherConfig
from dnadesign.cruncher.study.layout import study_manifest_path
from dnadesign.cruncher.study.manifest import load_study_manifest
from dnadesign.cruncher.tests.study._helpers import write_study_spec, write_workspace_config


def _write_grid_spec(spec_path: Path) -> None:
    payload = {
        "study": {
            "schema_version": 3,
            "name": "smoke_study",
            "base_config": "config.yaml",
            "target": {"kind": "regulator_set", "set_index": 1},
            "execution": {
                "parallelism": 1,
                "on_trial_error": "continue",
                "exit_code_policy": "nonzero_if_any_error",
                "summarize_after_run": True,
            },
            "artifacts": {"trial_output_profile": "analysis_ready"},
            "replicates": {"seed_path": "sample.seed", "seeds": [11]},
            "trials": [],
            "trial_grids": [
                {
                    "id_prefix": "G",
                    "factors": {
                        "sample.sequence_length": [6, 7],
                        "sample.elites.select.diversity": [0.0, 0.5],
                    },
                }
            ],
            "replays": {
                "mmr_sweep": {
                    "enabled": False,
                    "pool_size_values": ["auto"],
                    "diversity_values": [0.0, 0.5, 1.0],
                }
            },
        }
    }
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(yaml.safe_dump(payload))


def test_study_run_expands_trial_grid(tmp_path: Path) -> None:
    write_workspace_config(tmp_path)
    spec_path = tmp_path / "grid.study.yaml"
    _write_grid_spec(spec_path)

    run_dir = run_study(spec_path)
    manifest = load_study_manifest(study_manifest_path(run_dir))
    assert len(manifest.trial_runs) == 4
    assert sorted({item.trial_id for item in manifest.trial_runs}) == ["G_1", "G_2", "G_3", "G_4"]


def test_study_parse_preflight_fails_before_trial_runs(tmp_path: Path) -> None:
    write_workspace_config(tmp_path)
    motif_path = tmp_path / ".cruncher" / "normalized" / "motifs" / "regulondb" / "RBM1.json"
    motif_path.write_text("{not-json")

    spec_path = tmp_path / "smoke.study.yaml"
    write_study_spec(
        spec_path,
        profile="analysis_ready",
        mmr_enabled=False,
        seeds=[11],
        trials=[{"id": "L6", "factors": {"sample.sequence_length": 6}}],
    )

    with pytest.raises(ValueError, match="parse readiness failed"):
        run_study(spec_path)
    assert not (tmp_path / "outputs" / "studies" / "smoke_study").exists()


def test_study_trial_contract_preflight_fails_before_trial_runs(tmp_path: Path) -> None:
    config_path = write_workspace_config(tmp_path)
    payload = yaml.safe_load(config_path.read_text())
    assert isinstance(payload, dict)
    cruncher = payload.get("cruncher")
    assert isinstance(cruncher, dict)
    sample = cruncher.get("sample")
    assert isinstance(sample, dict)
    sample["motif_width"] = {"maxw": 6}
    config_path.write_text(yaml.safe_dump(payload))

    spec_path = tmp_path / "trial_contracts.study.yaml"
    write_study_spec(
        spec_path,
        profile="analysis_ready",
        mmr_enabled=False,
        seeds=[11],
        trials=[
            {"id": "L6", "factors": {"sample.sequence_length": 6}},
            {"id": "L5", "factors": {"sample.sequence_length": 5}},
        ],
    )

    with pytest.raises(ValueError, match="sample.motif_width.maxw must be <= sample.sequence_length"):
        run_study(spec_path)
    assert not (tmp_path / "outputs" / "studies" / "smoke_study").exists()


def test_study_length_sweep_requires_base_config_value(tmp_path: Path) -> None:
    write_workspace_config(tmp_path)
    spec_path = tmp_path / "length_missing_base.study.yaml"
    write_study_spec(
        spec_path,
        profile="analysis_ready",
        mmr_enabled=False,
        seeds=[11],
        trials=[{"id": "L7", "factors": {"sample.sequence_length": 7}}],
    )

    with pytest.raises(ValueError, match="must include the base config value"):
        run_study(spec_path)
    assert not (tmp_path / "outputs" / "studies" / "smoke_study").exists()


def test_study_mmr_replay_requires_base_diversity_value(tmp_path: Path) -> None:
    write_workspace_config(tmp_path)
    spec_path = tmp_path / "mmr_missing_base.study.yaml"
    write_study_spec(
        spec_path,
        profile="analysis_ready",
        mmr_enabled=True,
        seeds=[11],
        trials=[{"id": "BASE", "factors": {"sample.sequence_length": 6}}],
    )
    payload = yaml.safe_load(spec_path.read_text())
    assert isinstance(payload, dict)
    study = payload.get("study")
    assert isinstance(study, dict)
    replays = study.get("replays")
    assert isinstance(replays, dict)
    mmr_sweep = replays.get("mmr_sweep")
    assert isinstance(mmr_sweep, dict)
    mmr_sweep["diversity_values"] = [0.5, 1.0]
    spec_path.write_text(yaml.safe_dump(payload))

    with pytest.raises(ValueError, match="replay diversity_values must include base config diversity"):
        run_study(spec_path)
    assert not (tmp_path / "outputs" / "studies" / "smoke_study").exists()


def test_study_grid_id_collision_fails_before_execution(tmp_path: Path) -> None:
    write_workspace_config(tmp_path)
    spec_path = tmp_path / "grid_collision.study.yaml"
    payload = {
        "study": {
            "schema_version": 3,
            "name": "smoke_study",
            "base_config": "config.yaml",
            "target": {"kind": "regulator_set", "set_index": 1},
            "execution": {
                "parallelism": 1,
                "on_trial_error": "continue",
                "exit_code_policy": "nonzero_if_any_error",
                "summarize_after_run": True,
            },
            "artifacts": {"trial_output_profile": "analysis_ready"},
            "replicates": {"seed_path": "sample.seed", "seeds": [11]},
            "trials": [{"id": "G_1", "factors": {"sample.sequence_length": 6}}],
            "trial_grids": [
                {
                    "id_prefix": "G",
                    "factors": {"sample.elites.select.diversity": [0.0]},
                }
            ],
            "replays": {
                "mmr_sweep": {
                    "enabled": False,
                    "pool_size_values": ["auto"],
                    "diversity_values": [0.0, 0.5, 1.0],
                }
            },
        }
    }
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(yaml.safe_dump(payload))

    with pytest.raises(ValueError, match="Duplicate trial id after grid expansion"):
        run_study(spec_path)
    assert not (tmp_path / "outputs" / "studies" / "smoke_study").exists()


def test_study_grid_expansion_caps_total_trial_count(tmp_path: Path) -> None:
    write_workspace_config(tmp_path)
    spec_path = tmp_path / "grid_large.study.yaml"
    payload = {
        "study": {
            "schema_version": 3,
            "name": "smoke_study",
            "base_config": "config.yaml",
            "target": {"kind": "regulator_set", "set_index": 1},
            "execution": {
                "parallelism": 1,
                "on_trial_error": "continue",
                "exit_code_policy": "nonzero_if_any_error",
                "summarize_after_run": True,
            },
            "artifacts": {"trial_output_profile": "analysis_ready"},
            "replicates": {"seed_path": "sample.seed", "seeds": [11]},
            "trials": [],
            "trial_grids": [
                {
                    "id_prefix": "G",
                    "factors": {"sample.sequence_length": list(range(300))},
                },
                {
                    "id_prefix": "H",
                    "factors": {"sample.optimizer.chains": list(range(1, 301))},
                },
            ],
            "replays": {
                "mmr_sweep": {
                    "enabled": False,
                    "pool_size_values": ["auto"],
                    "diversity_values": [0.0, 0.5, 1.0],
                }
            },
        }
    }
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(yaml.safe_dump(payload))

    with pytest.raises(ValueError, match="too many expanded trials"):
        run_study(spec_path)
    assert not (tmp_path / "outputs" / "studies" / "smoke_study").exists()


def test_preflight_reuses_pwm_validation_for_overlapping_target_sets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = write_workspace_config(tmp_path)
    base_cfg = load_config(config_path)
    payload = base_cfg.model_dump(mode="python")
    workspace_payload = payload.get("workspace")
    assert isinstance(workspace_payload, dict)
    workspace_payload["regulator_sets"] = [["lexA", "cpxR"], ["lexA", "cpxR"]]
    payload["workspace"] = workspace_payload
    cfg = CruncherConfig.model_validate(payload)

    monkeypatch.setattr(study_workflow, "target_statuses", lambda **kwargs: [])
    monkeypatch.setattr(study_workflow, "has_blocking_target_errors", lambda statuses: False)
    monkeypatch.setattr(study_workflow, "_lockmap_for", lambda *_args, **_kwargs: {"lexA": object(), "cpxR": object()})
    load_calls: list[list[str]] = []

    def _fake_load_pwms_for_set(*, tfs: list[str], **_kwargs):
        load_calls.append(list(tfs))
        return {}

    monkeypatch.setattr(study_workflow, "_load_pwms_for_set", _fake_load_pwms_for_set)

    study_workflow._ensure_lock_parse_and_targets_ready(cfg, config_path)
    assert load_calls == [["cpxR", "lexA"]]
