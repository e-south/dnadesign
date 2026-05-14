"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/tests/test_record_loader.py

Focused tests for fail-fast ops.study.yaml contract loading.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.ops.preflight.models import supported_preflight_check_kinds
from dnadesign.studies.core.record_loader import load_study_ops_contract


def _base_payload() -> dict[str, object]:
    return {
        "version": 2,
        "study_id": "demo_study",
        "family": "promoter",
        "title": "Demo study",
        "record_sources": {
            "narrative_ref": "manifest:status.md",
            "datasets_ref": "manifest:datasets.yaml",
            "pipeline_ref": "manifest:pipeline.yaml",
            "campaign_ref": "manifest:campaign.yaml",
        },
        "lifecycle": {
            "current_phase": {
                "strategy": "explicit",
                "id": "densegen_growth",
            },
            "phase_order": [
                "densegen_growth",
                "infer_batch_preparation",
            ],
        },
        "phases": [
            {
                "id": "densegen_growth",
                "status": "in_progress",
                "next_surface": "repo:src/dnadesign/ops/runbooks/presets/demo.yaml",
            },
            {
                "id": "infer_batch_preparation",
                "status": "planned",
                "next_surface": "repo:src/dnadesign/usr/docs/operations/promoter-study-preflight.md",
            },
        ],
        "execution_surfaces": {
            "densegen_batch": {
                "surface_type": "runbook",
                "runbook_ref": "repo:src/dnadesign/ops/runbooks/presets/demo.yaml",
            }
        },
        "snapshot": {"summary_scope": "repo"},
        "preflight": {
            "default_scope": "next",
            "scopes": {
                "next": {"include_phases": ["current_phase", "next_in_progress_phase"]},
                "full": {"include_phases": ["all"]},
            },
            "group_phase_bindings": {
                "densegen": "densegen_growth",
                "notify_environment": "infer_batch_preparation",
            },
            "checks": {
                "densegen_growth": [
                    {
                        "kind": "runbook_plan",
                        "check_id": "densegen.batch.plan",
                        "check_group": "densegen",
                        "summary": "DenseGen batch runbook renders cleanly.",
                        "required": True,
                        "surface": "densegen_batch",
                    }
                ],
                "infer_batch_preparation": [
                    {
                        "kind": "environment",
                        "check_id": "notify.environment.contract",
                        "check_group": "notify_environment",
                        "summary": "Notify environment variables are present.",
                        "required": False,
                        "vars": ["NOTIFY_WEBHOOK", "NOTIFY_WEBHOOK_FILE"],
                        "match_mode": "any",
                    }
                ],
            },
            "next_scope": {
                "target_phase_groups": {
                    "densegen_growth": ["densegen"],
                    "infer_batch_preparation": ["notify_environment"],
                },
                "runtime_phase_groups": [],
                "runtime_shared_groups": ["notify_environment"],
            },
        },
    }


def _write_contract(tmp_path: Path, payload: dict[str, object]) -> Path:
    repo_root = tmp_path
    (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.0.0'\n", encoding="utf-8")
    study_root = repo_root / "docs" / "studies" / "demo_study"
    study_root.mkdir(parents=True, exist_ok=True)
    (study_root / "ops.study.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return study_root


def test_load_study_ops_contract_accepts_valid_repo_scoped_surface_refs(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["phases"][1]["required_for_main_study_state"] = False
    study_root = _write_contract(tmp_path, payload)

    contract = load_study_ops_contract(study_root)

    assert contract.snapshot_summary_scope == "repo"
    assert contract.phases[0].next_surface == "repo:src/dnadesign/ops/runbooks/presets/demo.yaml"
    assert contract.phases[0].required_for_main_study_state is True
    assert contract.phases[1].required_for_main_study_state is False
    assert contract.phases[1].as_dict()["required_for_main_study_state"] is False


def test_load_study_ops_contract_accepts_nonsequential_tracks(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["lifecycle"] = {
        "mode": "tracks",
        "current_track": {
            "strategy": "explicit",
            "id": "design_reference_catalog",
        },
        "track_order": [
            "design_reference_catalog",
            "source_ref_dogfood",
        ],
    }
    payload["tracks"] = [
        {
            "id": "design_reference_catalog",
            "status": "in_progress",
            "next_surface": "repo:src/dnadesign/ops/runbooks/presets/demo.yaml",
        },
        {
            "id": "source_ref_dogfood",
            "status": "planned",
            "next_surface": "repo:src/dnadesign/usr/docs/operations/promoter-study-preflight.md",
        },
    ]
    del payload["phases"]
    payload["preflight"]["scopes"] = {
        "next": {"include_tracks": ["current_track"]},
        "full": {"include_tracks": ["all"]},
    }
    payload["preflight"]["group_track_bindings"] = {
        "design_catalog": "design_reference_catalog",
        "source_ref": "source_ref_dogfood",
    }
    del payload["preflight"]["group_phase_bindings"]
    payload["preflight"]["checks"] = {
        "design_reference_catalog": [
            {
                "kind": "runbook_plan",
                "check_id": "design.catalog.plan",
                "check_group": "design_catalog",
                "summary": "Design catalog route renders cleanly.",
                "required": True,
                "surface": "densegen_batch",
            }
        ]
    }
    payload["preflight"]["next_scope"] = {
        "target_track_groups": {
            "design_reference_catalog": ["design_catalog"],
            "source_ref_dogfood": ["source_ref"],
        },
        "runtime_track_groups": [],
        "runtime_shared_groups": [],
    }
    study_root = _write_contract(tmp_path, payload)

    contract = load_study_ops_contract(study_root)

    assert contract.lifecycle_mode == "tracks"
    assert contract.lifecycle_item_label == "track"
    assert contract.current_phase_id == "design_reference_catalog"
    assert contract.phase_order == ("design_reference_catalog", "source_ref_dogfood")
    assert contract.preflight.group_phase_bindings == {
        "design_catalog": "design_reference_catalog",
        "source_ref": "source_ref_dogfood",
    }
    assert contract.preflight.next_scope.target_phase_groups["design_reference_catalog"] == ("design_catalog",)


def test_supported_preflight_kinds_come_from_single_registry() -> None:
    from dnadesign.studies.core import record_loader

    assert not hasattr(record_loader, "_SUPPORTED_PREFLIGHT_CHECK_KINDS")
    assert "scheduler_queue" in supported_preflight_check_kinds()
    assert "infer_validate_config" not in supported_preflight_check_kinds()


def test_load_study_ops_contract_accepts_generic_command_and_scheduler_queue_checks(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["lifecycle"]["phase_order"] = [
        "densegen_growth",
        "infer_batch_preparation",
        "infer_anchor_only_20b",
    ]
    payload["phases"].append(
        {
            "id": "infer_anchor_only_20b",
            "status": "planned",
            "next_surface": "repo:src/dnadesign/ops/runbooks/presets/infer_anchor_only_20b.yaml",
        }
    )
    payload["execution_surfaces"]["infer_validate_anchor_only_20b"] = {
        "surface_type": "command",
        "argv": [
            "uv",
            "run",
            "infer",
            "validate",
            "config",
            "--config",
            "workspace/infer/config.anchor_only.evo2_20b.yaml",
        ],
    }
    payload["execution_surfaces"]["scheduler_default"] = {
        "surface_type": "scheduler",
        "backend": "sge",
    }
    payload["preflight"]["checks"]["infer_batch_preparation"] = [
        {
            "kind": "environment",
            "check_id": "notify.environment.contract",
            "check_group": "notify_environment",
            "summary": "Notify environment variables are present.",
            "required": False,
            "vars": ["NOTIFY_WEBHOOK", "NOTIFY_WEBHOOK_FILE"],
            "match_mode": "any",
        },
        {
            "kind": "command",
            "check_id": "infer.validate.anchor_only_20b",
            "check_group": "infer",
            "summary": "Infer config validation completed.",
            "required": True,
            "phase_id": "infer_anchor_only_20b",
            "surface": "infer_validate_anchor_only_20b",
        },
        {
            "kind": "scheduler_queue",
            "check_id": "infer.batch.queue",
            "check_group": "infer_batch_plan",
            "summary": "Scheduler queue is below the declared submit thresholds.",
            "required": False,
            "surface": "scheduler_default",
            "max_running_jobs": 3,
        },
    ]
    payload["preflight"]["next_scope"]["target_phase_groups"]["infer_batch_preparation"] = [
        "notify_environment",
        "infer",
        "infer_batch_plan",
    ]
    payload["preflight"]["next_scope"]["runtime_phase_groups"] = ["infer", "infer_batch_plan"]

    study_root = _write_contract(tmp_path, payload)

    contract = load_study_ops_contract(study_root)

    infer_checks = contract.preflight.check_specs["infer_batch_preparation"]
    assert [spec["kind"] for spec in infer_checks[1:]] == [
        "command",
        "scheduler_queue",
    ]
    assert infer_checks[1]["surface"] == "infer_validate_anchor_only_20b"
    assert infer_checks[2]["surface"] == "scheduler_default"


def test_load_study_ops_contract_allows_repeated_tokens_in_command_argv(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["execution_surfaces"]["infer_dry_run_anchor_only_20b"] = {
        "surface_type": "command",
        "argv": [
            "uv",
            "run",
            "infer",
            "run",
            "--config",
            "workspace/infer/config.anchor_only.evo2_20b.yaml",
            "--dry-run",
        ],
    }

    study_root = _write_contract(tmp_path, payload)

    contract = load_study_ops_contract(study_root)

    assert contract.execution_surfaces["infer_dry_run_anchor_only_20b"]["argv"] == [
        "uv",
        "run",
        "infer",
        "run",
        "--config",
        "workspace/infer/config.anchor_only.evo2_20b.yaml",
        "--dry-run",
    ]


def test_load_study_ops_contract_rejects_invalid_command_cwd_ref(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["execution_surfaces"]["densegen_batch_probe"] = {
        "surface_type": "command",
        "argv": ["uv", "run", "python", "-c", "print('ok')"],
        "cwd_ref": "<replace-me>",
    }
    payload["preflight"]["checks"]["densegen_growth"].append(
        {
            "kind": "command",
            "check_id": "densegen.batch.probe",
            "check_group": "densegen",
            "summary": "DenseGen batch probe completed.",
            "required": True,
            "surface": "densegen_batch_probe",
        }
    )
    study_root = _write_contract(tmp_path, payload)

    with pytest.raises(ValueError, match="contains placeholder path text"):
        load_study_ops_contract(study_root)


def test_load_study_ops_contract_rejects_unknown_phase_status(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["phases"][0]["status"] = "later"
    study_root = _write_contract(tmp_path, payload)

    with pytest.raises(ValueError, match="unsupported status"):
        load_study_ops_contract(study_root)


def test_load_study_ops_contract_rejects_non_boolean_main_state_flag(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["phases"][1]["required_for_main_study_state"] = "false"
    study_root = _write_contract(tmp_path, payload)

    with pytest.raises(ValueError, match="required_for_main_study_state must be boolean"):
        load_study_ops_contract(study_root)


def test_load_study_ops_contract_rejects_invalid_snapshot_summary_scope(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["snapshot"]["summary_scope"] = "galaxy"
    study_root = _write_contract(tmp_path, payload)

    with pytest.raises(ValueError, match="snapshot.summary_scope must be one of"):
        load_study_ops_contract(study_root)


def test_load_study_ops_contract_rejects_invalid_default_scope(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["preflight"]["default_scope"] = "later"
    study_root = _write_contract(tmp_path, payload)

    with pytest.raises(ValueError, match="preflight.default_scope must be one of"):
        load_study_ops_contract(study_root)


def test_load_study_ops_contract_rejects_unknown_group_phase_binding(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["preflight"]["group_phase_bindings"]["densegen"] = "missing_phase"
    study_root = _write_contract(tmp_path, payload)

    with pytest.raises(ValueError, match="references undeclared phase 'missing_phase'"):
        load_study_ops_contract(study_root)


def test_load_study_ops_contract_rejects_invalid_next_scope_phase_reference(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["preflight"]["next_scope"]["target_phase_groups"]["missing_phase"] = ["densegen"]
    study_root = _write_contract(tmp_path, payload)

    with pytest.raises(ValueError, match="target_phase_groups references undeclared phase 'missing_phase'"):
        load_study_ops_contract(study_root)


def test_load_study_ops_contract_rejects_placeholder_next_surface(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["phases"][0]["next_surface"] = "<replace-me>"
    study_root = _write_contract(tmp_path, payload)

    with pytest.raises(ValueError, match="contains placeholder path text"):
        load_study_ops_contract(study_root)


def test_load_study_ops_contract_rejects_duplicate_preflight_check_ids(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["preflight"]["checks"]["infer_batch_preparation"][0]["check_id"] = "densegen.batch.plan"
    study_root = _write_contract(tmp_path, payload)

    with pytest.raises(ValueError, match="must not duplicate check_id"):
        load_study_ops_contract(study_root)


def test_load_study_ops_contract_rejects_specialized_preflight_kind_ids(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["preflight"]["checks"]["infer_batch_preparation"].append(
        {
            "kind": "infer_validate_config",
            "check_id": "infer.validate.anchor_only_20b",
            "check_group": "notify_environment",
            "summary": "Legacy specialized kind should fail fast.",
            "required": True,
            "phase_id": "infer_batch_preparation",
            "config_label": "anchor_only_20b",
        }
    )
    study_root = _write_contract(tmp_path, payload)

    with pytest.raises(ValueError, match="unsupported kind 'infer_validate_config'"):
        load_study_ops_contract(study_root)


def test_load_study_ops_contract_rejects_unknown_preflight_surface_reference(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["preflight"]["checks"]["densegen_growth"][0]["surface"] = "missing_surface"
    study_root = _write_contract(tmp_path, payload)

    with pytest.raises(ValueError, match="references unknown surface 'missing_surface'"):
        load_study_ops_contract(study_root)


def test_load_study_ops_contract_rejects_environment_checks_without_vars(tmp_path: Path) -> None:
    payload = _base_payload()
    del payload["preflight"]["checks"]["infer_batch_preparation"][0]["vars"]
    study_root = _write_contract(tmp_path, payload)

    with pytest.raises(ValueError, match="entry notify.environment.contract vars must be a list"):
        load_study_ops_contract(study_root)


def test_load_study_ops_contract_rejects_unknown_preflight_check_group(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["preflight"]["checks"]["densegen_growth"][0]["check_group"] = "missing_group"
    study_root = _write_contract(tmp_path, payload)

    with pytest.raises(ValueError, match="references unknown check_group 'missing_group'"):
        load_study_ops_contract(study_root)


def test_load_study_ops_contract_rejects_unknown_preflight_check_phase_override(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["preflight"]["checks"]["infer_batch_preparation"][0]["phase_id"] = "missing_phase"
    study_root = _write_contract(tmp_path, payload)

    with pytest.raises(ValueError, match="references undeclared phase_id 'missing_phase'"):
        load_study_ops_contract(study_root)


def test_load_study_ops_contract_rejects_invalid_environment_match_mode(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["preflight"]["checks"]["infer_batch_preparation"][0]["match_mode"] = "later"
    study_root = _write_contract(tmp_path, payload)

    with pytest.raises(ValueError, match="match_mode must be one of: all, any"):
        load_study_ops_contract(study_root)
