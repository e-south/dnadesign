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

from dnadesign.ops.preflight import supported_preflight_check_kinds
from dnadesign.studies.core.record_loader import load_study_ops_contract


def _base_payload() -> dict[str, object]:
    return {
        "version": 2,
        "study_id": "demo_study",
        "ops_surfaces": {
            "status_kind": "demo-study-status",
            "preflight_kind": "demo-study-preflight",
        },
        "title": "Demo study",
        "record_sources": {
            "narrative_ref": "manifest:record/status.md",
            "datasets_ref": "manifest:record/datasets.yaml",
            "pipeline_ref": "manifest:operations/runtime/command-groups/pipeline.yaml",
            "campaign_ref": "manifest:record/campaign.yaml",
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
                "next_surface": "repo:docs/studies/demo_study/preflight.md",
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
    (study_root / "operations").mkdir(parents=True, exist_ok=True)
    (study_root / "operations" / "ops.study.yaml").write_text(
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
    assert contract.status_kind == "demo-study-status"
    assert contract.preflight_kind == "demo-study-preflight"
    assert contract.phases[0].next_surface == "repo:src/dnadesign/ops/runbooks/presets/demo.yaml"
    assert contract.phases[0].required_for_main_study_state is True
    assert contract.phases[1].required_for_main_study_state is False
    assert contract.phases[1].as_dict()["required_for_main_study_state"] is False


def test_load_study_ops_contract_accepts_split_parts(tmp_path: Path) -> None:
    payload = _base_payload()
    split_keys = (
        "lifecycle",
        "phases",
        "artifacts",
        "execution_surfaces",
        "snapshot",
        "preflight",
    )
    payload["artifacts"] = {
        "status_note": {
            "artifact_type": "file",
            "ref": "manifest:record/status.md",
        }
    }
    root_payload = {key: value for key, value in payload.items() if key not in split_keys}
    root_payload["parts"] = {
        "lifecycle": "contract/lifecycle/mode.yaml",
        "phases": "contract/lifecycle/phases.yaml",
        "artifacts": "contract/surfaces/artifacts.yaml",
        "execution_surfaces": "contract/surfaces/execution_surfaces.yaml",
        "snapshot": "contract/status/snapshot.yaml",
        "preflight": "contract/readiness/preflight.yaml",
    }
    study_root = _write_contract(tmp_path, root_payload)
    contract_root = study_root / "operations" / "contract"
    for key, rel_path in root_payload["parts"].items():
        part_path = contract_root / rel_path.removeprefix("contract/")
        part_path.parent.mkdir(parents=True, exist_ok=True)
        part_path.write_text(
            yaml.safe_dump(payload[key], sort_keys=False),
            encoding="utf-8",
        )

    contract = load_study_ops_contract(study_root)

    assert contract.current_phase_id == "densegen_growth"
    assert contract.phase_order == ("densegen_growth", "infer_batch_preparation")
    assert contract.artifacts["status_note"]["ref"] == "manifest:record/status.md"
    assert contract.execution_surfaces["densegen_batch"]["surface_type"] == "runbook"


def test_load_study_ops_contract_accepts_multi_file_split_parts(tmp_path: Path) -> None:
    payload = _base_payload()
    split_keys = ("execution_surfaces", "preflight")
    root_payload = {key: value for key, value in payload.items() if key not in split_keys}
    root_payload["parts"] = {
        "execution_surfaces": [
            "contract/surfaces/execution/runbooks.yaml",
            "contract/surfaces/execution/commands.yaml",
        ],
        "preflight": [
            "contract/readiness/scope.yaml",
            "contract/readiness/groups.yaml",
            "contract/readiness/checks/densegen.yaml",
            "contract/readiness/checks/notify.yaml",
        ],
    }
    study_root = _write_contract(tmp_path, root_payload)
    operations_root = study_root / "operations"
    split_payloads = {
        "contract/surfaces/execution/runbooks.yaml": {
            "densegen_batch": payload["execution_surfaces"]["densegen_batch"],
        },
        "contract/surfaces/execution/commands.yaml": {
            "notify_profile_doctor": {
                "surface_type": "command",
                "argv": ["uv", "run", "notify", "profile", "doctor"],
            },
        },
        "contract/readiness/scope.yaml": {
            "default_scope": payload["preflight"]["default_scope"],
            "scopes": payload["preflight"]["scopes"],
        },
        "contract/readiness/groups.yaml": {
            "group_phase_bindings": payload["preflight"]["group_phase_bindings"],
            "next_scope": payload["preflight"]["next_scope"],
        },
        "contract/readiness/checks/densegen.yaml": {
            "checks": {"densegen_growth": payload["preflight"]["checks"]["densegen_growth"]},
        },
        "contract/readiness/checks/notify.yaml": {
            "checks": {
                "infer_batch_preparation": [
                    *payload["preflight"]["checks"]["infer_batch_preparation"],
                    {
                        "kind": "command",
                        "check_id": "notify.profile.doctor",
                        "check_group": "notify_environment",
                        "summary": "Notify profile doctor runs.",
                        "required": False,
                        "surface": "notify_profile_doctor",
                    },
                ]
            },
        },
    }
    for rel_path, part_payload in split_payloads.items():
        part_path = operations_root / rel_path
        part_path.parent.mkdir(parents=True, exist_ok=True)
        part_path.write_text(yaml.safe_dump(part_payload, sort_keys=False), encoding="utf-8")

    contract = load_study_ops_contract(study_root)

    assert contract.execution_surfaces["densegen_batch"]["surface_type"] == "runbook"
    assert contract.execution_surfaces["notify_profile_doctor"]["surface_type"] == "command"
    assert len(contract.preflight.check_specs["infer_batch_preparation"]) == 2
    assert contract.preflight.check_specs["infer_batch_preparation"][1]["check_id"] == "notify.profile.doctor"


def test_load_study_ops_contract_rejects_split_part_that_duplicates_inline_section(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["parts"] = {"preflight": "contract/readiness/preflight.yaml"}
    study_root = _write_contract(tmp_path, payload)
    contract_root = study_root / "operations" / "contract"
    (contract_root / "readiness").mkdir(parents=True)
    (contract_root / "readiness" / "preflight.yaml").write_text("default_scope: next\n", encoding="utf-8")

    with pytest.raises(ValueError, match="duplicates an inline preflight section"):
        load_study_ops_contract(study_root)


def test_load_study_ops_contract_rejects_legacy_family_key(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["family"] = "promoter"
    study_root = _write_contract(tmp_path, payload)

    with pytest.raises(ValueError, match="must not define legacy family"):
        load_study_ops_contract(study_root)


def test_load_study_ops_contract_rejects_unknown_top_level_keys(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["status_adapters"] = {"demo": "legacy"}
    study_root = _write_contract(tmp_path, payload)

    with pytest.raises(ValueError, match="unknown key\\(s\\) status_adapters"):
        load_study_ops_contract(study_root)


def test_load_study_ops_contract_rejects_unknown_ops_surface_keys(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["ops_surfaces"]["status_adapter"] = "legacy"
    study_root = _write_contract(tmp_path, payload)

    with pytest.raises(ValueError, match="ops_surfaces contains unknown key\\(s\\) status_adapter"):
        load_study_ops_contract(study_root)


def test_load_study_ops_contract_allows_studies_without_status_provider(tmp_path: Path) -> None:
    payload = _base_payload()
    payload.pop("ops_surfaces")
    study_root = _write_contract(tmp_path, payload)

    contract = load_study_ops_contract(study_root)

    assert contract.status_kind is None
    assert contract.preflight_kind is None


def test_load_study_ops_contract_requires_status_and_preflight_surfaces_together(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["ops_surfaces"] = {"status_kind": "demo-study-status"}
    study_root = _write_contract(tmp_path, payload)

    with pytest.raises(ValueError, match="must be declared together"):
        load_study_ops_contract(study_root)


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
            "next_surface": "repo:docs/studies/demo_study/preflight.md",
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
