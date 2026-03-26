"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/tests/test_record_loader.py

Focused tests for fail-fast ops.study.yaml contract loading.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

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
                        "summary": "DenseGen batch runbook renders cleanly.",
                        "required": True,
                        "surface": "densegen_batch",
                    }
                ],
                "infer_batch_preparation": [
                    {
                        "kind": "environment",
                        "check_id": "notify.environment.contract",
                        "summary": "Notify environment variables are present.",
                        "required": False,
                        "vars": ["SLACK_WEBHOOK_URL", "SLACK_BOT_TOKEN"],
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
    study_root = _write_contract(tmp_path, _base_payload())

    contract = load_study_ops_contract(study_root)

    assert contract.snapshot_summary_scope == "repo"
    assert contract.phases[0].next_surface == "repo:src/dnadesign/ops/runbooks/presets/demo.yaml"


def test_load_study_ops_contract_rejects_unknown_phase_status(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["phases"][0]["status"] = "later"
    study_root = _write_contract(tmp_path, payload)

    with pytest.raises(ValueError, match="unsupported status"):
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
