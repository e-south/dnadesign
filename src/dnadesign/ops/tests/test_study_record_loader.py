"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/tests/test_study_record_loader.py

Verify strict loading of study operations records.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.ops.preflight import supported_preflight_check_kinds
from dnadesign.ops.study import load_study_ops_contract


def _base_payload() -> dict[str, object]:
    return {
        "version": 2,
        "study_id": "demo_study",
        "ops_surfaces": {
            "status_kind": "demo-study-status",
            "preflight_kind": "demo-study-preflight",
        },
        "title": "Demo study",
        "record_sources": {"narrative_ref": "manifest:record/status.md"},
        "artifacts": {
            "status_note": {"artifact_type": "file", "ref": "manifest:record/status.md"},
        },
        "execution_surfaces": {
            "validate_status": {
                "surface_type": "command",
                "argv": ["python", "-c", "print('ok')"],
            },
        },
        "snapshot": {"summary_scope": "repo"},
        "preflight": {
            "default_scope": "next",
            "scopes": {
                "next": {"include_groups": ["study_record"]},
                "full": {"include_groups": ["all"]},
            },
            "checks": {
                "study_record": [
                    {
                        "kind": "path_exists",
                        "check_id": "study.record.present",
                        "check_group": "study_record",
                        "summary": "Study record is present.",
                        "required": True,
                        "artifact": "status_note",
                    },
                    {
                        "kind": "command",
                        "check_id": "study.record.valid",
                        "check_group": "study_record",
                        "summary": "Study record validates.",
                        "required": True,
                        "surface": "validate_status",
                    },
                ],
            },
        },
    }


def _write_contract(tmp_path: Path, payload: dict[str, object]) -> Path:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.0.0'\n", encoding="utf-8")
    study_root = tmp_path / "studies" / "demo-study"
    (study_root / "operations").mkdir(parents=True)
    (study_root / "record").mkdir()
    (study_root / "record" / "status.md").write_text("# Status\n", encoding="utf-8")
    (study_root / "operations" / "ops.study.yaml").write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return study_root


def test_loads_explicit_scope_contract(tmp_path: Path) -> None:
    contract = load_study_ops_contract(_write_contract(tmp_path, _base_payload()))

    assert contract.study_id == "demo_study"
    assert contract.snapshot_summary_scope == "repo"
    assert contract.preflight.scope_groups == {
        "next": ("study_record",),
        "full": ("study_record",),
    }
    assert tuple(contract.preflight.check_specs) == ("study_record",)


def test_full_scope_expands_all_declared_check_groups(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["preflight"]["checks"]["optional_review"] = [
        {
            "kind": "command",
            "check_id": "study.optional-review.valid",
            "check_group": "optional_review",
            "summary": "Optional review validates.",
            "required": False,
            "surface": "validate_status",
        }
    ]

    contract = load_study_ops_contract(_write_contract(tmp_path, payload))

    assert contract.preflight.scope_groups == {
        "next": ("study_record",),
        "full": ("study_record", "optional_review"),
    }


def test_accepts_split_mapping_parts(tmp_path: Path) -> None:
    payload = _base_payload()
    sections = {key: payload.pop(key) for key in ("artifacts", "execution_surfaces", "snapshot", "preflight")}
    payload["parts"] = {
        "artifacts": "contract/artifacts.yaml",
        "execution_surfaces": "contract/execution.yaml",
        "snapshot": "contract/snapshot.yaml",
        "preflight": "contract/preflight.yaml",
    }
    study_root = _write_contract(tmp_path, payload)
    for section, relative in payload["parts"].items():
        path = study_root / "operations" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(sections[section], sort_keys=False), encoding="utf-8")

    contract = load_study_ops_contract(study_root)

    assert contract.artifacts["status_note"]["artifact_type"] == "file"
    assert contract.execution_surfaces["validate_status"]["surface_type"] == "command"


@pytest.mark.parametrize("legacy_key", ["lifecycle", "phases", "tracks"])
def test_rejects_lifecycle_control_plane_keys(tmp_path: Path, legacy_key: str) -> None:
    payload = _base_payload()
    payload[legacy_key] = {}

    with pytest.raises(ValueError, match=rf"unknown key\(s\) {legacy_key}"):
        load_study_ops_contract(_write_contract(tmp_path, payload))


@pytest.mark.parametrize("legacy_key", ["group_phase_bindings", "group_track_bindings", "next_scope"])
def test_rejects_lifecycle_preflight_keys(tmp_path: Path, legacy_key: str) -> None:
    payload = _base_payload()
    payload["preflight"][legacy_key] = {}

    with pytest.raises(ValueError, match="must use scopes"):
        load_study_ops_contract(_write_contract(tmp_path, payload))


def test_requires_both_named_scopes(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["preflight"]["scopes"].pop("full")

    with pytest.raises(ValueError, match="must define full"):
        load_study_ops_contract(_write_contract(tmp_path, payload))


def test_full_scope_must_be_all(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["preflight"]["scopes"]["full"] = {"include_groups": ["study_record"]}

    with pytest.raises(ValueError, match=r"must be \[all\]"):
        load_study_ops_contract(_write_contract(tmp_path, payload))


def test_rejects_next_scope_group_without_checks(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["preflight"]["scopes"]["next"] = {"include_groups": ["unknown"]}

    with pytest.raises(ValueError, match="unknown check_group"):
        load_study_ops_contract(_write_contract(tmp_path, payload))


def test_rejects_unknown_artifact_reference(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["preflight"]["checks"]["study_record"][0]["artifact"] = "missing"

    with pytest.raises(ValueError, match="unknown artifact"):
        load_study_ops_contract(_write_contract(tmp_path, payload))


@pytest.mark.parametrize("legacy_key", ["phase_id", "phase"])
def test_rejects_legacy_preflight_check_phase_fields(tmp_path: Path, legacy_key: str) -> None:
    payload = _base_payload()
    payload["preflight"]["checks"]["study_record"][0][legacy_key] = "study_record"

    with pytest.raises(ValueError, match=rf"preflight check.*legacy key\(s\) {legacy_key}"):
        load_study_ops_contract(_write_contract(tmp_path, payload))


def test_supported_preflight_kinds_come_from_single_registry() -> None:
    assert "scheduler_queue" in supported_preflight_check_kinds()
    assert "infer_validate_config" not in supported_preflight_check_kinds()
