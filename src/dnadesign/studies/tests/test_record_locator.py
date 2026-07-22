"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/tests/test_record_locator.py

Focused tests for flat checked-in study index loading and active-study.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.studies.core.record_loader import load_study_ops_contract
from dnadesign.studies.core.record_locator import (
    discover_active_study_selection,
    discover_study_selection_for_status_kind,
)
from dnadesign.studies.core.registry import load_study_index


def _repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise AssertionError("Could not locate repository root")


def _write_repo(tmp_path: Path) -> Path:
    repo_root = tmp_path
    (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.0.0'\n", encoding="utf-8")
    _write_minimal_ops_contract(
        repo_root / "docs" / "studies" / "demo_study",
        study_id="demo_study",
        status_kind="demo-study-status",
        preflight_kind="demo-study-preflight",
    )
    (repo_root / "docs" / "studies" / "index.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "active_study_id": "demo_study",
                "studies": [
                    {
                        "study_id": "demo_study",
                        "title": "Demo study",
                        "record_root": "docs/studies/demo_study",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return repo_root


def _write_minimal_ops_contract(
    study_root: Path,
    *,
    study_id: str,
    status_kind: str,
    preflight_kind: str,
) -> None:
    (study_root / "operations").mkdir(parents=True, exist_ok=True)
    (study_root / "operations" / "ops.study.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 2,
                "study_id": study_id,
                "ops_surfaces": {
                    "status_kind": status_kind,
                    "preflight_kind": preflight_kind,
                },
                "lifecycle": {
                    "phase_order": ["ready"],
                    "current_phase": {"strategy": "explicit", "id": "ready"},
                },
                "phases": [{"id": "ready", "status": "ready"}],
                "snapshot": {"summary_scope": "repo"},
                "preflight": {
                    "default_scope": "next",
                    "scopes": {"next": {"include_phases": ["current_phase"]}},
                    "group_phase_bindings": {"study_record": "ready"},
                    "next_scope": {
                        "target_phase_groups": {"ready": ["study_record"]},
                        "runtime_phase_groups": [],
                        "runtime_shared_groups": [],
                    },
                    "checks": {},
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_load_study_index_reads_flat_study_first_layout(tmp_path: Path) -> None:
    repo_root = _write_repo(tmp_path)

    index = load_study_index(repo_root)

    assert index.active_study_id == "demo_study"
    assert index.studies[0].record_root == (repo_root / "docs" / "studies" / "demo_study").resolve()


def test_indexed_study_records_have_loadable_ops_contracts() -> None:
    index = load_study_index(_repo_root())

    for entry in index.studies:
        contract = load_study_ops_contract(entry.record_root)

        assert contract.study_id == entry.study_id


def test_discover_active_study_selection_uses_top_level_studies_index(tmp_path: Path) -> None:
    repo_root = _write_repo(tmp_path)

    selection = discover_active_study_selection(
        repo_root=repo_root,
        status_kind="demo-study-status",
    )

    assert selection.active_study_id == "demo_study"
    assert selection.index_path == (repo_root / "docs" / "studies" / "index.yaml").resolve()
    assert selection.study_root == (repo_root / "docs" / "studies" / "demo_study").resolve()


def test_discover_study_selection_for_status_kind_uses_matching_ops_surface(tmp_path: Path) -> None:
    repo_root = _write_repo(tmp_path)
    _write_minimal_ops_contract(
        repo_root / "docs" / "studies" / "retron_hairpin_design",
        study_id="retron_hairpin_design",
        status_kind="retron-hairpin-design-status",
        preflight_kind="retron-hairpin-design-preflight",
    )
    index_path = repo_root / "docs" / "studies" / "index.yaml"
    index_payload = yaml.safe_load(index_path.read_text(encoding="utf-8"))
    assert isinstance(index_payload, dict)
    index_payload["studies"].append(
        {
            "study_id": "retron_hairpin_design",
            "title": "Retron hairpin design",
            "record_root": "docs/studies/retron_hairpin_design",
        }
    )
    index_path.write_text(yaml.safe_dump(index_payload, sort_keys=False), encoding="utf-8")

    selection = discover_study_selection_for_status_kind(
        repo_root=repo_root,
        status_kind="retron-hairpin-design-status",
    )

    assert selection.active_study_id == "demo_study"
    assert selection.entry.study_id == "retron_hairpin_design"
    assert selection.study_root == (repo_root / "docs" / "studies" / "retron_hairpin_design").resolve()


def test_load_study_index_rejects_legacy_family_field(tmp_path: Path) -> None:
    repo_root = _write_repo(tmp_path)
    index_path = repo_root / "docs" / "studies" / "index.yaml"
    index_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "active_study_id": "demo_study",
                "studies": [
                    {
                        "study_id": "demo_study",
                        "family": "promoter",
                        "title": "Demo study",
                        "record_root": "docs/studies/demo_study",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    try:
        load_study_index(repo_root)
    except ValueError as exc:
        assert "must not define legacy family" in str(exc)
    else:
        raise AssertionError("Expected legacy family field to fail fast")
