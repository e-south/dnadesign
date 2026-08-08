"""Contract tests for a portable study workspace catalog."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dnadesign.studies.core.workspace import load_study_workspace


def _write_yaml(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _workspace(tmp_path: Path) -> Path:
    root = tmp_path / "research-studies"
    program_root = root / "programs" / "stress-response"
    study_root = program_root / "studies" / "promoter-response"
    (program_root / "README.md").parent.mkdir(parents=True, exist_ok=True)
    (program_root / "README.md").write_text("# Stress response\n", encoding="utf-8")
    (study_root / "README.md").parent.mkdir(parents=True, exist_ok=True)
    (study_root / "README.md").write_text("# Promoter response\n", encoding="utf-8")
    (study_root / "workflows" / "reader").mkdir(parents=True, exist_ok=True)
    (study_root / "workflows" / "reader" / "README.md").write_text("# Reader route\n", encoding="utf-8")
    _write_yaml(
        study_root / "evidence" / "index.yaml",
        {
            "schema": "study-evidence-index/v1",
            "study_id": "promoter_response",
            "artifacts": [],
        },
    )
    _write_yaml(
        study_root / "study.yaml",
        {
            "schema": "study/v1",
            "study_id": "promoter_response",
            "program_id": "stress_response",
            "title": "Promoter response",
            "summary": "Measures promoter responses across declared conditions.",
            "visibility": "private",
            "status": "active",
            "owners": ["stress-study-maintainers"],
            "last_verified": "2026-08-08",
            "entrypoint": "README.md",
            "evidence_index": "evidence/index.yaml",
            "workflows": [
                {
                    "tool_id": "reader",
                    "route": "workflows/reader/README.md",
                    "requires": "reader-workbench>=0.1",
                }
            ],
        },
    )
    _write_yaml(
        root / "catalog" / "studies.yaml",
        {
            "schema": "study-catalog/v1",
            "programs": [
                {
                    "program_id": "stress_response",
                    "title": "Stress response",
                    "entrypoint": "programs/stress-response/README.md",
                }
            ],
            "studies": [
                {
                    "study_id": "promoter_response",
                    "manifest": "programs/stress-response/studies/promoter-response/study.yaml",
                }
            ],
        },
    )
    return root


def test_load_study_workspace_resolves_declared_routes(tmp_path: Path) -> None:
    root = _workspace(tmp_path)

    workspace = load_study_workspace(root)

    assert workspace.schema == "study-catalog/v1"
    assert workspace.programs[0].program_id == "stress_response"
    assert workspace.studies[0].study_id == "promoter_response"
    assert workspace.studies[0].program_id == "stress_response"
    assert (
        workspace.studies[0].entrypoint
        == (root / "programs/stress-response/studies/promoter-response/README.md").resolve()
    )
    assert workspace.studies[0].workflows[0].tool_id == "reader"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda payload: payload.update({"active_study_id": "promoter_response"}), "unknown key"),
        (lambda payload: payload["studies"].append(dict(payload["studies"][0])), "duplicate study_id"),
        (lambda payload: payload["programs"].append(dict(payload["programs"][0])), "duplicate program_id"),
    ],
)
def test_catalog_rejects_ambiguous_or_undeclared_state(
    tmp_path: Path,
    mutation,
    message: str,
) -> None:
    root = _workspace(tmp_path)
    catalog_path = root / "catalog" / "studies.yaml"
    payload = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    mutation(payload)
    _write_yaml(catalog_path, payload)

    with pytest.raises(ValueError, match=message):
        load_study_workspace(root)


def test_catalog_rejects_manifest_identity_drift(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    manifest_path = root / "programs/stress-response/studies/promoter-response/study.yaml"
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    payload["study_id"] = "another_study"
    _write_yaml(manifest_path, payload)

    with pytest.raises(ValueError, match="does not match catalog study_id"):
        load_study_workspace(root)


def test_catalog_rejects_undeclared_program(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    manifest_path = root / "programs/stress-response/studies/promoter-response/study.yaml"
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    payload["program_id"] = "unregistered_program"
    _write_yaml(manifest_path, payload)

    with pytest.raises(ValueError, match="undeclared program_id"):
        load_study_workspace(root)


def test_catalog_rejects_path_traversal(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    catalog_path = root / "catalog" / "studies.yaml"
    payload = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    payload["studies"][0]["manifest"] = "../outside.yaml"
    _write_yaml(catalog_path, payload)

    with pytest.raises(ValueError, match="repository-relative path"):
        load_study_workspace(root)


def test_catalog_rejects_symlink_escape(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    outside = tmp_path / "outside.md"
    outside.write_text("outside\n", encoding="utf-8")
    program_entrypoint = root / "programs/stress-response/README.md"
    program_entrypoint.unlink()
    program_entrypoint.symlink_to(outside)

    with pytest.raises(ValueError, match="escapes workspace root"):
        load_study_workspace(root)


def test_manifest_rejects_unknown_keys(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    manifest_path = root / "programs/stress-response/studies/promoter-response/study.yaml"
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    payload["objective"] = "study-specific meaning must not enter the generic manifest"
    _write_yaml(manifest_path, payload)

    with pytest.raises(ValueError, match="unknown key.*objective"):
        load_study_workspace(root)


def test_manifest_rejects_unresolved_workflow_route(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    route = root / "programs/stress-response/studies/promoter-response/workflows/reader/README.md"
    route.unlink()

    with pytest.raises(ValueError, match="workflow reader route does not exist"):
        load_study_workspace(root)
