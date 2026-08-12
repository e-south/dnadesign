"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/tests/test_study_runbook_discovery.py

Contract tests for external-study Infer runbook discovery.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.ops.api import discover_infer_runbook_paths_for_study


def _write_study_contract(study_dir: Path, *, runbook_ref: str) -> None:
    repo_root = study_dir.parents[2]
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "pyproject.toml").write_text("[project]\nname='studies'\nversion='0.0.0'\n", encoding="utf-8")
    operations_dir = study_dir / "operations"
    operations_dir.mkdir(parents=True)
    (operations_dir / "ops.study.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 2,
                "study_id": "study",
                "execution_surfaces": {
                    "infer": {
                        "surface_type": "runbook",
                        "runbook_ref": runbook_ref,
                    }
                },
                "preflight": {
                    "default_scope": "next",
                    "scopes": {
                        "next": {"include_groups": ["infer_runbook"]},
                        "full": {"include_groups": ["all"]},
                    },
                    "checks": {
                        "infer_runbook": [
                            {
                                "kind": "runbook_plan",
                                "check_id": "infer.runbook.plan",
                                "check_group": "infer_runbook",
                                "summary": "Infer runbook can be planned.",
                                "required": True,
                                "surface": "infer",
                            }
                        ]
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_unprefixed_runbook_ref_resolves_from_external_study_root(tmp_path: Path) -> None:
    repo_root = tmp_path / "dnadesign"
    repo_root.mkdir()
    study_dir = tmp_path / "research-studies" / "studies" / "group" / "study"
    runbook = study_dir / "workflows" / "infer" / "runbook.yaml"
    runbook.parent.mkdir(parents=True)
    runbook.write_text("runbook: {}\n", encoding="utf-8")
    _write_study_contract(study_dir, runbook_ref="workflows/infer/runbook.yaml")

    discovered = discover_infer_runbook_paths_for_study(study_dir=study_dir, repo_root=repo_root)

    assert discovered == (runbook.resolve(),)


def test_repo_prefixed_runbook_ref_remains_explicitly_repo_relative(tmp_path: Path) -> None:
    repo_root = tmp_path / "dnadesign"
    runbook = repo_root / "runbooks" / "infer.yaml"
    runbook.parent.mkdir(parents=True)
    runbook.write_text("runbook: {}\n", encoding="utf-8")
    study_dir = tmp_path / "research-studies" / "studies" / "group" / "study"
    _write_study_contract(study_dir, runbook_ref="repo:runbooks/infer.yaml")

    discovered = discover_infer_runbook_paths_for_study(study_dir=study_dir, repo_root=repo_root)

    assert discovered == (runbook.resolve(),)
