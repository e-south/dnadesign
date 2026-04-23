"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/tests/test_cruncher_snapshot.py

Focused tests for the Cruncher study snapshot adapter.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.studies.families.cruncher.adapter import STUDY_FAMILY_ADAPTER
from dnadesign.studies.families.cruncher.ops.provider import provide_cruncher_status


def _write_cruncher_study_record(tmp_path: Path) -> Path:
    repo_root = tmp_path
    (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.0.0'\n", encoding="utf-8")

    snapback_workspace = repo_root / "src" / "dnadesign" / "cruncher" / "workspaces" / "de033"
    yiu_workspace = repo_root / "src" / "dnadesign" / "cruncher" / "workspaces" / "demo_monotypic_tetr"
    for path in (
        snapback_workspace / "configs" / "snapback",
        yiu_workspace / "configs" / "yiu",
        repo_root / "src" / "dnadesign" / "cruncher" / "docs" / "dev",
        repo_root / ".agents" / "skills" / "snapback-hairpin-study",
    ):
        path.mkdir(parents=True, exist_ok=True)

    (snapback_workspace / "configs" / "snapback" / "de033.released.snapback.yaml").write_text(
        "released_snapback: {}\n",
        encoding="utf-8",
    )
    (yiu_workspace / "configs" / "yiu" / "tetr_teto2_wt_direct.yiu.yaml").write_text("yiu: {}\n", encoding="utf-8")
    retron_note = (
        repo_root / "src" / "dnadesign" / "cruncher" / "docs" / "dev" / "2026-04-19-retron-p4-hairpin-variant-audit.md"
    )
    retron_note.write_text(
        "# retron\n",
        encoding="utf-8",
    )
    yiu_note = (
        repo_root / "src" / "dnadesign" / "cruncher" / "docs" / "dev" / "2026-04-19-yiu-retron-mismatch-bulge-audit.md"
    )
    yiu_note.write_text(
        "# yiu\n",
        encoding="utf-8",
    )
    (repo_root / ".agents" / "skills" / "snapback-hairpin-study" / "SKILL.md").write_text(
        "# skill\n",
        encoding="utf-8",
    )

    study_root = repo_root / "docs" / "studies" / "demo_study"
    study_root.mkdir(parents=True, exist_ok=True)
    (study_root / "status.md").write_text(
        (
            "# Demo study\n\n"
            "**Owner:** dnadesign-maintainers\n"
            "**Last verified:** 2026-04-21\n\n"
            "### At a glance\n\n"
            "- Released-product Snapback stays active.\n"
            "- YIU remains contrast only.\n\n"
            "### Current phase and surfaces\n\n"
            "- Current phase: `snapback_released_solve`\n"
            "- Next owner surface: `src/dnadesign/cruncher/workspaces/de033/runbook.md`\n"
            "- Primary workspace: `src/dnadesign/cruncher/workspaces/de033`\n"
            "- Contrast workspace: `src/dnadesign/cruncher/workspaces/demo_monotypic_tetr`\n"
        ),
        encoding="utf-8",
    )
    (study_root / "routes.md").write_text("# Routes\n", encoding="utf-8")
    (study_root / "datasets.yaml").write_text("study_id: demo_study\ndatasets: []\n", encoding="utf-8")
    (study_root / "campaign.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 2,
                "path_base": "repo",
                "campaign_id": "demo_study",
                "steps": [
                    {
                        "label": "cruncher-study-status",
                        "registry_id": "cruncher.data-plane.cruncher-study-status",
                        "inputs": {"study_dir": "docs/studies/demo_study"},
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (study_root / "pipeline.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "study_id": "demo_study",
                "intent": {
                    "primary_goal": "Evaluate released-product Snapback as the shortening lane.",
                    "primary_lane": "released-product snapback",
                    "operator_question": "Can released-product Snapback own the compact post-release object?",
                    "context_refs": [
                        "repo:src/dnadesign/cruncher/docs/dev/2026-04-19-retron-p4-hairpin-variant-audit.md",
                        "repo:src/dnadesign/cruncher/docs/dev/2026-04-19-yiu-retron-mismatch-bulge-audit.md",
                    ],
                    "decision_refs": [
                        "repo:src/dnadesign/cruncher/workspaces/de033/runbook.md",
                    ],
                },
                "command_groups": [
                    {
                        "id": "snapback_released_probe",
                        "purpose": "Check the released-product read-only probe.",
                        "workspace_ref": "repo:src/dnadesign/cruncher/workspaces/de033",
                        "validation_role": "read_only_probe",
                        "commands": [
                            (
                                "uv run cruncher snapback released-target-search --workspace-root . "
                                "--nick-preset neb_nicking_v1 --release-preset type_iis_release_v1 --json"
                            ),
                        ],
                    },
                    {
                        "id": "snapback_released_solve",
                        "purpose": (
                            "Materialize the whole-catalog released-product hit bundle after the "
                            "read-only probe is clean."
                        ),
                        "workspace_ref": "repo:src/dnadesign/cruncher/workspaces/de033",
                        "mutates_outputs": True,
                        "commands": [
                            (
                                "uv run cruncher snapback released-solve --workspace-root . "
                                "--nick-preset neb_nicking_v1 --release-preset type_iis_release_v1 "
                                "--nick-boundary 0 --paired-bp 3 --cap-nt 3 "
                                "--run-dir outputs/released_solve --materialize-top-k 8 "
                                "--render-format pdf --emit-renders --force-overwrite --json"
                            ),
                        ],
                    },
                    {
                        "id": "yiu_boundary_check",
                        "purpose": "Validate the direct TetR/TetO YIU boundary example.",
                        "workspace_ref": "repo:src/dnadesign/cruncher/workspaces/demo_monotypic_tetr",
                        "commands": [
                            "uv run cruncher yiu validate --spec configs/yiu/tetr_teto2_wt_direct.yiu.yaml",
                        ],
                    },
                ],
                "native_agent_bootstrap": {
                    "open_first": [
                        "repo:.agents/skills/snapback-hairpin-study/SKILL.md",
                        "manifest:status.md",
                        "manifest:routes.md",
                        "manifest:pipeline.yaml",
                    ],
                    "pair_with": [
                        "harness-engineering",
                        "pragmatic-programming-principles",
                    ],
                    "must_preserve": [
                        "released-product Snapback is the active shortening lane",
                        "YIU remains mismatch-centric boundary context only",
                    ],
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (study_root / "ops.study.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 2,
                "study_id": "demo_study",
                "family": "cruncher",
                "title": "Demo Cruncher study",
                "record_sources": {
                    "narrative_ref": "manifest:status.md",
                    "datasets_ref": "manifest:datasets.yaml",
                    "pipeline_ref": "manifest:pipeline.yaml",
                    "campaign_ref": "manifest:campaign.yaml",
                    "routes_ref": "manifest:routes.md",
                },
                "lifecycle": {
                    "phase_order": [
                        "context_consolidation",
                        "snapback_released_probe",
                        "snapback_released_solve",
                        "yiu_boundary_check",
                    ],
                    "current_phase": {
                        "strategy": "explicit",
                        "id": "snapback_released_solve",
                    },
                },
                "phases": [
                    {
                        "id": "context_consolidation",
                        "status": "complete",
                        "next_surface": "manifest:status.md",
                    },
                    {
                        "id": "snapback_released_probe",
                        "status": "complete",
                        "next_surface": "manifest:routes.md",
                    },
                    {
                        "id": "snapback_released_solve",
                        "status": "in_progress",
                        "next_surface": "repo:src/dnadesign/cruncher/workspaces/de033/runbook.md",
                    },
                    {
                        "id": "yiu_boundary_check",
                        "status": "planned",
                        "next_surface": "repo:src/dnadesign/cruncher/workspaces/demo_monotypic_tetr/runbook.md",
                    },
                ],
                "artifacts": {
                    "routes_doc": {
                        "artifact_type": "file",
                        "ref": "manifest:routes.md",
                    },
                    "pipeline_doc": {
                        "artifact_type": "file",
                        "ref": "manifest:pipeline.yaml",
                    },
                    "snapback_released_spec": {
                        "artifact_type": "file",
                        "ref": (
                            "repo:src/dnadesign/cruncher/workspaces/de033/configs/snapback/de033.released.snapback.yaml"
                        ),
                    },
                    "yiu_direct_spec": {
                        "artifact_type": "file",
                        "ref": (
                            "repo:src/dnadesign/cruncher/workspaces/demo_monotypic_tetr/"
                            "configs/yiu/tetr_teto2_wt_direct.yiu.yaml"
                        ),
                    },
                },
                "execution_surfaces": {
                    "de033_workspace": {
                        "surface_type": "workspace",
                        "workspace_ref": "repo:src/dnadesign/cruncher/workspaces/de033",
                    },
                    "demo_monotypic_tetr_workspace": {
                        "surface_type": "workspace",
                        "workspace_ref": "repo:src/dnadesign/cruncher/workspaces/demo_monotypic_tetr",
                    },
                    "snapback_released_target_search": {
                        "surface_type": "command",
                        "cwd_ref": "repo:src/dnadesign/cruncher/workspaces/de033",
                        "argv": [
                            "uv",
                            "run",
                            "cruncher",
                            "snapback",
                            "released-target-search",
                            "--workspace-root",
                            ".",
                            "--nick-preset",
                            "neb_nicking_v1",
                            "--release-preset",
                            "type_iis_release_v1",
                            "--nick-boundary",
                            "0",
                            "--paired-bp",
                            "3",
                            "--cap-nt",
                            "3",
                            "--json",
                        ],
                    },
                    "yiu_direct_validate": {
                        "surface_type": "command",
                        "cwd_ref": "repo:src/dnadesign/cruncher/workspaces/demo_monotypic_tetr",
                        "argv": [
                            "uv",
                            "run",
                            "cruncher",
                            "yiu",
                            "validate",
                            "--spec",
                            "configs/yiu/tetr_teto2_wt_direct.yiu.yaml",
                        ],
                    },
                },
                "snapshot": {
                    "summary_scope": "repo",
                },
                "preflight": {
                    "default_scope": "next",
                    "scopes": {
                        "next": {"include_phases": ["current_phase"]},
                        "full": {"include_phases": ["all"]},
                    },
                    "group_phase_bindings": {
                        "study_record": "context_consolidation",
                        "snapback_workspace": "snapback_released_probe",
                        "snapback_probe": "snapback_released_probe",
                        "yiu_workspace": "yiu_boundary_check",
                        "yiu_validate": "yiu_boundary_check",
                    },
                    "checks": {
                        "context_consolidation": [
                            {
                                "kind": "path_exists",
                                "check_id": "study.routes.present",
                                "check_group": "study_record",
                                "summary": "Study route map is present.",
                                "required": True,
                                "artifact": "routes_doc",
                            },
                            {
                                "kind": "path_exists",
                                "check_id": "study.pipeline.present",
                                "check_group": "study_record",
                                "summary": "Study pipeline context is present.",
                                "required": True,
                                "artifact": "pipeline_doc",
                            },
                        ],
                        "snapback_released_probe": [
                            {
                                "kind": "workspace_layout",
                                "check_id": "de033.workspace",
                                "check_group": "snapback_workspace",
                                "summary": "de033 workspace is present.",
                                "required": True,
                                "surface": "de033_workspace",
                            },
                            {
                                "kind": "path_exists",
                                "check_id": "de033.released_spec",
                                "check_group": "snapback_workspace",
                                "summary": "Released-product demo spec is present.",
                                "required": True,
                                "artifact": "snapback_released_spec",
                            },
                            {
                                "kind": "command",
                                "check_id": "de033.released_target_search",
                                "check_group": "snapback_probe",
                                "summary": "Released-product target-search probe completed.",
                                "required": True,
                                "surface": "snapback_released_target_search",
                            },
                        ],
                        "yiu_boundary_check": [
                            {
                                "kind": "workspace_layout",
                                "check_id": "demo_monotypic_tetr.workspace",
                                "check_group": "yiu_workspace",
                                "summary": "demo_monotypic_tetr workspace is present.",
                                "required": True,
                                "surface": "demo_monotypic_tetr_workspace",
                            },
                            {
                                "kind": "path_exists",
                                "check_id": "demo_monotypic_tetr.yiu_spec",
                                "check_group": "yiu_workspace",
                                "summary": "Direct TetR/TetO YIU spec is present.",
                                "required": True,
                                "artifact": "yiu_direct_spec",
                            },
                            {
                                "kind": "command",
                                "check_id": "demo_monotypic_tetr.yiu_validate",
                                "check_group": "yiu_validate",
                                "summary": "Direct TetR/TetO YIU validate completed.",
                                "required": True,
                                "surface": "yiu_direct_validate",
                            },
                        ],
                    },
                    "next_scope": {
                        "target_phase_groups": {
                            "context_consolidation": ["study_record"],
                            "snapback_released_probe": ["study_record", "snapback_workspace", "snapback_probe"],
                            "snapback_released_solve": ["study_record", "snapback_workspace", "snapback_probe"],
                            "yiu_boundary_check": ["study_record", "yiu_workspace", "yiu_validate"],
                        },
                        "runtime_phase_groups": [],
                        "runtime_shared_groups": [],
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return study_root


def test_provide_cruncher_status_exposes_command_groups_and_agent_bootstrap(tmp_path: Path) -> None:
    study_root = _write_cruncher_study_record(tmp_path)

    state, summary, evidence = provide_cruncher_status(
        repo_root=tmp_path,
        inputs={"study_dir": study_root},
    )

    assert state == "ok"
    assert "primary lane released-product snapback" in summary
    assert evidence["current_phase"] == "snapback_released_solve"
    assert [group["id"] for group in evidence["command_groups"]] == [
        "snapback_released_probe",
        "snapback_released_solve",
        "yiu_boundary_check",
    ]
    assert evidence["native_agent_bootstrap"]["open_first"] == [
        str(tmp_path / ".agents" / "skills" / "snapback-hairpin-study" / "SKILL.md"),
        str(study_root / "status.md"),
        str(study_root / "routes.md"),
        str(study_root / "pipeline.yaml"),
    ]
    assert evidence["native_agent_bootstrap"]["pair_with"] == [
        "harness-engineering",
        "pragmatic-programming-principles",
    ]
    assert "skill_ref" not in evidence["record_sources"]
    assert "repo_local_skill" not in evidence["artifacts"]
    assert evidence["status_excerpt"] == [
        "- Current phase: `snapback_released_solve`",
        "- Next owner surface: `src/dnadesign/cruncher/workspaces/de033/runbook.md`",
        "- Primary workspace: `src/dnadesign/cruncher/workspaces/de033`",
        "- Contrast workspace: `src/dnadesign/cruncher/workspaces/demo_monotypic_tetr`",
    ]


def test_cruncher_snapshot_reports_missing_required_record_files(tmp_path: Path) -> None:
    study_root = _write_cruncher_study_record(tmp_path)
    (study_root / "routes.md").unlink()

    context = STUDY_FAMILY_ADAPTER.load_context(repo_root=tmp_path, study_root=study_root)
    state, summary, evidence = STUDY_FAMILY_ADAPTER.build_snapshot(context)

    assert state == "missing"
    assert "routes.md" in summary
    assert evidence["missing_required_files"] == ["routes.md"]
