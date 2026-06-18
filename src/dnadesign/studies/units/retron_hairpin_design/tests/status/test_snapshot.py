"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/status/test_snapshot.py

Focused tests for the Retron hairpin design status service.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import yaml

from dnadesign.studies.units.retron_hairpin_design.status.ops.provider import provide_retron_hairpin_design_status
from dnadesign.studies.units.retron_hairpin_design.status.service import STUDY_STATUS_SERVICE


def _write_retron_hairpin_design_record(tmp_path: Path) -> Path:
    repo_root = tmp_path
    (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.0.0'\n", encoding="utf-8")

    snapback_workspace = repo_root / "src" / "dnadesign" / "cruncher" / "workspaces" / "de033"
    yiu_workspace = repo_root / "src" / "dnadesign" / "cruncher" / "workspaces" / "demo_monotypic_tetr"
    for path in (
        snapback_workspace / "configs" / "snapback",
        yiu_workspace / "configs" / "yiu",
        repo_root / "src" / "dnadesign" / "cruncher" / "docs" / "dev",
        repo_root / ".agents" / "skills" / "retron-hairpin-study",
    ):
        path.mkdir(parents=True, exist_ok=True)

    (snapback_workspace / "configs" / "snapback" / "de033.released.snapback.yaml").write_text(
        "released_snapback: {}\n",
        encoding="utf-8",
    )
    (yiu_workspace / "configs" / "yiu" / "tetr_teto2_wt_direct.yiu.yaml").write_text("yiu: {}\n", encoding="utf-8")
    retron_note = (
        repo_root
        / "src"
        / "dnadesign"
        / "cruncher"
        / "docs"
        / "dev"
        / "audits"
        / "2026-04-19-retron-p4-hairpin-variant.md"
    )
    retron_note.parent.mkdir(parents=True, exist_ok=True)
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
    (repo_root / ".agents" / "skills" / "retron-hairpin-study" / "SKILL.md").write_text(
        "# skill\n",
        encoding="utf-8",
    )

    study_root = repo_root / "docs" / "studies" / "retron_hairpin_design"
    (study_root / "record").mkdir(parents=True, exist_ok=True)
    (study_root / "operations").mkdir(parents=True, exist_ok=True)
    (study_root / "operations" / "runtime").mkdir(parents=True, exist_ok=True)
    (study_root / "routes").mkdir(parents=True, exist_ok=True)
    (study_root / "record" / "status.md").write_text(
        (
            "# Retron hairpin design\n\n"
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
    (study_root / "routes" / "README.md").write_text("# Routes\n", encoding="utf-8")
    (study_root / "record" / "datasets.yaml").write_text(
        "study_id: retron_hairpin_design\ndatasets: []\n",
        encoding="utf-8",
    )
    (study_root / "record" / "campaign.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 2,
                "path_base": "repo",
                "campaign_id": "retron_hairpin_design",
                "steps": [
                    {
                        "label": "retron-hairpin-design-status",
                        "registry_id": "studies.retron-hairpin-design.status",
                        "inputs": {"study_dir": "docs/studies/retron_hairpin_design"},
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (study_root / "operations" / "runtime" / "command-groups").mkdir(parents=True, exist_ok=True)
    (study_root / "operations" / "runtime" / "command-groups" / "pipeline.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "study_id": "retron_hairpin_design",
                "intent": {
                    "primary_goal": "Evaluate released-product Snapback as the shortening lane.",
                    "primary_lane": "released-product snapback",
                    "operator_question": "Can released-product Snapback own the compact post-release object?",
                    "context_refs": [
                        "repo:src/dnadesign/cruncher/docs/dev/audits/2026-04-19-retron-p4-hairpin-variant.md",
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
                        "repo:.agents/skills/retron-hairpin-study/SKILL.md",
                        "manifest:record/status.md",
                        "manifest:routes/README.md",
                        "manifest:operations/runtime/command-groups/pipeline.yaml",
                    ],
                    "pair_with": [
                        "harness-engineering",
                        "code-change-discipline (pragmatic-programming-principles lane)",
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
    (study_root / "operations" / "ops.study.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 2,
                "study_id": "retron_hairpin_design",
                "ops_surfaces": {
                    "status_kind": "retron-hairpin-design-status",
                    "preflight_kind": "retron-hairpin-design-preflight",
                },
                "title": "Demo Retron hairpin design",
                "record_sources": {
                    "narrative_ref": "manifest:record/status.md",
                    "datasets_ref": "manifest:record/datasets.yaml",
                    "pipeline_ref": "manifest:operations/runtime/command-groups/pipeline.yaml",
                    "campaign_ref": "manifest:record/campaign.yaml",
                    "routes_ref": "manifest:routes/README.md",
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
                        "next_surface": "manifest:record/status.md",
                    },
                    {
                        "id": "snapback_released_probe",
                        "status": "complete",
                        "next_surface": "manifest:routes/README.md",
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
                        "ref": "manifest:routes/README.md",
                    },
                    "pipeline_doc": {
                        "artifact_type": "file",
                        "ref": "manifest:operations/runtime/command-groups/pipeline.yaml",
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


def _write_stress_record_with_status_surface(repo_root: Path) -> Path:
    study_root = repo_root / "docs" / "studies" / "stress_ethanol_cipro_growth"
    (study_root / "operations").mkdir(parents=True, exist_ok=True)
    (study_root / "operations" / "ops.study.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 2,
                "study_id": "stress_ethanol_cipro_growth",
                "ops_surfaces": {
                    "status_kind": "stress-ethanol-cipro-growth-status",
                    "preflight_kind": "stress-ethanol-cipro-growth-preflight",
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
    return study_root


def _write_study_index_with_active_stress(repo_root: Path) -> None:
    (repo_root / "docs" / "studies").mkdir(parents=True, exist_ok=True)
    (repo_root / "docs" / "studies" / "index.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "active_study_id": "stress_ethanol_cipro_growth",
                "studies": [
                    {
                        "study_id": "stress_ethanol_cipro_growth",
                        "title": "Stress / ethanol / ciprofloxacin growth study",
                        "record_root": "docs/studies/stress_ethanol_cipro_growth",
                    },
                    {
                        "study_id": "retron_hairpin_design",
                        "title": "Retron hairpin design",
                        "record_root": "docs/studies/retron_hairpin_design",
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def test_provide_retron_hairpin_design_status_exposes_command_groups_and_agent_bootstrap(tmp_path: Path) -> None:
    study_root = _write_retron_hairpin_design_record(tmp_path)

    state, summary, evidence = provide_retron_hairpin_design_status(
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
        str(tmp_path / ".agents" / "skills" / "retron-hairpin-study" / "SKILL.md"),
        str(study_root / "record" / "status.md"),
        str(study_root / "routes" / "README.md"),
        str(study_root / "operations" / "runtime" / "command-groups" / "pipeline.yaml"),
    ]
    assert evidence["native_agent_bootstrap"]["pair_with"] == [
        "harness-engineering",
        "code-change-discipline (pragmatic-programming-principles lane)",
    ]
    assert "skill_ref" not in evidence["record_sources"]
    assert "repo_local_skill" not in evidence["artifacts"]
    assert evidence["status_excerpt"] == [
        "- Current phase: `snapback_released_solve`",
        "- Next owner surface: `src/dnadesign/cruncher/workspaces/de033/runbook.md`",
        "- Primary workspace: `src/dnadesign/cruncher/workspaces/de033`",
        "- Contrast workspace: `src/dnadesign/cruncher/workspaces/demo_monotypic_tetr`",
    ]


def test_provide_retron_hairpin_design_status_without_study_dir_uses_retron_surface(
    tmp_path: Path,
) -> None:
    _write_retron_hairpin_design_record(tmp_path)
    _write_stress_record_with_status_surface(tmp_path)
    _write_study_index_with_active_stress(tmp_path)

    state, summary, evidence = provide_retron_hairpin_design_status(
        repo_root=tmp_path,
        inputs={},
    )

    assert state == "ok"
    assert "primary lane released-product snapback" in summary
    assert evidence["selection_source"] == "status_kind_registry"
    assert evidence["active_study_id"] == "stress_ethanol_cipro_growth"
    assert evidence["record_root"] == str(tmp_path / "docs" / "studies" / "retron_hairpin_design")


def test_provide_retron_hairpin_design_status_uses_track_language_for_nonsequential_records(tmp_path: Path) -> None:
    study_root = _write_retron_hairpin_design_record(tmp_path)
    contract_path = study_root / "operations" / "ops.study.yaml"
    payload = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["lifecycle"] = {
        "mode": "tracks",
        "track_order": [
            "context_consolidation",
            "snapback_released_probe",
            "snapback_released_solve",
            "yiu_boundary_check",
        ],
        "current_track": {
            "strategy": "explicit",
            "id": "snapback_released_solve",
        },
    }
    payload["tracks"] = payload.pop("phases")
    payload["preflight"]["scopes"] = {
        "next": {"include_tracks": ["current_track"]},
        "full": {"include_tracks": ["all"]},
    }
    payload["preflight"]["group_track_bindings"] = payload["preflight"].pop("group_phase_bindings")
    payload["preflight"]["next_scope"]["target_track_groups"] = payload["preflight"]["next_scope"].pop(
        "target_phase_groups"
    )
    payload["preflight"]["next_scope"]["runtime_track_groups"] = payload["preflight"]["next_scope"].pop(
        "runtime_phase_groups"
    )
    contract_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    state, summary, evidence = provide_retron_hairpin_design_status(
        repo_root=tmp_path,
        inputs={"study_dir": study_root},
    )

    assert state == "ok"
    assert "primary lane released-product snapback" in summary
    assert evidence["lifecycle_mode"] == "tracks"
    assert evidence["lifecycle_item_label"] == "track"
    assert evidence["current_track"] == "snapback_released_solve"
    assert "current_phase" not in evidence
    assert "phase_states" not in evidence
    assert [track["id"] for track in evidence["track_states"]] == [
        "context_consolidation",
        "snapback_released_probe",
        "snapback_released_solve",
        "yiu_boundary_check",
    ]


def test_retron_hairpin_design_snapshot_reports_missing_required_record_files(tmp_path: Path) -> None:
    study_root = _write_retron_hairpin_design_record(tmp_path)
    (study_root / "routes" / "README.md").unlink()

    context = STUDY_STATUS_SERVICE.load_context(repo_root=tmp_path, study_root=study_root)
    state, summary, evidence = STUDY_STATUS_SERVICE.build_snapshot(context)

    assert state == "missing"
    assert "routes/README.md" in summary
    assert evidence["missing_required_files"] == ["routes/README.md"]
