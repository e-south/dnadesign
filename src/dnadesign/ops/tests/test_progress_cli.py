"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/tests/test_progress_cli.py

Contract tests for the read-only ops progress surface.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from typer.testing import CliRunner

from dnadesign.ops.cli import app
from dnadesign.ops.progress_command_support import CommandExecution
from dnadesign.studies.stress_promoter_ethanol_cipro.ops_provider import (
    provide_stress_promoter_ethanol_cipro_preflight,
    provide_stress_promoter_ethanol_cipro_status,
)


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def _promoter_study_status(study_dir: Path | None, *, repo_root: Path | None) -> tuple[str, str, dict[str, object]]:
    inputs: dict[str, object] = {}
    if study_dir is not None:
        inputs["study_dir"] = study_dir
    return provide_stress_promoter_ethanol_cipro_status(repo_root=repo_root, inputs=inputs)


def _promoter_study_preflight(
    study_dir: Path | None,
    *,
    repo_root: Path | None,
    scope: str | None = None,
) -> tuple[str, str, dict[str, object]]:
    inputs: dict[str, object] = {}
    if study_dir is not None:
        inputs["study_dir"] = study_dir
    if scope is not None:
        inputs["scope"] = scope
    return provide_stress_promoter_ethanol_cipro_preflight(repo_root=repo_root, inputs=inputs)


def _write_ops_audit(path: Path, *, ok: bool, queue_probe: str | None = None) -> None:
    preflight_stdout = ""
    preflight_stderr = ""
    if queue_probe is not None:
        preflight_stdout = f"queue_probe={queue_probe} running_jobs=unknown queued_jobs=unknown eqw_jobs=unknown"
        if queue_probe == "degraded":
            preflight_stderr = "queue probe degraded: qstat unavailable"
    payload = {
        "plan": {
            "workflow_id": "densegen_batch_submit",
            "project": "demo",
            "runbook_id": "demo_runbook",
            "workspace_root": "/tmp/demo_workspace",
        },
        "execution": {
            "ok": ok,
            "failed_phase": None if ok else "submit",
            "commands": [
                {
                    "phase": "preflight",
                    "command": "uv run python -m dnadesign.ops.orchestrator.gates session-counts",
                    "returncode": 0,
                    "stdout": preflight_stdout,
                    "stderr": preflight_stderr,
                },
                {
                    "phase": "submit",
                    "command": "echo submit",
                    "returncode": 0 if ok else 1,
                    "stdout": "",
                    "stderr": "",
                },
            ],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_sync_audit(path: Path, *, transfer_state: str, primary_changed: bool) -> None:
    payload = {
        "action": "pull",
        "dataset": "demo_dataset",
        "transfer_state": transfer_state,
        "verify": {"primary": "hash", "sidecars": "strict", "content_hashes": "on"},
        "primary": {"changed": primary_changed},
        "meta": {"changed": False},
        ".events.log": {"local": 4, "remote": 4},
        "_snapshots": {"changed": False, "remote_count": 0, "newer_than_local": 0},
        "_derived": {"changed": False, "local_files": 1, "remote_files": 1, "local_only": [], "remote_only": []},
        "_auxiliary": {"changed": False, "local_files": 0, "remote_files": 0, "local_only": [], "remote_only": []},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_usr_dataset(root: Path, dataset: str) -> None:
    dataset_dir = root / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    derived_dir = dataset_dir / "_derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "id": ["a", "b"],
            "sequence": ["AAAA", "CCCC"],
            "length": [4, 4],
            "infer__demo__embedding": [[1.0], [2.0]],
        }
    )
    pq.write_table(table, dataset_dir / "records.parquet")
    pq.write_table(pa.table({"id": ["a"], "infer__score": [0.1]}), derived_dir / "infer.parquet")
    (dataset_dir / ".events.log").write_text("{}\n{}\n", encoding="utf-8")


def _write_promoter_study_record(study_dir: Path, *, densegen_rows: int, densegen_target: int) -> None:
    repo_root = study_dir.parents[3]
    study_dir.mkdir(parents=True, exist_ok=True)
    (study_dir / "campaign.yaml").write_text("campaign_id: demo_study\nsteps: []\n", encoding="utf-8")
    (study_dir / "status.md").write_text("## demo_study\n\n- Current shared feature dataset: `n/a`\n", encoding="utf-8")
    (study_dir / "datasets.yaml").write_text(
        yaml.safe_dump(
            {
                "study_id": "demo_study",
                "datasets": [
                    {
                        "role": "densegen_anchor",
                        "dataset": "densegen/demo_anchor",
                        "usr_root": "usr_root",
                        "status": "present",
                    },
                    {
                        "role": "merged_anchor_source",
                        "dataset": "promoter/demo_anchor_set",
                        "usr_root": "usr_root",
                        "status": "planned",
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (study_dir / "pipeline.yaml").write_text(
        yaml.safe_dump(
            {
                "study_pipeline": {
                    "study_id": "demo_study",
                    "current_phase": "densegen_growth",
                    "canonical_usr_root": "usr_root",
                    "row_targets": {
                        "densegen_anchor_minimum_before_first_full_lane_infer": densegen_target,
                    },
                    "execution_surfaces": {
                        "construct_workspace": "workspace/construct",
                        "infer_workspace": "workspace/infer",
                    },
                    "datasets": {
                        "densegen_anchor_source": "densegen/demo_anchor",
                        "merged_anchor_dataset": "promoter/demo_anchor_set",
                    },
                    "phases": [
                        {
                            "id": "densegen_growth",
                            "status": "in_progress",
                            "primary_dataset": "densegen/demo_anchor",
                            "next_surface": "workspace/densegen_batch.yaml",
                        },
                        {
                            "id": "merged_anchor_set",
                            "status": "ready",
                            "output_dataset": "promoter/demo_anchor_set",
                            "next_surface": "workspace/merge.md",
                        },
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (study_dir / "ops.study.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "study_id": "demo_study",
                "family": "stress_promoter_ethanol_cipro",
                "phase_order": [
                    "densegen_growth",
                    "merged_anchor_set",
                ],
                "snapshot": {"summary_scope": "repo"},
                "preflight": {
                    "default_scope": "next",
                    "phase_targets": {
                        "densegen": "densegen_growth",
                        "construct": "construct_context_expansion",
                        "infer_preparation": "infer_batch_preparation",
                    },
                    "next_scope": {
                        "phase_groups": {
                            "densegen_growth": ["densegen"],
                            "merged_anchor_set": [],
                            "construct_context_expansion": ["construct"],
                            "infer_batch_preparation": ["infer", "notify", "infer_batch_plan"],
                        },
                        "infer_lane_groups": ["infer", "notify", "infer_batch_plan"],
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_usr_dataset(repo_root / "usr_root", "densegen/demo_anchor")
    if densegen_rows != 2:
        dataset_dir = repo_root / "usr_root" / "densegen" / "demo_anchor"
        table = pa.table(
            {
                "id": [f"id_{idx}" for idx in range(densegen_rows)],
                "sequence": ["AAAA"] * densegen_rows,
                "length": [4] * densegen_rows,
            }
        )
        pq.write_table(table, dataset_dir / "records.parquet")
    (repo_root / "workspace" / "construct").mkdir(parents=True, exist_ok=True)
    (repo_root / "workspace" / "infer").mkdir(parents=True, exist_ok=True)


def _write_promoter_study_preflight_record(study_dir: Path) -> None:
    repo_root = study_dir.parents[3]
    study_dir.mkdir(parents=True, exist_ok=True)
    (study_dir / "campaign.yaml").write_text("campaign_id: demo_study\nsteps: []\n", encoding="utf-8")
    (study_dir / "status.md").write_text("## demo_study\n\n- Current shared feature dataset: `n/a`\n", encoding="utf-8")
    (study_dir / "datasets.yaml").write_text(
        yaml.safe_dump(
            {
                "study_id": "demo_study",
                "datasets": [
                    {
                        "role": "densegen_anchor",
                        "dataset": "densegen/demo_anchor",
                        "usr_root": "usr_root",
                        "status": "present",
                    },
                    {
                        "role": "wildtype_manual",
                        "dataset": "mg1655_promoters",
                        "usr_root": "usr_root",
                        "status": "present",
                    },
                    {
                        "role": "construct_template_seed",
                        "dataset": "plasmids",
                        "usr_root": "usr_root",
                        "status": "present",
                    },
                    {
                        "role": "merged_anchor_source",
                        "dataset": "promoter/demo_anchor_set",
                        "usr_root": "usr_root",
                        "status": "planned",
                    },
                    {
                        "role": "construct_context",
                        "dataset": "promoter/demo_construct_contexts",
                        "usr_root": "usr_root",
                        "status": "planned",
                    },
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (study_dir / "pipeline.yaml").write_text(
        yaml.safe_dump(
            {
                "study_pipeline": {
                    "study_id": "demo_study",
                    "current_phase": "infer_batch_preparation",
                    "canonical_usr_root": "usr_root",
                    "execution_surfaces": {
                        "construct_workspace": "workspace/construct",
                        "infer_workspace": "workspace/infer",
                        "densegen_batch_with_notify": "workspace/runbooks/densegen.yaml",
                        "infer_batch_7b_with_notify": {
                            "anchor_only": "workspace/runbooks/infer_anchor_only_7b.yaml",
                            "anchor_plus_template": "workspace/runbooks/infer_anchor_plus_template_7b.yaml",
                        },
                        "infer_batch_20b_with_notify": {
                            "anchor_only": "workspace/runbooks/infer_anchor_only_20b.yaml",
                            "anchor_plus_template": "workspace/runbooks/infer_anchor_plus_template_20b.yaml",
                        },
                    },
                    "datasets": {
                        "densegen_anchor_source": "densegen/demo_anchor",
                        "merged_anchor_dataset": "promoter/demo_anchor_set",
                        "construct_context_dataset": "promoter/demo_construct_contexts",
                    },
                    "construct": {
                        "workspace_projects": [
                            {"id": "slot_a_window", "config": "workspace/construct/config.slot_a.yaml"},
                        ]
                    },
                    "infer": {
                        "preferred_model_family": "evo2_20b",
                        "supported_model_families": ["evo2_20b", "evo2_7b"],
                        "configs": {
                            "anchor_only_7b": "workspace/infer/config.anchor_only.evo2_7b.yaml",
                            "anchor_plus_template_7b": "workspace/infer/config.anchor_plus_template.evo2_7b.yaml",
                            "anchor_only_20b": "workspace/infer/config.anchor_only.evo2_20b.yaml",
                            "anchor_plus_template_20b": "workspace/infer/config.anchor_plus_template.evo2_20b.yaml",
                        },
                    },
                    "phases": [
                        {"id": "densegen_growth", "status": "parallel_optional"},
                        {"id": "merged_anchor_set", "status": "complete"},
                        {"id": "construct_context_expansion", "status": "complete"},
                        {"id": "infer_batch_preparation", "status": "in_progress"},
                        {
                            "id": "infer_anchor_only_20b",
                            "status": "planned",
                            "next_surface": "workspace/runbooks/infer_anchor_only_20b.yaml",
                        },
                        {
                            "id": "infer_anchor_plus_template_20b",
                            "status": "planned",
                            "next_surface": "workspace/runbooks/infer_anchor_plus_template_20b.yaml",
                        },
                        {
                            "id": "infer_anchor_only_7b",
                            "status": "planned",
                            "next_surface": "workspace/runbooks/infer_anchor_only_7b.yaml",
                        },
                        {
                            "id": "infer_anchor_plus_template_7b",
                            "status": "planned",
                            "next_surface": "workspace/runbooks/infer_anchor_plus_template_7b.yaml",
                        },
                    ],
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    (study_dir / "ops.study.yaml").write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "study_id": "demo_study",
                "family": "stress_promoter_ethanol_cipro",
                "phase_order": [
                    "densegen_growth",
                    "merged_anchor_set",
                    "construct_context_expansion",
                    "infer_batch_preparation",
                    "infer_anchor_only_20b",
                    "infer_anchor_plus_template_20b",
                    "infer_anchor_only_7b",
                ],
                "snapshot": {"summary_scope": "repo"},
                "preflight": {
                    "default_scope": "next",
                    "phase_targets": {
                        "densegen": "densegen_growth",
                        "construct": "construct_context_expansion",
                        "infer_preparation": "infer_batch_preparation",
                    },
                    "next_scope": {
                        "phase_groups": {
                            "densegen_growth": ["densegen"],
                            "merged_anchor_set": [],
                            "construct_context_expansion": ["construct"],
                            "infer_batch_preparation": ["infer", "notify", "infer_batch_plan"],
                        },
                        "infer_lane_groups": ["infer", "notify", "infer_batch_plan"],
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    _write_usr_dataset(repo_root / "usr_root", "densegen/demo_anchor")
    _write_usr_dataset(repo_root / "usr_root", "mg1655_promoters")
    _write_usr_dataset(repo_root / "usr_root", "plasmids")
    _write_usr_dataset(repo_root / "usr_root", "promoter/demo_anchor_set")
    _write_usr_dataset(repo_root / "usr_root", "promoter/demo_construct_contexts")

    construct_workspace = repo_root / "workspace" / "construct"
    construct_workspace.mkdir(parents=True, exist_ok=True)
    (construct_workspace / "construct.workspace.yaml").write_text(
        "workspace_id: demo\nprojects: []\n", encoding="utf-8"
    )

    infer_workspace = repo_root / "workspace" / "infer"
    infer_workspace.mkdir(parents=True, exist_ok=True)
    for name, dataset in (
        ("config.anchor_only.evo2_7b.yaml", "promoter/demo_anchor_set"),
        ("config.anchor_plus_template.evo2_7b.yaml", "promoter/demo_construct_contexts"),
        ("config.anchor_only.evo2_20b.yaml", "promoter/demo_anchor_set"),
        ("config.anchor_plus_template.evo2_20b.yaml", "promoter/demo_construct_contexts"),
    ):
        model_id = "evo2_20b" if "20b" in name else "evo2_7b"
        (infer_workspace / name).write_text(
            yaml.safe_dump(
                {
                    "model": {"id": model_id, "device": "cuda:0", "precision": "bf16", "alphabet": "dna"},
                    "jobs": [
                        {
                            "id": name,
                            "operation": "extract",
                            "ingest": {
                                "source": "usr",
                                "dataset": dataset,
                                "root": "../../usr_root",
                                "field": "sequence",
                            },
                        }
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    runbook_root = repo_root / "workspace" / "runbooks"
    runbook_root.mkdir(parents=True, exist_ok=True)
    (runbook_root / "densegen.yaml").write_text(
        yaml.safe_dump(
            {
                "runbook": {
                    "workflow_id": "densegen_batch_with_notify",
                    "densegen": {"config": "../densegen/config.yaml"},
                    "resources": {"h_rt": "08:00:00", "pe_omp": 12},
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    densegen_workspace = repo_root / "workspace" / "densegen"
    densegen_workspace.mkdir(parents=True, exist_ok=True)
    (densegen_workspace / "config.yaml").write_text("solver:\n  backend: gurobi\n", encoding="utf-8")
    for runbook_name in (
        "infer_anchor_only_7b.yaml",
        "infer_anchor_plus_template_7b.yaml",
        "infer_anchor_only_20b.yaml",
        "infer_anchor_plus_template_20b.yaml",
    ):
        (runbook_root / runbook_name).write_text(
            yaml.safe_dump(
                {
                    "runbook": {
                        "workflow_id": runbook_name.replace(".yaml", ""),
                        "infer": {
                            "config": f"../infer/{runbook_name.replace('infer_', 'config.').replace('.yaml', '.yaml')}"
                        },
                        "resources": {"h_rt": "12:00:00", "gpus": 1},
                    }
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )


def _fake_infer_validation_contract(config_path: Path) -> SimpleNamespace:
    dataset = "promoter/demo_construct_contexts" if "template" in config_path.name else "promoter/demo_anchor_set"
    model_id = "evo2_20b" if "20b" in config_path.name else "evo2_7b"
    return SimpleNamespace(
        config_path=config_path,
        model_id=model_id,
        device="cuda:0",
        job_ids=(config_path.stem,),
        usr_datasets=(dataset,),
    )


def _fake_infer_usr_output_contract(repo_root: Path, *, config_path: Path) -> SimpleNamespace:
    dataset = "promoter/demo_construct_contexts" if "template" in config_path.name else "promoter/demo_anchor_set"
    return SimpleNamespace(
        config_path=config_path,
        usr_root=repo_root / "usr_root",
        usr_dataset=dataset,
    )


def _write_cluster_index(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "kind": ["fit", "umap"],
            "run_slug": ["fit_001", "umap_001"],
            "created_utc": ["2026-03-19T10:00:00Z", "2026-03-19T10:05:00Z"],
            "status": ["complete", "complete"],
            "alias": ["demo", "demo"],
        }
    )
    pq.write_table(table, root / "index.parquet")


def _write_opal_campaign(config_path: Path, *, rounds: list[dict[str, object]]) -> None:
    workdir = config_path.parent / "opal_campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump({"campaign": {"workdir": str(workdir)}}, sort_keys=False),
        encoding="utf-8",
    )
    payload = {
        "campaign_slug": "demo_campaign",
        "campaign_name": "Demo campaign",
        "x_column_name": "infer__demo",
        "y_column_name": "measured_activity",
        "rounds": rounds,
    }
    (workdir / "state.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_relative_opal_campaign(config_path: Path, *, rounds: list[dict[str, object]]) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    campaign_root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    campaign_root.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump({"campaign": {"workdir": "."}}, sort_keys=False),
        encoding="utf-8",
    )
    payload = {
        "campaign_slug": "demo_campaign",
        "campaign_name": "Demo campaign",
        "x_column_name": "infer__demo",
        "y_column_name": "measured_activity",
        "rounds": rounds,
    }
    (campaign_root / "state.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def test_cli_progress_show_reports_ops_audit_surface() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        audit_path = Path("artifacts") / "latest.json"
        _write_ops_audit(audit_path, ok=True)

        result = runner.invoke(
            app,
            [
                "progress",
                "show",
                "ops.control-plane.orchestration",
                "--repo-root",
                str(_repo_root()),
                "--audit-json",
                str(audit_path),
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["progress_kind"] == "ops-audit-json"
        assert payload["state"] == "ok"
        assert payload["evidence"]["command_count"] == 2
        assert payload["evidence"]["phase_counts"] == {"preflight": 1, "submit": 1}


def test_cli_progress_show_reports_usr_sync_audit_surface() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        audit_path = Path("sync") / "pull.json"
        _write_sync_audit(audit_path, transfer_state="DRY-RUN", primary_changed=True)

        result = runner.invoke(
            app,
            [
                "progress",
                "show",
                "usr.data-plane.hpc-sync",
                "--repo-root",
                str(_repo_root()),
                "--sync-audit-json",
                str(audit_path),
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["progress_kind"] == "usr-sync-audit"
        assert payload["state"] == "attention"
        assert payload["evidence"]["transfer_state"] == "DRY-RUN"


def test_cli_progress_show_reports_usr_dataset_surface() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        usr_root = Path("usr_root")
        _write_usr_dataset(usr_root, "demo_dataset")

        result = runner.invoke(
            app,
            [
                "progress",
                "show",
                "usr.data-plane.promoter-feature-matrix",
                "--repo-root",
                str(_repo_root()),
                "--usr-root",
                str(usr_root),
                "--dataset",
                "demo_dataset",
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["progress_kind"] == "usr-dataset-state"
        assert payload["state"] == "ok"
        assert payload["evidence"]["rows"] == 2


def test_cli_progress_show_reports_promoter_study_record_surface() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        repo_root = Path.cwd()
        (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.0.0'\n", encoding="utf-8")
        (repo_root / "src" / "dnadesign").mkdir(parents=True, exist_ok=True)
        study_dir = repo_root / "docs" / "studies" / "promoter" / "demo_study"
        promoter_index = repo_root / "docs" / "studies" / "promoter" / "index.yaml"
        promoter_index.parent.mkdir(parents=True, exist_ok=True)
        promoter_index.write_text(
            (
                "active_study: demo_study\n"
                "studies:\n"
                "  - study_id: demo_study\n"
                "    path: docs/studies/promoter/demo_study\n"
            ),
            encoding="utf-8",
        )
        _write_promoter_study_record(study_dir, densegen_rows=2, densegen_target=5)

        result = runner.invoke(
            app,
            [
                "progress",
                "show",
                "usr.data-plane.promoter-study-status",
                "--repo-root",
                str(_repo_root()),
                "--study-dir",
                str(study_dir),
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["progress_kind"] == "promoter-study-record"
        assert payload["state"] == "attention"
        assert payload["evidence"]["study_id"] == "demo_study"
        assert payload["evidence"]["study_selection_source"] == "explicit"
        assert payload["evidence"]["is_active_study"] is True
        assert payload["evidence"]["densegen_rows"] == 2
        assert payload["evidence"]["densegen_row_target"] == 5
        assert payload["evidence"]["densegen_row_gap"] == 3
        assert payload["evidence"]["next_ready_phase"]["id"] == "merged_anchor_set"
        assert payload["evidence"]["datasets"][0]["dataset"] == "densegen/demo_anchor"
        assert "pending promoter/demo_anchor_set" in payload["summary"]


def test_promoter_study_progress_discovers_active_study_from_repo_root() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        repo_root = Path.cwd()
        (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.0.0'\n", encoding="utf-8")
        (repo_root / "src" / "dnadesign").mkdir(parents=True, exist_ok=True)
        study_dir = repo_root / "docs" / "studies" / "promoter" / "demo_study"
        promoter_index = repo_root / "docs" / "studies" / "promoter" / "index.yaml"
        promoter_index.parent.mkdir(parents=True, exist_ok=True)
        promoter_index.write_text(
            (
                "active_study: demo_study\n"
                "studies:\n"
                "  - study_id: demo_study\n"
                "    path: docs/studies/promoter/demo_study\n"
            ),
            encoding="utf-8",
        )
        _write_promoter_study_record(study_dir, densegen_rows=2, densegen_target=5)

        state, summary, evidence = _promoter_study_status(None, repo_root=repo_root)

        assert state == "attention"
        assert evidence["study_id"] == "demo_study"
        assert evidence["study_selection_source"] == "active_registry"
        assert evidence["active_study_registry_path"] == str(promoter_index)
        assert evidence["study_dir"] == str(study_dir)
        assert evidence["infer_notify_profiles"] == {}
        assert evidence["infer_notify_profile_errors"] == {}
        assert "pending promoter/demo_anchor_set" in summary


def test_promoter_study_progress_resolves_relative_study_dir_against_repo_root() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        repo_root = Path.cwd() / "repo"
        repo_root.mkdir(parents=True, exist_ok=True)
        (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.0.0'\n", encoding="utf-8")
        (repo_root / "src" / "dnadesign").mkdir(parents=True, exist_ok=True)
        study_dir = repo_root / "docs" / "studies" / "promoter" / "demo_study"
        promoter_index = repo_root / "docs" / "studies" / "promoter" / "index.yaml"
        promoter_index.parent.mkdir(parents=True, exist_ok=True)
        promoter_index.write_text(
            (
                "active_study: demo_study\n"
                "studies:\n"
                "  - study_id: demo_study\n"
                "    path: docs/studies/promoter/demo_study\n"
            ),
            encoding="utf-8",
        )
        _write_promoter_study_record(study_dir, densegen_rows=2, densegen_target=5)

        state, _, evidence = _promoter_study_status(
            Path("docs/studies/promoter/demo_study"),
            repo_root=repo_root,
        )

        assert state == "attention"
        assert evidence["study_selection_source"] == "explicit"
        assert evidence["study_dir"] == str(study_dir)


def test_promoter_study_preflight_reports_command_and_dataset_blockers(monkeypatch) -> None:
    with CliRunner().isolated_filesystem():
        repo_root = Path.cwd()
        (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.0.0'\n", encoding="utf-8")
        (repo_root / "src" / "dnadesign").mkdir(parents=True, exist_ok=True)
        study_dir = repo_root / "docs" / "studies" / "promoter" / "demo_study"
        promoter_index = repo_root / "docs" / "studies" / "promoter" / "index.yaml"
        promoter_index.parent.mkdir(parents=True, exist_ok=True)
        promoter_index.write_text(
            (
                "active_study: demo_study\n"
                "studies:\n"
                "  - study_id: demo_study\n"
                "    path: docs/studies/promoter/demo_study\n"
            ),
            encoding="utf-8",
        )
        _write_promoter_study_preflight_record(study_dir)

        def _fake_run(argv, *, cwd, timeout_seconds=180):
            command = " ".join(argv)
            if "dense validate-config" in command:
                return CommandExecution(tuple(argv), str(cwd), 1, "", "Solver probe failed", False)
            if "construct workspace doctor" in command:
                return CommandExecution(tuple(argv), str(cwd), 0, "workspace_doctor: ok", "", False)
            if "construct workspace validate-project" in command:
                return CommandExecution(
                    tuple(argv),
                    str(cwd),
                    0,
                    "construct runtime validation completed",
                    "",
                    False,
                )
            if "infer validate config" in command:
                return CommandExecution(tuple(argv), str(cwd), 0, "✔ Config validated.", "", False)
            if "infer run" in command:
                return CommandExecution(
                    tuple(argv),
                    str(cwd),
                    0,
                    "✔ Config validated (dry run).",
                    "",
                    False,
                )
            if "notify setup resolve-events" in command:
                config_path = Path(argv[-2])
                dataset = (
                    "promoter/demo_construct_contexts" if "template" in config_path.name else "promoter/demo_anchor_set"
                )
                payload = {
                    "ok": True,
                    "events": str(repo_root / "usr_root" / dataset / ".events.log"),
                    "policy": "infer",
                }
                return CommandExecution(tuple(argv), str(cwd), 0, json.dumps(payload), "", False)
            if "ops runbook plan" in command and "densegen" in command:
                payload = {
                    "selected_mode": "resume",
                    "workflow_id": "densegen_batch_with_notify",
                    "orchestration_notify": {"secret_ref": "file:///tmp/webhook"},
                }
                return CommandExecution(tuple(argv), str(cwd), 0, json.dumps(payload), "", False)
            if "ops runbook plan" in command:
                return CommandExecution(
                    tuple(argv),
                    str(cwd),
                    2,
                    "",
                    "notify webhook secret file is required for batch notify workflows",
                    False,
                )
            raise AssertionError(f"unexpected command: {command}")

        monkeypatch.setattr("dnadesign.studies.stress_promoter_ethanol_cipro.family.run_progress_command", _fake_run)
        monkeypatch.setattr(
            "dnadesign.studies.stress_promoter_ethanol_cipro.family.inspect_local_infer_gpu_inventory",
            lambda: {"count": 0, "devices": [], "probe_error": None},
        )
        monkeypatch.setattr(
            "dnadesign.construct.contracts.resolve_construct_workspace_config_path_from_root",
            lambda **kwargs: repo_root / "workspace" / "construct" / "config.slot_a.yaml",
        )
        monkeypatch.setattr(
            "dnadesign.construct.preflight_from_config",
            lambda config_path: SimpleNamespace(
                spec_id="spec-demo",
                records_total=2,
                existing_output_collisions=0,
                output_on_conflict="error",
            ),
        )
        monkeypatch.setattr(
            "dnadesign.infer.validate_infer_config_contract",
            lambda config_path: _fake_infer_validation_contract(config_path),
        )
        monkeypatch.setattr(
            "dnadesign.infer.validate_infer_dry_run_contract",
            lambda config_path: _fake_infer_validation_contract(config_path),
        )
        monkeypatch.setattr(
            "dnadesign.infer.contracts.resolve_infer_usr_output_contract",
            lambda config_path: _fake_infer_usr_output_contract(repo_root, config_path=config_path),
        )

        state, summary, evidence = _promoter_study_preflight(None, repo_root=repo_root, scope="full")

        assert state == "attention"
        assert "demo_study: preflight phase infer_batch_preparation" in summary
        assert evidence["preferred_infer_model_family"] == "evo2_20b"
        assert evidence["notify_environment"] == {
            "NOTIFY_WEBHOOK": False,
            "NOTIFY_WEBHOOK_FILE": False,
            "SSL_CERT_FILE": False,
        }
        checks = {check["id"]: check for check in evidence["checks"]}
        assert checks["notify.environment.webhook"]["state"] == "attention"
        assert checks["notify.environment.tls"]["state"] == "attention"
        assert checks["densegen.config.probe_solver"]["state"] == "attention"
        assert checks["densegen.batch.plan"]["state"] == "ok"
        assert checks["construct.workspace.doctor"]["state"] == "ok"
        assert checks["construct.runtime.slot_a_window"]["state"] == "ok"
        assert checks["infer.validate.anchor_only_7b"]["state"] == "ok"
        assert checks["infer.local_runtime.anchor_only_7b"]["state"] == "attention"
        assert checks["infer.local_runtime.anchor_only_20b"]["state"] == "attention"
        assert checks["notify.profile.anchor_only_7b"]["state"] == "attention"
        assert checks["notify.profile.anchor_only_20b"]["state"] == "attention"
        assert checks["notify.profile.anchor_only_20b"]["details"]["profile"].endswith(
            "workspace/infer/outputs/notify/infer/anchor_only_20b/profile.json"
        )
        assert "notify setup slack" in checks["notify.profile.anchor_only_7b"]["details"]["setup_command"]
        assert "--profile" not in checks["notify.profile.anchor_only_7b"]["details"]["setup_command"]
        assert checks["infer.dry_run.anchor_only_7b"]["state"] == "ok"
        assert checks["infer.dry_run.anchor_only_20b"]["state"] == "ok"
        assert checks["notify.resolve_events.anchor_only_7b"]["state"] == "ok"
        assert checks["notify.resolve_events.anchor_only_20b"]["state"] == "ok"
        assert checks["ops.runbook_plan.infer_batch_7b_with_notify.anchor_only"]["state"] == "attention"
        assert evidence["counts"]["ok"] >= 1
        assert evidence["counts"]["attention"] >= 1
        assert evidence["counts"]["missing"] == 0
        assert evidence["scope"] == "full"


def test_promoter_study_preflight_skips_construct_runtime_revalidation_once_materialized(monkeypatch) -> None:
    with CliRunner().isolated_filesystem():
        repo_root = Path.cwd()
        (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.0.0'\n", encoding="utf-8")
        (repo_root / "src" / "dnadesign").mkdir(parents=True, exist_ok=True)
        study_dir = repo_root / "docs" / "studies" / "promoter" / "demo_study"
        promoter_index = repo_root / "docs" / "studies" / "promoter" / "index.yaml"
        promoter_index.parent.mkdir(parents=True, exist_ok=True)
        promoter_index.write_text(
            (
                "active_study: demo_study\n"
                "studies:\n"
                "  - study_id: demo_study\n"
                "    path: docs/studies/promoter/demo_study\n"
            ),
            encoding="utf-8",
        )
        _write_promoter_study_preflight_record(study_dir)

        def _fake_run(argv, *, cwd, timeout_seconds=180):
            command = " ".join(argv)
            if "dense validate-config" in command:
                return CommandExecution(tuple(argv), str(cwd), 0, "solver ok", "", False)
            if "construct workspace doctor" in command:
                return CommandExecution(tuple(argv), str(cwd), 0, "workspace_doctor: ok", "", False)
            if "construct workspace validate-project" in command:
                payload = {
                    "status": "error",
                    "code": 1,
                    "error": (
                        "2 planned output id(s) already exist in dataset "
                        "'promoter/demo_construct_contexts'. Choose a different output dataset."
                    ),
                    "error_type": "ValidationError",
                }
                return CommandExecution(tuple(argv), str(cwd), 1, json.dumps(payload), "", False)
            if "infer validate config" in command:
                return CommandExecution(tuple(argv), str(cwd), 0, "✔ Config validated.", "", False)
            if "infer run" in command:
                return CommandExecution(
                    tuple(argv),
                    str(cwd),
                    0,
                    "✔ Config validated (dry run).",
                    "",
                    False,
                )
            if "notify setup resolve-events" in command:
                config_path = Path(argv[-2])
                dataset = (
                    "promoter/demo_construct_contexts" if "template" in config_path.name else "promoter/demo_anchor_set"
                )
                payload = {
                    "ok": True,
                    "events": str(repo_root / "usr_root" / dataset / ".events.log"),
                    "policy": "infer",
                }
                return CommandExecution(tuple(argv), str(cwd), 0, json.dumps(payload), "", False)
            if "ops runbook plan" in command and "densegen" in command:
                payload = {
                    "selected_mode": "resume",
                    "workflow_id": "densegen_batch_with_notify",
                    "orchestration_notify": {"secret_ref": "file:///tmp/webhook"},
                }
                return CommandExecution(tuple(argv), str(cwd), 0, json.dumps(payload), "", False)
            if "ops runbook plan" in command:
                return CommandExecution(
                    tuple(argv),
                    str(cwd),
                    0,
                    json.dumps({"selected_mode": "fresh"}),
                    "",
                    False,
                )
            raise AssertionError(f"unexpected command: {command}")

        monkeypatch.setattr("dnadesign.studies.stress_promoter_ethanol_cipro.family.run_progress_command", _fake_run)
        monkeypatch.setattr(
            "dnadesign.studies.stress_promoter_ethanol_cipro.family.inspect_local_infer_gpu_inventory",
            lambda: {"count": 1, "devices": [{"id": 0, "name": "GPU"}], "probe_error": None},
        )
        monkeypatch.setattr(
            "dnadesign.infer.validate_infer_config_contract",
            lambda config_path: _fake_infer_validation_contract(config_path),
        )
        monkeypatch.setattr(
            "dnadesign.infer.validate_infer_dry_run_contract",
            lambda config_path: _fake_infer_validation_contract(config_path),
        )
        monkeypatch.setattr(
            "dnadesign.infer.contracts.resolve_infer_usr_output_contract",
            lambda config_path: _fake_infer_usr_output_contract(repo_root, config_path=config_path),
        )
        state, summary, evidence = _promoter_study_preflight(None, repo_root=repo_root, scope="full")

        assert state == "attention"
        checks = {check["id"]: check for check in evidence["checks"]}
        assert checks["construct.runtime.slot_a_window"]["state"] == "ok"
        assert checks["construct.runtime.slot_a_window"]["details"]["skipped_runtime_revalidation"] is True
        assert "skipping rerun runtime preflight" in checks["construct.runtime.slot_a_window"]["summary"]
        assert "construct.runtime.slot_a_window" not in evidence["blocked_by"]


def test_promoter_study_preflight_scope_next_defers_later_lane_blockers(monkeypatch) -> None:
    with CliRunner().isolated_filesystem():
        repo_root = Path.cwd()
        (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.0.0'\n", encoding="utf-8")
        (repo_root / "src" / "dnadesign").mkdir(parents=True, exist_ok=True)
        study_dir = repo_root / "docs" / "studies" / "promoter" / "demo_study"
        promoter_index = repo_root / "docs" / "studies" / "promoter" / "index.yaml"
        promoter_index.parent.mkdir(parents=True, exist_ok=True)
        promoter_index.write_text(
            (
                "active_study: demo_study\n"
                "studies:\n"
                "  - study_id: demo_study\n"
                "    path: docs/studies/promoter/demo_study\n"
            ),
            encoding="utf-8",
        )
        _write_promoter_study_preflight_record(study_dir)

        def _fake_run(argv, *, cwd, timeout_seconds=180):
            command = " ".join(argv)
            if "dense validate-config" in command:
                return CommandExecution(tuple(argv), str(cwd), 1, "", "Solver probe failed", False)
            if "construct workspace doctor" in command:
                return CommandExecution(tuple(argv), str(cwd), 0, "workspace_doctor: ok", "", False)
            if "construct workspace validate-project" in command:
                return CommandExecution(
                    tuple(argv),
                    str(cwd),
                    0,
                    "construct runtime validation completed",
                    "",
                    False,
                )
            if "infer validate config" in command:
                return CommandExecution(tuple(argv), str(cwd), 0, "✔ Config validated.", "", False)
            if "infer run" in command:
                return CommandExecution(
                    tuple(argv),
                    str(cwd),
                    0,
                    "✔ Config validated (dry run).",
                    "",
                    False,
                )
            if "notify setup resolve-events" in command:
                config_path = Path(argv[-2])
                dataset = (
                    "promoter/demo_construct_contexts" if "template" in config_path.name else "promoter/demo_anchor_set"
                )
                payload = {
                    "ok": True,
                    "events": str(repo_root / "usr_root" / dataset / ".events.log"),
                    "policy": "infer",
                }
                return CommandExecution(tuple(argv), str(cwd), 0, json.dumps(payload), "", False)
            if "ops runbook plan" in command and "densegen" in command:
                payload = {
                    "selected_mode": "resume",
                    "workflow_id": "densegen_batch_with_notify",
                    "orchestration_notify": {"secret_ref": "file:///tmp/webhook"},
                }
                return CommandExecution(tuple(argv), str(cwd), 0, json.dumps(payload), "", False)
            if "ops runbook plan" in command:
                return CommandExecution(
                    tuple(argv),
                    str(cwd),
                    2,
                    "",
                    "notify webhook secret file is required for batch notify workflows",
                    False,
                )
            raise AssertionError(f"unexpected command: {command}")

        monkeypatch.setattr("dnadesign.studies.stress_promoter_ethanol_cipro.family.run_progress_command", _fake_run)
        monkeypatch.setattr(
            "dnadesign.studies.stress_promoter_ethanol_cipro.family.inspect_local_infer_gpu_inventory",
            lambda: {"count": 0, "devices": [], "probe_error": None},
        )
        monkeypatch.setattr(
            "dnadesign.construct.contracts.resolve_construct_workspace_config_path_from_root",
            lambda **kwargs: repo_root / "workspace" / "construct" / "config.slot_a.yaml",
        )
        monkeypatch.setattr(
            "dnadesign.construct.preflight_from_config",
            lambda config_path: SimpleNamespace(
                spec_id="spec-demo",
                records_total=2,
                existing_output_collisions=0,
                output_on_conflict="error",
            ),
        )
        monkeypatch.setattr(
            "dnadesign.infer.validate_infer_config_contract",
            lambda config_path: _fake_infer_validation_contract(config_path),
        )
        monkeypatch.setattr(
            "dnadesign.infer.validate_infer_dry_run_contract",
            lambda config_path: _fake_infer_validation_contract(config_path),
        )
        monkeypatch.setattr(
            "dnadesign.infer.contracts.resolve_infer_usr_output_contract",
            lambda config_path: _fake_infer_usr_output_contract(repo_root, config_path=config_path),
        )

        state, summary, evidence = _promoter_study_preflight(None, repo_root=repo_root, scope="next")

        assert state == "attention"
        assert "focus phase infer_batch_preparation" in summary
        assert "blocked by:" in summary
        assert evidence["scope"] == "next"
        assert evidence["target_phase"] == "infer_batch_preparation"
        assert "infer.local_runtime.anchor_only_20b" in evidence["blocked_by"]
        assert "notify.environment.webhook" in evidence["blocked_by"]
        assert evidence["deferred_check_ids"] == []


def test_promoter_study_preflight_lane_scope_keeps_notify_env_and_selected_lane(monkeypatch) -> None:
    with CliRunner().isolated_filesystem():
        repo_root = Path.cwd()
        (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.0.0'\n", encoding="utf-8")
        (repo_root / "src" / "dnadesign").mkdir(parents=True, exist_ok=True)
        study_dir = repo_root / "docs" / "studies" / "promoter" / "demo_study"
        promoter_index = repo_root / "docs" / "studies" / "promoter" / "index.yaml"
        promoter_index.parent.mkdir(parents=True, exist_ok=True)
        promoter_index.write_text(
            (
                "active_study: demo_study\n"
                "studies:\n"
                "  - study_id: demo_study\n"
                "    path: docs/studies/promoter/demo_study\n"
            ),
            encoding="utf-8",
        )
        _write_promoter_study_preflight_record(study_dir)

        pipeline_path = study_dir / "pipeline.yaml"
        pipeline_payload = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
        study_pipeline = pipeline_payload["study_pipeline"]
        study_pipeline["current_phase"] = "infer_anchor_only_20b"
        for phase in study_pipeline["phases"]:
            if phase["id"] == "infer_batch_preparation":
                phase["status"] = "complete"
            elif phase["id"] == "infer_anchor_only_20b":
                phase["status"] = "in_progress"
        pipeline_path.write_text(yaml.safe_dump(pipeline_payload, sort_keys=False), encoding="utf-8")

        def _fake_run(argv, *, cwd, timeout_seconds=180):
            command = " ".join(argv)
            if "dense validate-config" in command:
                return CommandExecution(tuple(argv), str(cwd), 0, "solver ok", "", False)
            if "construct workspace doctor" in command:
                return CommandExecution(tuple(argv), str(cwd), 0, "workspace_doctor: ok", "", False)
            if "construct workspace validate-project" in command:
                return CommandExecution(
                    tuple(argv),
                    str(cwd),
                    0,
                    "construct runtime validation completed",
                    "",
                    False,
                )
            if "infer validate config" in command:
                return CommandExecution(tuple(argv), str(cwd), 0, "✔ Config validated.", "", False)
            if "infer run" in command:
                return CommandExecution(
                    tuple(argv),
                    str(cwd),
                    0,
                    "✔ Config validated (dry run).",
                    "",
                    False,
                )
            if "notify setup resolve-events" in command:
                config_path = Path(argv[-2])
                dataset = (
                    "promoter/demo_construct_contexts" if "template" in config_path.name else "promoter/demo_anchor_set"
                )
                payload = {
                    "ok": True,
                    "events": str(repo_root / "usr_root" / dataset / ".events.log"),
                    "policy": "infer",
                }
                return CommandExecution(tuple(argv), str(cwd), 0, json.dumps(payload), "", False)
            if "ops runbook plan" in command and "densegen" in command:
                payload = {
                    "selected_mode": "resume",
                    "workflow_id": "densegen_batch_with_notify",
                    "orchestration_notify": {"secret_ref": "file:///tmp/webhook"},
                }
                return CommandExecution(tuple(argv), str(cwd), 0, json.dumps(payload), "", False)
            if "ops runbook plan" in command:
                return CommandExecution(
                    tuple(argv),
                    str(cwd),
                    2,
                    "",
                    "notify webhook secret file is required for batch notify workflows",
                    False,
                )
            raise AssertionError(f"unexpected command: {command}")

        monkeypatch.setattr("dnadesign.studies.stress_promoter_ethanol_cipro.family.run_progress_command", _fake_run)
        monkeypatch.setattr(
            "dnadesign.studies.stress_promoter_ethanol_cipro.family.inspect_local_infer_gpu_inventory",
            lambda: {"count": 0, "devices": [], "probe_error": None},
        )
        monkeypatch.setattr(
            "dnadesign.construct.contracts.resolve_construct_workspace_config_path_from_root",
            lambda **kwargs: repo_root / "workspace" / "construct" / "config.slot_a.yaml",
        )
        monkeypatch.setattr(
            "dnadesign.construct.preflight_from_config",
            lambda config_path: SimpleNamespace(
                spec_id="spec-demo",
                records_total=2,
                existing_output_collisions=0,
                output_on_conflict="error",
            ),
        )
        monkeypatch.setattr(
            "dnadesign.infer.validate_infer_config_contract",
            lambda config_path: _fake_infer_validation_contract(config_path),
        )
        monkeypatch.setattr(
            "dnadesign.infer.validate_infer_dry_run_contract",
            lambda config_path: _fake_infer_validation_contract(config_path),
        )
        monkeypatch.setattr(
            "dnadesign.infer.contracts.resolve_infer_usr_output_contract",
            lambda config_path: _fake_infer_usr_output_contract(repo_root, config_path=config_path),
        )

        state, summary, evidence = _promoter_study_preflight(None, repo_root=repo_root, scope="next")

        assert state == "attention"
        assert "focus phase infer_anchor_only_20b" in summary
        assert evidence["target_phase"] == "infer_anchor_only_20b"
        assert "notify.environment.webhook" in evidence["blocked_by"]
        assert "notify.environment.tls" in evidence["blocked_by"]
        assert "infer.local_runtime.anchor_only_20b" in evidence["blocked_by"]
        assert "ops.runbook_plan.infer_batch_20b_with_notify.anchor_only" in evidence["blocked_by"]
        assert "infer.local_runtime.anchor_only_7b" not in evidence["blocked_by"]
        assert "notify.profile.anchor_plus_template_20b" not in evidence["blocked_by"]
        assert "infer.local_runtime.anchor_only_7b" in evidence["deferred_check_ids"]


def test_promoter_study_preflight_full_scope_demotes_completed_infer_lane_attention(monkeypatch) -> None:
    with CliRunner().isolated_filesystem():
        repo_root = Path.cwd()
        (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.0.0'\n", encoding="utf-8")
        (repo_root / "src" / "dnadesign").mkdir(parents=True, exist_ok=True)
        study_dir = repo_root / "docs" / "studies" / "promoter" / "demo_study"
        promoter_index = repo_root / "docs" / "studies" / "promoter" / "index.yaml"
        promoter_index.parent.mkdir(parents=True, exist_ok=True)
        promoter_index.write_text(
            (
                "active_study: demo_study\n"
                "studies:\n"
                "  - study_id: demo_study\n"
                "    path: docs/studies/promoter/demo_study\n"
            ),
            encoding="utf-8",
        )
        _write_promoter_study_preflight_record(study_dir)

        pipeline_path = study_dir / "pipeline.yaml"
        pipeline_payload = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
        study_pipeline = pipeline_payload["study_pipeline"]
        for phase in study_pipeline["phases"]:
            if phase["id"] == "infer_anchor_only_20b":
                phase["status"] = "complete"
        pipeline_path.write_text(yaml.safe_dump(pipeline_payload, sort_keys=False), encoding="utf-8")

        def _fake_run(argv, *, cwd, timeout_seconds=180):
            command = " ".join(argv)
            if "dense validate-config" in command:
                return CommandExecution(tuple(argv), str(cwd), 0, "solver ok", "", False)
            if "construct workspace doctor" in command:
                return CommandExecution(tuple(argv), str(cwd), 0, "workspace_doctor: ok", "", False)
            if "construct workspace validate-project" in command:
                return CommandExecution(
                    tuple(argv),
                    str(cwd),
                    0,
                    "construct runtime validation completed",
                    "",
                    False,
                )
            if "infer validate config" in command:
                return CommandExecution(tuple(argv), str(cwd), 0, "✔ Config validated.", "", False)
            if "infer run" in command:
                return CommandExecution(
                    tuple(argv),
                    str(cwd),
                    0,
                    "✔ Config validated (dry run).",
                    "",
                    False,
                )
            if "notify setup resolve-events" in command:
                config_path = Path(argv[-2])
                dataset = (
                    "promoter/demo_construct_contexts" if "template" in config_path.name else "promoter/demo_anchor_set"
                )
                payload = {
                    "ok": True,
                    "events": str(repo_root / "usr_root" / dataset / ".events.log"),
                    "policy": "infer",
                }
                return CommandExecution(tuple(argv), str(cwd), 0, json.dumps(payload), "", False)
            if "ops runbook plan" in command and "densegen" in command:
                payload = {
                    "selected_mode": "resume",
                    "workflow_id": "densegen_batch_with_notify",
                    "orchestration_notify": {"secret_ref": "file:///tmp/webhook"},
                }
                return CommandExecution(tuple(argv), str(cwd), 0, json.dumps(payload), "", False)
            if "ops runbook plan" in command:
                return CommandExecution(
                    tuple(argv),
                    str(cwd),
                    2,
                    "",
                    "notify webhook secret file is required for batch notify workflows",
                    False,
                )
            raise AssertionError(f"unexpected command: {command}")

        monkeypatch.setattr("dnadesign.studies.stress_promoter_ethanol_cipro.family.run_progress_command", _fake_run)
        monkeypatch.setattr(
            "dnadesign.studies.stress_promoter_ethanol_cipro.family.inspect_local_infer_gpu_inventory",
            lambda: {"count": 0, "devices": [], "probe_error": None},
        )
        monkeypatch.setattr(
            "dnadesign.construct.contracts.resolve_construct_workspace_config_path_from_root",
            lambda **kwargs: repo_root / "workspace" / "construct" / "config.slot_a.yaml",
        )
        monkeypatch.setattr(
            "dnadesign.construct.preflight_from_config",
            lambda config_path: SimpleNamespace(
                spec_id="spec-demo",
                records_total=2,
                existing_output_collisions=0,
                output_on_conflict="error",
            ),
        )
        monkeypatch.setattr(
            "dnadesign.infer.validate_infer_config_contract",
            lambda config_path: _fake_infer_validation_contract(config_path),
        )
        monkeypatch.setattr(
            "dnadesign.infer.validate_infer_dry_run_contract",
            lambda config_path: _fake_infer_validation_contract(config_path),
        )
        monkeypatch.setattr(
            "dnadesign.infer.contracts.resolve_infer_usr_output_contract",
            lambda config_path: _fake_infer_usr_output_contract(repo_root, config_path=config_path),
        )

        state, _summary, evidence = _promoter_study_preflight(None, repo_root=repo_root, scope="full")

        assert state == "attention"
        assert "infer.local_runtime.anchor_only_20b" not in evidence["blocked_by"]
        assert "notify.profile.anchor_only_20b" not in evidence["blocked_by"]
        assert "ops.runbook_plan.infer_batch_20b_with_notify.anchor_only" not in evidence["blocked_by"]
        assert "infer.local_runtime.anchor_only_20b" in evidence["nonblocking_attention_ids"]
        assert "notify.profile.anchor_only_20b" in evidence["nonblocking_attention_ids"]
        assert "ops.runbook_plan.infer_batch_20b_with_notify.anchor_only" in evidence["nonblocking_attention_ids"]
        assert "infer.local_runtime.anchor_plus_template_20b" in evidence["blocked_by"]


def test_promoter_study_preflight_full_scope_demotes_parallel_optional_densegen_attention(monkeypatch) -> None:
    with CliRunner().isolated_filesystem():
        repo_root = Path.cwd()
        (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.0.0'\n", encoding="utf-8")
        (repo_root / "src" / "dnadesign").mkdir(parents=True, exist_ok=True)
        study_dir = repo_root / "docs" / "studies" / "promoter" / "demo_study"
        promoter_index = repo_root / "docs" / "studies" / "promoter" / "index.yaml"
        promoter_index.parent.mkdir(parents=True, exist_ok=True)
        promoter_index.write_text(
            (
                "active_study: demo_study\n"
                "studies:\n"
                "  - study_id: demo_study\n"
                "    path: docs/studies/promoter/demo_study\n"
            ),
            encoding="utf-8",
        )
        _write_promoter_study_preflight_record(study_dir)

        def _fake_run(argv, *, cwd, timeout_seconds=180):
            command = " ".join(argv)
            if "dense validate-config" in command:
                return CommandExecution(tuple(argv), str(cwd), 1, "", "Solver probe failed", False)
            if "construct workspace doctor" in command:
                return CommandExecution(tuple(argv), str(cwd), 0, "workspace_doctor: ok", "", False)
            if "construct workspace validate-project" in command:
                return CommandExecution(
                    tuple(argv),
                    str(cwd),
                    0,
                    "construct runtime validation completed",
                    "",
                    False,
                )
            if "ops runbook plan" in command and "densegen" in command:
                payload = {
                    "selected_mode": "resume",
                    "workflow_id": "densegen_batch_with_notify",
                    "orchestration_notify": {"secret_ref": "file:///tmp/webhook"},
                }
                return CommandExecution(tuple(argv), str(cwd), 0, json.dumps(payload), "", False)
            if "ops runbook plan" in command:
                return CommandExecution(
                    tuple(argv),
                    str(cwd),
                    2,
                    "",
                    "notify webhook secret file is required for batch notify workflows",
                    False,
                )
            raise AssertionError(f"unexpected command: {command}")

        monkeypatch.setattr("dnadesign.studies.stress_promoter_ethanol_cipro.family.run_progress_command", _fake_run)
        monkeypatch.setattr(
            "dnadesign.studies.stress_promoter_ethanol_cipro.family.inspect_local_infer_gpu_inventory",
            lambda: {"count": 0, "devices": [], "probe_error": None},
        )
        monkeypatch.setattr(
            "dnadesign.construct.contracts.resolve_construct_workspace_config_path_from_root",
            lambda **kwargs: repo_root / "workspace" / "construct" / "config.slot_a.yaml",
        )
        monkeypatch.setattr(
            "dnadesign.construct.preflight_from_config",
            lambda config_path: SimpleNamespace(
                spec_id="spec-demo",
                records_total=2,
                existing_output_collisions=0,
                output_on_conflict="error",
            ),
        )
        monkeypatch.setattr(
            "dnadesign.infer.validate_infer_config_contract",
            lambda config_path: _fake_infer_validation_contract(config_path),
        )
        monkeypatch.setattr(
            "dnadesign.infer.validate_infer_dry_run_contract",
            lambda config_path: _fake_infer_validation_contract(config_path),
        )
        monkeypatch.setattr(
            "dnadesign.infer.contracts.resolve_infer_usr_output_contract",
            lambda config_path: _fake_infer_usr_output_contract(repo_root, config_path=config_path),
        )

        state, summary, evidence = _promoter_study_preflight(None, repo_root=repo_root, scope="full")

        assert state == "attention"
        assert "densegen.config.probe_solver" not in evidence["blocked_by"]
        assert "densegen.config.probe_solver" in evidence["nonblocking_attention_ids"]
        assert "infer.local_runtime.anchor_only_20b" in evidence["blocked_by"]


def test_cli_progress_show_reports_cluster_run_index_surface() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        results_root = Path("cluster_results")
        _write_cluster_index(results_root)

        result = runner.invoke(
            app,
            [
                "progress",
                "show",
                "cluster.downstream.exploratory-clustering",
                "--repo-root",
                str(_repo_root()),
                "--cluster-results-root",
                str(results_root),
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["progress_kind"] == "cluster-run-index"
        assert payload["state"] == "ok"
        assert payload["evidence"]["entry_count"] == 2
        assert payload["evidence"]["latest_entry"]["run_slug"] == "umap_001"


def test_cli_progress_show_reports_opal_campaign_surface() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        config_path = Path("configs") / "campaign.yaml"
        _write_opal_campaign(
            config_path,
            rounds=[
                {
                    "round_index": 1,
                    "run_id": "run_001",
                    "round_dir": "/tmp/opal_campaign/outputs/rounds/round_1",
                    "selection_top_k_requested": 12,
                    "selection_top_k_effective_after_ties": 12,
                }
            ],
        )

        result = runner.invoke(
            app,
            [
                "progress",
                "show",
                "opal.downstream.usr-infer-x-active-learning",
                "--repo-root",
                str(_repo_root()),
                "--opal-config",
                str(config_path),
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["progress_kind"] == "opal-campaign-state"
        assert payload["state"] == "ok"
        assert payload["evidence"]["num_rounds"] == 1
        assert payload["evidence"]["latest_round"]["run_id"] == "run_001"


def test_cli_progress_show_resolves_opal_workdir_relative_to_campaign_root() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        config_path = Path("demo_campaign") / "configs" / "campaign.yaml"
        _write_relative_opal_campaign(config_path, rounds=[])

        result = runner.invoke(
            app,
            [
                "progress",
                "show",
                "opal.downstream.usr-infer-x-active-learning",
                "--repo-root",
                str(_repo_root()),
                "--opal-config",
                str(config_path),
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["evidence"]["opal_workdir"].endswith("demo_campaign")
        assert payload["evidence"]["state_path"].endswith("demo_campaign/state.json")


def test_cli_progress_show_reports_missing_artifact_state_without_exiting() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "progress",
            "show",
            "ops.control-plane.orchestration",
            "--repo-root",
            str(_repo_root()),
            "--audit-json",
            "missing/latest.json",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["progress_kind"] == "ops-audit-json"
    assert payload["state"] == "missing"
    assert payload["summary"] == "audit artifact not found"


def test_cli_progress_show_reports_attention_for_degraded_queue_probe() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        audit_path = Path("artifacts") / "latest.json"
        _write_ops_audit(audit_path, ok=True, queue_probe="degraded")

        result = runner.invoke(
            app,
            [
                "progress",
                "show",
                "ops.control-plane.orchestration",
                "--repo-root",
                str(_repo_root()),
                "--audit-json",
                str(audit_path),
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["state"] == "attention"
        assert payload["summary"] == "latest orchestration audit passed with degraded queue probe"
        assert payload["evidence"]["queue_probe"]["status"] == "degraded"


def test_cli_progress_campaign_aggregates_explicit_steps() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        audit_path = Path("artifacts") / "latest.json"
        _write_ops_audit(audit_path, ok=True)
        usr_root = Path("usr_root")
        _write_usr_dataset(usr_root, "demo_dataset")
        config_path = Path("configs") / "campaign.yaml"
        _write_opal_campaign(config_path, rounds=[])
        manifest_path = Path("manifests") / "campaign_status.yaml"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            yaml.safe_dump(
                {
                    "version": 2,
                    "path_base": "manifest",
                    "campaign_id": "demo_cross_tool_campaign",
                    "steps": [
                        {
                            "label": "orchestration",
                            "registry_id": "ops.control-plane.orchestration",
                            "inputs": {"audit_json": "../artifacts/latest.json"},
                        },
                        {
                            "label": "feature-matrix",
                            "registry_id": "usr.data-plane.promoter-feature-matrix",
                            "inputs": {
                                "usr_root": "../usr_root",
                                "dataset": "demo_dataset",
                            },
                        },
                        {
                            "label": "active-learning",
                            "registry_id": "opal.downstream.usr-infer-x-active-learning",
                            "inputs": {"opal_config": "../configs/campaign.yaml"},
                        },
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        result = runner.invoke(
            app,
            [
                "progress",
                "campaign",
                "--repo-root",
                str(_repo_root()),
                "--manifest",
                str(manifest_path),
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["campaign_id"] == "demo_cross_tool_campaign"
        assert payload["overall_state"] == "attention"
        assert payload["counts"] == {"ok": 2, "attention": 1, "missing": 0}
        assert [step["label"] for step in payload["steps"]] == [
            "orchestration",
            "feature-matrix",
            "active-learning",
        ]


def test_cli_progress_campaign_resolves_manifest_relative_artifact_paths() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        audit_path = Path("artifacts") / "latest.json"
        _write_ops_audit(audit_path, ok=True)
        manifest_path = Path("manifests") / "campaign_status.yaml"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            yaml.safe_dump(
                {
                    "version": 2,
                    "path_base": "manifest",
                    "campaign_id": "manifest_relative_paths",
                    "steps": [
                        {
                            "label": "orchestration",
                            "registry_id": "ops.control-plane.orchestration",
                            "inputs": {"audit_json": "../artifacts/latest.json"},
                        }
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        result = runner.invoke(
            app,
            [
                "progress",
                "campaign",
                "--repo-root",
                str(_repo_root()),
                "--manifest",
                str(manifest_path),
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["campaign_id"] == "manifest_relative_paths"
        assert payload["overall_state"] == "ok"
        assert payload["counts"] == {"ok": 1, "attention": 0, "missing": 0}
        assert payload["steps"][0]["label"] == "orchestration"
        assert payload["steps"][0]["evidence"]["audit_json"].endswith("artifacts/latest.json")


def test_cli_progress_show_reports_missing_opal_config_state_without_exiting() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        missing_config = Path("demo_campaign") / "configs" / "campaign.yaml"

        result = runner.invoke(
            app,
            [
                "progress",
                "show",
                "opal.downstream.usr-infer-x-active-learning",
                "--repo-root",
                str(_repo_root()),
                "--opal-config",
                str(missing_config),
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["state"] == "missing"
        assert payload["summary"] == "OPAL config not found"
        assert payload["evidence"]["opal_config"].endswith("demo_campaign/configs/campaign.yaml")
        assert payload["evidence"]["opal_workdir"].endswith("demo_campaign")


def test_cli_progress_campaign_reports_missing_step_for_scaffold_placeholder_paths() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        manifest_path = Path("manifests") / "campaign_status.yaml"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            yaml.safe_dump(
                {
                    "version": 2,
                    "path_base": "manifest",
                    "campaign_id": "placeholder_manifest",
                    "steps": [
                        {
                            "label": "active-learning",
                            "registry_id": "opal.downstream.usr-infer-x-active-learning",
                            "inputs": {"opal_config": "<opal-workdir>/configs/campaign.yaml"},
                        }
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        result = runner.invoke(
            app,
            [
                "progress",
                "campaign",
                "--repo-root",
                str(_repo_root()),
                "--manifest",
                str(manifest_path),
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["overall_state"] == "missing"
        assert payload["counts"] == {"ok": 0, "attention": 0, "missing": 1}
        assert payload["steps"][0]["state"] == "missing"
        assert payload["steps"][0]["summary"] == "OPAL config not found"
        assert payload["steps"][0]["evidence"]["opal_config"].endswith("<opal-workdir>/configs/campaign.yaml")


def test_cli_progress_scaffold_emits_yaml_manifest() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "progress",
            "scaffold",
            "ops.control-plane.orchestration",
            "usr.data-plane.promoter-feature-matrix",
            "--repo-root",
            str(_repo_root()),
        ],
    )

    assert result.exit_code == 0
    payload = yaml.safe_load(result.output)
    assert payload["campaign_id"] == "progress_campaign"
    assert payload["version"] == 2
    assert payload["path_base"] == "repo"
    assert payload["steps"] == [
        {
            "label": "orchestration",
            "registry_id": "ops.control-plane.orchestration",
            "inputs": {"audit_json": "<workspace-root>/outputs/logs/ops/audit/latest.json"},
        },
        {
            "label": "promoter-feature-matrix",
            "registry_id": "usr.data-plane.promoter-feature-matrix",
            "inputs": {
                "usr_root": "<usr-root>",
                "dataset": "<dataset>",
            },
        },
    ]


def test_cli_progress_scaffold_emits_json_metadata() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "progress",
            "scaffold",
            "opal.downstream.usr-infer-x-active-learning",
            "--repo-root",
            str(_repo_root()),
            "--campaign-id",
            "active_learning_demo",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["campaign_id"] == "active_learning_demo"
    assert payload["manifest"]["steps"] == [
        {
            "label": "usr-infer-x-active-learning",
            "registry_id": "opal.downstream.usr-infer-x-active-learning",
            "inputs": {"opal_config": "<opal-workdir>/configs/campaign.yaml"},
        }
    ]
    assert payload["steps"][0]["progress_kind"] == "opal-campaign-state"
    assert payload["steps"][0]["required_inputs"] == [
        {
            "cli_flag": "--opal-config",
            "manifest_key": "opal_config",
            "placeholder": "<opal-workdir>/configs/campaign.yaml",
            "summary": "Canonical OPAL campaign config used to resolve campaign.workdir.",
        }
    ]


def test_cli_progress_scaffold_supports_related_to_expansion() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "progress",
            "scaffold",
            "--repo-root",
            str(_repo_root()),
            "--related-to",
            "usr.data-plane.promoter-feature-matrix",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert [step["registry_id"] for step in payload["steps"]] == [
        "usr.data-plane.promoter-feature-matrix",
        "usr.data-plane.multi-source-source-of-truth",
        "usr.data-plane.construct-infer-source-of-truth",
        "cluster.downstream.exploratory-clustering",
        "opal.downstream.usr-infer-x-active-learning",
    ]
    assert payload["manifest"]["steps"][0]["label"] == "promoter-feature-matrix"
    assert payload["manifest"]["steps"][0]["inputs"]["dataset"] == "<dataset>"


def test_cli_progress_scaffold_dedupes_related_to_expansion_and_explicit_ids() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "progress",
            "scaffold",
            "opal.downstream.usr-infer-x-active-learning",
            "--repo-root",
            str(_repo_root()),
            "--related-to",
            "usr.data-plane.promoter-feature-matrix",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    registry_ids = [step["registry_id"] for step in payload["steps"]]
    assert registry_ids.count("opal.downstream.usr-infer-x-active-learning") == 1
    assert registry_ids[0] == "usr.data-plane.promoter-feature-matrix"


def test_cli_progress_scaffold_rejects_unknown_registry_id_with_suggestions() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "progress",
            "scaffold",
            "usr.data-plane.promoter-feature",
            "--repo-root",
            str(_repo_root()),
        ],
    )

    assert result.exit_code == 2
    assert "Progress contract error: unknown registry id: usr.data-plane.promoter-feature" in result.output
    assert "Did you mean:" in result.output
    assert "usr.data-plane.promoter-feature-matrix" in result.output


def test_cli_progress_scaffold_rejects_unknown_related_to_registry_id_with_suggestions() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "progress",
            "scaffold",
            "--repo-root",
            str(_repo_root()),
            "--related-to",
            "usr.data-plane.promoter-feature",
        ],
    )

    assert result.exit_code == 2
    assert "Progress contract error: unknown registry id: usr.data-plane.promoter-feature" in result.output
    assert "Did you mean:" in result.output
    assert "usr.data-plane.promoter-feature-matrix" in result.output


def test_cli_progress_scaffold_writes_manifest_file_and_requires_force() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        out_path = Path("manifests") / "campaign_status.yaml"

        result = runner.invoke(
            app,
            [
                "progress",
                "scaffold",
                "ops.control-plane.orchestration",
                "--repo-root",
                str(_repo_root()),
                "--out",
                str(out_path),
            ],
        )

        assert result.exit_code == 0
        assert out_path.exists()
        payload = yaml.safe_load(out_path.read_text(encoding="utf-8"))
        assert payload["steps"][0]["inputs"]["audit_json"] == "<workspace-root>/outputs/logs/ops/audit/latest.json"

        second_result = runner.invoke(
            app,
            [
                "progress",
                "scaffold",
                "ops.control-plane.orchestration",
                "--repo-root",
                str(_repo_root()),
                "--out",
                str(out_path),
            ],
        )

        assert second_result.exit_code == 2
        assert f"Progress contract error: file exists: {out_path}" in second_result.output


def test_cli_progress_scaffold_requires_registry_ids_or_related_to() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "progress",
            "scaffold",
            "--repo-root",
            str(_repo_root()),
        ],
    )

    assert result.exit_code == 2
    assert "progress scaffold requires at least one registry id or --related-to" in result.output
    assert "uv run ops catalog list --simple" in result.output
    assert "uv run ops progress scaffold --related-to <registry-id>" in result.output


def test_cli_progress_show_suggests_close_registry_ids() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "progress",
            "show",
            "usr.data-plane.promoter-feature",
            "--repo-root",
            str(_repo_root()),
        ],
    )

    assert result.exit_code == 2
    assert "Progress contract error: unknown registry id: usr.data-plane.promoter-feature" in result.output
    assert "Did you mean:" in result.output
    assert "usr.data-plane.promoter-feature-matrix" in result.output


def test_cli_progress_show_rejects_missing_required_surface_arguments() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "progress",
            "show",
            "ops.control-plane.orchestration",
            "--repo-root",
            str(_repo_root()),
        ],
    )

    assert result.exit_code == 2
    assert "progress kind 'ops-audit-json' requires --audit-json" in result.output
    assert "Required inputs for ops.control-plane.orchestration:" in result.output
    assert "--audit-json <workspace-root>/outputs/logs/ops/audit/latest.json" in result.output
    assert "Workspace-scoped orchestration audit JSON emitted by ops runbook execute." in result.output
    assert "uv run ops progress explain ops.control-plane.orchestration" in result.output
    assert "uv run ops progress scaffold ops.control-plane.orchestration" in result.output


def test_cli_progress_show_surfaces_optional_opal_workdir_hint_when_inputs_are_missing() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "progress",
            "show",
            "opal.downstream.usr-infer-x-active-learning",
            "--repo-root",
            str(_repo_root()),
        ],
    )

    assert result.exit_code == 2
    assert "progress kind 'opal-campaign-state' requires --opal-config or --opal-workdir" in result.output
    assert "Required inputs for opal.downstream.usr-infer-x-active-learning:" in result.output
    assert "Also accepted:" in result.output
    assert "--opal-workdir" in result.output


def test_cli_progress_explain_reports_required_inputs_and_next_commands() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "progress",
            "explain",
            "opal.downstream.usr-infer-x-active-learning",
            "--repo-root",
            str(_repo_root()),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["progress_kind"] == "opal-campaign-state"
    assert payload["provider_id"] == "builtin.opal"
    assert payload["required_inputs"] == [
        {
            "cli_flag": "--opal-config",
            "manifest_key": "opal_config",
            "placeholder": "<opal-workdir>/configs/campaign.yaml",
            "summary": "Canonical OPAL campaign config used to resolve campaign.workdir.",
        }
    ]
    assert payload["optional_inputs"] == [
        {
            "cli_flag": "--opal-workdir",
            "summary": (
                "Use when you want to point directly at the OPAL campaign workdir instead of resolving it from config."
            ),
        }
    ]
    assert payload["next_commands"]["catalog_show"] == (
        "uv run ops catalog show opal.downstream.usr-infer-x-active-learning"
    )
    assert payload["next_commands"]["progress_show"] == (
        "uv run ops progress show opal.downstream.usr-infer-x-active-learning "
        "--opal-config <opal-workdir>/configs/campaign.yaml"
    )
    assert payload["notes"] == [
        "Prefer `--opal-config` so Ops resolves `campaign.workdir` relative "
        "to the campaign root, matching OPAL's config contract."
    ]


def test_cli_progress_kinds_reports_provider_owned_inventory() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "progress",
            "kinds",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    kinds = {entry["progress_kind"]: entry for entry in payload["progress_kinds"]}
    assert kinds["promoter-study-preflight"]["provider_id"] == "study.stress_promoter_ethanol_cipro"
    assert kinds["promoter-study-preflight"]["optional_inputs"] == [
        {
            "cli_flag": "--study-dir",
            "summary": (
                "Checked-in promoter-study directory containing campaign.yaml, "
                "datasets.yaml, status.md, and ops.study.yaml."
            ),
        },
        {
            "cli_flag": "--scope",
            "summary": (
                "Preflight scope for promoter-study-preflight surfaces: "
                "`full` runs the whole suite; `next` focuses the next actionable "
                "phase and defers later-lane blockers."
            ),
        },
    ]
    assert kinds["ops-audit-json"]["notes"] == [
        "Smallest positive control-plane demo: run "
        "`uv run ops runbook execute ... --no-submit "
        "--audit-json <workspace-root>/outputs/logs/ops/audit/<file>.json`, "
        "then pass the same audit path to `ops progress show`.",
        "On workstations without `qstat`, add `--allow-missing-qstat` so the queue probe stays explicit "
        "but non-fatal during a dry-run demo.",
    ]


def test_cli_progress_explain_includes_related_scaffold_when_route_has_neighbors() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "progress",
            "explain",
            "usr.data-plane.promoter-feature-matrix",
            "--repo-root",
            str(_repo_root()),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["next_commands"]["progress_scaffold_related"] == (
        "uv run ops progress scaffold --related-to usr.data-plane.promoter-feature-matrix"
    )


def test_cli_progress_campaign_missing_manifest_suggests_scaffold() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "progress",
            "campaign",
            "--repo-root",
            str(_repo_root()),
            "--manifest",
            "missing/campaign.yaml",
        ],
    )

    assert result.exit_code == 2
    assert "Progress contract error: campaign manifest not found" in result.output
    assert "Hint: check the manifest path from `pwd` or pass an absolute path." in result.output
    assert "uv run ops progress scaffold <registry-id> ..." in result.output
    assert "uv run ops progress scaffold --related-to <registry-id>" in result.output


def test_cli_progress_campaign_requires_versioned_manifest_contract() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        manifest_path = Path("manifests") / "campaign_status.yaml"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            yaml.safe_dump(
                {
                    "campaign_id": "missing_version",
                    "steps": [],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        result = runner.invoke(
            app,
            [
                "progress",
                "campaign",
                "--repo-root",
                str(_repo_root()),
                "--manifest",
                str(manifest_path),
            ],
        )

        assert result.exit_code == 2
        assert "Progress contract error: campaign manifest must declare version: 2" in result.output


def test_cli_progress_campaign_rejects_top_level_step_inputs() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        manifest_path = Path("manifests") / "campaign_status.yaml"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            yaml.safe_dump(
                {
                    "version": 2,
                    "path_base": "repo",
                    "campaign_id": "invalid_step_shape",
                    "steps": [
                        {
                            "label": "orchestration",
                            "registry_id": "ops.control-plane.orchestration",
                            "audit_json": "repo:artifacts/latest.json",
                        }
                    ],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

        result = runner.invoke(
            app,
            [
                "progress",
                "campaign",
                "--repo-root",
                str(_repo_root()),
                "--manifest",
                str(manifest_path),
            ],
        )

        assert result.exit_code == 2
        assert "must place provider inputs under 'inputs': audit_json" in result.output
