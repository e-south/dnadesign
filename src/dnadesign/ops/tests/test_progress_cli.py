"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/tests/test_progress_cli.py

Contract tests for the read-only ops status surface.

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

import dnadesign.usr as usr_pkg
from dnadesign.devtools.tests.support.usr import register_test_namespace
from dnadesign.ops.cli import app
from dnadesign.ops.preflight import CommandExecution
from dnadesign.ops.status.service import run_status_kind
from dnadesign.usr import (
    Dataset,
    SequenceViewRecord,
    ensure_sequence_contract_namespaces,
    with_overlay_metadata,
    write_sequence_views,
)

STRESS_ETHANOL_CIPRO_GROWTH_STATUS_SERVICE_MODULE = (
    "dnadesign.studies.units.stress_ethanol_cipro_growth.operations.status.service"
)
STRESS_ETHANOL_CIPRO_GROWTH_RUN_PREFLIGHT_COMMAND_REF = (
    f"{STRESS_ETHANOL_CIPRO_GROWTH_STATUS_SERVICE_MODULE}.run_preflight_command"
)


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


def test_progress_command_source_is_decomposed_by_responsibility() -> None:
    source_root = _repo_root() / "src" / "dnadesign" / "ops" / "cli" / "commands"
    budgets = {
        "progress.py": 550,
        "progress_render.py": 350,
        "progress_status_specs.py": 150,
    }

    for filename, max_lines in budgets.items():
        path = source_root / filename
        assert path.is_file(), filename
        line_count = len(path.read_text(encoding="utf-8").splitlines())
        assert line_count <= max_lines, f"{filename} has {line_count} lines > {max_lines}"


def _stress_ethanol_cipro_growth_status(
    study_dir: Path | None, *, repo_root: Path | None
) -> tuple[str, str, dict[str, object]]:
    inputs: dict[str, object] = {}
    if study_dir is not None:
        inputs["study_dir"] = study_dir
    return run_status_kind(
        "stress-ethanol-cipro-growth-status",
        repo_root=repo_root,
        raw_inputs=inputs,
    )


def _stress_ethanol_cipro_growth_preflight(
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
    return run_status_kind(
        "stress-ethanol-cipro-growth-preflight",
        repo_root=repo_root,
        raw_inputs=inputs,
    )


def _write_study_index(index_path: Path) -> None:
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(
        (
            "version: 1\n"
            "active_study_id: stress_ethanol_cipro_growth\n"
            "studies:\n"
            "  - study_id: stress_ethanol_cipro_growth\n"
            "    title: Demo study\n"
            "    record_root: docs/studies/stress_ethanol_cipro_growth\n"
        ),
        encoding="utf-8",
    )


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
                    "command": "uv run ops runbook diagnostics session-counts",
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


def _write_sync_audit(path: Path, *, transfer_state: str, primary_changed: bool, wrapped: bool = False) -> None:
    data = {
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
    payload = {"usr_output_version": 1, "data": data} if wrapped else data
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_usr_dataset(root: Path, dataset: str, *, rows: int = 2) -> None:
    registry_src = Path(usr_pkg.__file__).resolve().parent / "datasets" / "registry.yaml"
    registry_path = root / "registry.yaml"
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    if not registry_path.exists():
        registry_path.write_text(registry_src.read_text(encoding="utf-8"), encoding="utf-8")
    register_test_namespace(
        root,
        namespace="densegen",
        columns_spec="densegen__plan:string,densegen__required_regulators:list<string>",
    )
    dataset_dir = root / dataset
    dataset_dir.mkdir(parents=True, exist_ok=True)
    derived_dir = dataset_dir / "_derived"
    derived_dir.mkdir(parents=True, exist_ok=True)
    ids = [f"{dataset.replace('/', '_')}_{idx}" for idx in range(rows)]
    sequences = ["ACGT"] * rows
    table = pa.table(
        {
            "id": ids,
            "sequence": sequences,
            "length": [4] * rows,
            "infer__demo__embedding": [[1.0]] * rows,
        }
    )
    pq.write_table(table, dataset_dir / "records.parquet")
    if dataset == "densegen_demo_anchor" or dataset.startswith("promoter/"):
        densegen_overlay = pa.table(
            {
                "id": ids,
                "densegen__plan": ["ethanol__sigma70_b"] * rows,
                "densegen__required_regulators": [["cpxR"]] * rows,
            }
        )
        densegen_overlay = with_overlay_metadata(
            densegen_overlay,
            namespace="densegen",
            key="id",
            created_at="2026-04-13T00:00:00Z",
        )
        pq.write_table(densegen_overlay, derived_dir / "densegen.parquet")
    (dataset_dir / ".events.log").write_text("{}\n{}\n", encoding="utf-8")


def _write_sequence_view_sidecar(
    root: Path,
    dataset: str,
    *,
    product_kind: str,
    context_kind: str,
    recommended_pooling: str,
    orientations: tuple[str, ...],
) -> None:
    ensure_sequence_contract_namespaces(root)
    ds = Dataset(root, dataset)
    table = pq.read_table(ds.records_path, columns=["id", "length"])
    ids = [str(value) for value in table.column("id").to_pylist()]
    lengths = [int(value) for value in table.column("length").to_pylist()]
    if len(ids) != len(orientations):
        raise AssertionError(f"test setup orientation count mismatch for {dataset}")
    rows = []
    for row_id, length, orientation in zip(ids, lengths, orientations, strict=True):
        rows.append(
            SequenceViewRecord(
                sequence_id=row_id,
                view_name=f"{row_id}_{product_kind}_{orientation}",
                product_kind=product_kind,
                context_kind=context_kind,
                orientation=orientation,  # type: ignore[arg-type]
                analysis_only=product_kind == "analysis_window",
                source_dataset_id=dataset,
                anchor_start_0=0 if recommended_pooling == "anchor_mean" else None,
                anchor_end_0=length if recommended_pooling == "anchor_mean" else None,
                forward_anchor_start_0=0 if recommended_pooling == "anchor_mean" else None,
                forward_anchor_end_0=length if recommended_pooling == "anchor_mean" else None,
                recommended_pooling=recommended_pooling,  # type: ignore[arg-type]
                created_at="2026-04-28T00:00:00+00:00",
                created_by="test",
            )
        )
    write_sequence_views(ds, rows, conflict_policy="error")


def _write_promoter_ops_contract(
    path: Path,
    *,
    current_phase_id: str,
    phase_rows: list[dict[str, object]],
) -> None:
    phase_ids = {str(phase["id"]) for phase in phase_rows}
    artifacts = {
        "densegen_anchor_source": {
            "artifact_type": "dataset",
            "dataset_id": "densegen_demo_anchor",
            "ref": "repo:usr_root/densegen_demo_anchor",
        }
    }
    if {"merged_anchor_set", "construct_context_expansion", "infer_batch_preparation"} & phase_ids:
        artifacts["merged_anchor_dataset"] = {
            "artifact_type": "dataset",
            "dataset_id": "promoter/demo_anchor_set",
            "ref": "repo:usr_root/promoter/demo_anchor_set",
        }
    if {"construct_context_expansion", "infer_batch_preparation"} & phase_ids:
        artifacts["construct_context_dataset"] = {
            "artifact_type": "dataset",
            "dataset_id": "promoter/demo_construct_contexts",
            "ref": "repo:usr_root/promoter/demo_construct_contexts",
        }

    execution_surfaces: dict[str, dict[str, object]] = {}
    if "construct_context_expansion" in phase_ids:
        execution_surfaces["construct_workspace"] = {
            "surface_type": "workspace",
            "workspace_ref": "repo:workspace/construct",
        }
        execution_surfaces["construct_workspace_doctor"] = {
            "surface_type": "command",
            "argv": [
                "uv",
                "run",
                "construct",
                "workspace",
                "doctor",
                "--workspace",
                "workspace/construct",
            ],
        }
        execution_surfaces["construct_runtime_slot_a_window"] = {
            "surface_type": "command",
            "argv": [
                "uv",
                "run",
                "construct",
                "workspace",
                "validate-project",
                "--workspace",
                "workspace/construct",
                "--project",
                "slot_a_window",
                "--runtime",
            ],
        }
    if "densegen_growth" in phase_ids:
        execution_surfaces["densegen_batch"] = {
            "surface_type": "runbook",
            "runbook_ref": "repo:workspace/runbooks/densegen.yaml",
        }
        execution_surfaces["densegen_probe_solver"] = {
            "surface_type": "command",
            "argv": [
                "uv",
                "run",
                "dense",
                "validate-config",
                "--probe-solver",
                "-c",
                "workspace/densegen/config.yaml",
            ],
        }
    if {"infer_batch_preparation", "infer_anchor_only_20b", "infer_anchor_plus_template_20b"} & phase_ids:
        execution_surfaces["infer_batch_20b_anchor_only"] = {
            "surface_type": "runbook",
            "runbook_ref": "repo:workspace/runbooks/infer_anchor_only_20b.yaml",
        }
        execution_surfaces["infer_batch_20b_anchor_plus_template"] = {
            "surface_type": "runbook",
            "runbook_ref": "repo:workspace/runbooks/infer_anchor_plus_template_20b.yaml",
        }
    if {"infer_batch_preparation", "infer_anchor_only_7b", "infer_anchor_plus_template_7b"} & phase_ids:
        execution_surfaces["infer_batch_7b_anchor_only"] = {
            "surface_type": "runbook",
            "runbook_ref": "repo:workspace/runbooks/infer_anchor_only_7b.yaml",
        }
        execution_surfaces["infer_batch_7b_anchor_plus_template"] = {
            "surface_type": "runbook",
            "runbook_ref": "repo:workspace/runbooks/infer_anchor_plus_template_7b.yaml",
        }
    if {
        "infer_batch_preparation",
        "infer_anchor_only_20b",
        "infer_anchor_plus_template_20b",
        "infer_anchor_only_7b",
        "infer_anchor_plus_template_7b",
    } & phase_ids:
        execution_surfaces["infer_workspace"] = {
            "surface_type": "workspace",
            "workspace_ref": "repo:workspace/infer",
        }
        execution_surfaces["scheduler_default"] = {
            "surface_type": "scheduler",
            "backend": "sge",
        }
        for config_label, config_path in (
            ("anchor_only_20b", "workspace/infer/config.anchor_only.evo2_20b.yaml"),
            ("anchor_plus_template_20b", "workspace/infer/config.anchor_plus_template.evo2_20b.yaml"),
            ("anchor_only_7b", "workspace/infer/config.anchor_only.evo2_7b.yaml"),
            ("anchor_plus_template_7b", "workspace/infer/config.anchor_plus_template.evo2_7b.yaml"),
        ):
            execution_surfaces[f"infer_validate_{config_label}"] = {
                "surface_type": "command",
                "argv": ["uv", "run", "infer", "validate", "config", "--config", config_path],
            }
            execution_surfaces[f"infer_dry_run_{config_label}"] = {
                "surface_type": "command",
                "argv": ["uv", "run", "infer", "run", "--config", config_path, "--dry-run"],
            }
            execution_surfaces[f"notify_profile_doctor_{config_label}"] = {
                "surface_type": "command",
                "argv": [
                    "uv",
                    "run",
                    "notify",
                    "profile",
                    "doctor",
                    "--profile",
                    f"workspace/infer/outputs/notify/infer/{config_label}/profile.json",
                    "--json",
                ],
            }
            execution_surfaces[f"notify_resolve_events_{config_label}"] = {
                "surface_type": "command",
                "argv": [
                    "uv",
                    "run",
                    "notify",
                    "setup",
                    "resolve-events",
                    "--tool",
                    "infer",
                    "--config",
                    config_path,
                    "--json",
                ],
            }
    group_phase_bindings = {
        group: phase_id
        for group, phase_id in {
            "densegen": "densegen_growth",
            "construct": "construct_context_expansion",
            "notify_environment": "infer_batch_preparation",
        }.items()
        if phase_id in phase_ids
    }
    target_phase_groups = {
        phase_id: groups
        for phase_id, groups in {
            "densegen_growth": ["densegen"],
            "merged_anchor_set": [],
            "construct_context_expansion": ["construct"],
            "infer_batch_preparation": [
                "infer",
                "notify_environment",
                "notify",
                "infer_batch_plan",
            ],
        }.items()
        if phase_id in phase_ids
    }
    checks = {phase_id: [] for phase_id in [str(phase["id"]) for phase in phase_rows]}
    if "densegen_growth" in phase_ids:
        checks["densegen_growth"] = [
            {
                "kind": "runbook_plan",
                "check_id": "densegen.batch.plan",
                "check_group": "densegen",
                "summary": "DenseGen batch runbook renders cleanly.",
                "required": True,
                "surface": "densegen_batch",
            },
            {
                "kind": "dataset_snapshot",
                "check_id": "densegen.anchor.rows",
                "check_group": "densegen",
                "summary": "DenseGen anchor dataset row count is visible.",
                "required": True,
                "artifact": "densegen_anchor_source",
                "target_rows": 2,
            },
            {
                "kind": "command",
                "check_id": "densegen.config.probe_solver",
                "check_group": "densegen",
                "summary": "DenseGen config probe completed.",
                "required": True,
                "surface": "densegen_probe_solver",
            },
        ]
    if "construct_context_expansion" in phase_ids:
        checks["construct_context_expansion"] = [
            {
                "kind": "path_exists",
                "check_id": "construct.anchor.dataset",
                "check_group": "construct",
                "summary": "Merged anchor dataset is materialized.",
                "required": True,
                "artifact": "merged_anchor_dataset",
            },
            {
                "kind": "workspace_layout",
                "check_id": "construct.workspace.layout",
                "check_group": "construct",
                "summary": "Construct workspace root is present.",
                "required": True,
                "surface": "construct_workspace",
            },
            {
                "kind": "command",
                "check_id": "construct.workspace.doctor",
                "check_group": "construct",
                "summary": "Construct workspace doctor completed.",
                "required": True,
                "surface": "construct_workspace_doctor",
            },
            {
                "kind": "command",
                "check_id": "construct.runtime.slot_a_window",
                "check_group": "construct",
                "summary": "Construct runtime validation completed.",
                "required": True,
                "surface": "construct_runtime_slot_a_window",
            },
        ]
    if "infer_batch_preparation" in phase_ids:
        infer_checks: list[dict[str, object]] = [
            {
                "kind": "path_exists",
                "check_id": "infer.construct.contexts",
                "check_group": "infer",
                "summary": "Construct contexts are present for infer.",
                "required": True,
                "artifact": "construct_context_dataset",
            },
            {
                "kind": "workspace_layout",
                "check_id": "infer.workspace.layout",
                "check_group": "infer",
                "summary": "Infer workspace root is present.",
                "required": True,
                "surface": "infer_workspace",
            },
            {
                "kind": "environment",
                "check_id": "notify.environment.webhook",
                "check_group": "notify_environment",
                "summary": "Batch notify secret is configured in the environment.",
                "required": True,
                "vars": ["NOTIFY_WEBHOOK", "NOTIFY_WEBHOOK_FILE"],
                "match_mode": "any",
            },
            {
                "kind": "environment",
                "check_id": "notify.environment.tls",
                "check_group": "notify_environment",
                "summary": "SSL_CERT_FILE is configured for notify profile doctor and live delivery.",
                "required": True,
                "vars": ["SSL_CERT_FILE"],
                "match_mode": "all",
            },
            {
                "kind": "gpu_availability",
                "check_id": "infer.local_gpu.visibility",
                "check_group": "infer",
                "summary": "At least one compatible GPU is visible on the current host.",
                "required": False,
                "min_visible": 1,
            },
        ]
        for config_label, phase_id in (
            ("anchor_only_20b", "infer_anchor_only_20b"),
            ("anchor_plus_template_20b", "infer_anchor_plus_template_20b"),
            ("anchor_only_7b", "infer_anchor_only_7b"),
            ("anchor_plus_template_7b", "infer_anchor_plus_template_7b"),
        ):
            infer_checks.append(
                {
                    "kind": "command",
                    "check_id": f"infer.validate.{config_label}",
                    "check_group": "infer",
                    "summary": "Infer config validation completed.",
                    "required": True,
                    "phase_id": phase_id,
                    "surface": f"infer_validate_{config_label}",
                }
            )
        for runtime_label, phase_id in (
            ("anchor_only_20b", "infer_anchor_only_20b"),
            ("anchor_plus_template_20b", "infer_anchor_plus_template_20b"),
            ("anchor_only_7b", "infer_anchor_only_7b"),
            ("anchor_plus_template_7b", "infer_anchor_plus_template_7b"),
        ):
            infer_checks.extend(
                [
                    {
                        "kind": "command",
                        "check_id": f"infer.dry_run.{runtime_label}",
                        "check_group": "infer",
                        "summary": "Infer dry-run completed.",
                        "required": True,
                        "phase_id": phase_id,
                        "surface": f"infer_dry_run_{runtime_label}",
                    },
                    {
                        "kind": "command",
                        "check_id": f"notify.profile.{runtime_label}",
                        "check_group": "notify",
                        "summary": "Notify profile doctor completed.",
                        "required": True,
                        "phase_id": phase_id,
                        "surface": f"notify_profile_doctor_{runtime_label}",
                    },
                    {
                        "kind": "command",
                        "check_id": f"notify.resolve_events.{runtime_label}",
                        "check_group": "notify",
                        "summary": "Notify events path resolved.",
                        "required": True,
                        "phase_id": phase_id,
                        "surface": f"notify_resolve_events_{runtime_label}",
                    },
                ]
            )
        infer_checks.extend(
            [
                {
                    "kind": "scheduler_queue",
                    "check_id": "infer.batch.queue",
                    "check_group": "infer_batch_plan",
                    "summary": "Scheduler queue is below the declared submit threshold.",
                    "required": False,
                    "surface": "scheduler_default",
                    "max_running_jobs": 3,
                },
                {
                    "kind": "runbook_plan",
                    "check_id": "infer.batch.20b.anchor_only.plan",
                    "check_group": "infer_batch_plan",
                    "phase_id": "infer_anchor_only_20b",
                    "summary": "Anchor-only 20B infer runbook renders cleanly.",
                    "required": True,
                    "surface": "infer_batch_20b_anchor_only",
                },
                {
                    "kind": "runbook_plan",
                    "check_id": "infer.batch.20b.anchor_plus_template.plan",
                    "check_group": "infer_batch_plan",
                    "phase_id": "infer_anchor_plus_template_20b",
                    "summary": "Anchor-plus-template 20B infer runbook renders cleanly.",
                    "required": True,
                    "surface": "infer_batch_20b_anchor_plus_template",
                },
                {
                    "kind": "runbook_plan",
                    "check_id": "infer.batch.7b.anchor_only.plan",
                    "check_group": "infer_batch_plan",
                    "phase_id": "infer_anchor_only_7b",
                    "summary": "Anchor-only 7B infer runbook renders cleanly.",
                    "required": True,
                    "surface": "infer_batch_7b_anchor_only",
                },
                {
                    "kind": "runbook_plan",
                    "check_id": "infer.batch.7b.anchor_plus_template.plan",
                    "check_group": "infer_batch_plan",
                    "phase_id": "infer_anchor_plus_template_7b",
                    "summary": "Anchor-plus-template 7B infer runbook renders cleanly.",
                    "required": True,
                    "surface": "infer_batch_7b_anchor_plus_template",
                },
            ]
        )
        checks["infer_batch_preparation"] = infer_checks
    path.write_text(
        yaml.safe_dump(
            {
                "version": 2,
                "study_id": "stress_ethanol_cipro_growth",
                "ops_surfaces": {
                    "status_kind": "stress-ethanol-cipro-growth-status",
                    "preflight_kind": "stress-ethanol-cipro-growth-preflight",
                },
                "title": "Demo study",
                "record_sources": {
                    "narrative_ref": "manifest:record/status.md",
                    "datasets_ref": "manifest:record/datasets.yaml",
                    "pipeline_ref": "manifest:operations/runtime/command-groups/pipeline.yaml",
                    "campaign_ref": "manifest:record/campaign.yaml",
                },
                "artifacts": artifacts,
                "execution_surfaces": execution_surfaces,
                "lifecycle": {
                    "current_phase": {
                        "strategy": "explicit",
                        "id": current_phase_id,
                    },
                    "phase_order": [str(phase["id"]) for phase in phase_rows],
                },
                "phases": phase_rows,
                "snapshot": {"summary_scope": "repo"},
                "preflight": {
                    "default_scope": "next",
                    "scopes": {
                        "next": {"include_phases": ["current_phase", "next_in_progress_phase"]},
                        "full": {"include_phases": ["all"]},
                    },
                    "group_phase_bindings": group_phase_bindings,
                    "checks": checks,
                    "next_scope": {
                        "target_phase_groups": target_phase_groups,
                        "runtime_phase_groups": ["infer", "notify", "infer_batch_plan"],
                        "runtime_shared_groups": ["notify_environment"],
                    },
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_stress_ethanol_cipro_growth_record(study_dir: Path, *, densegen_rows: int, densegen_target: int) -> None:
    repo_root = study_dir.parents[2]
    (study_dir / "record").mkdir(parents=True, exist_ok=True)
    (study_dir / "operations").mkdir(parents=True, exist_ok=True)
    (study_dir / "operations" / "runtime" / "command-groups").mkdir(parents=True, exist_ok=True)
    (study_dir / "record" / "campaign.yaml").write_text(
        "campaign_id: stress_ethanol_cipro_growth\nsteps: []\n",
        encoding="utf-8",
    )
    (study_dir / "record" / "status.md").write_text(
        "## stress_ethanol_cipro_growth\n\n- Canonical consolidated feature dataset: `n/a`\n",
        encoding="utf-8",
    )
    (study_dir / "record" / "datasets.yaml").write_text(
        yaml.safe_dump(
            {
                "study_id": "stress_ethanol_cipro_growth",
                "datasets": [
                    {
                        "role": "densegen_anchor",
                        "dataset": "densegen_demo_anchor",
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
    (study_dir / "operations" / "runtime" / "command-groups" / "pipeline.yaml").write_text(
        yaml.safe_dump(
            {
                "study_pipeline": {
                    "study_id": "stress_ethanol_cipro_growth",
                    "canonical_usr_root": "usr_root",
                    "row_targets": {
                        "densegen_anchor_minimum_before_first_full_lane_infer": densegen_target,
                    },
                    "execution_surfaces": {
                        "construct_workspace": "workspace/construct",
                        "infer_workspace": "workspace/infer",
                    },
                    "datasets": {
                        "densegen_anchor_source": "densegen_demo_anchor",
                        "merged_anchor_dataset": "promoter/demo_anchor_set",
                    },
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_promoter_ops_contract(
        study_dir / "operations" / "ops.study.yaml",
        current_phase_id="densegen_growth",
        phase_rows=[
            {
                "id": "densegen_growth",
                "status": "in_progress",
                "primary_dataset": "densegen_demo_anchor",
                "next_surface": "repo:workspace/densegen_batch.yaml",
            },
            {
                "id": "merged_anchor_set",
                "status": "ready",
                "output_dataset": "promoter/demo_anchor_set",
                "next_surface": "repo:workspace/merge.md",
            },
        ],
    )
    _write_usr_dataset(repo_root / "usr_root", "densegen_demo_anchor")
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
    (repo_root / "workspace" / "construct").mkdir(parents=True, exist_ok=True)
    (repo_root / "workspace" / "infer").mkdir(parents=True, exist_ok=True)


def _write_stress_ethanol_cipro_growth_preflight_record(
    study_dir: Path,
    *,
    densegen_rows: int = 2,
    anchor_rows: int = 2,
    construct_rows: int = 2,
) -> None:
    repo_root = study_dir.parents[2]
    (study_dir / "record").mkdir(parents=True, exist_ok=True)
    (study_dir / "operations").mkdir(parents=True, exist_ok=True)
    (study_dir / "operations" / "runtime" / "command-groups").mkdir(parents=True, exist_ok=True)
    (study_dir / "record" / "campaign.yaml").write_text(
        "campaign_id: stress_ethanol_cipro_growth\nsteps: []\n", encoding="utf-8"
    )
    (study_dir / "record" / "status.md").write_text(
        "## stress_ethanol_cipro_growth\n\n- Canonical consolidated feature dataset: `n/a`\n",
        encoding="utf-8",
    )
    (study_dir / "record" / "datasets.yaml").write_text(
        yaml.safe_dump(
            {
                "study_id": "stress_ethanol_cipro_growth",
                "datasets": [
                    {
                        "role": "densegen_anchor",
                        "dataset": "densegen_demo_anchor",
                        "usr_root": "usr_root",
                        "status": "present",
                    },
                    {
                        "role": "promoter_references",
                        "dataset": "usr_promoter_references",
                        "usr_root": "usr_root",
                        "status": "present",
                    },
                    {
                        "role": "construct_template_seed",
                        "dataset": "usr_pdual10_plasmid_template",
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
    (study_dir / "operations" / "runtime" / "command-groups" / "pipeline.yaml").write_text(
        yaml.safe_dump(
            {
                "study_pipeline": {
                    "study_id": "stress_ethanol_cipro_growth",
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
                        "densegen_anchor_source": "densegen_demo_anchor",
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
                }
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    _write_promoter_ops_contract(
        study_dir / "operations" / "ops.study.yaml",
        current_phase_id="infer_batch_preparation",
        phase_rows=[
            {"id": "densegen_growth", "status": "parallel_optional"},
            {"id": "merged_anchor_set", "status": "complete"},
            {"id": "construct_context_expansion", "status": "complete"},
            {"id": "infer_batch_preparation", "status": "in_progress"},
            {
                "id": "infer_anchor_only_20b",
                "status": "planned",
                "next_surface": "repo:workspace/runbooks/infer_anchor_only_20b.yaml",
            },
            {
                "id": "infer_anchor_plus_template_20b",
                "status": "planned",
                "next_surface": "repo:workspace/runbooks/infer_anchor_plus_template_20b.yaml",
            },
            {
                "id": "infer_anchor_only_7b",
                "status": "planned",
                "next_surface": "repo:workspace/runbooks/infer_anchor_only_7b.yaml",
            },
            {
                "id": "infer_anchor_plus_template_7b",
                "status": "planned",
                "next_surface": "repo:workspace/runbooks/infer_anchor_plus_template_7b.yaml",
            },
        ],
    )

    _write_usr_dataset(repo_root / "usr_root", "densegen_demo_anchor", rows=densegen_rows)
    _write_usr_dataset(repo_root / "usr_root", "usr_promoter_references")
    _write_usr_dataset(repo_root / "usr_root", "usr_pdual10_plasmid_template")
    _write_usr_dataset(repo_root / "usr_root", "promoter/demo_anchor_set", rows=anchor_rows)
    _write_usr_dataset(repo_root / "usr_root", "promoter/demo_construct_contexts", rows=construct_rows)

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
    payload = _opal_state_payload(workdir=workdir, rounds=rounds)
    (workdir / "state.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_relative_opal_campaign(config_path: Path, *, rounds: list[dict[str, object]]) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    campaign_root = config_path.parent.parent if config_path.parent.name == "configs" else config_path.parent
    campaign_root.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump({"campaign": {"workdir": "."}}, sort_keys=False),
        encoding="utf-8",
    )
    payload = _opal_state_payload(workdir=campaign_root, rounds=rounds)
    (campaign_root / "state.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _opal_state_payload(*, workdir: Path, rounds: list[dict[str, object]]) -> dict[str, object]:
    return {
        "version": 2,
        "campaign_slug": "demo_campaign",
        "campaign_name": "Demo campaign",
        "workdir": str(workdir),
        "data_location": {"kind": "local", "path": "records.parquet"},
        "x_column_name": "infer__demo",
        "y_column_name": "measured_activity",
        "created_at": "2026-06-25T00:00:00Z",
        "updated_at": "2026-06-25T00:00:00Z",
        "representation_vector_dimension": 2,
        "representation_transform": {"kind": "none"},
        "training_policy": {"kind": "test"},
        "performance": {},
        "rounds": [_opal_round_payload(round_payload) for round_payload in rounds],
        "backlog": {"number_of_selected_but_not_yet_labeled_candidates_total": 0},
    }


def _opal_round_payload(round_payload: dict[str, object]) -> dict[str, object]:
    round_index = int(round_payload.get("round_index", 0))
    payload: dict[str, object] = {
        "round_index": round_index,
        "run_id": f"run_{round_index:03d}",
        "round_name": f"round_{round_index}",
        "round_dir": f"/tmp/opal_campaign/outputs/rounds/round_{round_index}",
        "labels_used_rounds": [],
        "number_of_training_examples_used_in_round": 0,
        "number_of_candidates_scored_in_round": 0,
        "selection_top_k_requested": 0,
        "selection_top_k_effective_after_ties": 0,
        "model": {},
        "metrics": {},
        "durations_sec": {},
        "seeds": {},
        "artifacts": {},
        "writebacks": {},
        "warnings": [],
        "status": "completed",
    }
    payload.update(round_payload)
    return payload


def _write_opal_usr_campaign_without_records(config_path: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(
            {
                "campaign": {"workdir": "."},
                "data": {
                    "location": {
                        "kind": "usr",
                        "path": "usr/datasets",
                        "dataset": "demo_candidates",
                    },
                    "x_column_name": "X",
                    "y_column_name": "Y",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )


def _write_opal_usr_campaign_with_records(config_path: Path) -> Path:
    _write_opal_usr_campaign_without_records(config_path)
    records_path = config_path.parent.parent / "usr" / "datasets" / "demo_candidates" / "records.parquet"
    records_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "id": ["a", "b"],
                "bio_type": ["dna", "dna"],
                "sequence": ["AAAA", "CCCC"],
                "alphabet": ["dna_4", "dna_4"],
                "X": [[0.1, 0.2], [0.3, 0.4]],
            }
        ),
        records_path,
    )
    return records_path


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
        assert payload["status_kind"] == "ops-audit-json"
        assert payload["observes_plane"] == "control"
        assert payload["surface_type"] == "orchestration_audit"
        assert payload["cost_class"] == "cheap"
        assert payload["summary_scope"] == "workspace"
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
        assert payload["status_kind"] == "usr-sync-audit"
        assert payload["state"] == "attention"
        assert payload["evidence"]["transfer_state"] == "DRY-RUN"


def test_cli_progress_show_reports_wrapped_usr_diff_sync_audit_surface() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        audit_path = Path("sync") / "diff.json"
        _write_sync_audit(audit_path, transfer_state="DIFF-ONLY", primary_changed=False, wrapped=True)

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
        assert payload["status_kind"] == "usr-sync-audit"
        assert payload["state"] == "ok"
        assert payload["summary"] == "demo_dataset: DIFF-ONLY"
        assert payload["evidence"]["dataset"] == "demo_dataset"
        assert payload["evidence"]["transfer_state"] == "DIFF-ONLY"


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
        assert payload["status_kind"] == "usr-dataset-state"
        assert payload["state"] == "ok"
        assert payload["evidence"]["rows"] == 2


def test_cli_progress_show_reports_stress_ethanol_cipro_growth_record_surface() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        repo_root = Path.cwd()
        (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.0.0'\n", encoding="utf-8")
        (repo_root / "src" / "dnadesign").mkdir(parents=True, exist_ok=True)
        study_dir = repo_root / "docs" / "studies" / "stress_ethanol_cipro_growth"
        study_index = repo_root / "docs" / "studies" / "index.yaml"
        _write_study_index(study_index)
        _write_stress_ethanol_cipro_growth_record(study_dir, densegen_rows=2, densegen_target=5)

        result = runner.invoke(
            app,
            [
                "progress",
                "show",
                "studies.stress-ethanol-cipro-growth.status",
                "--repo-root",
                str(_repo_root()),
                "--study-dir",
                str(study_dir),
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["status_kind"] == "stress-ethanol-cipro-growth-status"
        assert payload["state"] == "attention"
        assert payload["evidence"]["study_id"] == "stress_ethanol_cipro_growth"
        assert payload["evidence"]["study_selection_source"] == "explicit"
        assert payload["evidence"]["is_active_study"] is True
        assert payload["evidence"]["densegen_rows"] == 2
        assert payload["evidence"]["densegen_row_target"] == 5
        assert payload["evidence"]["densegen_row_gap"] == 3
        assert payload["evidence"]["source_growth_state"]["state"] == "attention"
        assert payload["evidence"]["handoff_readiness_state"]["state"] == "attention"
        assert payload["evidence"]["planned_outputs_state"]["state"] == "ok"
        assert payload["evidence"]["next_ready_phase"]["id"] == "merged_anchor_set"
        assert payload["evidence"]["datasets"][0]["dataset"] == "densegen_demo_anchor"
        assert "source gate active densegen_demo_anchor 2/5 rows (gap=3)" in payload["summary"]
        assert "handoff outputs pending promoter/demo_anchor_set" in payload["summary"]


def test_cli_progress_show_marks_pipeline_only_execution_surfaces_as_derived() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        repo_root = Path.cwd()
        (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.0.0'\n", encoding="utf-8")
        (repo_root / "src" / "dnadesign").mkdir(parents=True, exist_ok=True)
        study_dir = repo_root / "docs" / "studies" / "stress_ethanol_cipro_growth"
        study_index = repo_root / "docs" / "studies" / "index.yaml"
        _write_study_index(study_index)
        _write_stress_ethanol_cipro_growth_preflight_record(study_dir)

        result = runner.invoke(
            app,
            [
                "progress",
                "show",
                "studies.stress-ethanol-cipro-growth.status",
                "--repo-root",
                str(_repo_root()),
                "--study-dir",
                str(study_dir),
                "--json",
            ],
        )

        assert result.exit_code == 0
        payload = json.loads(result.output)
        surface_labels = {item["label"] for item in payload["evidence"]["execution_surfaces"]}
        derived_labels = {item["label"] for item in payload["evidence"]["derived_execution_surfaces"]}

        assert "densegen_batch" in surface_labels
        assert "infer_batch_7b_anchor_only" in surface_labels
        assert "densegen_batch_with_notify" not in surface_labels
        assert "infer_batch_7b_with_notify.anchor_only" not in surface_labels
        assert "densegen_batch_with_notify" in derived_labels
        assert "infer_batch_7b_with_notify.anchor_only" in derived_labels
        assert payload["evidence"]["missing_execution_surfaces"] == []
        assert payload["evidence"]["missing_derived_execution_surfaces"] == []


def test_stress_ethanol_cipro_study_progress_discovers_active_study_from_repo_root() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        repo_root = Path.cwd()
        (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.0.0'\n", encoding="utf-8")
        (repo_root / "src" / "dnadesign").mkdir(parents=True, exist_ok=True)
        study_dir = repo_root / "docs" / "studies" / "stress_ethanol_cipro_growth"
        study_index = repo_root / "docs" / "studies" / "index.yaml"
        _write_study_index(study_index)
        _write_stress_ethanol_cipro_growth_record(study_dir, densegen_rows=2, densegen_target=5)

        state, summary, evidence = _stress_ethanol_cipro_growth_status(None, repo_root=repo_root)

        assert state == "attention"
        assert evidence["study_id"] == "stress_ethanol_cipro_growth"
        assert evidence["study_selection_source"] == "active_registry"
        assert evidence["active_study_registry_path"] == str(study_index)
        assert evidence["study_dir"] == str(study_dir)
        assert evidence["infer_notify_profiles"] == {}
        assert evidence["infer_notify_profile_errors"] == {}
        assert "source gate active densegen_demo_anchor 2/5 rows (gap=3)" in summary
        assert "handoff outputs pending promoter/demo_anchor_set" in summary


def test_stress_ethanol_cipro_study_progress_resolves_relative_study_dir_against_repo_root() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        repo_root = Path.cwd() / "repo"
        repo_root.mkdir(parents=True, exist_ok=True)
        (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.0.0'\n", encoding="utf-8")
        (repo_root / "src" / "dnadesign").mkdir(parents=True, exist_ok=True)
        study_dir = repo_root / "docs" / "studies" / "stress_ethanol_cipro_growth"
        study_index = repo_root / "docs" / "studies" / "index.yaml"
        _write_study_index(study_index)
        _write_stress_ethanol_cipro_growth_record(study_dir, densegen_rows=2, densegen_target=5)

        state, _, evidence = _stress_ethanol_cipro_growth_status(
            Path("docs/studies/stress_ethanol_cipro_growth"),
            repo_root=repo_root,
        )

        assert state == "attention"
        assert evidence["study_selection_source"] == "explicit"
        assert evidence["study_dir"] == str(study_dir)


def test_stress_ethanol_cipro_study_progress_demotes_source_gate_once_handoffs_exceed_target() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        repo_root = Path.cwd()
        (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.0.0'\n", encoding="utf-8")
        (repo_root / "src" / "dnadesign").mkdir(parents=True, exist_ok=True)
        study_dir = repo_root / "docs" / "studies" / "stress_ethanol_cipro_growth"
        study_index = repo_root / "docs" / "studies" / "index.yaml"
        _write_study_index(study_index)
        _write_stress_ethanol_cipro_growth_preflight_record(study_dir, densegen_rows=2, anchor_rows=7, construct_rows=7)

        pipeline_path = study_dir / "operations" / "runtime" / "command-groups" / "pipeline.yaml"
        pipeline_payload = yaml.safe_load(pipeline_path.read_text(encoding="utf-8"))
        pipeline_payload["study_pipeline"]["row_targets"] = {
            "densegen_anchor_minimum_before_first_full_lane_infer": 5,
        }
        pipeline_path.write_text(yaml.safe_dump(pipeline_payload, sort_keys=False), encoding="utf-8")
        datasets_path = study_dir / "record" / "datasets.yaml"
        datasets_payload = yaml.safe_load(datasets_path.read_text(encoding="utf-8"))
        for entry in datasets_payload["datasets"]:
            if entry["dataset"] in {
                "promoter/demo_anchor_set",
                "promoter/demo_construct_contexts",
            }:
                entry["status"] = "present"
        datasets_path.write_text(yaml.safe_dump(datasets_payload, sort_keys=False), encoding="utf-8")

        state, summary, evidence = _stress_ethanol_cipro_growth_status(None, repo_root=repo_root)

        assert state == "ok"
        assert "handoffs ready anchor=7 construct=7" in summary
        assert "source gate superseded by downstream handoffs densegen_demo_anchor 2/5 rows (gap=3)" in summary
        assert "attention_reasons" not in evidence
        assert evidence["source_growth_state"]["state"] == "ok"
        assert evidence["source_growth_state"]["target_met"] is False
        assert evidence["source_growth_state"]["gates_current_phase"] is False
        assert evidence["source_growth_state"]["superseded_by_handoffs"] is True
        assert evidence["planned_outputs_state"]["state"] == "ok"


def test_stress_ethanol_cipro_study_status_surfaces_stale_construct_handoff() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        repo_root = Path.cwd()
        (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.0.0'\n", encoding="utf-8")
        (repo_root / "src" / "dnadesign").mkdir(parents=True, exist_ok=True)
        study_dir = repo_root / "docs" / "studies" / "stress_ethanol_cipro_growth"
        study_index = repo_root / "docs" / "studies" / "index.yaml"
        _write_study_index(study_index)
        _write_stress_ethanol_cipro_growth_preflight_record(study_dir, densegen_rows=5, anchor_rows=2, construct_rows=2)

        state, summary, evidence = _stress_ethanol_cipro_growth_status(None, repo_root=repo_root)

        assert state == "attention"
        assert "source rows visible densegen_demo_anchor 5 rows" in summary
        assert "handoff lag promoter/demo_anchor_set, promoter/demo_construct_contexts" in summary
        assert evidence["source_growth_state"]["state"] == "ok"
        assert evidence["handoff_readiness_state"]["state"] == "attention"
        refresh_states = {item["id"]: item for item in evidence["dataset_refresh_states"]}
        assert refresh_states["merged_anchor_from_densegen"]["state"] == "attention"
        assert refresh_states["construct_contexts_from_merged_anchor"]["state"] == "attention"
        assert evidence["stale_dataset_ids"] == [
            "promoter/demo_anchor_set",
            "promoter/demo_construct_contexts",
        ]


def test_stress_ethanol_cipro_study_preflight_reports_command_and_dataset_blockers(monkeypatch) -> None:
    with CliRunner().isolated_filesystem():
        repo_root = Path.cwd()
        (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.0.0'\n", encoding="utf-8")
        (repo_root / "src" / "dnadesign").mkdir(parents=True, exist_ok=True)
        study_dir = repo_root / "docs" / "studies" / "stress_ethanol_cipro_growth"
        study_index = repo_root / "docs" / "studies" / "index.yaml"
        _write_study_index(study_index)
        _write_stress_ethanol_cipro_growth_preflight_record(study_dir)

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
            if "notify profile doctor" in command:
                payload = {"ok": False, "error": "notify profile missing"}
                return CommandExecution(tuple(argv), str(cwd), 1, json.dumps(payload), "", False)
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
            if "session-counts" in command:
                return CommandExecution(
                    tuple(argv),
                    str(cwd),
                    0,
                    "queue_probe=ok running_jobs=1 queued_jobs=0 eqw_jobs=0",
                    "",
                    False,
                )
            if "ops runbook plan" in command and "densegen" in command:
                payload = {
                    "selected_mode": "resume",
                    "workflow_id": "densegen_batch_with_notify",
                    "orchestration_notify": {"secret_ref": "file:///tmp/webhook"},  # pragma: allowlist secret
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

        monkeypatch.setattr(STRESS_ETHANOL_CIPRO_GROWTH_RUN_PREFLIGHT_COMMAND_REF, _fake_run)
        monkeypatch.setattr(
            "dnadesign.studies.units.stress_ethanol_cipro_growth.operations.status.service.execute_runbook_plan",
            lambda *, runbook_path, repo_root: _fake_run(
                (
                    "uv",
                    "run",
                    "ops",
                    "runbook",
                    "plan",
                    "--runbook",
                    str(runbook_path),
                    "--repo-root",
                    str(repo_root),
                ),
                cwd=repo_root,
            ),
        )
        monkeypatch.setattr(
            "dnadesign.studies.units.stress_ethanol_cipro_growth.operations.status.service.inspect_local_infer_gpu_inventory",
            lambda: {"count": 0, "devices": [], "probe_error": None},
        )
        monkeypatch.setattr(
            "dnadesign.construct.resolve_construct_workspace_config_path_from_root",
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
        state, summary, evidence = _stress_ethanol_cipro_growth_preflight(None, repo_root=repo_root, scope="full")

        assert state == "attention"
        assert "stress_ethanol_cipro_growth: preflight phase infer_batch_preparation" in summary
        assert evidence["preferred_infer_model_family"] == "evo2_20b"
        assert evidence["notify_environment"] == {
            "NOTIFY_WEBHOOK": False,
            "NOTIFY_WEBHOOK_FILE": False,
            "SSL_CERT_FILE": False,
        }
        checks = {check["id"]: check for check in evidence["checks"]}
        assert checks["notify.environment.webhook"]["state"] == "attention"
        assert checks["notify.environment.tls"]["state"] == "attention"
        assert (
            checks["notify.environment.webhook"]["summary"]
            == "None of the accepted environment variables are configured: NOTIFY_WEBHOOK, NOTIFY_WEBHOOK_FILE."
        )
        assert (
            checks["notify.environment.tls"]["summary"]
            == "Required environment variable is not configured: SSL_CERT_FILE."
        )
        assert checks["densegen.config.probe_solver"]["state"] == "attention"
        assert checks["densegen.batch.plan"]["state"] == "ok"
        assert checks["densegen.batch.plan"]["summary"] == "DenseGen batch runbook renders cleanly."
        assert checks["construct.workspace.doctor"]["state"] == "ok"
        assert checks["construct.runtime.slot_a_window"]["state"] == "ok"
        assert checks["infer.validate.anchor_only_7b"]["state"] == "ok"
        assert checks["infer.validate.anchor_only_7b"]["summary"] == "Infer config validation completed."
        assert checks["infer.local_gpu.visibility"]["state"] == "attention"
        assert checks["infer.local_gpu.visibility"]["kind"] == "gpu_availability"
        assert checks["notify.profile.anchor_only_7b"]["state"] == "attention"
        assert checks["notify.profile.anchor_only_20b"]["state"] == "attention"
        assert checks["notify.profile.anchor_only_20b"]["surface_id"] == "notify_profile_doctor_anchor_only_20b"
        assert checks["notify.profile.anchor_only_20b"]["summary"] == "notify profile missing"
        assert checks["notify.profile.anchor_only_20b"]["command"].endswith(
            "workspace/infer/outputs/notify/infer/anchor_only_20b/profile.json --json"
        )
        assert "notify profile doctor" in checks["notify.profile.anchor_only_7b"]["command"]
        assert checks["infer.dry_run.anchor_only_7b"]["state"] == "ok"
        assert checks["infer.dry_run.anchor_only_20b"]["state"] == "ok"
        assert checks["infer.dry_run.anchor_only_7b"]["summary"] == "Infer dry-run completed."
        assert checks["notify.resolve_events.anchor_only_7b"]["state"] == "ok"
        assert checks["notify.resolve_events.anchor_only_20b"]["state"] == "ok"
        assert checks["notify.resolve_events.anchor_only_7b"]["summary"] == "Notify events path resolved."
        assert checks["infer.batch.7b.anchor_only.plan"]["state"] == "attention"
        assert (
            checks["infer.batch.7b.anchor_only.plan"]["summary"]
            == "notify webhook secret file is required for batch notify workflows"
        )
        assert evidence["counts"]["ok"] >= 1
        assert evidence["counts"]["attention"] >= 1
        assert evidence["counts"]["missing"] == 0


def test_stress_ethanol_cipro_study_preflight_reports_sequence_view_contract_health(monkeypatch) -> None:
    with CliRunner().isolated_filesystem():
        repo_root = Path.cwd()
        (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.0.0'\n", encoding="utf-8")
        (repo_root / "src" / "dnadesign").mkdir(parents=True, exist_ok=True)
        study_dir = repo_root / "docs" / "studies" / "stress_ethanol_cipro_growth"
        _write_study_index(repo_root / "docs" / "studies" / "index.yaml")
        _write_stress_ethanol_cipro_growth_preflight_record(study_dir, anchor_rows=2, construct_rows=4)
        usr_root = repo_root / "usr_root"
        _write_sequence_view_sidecar(
            usr_root,
            "promoter/demo_anchor_set",
            product_kind="construct_insert",
            context_kind="anchor_only",
            recommended_pooling="seq_mean",
            orientations=("forward", "forward"),
        )
        _write_sequence_view_sidecar(
            usr_root,
            "promoter/demo_construct_contexts",
            product_kind="realized_context",
            context_kind="template_1kb",
            recommended_pooling="anchor_mean",
            orientations=("forward", "forward", "reverse_complement", "reverse_complement"),
        )
        ops_path = study_dir / "operations" / "ops.study.yaml"
        ops_payload = yaml.safe_load(ops_path.read_text(encoding="utf-8"))
        ops_payload["preflight"]["checks"]["infer_batch_preparation"].insert(
            1,
            {
                "kind": "sequence_view_contract",
                "check_id": "infer.sequence_views.context_contract",
                "check_group": "infer",
                "summary": "Construct context sequence views satisfy product and pooling contract.",
                "required": True,
                "artifact": "construct_context_dataset",
                "expected": {
                    "total_records": 4,
                    "total_views": 4,
                    "counts_by_product_kind": {"realized_context": 4},
                    "counts_by_orientation": {"forward": 2, "reverse_complement": 2},
                    "counts_by_context_kind": {"template_1kb": 4},
                    "counts_by_recommended_pooling": {"anchor_mean": 4},
                },
            },
        )
        ops_path.write_text(yaml.safe_dump(ops_payload, sort_keys=False), encoding="utf-8")

        monkeypatch.setattr(
            STRESS_ETHANOL_CIPRO_GROWTH_RUN_PREFLIGHT_COMMAND_REF,
            lambda argv, *, cwd, timeout_seconds=180: CommandExecution(tuple(argv), str(cwd), 0, "ok", "", False),
        )
        monkeypatch.setattr(
            "dnadesign.studies.units.stress_ethanol_cipro_growth.operations.status.service.execute_runbook_plan",
            lambda *, runbook_path, repo_root: CommandExecution((), str(repo_root), 0, "{}", "", False),
        )
        monkeypatch.setattr(
            "dnadesign.studies.units.stress_ethanol_cipro_growth.operations.status.service.inspect_local_infer_gpu_inventory",
            lambda: {"count": 1, "devices": [], "probe_error": None},
        )

        _state, _summary, evidence = _stress_ethanol_cipro_growth_preflight(None, repo_root=repo_root, scope="full")

        checks = {check["id"]: check for check in evidence["checks"]}
        check = checks["infer.sequence_views.context_contract"]
        assert check["kind"] == "sequence_view_contract"
        assert check["state"] == "ok"
        assert check["details"]["counts_by_product_kind"] == {"realized_context": 4}
        assert check["details"]["counts_by_orientation"] == {"forward": 2, "reverse_complement": 2}
        assert check["details"]["invalid_bounds"] == 0
        assert evidence["scope"] == "full"


def test_stress_ethanol_cipro_study_status_reports_sequence_view_summary_without_deep_infer_completion(
    monkeypatch,
) -> None:
    with CliRunner().isolated_filesystem():
        repo_root = Path.cwd()
        (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.0.0'\n", encoding="utf-8")
        (repo_root / "src" / "dnadesign").mkdir(parents=True, exist_ok=True)
        study_dir = repo_root / "docs" / "studies" / "stress_ethanol_cipro_growth"
        _write_study_index(repo_root / "docs" / "studies" / "index.yaml")
        _write_stress_ethanol_cipro_growth_preflight_record(study_dir, anchor_rows=2, construct_rows=4)
        datasets_path = study_dir / "record" / "datasets.yaml"
        datasets_payload = yaml.safe_load(datasets_path.read_text(encoding="utf-8"))
        for entry in datasets_payload["datasets"]:
            if entry["dataset"] in {
                "promoter/demo_anchor_set",
                "promoter/demo_construct_contexts",
            }:
                entry["status"] = "present"
        datasets_path.write_text(yaml.safe_dump(datasets_payload, sort_keys=False), encoding="utf-8")

        usr_root = repo_root / "usr_root"
        _write_sequence_view_sidecar(
            usr_root,
            "promoter/demo_construct_contexts",
            product_kind="realized_context",
            context_kind="template_1kb",
            recommended_pooling="anchor_mean",
            orientations=("forward", "forward", "reverse_complement", "reverse_complement"),
        )
        infer_config = repo_root / "workspace" / "infer" / "config.sequence_views.evo2_7b.yaml"
        infer_config.write_text(
            yaml.safe_dump(
                {
                    "model": {"id": "evo2_7b", "device": "cuda:0", "precision": "bf16", "alphabet": "dna"},
                    "jobs": [],
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        ops_path = study_dir / "operations" / "ops.study.yaml"
        ops_payload = yaml.safe_load(ops_path.read_text(encoding="utf-8"))
        ops_payload["execution_surfaces"]["infer_completion_context_7b"] = {
            "surface_type": "command",
            "cwd_ref": "repo:workspace/infer",
            "argv": [
                "uv",
                "run",
                "infer",
                "validate",
                "sequence-view-completion",
                "--config",
                "config.sequence_views.evo2_7b.yaml",
                "--format",
                "json",
            ],
        }
        ops_payload["preflight"]["checks"]["infer_batch_preparation"].insert(
            1,
            {
                "kind": "sequence_view_contract",
                "check_id": "infer.sequence_views.context_contract",
                "check_group": "infer",
                "summary": "Construct context sequence views satisfy product and pooling contract.",
                "required": True,
                "artifact": "construct_context_dataset",
                "expected": {
                    "total_records": 4,
                    "total_views": 4,
                    "counts_by_product_kind": {"realized_context": 4},
                    "counts_by_orientation": {"forward": 2, "reverse_complement": 2},
                    "counts_by_context_kind": {"template_1kb": 4},
                    "counts_by_recommended_pooling": {"anchor_mean": 4},
                },
            },
        )
        ops_payload["preflight"]["checks"]["infer_batch_preparation"].insert(
            2,
            {
                "kind": "infer_sequence_view_completion",
                "check_id": "infer.feature_completion.context_7b",
                "check_group": "infer",
                "summary": "Context sequence-view feature completion is classified.",
                "required": False,
                "phase_id": "infer_anchor_plus_template_7b",
                "surface": "infer_completion_context_7b",
                "expected": {
                    "max_missing_vectors": 0,
                    "max_stale_vectors": 0,
                    "max_missing_products": 0,
                },
            },
        )
        ops_path.write_text(yaml.safe_dump(ops_payload, sort_keys=False), encoding="utf-8")

        def _fake_plan(config_path: Path, job: str | None = None):
            raise AssertionError(
                "stress-ethanol-cipro-growth-status must not scan Infer "
                f"feature-completion sidecars: {config_path} {job}"
            )

        monkeypatch.setattr(
            "dnadesign.infer.plan_sequence_view_feature_inventory_completion_from_config",
            _fake_plan,
            raising=False,
        )

        state, summary, evidence = _stress_ethanol_cipro_growth_status(None, repo_root=repo_root)

        assert state == "ok"
        assert "sequence-view product contracts 1/1 ok" in summary
        assert "infer sequence-view feature completion" not in summary
        assert evidence["sequence_view_contract_state"]["state"] == "ok"
        assert evidence["sequence_view_contract_state"]["checks"][0]["counts_by_orientation"] == {
            "forward": 2,
            "reverse_complement": 2,
        }
        assert evidence["infer_feature_completion_state"] is None


def test_stress_ethanol_cipro_study_preflight_reports_infer_sequence_view_completion(monkeypatch) -> None:
    with CliRunner().isolated_filesystem():
        repo_root = Path.cwd()
        (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.0.0'\n", encoding="utf-8")
        (repo_root / "src" / "dnadesign").mkdir(parents=True, exist_ok=True)
        study_dir = repo_root / "docs" / "studies" / "stress_ethanol_cipro_growth"
        _write_study_index(repo_root / "docs" / "studies" / "index.yaml")
        _write_stress_ethanol_cipro_growth_preflight_record(study_dir, anchor_rows=2, construct_rows=4)
        ops_path = study_dir / "operations" / "ops.study.yaml"
        ops_payload = yaml.safe_load(ops_path.read_text(encoding="utf-8"))
        ops_payload["execution_surfaces"]["infer_completion_anchor_7b"] = {
            "surface_type": "command",
            "argv": [
                "uv",
                "run",
                "infer",
                "validate",
                "sequence-view-completion",
                "--config",
                "workspace/infer/config.anchor_sequence_views.evo2_7b.yaml",
                "--format",
                "json",
            ],
        }
        ops_payload["preflight"]["checks"]["infer_batch_preparation"].insert(
            1,
            {
                "kind": "infer_sequence_view_completion",
                "check_id": "infer.feature_completion.anchor_7b",
                "check_group": "infer",
                "summary": "Anchor sequence-view feature completion is classified.",
                "required": False,
                "phase_id": "infer_anchor_only_7b",
                "surface": "infer_completion_anchor_7b",
                "expected": {
                    "max_missing_vectors": 0,
                    "max_stale_vectors": 0,
                    "max_missing_products": 0,
                },
            },
        )
        ops_path.write_text(yaml.safe_dump(ops_payload, sort_keys=False), encoding="utf-8")

        def _fake_run(argv, *, cwd, timeout_seconds=180):
            command = " ".join(argv)
            if "infer validate sequence-view-completion" in command:
                payload = [
                    {
                        "dataset": "promoter/demo_anchor_set",
                        "bundle_id": "anchor_sequence_views_7b",
                        "model_family": "evo2_7b",
                        "required_views": 2,
                        "required_vectors": 4,
                        "required_scalars": 4,
                        "existing_vectors": 2,
                        "existing_scalars": 0,
                        "reusable_vectors": 1,
                        "reusable_scalars": 0,
                        "stale_vectors": 1,
                        "stale_scalars": 0,
                        "missing_vectors": 2,
                        "missing_scalars": 4,
                        "missing_products": 0,
                        "persisted_vector_reusable": 0,
                        "persisted_scalar_reusable": 0,
                        "existing_aliases": 0,
                        "existing_scalar_aliases": 0,
                        "by_product_kind": {"construct_insert": 2},
                        "by_orientation": {"forward": 2},
                        "by_pooling_operation": {"seq_mean": 2},
                        "commands": {
                            "construct_completion": [],
                            "infer_backfill": ["uv run infer run --config config.yaml --job anchor_sequence_views_7b"],
                        },
                    }
                ]
                return CommandExecution(tuple(argv), str(cwd), 0, json.dumps(payload), "", False)
            return CommandExecution(tuple(argv), str(cwd), 0, "ok", "", False)

        monkeypatch.setattr(STRESS_ETHANOL_CIPRO_GROWTH_RUN_PREFLIGHT_COMMAND_REF, _fake_run)
        monkeypatch.setattr(
            "dnadesign.studies.units.stress_ethanol_cipro_growth.operations.status.service.execute_runbook_plan",
            lambda *, runbook_path, repo_root: CommandExecution((), str(repo_root), 0, "{}", "", False),
        )
        monkeypatch.setattr(
            "dnadesign.studies.units.stress_ethanol_cipro_growth.operations.status.service.inspect_local_infer_gpu_inventory",
            lambda: {"count": 1, "devices": [], "probe_error": None},
        )

        _state, _summary, evidence = _stress_ethanol_cipro_growth_preflight(None, repo_root=repo_root, scope="full")

        checks = {check["id"]: check for check in evidence["checks"]}
        check = checks["infer.feature_completion.anchor_7b"]
        assert check["kind"] == "infer_sequence_view_completion"
        assert check["state"] == "attention"
        assert check["summary"] == (
            "Anchor sequence-view feature completion is classified. reusable_vectors=1 stale_vectors=1 "
            "missing_vectors=2 reusable_scalars=0 stale_scalars=0 missing_scalars=4 missing_products=0."
        )
        assert check["details"]["required_views"] == 2
        assert check["details"]["required_vectors"] == 4
        assert check["details"]["required_scalars"] == 4
        assert check["details"]["reusable_vectors"] == 1
        assert check["details"]["reusable_scalars"] == 0
        assert check["details"]["stale_vectors"] == 1
        assert check["details"]["stale_scalars"] == 0
        assert check["details"]["missing_vectors"] == 2
        assert check["details"]["missing_scalars"] == 4
        assert check["details"]["missing_products"] == 0
        assert check["details"]["counts_by_product_kind"] == {"construct_insert": 2}
        assert check["details"]["counts_by_orientation"] == {"forward": 2}
        assert check["details"]["counts_by_pooling_operation"] == {"seq_mean": 2}
        assert check["details"]["commands"]["infer_backfill"] == [
            "uv run infer run --config config.yaml --job anchor_sequence_views_7b"
        ]


def test_stress_ethanol_cipro_study_preflight_blocks_stale_construct_inputs_in_next_scope(monkeypatch) -> None:
    with CliRunner().isolated_filesystem():
        repo_root = Path.cwd()
        (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.0.0'\n", encoding="utf-8")
        (repo_root / "src" / "dnadesign").mkdir(parents=True, exist_ok=True)
        study_dir = repo_root / "docs" / "studies" / "stress_ethanol_cipro_growth"
        study_index = repo_root / "docs" / "studies" / "index.yaml"
        _write_study_index(study_index)
        _write_stress_ethanol_cipro_growth_preflight_record(study_dir, densegen_rows=5, anchor_rows=2, construct_rows=2)

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
                return CommandExecution(tuple(argv), str(cwd), 0, "✔ Config validated (dry run).", "", False)
            if "notify profile doctor" in command:
                return CommandExecution(tuple(argv), str(cwd), 0, json.dumps({"ok": True}), "", False)
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
            if "session-counts" in command:
                return CommandExecution(
                    tuple(argv),
                    str(cwd),
                    0,
                    "queue_probe=ok running_jobs=0 queued_jobs=0 eqw_jobs=0",
                    "",
                    False,
                )
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

        monkeypatch.setenv("NOTIFY_WEBHOOK", "https://example.invalid/webhook")
        monkeypatch.setenv("SSL_CERT_FILE", "/tmp/cert.pem")
        monkeypatch.setattr(STRESS_ETHANOL_CIPRO_GROWTH_RUN_PREFLIGHT_COMMAND_REF, _fake_run)
        monkeypatch.setattr(
            "dnadesign.studies.units.stress_ethanol_cipro_growth.operations.status.service.execute_runbook_plan",
            lambda *, runbook_path, repo_root: _fake_run(
                (
                    "uv",
                    "run",
                    "ops",
                    "runbook",
                    "plan",
                    "--runbook",
                    str(runbook_path),
                    "--repo-root",
                    str(repo_root),
                ),
                cwd=repo_root,
            ),
        )
        monkeypatch.setattr(
            "dnadesign.studies.units.stress_ethanol_cipro_growth.operations.status.service.inspect_local_infer_gpu_inventory",
            lambda: {"count": 1, "devices": [{"id": 0, "name": "GPU"}], "probe_error": None},
        )
        monkeypatch.setattr(
            "dnadesign.construct.resolve_construct_workspace_config_path_from_root",
            lambda **kwargs: repo_root / "workspace" / "construct" / "config.slot_a.yaml",
        )
        monkeypatch.setattr(
            "dnadesign.construct.preflight_from_config",
            lambda config_path: SimpleNamespace(
                spec_id="spec-demo",
                records_total=2,
                existing_output_collisions=0,
                output_on_conflict="ignore",
            ),
        )

        state, summary, evidence = _stress_ethanol_cipro_growth_preflight(None, repo_root=repo_root, scope="next")

        assert state == "attention"
        assert "blocked by:" in summary
        checks = {check["id"]: check for check in evidence["checks"]}
        assert checks["infer.input.merged_anchor_from_densegen"]["state"] == "attention"
        assert "lag=3" in checks["infer.input.merged_anchor_from_densegen"]["summary"]
        assert checks["infer.input.construct_contexts_from_merged_anchor"]["state"] == "attention"
        assert "infer.input.merged_anchor_from_densegen" in evidence["blocked_by"]


def test_stress_ethanol_cipro_study_preflight_demotes_construct_runtime_attention_once_materialized(
    monkeypatch,
) -> None:
    with CliRunner().isolated_filesystem():
        repo_root = Path.cwd()
        (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.0.0'\n", encoding="utf-8")
        (repo_root / "src" / "dnadesign").mkdir(parents=True, exist_ok=True)
        study_dir = repo_root / "docs" / "studies" / "stress_ethanol_cipro_growth"
        study_index = repo_root / "docs" / "studies" / "index.yaml"
        _write_study_index(study_index)
        _write_stress_ethanol_cipro_growth_preflight_record(study_dir)

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
            if "notify profile doctor" in command:
                payload = {"ok": False, "error": "notify profile missing"}
                return CommandExecution(tuple(argv), str(cwd), 1, json.dumps(payload), "", False)
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
            if "session-counts" in command:
                return CommandExecution(
                    tuple(argv),
                    str(cwd),
                    0,
                    "queue_probe=ok running_jobs=1 queued_jobs=0 eqw_jobs=0",
                    "",
                    False,
                )
            if "ops runbook plan" in command and "densegen" in command:
                payload = {
                    "selected_mode": "resume",
                    "workflow_id": "densegen_batch_with_notify",
                    "orchestration_notify": {"secret_ref": "file:///tmp/webhook"},  # pragma: allowlist secret
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

        monkeypatch.setattr(STRESS_ETHANOL_CIPRO_GROWTH_RUN_PREFLIGHT_COMMAND_REF, _fake_run)
        monkeypatch.setattr(
            "dnadesign.studies.units.stress_ethanol_cipro_growth.operations.status.service.execute_runbook_plan",
            lambda *, runbook_path, repo_root: _fake_run(
                (
                    "uv",
                    "run",
                    "ops",
                    "runbook",
                    "plan",
                    "--runbook",
                    str(runbook_path),
                    "--repo-root",
                    str(repo_root),
                ),
                cwd=repo_root,
            ),
        )
        monkeypatch.setattr(
            "dnadesign.studies.units.stress_ethanol_cipro_growth.operations.status.service.inspect_local_infer_gpu_inventory",
            lambda: {"count": 1, "devices": [{"id": 0, "name": "GPU"}], "probe_error": None},
        )
        state, summary, evidence = _stress_ethanol_cipro_growth_preflight(None, repo_root=repo_root, scope="full")

        assert state == "attention"
        checks = {check["id"]: check for check in evidence["checks"]}
        assert checks["construct.runtime.slot_a_window"]["state"] == "attention"
        assert "planned output id(s) already exist" in (checks["construct.runtime.slot_a_window"]["summary"])
        assert "construct.runtime.slot_a_window" not in evidence["blocked_by"]
        assert "construct.runtime.slot_a_window" in evidence["nonblocking_attention_ids"]


def test_stress_ethanol_cipro_study_preflight_scope_next_defers_later_lane_blockers(monkeypatch) -> None:
    with CliRunner().isolated_filesystem():
        repo_root = Path.cwd()
        (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.0.0'\n", encoding="utf-8")
        (repo_root / "src" / "dnadesign").mkdir(parents=True, exist_ok=True)
        study_dir = repo_root / "docs" / "studies" / "stress_ethanol_cipro_growth"
        study_index = repo_root / "docs" / "studies" / "index.yaml"
        _write_study_index(study_index)
        _write_stress_ethanol_cipro_growth_preflight_record(study_dir)

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
            if "notify profile doctor" in command:
                payload = {"ok": False, "error": "notify profile missing"}
                return CommandExecution(tuple(argv), str(cwd), 1, json.dumps(payload), "", False)
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
            if "session-counts" in command:
                return CommandExecution(
                    tuple(argv),
                    str(cwd),
                    0,
                    "queue_probe=ok running_jobs=1 queued_jobs=0 eqw_jobs=0",
                    "",
                    False,
                )
            if "ops runbook plan" in command and "densegen" in command:
                payload = {
                    "selected_mode": "resume",
                    "workflow_id": "densegen_batch_with_notify",
                    "orchestration_notify": {"secret_ref": "file:///tmp/webhook"},  # pragma: allowlist secret
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

        monkeypatch.setattr(STRESS_ETHANOL_CIPRO_GROWTH_RUN_PREFLIGHT_COMMAND_REF, _fake_run)
        monkeypatch.setattr(
            "dnadesign.studies.units.stress_ethanol_cipro_growth.operations.status.service.execute_runbook_plan",
            lambda *, runbook_path, repo_root: _fake_run(
                (
                    "uv",
                    "run",
                    "ops",
                    "runbook",
                    "plan",
                    "--runbook",
                    str(runbook_path),
                    "--repo-root",
                    str(repo_root),
                ),
                cwd=repo_root,
            ),
        )
        monkeypatch.setattr(
            "dnadesign.studies.units.stress_ethanol_cipro_growth.operations.status.service.inspect_local_infer_gpu_inventory",
            lambda: {"count": 0, "devices": [], "probe_error": None},
        )
        monkeypatch.setattr(
            "dnadesign.construct.resolve_construct_workspace_config_path_from_root",
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
        state, summary, evidence = _stress_ethanol_cipro_growth_preflight(None, repo_root=repo_root, scope="next")

        assert state == "attention"
        assert "focus phase infer_batch_preparation" in summary
        assert "blocked by:" in summary
        assert evidence["scope"] == "next"
        assert evidence["target_phase"] == "infer_batch_preparation"
        assert evidence["blocked_by"][:3] == [
            "notify.environment.tls",
            "notify.environment.webhook",
            "infer.batch.20b.anchor_only.plan",
        ]
        assert "infer.local_gpu.visibility" in evidence["nonblocking_attention_ids"]
        assert "notify.environment.webhook" not in evidence["nonblocking_attention_ids"]
        assert "infer.batch.20b.anchor_only.plan" not in evidence["nonblocking_attention_ids"]
        assert evidence["deferred_check_ids"] == []


def test_stress_ethanol_cipro_study_preflight_lane_scope_keeps_notify_env_and_selected_lane(monkeypatch) -> None:
    with CliRunner().isolated_filesystem():
        repo_root = Path.cwd()
        (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.0.0'\n", encoding="utf-8")
        (repo_root / "src" / "dnadesign").mkdir(parents=True, exist_ok=True)
        study_dir = repo_root / "docs" / "studies" / "stress_ethanol_cipro_growth"
        study_index = repo_root / "docs" / "studies" / "index.yaml"
        _write_study_index(study_index)
        _write_stress_ethanol_cipro_growth_preflight_record(study_dir)

        contract_path = study_dir / "operations" / "ops.study.yaml"
        contract_payload = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        contract_payload["lifecycle"]["current_phase"]["id"] = "infer_anchor_only_20b"
        for phase in contract_payload["phases"]:
            if phase["id"] == "infer_batch_preparation":
                phase["status"] = "complete"
            elif phase["id"] == "infer_anchor_only_20b":
                phase["status"] = "in_progress"
        contract_path.write_text(yaml.safe_dump(contract_payload, sort_keys=False), encoding="utf-8")

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
            if "notify profile doctor" in command:
                payload = {"ok": False, "error": "notify profile missing"}
                return CommandExecution(tuple(argv), str(cwd), 1, json.dumps(payload), "", False)
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
            if "session-counts" in command:
                return CommandExecution(
                    tuple(argv),
                    str(cwd),
                    0,
                    "queue_probe=ok running_jobs=1 queued_jobs=0 eqw_jobs=0",
                    "",
                    False,
                )
            if "ops runbook plan" in command and "densegen" in command:
                payload = {
                    "selected_mode": "resume",
                    "workflow_id": "densegen_batch_with_notify",
                    "orchestration_notify": {"secret_ref": "file:///tmp/webhook"},  # pragma: allowlist secret
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

        monkeypatch.setattr(STRESS_ETHANOL_CIPRO_GROWTH_RUN_PREFLIGHT_COMMAND_REF, _fake_run)
        monkeypatch.setattr(
            "dnadesign.studies.units.stress_ethanol_cipro_growth.operations.status.service.execute_runbook_plan",
            lambda *, runbook_path, repo_root: _fake_run(
                (
                    "uv",
                    "run",
                    "ops",
                    "runbook",
                    "plan",
                    "--runbook",
                    str(runbook_path),
                    "--repo-root",
                    str(repo_root),
                ),
                cwd=repo_root,
            ),
        )
        monkeypatch.setattr(
            "dnadesign.studies.units.stress_ethanol_cipro_growth.operations.status.service.inspect_local_infer_gpu_inventory",
            lambda: {"count": 0, "devices": [], "probe_error": None},
        )
        monkeypatch.setattr(
            "dnadesign.construct.resolve_construct_workspace_config_path_from_root",
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
        state, summary, evidence = _stress_ethanol_cipro_growth_preflight(None, repo_root=repo_root, scope="next")

        assert state == "attention"
        assert "focus phase infer_anchor_only_20b" in summary
        assert evidence["target_phase"] == "infer_anchor_only_20b"
        assert "blocked by:" in summary
        assert "infer.local_gpu.visibility" not in evidence["blocked_by"]
        assert "infer.batch.7b.anchor_only.plan" not in evidence["blocked_by"]
        assert "notify.profile.anchor_plus_template_20b" not in evidence["blocked_by"]
        assert evidence["blocked_by"][:4] == [
            "notify.environment.tls",
            "notify.environment.webhook",
            "infer.batch.20b.anchor_only.plan",
            "notify.profile.anchor_only_20b",
        ]
        assert "notify.environment.webhook" not in evidence["nonblocking_attention_ids"]
        assert "notify.environment.tls" not in evidence["nonblocking_attention_ids"]
        assert "infer.batch.20b.anchor_only.plan" not in evidence["nonblocking_attention_ids"]
        assert "notify.profile.anchor_only_20b" not in evidence["nonblocking_attention_ids"]


def test_stress_ethanol_cipro_study_preflight_full_scope_demotes_completed_infer_lane_attention(monkeypatch) -> None:
    with CliRunner().isolated_filesystem():
        repo_root = Path.cwd()
        (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.0.0'\n", encoding="utf-8")
        (repo_root / "src" / "dnadesign").mkdir(parents=True, exist_ok=True)
        study_dir = repo_root / "docs" / "studies" / "stress_ethanol_cipro_growth"
        study_index = repo_root / "docs" / "studies" / "index.yaml"
        _write_study_index(study_index)
        _write_stress_ethanol_cipro_growth_preflight_record(study_dir)

        contract_path = study_dir / "operations" / "ops.study.yaml"
        contract_payload = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
        for phase in contract_payload["phases"]:
            if phase["id"] == "infer_anchor_only_20b":
                phase["status"] = "complete"
        contract_path.write_text(yaml.safe_dump(contract_payload, sort_keys=False), encoding="utf-8")

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
            if "notify profile doctor" in command:
                payload = {"ok": False, "error": "notify profile missing"}
                return CommandExecution(tuple(argv), str(cwd), 1, json.dumps(payload), "", False)
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
            if "session-counts" in command:
                return CommandExecution(
                    tuple(argv),
                    str(cwd),
                    0,
                    "queue_probe=ok running_jobs=1 queued_jobs=0 eqw_jobs=0",
                    "",
                    False,
                )
            if "ops runbook plan" in command and "densegen" in command:
                payload = {
                    "selected_mode": "resume",
                    "workflow_id": "densegen_batch_with_notify",
                    "orchestration_notify": {"secret_ref": "file:///tmp/webhook"},  # pragma: allowlist secret
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

        monkeypatch.setattr(STRESS_ETHANOL_CIPRO_GROWTH_RUN_PREFLIGHT_COMMAND_REF, _fake_run)
        monkeypatch.setattr(
            "dnadesign.studies.units.stress_ethanol_cipro_growth.operations.status.service.execute_runbook_plan",
            lambda *, runbook_path, repo_root: _fake_run(
                (
                    "uv",
                    "run",
                    "ops",
                    "runbook",
                    "plan",
                    "--runbook",
                    str(runbook_path),
                    "--repo-root",
                    str(repo_root),
                ),
                cwd=repo_root,
            ),
        )
        monkeypatch.setattr(
            "dnadesign.studies.units.stress_ethanol_cipro_growth.operations.status.service.inspect_local_infer_gpu_inventory",
            lambda: {"count": 0, "devices": [], "probe_error": None},
        )
        monkeypatch.setattr(
            "dnadesign.construct.resolve_construct_workspace_config_path_from_root",
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
        state, _summary, evidence = _stress_ethanol_cipro_growth_preflight(None, repo_root=repo_root, scope="full")

        assert state == "attention"
        assert "infer.local_gpu.visibility" not in evidence["blocked_by"]
        assert "notify.profile.anchor_only_20b" not in evidence["blocked_by"]
        assert "infer.batch.20b.anchor_only.plan" not in evidence["blocked_by"]
        assert "infer.local_gpu.visibility" in evidence["nonblocking_attention_ids"]
        assert "notify.profile.anchor_only_20b" in evidence["nonblocking_attention_ids"]
        assert "infer.batch.20b.anchor_only.plan" in evidence["nonblocking_attention_ids"]
        assert "notify.environment.webhook" in evidence["blocked_by"]
        assert "infer.batch.20b.anchor_plus_template.plan" in evidence["blocked_by"]


def test_stress_ethanol_cipro_study_preflight_full_scope_demotes_parallel_optional_densegen_attention(
    monkeypatch,
) -> None:
    with CliRunner().isolated_filesystem():
        repo_root = Path.cwd()
        (repo_root / "pyproject.toml").write_text("[project]\nname='demo'\nversion='0.0.0'\n", encoding="utf-8")
        (repo_root / "src" / "dnadesign").mkdir(parents=True, exist_ok=True)
        study_dir = repo_root / "docs" / "studies" / "stress_ethanol_cipro_growth"
        study_index = repo_root / "docs" / "studies" / "index.yaml"
        _write_study_index(study_index)
        _write_stress_ethanol_cipro_growth_preflight_record(study_dir)

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
            if "notify profile doctor" in command:
                payload = {"ok": False, "error": "notify profile missing"}
                return CommandExecution(tuple(argv), str(cwd), 1, json.dumps(payload), "", False)
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
            if "session-counts" in command:
                return CommandExecution(
                    tuple(argv),
                    str(cwd),
                    0,
                    "queue_probe=ok running_jobs=1 queued_jobs=0 eqw_jobs=0",
                    "",
                    False,
                )
            if "ops runbook plan" in command and "densegen" in command:
                payload = {
                    "selected_mode": "resume",
                    "workflow_id": "densegen_batch_with_notify",
                    "orchestration_notify": {"secret_ref": "file:///tmp/webhook"},  # pragma: allowlist secret
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

        monkeypatch.setattr(STRESS_ETHANOL_CIPRO_GROWTH_RUN_PREFLIGHT_COMMAND_REF, _fake_run)
        monkeypatch.setattr(
            "dnadesign.studies.units.stress_ethanol_cipro_growth.operations.status.service.execute_runbook_plan",
            lambda *, runbook_path, repo_root: _fake_run(
                (
                    "uv",
                    "run",
                    "ops",
                    "runbook",
                    "plan",
                    "--runbook",
                    str(runbook_path),
                    "--repo-root",
                    str(repo_root),
                ),
                cwd=repo_root,
            ),
        )
        monkeypatch.setattr(
            "dnadesign.studies.units.stress_ethanol_cipro_growth.operations.status.service.inspect_local_infer_gpu_inventory",
            lambda: {"count": 0, "devices": [], "probe_error": None},
        )
        monkeypatch.setattr(
            "dnadesign.construct.resolve_construct_workspace_config_path_from_root",
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
        state, summary, evidence = _stress_ethanol_cipro_growth_preflight(None, repo_root=repo_root, scope="full")

        assert state == "attention"
        assert "densegen.config.probe_solver" not in evidence["blocked_by"]
        assert "densegen.config.probe_solver" in evidence["nonblocking_attention_ids"]
        assert "infer.local_gpu.visibility" in evidence["nonblocking_attention_ids"]
        assert "notify.environment.webhook" in evidence["blocked_by"]


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
        assert payload["status_kind"] == "cluster-run-index"
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
        assert payload["status_kind"] == "opal-campaign-state"
        assert payload["state"] == "ok"
        assert payload["evidence"]["num_rounds"] == 1
        assert payload["evidence"]["latest_round"]["run_id"] == "run_001"


def test_cli_progress_show_rejects_incompatible_opal_state_schema() -> None:
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
        state_path = config_path.parent / "opal_campaign" / "state.json"
        state_path.write_text(
            json.dumps(
                {
                    "version": 1,
                    "campaign_slug": "demo_campaign",
                    "campaign_name": "Demo campaign",
                    "x_column_name": "infer__demo",
                    "y_column_name": "measured_activity",
                    "rounds": [{"round_index": 1, "run_id": "run_001"}],
                },
                indent=2,
                sort_keys=True,
            ),
            encoding="utf-8",
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
        assert payload["status_kind"] == "opal-campaign-state"
        assert payload["state"] == "attention"
        assert payload["summary"] == "OPAL state.json is not loadable"
        assert "state.json version must be 2" in payload["evidence"]["state_load_error"]


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


def test_cli_progress_show_reports_missing_opal_candidate_records_before_state_json() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        config_path = Path("demo_campaign") / "configs" / "campaign.yaml"
        _write_opal_usr_campaign_without_records(config_path)

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
        assert payload["status_kind"] == "opal-campaign-state"
        assert payload["state"] == "missing"
        assert payload["summary"] == "OPAL candidate records.parquet not found"
        assert payload["evidence"]["records_path"].endswith("usr/datasets/demo_candidates/records.parquet")
        assert payload["evidence"]["dataset"] == "demo_candidates"


def test_cli_progress_show_reports_ready_opal_candidate_records_before_state_json() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        config_path = Path("demo_campaign") / "configs" / "campaign.yaml"
        records_path = _write_opal_usr_campaign_with_records(config_path)

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
        assert payload["status_kind"] == "opal-campaign-state"
        assert payload["state"] == "missing"
        assert payload["summary"] == "OPAL state.json not found; candidate records.parquet exists"
        assert payload["evidence"]["records_path"] == str(records_path.resolve())
        assert payload["evidence"]["records_present"] is True
        assert payload["evidence"]["records_row_count"] == 2
        assert payload["evidence"]["dataset"] == "demo_candidates"


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
    assert payload["status_kind"] == "ops-audit-json"
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


def test_cli_progress_campaign_rejects_scaffold_placeholder_paths() -> None:
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

        assert result.exit_code == 2
        assert "placeholder path text" in result.output
        assert "<opal-workdir>/configs/campaign.yaml" in result.output


def test_cli_progress_campaign_rejects_narrative_placeholder_paths() -> None:
    runner = CliRunner()
    with runner.isolated_filesystem():
        manifest_path = Path("manifests") / "campaign_status.yaml"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            yaml.safe_dump(
                {
                    "version": 2,
                    "path_base": "repo",
                    "campaign_id": "placeholder_manifest",
                    "steps": [
                        {
                            "label": "clustering",
                            "registry_id": "cluster.downstream.exploratory-clustering",
                            "inputs": {"cluster_results_root": "repo:n/a"},
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

        assert result.exit_code == 2
        assert "placeholder path text" in result.output
        assert "repo:n/a" in result.output


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
    assert payload["campaign_id"] == "status_campaign"
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
    assert payload["steps"][0]["status_kind"] == "opal-campaign-state"
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
    assert "Progress contract error: unknown registry id:" in result.output
    assert "usr.data-plane.promoter-feature" in result.output
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
    assert "Progress contract error: unknown registry id:" in result.output
    assert "usr.data-plane.promoter-feature" in result.output
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
    assert "campaign scaffold requires at least one registry id" in result.output
    assert "--related-to" in result.output
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
    assert "Progress contract error: unknown registry id:" in result.output
    assert "usr.data-plane.promoter-feature" in result.output
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
    assert "status kind 'ops-audit-json' requires --audit-json" in result.output
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
    assert "status kind 'opal-campaign-state' requires" in result.output
    assert "--opal-config" in result.output
    assert "--opal-workdir" in result.output
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
    assert payload["status_kind"] == "opal-campaign-state"
    assert payload["observes_plane"] == "control"
    assert payload["surface_type"] == "campaign_snapshot"
    assert payload["cost_class"] == "cheap"
    assert payload["summary_scope"] == "workspace"
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


def test_cli_progress_explain_text_reports_status_ontology() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "progress",
            "explain",
            "opal.downstream.usr-infer-x-active-learning",
            "--repo-root",
            str(_repo_root()),
        ],
    )

    assert result.exit_code == 0
    assert "Observes plane: control" in result.output
    assert "Surface type: campaign_snapshot" in result.output
    assert "Cost class: cheap" in result.output
    assert "Summary scope: workspace" in result.output


def test_cli_status_kinds_reports_provider_owned_inventory() -> None:
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
    kinds = {entry["status_kind"]: entry for entry in payload["status_kinds"]}
    assert kinds["stress-ethanol-cipro-growth-preflight"]["observes_plane"] == "execution_readiness"
    assert kinds["stress-ethanol-cipro-growth-preflight"]["surface_type"] == "study_preflight"
    assert kinds["stress-ethanol-cipro-growth-preflight"]["cost_class"] == "deep"
    assert kinds["stress-ethanol-cipro-growth-preflight"]["summary_scope"] == "host"
    assert kinds["stress-ethanol-cipro-growth-preflight"]["provider_id"] == "study.stress_ethanol_cipro_growth"
    assert kinds["stress-ethanol-cipro-growth-preflight"]["optional_inputs"] == [
        {
            "cli_flag": "--study-dir",
            "summary": (
                "Checked-in stress_ethanol_cipro_growth study directory containing record/campaign.yaml, "
                "record/datasets.yaml, record/status.md, and operations/ops.study.yaml."
            ),
        },
        {
            "cli_flag": "--scope",
            "summary": (
                "Preflight scope for stress-ethanol-cipro-growth-preflight surfaces: "
                "`full` runs the whole suite; `next` focuses the next actionable "
                "phase and defers later-lane blockers."
            ),
        },
        {
            "cli_flag": "--command-timeout-seconds",
            "summary": "Per-command timeout for command-backed preflight checks on this host.",
        },
    ]
    assert kinds["retron-hairpin-design-preflight"]["observes_plane"] == "execution_readiness"
    assert kinds["retron-hairpin-design-preflight"]["surface_type"] == "study_preflight"
    assert kinds["retron-hairpin-design-preflight"]["cost_class"] == "deep"
    assert kinds["retron-hairpin-design-preflight"]["summary_scope"] == "host"
    assert kinds["retron-hairpin-design-preflight"]["provider_id"] == "study.retron_hairpin_design"
    assert kinds["retron-hairpin-design-preflight"]["optional_inputs"] == [
        {
            "cli_flag": "--study-dir",
            "summary": (
                "Checked-in retron_hairpin_design record directory containing record/campaign.yaml, "
                "record/datasets.yaml, record/status.md, operations/ops.study.yaml, routes/README.md, "
                "and operations/runtime/command-groups/pipeline.yaml."
            ),
        },
        {
            "cli_flag": "--scope",
            "summary": (
                "Preflight scope for retron-hairpin-design-preflight surfaces: "
                "`full` runs the whole suite; `next` focuses the current actionable route, track, "
                "or phase declared by the study contract."
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
        "If `plan.runtime_visibility.active_job_resolution_state=unknown`, submit stays blocked by default "
        "unless the operator explicitly passes `--allow-unknown-active-jobs`.",
    ]


def test_cli_progress_show_help_includes_registry_specific_inputs_for_status_surface() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "progress",
            "show",
            "studies.stress-ethanol-cipro-growth.status",
            "--help",
        ],
    )

    assert result.exit_code == 0
    assert "--study-dir" in result.output
    assert "--scope" not in result.output


def test_cli_progress_show_help_includes_registry_specific_inputs_for_preflight_surface() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "progress",
            "show",
            "studies.stress-ethanol-cipro-growth.preflight",
            "--help",
        ],
    )

    assert result.exit_code == 0
    assert "--study-dir" in result.output
    assert "--scope" in result.output
    assert "--command-timeout-seconds" in result.output


def test_cli_progress_show_help_includes_registry_specific_inputs_for_retron_preflight_surface() -> None:
    runner = CliRunner()

    result = runner.invoke(
        app,
        [
            "progress",
            "show",
            "studies.retron-hairpin-design.preflight",
            "--help",
        ],
    )

    assert result.exit_code == 0
    assert "--study-dir" in result.output
    assert "--scope" in result.output
    assert "Preflight scope for retron-hairpin-design-" in result.output
    assert "preflight surfaces" in result.output


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
    assert "uv run ops progress scaffold --related-to" in result.output
    assert "<registry-id>" in result.output


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
        assert "must place provider inputs under 'inputs':" in result.output
        assert "audit_json" in result.output
