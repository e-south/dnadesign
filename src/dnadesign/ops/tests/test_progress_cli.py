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

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from typer.testing import CliRunner

from dnadesign.ops.cli import app


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    raise RuntimeError("repo root not found")


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
        assert payload["evidence"]["namespace_column_counts"]["infer"] == 1
        assert payload["evidence"]["overlay_namespaces"] == ["infer"]


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
                    "campaign_id": "demo_cross_tool_campaign",
                    "steps": [
                        {
                            "label": "orchestration",
                            "registry_id": "ops.control-plane.orchestration",
                            "audit_json": "../artifacts/latest.json",
                        },
                        {
                            "label": "feature-matrix",
                            "registry_id": "usr.data-plane.promoter-feature-matrix",
                            "usr_root": "../usr_root",
                            "dataset": "demo_dataset",
                        },
                        {
                            "label": "active-learning",
                            "registry_id": "opal.downstream.usr-infer-x-active-learning",
                            "opal_config": "../configs/campaign.yaml",
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
                    "campaign_id": "manifest_relative_paths",
                    "steps": [
                        {
                            "label": "orchestration",
                            "registry_id": "ops.control-plane.orchestration",
                            "audit_json": "../artifacts/latest.json",
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
                    "campaign_id": "placeholder_manifest",
                    "steps": [
                        {
                            "label": "active-learning",
                            "registry_id": "opal.downstream.usr-infer-x-active-learning",
                            "opal_config": "<opal-workdir>/configs/campaign.yaml",
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
    assert payload["steps"] == [
        {
            "label": "orchestration",
            "registry_id": "ops.control-plane.orchestration",
            "audit_json": "<workspace-root>/outputs/logs/ops/audit/latest.json",
        },
        {
            "label": "promoter-feature-matrix",
            "registry_id": "usr.data-plane.promoter-feature-matrix",
            "usr_root": "<usr-root>",
            "dataset": "<dataset>",
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
            "opal_config": "<opal-workdir>/configs/campaign.yaml",
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
    assert payload["manifest"]["steps"][0]["dataset"] == "<dataset>"


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
        assert payload["steps"][0]["audit_json"] == "<workspace-root>/outputs/logs/ops/audit/latest.json"

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
