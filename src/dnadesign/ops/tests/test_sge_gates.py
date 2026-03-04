"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/ops/tests/test_sge_gates.py

Unit tests for native Ops SGE preflight gate helpers.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

import pytest

import dnadesign.ops.orchestrator.gates as gates
from dnadesign.ops.orchestrator.gates import (
    SessionCounts,
    build_operator_brief,
    build_shape_advisor,
    main,
    parse_qstat_output,
    qa_submit_template,
)


def _write_densegen_config(
    path: Path,
    *,
    dataset: str = "densegen/study_stress_ethanol_cipro",
    run_root: str = ".",
    usr_root: str = "outputs/usr_datasets",
    chunk_size: int = 128,
    parquet_path: str = "outputs/tables/records.parquet",
    parquet_chunk_size: int = 2048,
    max_accepted_per_library: int = 2,
    total_sequences: int = 1_000_000,
) -> None:
    path.write_text(
        textwrap.dedent(
            f"""
            densegen:
              run:
                id: study_stress_ethanol_cipro
                root: {run_root}
              output:
                targets: [parquet, usr]
                schema:
                  bio_type: dna
                  alphabet: dna_4
                parquet:
                  path: {parquet_path}
                  deduplicate: true
                  chunk_size: {parquet_chunk_size}
                usr:
                  root: {usr_root}
                  dataset: {dataset}
                  chunk_size: {chunk_size}
              generation:
                plan:
                  - name: stress
                    sequences: {total_sequences}
              runtime:
                round_robin: true
                max_accepted_per_library: {max_accepted_per_library}
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def _write_infer_config(
    path: Path,
    *,
    dataset: str = "infer/demo",
    usr_root: str = "outputs/usr_datasets",
) -> None:
    path.write_text(
        textwrap.dedent(
            f"""
            model:
              id: evo2_7b
              device: cpu
              precision: fp32
              alphabet: dna
            jobs:
              - id: infer_extract
                operation: extract
                ingest:
                  source: usr
                  root: {usr_root}
                  dataset: {dataset}
                  field: sequence
                outputs:
                  - id: ll_mean
                    fn: evo2.log_likelihood
                    format: float
                io:
                  write_back: true
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def test_parse_qstat_output_counts_running_queued_and_eqw() -> None:
    qstat_text = """
job-ID prior name user state submit/start at queue slots ja-task-ID
--------------------------------------------------------------------------------
81001 0.555 a b r 03/01/2026 queueA 16
81002 0.555 a b qw 03/01/2026 queueA 1
81003 0.555 a b Eqw 03/01/2026 queueA 1
"""
    counts = parse_qstat_output(qstat_text)
    assert counts.running_jobs == 1
    assert counts.queued_jobs == 2
    assert counts.eqw_jobs == 1


def test_qa_submit_template_rejects_now_y_directive(tmp_path: Path) -> None:
    template = tmp_path / "bad.qsub"
    template.write_text("#!/bin/bash -l\n#$ -now y\n", encoding="utf-8")

    failures = qa_submit_template(template)
    assert failures
    assert any("disallowed_queue_bypass_directive" in failure for failure in failures)


def test_build_shape_advisor_prefers_hold_jid_for_ordered_pipeline() -> None:
    advisor = build_shape_advisor(
        counts=SessionCounts(running_jobs=0, queued_jobs=0, eqw_jobs=0),
        planned_submits=2,
        warn_over_running=3,
        requires_order=True,
    )
    assert advisor["advisor"] == "hold_jid"
    assert advisor["recommended_action"] == "dependency_chain"


def test_build_operator_brief_blocks_when_eqw_jobs_present() -> None:
    brief, exit_code = build_operator_brief(
        counts=SessionCounts(running_jobs=0, queued_jobs=0, eqw_jobs=1),
        planned_submits=2,
        warn_over_running=3,
    )
    assert brief["submit_gate"] == "blocked"
    assert brief["next_action"] == "triage_eqw"
    assert exit_code == 2


def test_qa_submit_template_accepts_valid_template(tmp_path: Path) -> None:
    template = tmp_path / "ok.qsub"
    template.write_text("#!/bin/bash -l\n#$ -N demo\n", encoding="utf-8")

    failures = qa_submit_template(template)
    assert failures == []


def test_main_qa_submit_preflight_passes_for_valid_template(tmp_path: Path, capsys) -> None:
    template = tmp_path / "ok.qsub"
    template.write_text("#!/bin/bash -l\n#$ -N demo\n", encoding="utf-8")

    exit_code = main(["qa-submit-preflight", "--template", str(template)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "qa_preflight=pass" in captured.out


def test_main_qa_submit_preflight_fails_for_missing_template(capsys) -> None:
    exit_code = main(["qa-submit-preflight", "--template", "/tmp/does-not-exist.qsub"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "template_missing=" in captured.err


def test_main_submit_shape_advisor_emits_record(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(
        gates,
        "_load_session_counts",
        lambda: SessionCounts(running_jobs=1, queued_jobs=2, eqw_jobs=0),
    )
    exit_code = main(["submit-shape-advisor", "--planned-submits", "2", "--warn-over-running", "3"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "advisor=array" in captured.out
    assert "queue_policy=respect_queue" in captured.out


def test_main_operator_brief_returns_non_zero_for_eqw(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(
        gates,
        "_load_session_counts",
        lambda: SessionCounts(running_jobs=0, queued_jobs=0, eqw_jobs=1),
    )
    exit_code = main(["operator-brief", "--planned-submits", "2", "--warn-over-running", "3"])
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "submit_gate=blocked" in captured.out


def test_main_session_counts_emits_record(monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    monkeypatch.setattr(
        gates,
        "_load_session_counts",
        lambda: SessionCounts(running_jobs=2, queued_jobs=1, eqw_jobs=0),
    )
    exit_code = main(["session-counts"])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "running_jobs=2" in captured.out
    assert "queued_jobs=1" in captured.out
    assert "eqw_jobs=0" in captured.out


def test_main_ensure_dir_writable_creates_directory_and_emits_record(tmp_path: Path, capsys) -> None:
    target = tmp_path / "workspace" / "outputs" / "logs" / "ops" / "runtime"
    exit_code = main(["ensure-dir-writable", "--path", str(target)])
    captured = capsys.readouterr()

    assert exit_code == 0
    assert target.exists()
    assert target.is_dir()
    assert f"path={target.resolve()}" in captured.out


def test_main_prune_ops_logs_prunes_old_files_and_writes_manifest(tmp_path: Path, capsys) -> None:
    stdout_dir = tmp_path / "workspace" / "outputs" / "logs" / "ops" / "sge" / "study_stress_ethanol_cipro"
    stdout_dir.mkdir(parents=True, exist_ok=True)
    keep_file = stdout_dir / "dnadesign_densegen_cpu.100.out"
    prune_file = stdout_dir / "dnadesign_densegen_cpu.101.out"
    keep_file.write_text("keep\n", encoding="utf-8")
    prune_file.write_text("prune\n", encoding="utf-8")
    old_timestamp = 946684800  # 2000-01-01T00:00:00Z
    os.utime(prune_file, (old_timestamp, old_timestamp))
    manifest_path = stdout_dir / "retention-manifest.json"

    exit_code = main(
        [
            "prune-ops-logs",
            "--stdout-dir",
            str(stdout_dir),
            "--runbook-id",
            "study_stress_ethanol_cipro",
            "--keep-last",
            "1",
            "--max-age-days",
            "7",
            "--manifest-path",
            str(manifest_path),
            "--json",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert keep_file.exists()
    assert not prune_file.exists()
    payload = json.loads(captured.out)
    assert payload["pruned_count"] == 1
    assert payload["manifest_path"] == str(manifest_path)
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest_payload["runbook_id"] == "study_stress_ethanol_cipro"
    assert manifest_payload["pruned_count"] == 1


def test_main_prune_ops_logs_rejects_stdout_dir_outside_runbook_scope(tmp_path: Path, capsys) -> None:
    stdout_dir = tmp_path / "outside"
    stdout_dir.mkdir(parents=True, exist_ok=True)

    exit_code = main(
        [
            "prune-ops-logs",
            "--stdout-dir",
            str(stdout_dir),
            "--runbook-id",
            "study_stress_ethanol_cipro",
            "--keep-last",
            "5",
            "--max-age-days",
            "30",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "must be exactly workspace/outputs/logs/ops/sge/<runbook-id>" in captured.err


def test_main_prune_ops_logs_runtime_kind_prunes_old_files(tmp_path: Path, capsys) -> None:
    runtime_dir = tmp_path / "workspace" / "outputs" / "logs" / "ops" / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    keep_file = runtime_dir / "dnadesign_densegen_cpu.300.trace.log"
    prune_file = runtime_dir / "dnadesign_densegen_cpu.301.trace.log"
    keep_file.write_text("keep\n", encoding="utf-8")
    prune_file.write_text("prune\n", encoding="utf-8")
    old_timestamp = 946684800  # 2000-01-01T00:00:00Z
    os.utime(prune_file, (old_timestamp, old_timestamp))
    manifest_path = runtime_dir / "retention-manifest.json"

    exit_code = main(
        [
            "prune-ops-logs",
            "--stdout-dir",
            str(runtime_dir),
            "--log-kind",
            "runtime",
            "--runbook-id",
            "study_stress_ethanol_cipro",
            "--keep-last",
            "1",
            "--max-age-days",
            "7",
            "--manifest-path",
            str(manifest_path),
            "--json",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    assert keep_file.exists()
    assert not prune_file.exists()
    payload = json.loads(captured.out)
    assert payload["log_kind"] == "runtime"
    assert payload["pruned_count"] == 1


def test_main_prune_ops_logs_runtime_kind_rejects_invalid_scope(tmp_path: Path, capsys) -> None:
    wrong_dir = tmp_path / "workspace" / "outputs" / "logs" / "ops" / "sge" / "study_stress_ethanol_cipro"
    wrong_dir.mkdir(parents=True, exist_ok=True)

    exit_code = main(
        [
            "prune-ops-logs",
            "--stdout-dir",
            str(wrong_dir),
            "--log-kind",
            "runtime",
            "--runbook-id",
            "study_stress_ethanol_cipro",
            "--keep-last",
            "5",
            "--max-age-days",
            "30",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "must be exactly workspace/outputs/logs/ops/runtime" in captured.err


def test_main_usr_overlay_guard_blocks_when_projected_overlay_parts_exceed_threshold(tmp_path: Path, capsys) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    config_path = workspace_root / "config.yaml"
    _write_densegen_config(
        config_path,
        chunk_size=128,
        max_accepted_per_library=2,
        total_sequences=1_000_000,
    )

    exit_code = main(
        [
            "usr-overlay-guard",
            "--config",
            str(config_path),
            "--workspace-root",
            str(workspace_root),
            "--mode",
            "fresh",
            "--run-args",
            "--fresh --no-plot",
            "--max-projected-overlay-parts",
            "1000",
            "--max-existing-overlay-parts",
            "5000",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "projected_overlay_parts" in captured.err
    assert "max_accepted_per_library" in captured.err


def test_main_usr_overlay_guard_auto_compacts_when_existing_parts_exceed_threshold(tmp_path: Path, capsys) -> None:
    pyarrow = pytest.importorskip("pyarrow")

    import dnadesign.usr as usr_pkg

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    config_path = workspace_root / "config.yaml"
    _write_densegen_config(
        config_path,
        dataset="densegen/demo",
        chunk_size=128,
        max_accepted_per_library=64,
        total_sequences=1000,
    )

    usr_root = workspace_root / "outputs" / "usr_datasets"
    registry_src = Path(usr_pkg.__file__).resolve().parent / "datasets" / "registry.yaml"
    registry_dst = usr_root / "registry.yaml"
    registry_dst.parent.mkdir(parents=True, exist_ok=True)
    registry_dst.write_text(registry_src.read_text(encoding="utf-8"), encoding="utf-8")

    dataset = usr_pkg.Dataset(usr_root, "densegen/demo")
    dataset.init(source="test")
    dataset.import_rows([{"sequence": "ACGT", "bio_type": "dna", "alphabet": "dna_4"}], source="seed")
    row_id = str(dataset.head(1, columns=["id"]).iloc[0]["id"])
    for idx in range(2):
        table = pyarrow.table(
            {
                "id": [row_id],
                "densegen__run_id": [f"run-{idx}"],
            }
        )
        dataset.write_overlay_part("densegen", table, key="id")
    overlay_parts_dir = dataset.dir / "_derived" / "densegen"
    assert len(sorted(overlay_parts_dir.glob("part-*.parquet"))) == 2

    exit_code = main(
        [
            "usr-overlay-guard",
            "--config",
            str(config_path),
            "--workspace-root",
            str(workspace_root),
            "--mode",
            "resume",
            "--run-args",
            "--resume --no-plot",
            "--max-projected-overlay-parts",
            "100000",
            "--max-existing-overlay-parts",
            "1",
            "--auto-compact-existing-overlay-parts",
            "--json",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["compacted"] is True
    assert payload["existing_overlay_parts_before"] == 2
    compacted_overlay = dataset.dir / "_derived" / "densegen.parquet"
    assert compacted_overlay.exists()
    assert not overlay_parts_dir.exists()


def test_main_usr_overlay_guard_resolves_usr_root_relative_to_densegen_config_path(tmp_path: Path, capsys) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    config_path = workspace_root / "configs" / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    _write_densegen_config(
        config_path,
        run_root="..",
        usr_root="../outputs/usr_datasets",
        chunk_size=256,
        max_accepted_per_library=32,
        total_sequences=1000,
    )

    exit_code = main(
        [
            "usr-overlay-guard",
            "--tool",
            "densegen",
            "--config",
            str(config_path),
            "--workspace-root",
            str(workspace_root),
            "--mode",
            "fresh",
            "--run-args",
            "--fresh --no-plot",
            "--max-projected-overlay-parts",
            "100000",
            "--max-existing-overlay-parts",
            "5000",
            "--json",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["usr_root"] == str((workspace_root / "outputs" / "usr_datasets").resolve())


def test_main_usr_overlay_guard_rejects_workspace_root_mismatch(tmp_path: Path, capsys) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    config_path = workspace_root / "config.yaml"
    _write_densegen_config(config_path)
    mismatched_workspace = tmp_path / "other_workspace"
    mismatched_workspace.mkdir(parents=True, exist_ok=True)

    exit_code = main(
        [
            "usr-overlay-guard",
            "--tool",
            "densegen",
            "--config",
            str(config_path),
            "--workspace-root",
            str(mismatched_workspace),
            "--mode",
            "fresh",
            "--run-args",
            "--fresh --no-plot",
            "--max-projected-overlay-parts",
            "100000",
            "--max-existing-overlay-parts",
            "5000",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "workspace-root must match tool config run root" in captured.err


def test_main_usr_overlay_guard_rejects_unsupported_tool(tmp_path: Path, capsys) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    config_path = workspace_root / "config.yaml"
    _write_densegen_config(config_path)

    exit_code = main(
        [
            "usr-overlay-guard",
            "--tool",
            "unknown_tool",
            "--config",
            str(config_path),
            "--workspace-root",
            str(workspace_root),
            "--mode",
            "fresh",
            "--run-args",
            "--fresh --no-plot",
            "--max-projected-overlay-parts",
            "1000",
            "--max-existing-overlay-parts",
            "1000",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "unsupported usr-overlay-guard tool" in captured.err


def test_main_usr_overlay_guard_infer_tool_is_explicitly_skipped(tmp_path: Path, capsys) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    config_path = workspace_root / "configs" / "infer_config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    _write_infer_config(
        config_path,
        dataset="infer/demo",
        usr_root="../outputs/usr_datasets",
    )

    exit_code = main(
        [
            "usr-overlay-guard",
            "--tool",
            "infer_evo2",
            "--config",
            str(config_path),
            "--workspace-root",
            str(workspace_root),
            "--mode",
            "fresh",
            "--run-args",
            "",
            "--max-projected-overlay-parts",
            "1000",
            "--max-existing-overlay-parts",
            "1000",
            "--json",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["guard_status"] == "skipped"
    assert payload["tool"] == "infer_evo2"
    assert "does not emit overlay parts" in payload["reason"]


def test_main_usr_records_part_guard_infer_tool_is_explicitly_skipped(tmp_path: Path, capsys) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    config_path = workspace_root / "config.yaml"
    _write_infer_config(config_path, dataset="infer/demo", usr_root="outputs/usr_datasets")

    exit_code = main(
        [
            "usr-records-part-guard",
            "--tool",
            "infer_evo2",
            "--config",
            str(config_path),
            "--workspace-root",
            str(workspace_root),
            "--mode",
            "fresh",
            "--run-args",
            "",
            "--max-projected-records-parts",
            "1000",
            "--max-existing-records-parts",
            "1000",
            "--max-existing-records-part-age-days",
            "14",
            "--json",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["guard_status"] == "skipped"
    assert payload["tool"] == "infer_evo2"
    assert "does not emit records part files" in payload["reason"]


def test_main_usr_records_part_guard_blocks_when_projected_records_parts_exceed_threshold(
    tmp_path: Path, capsys
) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    config_path = workspace_root / "config.yaml"
    _write_densegen_config(
        config_path,
        chunk_size=512,
        parquet_chunk_size=128,
        max_accepted_per_library=2,
        total_sequences=1_000_000,
    )

    exit_code = main(
        [
            "usr-records-part-guard",
            "--config",
            str(config_path),
            "--workspace-root",
            str(workspace_root),
            "--mode",
            "fresh",
            "--run-args",
            "--fresh --no-plot",
            "--max-projected-records-parts",
            "1000",
            "--max-existing-records-parts",
            "5000",
            "--max-existing-records-part-age-days",
            "14",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "projected_records_parts" in captured.err
    assert "output.parquet.chunk_size" in captured.err


def test_main_usr_records_part_guard_auto_compacts_when_existing_parts_are_stale(tmp_path: Path, capsys) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    config_path = workspace_root / "config.yaml"
    _write_densegen_config(
        config_path,
        chunk_size=512,
        parquet_chunk_size=1024,
        max_accepted_per_library=64,
        total_sequences=1000,
    )

    tables_root = workspace_root / "outputs" / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)
    old_part_path = tables_root / "records__part-old.parquet"
    pyarrow_parquet.write_table(pyarrow.table({"id": ["row-1"], "sequence": ["ACGT"]}), old_part_path)
    old_timestamp = 946684800  # 2000-01-01T00:00:00Z
    os.utime(old_part_path, (old_timestamp, old_timestamp))

    exit_code = main(
        [
            "usr-records-part-guard",
            "--config",
            str(config_path),
            "--workspace-root",
            str(workspace_root),
            "--mode",
            "resume",
            "--run-args",
            "--resume --no-plot",
            "--max-projected-records-parts",
            "100000",
            "--max-existing-records-parts",
            "1000",
            "--max-existing-records-part-age-days",
            "7",
            "--auto-compact-existing-records-parts",
            "--json",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 0
    payload = json.loads(captured.out)
    assert payload["compacted"] is True
    assert payload["existing_records_parts_before"] == 1
    assert payload["existing_records_parts_after"] == 0
    assert payload["oldest_existing_records_part_age_days_before"] > 7
    assert (tables_root / "records.parquet").exists()
    assert not old_part_path.exists()


def test_main_usr_records_part_guard_blocks_when_existing_parts_are_stale_without_auto_compact(
    tmp_path: Path, capsys
) -> None:
    pyarrow = pytest.importorskip("pyarrow")
    pyarrow_parquet = pytest.importorskip("pyarrow.parquet")

    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    config_path = workspace_root / "config.yaml"
    _write_densegen_config(
        config_path,
        chunk_size=512,
        parquet_chunk_size=1024,
        max_accepted_per_library=64,
        total_sequences=1000,
    )

    tables_root = workspace_root / "outputs" / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)
    stale_part_path = tables_root / "records__part-stale.parquet"
    pyarrow_parquet.write_table(pyarrow.table({"id": ["row-1"], "sequence": ["ACGT"]}), stale_part_path)
    old_timestamp = 946684800  # 2000-01-01T00:00:00Z
    os.utime(stale_part_path, (old_timestamp, old_timestamp))

    exit_code = main(
        [
            "usr-records-part-guard",
            "--config",
            str(config_path),
            "--workspace-root",
            str(workspace_root),
            "--mode",
            "resume",
            "--run-args",
            "--resume --no-plot",
            "--max-projected-records-parts",
            "100000",
            "--max-existing-records-parts",
            "1000",
            "--max-existing-records-part-age-days",
            "7",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "existing records parts require maintenance" in captured.err
    assert "auto-compact-existing-records-parts" in captured.err


def test_main_usr_archived_overlay_guard_blocks_when_archived_entries_exceed_threshold(tmp_path: Path, capsys) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    config_path = workspace_root / "config.yaml"
    _write_densegen_config(config_path)

    archived_dir = (
        workspace_root
        / "outputs"
        / "usr_datasets"
        / "densegen"
        / "study_stress_ethanol_cipro"
        / "_derived"
        / "_archived"
    )
    archived_dir.mkdir(parents=True, exist_ok=True)
    (archived_dir / "a.parquet").write_bytes(b"a")
    (archived_dir / "b.parquet").write_bytes(b"b")

    exit_code = main(
        [
            "usr-archived-overlay-guard",
            "--tool",
            "densegen",
            "--config",
            str(config_path),
            "--workspace-root",
            str(workspace_root),
            "--max-archived-entries",
            "1",
            "--max-archived-bytes",
            "1024",
            "--json",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "archived_entries exceeds threshold" in captured.err
    assert "archived_entries" in captured.err


def test_main_usr_archived_overlay_guard_blocks_when_archived_bytes_exceed_threshold(tmp_path: Path, capsys) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir(parents=True, exist_ok=True)
    config_path = workspace_root / "config.yaml"
    _write_densegen_config(config_path)

    archived_dir = (
        workspace_root
        / "outputs"
        / "usr_datasets"
        / "densegen"
        / "study_stress_ethanol_cipro"
        / "_derived"
        / "_archived"
    )
    archived_dir.mkdir(parents=True, exist_ok=True)
    (archived_dir / "big.parquet").write_bytes(b"0123456789")

    exit_code = main(
        [
            "usr-archived-overlay-guard",
            "--tool",
            "densegen",
            "--config",
            str(config_path),
            "--workspace-root",
            str(workspace_root),
            "--max-archived-entries",
            "100",
            "--max-archived-bytes",
            "5",
            "--json",
        ]
    )
    captured = capsys.readouterr()

    assert exit_code == 2
    assert "archived_bytes exceeds threshold" in captured.err
    assert "max_archived_bytes" in captured.err
