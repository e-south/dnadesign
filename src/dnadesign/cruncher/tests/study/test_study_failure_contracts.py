"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/study/test_study_failure_contracts.py

Validate Study failure contracts (replay errors, summarize ordering, partial data).

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

import dnadesign.cruncher.app.study_workflow as study_workflow
from dnadesign.cruncher.analysis.layout import analysis_root, analysis_table_path
from dnadesign.cruncher.app.study_workflow import run_study, summarize_study_run
from dnadesign.cruncher.artifacts.layout import elites_path
from dnadesign.cruncher.study.layout import (
    spec_frozen_path,
    study_manifest_path,
    study_plot_path,
    study_status_path,
    study_table_path,
)
from dnadesign.cruncher.study.manifest import load_study_manifest, load_study_status, write_study_manifest
from dnadesign.cruncher.tests.study._helpers import write_study_spec, write_workspace_config


def _only_study_run_dir(tmp_path: Path, *, study_name: str = "smoke_study") -> Path:
    roots = sorted((tmp_path / "outputs" / "studies" / study_name).glob("*"))
    assert len(roots) == 1
    return roots[0]


def test_shutdown_trial_worker_process_terminates_and_kills_when_still_alive() -> None:
    events: list[str] = []

    class _FakeProcess:
        def __init__(self) -> None:
            self._alive = True

        def is_alive(self) -> bool:
            return bool(self._alive)

        def terminate(self) -> None:
            events.append("terminate")

        def kill(self) -> None:
            events.append("kill")
            self._alive = False

        def join(self, timeout: float | None = None) -> None:
            if timeout is None:
                events.append("join")
            else:
                events.append(f"join:{float(timeout)}")

    process = _FakeProcess()
    study_workflow._shutdown_trial_worker_process(process, terminate=True)

    assert events == ["terminate", "join:5.0", "kill", "join"]


def test_replay_failure_updates_manifest_and_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_workspace_config(tmp_path)
    spec_path = tmp_path / "smoke.study.yaml"
    write_study_spec(
        spec_path,
        profile="analysis_ready",
        mmr_enabled=True,
        seeds=[11],
        trials=[{"id": "L6", "factors": {"sample.sequence_length": 6}}],
    )

    def _boom(*args, **kwargs):
        raise RuntimeError("mmr replay boom")

    monkeypatch.setattr(study_workflow, "run_mmr_sweep_for_run", _boom)

    with pytest.raises(RuntimeError, match="Study completed with trial errors"):
        run_study(spec_path)

    run_dir = _only_study_run_dir(tmp_path)
    manifest = load_study_manifest(study_manifest_path(run_dir))
    status = load_study_status(study_status_path(run_dir))
    assert any(item.status == "error" for item in manifest.trial_runs)
    assert status.error_runs == 1
    assert status.finished_at is not None
    assert status.status == "completed_with_errors"


def test_run_study_always_zero_skips_summary_when_trials_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_workspace_config(tmp_path)
    spec_path = tmp_path / "always_zero.study.yaml"
    write_study_spec(
        spec_path,
        profile="analysis_ready",
        mmr_enabled=False,
        seeds=[11],
        trials=[
            {"id": "fail", "factors": {"sample.sequence_length": 6}},
            {"id": "ok", "factors": {"sample.sequence_length": 6}},
        ],
    )
    raw = spec_path.read_text()
    spec_path.write_text(raw.replace("exit_code_policy: nonzero_if_any_error", "exit_code_policy: always_zero"))

    original_run_sample = study_workflow.run_sample

    def _run_sample_with_failure(*args, **kwargs):
        cfg = args[0]
        out_dir = str(cfg.workspace.out_dir)
        if "/fail/" in out_dir:
            raise RuntimeError("forced trial failure")
        return original_run_sample(*args, **kwargs)

    monkeypatch.setattr(study_workflow, "run_sample", _run_sample_with_failure)

    run_dir = run_study(spec_path)
    status = load_study_status(study_status_path(run_dir))
    assert status.status == "completed_with_errors"
    assert status.error_runs >= 1
    assert any("Summary skipped due trial errors" in warning for warning in status.warnings)
    assert not study_table_path(run_dir, "trial_metrics", "parquet").exists()


def test_parallel_abort_marks_not_started_trials_skipped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    write_workspace_config(tmp_path)
    spec_path = tmp_path / "parallel_abort.study.yaml"
    write_study_spec(
        spec_path,
        profile="analysis_ready",
        mmr_enabled=False,
        seeds=[11],
        parallelism=2,
        on_trial_error="abort",
        trials=[
            {"id": "BAD", "factors": {"sample.sequence_length": "bad"}},
            {"id": "OK6", "factors": {"sample.sequence_length": 6}},
            {"id": "OK7", "factors": {"sample.sequence_length": 7}},
        ],
    )

    monkeypatch.setattr(study_workflow, "_preflight_trial_config_contracts", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="Study completed with trial errors"):
        run_study(spec_path, progress_bar=False, quiet_logs=True)

    run_dir = _only_study_run_dir(tmp_path)
    manifest = load_study_manifest(study_manifest_path(run_dir))
    status = load_study_status(study_status_path(run_dir))
    trial_statuses = [item.status for item in manifest.trial_runs]

    assert "error" in trial_statuses
    assert "skipped" in trial_statuses
    assert status.pending_runs == 0
    assert status.status == "failed"
    skipped_reasons = [str(item.error or "") for item in manifest.trial_runs if item.status == "skipped"]
    assert skipped_reasons and all("aborted" in reason.lower() for reason in skipped_reasons)


def test_summarize_allow_partial_annotates_missing_counts(tmp_path: Path) -> None:
    write_workspace_config(tmp_path)
    spec_path = tmp_path / "smoke.study.yaml"
    write_study_spec(
        spec_path,
        profile="analysis_ready",
        mmr_enabled=False,
        seeds=[11, 12],
        trials=[{"id": "L6", "factors": {"sample.sequence_length": 6}}],
    )

    run_dir = run_study(spec_path)
    manifest_file = study_manifest_path(run_dir)
    manifest = load_study_manifest(manifest_file)
    first = manifest.trial_runs[0]
    assert first.run_dir is not None
    first.run_dir = str(Path(first.run_dir).with_name("missing_study_trial_dir"))
    manifest.trial_runs[0] = first
    write_study_manifest(manifest_file, manifest)

    result = summarize_study_run(run_dir, allow_partial=True)
    assert result.n_missing_total > 0
    assert result.exit_code_policy == "nonzero_if_any_error"

    agg = pd.read_parquet(study_table_path(run_dir, "trial_metrics_agg", "parquet"))
    assert "n_missing_total" in agg.columns
    assert int(agg["n_missing_total"].iloc[0]) == int(result.n_missing_total)


def test_summarize_allow_partial_does_not_double_count_missing_trial_for_mmr(tmp_path: Path) -> None:
    write_workspace_config(tmp_path)
    spec_path = tmp_path / "smoke.study.yaml"
    write_study_spec(
        spec_path,
        profile="analysis_ready",
        mmr_enabled=True,
        seeds=[11, 12],
        trials=[{"id": "L6", "factors": {"sample.sequence_length": 6}}],
    )

    run_dir = run_study(spec_path)
    manifest_file = study_manifest_path(run_dir)
    manifest = load_study_manifest(manifest_file)
    first = manifest.trial_runs[0]
    assert first.run_dir is not None
    first.run_dir = str(Path(first.run_dir).with_name("missing_study_trial_dir_for_mmr"))
    manifest.trial_runs[0] = first
    write_study_manifest(manifest_file, manifest)

    result = summarize_study_run(run_dir, allow_partial=True)
    assert result.n_missing_total == 1
    assert result.n_missing_run_dirs == 1
    assert result.n_missing_mmr_tables == 0


def test_summarize_rejects_missing_execution_contract_in_frozen_spec(tmp_path: Path) -> None:
    write_workspace_config(tmp_path)
    spec_path = tmp_path / "smoke.study.yaml"
    write_study_spec(
        spec_path,
        profile="analysis_ready",
        mmr_enabled=False,
        seeds=[11],
        trials=[{"id": "L6", "factors": {"sample.sequence_length": 6}}],
    )

    run_dir = run_study(spec_path)
    frozen_path = spec_frozen_path(run_dir)
    payload = yaml.safe_load(frozen_path.read_text())
    assert isinstance(payload, dict) and isinstance(payload.get("study"), dict)
    study_payload = dict(payload["study"])
    study_payload.pop("execution", None)
    payload["study"] = study_payload
    frozen_path.write_text(yaml.safe_dump(payload))

    with pytest.raises(ValueError, match="missing required key: execution"):
        summarize_study_run(run_dir, allow_partial=False)


def test_summarize_allow_partial_counts_missing_metric_artifacts(tmp_path: Path) -> None:
    write_workspace_config(tmp_path)
    spec_path = tmp_path / "smoke.study.yaml"
    write_study_spec(
        spec_path,
        profile="analysis_ready",
        mmr_enabled=False,
        seeds=[11, 12],
        trials=[{"id": "L6", "factors": {"sample.sequence_length": 6}}],
    )

    run_dir = run_study(spec_path)
    manifest = load_study_manifest(study_manifest_path(run_dir))
    first = manifest.trial_runs[0]
    assert first.run_dir is not None
    elites_file = elites_path(Path(first.run_dir))
    assert elites_file.exists()
    elites_file.unlink()

    result = summarize_study_run(run_dir, allow_partial=True)
    assert result.n_missing_total > 0
    assert result.n_missing_metric_artifacts == 1

    agg = pd.read_parquet(study_table_path(run_dir, "trial_metrics_agg", "parquet"))
    assert "n_missing_metric_artifacts" in agg.columns
    assert int(agg["n_missing_metric_artifacts"].iloc[0]) == 1


def test_summarize_allow_partial_removes_stale_mmr_outputs(tmp_path: Path) -> None:
    write_workspace_config(tmp_path)
    spec_path = tmp_path / "smoke.study.yaml"
    write_study_spec(
        spec_path,
        profile="analysis_ready",
        mmr_enabled=True,
        seeds=[11],
        trials=[{"id": "L6", "factors": {"sample.sequence_length": 6}}],
    )

    run_dir = run_study(spec_path)
    mmr_agg_path = study_table_path(run_dir, "mmr_tradeoff_agg", "parquet")
    mmr_plot_path = study_plot_path(run_dir, "mmr_diversity_tradeoff", "pdf")
    assert mmr_agg_path.exists()
    assert mmr_plot_path.exists()

    manifest = load_study_manifest(study_manifest_path(run_dir))
    trial = manifest.trial_runs[0]
    assert trial.run_dir is not None
    mmr_run_table = analysis_table_path(analysis_root(Path(trial.run_dir)), "elites_mmr_sweep", "parquet")
    assert mmr_run_table.exists()
    mmr_run_table.unlink()

    result = summarize_study_run(run_dir, allow_partial=True)
    assert result.n_missing_mmr_tables == 1
    assert not mmr_agg_path.exists()
    assert not mmr_plot_path.exists()


def test_mmr_tradeoff_keeps_distinct_trials_with_same_sequence_length(tmp_path: Path) -> None:
    write_workspace_config(tmp_path)
    spec_path = tmp_path / "same_len.study.yaml"
    write_study_spec(
        spec_path,
        profile="analysis_ready",
        mmr_enabled=True,
        seeds=[11],
        trials=[
            {
                "id": "A",
                "factors": {
                    "sample.sequence_length": 6,
                    "sample.elites.select.diversity": 0.0,
                },
            },
            {
                "id": "B",
                "factors": {
                    "sample.sequence_length": 6,
                    "sample.elites.select.diversity": 1.0,
                },
            },
        ],
    )
    run_dir = run_study(spec_path)
    mmr_agg = pd.read_parquet(study_table_path(run_dir, "mmr_tradeoff_agg", "parquet"))
    assert "trial_id" in mmr_agg.columns
    assert set(mmr_agg["trial_id"].astype(str).tolist()) == {"A", "B"}
    assert mmr_agg["series_label"].nunique() == 2
    assert len(mmr_agg) == 6


def test_sequence_length_plot_is_skipped_when_length_does_not_vary(tmp_path: Path) -> None:
    write_workspace_config(tmp_path)
    spec_path = tmp_path / "same_len_single.study.yaml"
    write_study_spec(
        spec_path,
        profile="analysis_ready",
        mmr_enabled=True,
        seeds=[11],
        trials=[{"id": "BASE", "factors": {"sample.sequence_length": 6}}],
    )

    run_dir = run_study(spec_path)
    mmr_plot_path = study_plot_path(run_dir, "mmr_diversity_tradeoff", "pdf")
    seq_plot_path = study_plot_path(run_dir, "sequence_length_tradeoff", "pdf")
    assert mmr_plot_path.exists()
    assert not seq_plot_path.exists()
