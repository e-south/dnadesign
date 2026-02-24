"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/app/test_study_compaction.py

Validate study-run compaction contracts for transient trial artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnadesign.cruncher.app.study_compaction import compact_study_run
from dnadesign.cruncher.study.layout import study_manifest_path, study_status_path
from dnadesign.cruncher.study.manifest import (
    StudyManifestV1,
    StudyStatusV1,
    StudyTrialRun,
    write_study_manifest,
    write_study_status,
)


def _seed_study_run(tmp_path: Path) -> tuple[Path, Path]:
    study_run_dir = tmp_path / "outputs" / "studies" / "sweep" / "abc123"
    trial_run_dir = study_run_dir / "trials" / "L6" / "seed_11" / "sample_run"
    (trial_run_dir / "optimize" / "tables").mkdir(parents=True, exist_ok=True)
    (trial_run_dir / "optimize" / "state").mkdir(parents=True, exist_ok=True)
    (trial_run_dir / "meta").mkdir(parents=True, exist_ok=True)

    (trial_run_dir / "optimize" / "tables" / "elites.parquet").write_bytes(b"elites")
    (trial_run_dir / "optimize" / "tables" / "elites_hits.parquet").write_bytes(b"hits")
    (trial_run_dir / "optimize" / "tables" / "sequences.parquet").write_bytes(b"sequences")
    (trial_run_dir / "optimize" / "tables" / "random_baseline.parquet").write_bytes(b"baseline")
    (trial_run_dir / "optimize" / "tables" / "random_baseline_hits.parquet").write_bytes(b"baseline-hits")
    (trial_run_dir / "optimize" / "state" / "trace.nc").write_bytes(b"trace")
    (trial_run_dir / "optimize" / "optimizer_move_stats.json.gz").write_bytes(b"stats")
    (trial_run_dir / "optimize" / "tables" / "sequences.parquet.tmp").write_bytes(b"tmp")

    manifest = StudyManifestV1(
        study_name="sweep",
        study_id="abc123",
        spec_path=str(study_run_dir / "study" / "spec_frozen.yaml"),
        spec_sha256="spec",
        base_config_path=str(study_run_dir / "configs" / "config.yaml"),
        base_config_sha256="config",
        created_at="2026-02-21T00:00:00+00:00",
        trial_runs=[
            StudyTrialRun(
                trial_id="L6",
                seed=11,
                target_set_index=1,
                target_tfs=["lexA", "cpxR"],
                status="success",
                run_dir=str(trial_run_dir.relative_to(study_run_dir)),
            )
        ],
    )
    status = StudyStatusV1(
        study_name="sweep",
        study_id="abc123",
        status="completed",
        total_runs=1,
        pending_runs=0,
        running_runs=0,
        success_runs=1,
        error_runs=0,
        skipped_runs=0,
        warnings=[],
        started_at="2026-02-21T00:00:00+00:00",
        updated_at="2026-02-21T00:00:00+00:00",
        finished_at="2026-02-21T00:00:00+00:00",
    )
    write_study_manifest(study_manifest_path(study_run_dir), manifest)
    write_study_status(study_status_path(study_run_dir), status)
    return study_run_dir, trial_run_dir


def test_compact_study_run_dry_run_reports_candidates(tmp_path: Path) -> None:
    study_run_dir, trial_run_dir = _seed_study_run(tmp_path)

    summary = compact_study_run(study_run_dir, confirm=False)

    assert summary.trial_count == 1
    assert summary.candidate_file_count >= 5
    assert summary.candidate_bytes > 0
    assert summary.deleted_file_count == 0
    assert summary.deleted_bytes == 0
    assert (trial_run_dir / "optimize" / "tables" / "sequences.parquet").exists()


def test_compact_study_run_confirm_deletes_bulk_trial_artifacts(tmp_path: Path) -> None:
    study_run_dir, trial_run_dir = _seed_study_run(tmp_path)

    summary = compact_study_run(study_run_dir, confirm=True)

    assert summary.trial_count == 1
    assert summary.deleted_file_count >= 5
    assert summary.deleted_bytes > 0
    assert not (trial_run_dir / "optimize" / "tables" / "sequences.parquet").exists()
    assert not (trial_run_dir / "optimize" / "tables" / "random_baseline.parquet").exists()
    assert not (trial_run_dir / "optimize" / "tables" / "random_baseline_hits.parquet").exists()
    assert not (trial_run_dir / "optimize" / "state" / "trace.nc").exists()
    assert not (trial_run_dir / "optimize" / "optimizer_move_stats.json.gz").exists()
    assert not (trial_run_dir / "optimize" / "tables" / "sequences.parquet.tmp").exists()
    assert (trial_run_dir / "optimize" / "tables" / "elites.parquet").exists()
    assert (trial_run_dir / "optimize" / "tables" / "elites_hits.parquet").exists()


def test_compact_study_run_requires_elites_for_success_trials(tmp_path: Path) -> None:
    study_run_dir, trial_run_dir = _seed_study_run(tmp_path)
    (trial_run_dir / "optimize" / "tables" / "elites.parquet").unlink()

    with pytest.raises(FileNotFoundError, match="Missing elites parquet for successful trial"):
        compact_study_run(study_run_dir, confirm=False)


def test_compact_study_run_rejects_trial_run_dir_escape(tmp_path: Path) -> None:
    study_run_dir, _trial_run_dir = _seed_study_run(tmp_path)
    escaped_dir = tmp_path / "escaped_trial"
    (escaped_dir / "optimize" / "tables").mkdir(parents=True, exist_ok=True)
    (escaped_dir / "optimize" / "tables" / "elites.parquet").write_bytes(b"elites")
    (escaped_dir / "optimize" / "tables" / "sequences.parquet").write_bytes(b"sequences")

    manifest = StudyManifestV1(
        study_name="sweep",
        study_id="abc123",
        spec_path=str(study_run_dir / "study" / "spec_frozen.yaml"),
        spec_sha256="spec",
        base_config_path=str(study_run_dir / "configs" / "config.yaml"),
        base_config_sha256="config",
        created_at="2026-02-21T00:00:00+00:00",
        trial_runs=[
            StudyTrialRun(
                trial_id="L6",
                seed=11,
                target_set_index=1,
                target_tfs=["lexA", "cpxR"],
                status="success",
                run_dir=str(Path("..") / ".." / ".." / ".." / escaped_dir.name),
            )
        ],
    )
    write_study_manifest(study_manifest_path(study_run_dir), manifest)

    with pytest.raises(ValueError, match="must not contain '\\.\\.'"):
        compact_study_run(study_run_dir, confirm=True)
    assert (escaped_dir / "optimize" / "tables" / "sequences.parquet").exists()


def test_compact_study_run_rejects_absolute_trial_run_dir(tmp_path: Path) -> None:
    study_run_dir, _trial_run_dir = _seed_study_run(tmp_path)
    external_run_dir = tmp_path / "external_trial"
    (external_run_dir / "optimize" / "tables").mkdir(parents=True, exist_ok=True)
    (external_run_dir / "optimize" / "tables" / "elites.parquet").write_bytes(b"elites")

    manifest = StudyManifestV1(
        study_name="sweep",
        study_id="abc123",
        spec_path=str(study_run_dir / "study" / "spec_frozen.yaml"),
        spec_sha256="spec",
        base_config_path=str(study_run_dir / "configs" / "config.yaml"),
        base_config_sha256="config",
        created_at="2026-02-21T00:00:00+00:00",
        trial_runs=[
            StudyTrialRun(
                trial_id="L6",
                seed=11,
                target_set_index=1,
                target_tfs=["lexA", "cpxR"],
                status="success",
                run_dir=str(external_run_dir.resolve()),
            )
        ],
    )
    write_study_manifest(study_manifest_path(study_run_dir), manifest)

    with pytest.raises(ValueError, match="must resolve under"):
        compact_study_run(study_run_dir, confirm=False)


def test_compact_study_run_rejects_symlink_candidates(tmp_path: Path) -> None:
    study_run_dir, trial_run_dir = _seed_study_run(tmp_path)
    external = tmp_path / "outside.txt"
    external.write_text("outside")
    target = trial_run_dir / "optimize" / "tables" / "sequences.parquet"
    target.unlink()
    target.symlink_to(external)

    with pytest.raises(ValueError, match="must not be a symlink"):
        compact_study_run(study_run_dir, confirm=True)
    assert external.exists()
