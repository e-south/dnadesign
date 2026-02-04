from __future__ import annotations

from dnadesign.densegen.src.core.run_manifest import PlanManifest, RunManifest, load_run_manifest
from dnadesign.densegen.src.core.run_paths import ensure_run_meta_dir, run_manifest_path


def test_run_manifest_roundtrip(tmp_path) -> None:
    items = [
        PlanManifest(
            input_name="demo",
            plan_name="demo",
            generated=5,
            duplicates_skipped=1,
            failed_solutions=2,
            total_resamples=0,
            libraries_built=1,
            stall_events=0,
            failed_min_count_per_tf=1,
            failed_required_regulators=0,
            failed_min_count_by_regulator=1,
            failed_min_required_regulators=0,
            duplicate_solutions=3,
            leaderboard_latest={
                "tf": [{"tf": "lexA", "count": 3}],
                "tfbs": [{"tf": "lexA", "tfbs": "TTAC", "count": 3}],
                "failed_tfbs": [],
                "diversity": {"tf_coverage": 1.0, "tfbs_coverage": 1.0, "tfbs_entropy": 0.0},
            },
        )
    ]
    manifest = RunManifest(
        run_id="demo_run",
        created_at="2026-01-16T12:00:00Z",
        schema_version="2.7",
        config_sha256="abc123",
        run_root="outputs",
        random_seed=42,
        seed_stage_a=101,
        seed_stage_b=202,
        seed_solver=303,
        solver_backend="CBC",
        solver_strategy="iterate",
        solver_time_limit_seconds=5.0,
        solver_threads=2,
        solver_strands="double",
        dense_arrays_version="0.0.0",
        dense_arrays_version_source="lock",
        items=items,
    )
    ensure_run_meta_dir(tmp_path)
    path = run_manifest_path(tmp_path)
    manifest.write_json(path)
    loaded = load_run_manifest(path)
    assert loaded.schema_version == "2.7"
    assert loaded.dense_arrays_version == "0.0.0"
    assert loaded.dense_arrays_version_source == "lock"
    assert loaded.items[0].failed_min_count_per_tf == 1
    assert loaded.items[0].duplicate_solutions == 3
    assert loaded.items[0].leaderboard_latest is not None
    assert loaded.random_seed == 42
    assert loaded.solver_time_limit_seconds == 5.0
    assert loaded.solver_threads == 2
