"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/densegen/tests/runtime/test_run_finalization.py

Unit tests for run finalization artifact merge behavior.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from dnadesign.densegen.src.core.pipeline import run_finalization as run_finalization_module


def _minimal_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        run=SimpleNamespace(id="demo-run"),
        schema_version="2.9",
        solver=SimpleNamespace(strategy="iterate", strands="double"),
        generation=SimpleNamespace(sampling=SimpleNamespace(model_dump=lambda: {"library_source": "build"})),
    )


def test_finalize_run_outputs_raises_on_invalid_existing_composition_parquet(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path
    tables_root = run_root / "outputs" / "tables"
    tables_root.mkdir(parents=True, exist_ok=True)
    (tables_root / "composition.parquet").write_text("not parquet", encoding="utf-8")

    monkeypatch.setattr(run_finalization_module, "_consolidate_parts", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(run_finalization_module, "write_library_artifacts", lambda **_kwargs: None)
    monkeypatch.setattr(run_finalization_module, "write_run_metrics", lambda **_kwargs: None)

    with pytest.raises(RuntimeError, match="Failed to read existing composition.parquet"):
        run_finalization_module.finalize_run_outputs(
            cfg=_minimal_cfg(),
            run_root=run_root,
            run_root_str=str(run_root),
            cfg_path=run_root / "config.yaml",
            config_sha="abc123",
            seed=1,
            seeds={"stage_a": 1, "stage_b": 2, "solver": 3},
            chosen_solver="CBC",
            solver_time_limit_seconds=5.0,
            solver_threads=1,
            dense_arrays_version="0.0.0",
            dense_arrays_version_source="lock",
            plan_stats={},
            plan_order=[],
            plan_leaderboards={},
            plan_pools={},
            plan_items=[],
            inputs_manifest_entries={},
            library_source="build",
            library_artifact=None,
            library_build_rows=[],
            library_member_rows=[],
            composition_rows=[{"solution_id": "sol-1", "placement_index": 0, "tf": "TF1", "tfbs": "AAAA"}],
        )
