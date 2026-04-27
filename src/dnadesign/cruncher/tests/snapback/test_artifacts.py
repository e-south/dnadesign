"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/snapback/test_artifacts.py

Snapback artifact layout contracts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.snapback.artifacts import build_run_dir, build_solve_run_dir, solve_hit_run_dir


def test_build_run_dir_nests_spec_name_and_design_id_under_workspace_root(tmp_path: Path) -> None:
    run_dir = build_run_dir(
        workspace_root=tmp_path,
        run_root=Path("outputs/design"),
        spec_name="demo_snapback",
        snapback_design_id="abc123def456",
    )

    assert run_dir == tmp_path / "outputs" / "design"


def test_build_solve_run_dir_nests_solve_id_under_workspace_root(tmp_path: Path) -> None:
    run_dir = build_solve_run_dir(
        workspace_root=tmp_path,
        run_root=Path("outputs/solve"),
        snapback_solve_id="5b7d04e72af4",
    )

    assert run_dir == tmp_path / "outputs" / "solve"


def test_solve_hit_run_dir_uses_stable_rank_path_under_analysis(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "solve"

    assert solve_hit_run_dir(run_dir, rank=1) == run_dir / "analysis" / "materialized_hits" / "hit_01"
