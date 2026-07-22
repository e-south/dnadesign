"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/storage/test_run_artifact_paths.py

Tests deterministic, confined paths for immutable run-scoped artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from dnadesign.opal.src.core.utils import OpalError, file_sha256
from dnadesign.opal.src.storage.artifacts import (
    reserve_run_artifact_directory,
    run_artifact_slug,
    run_scoped_artifact_path,
    snapshot_run_artifacts,
)


def test_run_artifact_slug_is_path_safe_deterministic_and_collision_resistant() -> None:
    run_id = "r0-2026-07-16T14:53:58+00:00"

    first = run_artifact_slug(run_id)
    second = run_artifact_slug(run_id)

    assert first == second
    assert re.fullmatch(r"[a-z0-9-]+", first)
    assert first != run_artifact_slug("r0-2026-07-16T14/53/58+00/00")


def test_run_scoped_artifact_path_is_confined_beneath_exact_round(tmp_path: Path) -> None:
    round_dir = tmp_path / "outputs" / "rounds" / "round_0"
    run_id = "r0-2026-07-16T14:53:58+00:00"

    path = run_scoped_artifact_path(
        round_dir,
        run_id=run_id,
        artifact_key="labels/observed_events.parquet",
    )

    assert path.relative_to(round_dir) == (
        Path("run_artifacts") / run_artifact_slug(run_id) / "labels" / "observed_events.parquet"
    )


@pytest.mark.parametrize("artifact_key", ["", "/labels/events.parquet", "../events.parquet"])
def test_run_scoped_artifact_path_rejects_non_relative_artifact_keys(
    tmp_path: Path,
    artifact_key: str,
) -> None:
    with pytest.raises(OpalError, match="artifact key"):
        run_scoped_artifact_path(
            tmp_path / "round_0",
            run_id="run-0",
            artifact_key=artifact_key,
        )


def test_run_scoped_artifact_path_rejects_symlink_escape(tmp_path: Path) -> None:
    round_dir = tmp_path / "outputs" / "rounds" / "round_0"
    round_dir.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (round_dir / "run_artifacts").symlink_to(outside, target_is_directory=True)

    with pytest.raises(OpalError, match="outside its round directory"):
        run_scoped_artifact_path(
            round_dir,
            run_id="run-0",
            artifact_key="labels/observed_events.parquet",
        )


def test_reserve_run_artifact_directory_is_create_only(tmp_path: Path) -> None:
    round_dir = tmp_path / "outputs" / "rounds" / "round_0"

    reserved = reserve_run_artifact_directory(round_dir, run_id="run-0")

    assert reserved.is_dir()
    with pytest.raises(OpalError, match="already exists"):
        reserve_run_artifact_directory(round_dir, run_id="run-0")


def test_snapshot_run_artifacts_copies_latest_file_without_replacing_prior_snapshot(tmp_path: Path) -> None:
    round_dir = tmp_path / "outputs" / "rounds" / "round_0"
    source = round_dir / "selection" / "selections.parquet"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"first run")
    reserve_run_artifact_directory(round_dir, run_id="run-0")

    first = snapshot_run_artifacts(
        round_dir,
        run_id="run-0",
        artifacts={"selection/selections.parquet": (file_sha256(source), str(source))},
    )
    first_path = Path(first["selection/selections.parquet"][1])
    source.write_bytes(b"second run")
    reserve_run_artifact_directory(round_dir, run_id="run-1")
    second = snapshot_run_artifacts(
        round_dir,
        run_id="run-1",
        artifacts={"selection/selections.parquet": (file_sha256(source), str(source))},
    )

    assert first_path.read_bytes() == b"first run"
    assert Path(second["selection/selections.parquet"][1]).read_bytes() == b"second run"
    with pytest.raises(FileExistsError):
        snapshot_run_artifacts(
            round_dir,
            run_id="run-0",
            artifacts={"selection/selections.parquet": (file_sha256(source), str(source))},
        )
