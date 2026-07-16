"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/analysis/test_run_labels_used.py

Run-pinned observed-label analysis contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from dnadesign.opal.src.analysis.campaign import CampaignAnalysis
from dnadesign.opal.src.analysis.ledger import read_run_labels_used, read_runs
from dnadesign.opal.src.core.utils import OpalError, file_sha256
from dnadesign.opal.src.storage.artifacts import LABELS_USED_ARTIFACT_KEY, run_scoped_artifact_path
from dnadesign.opal.tests._cli_helpers import write_ledger


def _labels_used_artifact(outputs_dir: Path) -> tuple[Path, str]:
    path = run_scoped_artifact_path(
        outputs_dir / "rounds" / "round_0",
        run_id="run-0",
        artifact_key=LABELS_USED_ARTIFACT_KEY,
    )
    path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "run_id": ["run-0"],
            "as_of_round": [0],
            "observed_round": [0],
            "id": ["candidate-a"],
            "sequence": ["ACGT"],
            "y_obs": [[0.1] * 8],
            "src": ["training_snapshot"],
        }
    ).write_parquet(path)
    return path, file_sha256(path)


def _runs(*, path: Path, sha256: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "run_id": ["run-0"],
            "as_of_round": [0],
            "artifacts": [{"labels/labels_used.parquet": [sha256, str(path)]}],
        }
    )


def test_read_run_labels_used_verifies_run_pinned_snapshot(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    path, sha256 = _labels_used_artifact(outputs_dir)

    snapshot = read_run_labels_used(
        _runs(path=path, sha256=sha256),
        outputs_dir=outputs_dir,
        round_k=0,
        run_id="run-0",
    )

    assert snapshot.path == path.resolve()
    assert snapshot.sha256 == sha256
    assert snapshot.round_k == 0
    assert snapshot.run_id == "run-0"
    assert snapshot.frame.get_column("id").to_list() == ["candidate-a"]


def test_campaign_analysis_reads_selected_run_labels_without_mutable_ledger(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    path, sha256 = _labels_used_artifact(outputs_dir)
    analysis = CampaignAnalysis(data=SimpleNamespace(workspace=SimpleNamespace(outputs_dir=outputs_dir)))

    frame = analysis.read_run_labels_used(
        round_selector=0,
        run_id="run-0",
        runs_df=_runs(path=path, sha256=sha256),
    )

    assert frame.get_column("id").to_list() == ["candidate-a"]
    assert not (outputs_dir / "ledger" / "labels.parquet").exists()


def test_read_run_labels_used_accepts_serialized_run_ledger_artifact(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    outputs_dir = workdir / "outputs"
    path, sha256 = _labels_used_artifact(outputs_dir)
    write_ledger(
        workdir,
        run_id="run-0",
        round_index=0,
        artifact_paths_and_hashes={"labels/labels_used.parquet": (sha256, str(path.resolve()))},
    )

    snapshot = read_run_labels_used(
        read_runs(outputs_dir / "ledger" / "runs.parquet"),
        outputs_dir=outputs_dir,
        round_k=0,
        run_id="run-0",
    )

    assert snapshot.frame.get_column("id").to_list() == ["candidate-a"]


def test_read_run_labels_used_rejects_digest_drift(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    path, _sha256 = _labels_used_artifact(outputs_dir)

    with pytest.raises(OpalError, match="SHA-256"):
        read_run_labels_used(
            _runs(path=path, sha256="0" * 64),
            outputs_dir=outputs_dir,
            round_k=0,
            run_id="run-0",
        )


def test_read_run_labels_used_rejects_round_mutable_artifact_path(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    run_scoped_path, _sha256 = _labels_used_artifact(outputs_dir)
    mutable_path = outputs_dir / "rounds" / "round_0" / "labels" / "labels_used.parquet"
    mutable_path.parent.mkdir(parents=True)
    pl.read_parquet(run_scoped_path).write_parquet(mutable_path)

    with pytest.raises(OpalError, match="run-scoped path"):
        read_run_labels_used(
            _runs(path=mutable_path, sha256=file_sha256(mutable_path)),
            outputs_dir=outputs_dir,
            round_k=0,
            run_id="run-0",
        )


def test_read_run_labels_used_rejects_noncanonical_candidate_ids(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    path, _sha256 = _labels_used_artifact(outputs_dir)
    pl.read_parquet(path).with_columns(pl.lit(" candidate-a ").alias("id")).write_parquet(path)

    with pytest.raises(OpalError, match="canonical"):
        read_run_labels_used(
            _runs(path=path, sha256=file_sha256(path)),
            outputs_dir=outputs_dir,
            round_k=0,
            run_id="run-0",
        )


def test_read_run_labels_used_rejects_artifact_outside_round_scope(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    path = tmp_path / "outside" / "labels_used.parquet"
    path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "run_id": ["run-0"],
            "as_of_round": [0],
            "observed_round": [0],
            "id": ["candidate-a"],
            "y_obs": [[0.1] * 8],
        }
    ).write_parquet(path)

    with pytest.raises(OpalError, match="outside its round directory"):
        read_run_labels_used(
            _runs(path=path, sha256=file_sha256(path)),
            outputs_dir=outputs_dir,
            round_k=0,
            run_id="run-0",
        )
