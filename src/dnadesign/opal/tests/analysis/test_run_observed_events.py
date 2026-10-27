"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/analysis/test_run_observed_events.py

Run-pinned observed-event analysis contract tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from dnadesign.opal.src.analysis.campaign import CampaignAnalysis
from dnadesign.opal.src.analysis.ledger import read_run_observed_events
from dnadesign.opal.src.core.utils import OpalError, file_sha256
from dnadesign.opal.src.storage.artifacts import OBSERVED_EVENTS_ARTIFACT_KEY, run_scoped_artifact_path


def _observed_events_artifact(outputs_dir: Path) -> tuple[Path, str]:
    path = run_scoped_artifact_path(
        outputs_dir / "rounds" / "round_1",
        run_id="run-1",
        artifact_key=OBSERVED_EVENTS_ARTIFACT_KEY,
    )
    path.parent.mkdir(parents=True)
    pl.DataFrame(
        {
            "run_id": ["run-1", "run-1", "run-1"],
            "as_of_round": [1, 1, 1],
            "id": ["candidate-a", "candidate-a", "candidate-b"],
            "display_label": ["Candidate A", "Candidate A", None],
            "sequence": ["ACGT", "ACGT", "TGCA"],
            "observed_round": [0, 1, 1],
            "batch_id": ["batch-0", "batch-1", "batch-1"],
            "y_space": ["response_window_vec8_v1"] * 3,
            "y_obs": [[0.1] * 8, [0.2] * 8, [0.3] * 8],
            "label_source_kind": ["usr_sidecar"] * 3,
        }
    ).write_parquet(path)
    return path, file_sha256(path)


def _runs(*, path: Path, sha256: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "run_id": ["run-1"],
            "as_of_round": [1],
            "artifacts": [{OBSERVED_EVENTS_ARTIFACT_KEY: [sha256, str(path)]}],
        }
    )


def test_read_run_observed_events_verifies_repeated_candidate_events(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    path, sha256 = _observed_events_artifact(outputs_dir)

    snapshot = read_run_observed_events(
        _runs(path=path, sha256=sha256),
        outputs_dir=outputs_dir,
        round_k=1,
        run_id="run-1",
    )

    assert snapshot.path == path.resolve()
    assert snapshot.sha256 == sha256
    assert snapshot.frame.get_column("id").to_list() == [
        "candidate-a",
        "candidate-a",
        "candidate-b",
    ]
    assert snapshot.frame.get_column("batch_id").to_list() == ["batch-0", "batch-1", "batch-1"]
    assert snapshot.frame.get_column("display_label").to_list() == ["Candidate A", "Candidate A", None]


def test_campaign_analysis_reads_run_observed_events_without_mutable_label_ledger(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    path, sha256 = _observed_events_artifact(outputs_dir)
    analysis = CampaignAnalysis(data=SimpleNamespace(workspace=SimpleNamespace(outputs_dir=outputs_dir)))

    frame = analysis.read_run_observed_events(
        round_selector=1,
        run_id="run-1",
        runs_df=_runs(path=path, sha256=sha256),
    )

    assert frame.get_column("batch_id").to_list() == ["batch-0", "batch-1", "batch-1"]
    assert not (outputs_dir / "ledger" / "labels.parquet").exists()


def test_read_run_observed_events_rejects_future_events(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    path, _sha256 = _observed_events_artifact(outputs_dir)
    pl.read_parquet(path).with_columns(
        pl.when(pl.col("id") == "candidate-b")
        .then(pl.lit(2))
        .otherwise(pl.col("observed_round"))
        .alias("observed_round")
    ).write_parquet(path)

    with pytest.raises(OpalError, match="observed after the run scope"):
        read_run_observed_events(
            _runs(path=path, sha256=file_sha256(path)),
            outputs_dir=outputs_dir,
            round_k=1,
            run_id="run-1",
        )


@pytest.mark.parametrize("display_label", ["", " Candidate B "])
def test_read_run_observed_events_rejects_malformed_display_labels(
    tmp_path: Path,
    display_label: str,
) -> None:
    outputs_dir = tmp_path / "outputs"
    path, _sha256 = _observed_events_artifact(outputs_dir)
    pl.read_parquet(path).with_columns(
        pl.when(pl.col("id") == "candidate-b")
        .then(pl.lit(display_label))
        .otherwise(pl.col("display_label"))
        .alias("display_label")
    ).write_parquet(path)

    with pytest.raises(OpalError, match="display_label.*canonical non-blank"):
        read_run_observed_events(
            _runs(path=path, sha256=file_sha256(path)),
            outputs_dir=outputs_dir,
            round_k=1,
            run_id="run-1",
        )


def test_read_run_observed_events_requires_display_label_projection(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    path, _sha256 = _observed_events_artifact(outputs_dir)
    pl.read_parquet(path).drop("display_label").write_parquet(path)

    with pytest.raises(OpalError, match="display_label"):
        read_run_observed_events(
            _runs(path=path, sha256=file_sha256(path)),
            outputs_dir=outputs_dir,
            round_k=1,
            run_id="run-1",
        )


def test_read_run_observed_events_rejects_non_string_display_labels(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    path, _sha256 = _observed_events_artifact(outputs_dir)
    pl.read_parquet(path).with_columns(pl.lit(7).alias("display_label")).write_parquet(path)

    with pytest.raises(OpalError, match="display_label.*canonical non-blank strings"):
        read_run_observed_events(
            _runs(path=path, sha256=file_sha256(path)),
            outputs_dir=outputs_dir,
            round_k=1,
            run_id="run-1",
        )


@pytest.mark.parametrize("batch_id", [None, "", " batch-1 "])
def test_read_run_observed_events_rejects_missing_shared_batch_identity(
    tmp_path: Path,
    batch_id: str | None,
) -> None:
    outputs_dir = tmp_path / "outputs"
    path, _sha256 = _observed_events_artifact(outputs_dir)
    pl.read_parquet(path).with_columns(
        pl.when(pl.col("id") == "candidate-b")
        .then(pl.lit(batch_id, dtype=pl.String))
        .otherwise(pl.col("batch_id"))
        .alias("batch_id")
    ).write_parquet(path)

    with pytest.raises(OpalError, match="usr_sidecar.*batch_id"):
        read_run_observed_events(
            _runs(path=path, sha256=file_sha256(path)),
            outputs_dir=outputs_dir,
            round_k=1,
            run_id="run-1",
        )


def test_read_run_observed_events_allows_null_batch_for_campaign_history(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    path, _sha256 = _observed_events_artifact(outputs_dir)
    pl.read_parquet(path).with_columns(
        pl.lit(None, dtype=pl.String).alias("batch_id"),
        pl.lit(None, dtype=pl.String).alias("display_label"),
        pl.lit(None, dtype=pl.String).alias("y_space"),
        pl.lit("campaign_history").alias("label_source_kind"),
    ).write_parquet(path)

    snapshot = read_run_observed_events(
        _runs(path=path, sha256=file_sha256(path)),
        outputs_dir=outputs_dir,
        round_k=1,
        run_id="run-1",
    )

    assert snapshot.frame.get_column("batch_id").null_count() == 3
    assert snapshot.frame.get_column("display_label").null_count() == 3
    assert snapshot.frame.get_column("y_space").null_count() == 3


def test_read_run_observed_events_rejects_unknown_label_source_kind(tmp_path: Path) -> None:
    outputs_dir = tmp_path / "outputs"
    path, _sha256 = _observed_events_artifact(outputs_dir)
    pl.read_parquet(path).with_columns(pl.lit("unregistered_source").alias("label_source_kind")).write_parquet(path)

    with pytest.raises(OpalError, match="label-source kind.*supported"):
        read_run_observed_events(
            _runs(path=path, sha256=file_sha256(path)),
            outputs_dir=outputs_dir,
            round_k=1,
            run_id="run-1",
        )
