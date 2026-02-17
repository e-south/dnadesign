"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/test_cli_pressure_edge_matrix.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.src.storage.data_access import RecordsStore

from ._cli_helpers import write_campaign_yaml


def _write_records(path: Path) -> None:
    df = pd.DataFrame(
        {
            "id": ["a", "b"],
            "sequence": ["AAA", "BBB"],
            "bio_type": ["dna", "dna"],
            "alphabet": ["dna_4", "dna_4"],
            "X": [[0.1, 0.2], [0.2, 0.3]],
        }
    )
    df.to_parquet(path, index=False)


def _write_labels_with_extras(
    path: Path,
    *,
    seqs: list[str],
    ys: list[float],
    as_parquet: bool = False,
    include_x: bool = True,
    include_bio_type: bool = True,
    include_alphabet: bool = True,
) -> None:
    payload = {
        "sequence": seqs,
        "y": ys,
    }
    if include_bio_type:
        payload["bio_type"] = ["dna"] * len(seqs)
    if include_alphabet:
        payload["alphabet"] = ["dna_4"] * len(seqs)
    if include_x:
        payload["X"] = [[0.1, 0.2]] * len(seqs)
    df = pd.DataFrame(payload)
    if as_parquet:
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def test_cli_pressure_unknown_sequences_create_rows(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    _write_records(records)

    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(
        campaign,
        workdir=workdir,
        records_path=records,
        transforms_y_name="scalar_from_table_v1",
        transforms_y_params={"sequence_column": "sequence", "y_column": "y", "id_column": "id"},
        objective_name="scalar_identity_v1",
        objective_params={},
        y_expected_length=1,
        model_params={"n_estimators": 5, "random_state": 0, "oob_score": False},
        selection_params={"top_k": 1},
    )

    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "init", "-c", str(campaign)])
    assert res.exit_code == 0, res.stdout

    labels = workdir / "labels_r0.parquet"
    _write_labels_with_extras(labels, seqs=["AAA", "EEE"], ys=[0.2, 0.9], as_parquet=True)
    res = runner.invoke(
        app,
        [
            "--no-color",
            "ingest-y",
            "-c",
            str(campaign),
            "--round",
            "0",
            "--csv",
            str(labels),
            "--apply",
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert "unknown sequences" in res.stdout.lower() or "new sequences will be created" in res.stdout.lower()

    df = pd.read_parquet(records)
    assert "EEE" in df["sequence"].astype(str).tolist()
    new_row = df.loc[df["sequence"].astype(str) == "EEE"].iloc[0]
    assert pd.notna(new_row["id"])

    store = RecordsStore(
        kind="local",
        records_path=records,
        campaign_slug="demo",
        x_col="X",
        y_col="Y",
        x_transform_name="identity",
        x_transform_params={},
    )
    lh = store.label_hist_col()
    hist = store._normalize_hist_cell(new_row[lh])
    assert any(e.get("observed_round") == 0 for e in hist)


def test_cli_pressure_unknown_sequences_drop_skips_rows(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    _write_records(records)

    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(
        campaign,
        workdir=workdir,
        records_path=records,
        transforms_y_name="scalar_from_table_v1",
        transforms_y_params={"sequence_column": "sequence", "y_column": "y", "id_column": "id"},
        objective_name="scalar_identity_v1",
        objective_params={},
        y_expected_length=1,
        model_params={"n_estimators": 5, "random_state": 0, "oob_score": False},
        selection_params={"top_k": 1},
    )

    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "init", "-c", str(campaign)])
    assert res.exit_code == 0, res.stdout

    labels = workdir / "labels_r0.csv"
    _write_labels_with_extras(labels, seqs=["AAA", "EEE"], ys=[0.2, 0.9], include_x=False, as_parquet=False)
    res = runner.invoke(
        app,
        [
            "--no-color",
            "ingest-y",
            "-c",
            str(campaign),
            "--round",
            "0",
            "--csv",
            str(labels),
            "--apply",
            "--unknown-sequences",
            "drop",
        ],
    )
    assert res.exit_code == 0, res.stdout

    df = pd.read_parquet(records)
    assert "EEE" not in df["sequence"].astype(str).tolist()


def test_cli_pressure_unknown_sequences_missing_x_auto_drop(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    _write_records(records)

    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(
        campaign,
        workdir=workdir,
        records_path=records,
        transforms_y_name="scalar_from_table_v1",
        transforms_y_params={"sequence_column": "sequence", "y_column": "y"},
        objective_name="scalar_identity_v1",
        objective_params={},
        y_expected_length=1,
        model_params={"n_estimators": 5, "random_state": 0, "oob_score": False},
        selection_params={"top_k": 1},
    )

    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "init", "-c", str(campaign)])
    assert res.exit_code == 0, res.stdout

    labels = workdir / "labels_r0.csv"
    _write_labels_with_extras(labels, seqs=["AAA", "EEE"], ys=[0.2, 0.9], include_x=False, as_parquet=False)
    res = runner.invoke(
        app,
        [
            "--no-color",
            "ingest-y",
            "-c",
            str(campaign),
            "--round",
            "0",
            "--csv",
            str(labels),
            "--apply",
        ],
    )
    assert res.exit_code == 0, res.stdout

    df = pd.read_parquet(records)
    assert "EEE" not in df["sequence"].astype(str).tolist()


def test_cli_pressure_sequence_exists_with_new_id_errors(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    _write_records(records)

    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(
        campaign,
        workdir=workdir,
        records_path=records,
        transforms_y_name="scalar_from_table_v1",
        transforms_y_params={"sequence_column": "sequence", "y_column": "y", "id_column": "id"},
        objective_name="scalar_identity_v1",
        objective_params={},
        y_expected_length=1,
        model_params={"n_estimators": 5, "random_state": 0, "oob_score": False},
        selection_params={"top_k": 1},
    )

    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "init", "-c", str(campaign)])
    assert res.exit_code == 0, res.stdout

    labels = workdir / "labels_r0.csv"
    df = pd.DataFrame({"id": ["zzz"], "sequence": ["AAA"], "y": [0.2]})
    df.to_csv(labels, index=False)

    res = runner.invoke(
        app,
        [
            "--no-color",
            "ingest-y",
            "-c",
            str(campaign),
            "--round",
            "0",
            "--csv",
            str(labels),
            "--apply",
        ],
    )
    assert res.exit_code != 0
    assert "sequence" in (res.stdout + res.stderr).lower()


def test_cli_pressure_missing_required_defaults_prompt(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    _write_records(records)

    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(
        campaign,
        workdir=workdir,
        records_path=records,
        transforms_y_name="scalar_from_table_v1",
        transforms_y_params={"sequence_column": "sequence", "y_column": "y"},
        objective_name="scalar_identity_v1",
        objective_params={},
        y_expected_length=1,
        model_params={"n_estimators": 5, "random_state": 0, "oob_score": False},
        selection_params={"top_k": 1},
    )

    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "init", "-c", str(campaign)])
    assert res.exit_code == 0, res.stdout

    labels = workdir / "labels_r0.parquet"
    _write_labels_with_extras(
        labels,
        seqs=["AAA", "EEE"],
        ys=[0.2, 0.9],
        as_parquet=True,
        include_x=True,
        include_bio_type=False,
        include_alphabet=False,
    )
    res = runner.invoke(
        app,
        [
            "--no-color",
            "ingest-y",
            "-c",
            str(campaign),
            "--round",
            "0",
            "--csv",
            str(labels),
            "--apply",
            "--infer-missing-required",
        ],
    )
    assert res.exit_code == 0, res.stdout

    df = pd.read_parquet(records)
    new_row = df.loc[df["sequence"].astype(str) == "EEE"].iloc[0]
    assert str(new_row["bio_type"]) == "dna"
    assert str(new_row["alphabet"]) == "dna_4"


def test_cli_pressure_duplicate_sequences_require_ids(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    df = pd.DataFrame(
        {
            "id": ["a", "b"],
            "sequence": ["AAA", "AAA"],
            "bio_type": ["dna", "dna"],
            "alphabet": ["dna_4", "dna_4"],
            "X": [[0.1, 0.2], [0.2, 0.3]],
        }
    )
    df.to_parquet(records, index=False)

    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(
        campaign,
        workdir=workdir,
        records_path=records,
        transforms_y_name="scalar_from_table_v1",
        transforms_y_params={"sequence_column": "sequence", "y_column": "y"},
        objective_name="scalar_identity_v1",
        objective_params={},
        y_expected_length=1,
        model_params={"n_estimators": 5, "random_state": 0, "oob_score": False},
        selection_params={"top_k": 1},
    )

    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "init", "-c", str(campaign)])
    assert res.exit_code == 0, res.stdout

    labels = workdir / "labels_r0.csv"
    pd.DataFrame({"sequence": ["AAA"], "y": [0.2]}).to_csv(labels, index=False)

    res = runner.invoke(
        app,
        [
            "--no-color",
            "ingest-y",
            "-c",
            str(campaign),
            "--round",
            "0",
            "--csv",
            str(labels),
            "--apply",
        ],
    )
    assert res.exit_code != 0
    assert "duplicate sequences" in (res.stdout + res.stderr).lower()


def test_cli_pressure_if_exists_skip_keeps_history(tmp_path: Path) -> None:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    _write_records(records)

    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(
        campaign,
        workdir=workdir,
        records_path=records,
        transforms_y_name="scalar_from_table_v1",
        transforms_y_params={"sequence_column": "sequence", "y_column": "y"},
        objective_name="scalar_identity_v1",
        objective_params={},
        y_expected_length=1,
        model_params={"n_estimators": 5, "random_state": 0, "oob_score": False},
        selection_params={"top_k": 1},
    )

    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "init", "-c", str(campaign)])
    assert res.exit_code == 0, res.stdout

    labels0 = workdir / "labels_r0.csv"
    _write_labels_with_extras(labels0, seqs=["AAA"], ys=[0.2])
    res = runner.invoke(
        app,
        [
            "--no-color",
            "ingest-y",
            "-c",
            str(campaign),
            "--round",
            "0",
            "--csv",
            str(labels0),
            "--apply",
        ],
    )
    assert res.exit_code == 0, res.stdout

    labels_dup = workdir / "labels_r0_dup.csv"
    _write_labels_with_extras(labels_dup, seqs=["AAA"], ys=[0.9])
    res = runner.invoke(
        app,
        [
            "--no-color",
            "ingest-y",
            "-c",
            str(campaign),
            "--round",
            "0",
            "--csv",
            str(labels_dup),
            "--if-exists",
            "skip",
            "--apply",
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert "labels skipped" in res.stdout.lower()

    df = pd.read_parquet(records)
    store = RecordsStore(
        kind="local",
        records_path=records,
        campaign_slug="demo",
        x_col="X",
        y_col="Y",
        x_transform_name="identity",
        x_transform_params={},
    )
    lh = store.label_hist_col()
    hist = store._normalize_hist_cell(df.loc[df["id"] == "a", lh].iloc[0])
    assert len(hist) == 1
