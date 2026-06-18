"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/cli/test_cli_guide_next.py

State-aware CLI tests for guided next-step recommendations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.tests._cli_helpers import write_campaign_yaml, write_records, write_state


def _setup_workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records, include_opal_cols=True, slug="demo")
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records, slug="demo")
    return workdir, campaign, records


def _setup_usr_sidecar_workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    usr_root = tmp_path / "usr" / "datasets"
    dataset_root = usr_root / "demo_candidates"
    dataset_root.mkdir(parents=True)
    records = dataset_root / "records.parquet"
    pd.DataFrame(
        {
            "id": ["a", "b"],
            "sequence": ["AAA", "BBB"],
            "bio_type": ["dna", "dna"],
            "alphabet": ["dna_4", "dna_4"],
            "X": [[0.1], [0.2]],
        }
    ).to_parquet(records, index=False)
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True)
    campaign = workdir / "campaign.yaml"
    campaign.write_text(
        f"""
campaign:
  name: Demo
  slug: demo
  workdir: "{workdir}"
data:
  location: {{ kind: usr, path: "{usr_root}", dataset: demo_candidates }}
  x_column_name: X
  y_column_name: opal__demo__y
  y_expected_length: 1
labels:
  source:
    kind: usr_sidecar
    dataset: demo_candidates
    path: _opal/observed_labels.parquet
  y_space: scalar_test
writeback:
  prediction_records: ledger_only
transforms_x: {{ name: identity, params: {{}} }}
transforms_y: {{ name: scalar_from_table_v1, params: {{ y_column: y }} }}
model: {{ name: random_forest, params: {{ n_estimators: 5, random_state: 0 }} }}
objectives:
  - {{ name: scalar_identity_v1, params: {{}} }}
selection:
  name: top_n
  params: {{ top_k: 1, score_ref: scalar_identity_v1/scalar, objective_mode: maximize, tie_handling: competition_rank }}
""".strip(),
        encoding="utf-8",
    )
    return workdir, campaign, dataset_root / "_opal" / "observed_labels.parquet"


def _setup_missing_usr_sidecar_workspace(tmp_path: Path) -> tuple[Path, Path, Path]:
    usr_root = tmp_path / "usr" / "datasets"
    dataset_root = usr_root / "demo_candidates"
    dataset_root.mkdir(parents=True)
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True)
    campaign = workdir / "campaign.yaml"
    campaign.write_text(
        f"""
campaign:
  name: Demo
  slug: demo
  workdir: "{workdir}"
data:
  location: {{ kind: usr, path: "{usr_root}", dataset: demo_candidates }}
  x_column_name: X
  y_column_name: opal__demo__y
  y_expected_length: 1
labels:
  source:
    kind: usr_sidecar
    dataset: demo_candidates
    path: _opal/observed_labels.parquet
  y_space: scalar_test
writeback:
  prediction_records: ledger_only
transforms_x: {{ name: identity, params: {{}} }}
transforms_y: {{ name: scalar_from_table_v1, params: {{ y_column: y }} }}
model: {{ name: random_forest, params: {{ n_estimators: 5, random_state: 0 }} }}
objectives:
  - {{ name: scalar_identity_v1, params: {{}} }}
selection:
  name: top_n
  params: {{ top_k: 1, score_ref: scalar_identity_v1/scalar, objective_mode: maximize, tie_handling: competition_rank }}
""".strip(),
        encoding="utf-8",
    )
    return workdir, campaign, dataset_root / "records.parquet"


def _write_round0_label(records_path: Path) -> None:
    df = pd.read_parquet(records_path)
    lh_col = "opal__demo__label_hist"
    df.at[0, lh_col] = [
        {
            "kind": "label",
            "observed_round": 0,
            "ts": "2026-01-01T00:00:00Z",
            "src": "ingest_y",
            "y_obs": {
                "value": [0.1, 0.0, 0.0, 1.0, 0.2, 0.3, 0.1, 2.0],
                "dtype": "vector",
                "schema": {"length": 8},
            },
        }
    ]
    df.to_parquet(records_path, index=False)


def test_guide_next_recommends_validate_before_init_when_state_missing(tmp_path: Path) -> None:
    _, campaign, records = _setup_workspace(tmp_path)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "guide", "next", "-c", str(campaign), "--json"])
    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)
    assert out["stage"] == "validate"
    assert "opal validate -c" in out["next_commands"][0]
    assert "opal init -c" in out["next_commands"][1]
    assert out["records_path"] == str(records)
    assert out["records_exists"] is True


def test_guide_next_reports_missing_candidate_table_before_init(tmp_path: Path) -> None:
    _, campaign, records = _setup_missing_usr_sidecar_workspace(tmp_path)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "guide", "next", "-c", str(campaign), "--json"])
    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)
    assert out["stage"] == "candidate_table"
    assert out["records_path"] == str(records)
    assert out["records_exists"] is False
    assert out["label_source"]["kind"] == "usr_sidecar"


def test_status_reports_missing_candidate_table_without_state_json(tmp_path: Path) -> None:
    _, campaign, records = _setup_missing_usr_sidecar_workspace(tmp_path)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "status", "-c", str(campaign), "--json"])
    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)
    assert out["state_exists"] is False
    assert out["data"]["records_path"] == str(records)
    assert out["data"]["records_exists"] is False
    assert out["label_source"]["kind"] == "usr_sidecar"


def test_status_reads_label_status_columns_without_loading_x(tmp_path: Path, monkeypatch) -> None:
    _, campaign, _ = _setup_workspace(tmp_path)
    from dnadesign.opal.src.storage import records_io

    calls: list[tuple[str, ...] | None] = []
    original = records_io.read_parquet_df

    def spy_read_parquet_df(path, *, columns=None, dtype_backend=None):
        calls.append(tuple(columns) if columns is not None else None)
        assert columns is not None
        assert "X" not in columns
        return original(path, columns=columns, dtype_backend=dtype_backend)

    monkeypatch.setattr(records_io, "read_parquet_df", spy_read_parquet_df)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "status", "-c", str(campaign), "--json"])

    assert res.exit_code == 0, res.stdout
    assert calls == [("id", "Y", "opal__demo__label_hist")]


def test_guide_next_before_init_does_not_load_candidate_x(tmp_path: Path, monkeypatch) -> None:
    _, campaign, _ = _setup_workspace(tmp_path)
    from dnadesign.opal.src.storage import records_io

    calls: list[tuple[str, ...] | None] = []
    original = records_io.read_parquet_df

    def spy_read_parquet_df(path, *, columns=None, dtype_backend=None):
        calls.append(tuple(columns) if columns is not None else None)
        assert columns is not None
        assert "X" not in columns
        return original(path, columns=columns, dtype_backend=dtype_backend)

    monkeypatch.setattr(records_io, "read_parquet_df", spy_read_parquet_df)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "guide", "next", "-c", str(campaign), "--json"])

    assert res.exit_code == 0, res.stdout
    assert calls == []


def test_guide_next_recommends_ingest_when_state_exists_but_round_has_no_labels(tmp_path: Path) -> None:
    workdir, campaign, records = _setup_workspace(tmp_path)
    write_state(workdir, records_path=records, run_id="seed", round_index=0)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(
        app,
        ["--no-color", "guide", "next", "-c", str(campaign), "--labels-as-of", "0", "--json"],
    )
    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)
    assert out["stage"] == "ingest"
    assert "--observed-round 0" in out["next_commands"][0]


def test_guide_next_recommends_run_after_labels_exist(tmp_path: Path) -> None:
    workdir, campaign, records = _setup_workspace(tmp_path)
    _ = workdir
    app = _build()
    runner = CliRunner()

    init_res = runner.invoke(app, ["--no-color", "init", "-c", str(campaign)])
    assert init_res.exit_code == 0, init_res.stdout
    _write_round0_label(records)

    res = runner.invoke(
        app,
        ["--no-color", "guide", "next", "-c", str(campaign), "--labels-as-of", "0", "--json"],
    )
    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)
    assert out["stage"] == "run"
    assert "--labels-as-of 0" in out["next_commands"][0]


def test_guide_next_counts_shared_usr_sidecar_labels(tmp_path: Path) -> None:
    _, campaign, sidecar = _setup_usr_sidecar_workspace(tmp_path)
    sidecar.parent.mkdir(parents=True)
    pd.DataFrame(
        {
            "id": ["a"],
            "observed_round": [0],
            "batch_id": ["round_0"],
            "y_space": ["scalar_test"],
            "y_obs": [[0.1]],
        }
    ).to_parquet(sidecar, index=False)
    app = _build()
    runner = CliRunner()

    init_res = runner.invoke(app, ["--no-color", "init", "-c", str(campaign)])
    assert init_res.exit_code == 0, init_res.stdout

    res = runner.invoke(
        app,
        ["--no-color", "guide", "next", "-c", str(campaign), "--labels-as-of", "0", "--json"],
    )
    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)
    assert out["stage"] == "run"
    assert out["labels_in_observed_round"] == 1


def test_guide_next_recommends_verify_after_round_exists(tmp_path: Path) -> None:
    workdir, campaign, records = _setup_workspace(tmp_path)
    _write_round0_label(records)
    write_state(workdir, records_path=records, run_id="r0", round_index=0)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(
        app,
        ["--no-color", "guide", "next", "-c", str(campaign), "--labels-as-of", "0", "--json"],
    )
    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)
    assert out["stage"] == "post_run"
    assert any("verify-outputs" in cmd for cmd in out["next_commands"])
