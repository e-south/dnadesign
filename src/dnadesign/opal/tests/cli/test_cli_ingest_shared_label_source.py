"""
CLI ingest contracts for shared observed-label sources.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.src.registries.transforms_y import register_transform_y


@register_transform_y("test_shared_scalar_labels")
def _test_shared_scalar_labels(df_tidy: pd.DataFrame, params: dict, *, ctx=None) -> pd.DataFrame:
    _unused = (params, ctx)
    return pd.DataFrame(
        {
            "id": df_tidy["id"].astype(str),
            "y": df_tidy["y_val"].map(lambda v: [float(v)]),
        }
    )


def _write_records(path: Path, *, ids: list[str], sequences: list[str], x_values: list[list[float]]) -> None:
    row_count = len(ids)
    if len(sequences) != row_count or len(x_values) != row_count:
        raise ValueError("ids, sequences, and x_values must have the same length")
    x_dim = len(x_values[0]) if x_values else 0
    pq.write_table(
        pa.table(
            {
                "id": pa.array(ids, type=pa.string()),
                "sequence": pa.array(sequences, type=pa.string()),
                "bio_type": pa.array(["dna"] * row_count, type=pa.string()),
                "alphabet": pa.array(["dna_4"] * row_count, type=pa.string()),
                "X": pa.array(x_values, type=pa.list_(pa.float32(), list_size=x_dim)),
            }
        ),
        path,
    )


def test_ingest_y_writes_usr_sidecar_label_source(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr" / "datasets"
    dataset_root = usr_root / "demo_candidates"
    dataset_root.mkdir(parents=True)
    records = dataset_root / "records.parquet"
    _write_records(records, ids=["a", "b"], sequences=["AAA", "BBB"], x_values=[[0.1], [0.2]])

    workdir = tmp_path / "campaign"
    workdir.mkdir()
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
transforms_y: {{ name: test_shared_scalar_labels, params: {{}} }}
model: {{ name: random_forest, params: {{ n_estimators: 5, random_state: 0 }} }}
objectives:
  - {{ name: scalar_identity_v1, params: {{}} }}
selection:
  name: top_n
  params: {{ top_k: 1, score_ref: scalar_identity_v1/scalar, objective_mode: maximize, tie_handling: competition_rank }}
""".strip()
    )
    labels = workdir / "labels.parquet"
    pd.DataFrame({"id": ["a", "b"], "y_val": [0.1, 0.2]}).to_parquet(labels, index=False)
    runner = CliRunner()
    app = _build()

    pre_validate = runner.invoke(app, ["--no-color", "validate", "-c", str(campaign)])
    assert pre_validate.exit_code == 0, pre_validate.stdout
    assert "usr_sidecar" in pre_validate.stdout
    assert "validation passed" in pre_validate.stdout.lower()

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
            "--unknown-sequences",
            "error",
            "--apply",
        ],
    )

    assert res.exit_code == 0, res.stdout
    sidecar = pd.read_parquet(dataset_root / "_opal" / "observed_labels.parquet")
    assert sidecar[["id", "observed_round", "batch_id", "y_space", "y_obs"]].to_dict(orient="records") == [
        {
            "id": "a",
            "observed_round": 0,
            "batch_id": "round_0",
            "y_space": "scalar_test",
            "y_obs": [0.1],
        },
        {
            "id": "b",
            "observed_round": 0,
            "batch_id": "round_0",
            "y_space": "scalar_test",
            "y_obs": [0.2],
        },
    ]
    records_after = pd.read_parquet(records)
    assert "opal__demo__label_hist" not in records_after.columns
    assert "opal__demo__y" not in records_after.columns

    post_validate = runner.invoke(app, ["--no-color", "validate", "-c", str(campaign)])
    assert post_validate.exit_code == 0, post_validate.stdout
    assert "label_count" in post_validate.stdout
    assert "2" in post_validate.stdout

    init = runner.invoke(app, ["--no-color", "init", "-c", str(campaign), "--json"])
    assert init.exit_code == 0, init.stdout
    records_after_init = pd.read_parquet(records)
    assert "opal__demo__label_hist" not in records_after_init.columns
    status = runner.invoke(app, ["--no-color", "status", "-c", str(campaign), "--json"])
    assert status.exit_code == 0, status.stdout
    status_json = json.loads(status.stdout)
    assert status_json["label_source"]["kind"] == "usr_sidecar"
    assert status_json["label_source"]["label_count"] == 2
    assert status_json["label_source"]["available_rounds"] == [0]
    assert status_json["writeback"]["prediction_records"] == "ledger_only"


def test_ingest_y_rejects_unknown_ids_for_shared_label_source(tmp_path: Path) -> None:
    usr_root = tmp_path / "usr" / "datasets"
    dataset_root = usr_root / "demo_candidates"
    dataset_root.mkdir(parents=True)
    records = dataset_root / "records.parquet"
    _write_records(records, ids=["a"], sequences=["AAA"], x_values=[[0.1]])

    workdir = tmp_path / "campaign"
    workdir.mkdir()
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
transforms_y: {{ name: test_shared_scalar_labels, params: {{}} }}
model: {{ name: random_forest, params: {{ n_estimators: 5, random_state: 0 }} }}
objectives:
  - {{ name: scalar_identity_v1, params: {{}} }}
selection:
  name: top_n
  params: {{ top_k: 1, score_ref: scalar_identity_v1/scalar, objective_mode: maximize, tie_handling: competition_rank }}
""".strip()
    )
    labels = workdir / "labels.parquet"
    pd.DataFrame({"id": ["missing"], "y_val": [0.2]}).to_parquet(labels, index=False)

    res = CliRunner().invoke(
        _build(),
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
    assert "fixed candidate universe" in res.output
    assert not (dataset_root / "_opal" / "observed_labels.parquet").exists()
    assert pd.read_parquet(records)["id"].tolist() == ["a"]
