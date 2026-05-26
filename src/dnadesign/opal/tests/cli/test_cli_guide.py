"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/cli/test_cli_guide.py

CLI tests for guided workflow runbook generation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.tests._cli_helpers import write_campaign_yaml, write_records


def _setup_workspace(tmp_path: Path) -> tuple[Path, Path]:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records, include_opal_cols=True, slug="demo")
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records, slug="demo")
    return workdir, campaign


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
    return campaign, records, dataset_root / "_opal" / "observed_labels.parquet"


def test_guide_json_includes_campaign_plugins_steps_and_doc_pointers(tmp_path: Path) -> None:
    _, campaign = _setup_workspace(tmp_path)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "guide", "-c", str(campaign), "--format", "json"])
    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)

    assert out["campaign"]["slug"] == "demo"
    assert out["plugins"]["model"]["name"] == "random_forest"
    assert out["plugins"]["selection"]["name"] == "top_n"
    assert out["workflow_key"] == "rf_sfxi_topn"
    assert any("opal run -c" in str(step["command"]) for step in out["steps"])
    assert out["steps"][0]["title"] == "Validate schema and plugin wiring"
    assert "opal validate -c" in out["steps"][0]["command"]
    assert out["steps"][1]["title"] == "Initialize campaign workspace"
    assert "opal init -c" in out["steps"][1]["command"]
    assert "docs/plugins/objectives/sfxi.md" in out["learn_more"]["docs"]
    assert "src/dnadesign/opal/src/models/random_forest.py" in out["learn_more"]["source"]
    assert "src/dnadesign/opal/src/runtime/round/stages/scoring.py" in out["learn_more"]["source"]


def test_guide_json_includes_spop_objective_pointers_without_sfxi_warning(tmp_path: Path) -> None:
    workdir, campaign = _setup_workspace(tmp_path)
    records = workdir / "records.parquet"
    write_campaign_yaml(
        campaign,
        workdir=workdir,
        records_path=records,
        transforms_y_name="scalar_from_table_v1",
        objective_name="spop_v1",
        objective_params={},
        y_expected_length=1,
        selection_params={
            "score_ref": "spop_v1/spop",
            "objective_mode": "maximize",
        },
    )
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "guide", "-c", str(campaign), "--format", "json"])
    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)

    assert "docs/plugins/objectives/spop.md" in out["learn_more"]["docs"]
    assert "src/dnadesign/opal/src/objectives/spop_v1.py" in out["learn_more"]["source"]
    assert any("SPOP campaigns require scalar Y" in item for item in out["common_errors"])
    assert not any("SFXI min_n" in item for item in out["common_errors"])


def test_guide_json_uses_usr_records_and_shared_label_source_sidecar(tmp_path: Path) -> None:
    campaign, records, sidecar = _setup_usr_sidecar_workspace(tmp_path)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "guide", "-c", str(campaign), "--format", "json"])
    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)

    assert out["campaign"]["records_path"] == str(records.resolve())
    assert out["campaign"]["label_source"]["kind"] == "usr_sidecar"
    assert out["campaign"]["label_source"]["path"] == str(sidecar.resolve())
    ingest_step = next(step for step in out["steps"] if step["title"] == "Ingest observed labels")
    assert "--unknown-sequences error" in ingest_step["command"]
    assert "shared USR observed-label sidecar" in ingest_step["why"]
    assert str(sidecar.resolve()) in ingest_step["writes"]
    assert "records.parquet" not in ingest_step["writes"]


def test_guide_markdown_contains_round_semantics_and_commands(tmp_path: Path) -> None:
    _, campaign = _setup_workspace(tmp_path)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "guide", "-c", str(campaign), "--format", "markdown"])
    assert res.exit_code == 0, res.stdout
    text = res.stdout

    assert "## Guided Workflow" in text
    assert "--observed-round" in text
    assert "--labels-as-of" in text
    assert "opal init -c" in text
    assert "opal run -c" in text
