"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/tests/cli/test_cli_workflows.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.tests._cli_helpers import write_campaign_yaml, write_records, write_records_with_x_values


def _plain_output(text: str) -> str:
    return " ".join(str(text).split())


def _setup_workspace(tmp_path: Path, *, include_opal_cols: bool = False) -> tuple[Path, Path, Path]:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records, include_opal_cols=include_opal_cols)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)
    return workdir, campaign, records


def test_init_validate_explain_cli(tmp_path: Path) -> None:
    workdir, campaign, _ = _setup_workspace(tmp_path, include_opal_cols=True)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "init", "-c", str(campaign)])
    assert res.exit_code == 0, res.stdout
    assert (workdir / "state.json").exists()
    assert not (workdir / ".opal" / "config").exists()
    assert (workdir / "outputs").exists()
    assert (workdir / "outputs" / "ledger").exists()
    assert (workdir / "outputs" / "rounds").exists()
    assert not (workdir / "inputs").exists()

    res = runner.invoke(app, ["--no-color", "validate", "-c", str(campaign)])
    assert res.exit_code == 0, res.stdout
    assert "validation passed" in res.stdout.lower()

    res = runner.invoke(app, ["--no-color", "explain", "-c", str(campaign), "--round", "0", "--json"])
    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)
    assert out["round_index"] == 0


def test_init_does_not_materialize_candidate_x_when_label_history_exists(tmp_path: Path, monkeypatch) -> None:
    workdir, campaign, _ = _setup_workspace(tmp_path, include_opal_cols=True)
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

    res = runner.invoke(app, ["--no-color", "init", "-c", str(campaign), "--json"])

    assert res.exit_code == 0, res.stdout
    assert calls == [("id", "bio_type", "alphabet")]
    state = json.loads((workdir / "state.json").read_text())
    assert "records_sha256" not in state["data_location"]
    assert state["data_location"]["records_fingerprint_kind"] == "file_metadata"
    assert state["data_location"]["records_size_bytes"] > 0


def test_validate_does_not_materialize_candidate_x_in_pandas(tmp_path: Path, monkeypatch) -> None:
    _, campaign, _ = _setup_workspace(tmp_path, include_opal_cols=True)
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

    res = runner.invoke(app, ["--no-color", "validate", "-c", str(campaign)])

    assert res.exit_code == 0, res.stdout
    assert calls == [("id", "bio_type", "sequence", "alphabet", "Y", "opal__demo__label_hist")]


def test_run_rejects_x_matrix_memory_budget_before_records_load(tmp_path: Path, monkeypatch) -> None:
    workdir, campaign, records = _setup_workspace(
        tmp_path,
        include_opal_cols=True,
    )
    write_campaign_yaml(
        campaign,
        workdir=workdir,
        records_path=records,
        safety={"max_x_matrix_gib": 1.0e-9},
    )

    from dnadesign.opal.src.storage.data_access import RecordsStore

    def fail_load(self):
        raise AssertionError("records load should not happen after X memory guard failure")

    monkeypatch.setattr(RecordsStore, "load", fail_load)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "run", "-c", str(campaign), "--round", "0", "--json"])

    assert res.exit_code == 2
    assert "exceeds safety.max_x_matrix_gib" in _plain_output(res.output)
    round_log = workdir / "outputs" / "rounds" / "round_0" / "logs" / "round.log.jsonl"
    log_text = round_log.read_text(encoding="utf-8")
    assert '"stage":"x_validate_done"' in log_text
    assert '"stage":"records_load_start"' not in log_text


def test_run_ledger_only_streams_candidate_x_without_full_records_load(tmp_path: Path, monkeypatch) -> None:
    workdir, campaign, _ = _setup_workspace(tmp_path)
    raw = yaml.safe_load(campaign.read_text())
    raw["writeback"] = {"prediction_records": "ledger_only"}
    raw["objectives"][0]["params"]["scaling"] = {"percentile": 95, "min_n": 1, "eps": 1.0e-8}
    campaign.write_text(yaml.safe_dump(raw, sort_keys=False))

    app = _build()
    runner = CliRunner()
    init_res = runner.invoke(app, ["--no-color", "init", "-c", str(campaign)])
    assert init_res.exit_code == 0, init_res.stdout

    csv_path = workdir / "labels.csv"
    pd.DataFrame(
        {
            "sequence": ["AAA"],
            "v00": [0.0],
            "v10": [0.0],
            "v01": [0.0],
            "v11": [1.0],
            "y00_star": [0.1],
            "y10_star": [0.1],
            "y01_star": [0.1],
            "y11_star": [0.1],
            "intensity_log2_offset_delta": [0.0],
        }
    ).to_csv(csv_path, index=False)
    ingest_res = runner.invoke(
        app,
        [
            "--no-color",
            "ingest-y",
            "-c",
            str(campaign),
            "--round",
            "0",
            "--csv",
            str(csv_path),
            "--apply",
        ],
    )
    assert ingest_res.exit_code == 0, ingest_res.stdout

    from dnadesign.opal.src.storage.data_access import RecordsStore

    def fail_full_load(self):
        raise AssertionError("ledger_only run should not materialize the full records frame")

    monkeypatch.setattr(RecordsStore, "load", fail_full_load)

    run_res = runner.invoke(
        app,
        [
            "--no-color",
            "run",
            "-c",
            str(campaign),
            "--round",
            "0",
            "--score-batch-size",
            "1",
            "--json",
        ],
    )
    assert run_res.exit_code == 0, run_res.stdout
    out = json.loads(run_res.stdout)
    assert out["scored"] == 1
    assert (workdir / "outputs" / "ledger" / "predictions").exists()
    round_log = workdir / "outputs" / "rounds" / "round_0" / "logs" / "round.log.jsonl"
    log_text = round_log.read_text(encoding="utf-8")
    assert '"pool_mode":"streaming"' in log_text
    assert '"stage":"predict_batch"' in log_text


def test_validate_rejects_inconsistent_x_lengths(tmp_path: Path) -> None:
    _, campaign, records = _setup_workspace(tmp_path, include_opal_cols=True)
    pd.DataFrame(
        {
            "id": ["a", "b"],
            "sequence": ["AAA", "BBB"],
            "bio_type": ["dna", "dna"],
            "alphabet": ["dna_4", "dna_4"],
            "X": [[0.1, 0.2], [0.2, 0.3, 0.4]],
            "opal__demo__label_hist": [[], []],
            "Y": [None, None],
        }
    ).to_parquet(records, index=False)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "validate", "-c", str(campaign)])

    assert res.exit_code != 0
    assert "fixed_size_list" in res.output


def test_validate_rejects_null_x_values(tmp_path: Path) -> None:
    _, campaign, records = _setup_workspace(tmp_path, include_opal_cols=True)
    write_records_with_x_values(records, values=[[0.1, 0.2], None], include_opal_cols=True)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "validate", "-c", str(campaign)])

    assert res.exit_code != 0
    assert "null or ragged fixed-size-list rows" in res.output


def test_validate_rejects_unknown_plugin_names(tmp_path: Path) -> None:
    _, campaign, _ = _setup_workspace(tmp_path, include_opal_cols=True)
    app = _build()
    runner = CliRunner()

    raw = yaml.safe_load(campaign.read_text())
    raw["transforms_x"]["name"] = "does_not_exist_tx"
    raw["transforms_y"]["name"] = "does_not_exist_ty"
    raw["model"]["name"] = "does_not_exist_model"
    raw["selection"]["name"] = "does_not_exist_selection"
    raw["objectives"][0]["name"] = "does_not_exist_objective"
    campaign.write_text(yaml.safe_dump(raw, sort_keys=False))

    res = runner.invoke(app, ["--no-color", "validate", "-c", str(campaign)])
    assert res.exit_code != 0
    out = res.output.lower()
    assert "unknown transform_x plugin" in out


def test_validate_rejects_duplicate_yaml_keys_as_bad_args(tmp_path: Path) -> None:
    _, campaign, records = _setup_workspace(tmp_path, include_opal_cols=True)
    app = _build()
    runner = CliRunner()

    campaign.write_text(
        f"""
campaign:
  name: "Demo"
  slug: "demo"
  workdir: "."
data:
  location: {{ kind: local, path: "{records}" }}
  x_column_name: "X"
  y_column_name: "Y"
transforms_x: {{ name: identity, params: {{}} }}
transforms_y: {{ name: scalar_from_table_v1, params: {{}} }}
model: {{ name: random_forest, params: {{ n_estimators: 5, random_state: 0 }} }}
objectives:
  - {{ name: scalar_identity_v1, params: {{}} }}
selection:
  name: top_n
  params:
    top_k: 2
    score_ref: "scalar_identity_v1/scalar"
    objective_mode: maximize
    tie_handling: competition_rank
objectives:
  - {{ name: scalar_identity_v1, params: {{}} }}
""".strip()
    )

    res = runner.invoke(app, ["--no-color", "validate", "-c", str(campaign)])
    assert res.exit_code == 2
    out = res.output.lower()
    assert "duplicate key in yaml" in out
    assert "internal error during validate" not in out


def test_validate_unknown_model_error_lists_available_plugins_in_default_output(tmp_path: Path) -> None:
    _, campaign, _ = _setup_workspace(tmp_path, include_opal_cols=True)
    app = _build()
    runner = CliRunner()

    raw = yaml.safe_load(campaign.read_text())
    raw["model"]["name"] = "does_not_exist_model"
    campaign.write_text(yaml.safe_dump(raw, sort_keys=False))

    res = runner.invoke(app, ["validate", "-c", str(campaign)])
    assert res.exit_code != 0
    out = res.output.lower()
    assert "unknown model plugin" in out
    assert "gaussian_process" in out
    assert "random_forest" in out


def test_validate_requires_explicit_selection_contract_fields(tmp_path: Path) -> None:
    _, campaign, _ = _setup_workspace(tmp_path, include_opal_cols=True)
    app = _build()
    runner = CliRunner()

    raw = yaml.safe_load(campaign.read_text())
    params = raw["selection"]["params"]
    params.pop("top_k", None)
    params.pop("tie_handling", None)
    params.pop("objective_mode", None)
    campaign.write_text(yaml.safe_dump(raw, sort_keys=False))

    res = runner.invoke(app, ["--no-color", "validate", "-c", str(campaign)])
    assert res.exit_code != 0
    out = res.output.lower()
    assert "top_k" in out
    assert "tie_handling" in out
    assert "objective_mode" in out


def test_validate_rejects_unknown_selection_score_channel_for_declared_objective(tmp_path: Path) -> None:
    _, campaign, _ = _setup_workspace(tmp_path, include_opal_cols=True)
    app = _build()
    runner = CliRunner()

    raw = yaml.safe_load(campaign.read_text())
    raw["selection"]["params"]["score_ref"] = "sfxi_v1/missing_channel"
    campaign.write_text(yaml.safe_dump(raw, sort_keys=False))

    res = runner.invoke(app, ["--no-color", "validate", "-c", str(campaign)])
    assert res.exit_code != 0
    out = res.output.lower()
    assert "score_ref channel" in out
    assert "missing_channel" in out
    assert "available" in out


def test_validate_rejects_ei_channel_typo_before_runtime(tmp_path: Path) -> None:
    _, campaign, _ = _setup_workspace(tmp_path, include_opal_cols=True)
    app = _build()
    runner = CliRunner()

    raw = yaml.safe_load(campaign.read_text())
    raw["model"]["name"] = "gaussian_process"
    raw["model"]["params"] = {
        "alpha": 1.0e-6,
        "normalize_y": True,
        "kernel": {"name": "rbf", "length_scale": 1.0},
    }
    raw["selection"]["name"] = "expected_improvement"
    raw["selection"]["params"]["uncertainty_ref"] = "sfxi_v1/missing_channel"
    raw["selection"]["params"]["alpha"] = 1.0
    raw["selection"]["params"]["beta"] = 1.0
    campaign.write_text(yaml.safe_dump(raw, sort_keys=False))

    res = runner.invoke(app, ["--no-color", "validate", "-c", str(campaign)])
    assert res.exit_code != 0
    out = res.output.lower()
    assert "uncertainty_ref channel" in out
    assert "missing_channel" in out
    assert "available" in out


def test_validate_rejects_ei_with_model_without_predictive_std(tmp_path: Path) -> None:
    _, campaign, _ = _setup_workspace(tmp_path, include_opal_cols=True)
    app = _build()
    runner = CliRunner()

    raw = yaml.safe_load(campaign.read_text())
    raw["selection"]["name"] = "expected_improvement"
    raw["selection"]["params"]["uncertainty_ref"] = "sfxi_v1/sfxi"
    raw["selection"]["params"]["alpha"] = 1.0
    raw["selection"]["params"]["beta"] = 1.0
    campaign.write_text(yaml.safe_dump(raw, sort_keys=False))

    res = runner.invoke(app, ["--no-color", "validate", "-c", str(campaign)])
    assert res.exit_code != 0
    out = _plain_output(res.output).lower()
    assert "expected_improvement" in out
    assert "predictive std" in out


def test_validate_rejects_objective_mode_mismatch_for_score_ref(tmp_path: Path) -> None:
    _, campaign, _ = _setup_workspace(tmp_path, include_opal_cols=True)
    app = _build()
    runner = CliRunner()

    raw = yaml.safe_load(campaign.read_text())
    raw["selection"]["params"]["objective_mode"] = "minimize"
    campaign.write_text(yaml.safe_dump(raw, sort_keys=False))

    res = runner.invoke(app, ["--no-color", "validate", "-c", str(campaign)])
    assert res.exit_code != 0
    out = res.output.lower()
    assert "objective mode mismatch" in out
    assert "sfxi_v1/sfxi" in out
    assert "maximize" in out
    assert "minimize" in out


def test_validate_rejects_ei_negative_weights(tmp_path: Path) -> None:
    _, campaign, _ = _setup_workspace(tmp_path, include_opal_cols=True)
    app = _build()
    runner = CliRunner()

    raw = yaml.safe_load(campaign.read_text())
    raw["model"]["name"] = "gaussian_process"
    raw["model"]["params"] = {
        "alpha": 1.0e-6,
        "normalize_y": True,
        "kernel": {"name": "rbf", "length_scale": 1.0},
    }
    raw["selection"]["name"] = "expected_improvement"
    raw["selection"]["params"]["uncertainty_ref"] = "sfxi_v1/sfxi"
    raw["selection"]["params"]["alpha"] = -0.1
    raw["selection"]["params"]["beta"] = -0.5
    campaign.write_text(yaml.safe_dump(raw, sort_keys=False))

    res = runner.invoke(app, ["--no-color", "validate", "-c", str(campaign)])
    assert res.exit_code != 0
    out = res.output.lower()
    assert "alpha" in out
    assert "beta" in out
    assert ">= 0" in out


def test_run_rejects_corrupt_state_json(tmp_path: Path) -> None:
    workdir, campaign, _ = _setup_workspace(tmp_path, include_opal_cols=True)
    app = _build()
    runner = CliRunner()

    (workdir / "state.json").write_text("{not-valid-json")

    res = runner.invoke(app, ["--no-color", "run", "-c", str(campaign), "--round", "0", "--quiet"])
    assert res.exit_code != 0
    out = res.output.lower()
    assert "failed to load state.json" in out


def test_run_surfaces_sfxi_round_label_requirements_as_opal_error(tmp_path: Path) -> None:
    workdir, campaign, _ = _setup_workspace(tmp_path)
    app = _build()
    runner = CliRunner()

    raw = yaml.safe_load(campaign.read_text())
    raw["objectives"][0]["params"]["scaling"] = {"percentile": 95, "min_n": 1, "eps": 1.0e-8}
    campaign.write_text(yaml.safe_dump(raw, sort_keys=False))

    init_res = runner.invoke(app, ["--no-color", "init", "-c", str(campaign)])
    assert init_res.exit_code == 0, init_res.stdout

    csv_path = workdir / "labels.csv"
    df = pd.DataFrame(
        {
            "sequence": ["AAA"],
            "v00": [0.0],
            "v10": [0.0],
            "v01": [0.0],
            "v11": [1.0],
            "y00_star": [0.1],
            "y10_star": [0.1],
            "y01_star": [0.1],
            "y11_star": [0.1],
            "intensity_log2_offset_delta": [0.0],
        }
    )
    df.to_csv(csv_path, index=False)

    ingest_res = runner.invoke(
        app,
        [
            "--no-color",
            "ingest-y",
            "-c",
            str(campaign),
            "--round",
            "0",
            "--csv",
            str(csv_path),
            "--apply",
        ],
    )
    assert ingest_res.exit_code == 0, ingest_res.stdout

    res = runner.invoke(app, ["--no-color", "run", "-c", str(campaign), "--round", "1"])
    assert res.exit_code == 2, res.output
    out = res.output.lower()
    assert "objective plugin 'sfxi_v1' failed" in out
    assert "min_n=1" in out
    assert "current round" in out


def test_run_reports_empty_candidate_pool_as_opal_error(tmp_path: Path) -> None:
    workdir, campaign, _ = _setup_workspace(tmp_path)
    app = _build()
    runner = CliRunner()

    raw = yaml.safe_load(campaign.read_text())
    raw["objectives"][0]["params"]["scaling"] = {"percentile": 95, "min_n": 1, "eps": 1.0e-8}
    campaign.write_text(yaml.safe_dump(raw, sort_keys=False))

    init_res = runner.invoke(app, ["--no-color", "init", "-c", str(campaign)])
    assert init_res.exit_code == 0, init_res.stdout

    csv_path = workdir / "labels.csv"
    df = pd.DataFrame(
        {
            "sequence": ["AAA", "BBB"],
            "v00": [0.0, 0.0],
            "v10": [0.0, 0.0],
            "v01": [0.0, 0.0],
            "v11": [1.0, 0.5],
            "y00_star": [0.1, 0.2],
            "y10_star": [0.1, 0.2],
            "y01_star": [0.1, 0.2],
            "y11_star": [0.1, 0.2],
            "intensity_log2_offset_delta": [0.0, 0.0],
        }
    )
    df.to_csv(csv_path, index=False)

    ingest_res = runner.invoke(
        app,
        [
            "--no-color",
            "ingest-y",
            "-c",
            str(campaign),
            "--round",
            "0",
            "--csv",
            str(csv_path),
            "--apply",
        ],
    )
    assert ingest_res.exit_code == 0, ingest_res.stdout

    res = runner.invoke(app, ["--no-color", "run", "-c", str(campaign), "--round", "0"])
    assert res.exit_code == 2, res.output
    out = res.output.lower()
    assert "candidate pool is empty after filtering" in out


def test_label_hist_validate_and_repair(tmp_path: Path) -> None:
    _, campaign, _ = _setup_workspace(tmp_path, include_opal_cols=True)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "label-hist", "validate", "-c", str(campaign), "--json"])
    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)
    assert out["ok"] is True
    assert out["action"] == "validate"

    res = runner.invoke(app, ["--no-color", "label-hist", "repair", "-c", str(campaign), "--json"])
    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)
    assert out["ok"] is True
    assert out["action"] == "repair"
    assert out["applied"] is False


def test_ctx_show_audit_diff(tmp_path: Path) -> None:
    workdir, campaign, _ = _setup_workspace(tmp_path)
    app = _build()
    runner = CliRunner()

    round0 = workdir / "outputs" / "rounds" / "round_0"
    round1 = workdir / "outputs" / "rounds" / "round_1"
    round0.mkdir(parents=True, exist_ok=True)
    round1.mkdir(parents=True, exist_ok=True)

    ctx0 = {
        "core/run_id": "r0",
        "core/contracts/model/random_forest/produced": ["model/random_forest/x_dim"],
    }
    ctx1 = {
        "core/run_id": "r1",
        "core/contracts/model/random_forest/produced": [
            "model/random_forest/x_dim",
            "model/random_forest/y_dim",
        ],
    }
    ctx0_path = round0 / "metadata" / "round_ctx.json"
    ctx1_path = round1 / "metadata" / "round_ctx.json"
    ctx0_path.parent.mkdir(parents=True, exist_ok=True)
    ctx1_path.parent.mkdir(parents=True, exist_ok=True)
    ctx0_path.write_text(json.dumps(ctx0))
    ctx1_path.write_text(json.dumps(ctx1))

    res = runner.invoke(
        app,
        ["--no-color", "ctx", "show", "-c", str(campaign), "--round", "0", "--json"],
    )
    assert res.exit_code == 0, res.stdout
    out = json.loads(res.stdout)
    assert out["core/run_id"] == "r0"

    res = runner.invoke(
        app,
        ["--no-color", "ctx", "audit", "-c", str(campaign), "--round", "0", "--json"],
    )
    assert res.exit_code == 0, res.stdout
    audit = json.loads(res.stdout)
    assert "model" in audit
    assert "random_forest" in audit["model"]

    res = runner.invoke(
        app,
        [
            "--no-color",
            "ctx",
            "diff",
            "-c",
            str(campaign),
            "--round-a",
            "0",
            "--round-b",
            "1",
            "--json",
        ],
    )
    assert res.exit_code == 0, res.stdout
    diff = json.loads(res.stdout)
    assert "core/run_id" in diff.get("changed", {})


def test_ingest_y_cli(tmp_path: Path) -> None:
    workdir, campaign, _ = _setup_workspace(tmp_path)
    app = _build()
    runner = CliRunner()

    csv_path = workdir / "labels.csv"
    df = pd.DataFrame(
        {
            "sequence": ["AAA", "BBB"],
            "v00": [0.0, 0.0],
            "v10": [0.0, 0.0],
            "v01": [0.0, 0.0],
            "v11": [1.0, 0.5],
            "y00_star": [0.1, 0.2],
            "y10_star": [0.1, 0.2],
            "y01_star": [0.1, 0.2],
            "y11_star": [0.1, 0.2],
            "intensity_log2_offset_delta": [0.0, 0.0],
        }
    )
    df.to_csv(csv_path, index=False)

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
            str(csv_path),
            "--apply",
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert (workdir / "outputs" / "ledger" / "labels.parquet").exists()


def test_ingest_y_uses_apply_flag(tmp_path: Path) -> None:
    workdir, campaign, _ = _setup_workspace(tmp_path)
    app = _build()
    runner = CliRunner()

    csv_path = workdir / "labels.csv"
    df = pd.DataFrame(
        {
            "sequence": ["AAA", "BBB"],
            "v00": [0.0, 0.0],
            "v10": [0.0, 0.0],
            "v01": [0.0, 0.0],
            "v11": [1.0, 0.5],
            "y00_star": [0.1, 0.2],
            "y10_star": [0.1, 0.2],
            "y01_star": [0.1, 0.2],
            "y11_star": [0.1, 0.2],
            "intensity_log2_offset_delta": [0.0, 0.0],
        }
    )
    df.to_csv(csv_path, index=False)

    res_bad = runner.invoke(
        app,
        [
            "--no-color",
            "ingest-y",
            "-c",
            str(campaign),
            "--round",
            "0",
            "--csv",
            str(csv_path),
            "--yes",
        ],
    )
    assert res_bad.exit_code != 0

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
            str(csv_path),
            "--apply",
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert (workdir / "outputs" / "ledger" / "labels.parquet").exists()


def test_ingest_y_accepts_xlsx(tmp_path: Path) -> None:
    workdir, campaign, _ = _setup_workspace(tmp_path)
    app = _build()
    runner = CliRunner()

    xlsx_path = workdir / "labels.xlsx"
    df = pd.DataFrame(
        {
            "sequence": ["AAA", "BBB"],
            "v00": [0.0, 0.0],
            "v10": [0.0, 0.0],
            "v01": [0.0, 0.0],
            "v11": [1.0, 0.5],
            "y00_star": [0.1, 0.2],
            "y10_star": [0.1, 0.2],
            "y01_star": [0.1, 0.2],
            "y11_star": [0.1, 0.2],
            "intensity_log2_offset_delta": [0.0, 0.0],
        }
    )
    df.to_excel(xlsx_path, index=False)

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
            str(xlsx_path),
            "--apply",
        ],
    )
    assert res.exit_code == 0, res.stdout
    assert (workdir / "outputs" / "ledger" / "labels.parquet").exists()


def test_ingest_y_drop_unknown_sequences_preview(tmp_path: Path) -> None:
    workdir, campaign, _ = _setup_workspace(tmp_path)
    app = _build()
    runner = CliRunner()

    csv_path = workdir / "labels.csv"
    df = pd.DataFrame(
        {
            "sequence": ["AAA", "ZZZ"],
            "v00": [0.0, 0.0],
            "v10": [0.0, 0.0],
            "v01": [0.0, 0.0],
            "v11": [1.0, 0.5],
            "y00_star": [0.1, 0.2],
            "y10_star": [0.1, 0.2],
            "y01_star": [0.1, 0.2],
            "y11_star": [0.1, 0.2],
            "intensity_log2_offset_delta": [0.0, 0.0],
        }
    )
    df.to_csv(csv_path, index=False)

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
            str(csv_path),
            "--unknown-sequences",
            "drop",
            "--apply",
        ],
    )
    assert res.exit_code == 0, res.stdout
    lowered = res.stdout.lower()
    assert "new rows will be created" not in lowered
    assert "dropping 1 unknown sequences" in lowered


def test_ingest_y_rejects_unsupported_extension(tmp_path: Path) -> None:
    workdir, campaign, _ = _setup_workspace(tmp_path)
    app = _build()
    runner = CliRunner()

    bad_path = workdir / "labels.txt"
    df = pd.DataFrame(
        {
            "sequence": ["AAA"],
            "v00": [0.0],
            "v10": [0.0],
            "v01": [0.0],
            "v11": [1.0],
            "y00_star": [0.1],
            "y10_star": [0.1],
            "y01_star": [0.1],
            "y11_star": [0.1],
            "intensity_log2_offset_delta": [0.0],
        }
    )
    df.to_csv(bad_path, index=False)

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
            str(bad_path),
            "--apply",
        ],
    )
    assert res.exit_code != 0, res.stdout
    assert "must be a table file with extension" in res.output


def test_ingest_y_rejects_params_non_json(tmp_path: Path) -> None:
    workdir, campaign, _ = _setup_workspace(tmp_path)
    app = _build()
    runner = CliRunner()

    csv_path = workdir / "labels.csv"
    df = pd.DataFrame(
        {
            "sequence": ["AAA"],
            "v00": [0.0],
            "v10": [0.0],
            "v01": [0.0],
            "v11": [1.0],
            "y00_star": [0.1],
            "y10_star": [0.1],
            "y01_star": [0.1],
            "y11_star": [0.1],
            "intensity_log2_offset_delta": [0.0],
        }
    )
    df.to_csv(csv_path, index=False)

    bad_params = workdir / "params.txt"
    bad_params.write_text("{}")

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
            str(csv_path),
            "--params",
            str(bad_params),
            "--apply",
        ],
    )
    assert res.exit_code != 0, res.stdout
    assert "must be a JSON file" in res.output
