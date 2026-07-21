"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/cli/test_cli_workflows.py

Regression tests for CLI workflows OPAL CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml
from click import unstyle
from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.tests._cli_helpers import (
    write_campaign_yaml,
    write_ledger,
    write_records,
    write_records_with_x_values,
    write_state,
)


def _plain_output(text: str) -> str:
    return " ".join(unstyle(str(text)).split())


def _setup_workspace(tmp_path: Path, *, include_opal_cols: bool = False) -> tuple[Path, Path, Path]:
    workdir = tmp_path / "campaign"
    workdir.mkdir(parents=True, exist_ok=True)
    records = workdir / "records.parquet"
    write_records(records, include_opal_cols=include_opal_cols)
    campaign = workdir / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=workdir, records_path=records)
    return workdir, campaign, records


def _configure_metadata_backed_restriction_site_exclusion(campaign: Path, *, column: str) -> None:
    raw = yaml.safe_load(campaign.read_text(encoding="utf-8"))
    raw["candidate_eligibility"] = {
        "rules": [
            {
                "name": "restriction_site_exclusion",
                "params": {
                    "sequence_column": "sequence",
                    "scan_space": "final_assembled_insert",
                    "assembly_strategy_ref": "test_insert:v1",
                    "left_flank": "a",
                    "right_flank": "a",
                    "expected_core_length": 3,
                    "min_remaining_candidates": 1,
                    "forbidden_sites": [
                        {
                            "enzyme": "EcoRI",
                            "motif": "GAATTC",
                            "allowed_regions": ["right_flank"],
                        }
                    ],
                    "exclude_rows_where": [{"column": column, "equals": "control"}],
                },
            }
        ]
    }
    campaign.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


@pytest.mark.parametrize(
    ("args", "retired_flag"),
    [
        (["ingest-y", "--observed-round", "0", "--in", "labels.csv"], "--observed-round"),
        (["run", "--labels-as-of", "0"], "--labels-as-of"),
        (["explain", "--labels-as-of", "0"], "--labels-as-of"),
        (["guide", "--labels-as-of", "0"], "--labels-as-of"),
        (["guide", "next", "--observed-round", "0"], "--observed-round"),
    ],
)
def test_round_commands_reject_retired_option_aliases(
    tmp_path: Path,
    args: list[str],
    retired_flag: str,
) -> None:
    _, campaign, _ = _setup_workspace(tmp_path, include_opal_cols=True)

    result = CliRunner().invoke(_build(), ["--no-color", *args, "-c", str(campaign)])

    assert result.exit_code == 2
    assert f"No such option: {retired_flag}" in _plain_output(result.output)


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


def test_validate_json_writes_machine_readable_contract(tmp_path: Path) -> None:
    workdir, campaign, records = _setup_workspace(tmp_path, include_opal_cols=True)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "validate", "-c", str(campaign), "--json"])

    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["schema_version"] == "opal.validate.v1"
    assert payload["ok"] is True
    assert payload["config_path"] == str(campaign.resolve())
    assert payload["campaign"]["workdir"] == str(workdir.resolve())
    assert payload["records"]["path"] == str(records.resolve())
    assert payload["records"]["row_count"] == 2
    assert payload["records"]["column_count"] >= 5
    assert payload["data"]["x_column"] == "X"
    assert payload["data"]["y_column"] == "Y"
    assert payload["x_contract"] == {"row_count": 2, "x_dim": 2}
    assert payload["label_source"]["kind"] == "campaign_history"
    assert payload["label_source"]["prediction_records"] == "ledger_only"
    assert "validation passed" not in res.stdout.lower()


def test_validate_json_error_is_machine_readable(tmp_path: Path) -> None:
    _, campaign, _ = _setup_workspace(tmp_path, include_opal_cols=True)
    cfg = yaml.safe_load(campaign.read_text(encoding="utf-8"))
    cfg["data"]["x_column_name"] = "missing_x"
    campaign.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "validate", "-c", str(campaign), "--json"])

    assert res.exit_code != 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["ok"] is False
    assert payload["error"]["schema_version"] == "opal.cli_error.v1"
    assert payload["error"]["context"] == "validate"
    assert "Missing X column" in payload["error"]["message"]
    assert "Missing X column" not in res.stderr


def test_validate_projects_metadata_required_by_candidate_eligibility(tmp_path: Path) -> None:
    _, campaign, records = _setup_workspace(tmp_path, include_opal_cols=True)
    table = pq.read_table(records).append_column(
        "design_family",
        pa.array(["candidate", "control"], type=pa.string()),
    )
    pq.write_table(table, records)
    _configure_metadata_backed_restriction_site_exclusion(campaign, column="design_family")

    result = CliRunner().invoke(_build(), ["--no-color", "validate", "-c", str(campaign), "--json"])

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["candidate_eligibility"]["input_rows"] == 2
    assert payload["candidate_eligibility"]["output_rows"] == 1
    assert payload["candidate_eligibility"]["rules"][0]["pre_excluded_rows"] == 1


def test_validate_rejects_missing_metadata_required_by_candidate_eligibility(tmp_path: Path) -> None:
    _, campaign, _ = _setup_workspace(tmp_path, include_opal_cols=True)
    _configure_metadata_backed_restriction_site_exclusion(campaign, column="design_family")

    result = CliRunner().invoke(_build(), ["--no-color", "validate", "-c", str(campaign), "--json"])

    assert result.exit_code == 2
    payload = json.loads(result.stdout)
    assert payload["error"]["context"] == "validate"
    assert payload["error"]["message"] == ("records.parquet missing configured candidate column(s): ['design_family'].")


def test_notebook_generate_json_writes_machine_readable_summary(tmp_path: Path) -> None:
    workdir, campaign, _ = _setup_workspace(tmp_path, include_opal_cols=True)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(app, ["--no-color", "notebook", "generate", "-c", str(campaign), "--force", "--json"])

    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["schema_version"] == "opal.notebook_generate.v1"
    assert payload["ok"] is True
    assert payload["kind"] == "campaign"
    assert payload["campaign_count"] == 1
    assert payload["round_selector"] == "latest"
    assert payload["run_id"] is None
    assert payload["config_paths"] == [str(campaign)]
    assert Path(payload["notebook_path"]).exists()
    assert str(workdir) in payload["workdirs"]
    assert "Notebook written" not in res.stdout


def test_notebook_generate_json_pins_run_id_and_round(tmp_path: Path) -> None:
    _, campaign, _ = _setup_workspace(tmp_path, include_opal_cols=True)
    workdir = campaign.parent
    write_ledger(workdir, run_id="run-0", round_index=0)
    write_ledger(workdir, run_id="run-1", round_index=0)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(
        app,
        [
            "--no-color",
            "notebook",
            "generate",
            "-c",
            str(campaign),
            "--run-id",
            "run-1",
            "--force",
            "--json",
        ],
    )

    assert res.exit_code == 0, res.stdout
    payload = json.loads(res.stdout)
    assert payload["round_selector"] == "0"
    assert payload["run_id"] == "run-1"
    text = Path(payload["notebook_path"]).read_text(encoding="utf-8")
    assert "selected_round_selector = '0'" in text
    assert "run_id='run-1'" in text


def test_notebook_generate_rejects_run_id_round_mismatch_json(tmp_path: Path) -> None:
    _, campaign, _ = _setup_workspace(tmp_path, include_opal_cols=True)
    write_ledger(campaign.parent, run_id="run-0", round_index=0)
    app = _build()
    runner = CliRunner()

    res = runner.invoke(
        app,
        [
            "--no-color",
            "notebook",
            "generate",
            "-c",
            str(campaign),
            "--round",
            "1",
            "--run-id",
            "run-0",
            "--json",
        ],
    )

    assert res.exit_code == 2
    payload = json.loads(res.stdout)
    assert payload["error"]["schema_version"] == "opal.cli_error.v1"
    assert payload["error"]["context"] == "notebook.generate"
    assert "belongs to round 0" in payload["error"]["message"]


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


@pytest.mark.parametrize("operator_declines", [False, True])
def test_run_rejected_state_rerun_does_not_mutate_completed_round_log(
    tmp_path: Path,
    monkeypatch,
    operator_declines: bool,
) -> None:
    workdir, campaign, records = _setup_workspace(tmp_path, include_opal_cols=True)
    write_state(workdir, records_path=records, run_id="completed-run", round_index=0)
    round_log = workdir / "outputs" / "rounds" / "round_0" / "logs" / "round.log.jsonl"
    original_log = round_log.read_bytes()
    if operator_declines:
        monkeypatch.setattr(
            "dnadesign.opal.src.cli.commands.run.prompt_confirm",
            lambda *_args, **_kwargs: False,
        )

    result = CliRunner().invoke(
        _build(),
        ["--no-color", "run", "-c", str(campaign), "--round", "0", "--quiet"],
    )

    assert result.exit_code == 2
    expected_message = "Aborted." if operator_declines else "Re-run with --resume"
    assert expected_message in _plain_output(result.output)
    assert round_log.read_bytes() == original_log


def test_run_rejected_artifact_rerun_does_not_mutate_round_log(tmp_path: Path) -> None:
    workdir, campaign, _ = _setup_workspace(tmp_path, include_opal_cols=True)
    model_artifact = workdir / "outputs" / "rounds" / "round_0" / "model" / "model.joblib"
    model_artifact.parent.mkdir(parents=True, exist_ok=True)
    model_artifact.write_bytes(b"completed-model")
    round_log = workdir / "outputs" / "rounds" / "round_0" / "logs" / "round.log.jsonl"

    result = CliRunner().invoke(
        _build(),
        ["--no-color", "run", "-c", str(campaign), "--round", "0", "--quiet"],
    )

    assert result.exit_code == 2
    assert "already contains artifacts" in _plain_output(result.output)
    assert model_artifact.read_bytes() == b"completed-model"
    assert not round_log.exists()


def test_run_rejected_campaign_lock_does_not_mutate_round_log(tmp_path: Path) -> None:
    workdir, campaign, _ = _setup_workspace(tmp_path, include_opal_cols=True)
    round_log = workdir / "outputs" / "rounds" / "round_0" / "logs" / "round.log.jsonl"
    round_log.parent.mkdir(parents=True, exist_ok=True)
    round_log.write_text(json.dumps({"stage": "command_start"}) + "\n", encoding="utf-8")
    original_log = round_log.read_bytes()
    (workdir / ".opal.lock").write_text(
        json.dumps({"pid": os.getpid(), "ts": "2026-07-20T00:00:00+00:00"}),
        encoding="utf-8",
    )

    result = CliRunner().invoke(
        _build(),
        ["--no-color", "run", "-c", str(campaign), "--round", "0", "--quiet"],
    )

    assert result.exit_code == 4
    assert "locked by another process" in _plain_output(result.output)
    assert round_log.read_bytes() == original_log


def test_run_rechecks_round_artifacts_after_acquiring_campaign_lock(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workdir, campaign, _ = _setup_workspace(tmp_path, include_opal_cols=True)
    from dnadesign.opal.src.cli.commands import run as run_command

    model_artifact = workdir / "outputs" / "rounds" / "round_0" / "model" / "model.joblib"
    round_log = workdir / "outputs" / "rounds" / "round_0" / "logs" / "round.log.jsonl"
    original_check = run_command.assert_round_artifacts_writable
    completed_log = json.dumps({"stage": "command_complete"}).encode() + b"\n"
    check_count = 0

    def check_after_competing_run(*args, **kwargs):
        nonlocal check_count
        result = original_check(*args, **kwargs)
        check_count += 1
        if check_count == 1:
            model_artifact.parent.mkdir(parents=True, exist_ok=True)
            model_artifact.write_bytes(b"completed-by-competing-run")
            round_log.parent.mkdir(parents=True, exist_ok=True)
            round_log.write_bytes(completed_log)
        return result

    monkeypatch.setattr(run_command, "assert_round_artifacts_writable", check_after_competing_run)

    result = CliRunner().invoke(
        _build(),
        ["--no-color", "run", "-c", str(campaign), "--round", "0", "--quiet"],
    )

    assert result.exit_code == 2
    assert "already contains artifacts" in _plain_output(result.output)
    assert model_artifact.read_bytes() == b"completed-by-competing-run"
    assert round_log.read_bytes() == completed_log
    assert not (workdir / ".opal.lock").exists()


def test_run_writes_every_round_log_event_while_campaign_lock_is_held(
    tmp_path: Path,
    monkeypatch,
) -> None:
    workdir, campaign, _ = _setup_workspace(tmp_path, include_opal_cols=True)
    from dnadesign.opal.src.cli.commands import run as run_command

    original_append = run_command._append_cli_round_event
    observed: list[tuple[str, bool]] = []

    def recording_append(cfg, cfg_path, round_index, stage, **payload):
        observed.append((str(stage), (workdir / ".opal.lock").exists()))
        return original_append(cfg, cfg_path, round_index, stage, **payload)

    monkeypatch.setattr(run_command, "_append_cli_round_event", recording_append)

    result = CliRunner().invoke(
        _build(),
        ["--no-color", "run", "-c", str(campaign), "--round", "0", "--quiet"],
    )

    assert result.exit_code == 2
    assert "state.json not found" in _plain_output(result.output)
    assert observed
    assert all(lock_held for _, lock_held in observed), observed
    assert [stage for stage, _ in observed][-2:] == ["abort", "lock_release_start"]
    assert not (workdir / ".opal.lock").exists()


def test_run_ledger_only_streams_candidate_x_without_full_records_load(tmp_path: Path, monkeypatch) -> None:
    workdir, campaign, records = _setup_workspace(tmp_path)
    records_table = pq.read_table(records).append_column(
        "candidate_key",
        pa.array(["candidate-a", "candidate-b"], type=pa.string()),
    )
    pq.write_table(records_table, records)
    raw = yaml.safe_load(campaign.read_text())
    raw["writeback"] = {"prediction_records": "ledger_only"}
    raw["selection_batch"] = {"deduplicate_by": "candidate_key"}
    raw["selection_views"][0]["objective"]["params"]["scaling"] = {
        "percentile": 95,
        "min_n": 1,
        "eps": 1.0e-8,
    }
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
    assert run_res.exit_code == 0, run_res.output
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
    raw["selection_views"][0]["selection"]["name"] = "does_not_exist_selection"
    raw["selection_views"][0]["objective"]["name"] = "does_not_exist_objective"
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
schema_version: opal.campaign.v3
ownership: {{owner_scope: opal_demo, portable: true}}
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
selection_views:
  - id: primary
    objective: {{ name: scalar_identity_v1, params: {{}} }}
    selection:
      name: top_n
      params:
        top_k: 2
        score_ref: scalar
        objective_mode: maximize
        tie_handling: competition_rank
selection_views:
  - id: duplicate
    objective: {{ name: scalar_identity_v1, params: {{}} }}
    selection:
      name: top_n
      params: {{ top_k: 2, score_ref: scalar, objective_mode: maximize, tie_handling: competition_rank }}
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
    params = raw["selection_views"][0]["selection"]["params"]
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
    raw["selection_views"][0]["selection"]["params"]["score_ref"] = "missing_channel"
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
    selection = raw["selection_views"][0]["selection"]
    selection["name"] = "expected_improvement"
    selection["params"]["uncertainty_ref"] = "missing_channel"
    selection["params"]["alpha"] = 1.0
    selection["params"]["beta"] = 1.0
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
    selection = raw["selection_views"][0]["selection"]
    selection["name"] = "expected_improvement"
    selection["params"]["uncertainty_ref"] = "sfxi"
    selection["params"]["alpha"] = 1.0
    selection["params"]["beta"] = 1.0
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
    raw["selection_views"][0]["selection"]["params"]["objective_mode"] = "minimize"
    campaign.write_text(yaml.safe_dump(raw, sort_keys=False))

    res = runner.invoke(app, ["--no-color", "validate", "-c", str(campaign)])
    assert res.exit_code != 0
    out = res.output.lower()
    assert "objective mode mismatch" in out
    assert "sfxi" in out
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
    selection = raw["selection_views"][0]["selection"]
    selection["name"] = "expected_improvement"
    selection["params"]["uncertainty_ref"] = "sfxi"
    selection["params"]["alpha"] = -0.1
    selection["params"]["beta"] = -0.5
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
    assert not (workdir / "outputs" / "rounds" / "round_0" / "logs" / "round.log.jsonl").exists()


def test_run_surfaces_sfxi_round_label_requirements_as_opal_error(tmp_path: Path) -> None:
    workdir, campaign, _ = _setup_workspace(tmp_path)
    app = _build()
    runner = CliRunner()

    raw = yaml.safe_load(campaign.read_text())
    raw["selection_views"][0]["objective"]["params"]["scaling"] = {
        "percentile": 95,
        "min_n": 1,
        "eps": 1.0e-8,
    }
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
    raw["selection_views"][0]["objective"]["params"]["scaling"] = {
        "percentile": 95,
        "min_n": 1,
        "eps": 1.0e-8,
    }
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
