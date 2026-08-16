"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/cli/test_cli_history_import.py

Exercises the OPAL CLI contract for consolidating disjoint campaign histories.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml
from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.src.storage.parquet_io import read_parquet_df
from dnadesign.opal.src.storage.state import CampaignState
from dnadesign.opal.tests._cli_helpers import (
    write_campaign_yaml,
    write_ledger,
    write_ledger_labels,
    write_records,
    write_state,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_round(workdir: Path, *, round_index: int, run_id: str) -> None:
    round_dir = workdir / "outputs" / "rounds" / f"round_{round_index}"
    model = round_dir / "model" / "model.joblib"
    model.parent.mkdir(parents=True, exist_ok=True)
    model.write_bytes(f"model-{round_index}".encode())
    (round_dir / "model" / "model_meta.json").write_text(
        json.dumps({"schema_version": "opal.model_meta.v1", "model": "random_forest"}),
        encoding="utf-8",
    )
    metadata = round_dir / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    (metadata / "objective_meta.json").write_text(
        json.dumps({"schema_version": "opal.objective_meta.v1", "selection_view_id": "primary"}),
        encoding="utf-8",
    )
    (metadata / "round_ctx.json").write_text(
        json.dumps(
            {
                "core/campaign_slug": "demo",
                "core/run_id": run_id,
                "core/round_index": round_index,
                "core/labels_as_of_round": round_index,
                "core/data/n_train": 2,
                "core/data/n_scored": 2,
                "core/data/x_dim": 2,
                "core/plugins/selection_views/ids": ["primary"],
                "core/selection_batch/allocation": {
                    "strategy": "round_robin_next_best_unallocated",
                    "deduplicate_by": "id",
                    "view_priority": ["primary"],
                    "quota_by_view": {"primary": 1},
                    "initial_membership_count": 1,
                    "initial_unique_count": 1,
                    "overlap_membership_count": 0,
                    "skipped_overlap_count": 0,
                    "replacement_count": 0,
                    "final_unique_count": 1,
                    "expected_unique_count": 1,
                    "per_view": {
                        "primary": {
                            "quota": 1,
                            "allocated": 1,
                            "skipped_overlap_count": 0,
                            "replacement_count": 0,
                        }
                    },
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    labels = pd.DataFrame(
        {
            "run_id": [run_id] * (round_index + 1),
            "as_of_round": [round_index] * (round_index + 1),
            "observed_round": list(range(round_index + 1)),
            "id": [f"observed-{index}" for index in range(round_index + 1)],
            "sequence": [f"SEQ{index}" for index in range(round_index + 1)],
            "y_obs": [[0.1] for _ in range(round_index + 1)],
            "src": ["test"] * (round_index + 1),
            "note": [""] * (round_index + 1),
        }
    )
    artifact_root = round_dir / "run_artifacts" / run_id
    labels_dir = artifact_root / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(labels_dir / "labels_used.parquet", index=False)
    labels.to_parquet(labels_dir / "observed_events.parquet", index=False)
    selections = pd.DataFrame(
        {
            "run_id": [run_id],
            "as_of_round": [round_index],
            "campaign_slug": ["demo"],
            "selection_view_id": ["primary"],
            "selection_name": ["top_n"],
            "objective_name": ["sfxi_v1"],
            "id": [f"selected-{round_index}"],
            "score_ref": ["primary/sfxi"],
        }
    )
    selection_dir = round_dir / "selection"
    selection_dir.mkdir(parents=True, exist_ok=True)
    selections.to_parquet(selection_dir / "selections.parquet", index=False)
    batch = pd.DataFrame(
        {
            "run_id": [run_id],
            "as_of_round": [round_index],
            "campaign_slug": ["demo"],
            "id": [f"selected-{round_index}"],
            "deduplicate_by": ["id"],
            "allocation_view_id": ["primary"],
        }
    )
    batch.to_parquet(selection_dir / "selection_batch.parquet", index=False)
    artifact_selection = artifact_root / "selection"
    artifact_selection.mkdir(parents=True, exist_ok=True)
    selections.to_parquet(artifact_selection / "selections.parquet", index=False)
    batch.to_parquet(artifact_selection / "selection_batch.parquet", index=False)
    artifact_model = artifact_root / "model"
    artifact_model.mkdir(parents=True, exist_ok=True)
    artifact_model.joinpath("model.joblib").write_bytes(model.read_bytes())
    artifact_metadata = artifact_root / "metadata"
    artifact_metadata.mkdir(parents=True, exist_ok=True)
    artifact_metadata.joinpath("round_ctx.json").write_bytes((metadata / "round_ctx.json").read_bytes())
    artifact_metadata.joinpath("objective_meta.json").write_bytes((metadata / "objective_meta.json").read_bytes())
    log = round_dir / "logs" / "round.log.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "ts": f"2026-01-0{round_index + 1}T00:00:00+00:00",
                        "stage": "start",
                        "round": round_index,
                        "campaign": {"slug": "demo", "workdir": str(workdir.resolve())},
                        "data": {"x_column": "X", "y_column": "Y", "label_source": "campaign_history"},
                    }
                ),
                json.dumps(
                    {
                        "ts": f"2026-01-0{round_index + 1}T00:00:01+00:00",
                        "stage": "done",
                        "round": round_index,
                        "run_id": run_id,
                        "campaign": {"slug": "demo", "workdir": str(workdir.resolve())},
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    write_ledger(
        workdir,
        run_id=run_id,
        round_index=round_index,
        artifact_paths_and_hashes={"model/model.joblib": (_sha256(model), str(model.resolve()))},
    )


def _workspace(root: Path, *, round_index: int, run_id: str, with_state: bool) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    records = root / "records.parquet"
    write_records(records, include_opal_cols=True)
    campaign = root / "campaign.yaml"
    write_campaign_yaml(campaign, workdir=root, records_path=records)
    _write_round(root, round_index=round_index, run_id=run_id)
    if with_state:
        write_state(root, records_path=records, run_id=run_id, round_index=round_index)
    return campaign, records


def test_history_import_consolidates_disjoint_rounds_without_retraining(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    _workspace(source, round_index=0, run_id="run-0", with_state=True)
    target_campaign, _ = _workspace(target, round_index=1, run_id="run-1", with_state=False)
    source_model = source / "outputs" / "rounds" / "round_0" / "model" / "model.joblib"
    source_model_sha256 = _sha256(source_model)
    runner = CliRunner()
    app = _build()

    preview = runner.invoke(
        app,
        [
            "--no-color",
            "history",
            "import",
            "-c",
            str(target_campaign),
            "--source-workdir",
            str(source),
            "--json",
        ],
    )

    assert preview.exit_code == 0, preview.output
    preview_payload = json.loads(preview.stdout)
    assert preview_payload["applied"] is False
    assert preview_payload["imported_rounds"] == [0]
    assert preview_payload["existing_rounds"] == [1]
    assert preview_payload["canonical_rounds"] == [0, 1]
    assert not (target / "outputs" / "rounds" / "round_0").exists()
    assert not (target / "state.json").exists()

    applied = runner.invoke(
        app,
        [
            "--no-color",
            "history",
            "import",
            "-c",
            str(target_campaign),
            "--source-workdir",
            str(source),
            "--apply",
            "--json",
        ],
    )

    assert applied.exit_code == 0, applied.output
    payload = json.loads(applied.stdout)
    assert payload["applied"] is True
    receipt_path = Path(payload["receipt_path"])
    assert receipt_path.is_file()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert "state.json" in {entry["path"] for entry in receipt["imported_source_files"]}
    assert any(entry["path"].startswith("outputs/rounds/round_1/") for entry in receipt["existing_target_files"])
    assert {entry["path"] for entry in receipt["canonical_files"]} >= {
        "outputs/rounds/round_0/model/model.joblib",
        "outputs/rounds/round_1/model/model.joblib",
        "state.json",
    }
    state = CampaignState.load(target / "state.json")
    assert [entry.round_index for entry in state.rounds] == [0, 1]
    assert state.workdir == str(target.resolve())
    assert str(source.resolve()) not in json.dumps(state.to_dict())
    runs = read_parquet_df(target / "outputs" / "ledger" / "runs.parquet")
    assert sorted(runs["as_of_round"].astype(int).tolist()) == [0, 1]
    assert str(source.resolve()) not in runs.to_json()
    predictions = read_parquet_df(target / "outputs" / "ledger" / "predictions")
    assert sorted(predictions["as_of_round"].astype(int).unique().tolist()) == [0, 1]
    target_model = target / "outputs" / "rounds" / "round_0" / "model" / "model.joblib"
    assert _sha256(target_model) == source_model_sha256
    target_log = target / "outputs" / "rounds" / "round_0" / "logs" / "round.log.jsonl"
    assert str(source.resolve()) not in target_log.read_text(encoding="utf-8")
    assert (source / "state.json").is_file()


def test_history_import_rejects_artifact_bytes_that_differ_from_the_run_ledger(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    _workspace(source, round_index=0, run_id="run-0", with_state=True)
    target_campaign, _ = _workspace(target, round_index=1, run_id="run-1", with_state=True)
    source.joinpath(
        "outputs",
        "rounds",
        "round_0",
        "run_artifacts",
        "run-0",
        "model",
        "model.joblib",
    ).write_bytes(b"corrupted-model")

    result = CliRunner().invoke(
        _build(),
        [
            "--no-color",
            "history",
            "import",
            "-c",
            str(target_campaign),
            "--source-workdir",
            str(source),
            "--json",
        ],
    )

    assert result.exit_code == 4
    assert "artifact digest differs from run metadata" in result.output


def test_history_import_rejects_a_target_config_that_diverges_from_run_history(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    _workspace(source, round_index=0, run_id="run-0", with_state=True)
    target_campaign, _ = _workspace(target, round_index=1, run_id="run-1", with_state=True)
    config = yaml.safe_load(target_campaign.read_text(encoding="utf-8"))
    config["selection_views"][0]["objective"]["params"]["setpoint_vector"] = [1, 0, 0, 0]
    target_campaign.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = CliRunner().invoke(
        _build(),
        [
            "--no-color",
            "history",
            "import",
            "-c",
            str(target_campaign),
            "--source-workdir",
            str(source),
            "--json",
        ],
    )

    assert result.exit_code == 2
    assert "target campaign config differs from the verified run history" in result.output.lower()
    assert not (target / "outputs" / "rounds" / "round_0").exists()


def test_history_import_rejects_target_x_and_y_columns_that_differ_from_run_history(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    _workspace(source, round_index=0, run_id="run-0", with_state=True)
    target_campaign, _ = _workspace(target, round_index=1, run_id="run-1", with_state=True)
    config = yaml.safe_load(target_campaign.read_text(encoding="utf-8"))
    config["data"]["x_column_name"] = "alternate_X"
    config["data"]["y_column_name"] = "alternate_Y"
    target_campaign.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = CliRunner().invoke(
        _build(),
        [
            "--no-color",
            "history",
            "import",
            "-c",
            str(target_campaign),
            "--source-workdir",
            str(source),
            "--json",
        ],
    )

    assert result.exit_code == 2
    assert "target campaign config differs from the verified run history" in result.output.lower()


def test_history_import_consolidates_the_append_only_label_ledger(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    _workspace(source, round_index=0, run_id="run-0", with_state=True)
    target_campaign, _ = _workspace(target, round_index=1, run_id="run-1", with_state=True)
    write_ledger_labels(source, round_index=0)
    write_ledger_labels(target, round_index=1)

    result = CliRunner().invoke(
        _build(),
        [
            "--no-color",
            "history",
            "import",
            "-c",
            str(target_campaign),
            "--source-workdir",
            str(source),
            "--apply",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    labels = read_parquet_df(target / "outputs" / "ledger" / "labels.parquet")
    assert sorted(labels["observed_round"].astype(int).tolist()) == [0, 1]
    receipt = json.loads(Path(json.loads(result.output)["receipt_path"]).read_text(encoding="utf-8"))
    label_paths = {
        entry["path"] for entry in receipt["canonical_files"] if "outputs/ledger/labels.parquet/" in entry["path"]
    }
    assert len(label_paths) == 2


def test_history_import_merges_identical_label_event_keys_once(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    _workspace(source, round_index=0, run_id="run-0", with_state=True)
    target_campaign, _ = _workspace(target, round_index=1, run_id="run-1", with_state=True)
    write_ledger_labels(source, round_index=0)
    write_ledger_labels(target, round_index=0)

    result = CliRunner().invoke(
        _build(),
        [
            "--no-color",
            "history",
            "import",
            "-c",
            str(target_campaign),
            "--source-workdir",
            str(source),
            "--apply",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    labels = read_parquet_df(target / "outputs" / "ledger" / "labels.parquet")
    assert labels[["id", "observed_round"]].to_dict(orient="records") == [{"id": "a", "observed_round": 0}]


def test_history_import_rejects_conflicting_label_events_with_the_same_immutable_key(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    _workspace(source, round_index=0, run_id="run-0", with_state=True)
    target_campaign, _ = _workspace(target, round_index=1, run_id="run-1", with_state=True)
    write_ledger_labels(source, round_index=0)
    write_ledger_labels(target, round_index=0)
    target_part = next((target / "outputs" / "ledger" / "labels.parquet").rglob("*.parquet"))
    target_labels = read_parquet_df(target_part)
    target_labels["y_obs"] = [[0.9]]
    target_labels.to_parquet(target_part, index=False)

    result = CliRunner().invoke(
        _build(),
        [
            "--no-color",
            "history",
            "import",
            "-c",
            str(target_campaign),
            "--source-workdir",
            str(source),
            "--apply",
            "--json",
        ],
    )

    assert result.exit_code == 2
    assert "conflicting immutable label event" in result.output.lower()
