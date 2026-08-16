"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/cli/test_cli_history_import_provenance.py

Exercises immutable provenance boundaries during OPAL history relocation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import yaml
from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.src.config.loader import load_config
from dnadesign.opal.src.runtime.retention import apply_runtime_artifact_retention
from dnadesign.opal.src.storage.history_relocation.inspection import inspect_campaign_history
from dnadesign.opal.src.storage.ledger import compact_runs_ledger
from dnadesign.opal.src.storage.parquet_io import read_parquet_df
from dnadesign.opal.src.storage.workspace import CampaignWorkspace
from dnadesign.opal.tests.cli.test_cli_history_import import _workspace


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _invoke_import(source: Path, target_campaign: Path):
    return CliRunner().invoke(
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


def _apply_selected_history_retention(campaign: Path) -> None:
    payload = yaml.safe_load(campaign.read_text(encoding="utf-8"))
    payload["artifact_retention"] = {
        "mode": "production_review",
        "prediction_ledger": "selected_history_only",
        "plot_tidy_data": "full",
        "model_artifacts": "all",
        "tabular_format": "parquet",
        "max_estimated_bytes": 1_000_000,
        "fail_if_estimate_exceeds": True,
        "final_round": None,
    }
    campaign.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    cfg = load_config(campaign)
    apply_runtime_artifact_retention(cfg, CampaignWorkspace.from_config(cfg, campaign))


def _add_artifact_receipt(workdir: Path, *, round_index: int, run_id: str, key: str) -> bytes:
    payload = f"{run_id}:{key}".encode()
    artifact = workdir / "outputs" / "rounds" / f"round_{round_index}" / "run_artifacts" / run_id / key
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(payload)
    run_part = next((workdir / "outputs" / "ledger" / "runs.parquet").glob("*.parquet"))
    frame = read_parquet_df(run_part)
    receipts = dict(frame.at[0, "artifacts"])
    receipts[key] = (_sha256(artifact), str(artifact.resolve()))
    frame.at[0, "artifacts"] = receipts
    frame.to_parquet(run_part, index=False)
    return payload


def _remove_embedded_columns(workdir: Path, *, round_index: int, run_id: str) -> dict[str, object]:
    round_dir = workdir / "outputs" / "rounds" / f"round_{round_index}"
    mirror = round_dir / "metadata" / "round_ctx.json"
    snapshot = round_dir / "run_artifacts" / run_id / "metadata" / "round_ctx.json"
    context = json.loads(snapshot.read_text(encoding="utf-8"))
    context.pop("core/data/x_column_name")
    context.pop("core/data/y_column_name")
    payload = json.dumps(context, sort_keys=True).encode()
    mirror.write_bytes(payload)
    snapshot.write_bytes(payload)
    run_part = next((workdir / "outputs" / "ledger" / "runs.parquet").glob("*.parquet"))
    frame = read_parquet_df(run_part)
    receipts = dict(frame.at[0, "artifacts"])
    receipts["metadata/round_ctx.json"] = (_sha256(snapshot), str(mirror.resolve()))
    frame.at[0, "artifacts"] = receipts
    frame.to_parquet(run_part, index=False)
    return {
        "round_index": round_index,
        "run_id": run_id,
        "round_context_sha256": _sha256(snapshot),
    }


def test_history_import_reconstructs_mutable_round_mirrors_from_the_verified_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    _workspace(source, round_index=0, run_id="run-0", with_state=True)
    target_campaign, _ = _workspace(target, round_index=1, run_id="run-1", with_state=False)
    source_round = source / "outputs" / "rounds" / "round_0"
    source_round.joinpath("model", "model.joblib").write_bytes(b"mutable-model-drift")
    mutable_selections = pd.read_parquet(source_round / "selection" / "selections.parquet")
    mutable_selections.loc[:, "id"] = "mutable-selection-drift"
    mutable_selections.to_parquet(source_round / "selection" / "selections.parquet", index=False)

    result = _invoke_import(source, target_campaign)

    assert result.exit_code == 0, result.output
    imported_round = target / "outputs" / "rounds" / "round_0"
    snapshot = imported_round / "run_artifacts" / "run-0"
    assert (
        imported_round.joinpath("model", "model.joblib").read_bytes()
        == snapshot.joinpath("model", "model.joblib").read_bytes()
    )
    assert pd.read_parquet(imported_round / "selection" / "selections.parquet").equals(
        pd.read_parquet(snapshot / "selection" / "selections.parquet")
    )


def test_history_import_reads_x_and_y_identity_from_the_verified_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    _workspace(source, round_index=0, run_id="run-0", with_state=True)
    target_campaign, _ = _workspace(target, round_index=1, run_id="run-1", with_state=False)
    log_path = source / "outputs" / "rounds" / "round_0" / "logs" / "round.log.jsonl"
    events = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    events[0]["data"] = {"x_column": "mutable_X", "y_column": "mutable_Y"}
    log_path.write_text("\n".join(json.dumps(event) for event in events) + "\n", encoding="utf-8")

    result = _invoke_import(source, target_campaign)

    assert result.exit_code == 0, result.output


def test_history_import_uses_an_explicit_contract_for_pre_column_identity_snapshots(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    _workspace(source, round_index=0, run_id="run-0", with_state=True)
    target_campaign, _ = _workspace(target, round_index=1, run_id="run-1", with_state=False)
    evidence = [
        _remove_embedded_columns(source, round_index=0, run_id="run-0"),
        _remove_embedded_columns(target, round_index=1, run_id="run-1"),
    ]
    contract = tmp_path / "history-column-contract.json"
    contract.write_text(
        json.dumps(
            {
                "schema_version": "opal.history_column_contract.v1",
                "campaign_slug": "demo",
                "x_column_name": "X",
                "y_column_name": "Y",
                "rounds": evidence,
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )

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
            "--column-contract",
            str(contract),
            "--apply",
            "--json",
        ],
    )

    assert result.exit_code == 0, result.output
    receipt = json.loads(Path(json.loads(result.output)["receipt_path"]).read_text(encoding="utf-8"))
    assert receipt["column_contract_sha256"] == _sha256(contract)


def test_history_import_preserves_distinct_artifact_receipt_keys_in_both_run_rows(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    _workspace(source, round_index=0, run_id="run-0", with_state=True)
    target_campaign, _ = _workspace(target, round_index=1, run_id="run-1", with_state=False)
    source_payload = _add_artifact_receipt(
        source,
        round_index=0,
        run_id="run-0",
        key="analysis/source-only.json",
    )
    target_payload = _add_artifact_receipt(
        target,
        round_index=1,
        run_id="run-1",
        key="analysis/target-only.json",
    )

    result = _invoke_import(source, target_campaign)

    assert result.exit_code == 0, result.output
    rows = read_parquet_df(target / "outputs" / "ledger" / "runs.parquet").set_index("run_id")
    source_receipt = rows.at["run-0", "artifacts"]["analysis/source-only.json"]
    target_receipt = rows.at["run-1", "artifacts"]["analysis/target-only.json"]
    assert source_receipt[0] == hashlib.sha256(source_payload).hexdigest()
    assert target_receipt[0] == hashlib.sha256(target_payload).hexdigest()
    runs_path = target / "outputs" / "ledger" / "runs.parquet"
    shutil.copyfile(next(runs_path.glob("*.parquet")), runs_path / "part-duplicate.parquet")
    assert compact_runs_ledger(runs_path) == {
        "duplicates_removed": 1,
        "rows_after": 2,
        "rows_before": 3,
    }
    history = inspect_campaign_history(target, label="Relocated campaign")
    assert history.rounds == (0, 1)
    assert len(list(runs_path.glob("*.parquet"))) == 1


def test_history_import_projects_compacted_prediction_parts_by_run(tmp_path: Path) -> None:
    source = tmp_path / "source"
    canonical = tmp_path / "canonical"
    future = tmp_path / "future"
    _workspace(source, round_index=0, run_id="run-0", with_state=True)
    canonical_campaign, _ = _workspace(canonical, round_index=1, run_id="run-1", with_state=False)
    first_import = _invoke_import(source, canonical_campaign)
    assert first_import.exit_code == 0, first_import.output
    _apply_selected_history_retention(canonical_campaign)
    future_campaign, _ = _workspace(future, round_index=2, run_id="run-2", with_state=False)

    result = _invoke_import(canonical, future_campaign)

    assert result.exit_code == 0, result.output
    receipt = json.loads(Path(json.loads(result.output)["receipt_path"]).read_text(encoding="utf-8"))
    projections = [item for item in receipt["transformations"] if item["kind"] == "prediction_run_projection"]
    assert {(item["round_index"], item["run_id"]) for item in projections} == {(0, "run-0"), (1, "run-1")}
    history = inspect_campaign_history(future, label="Relocated campaign")
    assert history.rounds == (0, 1, 2)
    for run in history.runs:
        frame = pd.concat([read_parquet_df(part) for part in run.prediction_parts], ignore_index=True)
        assert set(zip(frame["as_of_round"].astype(int), frame["run_id"].astype(str), strict=True)) == {
            (run.round_index, run.run_id)
        }
        assert len(frame) == (2 if run.round_index == 2 else 1)


def test_history_import_accepts_retention_selected_prediction_history(tmp_path: Path) -> None:
    source = tmp_path / "source"
    canonical = tmp_path / "canonical"
    future = tmp_path / "future"
    _workspace(source, round_index=0, run_id="run-0", with_state=True)
    canonical_campaign, _ = _workspace(canonical, round_index=1, run_id="run-1", with_state=False)
    first_import = _invoke_import(source, canonical_campaign)
    assert first_import.exit_code == 0, first_import.output
    _apply_selected_history_retention(canonical_campaign)
    retained = read_parquet_df(canonical / "outputs" / "ledger" / "predictions")
    assert len(retained) == 2
    future_campaign, _ = _workspace(future, round_index=2, run_id="run-2", with_state=False)

    result = _invoke_import(canonical, future_campaign)

    assert result.exit_code == 0, result.output
    history = inspect_campaign_history(future, label="Relocated campaign")
    assert history.rounds == (0, 1, 2)
    assert [(run.round_index, run.prediction_retention) for run in history.runs] == [
        (0, "selected_history"),
        (1, "selected_history"),
        (2, "full"),
    ]


def test_history_import_preserves_round_scoped_mixed_retention(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    _workspace(source, round_index=0, run_id="run-0", with_state=True)
    target_campaign, _ = _workspace(target, round_index=1, run_id="run-1", with_state=False)
    _apply_selected_history_retention(target_campaign)

    result = _invoke_import(source, target_campaign)

    assert result.exit_code == 0, result.output
    history = inspect_campaign_history(target, label="Relocated campaign")
    assert [(run.round_index, run.prediction_retention) for run in history.runs] == [
        (0, "full"),
        (1, "selected_history"),
    ]


def test_history_import_rejects_a_tampered_retention_ledger_digest(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    source_campaign, _ = _workspace(source, round_index=0, run_id="run-0", with_state=True)
    _apply_selected_history_retention(source_campaign)
    manifest_path = source / "outputs" / "retention_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["actions"][0]["sha256"] = "sha256:" + "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    target_campaign, _ = _workspace(target, round_index=1, run_id="run-1", with_state=False)

    result = _invoke_import(source, target_campaign)

    assert result.exit_code == 4
    assert "retention manifest prediction-ledger digest mismatch" in result.output.lower()


def test_history_import_keeps_inspection_fields_out_of_canonical_run_rows(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    _workspace(source, round_index=0, run_id="run-0", with_state=True)
    target_campaign, _ = _workspace(target, round_index=1, run_id="run-1", with_state=False)

    result = _invoke_import(source, target_campaign)

    assert result.exit_code == 0, result.output
    for part in sorted((target / "outputs" / "ledger" / "runs.parquet").glob("*.parquet")):
        columns = set(read_parquet_df(part).columns)
        assert "data__x_column_name" not in columns
        assert "data__y_column_name" not in columns
