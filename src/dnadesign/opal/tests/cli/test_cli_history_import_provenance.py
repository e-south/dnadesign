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
from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.src.storage.history_relocation.inspection import inspect_campaign_history
from dnadesign.opal.src.storage.ledger import compact_runs_ledger
from dnadesign.opal.src.storage.parquet_io import read_parquet_df
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
