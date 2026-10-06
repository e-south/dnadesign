"""Whole-directory moves must preserve history and reject different models/runs."""

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.src.core.utils import OpalError, file_sha256
from dnadesign.opal.src.storage.history_relocation.state_rebinding import rebind_state_paths
from dnadesign.opal.tests._cli_helpers import write_campaign_yaml, write_records, write_state


def _moved(tmp_path: Path):
    old = tmp_path / "old"
    old.mkdir()
    records = tmp_path / "records.parquet"
    write_records(records)
    write_campaign_yaml(old / "campaign.yaml", workdir=old, records_path=records)
    write_state(old, records_path=records, run_id="retained")
    round_dir = old / "outputs/rounds/round_0"
    model = round_dir / "model/model.joblib"
    model.parent.mkdir()
    model.write_bytes(b"frozen model")
    selection = round_dir / "selection/selection_batch.parquet"
    selection.parent.mkdir()
    pd.DataFrame({"run_id": ["retained"], "as_of_round": [0]}).to_parquet(selection)
    state = json.loads((old / "state.json").read_text())
    state["rounds"][0]["model"]["artifact_sha256"] = file_sha256(model)
    state["rounds"][0]["artifacts"]["selection_batch_parquet"] = str(selection)
    state["rounds"][0]["artifacts"]["ledger_labels_parquet"] = str(old / "outputs/ledger/labels.parquet")
    (old / "state.json").write_text(json.dumps(state))
    new = tmp_path / "new"
    old.rename(new)
    write_campaign_yaml(new / "campaign.yaml", workdir=new, records_path=records)
    return old, new


def test_cli_preview_then_apply_preserves_payload_and_original_state(tmp_path):
    old, new = _moved(tmp_path)
    before = (new / "state.json").read_bytes()
    model = new / "outputs/rounds/round_0/model/model.joblib"
    digest = file_sha256(model)
    args = ["history", "rebind-state", "-c", str(new / "campaign.yaml"), "--previous-workdir", str(old), "--json"]
    preview = CliRunner().invoke(_build(), args)
    assert preview.exit_code == 0, preview.output
    assert (new / "state.json").read_bytes() == before
    result = CliRunner().invoke(_build(), [*args, "--apply"])
    assert result.exit_code == 0, result.output
    receipt = json.loads(result.output)
    assert receipt["applied"] is True
    assert (Path(receipt["receipt_path"]).parent / "state-before.json").read_bytes() == before
    state = json.loads((new / "state.json").read_text())
    assert state["workdir"] == str(new)
    assert state["rounds"][0]["model"]["artifact_path"] == str(model)
    assert file_sha256(model) == digest
    assert state["data_location"]["records_path"] == str(tmp_path / "records.parquet")
    assert json.loads(json.dumps(state).replace(str(new), str(old))) == json.loads(before)
    assert not (new / "outputs/ledger/labels.parquet").exists()
    assert next(item for item in receipt["artifacts"] if item["field"] == "ledger_labels_parquet")["exists"] is False


@pytest.mark.parametrize("corruption", ["model", "run", "missing", "symlink", "wrong_root"])
def test_rebinding_rejects_drift_without_changing_state(tmp_path, corruption):
    old, new = _moved(tmp_path)
    before = (new / "state.json").read_bytes()
    selection = new / "outputs/rounds/round_0/selection/selection_batch.parquet"
    if corruption == "model":
        (new / "outputs/rounds/round_0/model/model.joblib").write_bytes(b"wrong")
    elif corruption == "run":
        pd.DataFrame({"run_id": ["different"], "as_of_round": [0]}).to_parquet(selection)
    elif corruption == "missing":
        selection.unlink()
    elif corruption == "symlink":
        target = tmp_path / "outside.parquet"
        selection.rename(target)
        selection.symlink_to(target)
    else:
        old = tmp_path / "incorrect"
    with pytest.raises(OpalError):
        rebind_state_paths(new, old, expected_slug="demo", apply=True)
    assert (new / "state.json").read_bytes() == before
    assert not (new / ".opal.lock").exists()


@pytest.mark.parametrize("apply", [False, True])
def test_rebinding_rejects_symlinked_receipt_directory(tmp_path, apply):
    old, new = _moved(tmp_path)
    before = (new / "state.json").read_bytes()
    outside = tmp_path / "outside"
    outside.mkdir()
    (new / "outputs/history").symlink_to(outside, target_is_directory=True)

    with pytest.raises(OpalError, match="symlink"):
        rebind_state_paths(new, old, expected_slug="demo", apply=apply)

    assert (new / "state.json").read_bytes() == before
    assert list(outside.iterdir()) == []
    assert not (new / ".opal.lock").exists()


def test_rebinding_missing_campaign_does_not_create_it(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(OpalError, match="directory"):
        rebind_state_paths(missing, tmp_path / "old", expected_slug="demo", apply=True)
    assert not missing.exists()


def test_rebinding_failed_state_install_preserves_original_and_allows_retry(tmp_path, monkeypatch):
    from dnadesign.opal.src.storage.history_relocation import state_rebinding

    old, new = _moved(tmp_path)
    before = (new / "state.json").read_bytes()

    def fail_install(*args):
        raise OSError("state replacement failed")

    with monkeypatch.context() as patch:
        patch.setattr(state_rebinding.os, "replace", fail_install)
        with pytest.raises(OSError, match="state replacement failed"):
            rebind_state_paths(new, old, expected_slug="demo", apply=True)
    assert (new / "state.json").read_bytes() == before
    assert list((new / "outputs/history/state-path-rebindings").iterdir()) == []
    assert not (new / ".opal.lock").exists()
    assert rebind_state_paths(new, old, expected_slug="demo", apply=True)["applied"] is True
