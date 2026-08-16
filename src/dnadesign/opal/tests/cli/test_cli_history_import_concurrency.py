"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/tests/cli/test_cli_history_import_concurrency.py

Tests locked state publication and rollback during campaign-history imports.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from dnadesign.opal.src.cli.app import _build
from dnadesign.opal.tests.cli.test_cli_history_import import _workspace


def _tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(path.rglob("*")):
        if file_path.is_file():
            digest.update(file_path.relative_to(path).as_posix().encode())
            digest.update(file_path.read_bytes())
    return digest.hexdigest()


def test_history_import_does_not_restore_stale_state_after_concurrent_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    _workspace(source, round_index=0, run_id="run-0", with_state=True)
    target_campaign, _ = _workspace(target, round_index=1, run_id="run-1", with_state=True)
    state_path = target / "state.json"
    concurrent_state = json.loads(state_path.read_text(encoding="utf-8"))
    concurrent_state["updated_at"] = "2026-08-15T23:59:59+00:00"

    from dnadesign.opal.src.storage.history_relocation import materialization

    stage_history = materialization._stage_history

    def stage_then_drift(*args, **kwargs):
        staged = stage_history(*args, **kwargs)
        state_path.write_text(json.dumps(concurrent_state, sort_keys=True), encoding="utf-8")
        return staged

    monkeypatch.setattr(materialization, "_stage_history", stage_then_drift)

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
    assert "changed while the relocation was staged" in result.output
    assert json.loads(state_path.read_text(encoding="utf-8"))["updated_at"] == concurrent_state["updated_at"]
    assert not (target / "outputs" / "rounds" / "round_0").exists()


def test_history_import_rolls_back_to_the_state_seen_under_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    _workspace(source, round_index=0, run_id="run-0", with_state=True)
    target_campaign, _ = _workspace(target, round_index=1, run_id="run-1", with_state=True)
    state_path = target / "state.json"
    runs_path = target / "outputs" / "ledger" / "runs.parquet"
    runs_before = _tree_digest(runs_path)
    concurrent_state = json.loads(state_path.read_text(encoding="utf-8"))
    concurrent_state["updated_at"] = "2026-08-16T00:00:00+00:00"

    from dnadesign.opal.src.storage.history_relocation import materialization

    stage_history = materialization._stage_history

    def stage_after_concurrent_update(*args, **kwargs):
        state_path.write_text(json.dumps(concurrent_state, sort_keys=True), encoding="utf-8")
        staged = stage_history(*args, **kwargs)
        receipt_path = Path(staged["receipt_path"])
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.write_text("collision", encoding="utf-8")
        return staged

    monkeypatch.setattr(materialization, "_stage_history", stage_after_concurrent_update)

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
    assert "destination already exists" in result.output
    assert json.loads(state_path.read_text(encoding="utf-8"))["updated_at"] == concurrent_state["updated_at"]
    assert _tree_digest(runs_path) == runs_before
    assert not (target / "outputs" / "rounds" / "round_0").exists()


def test_history_import_rolls_back_the_run_ledger_when_the_operator_interrupts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    _workspace(source, round_index=0, run_id="run-0", with_state=True)
    target_campaign, _ = _workspace(target, round_index=1, run_id="run-1", with_state=True)
    runs_path = target / "outputs" / "ledger" / "runs.parquet"
    runs_before = _tree_digest(runs_path)

    from dnadesign.opal.src.storage.history_relocation import materialization

    replace = materialization.os.replace
    imported_round = target / "outputs" / "rounds" / "round_0"

    def replace_then_interrupt(source_path, target_path):
        if Path(target_path) == imported_round:
            raise KeyboardInterrupt
        return replace(source_path, target_path)

    monkeypatch.setattr(materialization.os, "replace", replace_then_interrupt)

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

    assert result.exit_code == 130
    assert _tree_digest(runs_path) == runs_before
    assert not imported_round.exists()


@pytest.mark.parametrize("interruption_target", ["run_ledger", "round", "state"])
def test_history_import_rolls_back_when_interrupted_after_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    interruption_target: str,
) -> None:
    source = tmp_path / "source"
    target = tmp_path / "target"
    _workspace(source, round_index=0, run_id="run-0", with_state=True)
    target_campaign, _ = _workspace(target, round_index=1, run_id="run-1", with_state=True)
    tree_before = _tree_digest(target)
    destinations = {
        "run_ledger": target / "outputs" / "ledger" / "runs.parquet",
        "round": target / "outputs" / "rounds" / "round_0",
        "state": target / "state.json",
    }

    from dnadesign.opal.src.storage.history_relocation import materialization

    replace = materialization.os.replace
    expected_target = destinations[interruption_target]
    interrupted = False

    def replace_then_interrupt(source_path, target_path):
        nonlocal interrupted
        result = replace(source_path, target_path)
        if not interrupted and Path(target_path) == expected_target:
            interrupted = True
            raise KeyboardInterrupt
        return result

    monkeypatch.setattr(materialization.os, "replace", replace_then_interrupt)

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

    assert interrupted is True
    assert result.exit_code == 130
    assert _tree_digest(target) == tree_before
