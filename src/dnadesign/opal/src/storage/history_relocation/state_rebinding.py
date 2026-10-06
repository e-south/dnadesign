"""Explicit path-only repair of state after a whole campaign directory moves."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import pandas as pd

from ...core.utils import OpalError, file_sha256
from ..locks import CampaignLock
from ..state import CampaignState


def _destination(value: str, *, previous: Path, current: Path, require_exists: bool = True) -> Path:
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise OpalError("State artifact paths must be absolute and canonical.")
    try:
        relative = path.relative_to(previous)
    except ValueError as exc:
        raise OpalError(f"State artifact is outside the declared previous workdir: {path}") from exc
    target = current / relative
    if any(p.is_symlink() for p in (target, *target.parents)):
        raise OpalError("State rebinding cannot traverse symlinks.")
    if (require_exists and not target.exists()) or not target.resolve().is_relative_to(current):
        raise OpalError(f"Moved state artifact is missing or escapes the campaign: {target}")
    return target


def _campaign_roots(workdir: Path, previous_workdir: Path) -> tuple[Path, Path]:
    current = workdir.absolute()
    previous = previous_workdir.absolute()
    if any(p.is_symlink() for p in (current, *current.parents)) or current == previous:
        raise OpalError("State rebinding requires distinct, non-symlink campaign roots.")
    if ".." in current.parts or ".." in previous.parts:
        raise OpalError("State rebinding requires canonical campaign roots without '..'.")
    if not current.is_dir():
        raise OpalError(f"Moved campaign directory does not exist: {current}")
    return current, previous


def _receipt_path(current: Path, state_digest: str) -> Path:
    return _destination(
        str(current / "outputs/history/state-path-rebindings" / state_digest),
        previous=current,
        current=current,
        require_exists=False,
    )


def plan_state_rebinding(workdir: Path, previous_workdir: Path, *, expected_slug: str) -> tuple[dict, bytes, dict]:
    current, previous = _campaign_roots(workdir, previous_workdir)
    state_path = current / "state.json"
    if state_path.is_symlink():
        raise OpalError("State rebinding cannot read a symlinked state file.")
    CampaignState.load(state_path)
    before = state_path.read_bytes()
    state = json.loads(before)
    if state["campaign_slug"] != expected_slug or state["workdir"] != str(previous):
        raise OpalError("State campaign identity or declared previous workdir differs.")
    if not state["rounds"]:
        raise OpalError("State rebinding requires retained completed rounds.")
    changed = []
    for entry in state["rounds"]:
        if entry["status"] != "completed":
            raise OpalError("Only completed retained state rounds can be rebound.")
        model = _destination(entry["model"]["artifact_path"], previous=previous, current=current)
        if file_sha256(model) != entry["model"]["artifact_sha256"]:
            raise OpalError("Moved model digest differs from the retained state.")
        entry["model"]["artifact_path"] = str(model)
        entry["round_dir"] = str(_destination(entry["round_dir"], previous=previous, current=current))
        for key, value in entry["artifacts"].items():
            # State records this storage locator even when external sidecar labels
            # mean no campaign label-event ledger has ever been written.
            target = _destination(
                value, previous=previous, current=current, require_exists=key != "ledger_labels_parquet"
            )
            entry["artifacts"][key] = str(target)
            changed.append(
                {
                    "field": key,
                    "round_index": entry["round_index"],
                    "relative_path": target.relative_to(current).as_posix(),
                    "exists": target.exists(),
                    "sha256": file_sha256(target) if target.is_file() else None,
                }
            )
        selection = pd.read_parquet(entry["artifacts"]["selection_batch_parquet"], columns=["run_id", "as_of_round"])
        if (
            selection.empty
            or set(selection.run_id) != {entry["run_id"]}
            or set(selection.as_of_round) != {entry["round_index"]}
        ):
            raise OpalError("Moved selection identity differs from the retained state.")
    state["workdir"] = str(current)
    after = json.dumps(state, indent=2, sort_keys=True) + "\n"
    receipt = {
        "schema_version": "opal.state_path_rebinding.v1",
        "campaign_slug": expected_slug,
        "previous_workdir": str(previous),
        "current_workdir": str(current),
        "state_before_sha256": hashlib.sha256(before).hexdigest(),
        "state_after_sha256": hashlib.sha256(after.encode()).hexdigest(),
        "rounds": [entry["round_index"] for entry in state["rounds"]],
        "artifacts": changed,
        "scope": "state_paths_only; immutable run artifacts and ledgers unchanged",
    }
    target = _receipt_path(current, receipt["state_before_sha256"])
    if target.exists():
        raise OpalError("A receipt already exists for this original state; inspect it before rebinding.")
    return state, before, receipt


def rebind_state_paths(workdir: Path, previous_workdir: Path, *, expected_slug: str, apply: bool = False) -> dict:
    """Verify moved model/selection identities; optionally rewrite only state paths."""
    current, previous = _campaign_roots(workdir, previous_workdir)
    with CampaignLock(current):
        state, before, receipt = plan_state_rebinding(current, previous, expected_slug=expected_slug)
        if not apply:
            return {**receipt, "applied": False}
        target = _receipt_path(current, receipt["state_before_sha256"])
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".rebind-", dir=current) as tmp:
            stage = Path(tmp)
            (stage / "state.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
            CampaignState.load(stage / "state.json")
            bundle = stage / "receipt"
            bundle.mkdir()
            (bundle / "state-before.json").write_bytes(before)
            (bundle / "manifest.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            bundle.rename(target)
            try:
                os.replace(stage / "state.json", current / "state.json")
            except BaseException:
                shutil.rmtree(target)
                raise
        return {**receipt, "applied": True, "receipt_path": str(target / "manifest.json")}
