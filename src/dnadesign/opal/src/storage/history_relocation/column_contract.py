"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/storage/history_relocation/column_contract.py

Loads explicit X/Y identity evidence for campaign snapshots that predate it.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from ...core.utils import ExitCodes, OpalError, file_sha256
from .contracts import HistoryColumnContract, RoundColumnEvidence

_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_ROOT_KEYS = {"schema_version", "campaign_slug", "x_column_name", "y_column_name", "rounds"}
_ROUND_KEYS = {"round_index", "run_id", "round_context_sha256"}


def _canonical_text(value: object, *, field: str) -> str:
    result = str(value or "").strip()
    if not result:
        raise OpalError(f"History column contract field {field} must be non-blank.", ExitCodes.CONTRACT_VIOLATION)
    return result


def load_history_column_contract(path: Path) -> HistoryColumnContract:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise OpalError(f"History column contract must not be a symlink: {candidate}", ExitCodes.BAD_ARGS)
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise OpalError(f"History column contract not found: {resolved}", ExitCodes.BAD_ARGS)
    try:
        payload = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OpalError(f"Invalid history column contract at {resolved}: {exc}") from exc
    if not isinstance(payload, dict) or set(payload) != _ROOT_KEYS:
        raise OpalError("History column contract must contain exactly the declared root fields.")
    if payload["schema_version"] != "opal.history_column_contract.v1":
        raise OpalError("Unsupported history column contract schema version.")
    raw_rounds = payload["rounds"]
    if not isinstance(raw_rounds, list) or not raw_rounds:
        raise OpalError("History column contract must contain round evidence.")
    rounds: list[RoundColumnEvidence] = []
    for item in raw_rounds:
        if not isinstance(item, dict) or set(item) != _ROUND_KEYS:
            raise OpalError("History column contract round evidence has unexpected fields.")
        digest = _canonical_text(item["round_context_sha256"], field="round_context_sha256")
        if not _DIGEST.fullmatch(digest):
            raise OpalError("History column contract round context digest must be lowercase SHA-256.")
        rounds.append(
            RoundColumnEvidence(
                round_index=int(item["round_index"]),
                run_id=_canonical_text(item["run_id"], field="run_id"),
                round_context_sha256=digest,
            )
        )
    identities = {(item.round_index, item.run_id) for item in rounds}
    if len(identities) != len(rounds):
        raise OpalError("History column contract contains duplicate round identities.")
    return HistoryColumnContract(
        campaign_slug=_canonical_text(payload["campaign_slug"], field="campaign_slug"),
        x_column_name=_canonical_text(payload["x_column_name"], field="x_column_name"),
        y_column_name=_canonical_text(payload["y_column_name"], field="y_column_name"),
        rounds=tuple(rounds),
        sha256=file_sha256(resolved),
    )


def columns_for_round(
    contract: HistoryColumnContract,
    *,
    round_index: int,
    run_id: str,
    round_context_sha256: str,
) -> tuple[str, str]:
    matches = [item for item in contract.rounds if item.round_index == round_index and item.run_id == run_id]
    if len(matches) != 1 or matches[0].round_context_sha256 != round_context_sha256:
        raise OpalError(f"History column contract does not bind round {round_index} context bytes.")
    return contract.x_column_name, contract.y_column_name
