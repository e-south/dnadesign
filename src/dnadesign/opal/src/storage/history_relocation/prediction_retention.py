"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/storage/history_relocation/prediction_retention.py

Validates and preserves prediction-retention evidence across history relocation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ...core.utils import ExitCodes, OpalError, file_sha256, now_iso
from ...runtime.retention import RETENTION_MANIFEST_SCHEMA_VERSION
from ..prediction_selection import selected_by_any_view
from .contracts import HistoryRelocationPlan

FULL = "full"
SELECTED_HISTORY = "selected_history"
HISTORY_RETENTION_SCHEMA_VERSION = "opal.history_prediction_retention.v1"


@dataclass(frozen=True, slots=True)
class PredictionRetentionEvidence:
    mode_by_round: Mapping[int, str]
    manifest_path: Path | None
    run_id_by_round: Mapping[int, str]
    scored_rows_by_round: Mapping[int, int]
    retained_rows_by_round: Mapping[int, int]


def prediction_dataset_sha256(path: Path) -> str:
    """Digest a prediction dataset using its runtime retention contract."""

    files = sorted(candidate for candidate in path.rglob("*.parquet") if candidate.is_file())
    if not files:
        raise OpalError(
            f"Retained prediction ledger has no parquet parts: {path}",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return "sha256:" + "|".join(file_sha256(file) for file in files)


def load_prediction_retention(*, workdir: Path, rounds: set[int]) -> PredictionRetentionEvidence:
    """Load exact full-versus-selected-history semantics for every round."""

    manifest_path = workdir / "outputs" / "retention_manifest.json"
    if not manifest_path.is_file():
        return PredictionRetentionEvidence(
            mode_by_round={round_index: FULL for round_index in rounds},
            manifest_path=None,
            run_id_by_round={},
            scored_rows_by_round={},
            retained_rows_by_round={},
        )
    payload = _load_manifest(manifest_path)
    if payload.get("schema_version") == HISTORY_RETENTION_SCHEMA_VERSION:
        return _history_relocation_retention(
            payload,
            manifest_path=manifest_path,
            prediction_root=workdir / "outputs" / "ledger" / "predictions",
            rounds=rounds,
        )
    if payload.get("schema_version") != RETENTION_MANIFEST_SCHEMA_VERSION:
        raise OpalError("Prediction retention manifest schema is unsupported.")
    policy = _mapping(payload.get("policy"), label="retention policy")
    prediction_policy = str(policy.get("prediction_ledger") or "")
    if prediction_policy == "all_rounds_full":
        return PredictionRetentionEvidence(
            mode_by_round={round_index: FULL for round_index in rounds},
            manifest_path=manifest_path,
            run_id_by_round={},
            scored_rows_by_round={},
            retained_rows_by_round={},
        )
    if prediction_policy not in {"latest_full_plus_selected_history", "selected_history_only"}:
        raise OpalError(
            f"Retention manifest has an unsupported prediction policy: {prediction_policy!r}",
            ExitCodes.CONTRACT_VIOLATION,
        )
    action = _prediction_action(payload)
    if str(action.get("policy")) != prediction_policy:
        raise OpalError("Retention manifest action and policy disagree.", ExitCodes.CONTRACT_VIOLATION)
    prediction_root = workdir / "outputs" / "ledger" / "predictions"
    observed_sha256 = prediction_dataset_sha256(prediction_root)
    if str(action.get("sha256")) != observed_sha256:
        raise OpalError("Retention manifest prediction-ledger digest mismatch.", ExitCodes.CONTRACT_VIOLATION)
    observed_rows = sum(
        len(pd.read_parquet(part, columns=["run_id"])) for part in sorted(prediction_root.rglob("*.parquet"))
    )
    if _exact_nonnegative_int(action.get("rows_after"), label="retention rows_after") != observed_rows:
        raise OpalError("Retention manifest prediction row count mismatch.", ExitCodes.CONTRACT_VIOLATION)
    full_rounds = {
        _exact_nonnegative_int(value, label="retained full round")
        for value in _list(action.get("retained_full_rounds"), label="retained_full_rounds")
    }
    if full_rounds - rounds:
        raise OpalError("Retention manifest names prediction rounds absent from the ledger.")
    if prediction_policy == "selected_history_only" and full_rounds:
        raise OpalError("selected_history_only retention cannot declare full rounds.")
    if prediction_policy == "latest_full_plus_selected_history" and max(rounds) not in full_rounds:
        raise OpalError("latest_full_plus_selected_history must retain the latest round in full.")
    return PredictionRetentionEvidence(
        mode_by_round={
            round_index: (FULL if round_index in full_rounds else SELECTED_HISTORY) for round_index in rounds
        },
        manifest_path=manifest_path,
        run_id_by_round={},
        scored_rows_by_round={},
        retained_rows_by_round={},
    )


def validate_prediction_retention(
    frame: pd.DataFrame,
    *,
    expected_scored_rows: int,
    mode: str,
    label: str,
) -> None:
    """Validate one run against its declared prediction-retention mode."""

    observed_rows = len(frame)
    if mode == FULL:
        if observed_rows != expected_scored_rows:
            raise OpalError(
                f"{label} prediction count differs from run metadata: "
                f"predictions={observed_rows}, expected={expected_scored_rows}.",
                ExitCodes.CONTRACT_VIOLATION,
            )
        return
    if mode != SELECTED_HISTORY:
        raise OpalError(f"{label} has an unsupported prediction retention mode: {mode!r}.")
    if observed_rows <= 0 or observed_rows > expected_scored_rows:
        raise OpalError(
            f"{label} selected prediction history has an invalid row count: "
            f"predictions={observed_rows}, originally_scored={expected_scored_rows}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    if "pred__selection_views" not in frame:
        raise OpalError(f"{label} selected prediction history is missing selection memberships.")
    if not frame["pred__selection_views"].map(selected_by_any_view).all():
        raise OpalError(f"{label} selected prediction history contains an unselected candidate.")


def stage_prediction_retention_manifest(
    plan: HistoryRelocationPlan,
    *,
    prediction_files: Mapping[Path, Path],
    staging_root: Path,
) -> Path:
    """Write one canonical retention manifest for the relocated history."""

    runs = sorted((*plan.source.runs, *plan.target.runs), key=lambda item: item.round_index)
    rows_before = sum(int(run.run_row["stats__n_scored"]) for run in runs)
    rows_after = sum(run.prediction_row_count for run in runs)
    source_manifests = []
    for history in (plan.source, plan.target):
        if history.retention_manifest is not None:
            source_manifests.append(
                {
                    "path": history.retention_manifest.relative_to(history.workdir).as_posix(),
                    "sha256": file_sha256(history.retention_manifest),
                }
            )
    payload = {
        "schema_version": HISTORY_RETENTION_SCHEMA_VERSION,
        "generated_at": now_iso(),
        "campaign": {"slug": plan.campaign_slug, "workdir": str(plan.target.workdir)},
        "prediction_ledger": {
            "path": str((plan.target.workdir / "outputs" / "ledger" / "predictions").resolve()),
            "rows_before_retention": rows_before,
            "rows_retained": rows_after,
            "sha256": _prediction_file_set_sha256(prediction_files),
        },
        "rounds": [
            {
                "round_index": run.round_index,
                "run_id": run.run_id,
                "mode": run.prediction_retention,
                "rows_scored": int(run.run_row["stats__n_scored"]),
                "rows_retained": run.prediction_row_count,
            }
            for run in runs
        ],
        "source_manifests": source_manifests,
        "status": "PASS",
    }
    output = staging_root / "outputs" / "retention_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    return output


def _prediction_file_set_sha256(prediction_files: Mapping[Path, Path]) -> str:
    if not prediction_files:
        raise OpalError("Canonical prediction history contains no parquet files.")
    return "sha256:" + "|".join(file_sha256(prediction_files[destination]) for destination in sorted(prediction_files))


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OpalError(f"Invalid prediction retention manifest at {path}: {exc}") from exc
    manifest = _mapping(payload, label="retention manifest")
    if manifest.get("status") != "PASS":
        raise OpalError("Prediction retention manifest status is invalid.")
    return manifest


def _history_relocation_retention(
    payload: Mapping[str, Any],
    *,
    manifest_path: Path,
    prediction_root: Path,
    rounds: set[int],
) -> PredictionRetentionEvidence:
    ledger = _mapping(payload.get("prediction_ledger"), label="history prediction ledger")
    if str(ledger.get("sha256")) != prediction_dataset_sha256(prediction_root):
        raise OpalError("History prediction-retention ledger digest mismatch.")
    observed_rows = sum(
        len(pd.read_parquet(part, columns=["run_id"])) for part in sorted(prediction_root.rglob("*.parquet"))
    )
    if _exact_nonnegative_int(ledger.get("rows_retained"), label="history retained rows") != observed_rows:
        raise OpalError("History prediction-retention row count mismatch.")
    entries = [_mapping(item, label="history retained round") for item in _list(payload.get("rounds"), label="rounds")]
    mode_by_round: dict[int, str] = {}
    run_id_by_round: dict[int, str] = {}
    scored_rows_by_round: dict[int, int] = {}
    retained_rows_by_round: dict[int, int] = {}
    for entry in entries:
        round_index = _exact_nonnegative_int(entry.get("round_index"), label="history retained round index")
        mode = str(entry.get("mode"))
        if mode not in {FULL, SELECTED_HISTORY} or round_index in mode_by_round:
            raise OpalError("History prediction-retention round entries are invalid.")
        run_id = str(entry.get("run_id") or "").strip()
        if not run_id:
            raise OpalError("History prediction-retention round run_id is invalid.")
        mode_by_round[round_index] = mode
        run_id_by_round[round_index] = run_id
        scored_rows_by_round[round_index] = _exact_nonnegative_int(
            entry.get("rows_scored"), label="history scored rows"
        )
        retained_rows_by_round[round_index] = _exact_nonnegative_int(
            entry.get("rows_retained"), label="history retained round rows"
        )
    if set(mode_by_round) != rounds:
        raise OpalError("History prediction-retention rounds differ from the ledgers.")
    if sum(retained_rows_by_round.values()) != observed_rows:
        raise OpalError("History prediction-retention per-round row counts disagree with the ledger total.")
    return PredictionRetentionEvidence(
        mode_by_round=mode_by_round,
        manifest_path=manifest_path,
        run_id_by_round=run_id_by_round,
        scored_rows_by_round=scored_rows_by_round,
        retained_rows_by_round=retained_rows_by_round,
    )


def _prediction_action(payload: Mapping[str, Any]) -> dict[str, Any]:
    actions = [
        _mapping(item, label="retention action") for item in _list(payload.get("actions"), label="retention actions")
    ]
    matching = [item for item in actions if item.get("kind") == "prediction_ledger_compaction"]
    if len(matching) != 1:
        raise OpalError("Retention manifest must contain one prediction-ledger compaction action.")
    return matching[0]


def _mapping(value: object, *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OpalError(f"{label} must be an object.")
    return {str(key): item for key, item in value.items()}


def _list(value: object, *, label: str) -> list[Any]:
    if not isinstance(value, list):
        raise OpalError(f"{label} must be a list.")
    return value


def _exact_nonnegative_int(value: object, *, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OpalError(f"{label} must be a non-negative integer.")
    return value


__all__ = [
    "FULL",
    "SELECTED_HISTORY",
    "PredictionRetentionEvidence",
    "load_prediction_retention",
    "prediction_dataset_sha256",
    "stage_prediction_retention_manifest",
    "validate_prediction_retention",
]
