"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/storage/history_relocation/inspection.py

Inspects OPAL histories and rejects incompatible or discontinuous relocations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd

from ...core.utils import ExitCodes, OpalError, file_sha256
from ..parquet_io import read_parquet_df
from ..state import CampaignState
from .column_contract import columns_for_round
from .contracts import CampaignHistory, HistoryColumnContract, HistoryRelocationPlan, RunHistory
from .prediction_ledger import prediction_rows_for_run
from .prediction_retention import FULL, load_prediction_retention, validate_prediction_retention

_RUN_INVARIANT_FIELDS = (
    "data__x_column_name",
    "data__y_column_name",
    "x_transform__name",
    "x_transform__params",
    "y_ingest__name",
    "y_ingest__params",
    "objective__defs_json",
    "schema__version",
)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [jsonable(item) for item in list(value)]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if pd.isna(value):
            return None
        return float(value)
    if isinstance(value, np.generic):
        return jsonable(value.item())
    return str(value)


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _canonical_root(path: Path, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_symlink():
        raise OpalError(f"{label} must not be a symlink: {candidate}", ExitCodes.BAD_ARGS)
    resolved = candidate.resolve()
    if not resolved.is_dir():
        raise OpalError(f"{label} directory not found: {resolved}", ExitCodes.BAD_ARGS)
    return resolved


def _part_identities(path: Path, *, label: str) -> list[tuple[int, str, int]]:
    frame = read_parquet_df(path, columns=["as_of_round", "run_id"])
    if frame.empty:
        raise OpalError(f"{label} part is empty: {path}", ExitCodes.CONTRACT_VIOLATION)
    counts: dict[tuple[int, str], int] = {}
    for round_value, run_value in zip(frame["as_of_round"], frame["run_id"], strict=True):
        identity = (int(round_value), str(run_value))
        counts[identity] = counts.get(identity, 0) + 1
    return [(round_index, run_id, counts[(round_index, run_id)]) for round_index, run_id in sorted(counts)]


def _parts_by_round(root: Path, *, label: str) -> dict[int, list[tuple[Path, str, int]]]:
    if not root.is_dir():
        raise OpalError(f"{label} dataset not found: {root}", ExitCodes.CONTRACT_VIOLATION)
    result: dict[int, list[tuple[Path, str, int]]] = {}
    for path in sorted(root.rglob("*.parquet")):
        for round_index, run_id, row_count in _part_identities(path, label=label):
            result.setdefault(round_index, []).append((path, run_id, row_count))
    if not result:
        raise OpalError(f"{label} dataset is empty: {root}", ExitCodes.CONTRACT_VIOLATION)
    return result


def _read_round_context(round_dir: Path) -> dict[str, Any]:
    path = round_dir / "metadata" / "round_ctx.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OpalError(f"Invalid round context at {path}: {exc}", ExitCodes.CONTRACT_VIOLATION) from exc
    if not isinstance(payload, dict):
        raise OpalError(f"Round context must be an object: {path}", ExitCodes.CONTRACT_VIOLATION)
    return payload


def _read_run_column_contract(
    context: dict[str, Any],
    *,
    round_index: int,
    run_id: str,
    round_context_sha256: str,
    column_contract: HistoryColumnContract | None,
) -> tuple[str, str]:
    x_column = str(context.get("core/data/x_column_name") or "").strip()
    y_column = str(context.get("core/data/y_column_name") or "").strip()
    if x_column and y_column:
        return x_column, y_column
    if x_column or y_column:
        raise OpalError(f"Round {round_index} immutable context has an incomplete X/Y column identity.")
    if column_contract is None:
        raise OpalError(
            f"Round {round_index} predates embedded X/Y identity; pass an explicit history column contract.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return columns_for_round(
        column_contract,
        round_index=round_index,
        run_id=run_id,
        round_context_sha256=round_context_sha256,
    )


def run_artifact_root(round_dir: Path, *, run_id: str) -> Path:
    root = round_dir / "run_artifacts"
    candidates = sorted(path for path in root.iterdir() if path.is_dir()) if root.is_dir() else []
    matches: list[Path] = []
    for candidate in candidates:
        labels_path = candidate / "labels" / "labels_used.parquet"
        if not labels_path.is_file():
            continue
        frame = read_parquet_df(labels_path, columns=["run_id"])
        if set(frame["run_id"].astype(str).tolist()) == {run_id}:
            matches.append(candidate)
    if len(matches) != 1:
        raise OpalError(f"Round directory must contain one immutable artifact snapshot for run_id={run_id}.")
    return matches[0].resolve()


def verified_artifact_path(root: Path, *, artifact_key: str) -> Path:
    key = str(artifact_key)
    logical = PurePosixPath(key)
    if (
        not key
        or key != key.strip()
        or logical.is_absolute()
        or "\\" in key
        or any(part in {"", ".", ".."} for part in logical.parts)
    ):
        raise OpalError(f"Run artifact key must be a canonical relative path: {artifact_key!r}.")
    path = (root / Path(*logical.parts)).resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise OpalError(f"Run artifact path is outside its immutable snapshot: {path}.") from exc
    return path


def _verify_run_artifacts(round_dir: Path, *, run_id: str, artifacts: Any, label: str) -> None:
    recorded = jsonable(artifacts)
    if not isinstance(recorded, dict) or not recorded:
        raise OpalError(f"{label} run metadata has no immutable artifact receipts.", ExitCodes.CONTRACT_VIOLATION)
    artifact_root = run_artifact_root(round_dir, run_id=run_id)
    for artifact_key, receipt in recorded.items():
        if receipt is None:
            continue
        if not isinstance(receipt, list) or len(receipt) != 2:
            raise OpalError(
                f"{label} artifact receipt is invalid for {artifact_key!r}.",
                ExitCodes.CONTRACT_VIOLATION,
            )
        expected_sha256 = str(receipt[0])
        artifact_path = verified_artifact_path(artifact_root, artifact_key=str(artifact_key))
        if not artifact_path.is_file():
            raise OpalError(
                f"{label} immutable artifact is missing for {artifact_key!r}.",
                ExitCodes.CONTRACT_VIOLATION,
            )
        observed_sha256 = file_sha256(artifact_path)
        if observed_sha256 != expected_sha256:
            raise OpalError(
                f"{label} artifact digest differs from run metadata for {artifact_key!r}: "
                f"expected={expected_sha256}, observed={observed_sha256}.",
                ExitCodes.CONTRACT_VIOLATION,
            )


def inspect_campaign_history(
    workdir: Path,
    *,
    label: str,
    column_contract: HistoryColumnContract | None = None,
) -> CampaignHistory:
    root = _canonical_root(workdir, label=label)
    outputs = root / "outputs"
    run_parts = _parts_by_round(outputs / "ledger" / "runs.parquet", label=f"{label} run ledger")
    prediction_parts = _parts_by_round(outputs / "ledger" / "predictions", label=f"{label} prediction ledger")
    if set(run_parts) != set(prediction_parts):
        raise OpalError(
            f"{label} run/prediction rounds differ: runs={sorted(run_parts)}, predictions={sorted(prediction_parts)}",
            ExitCodes.CONTRACT_VIOLATION,
        )
    retention = load_prediction_retention(workdir=root, rounds=set(run_parts))
    runs: list[RunHistory] = []
    campaign_slug: str | None = None
    for round_index in sorted(run_parts):
        if len(run_parts[round_index]) != 1:
            raise OpalError(
                f"{label} round {round_index} has multiple run ledger parts; "
                "compact or select one run before relocation.",
                ExitCodes.CONTRACT_VIOLATION,
            )
        run_part, run_id, run_row_count = run_parts[round_index][0]
        if run_row_count != 1:
            raise OpalError(
                f"{label} run ledger part must contain one metadata row: {run_part}",
                ExitCodes.CONTRACT_VIOLATION,
            )
        for _, prediction_run_id, _ in prediction_parts[round_index]:
            if prediction_run_id != run_id:
                raise OpalError(
                    f"{label} round {round_index} run_id differs between run and prediction ledgers.",
                    ExitCodes.CONTRACT_VIOLATION,
                )
        run_frame = read_parquet_df(run_part)
        matching_runs = run_frame.loc[
            run_frame["as_of_round"].astype(int).eq(round_index) & run_frame["run_id"].astype(str).eq(run_id)
        ]
        if len(matching_runs) != 1:
            raise OpalError(
                f"{label} round {round_index} run ledger must contain one metadata row for run_id={run_id}.",
                ExitCodes.CONTRACT_VIOLATION,
            )
        run_row = matching_runs.iloc[0].to_dict()
        prediction_frame = prediction_rows_for_run(
            (item[0] for item in prediction_parts[round_index]),
            round_index=round_index,
            run_id=run_id,
            columns=("pred__selection_views",),
        )
        predicted_rows = len(prediction_frame)
        expected_rows = int(run_row["stats__n_scored"])
        retention_mode = retention.mode_by_round[round_index]
        if retention.run_id_by_round.get(round_index, run_id) != run_id:
            raise OpalError(f"{label} round {round_index} retention run_id differs from the run ledger.")
        if retention.scored_rows_by_round.get(round_index, expected_rows) != expected_rows:
            raise OpalError(f"{label} round {round_index} retention scored rows differ from the run ledger.")
        if retention.retained_rows_by_round.get(round_index, predicted_rows) != predicted_rows:
            raise OpalError(f"{label} round {round_index} retention rows differ from the prediction ledger.")
        validate_prediction_retention(
            prediction_frame,
            expected_scored_rows=expected_rows,
            mode=retention_mode,
            label=f"{label} round {round_index}",
        )
        round_dir = outputs / "rounds" / f"round_{round_index}"
        if not round_dir.is_dir():
            raise OpalError(f"{label} round directory not found: {round_dir}", ExitCodes.CONTRACT_VIOLATION)
        _verify_run_artifacts(
            round_dir,
            run_id=run_id,
            artifacts=run_row.get("artifacts"),
            label=f"{label} round {round_index}",
        )
        artifact_root = run_artifact_root(round_dir, run_id=run_id)
        round_context_path = artifact_root / "metadata" / "round_ctx.json"
        context = _read_round_context(artifact_root)
        context_slug = str(context.get("core/campaign_slug") or "").strip()
        if not context_slug:
            raise OpalError(f"{label} round {round_index} context has no campaign slug.", ExitCodes.CONTRACT_VIOLATION)
        if campaign_slug is None:
            campaign_slug = context_slug
        elif context_slug != campaign_slug:
            raise OpalError(f"{label} contains multiple campaign slugs.", ExitCodes.CONTRACT_VIOLATION)
        if int(context.get("core/round_index", -1)) != round_index or str(context.get("core/run_id")) != run_id:
            raise OpalError(
                f"{label} round {round_index} context does not match its ledger identity.",
                ExitCodes.CONTRACT_VIOLATION,
            )
        x_column_name, y_column_name = _read_run_column_contract(
            context,
            round_index=round_index,
            run_id=run_id,
            round_context_sha256=file_sha256(round_context_path),
            column_contract=column_contract,
        )
        run_row["data__x_column_name"] = x_column_name
        run_row["data__y_column_name"] = y_column_name
        invariant = {field: run_row.get(field) for field in _RUN_INVARIANT_FIELDS}
        runs.append(
            RunHistory(
                round_index=round_index,
                run_id=run_id,
                round_dir=round_dir,
                run_part=run_part,
                prediction_parts=tuple(item[0] for item in prediction_parts[round_index]),
                run_row=run_row,
                round_context=context,
                invariant_sha256=canonical_sha256(invariant),
                prediction_row_count=predicted_rows,
                prediction_retention=retention_mode,
            )
        )
    state_path = root / "state.json"
    state = CampaignState.load(state_path) if state_path.is_file() else None
    if state is not None:
        if state.campaign_slug != campaign_slug:
            raise OpalError(f"{label} state and round contexts use different campaign slugs.")
        state_rounds = sorted(int(entry.round_index) for entry in state.rounds)
        if state_rounds != sorted(run_parts):
            raise OpalError(
                f"{label} state and ledgers contain different rounds: "
                f"state={state_rounds}, ledgers={sorted(run_parts)}.",
                ExitCodes.CONTRACT_VIOLATION,
            )
    return CampaignHistory(
        workdir=root,
        campaign_slug=str(campaign_slug),
        runs=tuple(runs),
        state=state,
        retention_manifest=retention.manifest_path,
    )


def _assert_candidate_lineage(runs: list[RunHistory]) -> None:
    prior: pd.DataFrame | None = None
    prior_is_full = False
    for run in sorted(runs, key=lambda item: item.round_index):
        current = prediction_rows_for_run(
            run.prediction_parts,
            round_index=run.round_index,
            run_id=run.run_id,
            columns=("id", "sequence"),
        )
        current["id"] = current["id"].astype(str)
        current["sequence"] = current["sequence"].astype(str)
        if current["id"].duplicated().any():
            raise OpalError(f"Round {run.round_index} prediction ledger contains duplicate candidate IDs.")
        if prior is not None:
            prior_by_id = prior.set_index("id")["sequence"]
            current_by_id = current.set_index("id")["sequence"]
            additions = sorted(set(current_by_id.index) - set(prior_by_id.index))
            if prior_is_full and additions:
                raise OpalError(
                    f"Round {run.round_index} introduces candidate IDs absent from the prior campaign universe "
                    f"(sample={additions[:5]}).",
                    ExitCodes.CONTRACT_VIOLATION,
                )
            shared = current_by_id.index.intersection(prior_by_id.index)
            changed = shared[current_by_id.loc[shared].to_numpy() != prior_by_id.loc[shared].to_numpy()].tolist()
            if changed:
                raise OpalError(
                    f"Round {run.round_index} changes candidate sequences for existing IDs (sample={changed[:5]}).",
                    ExitCodes.CONTRACT_VIOLATION,
                )
        prior = current
        prior_is_full = run.prediction_retention == FULL


def plan_history_relocation(
    *,
    source_workdir: Path,
    target_workdir: Path,
    expected_slug: str,
    column_contract: HistoryColumnContract | None = None,
) -> HistoryRelocationPlan:
    source = inspect_campaign_history(
        source_workdir,
        label="Source campaign history",
        column_contract=column_contract,
    )
    target = inspect_campaign_history(
        target_workdir,
        label="Target campaign history",
        column_contract=column_contract,
    )
    if source.workdir == target.workdir:
        raise OpalError("Source and target campaign histories must be different directories.", ExitCodes.BAD_ARGS)
    if source.campaign_slug != expected_slug or target.campaign_slug != expected_slug:
        raise OpalError(
            f"Campaign history slug mismatch: expected={expected_slug!r}, "
            f"source={source.campaign_slug!r}, target={target.campaign_slug!r}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    overlap = sorted(set(source.rounds) & set(target.rounds))
    if overlap:
        raise OpalError(f"Source and target campaign histories overlap at rounds {overlap}.")
    canonical_rounds = tuple(sorted((*source.rounds, *target.rounds)))
    expected_rounds = tuple(range(canonical_rounds[-1] + 1))
    if canonical_rounds != expected_rounds:
        raise OpalError(
            f"Combined campaign history is not contiguous: observed={list(canonical_rounds)}, "
            f"expected={list(expected_rounds)}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    all_runs = sorted((*source.runs, *target.runs), key=lambda item: item.round_index)
    if column_contract is not None:
        if column_contract.campaign_slug != expected_slug:
            raise OpalError("History column contract campaign slug differs from the relocation campaign.")
        observed = {
            (
                run.round_index,
                run.run_id,
                file_sha256(run_artifact_root(run.round_dir, run_id=run.run_id) / "metadata" / "round_ctx.json"),
            )
            for run in all_runs
        }
        declared = {(item.round_index, item.run_id, item.round_context_sha256) for item in column_contract.rounds}
        if observed != declared:
            raise OpalError("History column contract does not exactly cover the relocated round contexts.")
        if any(
            str(run.run_row["data__x_column_name"]) != column_contract.x_column_name
            or str(run.run_row["data__y_column_name"]) != column_contract.y_column_name
            for run in all_runs
        ):
            raise OpalError("History column contract X/Y identities differ from the relocated runs.")
    invariant_digests = {run.invariant_sha256 for run in all_runs}
    if len(invariant_digests) != 1:
        raise OpalError(
            "Source and target histories do not share one X/Y/objective contract.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    _assert_candidate_lineage(all_runs)
    return HistoryRelocationPlan(
        source=source,
        target=target,
        campaign_slug=expected_slug,
        canonical_rounds=canonical_rounds,
        invariant_sha256=next(iter(invariant_digests)),
        column_contract=column_contract,
    )
