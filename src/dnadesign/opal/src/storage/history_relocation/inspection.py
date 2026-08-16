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
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...core.utils import ExitCodes, OpalError
from ..parquet_io import read_parquet_df
from ..state import CampaignState
from .contracts import CampaignHistory, HistoryRelocationPlan, RunHistory

_RUN_INVARIANT_FIELDS = (
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


def _part_identity(path: Path, *, label: str) -> tuple[int, str, int]:
    frame = read_parquet_df(path, columns=["as_of_round", "run_id"])
    rounds = sorted({int(value) for value in frame["as_of_round"].tolist()})
    run_ids = sorted({str(value) for value in frame["run_id"].tolist()})
    if len(rounds) != 1 or len(run_ids) != 1:
        raise OpalError(
            f"{label} part must contain exactly one run and round: {path}",
            ExitCodes.CONTRACT_VIOLATION,
        )
    return rounds[0], run_ids[0], len(frame)


def _parts_by_round(root: Path, *, label: str) -> dict[int, list[tuple[Path, str, int]]]:
    if not root.is_dir():
        raise OpalError(f"{label} dataset not found: {root}", ExitCodes.CONTRACT_VIOLATION)
    result: dict[int, list[tuple[Path, str, int]]] = {}
    for path in sorted(root.rglob("*.parquet")):
        round_index, run_id, row_count = _part_identity(path, label=label)
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


def inspect_campaign_history(workdir: Path, *, label: str) -> CampaignHistory:
    root = _canonical_root(workdir, label=label)
    outputs = root / "outputs"
    run_parts = _parts_by_round(outputs / "ledger" / "runs.parquet", label=f"{label} run ledger")
    prediction_parts = _parts_by_round(outputs / "ledger" / "predictions", label=f"{label} prediction ledger")
    if set(run_parts) != set(prediction_parts):
        raise OpalError(
            f"{label} run/prediction rounds differ: runs={sorted(run_parts)}, predictions={sorted(prediction_parts)}",
            ExitCodes.CONTRACT_VIOLATION,
        )
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
        run_row = read_parquet_df(run_part).iloc[0].to_dict()
        predicted_rows = sum(item[2] for item in prediction_parts[round_index])
        expected_rows = int(run_row["stats__n_scored"])
        if predicted_rows != expected_rows:
            raise OpalError(
                f"{label} round {round_index} prediction count differs from run metadata: "
                f"predictions={predicted_rows}, expected={expected_rows}.",
                ExitCodes.CONTRACT_VIOLATION,
            )
        round_dir = outputs / "rounds" / f"round_{round_index}"
        if not round_dir.is_dir():
            raise OpalError(f"{label} round directory not found: {round_dir}", ExitCodes.CONTRACT_VIOLATION)
        context = _read_round_context(round_dir)
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
    )


def _assert_candidate_lineage(runs: list[RunHistory]) -> None:
    prior: pd.DataFrame | None = None
    for run in sorted(runs, key=lambda item: item.round_index):
        current = pd.concat(
            [read_parquet_df(part, columns=["id", "sequence"]) for part in run.prediction_parts],
            ignore_index=True,
        )
        current["id"] = current["id"].astype(str)
        current["sequence"] = current["sequence"].astype(str)
        if current["id"].duplicated().any():
            raise OpalError(f"Round {run.round_index} prediction ledger contains duplicate candidate IDs.")
        if prior is not None:
            prior_by_id = prior.set_index("id")["sequence"]
            current_by_id = current.set_index("id")["sequence"]
            missing = sorted(set(current_by_id.index) - set(prior_by_id.index))
            if missing:
                raise OpalError(
                    f"Round {run.round_index} introduces candidate IDs absent from the prior campaign universe "
                    f"(sample={missing[:5]}).",
                    ExitCodes.CONTRACT_VIOLATION,
                )
            shared = current_by_id.index
            changed = shared[current_by_id.loc[shared].to_numpy() != prior_by_id.loc[shared].to_numpy()].tolist()
            if changed:
                raise OpalError(
                    f"Round {run.round_index} changes candidate sequences for existing IDs (sample={changed[:5]}).",
                    ExitCodes.CONTRACT_VIOLATION,
                )
        prior = current


def plan_history_relocation(*, source_workdir: Path, target_workdir: Path, expected_slug: str) -> HistoryRelocationPlan:
    source = inspect_campaign_history(source_workdir, label="Source campaign history")
    target = inspect_campaign_history(target_workdir, label="Target campaign history")
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
    )
