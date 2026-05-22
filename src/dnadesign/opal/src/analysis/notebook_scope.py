"""Notebook round/run scope resolution."""

from __future__ import annotations

from ..core.utils import ExitCodes, OpalError
from .facade import CampaignAnalysis


def resolve_notebook_run_scope(
    analysis: CampaignAnalysis,
    *,
    round_selector: str | None,
    run_id: str | None,
) -> tuple[str | None, str | None]:
    requested_run_id = str(run_id).strip() if run_id is not None else ""
    if not requested_run_id:
        return round_selector, None
    runs_df = analysis.read_runs()
    if runs_df.is_empty():
        raise OpalError("run_id was provided but outputs/ledger/runs.parquet is empty.", ExitCodes.BAD_ARGS)
    if "run_id" not in runs_df.columns or "as_of_round" not in runs_df.columns:
        raise OpalError(
            "outputs/ledger/runs.parquet missing required columns (run_id, as_of_round).",
            ExitCodes.CONTRACT_VIOLATION,
        )
    matched = runs_df.filter(runs_df["run_id"] == requested_run_id)
    if matched.is_empty():
        raise OpalError(f"run_id {requested_run_id!r} not found in outputs/ledger/runs.parquet.", ExitCodes.BAD_ARGS)
    rounds = sorted({int(value) for value in matched["as_of_round"].to_list() if value is not None})
    if len(rounds) != 1:
        raise OpalError(
            f"run_id {requested_run_id!r} belongs to multiple rounds {rounds}.",
            ExitCodes.CONTRACT_VIOLATION,
        )
    run_round = rounds[0]
    selector = str(round_selector or "latest").strip().lower()
    if selector == "all":
        raise OpalError(
            "notebook generation with --run-id requires a single round selector, not --round all.",
            ExitCodes.BAD_ARGS,
        )
    if selector not in ("", "latest", "unspecified"):
        try:
            selected_round = int(selector)
        except Exception as exc:
            raise OpalError("Invalid --round: must be an integer or 'latest'.", ExitCodes.BAD_ARGS) from exc
        if selected_round != run_round:
            raise OpalError(
                f"run_id {requested_run_id!r} belongs to round {run_round}, but --round selected {selector}.",
                ExitCodes.BAD_ARGS,
            )
    return str(run_round), requested_run_id
