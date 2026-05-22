from __future__ import annotations

from typing import Any, Iterable, Mapping

from ._support import mapping, predict_progress_text, resolved_run_id, sequence


def build_notebook_change_lines(view_model: Mapping[str, Any]) -> list[str]:
    """Build a compact round/run change summary for generated notebooks."""

    status = mapping(view_model.get("status"))
    progress = mapping(view_model.get("progress"))
    event_contract = mapping(progress.get("event_contract"))
    rounds = sequence(progress.get("rounds"))
    lines = [
        "### Changes",
        "",
        f"- Round selector: `{status.get('round_selector') or progress.get('round_selector') or 'latest'}`",
        f"- Rounds visible: `{len(rounds)}`",
        f"- Latest run ID: `{status.get('latest_run_id')}`",
        (
            "- Event phases: "
            f"command=`{event_contract.get('command_events', 0)}`, "
            f"preflight=`{event_contract.get('preflight_events', 0)}`, "
            f"run=`{event_contract.get('run_events', 0)}`, "
            f"finalize=`{event_contract.get('finalize_events', 0)}`"
        ),
        f"- Aborted rounds: `{len(sequence(event_contract.get('aborted_rounds')))}`",
        f"- Ambiguous run-scope rounds: `{len(sequence(event_contract.get('ambiguous_rounds')))}`",
    ]
    if not rounds:
        lines.append("- Round history: `not started`")
    return lines


def build_notebook_change_rows(view_model: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return progress-derived round/run rows for notebook change tables."""

    progress = mapping(view_model.get("progress"))
    rows: list[dict[str, Any]] = []
    for round_row in sequence(progress.get("rounds")):
        if not isinstance(round_row, Mapping):
            continue
        summary = mapping(round_row.get("summary"))
        run_scope = mapping(summary.get("run_scope"))
        predict = mapping(round_row.get("predict"))
        rows.append(
            {
                "round": round_row.get("round_index"),
                "status": round_row.get("status"),
                "last_stage": round_row.get("last_stage"),
                "run_id": resolved_run_id(run_scope),
                "attempts": len(sequence(run_scope.get("attempt_ids"))),
                "events": round_row.get("events"),
                "elapsed_sec": round_row.get("elapsed_sec"),
                "predict": predict_progress_text(predict),
                "aborted": bool(summary.get("aborted")),
                "ambiguous_run_scope": bool(run_scope.get("ambiguous_run_scope")),
                "log_path": round_row.get("path"),
            }
        )
    return rows


def resolve_notebook_round_default(default_round: Any, rounds: Iterable[Any], latest_round_value: Any) -> int:
    """Resolve the generated notebook's initial round selector."""

    round_values = [int(round_value) for round_value in rounds]
    if str(default_round).strip().lower() in ("latest", ""):
        return int(latest_round_value)
    resolved = int(default_round)
    if resolved not in round_values:
        raise ValueError(f"default round {resolved} not in available rounds: {round_values}")
    return resolved


def build_notebook_run_options(runs_for_round: Any) -> list[str]:
    """Return stable run_id dropdown options for a selected round."""

    return runs_for_round.select("run_id").unique().sort("run_id")["run_id"].to_list()


def build_notebook_run_summary_lines(run_id: str, run_meta: Mapping[str, Any], objective_name: str) -> list[str]:
    """Build the selected-run summary lines used by generated notebooks."""

    return [
        "## Run Summary",
        "",
        (
            f"Run `{run_id}` (round {run_meta.get('as_of_round', -1)}) uses "
            f"objective `{objective_name}` and selection `{run_meta.get('selection__name')}`."
        ),
        f"Model: `{run_meta.get('model__name')}`",
        f"Train size: {run_meta.get('stats__n_train')} | Scored: {run_meta.get('stats__n_scored')}",
    ]


def build_notebook_no_run_lines(
    table_status_lines: Iterable[str],
    *,
    no_run_message: str = "No runs available yet.",
    expected_runs_ledger: str = "outputs/ledger/runs.parquet",
) -> list[str]:
    """Build no-run guidance for generated notebooks."""

    return [
        "### Round and run",
        "",
        no_run_message,
        "The campaign contract and records remain inspectable before the first OPAL run.",
        f"Expected runs ledger: `{expected_runs_ledger}`.",
        "",
        *list(table_status_lines),
    ]
