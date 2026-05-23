from __future__ import annotations

from typing import Any, Iterable, Mapping

from ._support import compact_path, mapping, predict_progress_text, resolved_run_id, sequence


def build_notebook_change_lines(view_model: Mapping[str, Any]) -> list[str]:
    """Build a compact round/run change summary for generated notebooks."""

    return [f"{row['field']}: `{row['value']}`" for row in build_notebook_change_summary_rows(view_model)]


def build_notebook_change_summary_rows(view_model: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Build compact round/run change rows for generated notebooks."""

    status = mapping(view_model.get("status"))
    progress = mapping(view_model.get("progress"))
    event_contract = mapping(progress.get("event_contract"))
    rounds = sequence(progress.get("rounds"))
    rows = [
        {
            "field": "Round selector",
            "value": status.get("round_selector") or progress.get("round_selector") or "latest",
        },
        {"field": "Rounds visible", "value": len(rounds)},
        {"field": "Latest run ID", "value": status.get("latest_run_id")},
        {
            "field": "Event phases",
            "value": (
                f"command={event_contract.get('command_events', 0)}, "
                f"preflight={event_contract.get('preflight_events', 0)}, "
                f"run={event_contract.get('run_events', 0)}, "
                f"finalize={event_contract.get('finalize_events', 0)}"
            ),
        },
        {"field": "Aborted rounds", "value": len(sequence(event_contract.get("aborted_rounds")))},
        {"field": "Ambiguous run-scope rounds", "value": len(sequence(event_contract.get("ambiguous_rounds")))},
    ]
    if not rounds:
        rows.append({"field": "Round history", "value": "not started"})
    return rows


def build_notebook_change_rows(view_model: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return progress-derived round/run rows for notebook change tables."""

    progress = mapping(view_model.get("progress"))
    campaign = mapping(view_model.get("campaign"))
    workdir = campaign.get("workdir")
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
                "log": compact_path(round_row.get("path"), base=workdir),
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


def build_notebook_run_summary_lines(
    run_id: str,
    run_meta: Mapping[str, Any],
    objective_name: str,
    *,
    selected_round: Any | None = None,
    default_round: Any | None = None,
    run_options: Iterable[str] | None = None,
) -> list[str]:
    """Build the selected-run summary lines used by generated notebooks."""

    run_ids = [str(option) for option in sequence(run_options)]
    selected = selected_round if selected_round is not None else run_meta.get("as_of_round", -1)
    default = "latest" if str(default_round or "").strip().lower() in ("", "latest") else str(default_round)
    lines = [
        (
            f"Run `{run_id}` (round {run_meta.get('as_of_round', -1)}) uses "
            f"objective `{objective_name}` and selection `{run_meta.get('selection__name')}`."
        ),
        f"Run scope: selected round `{selected}`, selected run `{run_id}`.",
        f"Generated default round selector: `{default}`.",
    ]
    if len(run_ids) > 1:
        lines.append(
            f"Available runs for this round: `{len(run_ids)}`. "
            "The notebook scopes labels, predictions, and selections to the selected Run ID dropdown value."
        )
    elif len(run_ids) == 1:
        lines.append("Available runs for this round: `1`.")
    lines.extend(
        [
            f"Model: `{run_meta.get('model__name')}`",
            f"Train size: {run_meta.get('stats__n_train')} | Scored: {run_meta.get('stats__n_scored')}",
        ]
    )
    return [
        *lines,
    ]


def build_notebook_no_run_lines(
    table_status_lines: Iterable[str],
    *,
    no_run_message: str = "No runs available yet.",
    expected_runs_ledger: str = "outputs/ledger/runs.parquet",
) -> list[str]:
    """Build no-run guidance for generated notebooks."""

    return [
        no_run_message,
        "The campaign contract and records remain inspectable before the first OPAL run.",
        f"Expected runs ledger: `{expected_runs_ledger}`.",
        "",
        *list(table_status_lines),
    ]
