"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/cli/commands/run.py

CLI wiring for run OPAL CLI commands.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Optional

import typer
from typer.models import OptionInfo

from ...core.selection_contracts import (
    resolve_selection_objective_mode,
    resolve_selection_tie_handling,
)
from ...core.utils import ExitCodes, OpalError, now_iso, print_stdout
from ...runtime.memory_guard import enforce_x_matrix_memory_budget
from ...runtime.run_round import RunRoundRequest, run_round
from ...storage.artifacts import append_round_log_event
from ...storage.candidate_scope import load_candidate_scope_ids
from ...storage.locks import CampaignLock
from ...storage.state import CampaignState
from ...storage.workspace import CampaignWorkspace
from ...storage.x_contracts import validate_x_parquet_column
from ..formatting import render_run_summary_text
from ..guidance_hints import maybe_print_hints
from ..registry import cli_command
from ..tui import progress_factory as tui_progress_factory
from ._common import (
    internal_error,
    json_out,
    load_cli_config,
    opal_error,
    print_config_context,
    prompt_confirm,
    resolve_config_path,
    store_from_cfg,
)


def _resolve_summary_selection_mode(sel_params: dict[str, object]) -> tuple[str, str]:
    tie_handling = resolve_selection_tie_handling(sel_params, error_cls=OpalError)
    objective_mode = resolve_selection_objective_mode(sel_params, error_cls=OpalError)
    return tie_handling, objective_mode


def _direct_call_default(value, default):
    return default if isinstance(value, OptionInfo) else value


def _append_cli_round_event(
    cfg,
    cfg_path: Path,
    round_index: int,
    stage: str,
    *,
    attempt_id: str | None = None,
    **payload: object,
) -> None:
    ws = CampaignWorkspace.from_config(cfg, cfg_path)
    if attempt_id is not None:
        payload.setdefault("attempt_id", str(attempt_id))
    append_round_log_event(
        ws.round_logs_dir(int(round_index)) / "round.log.jsonl",
        {"ts": now_iso(), "round": int(round_index), "stage": stage, **payload},
    )


def _append_abort_event(
    cfg,
    cfg_path: Path,
    round_index: int,
    error: BaseException,
    *,
    attempt_id: str | None = None,
) -> None:
    try:
        _append_cli_round_event(
            cfg,
            cfg_path,
            int(round_index),
            "abort",
            attempt_id=attempt_id,
            severity="error",
            error_type=type(error).__name__,
            message=str(error),
        )
    except Exception:
        pass


@cli_command("run", help="Train on labels ≤ round, score, select, append events.")
def cmd_run(
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
    round: int = typer.Option(
        ...,
        "--round",
        "-r",
        "--labels-as-of",
        help="Labels cutoff for training (use labels with observed_round ≤ this value).",
    ),
    k: Optional[int] = typer.Option(None, "--k", "-k", help="Top-k (default from YAML)."),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Allow overwriting existing round artifacts (required when rerunning a round).",
    ),
    score_batch_size: Optional[int] = typer.Option(None, "--score-batch-size", help="Override batch size."),
    max_x_matrix_gib: Optional[float] = typer.Option(
        None,
        "--max-x-matrix-gib",
        help="Override safety.max_x_matrix_gib for this run. Use only when the host has enough RAM.",
    ),
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
    no_hints: bool = typer.Option(False, "--no-hints", help="Disable next-step hints in text output."),
    json: bool = typer.Option(False, "--json/--text", help="Output format (default: text)"),
) -> None:
    config = _direct_call_default(config, None)
    k = _direct_call_default(k, None)
    resume = bool(_direct_call_default(resume, False))
    score_batch_size = _direct_call_default(score_batch_size, None)
    max_x_matrix_gib = _direct_call_default(max_x_matrix_gib, None)
    verbose = bool(_direct_call_default(verbose, True))
    no_hints = bool(_direct_call_default(no_hints, False))
    json = bool(_direct_call_default(json, False))

    cfg_path: Path | None = None
    cfg = None
    attempt_id: str | None = None
    try:
        attempt_id = uuid.uuid4().hex
        cfg_path = resolve_config_path(config)
        cfg = load_cli_config(cfg_path)
        _append_cli_round_event(
            cfg,
            cfg_path,
            int(round),
            "command_start",
            attempt_id=attempt_id,
            command="run",
            status="started",
        )
        store = store_from_cfg(cfg)
        _append_cli_round_event(
            cfg,
            cfg_path,
            int(round),
            "x_validate_start",
            attempt_id=attempt_id,
            status="started",
        )
        x_contract = validate_x_parquet_column(store.records_path, x_column=cfg.data.x_column_name)
        _append_cli_round_event(
            cfg,
            cfg_path,
            int(round),
            "x_validate_done",
            attempt_id=attempt_id,
            status="ok",
            rows=int(x_contract.row_count),
            x_dim=int(x_contract.x_dim),
        )
        sbatch = int(score_batch_size or cfg.scoring.score_batch_size)
        if sbatch <= 0:
            raise OpalError("score_batch_size must be a positive integer.", ExitCodes.BAD_ARGS)
        materializes_prediction_records = str(cfg.writeback.prediction_records) == "label_history"
        scoped_candidate_rows = (
            len(load_candidate_scope_ids(cfg.data.candidate_scope))
            if cfg.data.candidate_scope is not None
            else int(x_contract.row_count)
        )
        memory_guard_rows = (
            int(scoped_candidate_rows)
            if materializes_prediction_records
            else min(
                int(scoped_candidate_rows),
                int(sbatch),
            )
        )
        memory_guard_context = (
            "OPAL run full X matrix for label_history prediction writeback"
            if materializes_prediction_records
            else "OPAL run streaming score batch X matrix"
        )
        memory_estimate = enforce_x_matrix_memory_budget(
            row_count=int(memory_guard_rows),
            x_dim=int(x_contract.x_dim),
            item_size_bytes=max(8, int(x_contract.item_size_bytes)),
            max_gib=max_x_matrix_gib if max_x_matrix_gib is not None else cfg.safety.max_x_matrix_gib,
            context=memory_guard_context,
        )
        _append_cli_round_event(
            cfg,
            cfg_path,
            int(round),
            "x_memory_guard_done",
            attempt_id=attempt_id,
            status="ok",
            scope="full_records" if materializes_prediction_records else "streaming_score_batch",
            candidate_rows=int(scoped_candidate_rows),
            score_batch_size=int(sbatch),
            rows=int(memory_estimate.row_count),
            x_dim=int(memory_estimate.x_dim),
            raw_gib=float(memory_estimate.raw_gib),
            estimated_gib=float(memory_estimate.estimated_gib),
            max_gib=float(memory_estimate.max_gib),
        )
        _append_cli_round_event(
            cfg,
            cfg_path,
            int(round),
            "records_load_start",
            attempt_id=attempt_id,
            status="started",
        )
        df = store.load() if materializes_prediction_records else store.load_runtime_frame(include_x=False)
        _append_cli_round_event(
            cfg,
            cfg_path,
            int(round),
            "records_load_done",
            attempt_id=attempt_id,
            status="ok",
            rows=int(len(df)),
            columns=int(len(df.columns)),
        )
        if not json:
            print_config_context(cfg_path, cfg=cfg, records_path=store.records_path)

        # Guard: if this round already exists in state.json, prompt unless --resume
        st_path = Path(cfg.campaign.workdir) / "state.json"
        if st_path.exists():
            try:
                st = CampaignState.load(st_path)
            except Exception as e:
                raise OpalError(f"Failed to load state.json at {st_path}: {e}", ExitCodes.BAD_ARGS) from e
            exists = any(int(r.round_index) == int(round) for r in st.rounds)
            if exists and not resume:
                if not prompt_confirm(
                    f"[guard] Round r={int(round)} already recorded in {st_path.name}. "
                    "Overwrite this round entry and artifacts? (y/N): ",
                    non_interactive_hint="No TTY available. Re-run with --resume to overwrite this round.",
                ):
                    _append_cli_round_event(
                        cfg,
                        cfg_path,
                        int(round),
                        "abort",
                        attempt_id=attempt_id,
                        severity="warning",
                        status="operator_cancelled",
                        message="operator declined resume confirmation",
                    )
                    print_stdout("Aborted.")
                    raise typer.Exit(code=ExitCodes.BAD_ARGS)
                resume = True

        lockfile = Path(cfg.campaign.workdir) / ".opal.lock"
        _append_cli_round_event(
            cfg,
            cfg_path,
            int(round),
            "lock_acquire_start",
            attempt_id=attempt_id,
            lock_scope="local_host",
            lockfile=str(lockfile),
            status="started",
        )
        req = RunRoundRequest(
            cfg=cfg,
            as_of_round=int(round),
            config_path=cfg_path,
            k_override=k,
            score_batch_size_override=score_batch_size,
            max_x_matrix_gib_override=max_x_matrix_gib,
            x_dim_override=int(x_contract.x_dim),
            x_item_size_bytes=int(x_contract.item_size_bytes),
            verbose=verbose,
            allow_resume=bool(resume),
            progress_factory=(tui_progress_factory() if verbose and not json else None),
        )
        lock_acquired = False
        lock_path = str(lockfile)
        try:
            with CampaignLock(
                Path(cfg.campaign.workdir),
                payload_extra={"attempt_id": attempt_id, "round": int(round), "command": "run"},
            ) as lock:
                lock_acquired = True
                lock_path = str(lock.lockfile)
                _append_cli_round_event(
                    cfg,
                    cfg_path,
                    int(round),
                    "lock_acquired",
                    attempt_id=attempt_id,
                    lock_scope="local_host",
                    lockfile=lock_path,
                    status="locked",
                )
                res = run_round(store, df, req)
        finally:
            if lock_acquired:
                _append_cli_round_event(
                    cfg,
                    cfg_path,
                    int(round),
                    "lock_released",
                    attempt_id=attempt_id,
                    phase="finalize",
                    lock_scope="local_host",
                    lockfile=lock_path,
                    status="released",
                )
        sel_params = dict(cfg.selection.selection.params or {})
        tie_handling, objective_mode = _resolve_summary_selection_mode(sel_params)
        summary = {
            "ok": res.ok,
            "run_id": res.run_id,
            "as_of_round": res.as_of_round,
            "trained_on": res.trained_on,
            "scored": res.scored,
            "top_k_requested": res.top_k_requested,
            "top_k_effective": res.top_k_effective,
            "ledger": res.ledger_path,
            "top_k_source": "cli_override" if k is not None else "yaml_default",
            "tie_handling": tie_handling,
            "objective_mode": objective_mode,
        }
        if json:
            json_out(summary)
        else:
            print_stdout(render_run_summary_text(summary))
            maybe_print_hints(
                command_name="run",
                cfg_path=cfg_path,
                no_hints=no_hints,
                json_output=json,
                labels_as_of=int(round),
            )
    except typer.Exit:
        raise
    except OpalError as e:
        if cfg is not None and cfg_path is not None:
            _append_abort_event(cfg, cfg_path, int(round), e, attempt_id=attempt_id)
        opal_error("run", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        if cfg is not None and cfg_path is not None:
            _append_abort_event(cfg, cfg_path, int(round), e, attempt_id=attempt_id)
        internal_error("run", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
