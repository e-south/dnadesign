"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/run.py

Thin CLI wrapper: parse flags -> load cfg/store -> call app.run_round

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ...core.utils import ExitCodes, OpalError, print_stdout
from ...runtime.run_round import RunRoundRequest, run_round
from ...storage.locks import CampaignLock
from ...storage.state import CampaignState
from ..formatting import render_run_summary_human
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
        help="Allow overwrites in artifacts dir (no-op here; caller's concern).",
    ),
    score_batch_size: Optional[int] = typer.Option(None, "--score-batch-size", help="Override batch size."),
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
    json: bool = typer.Option(False, "--json/--human", help="Output format (default: human)"),
) -> None:
    try:
        cfg_path = resolve_config_path(config)
        cfg = load_cli_config(cfg_path)
        store = store_from_cfg(cfg)
        df = store.load()
        if not json:
            print_config_context(cfg_path, cfg=cfg, records_path=store.records_path)

        # Guard: if this round already exists in state.json, prompt unless --resume
        st_path = Path(cfg.campaign.workdir) / "state.json"
        if st_path.exists():
            try:
                st = CampaignState.load(st_path)
                exists = any(int(r.round_index) == int(round) for r in st.rounds)
            except Exception:
                exists = False
            if exists and not resume:
                if not prompt_confirm(
                    f"[guard] Round r={int(round)} already recorded in {st_path.name}. "
                    "Overwrite this round entry and artifacts? (y/N): ",
                    non_interactive_hint="No TTY available. Re-run with --resume to overwrite this round.",
                ):
                    print_stdout("Aborted.")
                    raise typer.Exit(code=ExitCodes.BAD_ARGS)
                resume = True

        req = RunRoundRequest(
            cfg=cfg,
            as_of_round=int(round),
            config_path=cfg_path,
            k_override=k,
            score_batch_size_override=score_batch_size,
            verbose=verbose,
            allow_resume=bool(resume),
            progress_factory=(tui_progress_factory() if verbose and not json else None),
        )
        with CampaignLock(Path(cfg.campaign.workdir)):
            res = run_round(store, df, req)
        sel_params = dict(cfg.selection.selection.params or {})
        tie_handling = str(sel_params.get("tie_handling", "competition_rank"))
        objective_mode = str((sel_params.get("objective_mode") or "maximize")).strip().lower()
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
            print_stdout(render_run_summary_human(summary))
    except OpalError as e:
        opal_error("run", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("run", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
