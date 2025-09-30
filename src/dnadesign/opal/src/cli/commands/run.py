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

from ...locks import CampaignLock
from ...run_round import RunRoundRequest, run_round
from ...state import CampaignState
from ...utils import ExitCodes, OpalError, print_stdout
from ..formatting import render_run_summary_human
from ..registry import cli_command
from ._common import (
    internal_error,
    json_out,
    load_cli_config,
    opal_error,
    store_from_cfg,
)


@cli_command("run", help="Train on labels ≤ round, score, select, append events.")
def cmd_run(
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
    round: int = typer.Option(
        ..., "--round", "-r", "--labels-as-of", help="Labels cutoff (as_of_round)."
    ),
    k: Optional[int] = typer.Option(
        None, "--k", "-k", help="Top-k (default from YAML)."
    ),
    resume: bool = typer.Option(
        False,
        "--resume",
        help="Allow overwrites in artifacts dir (no-op here; caller's concern).",
    ),
    score_batch_size: Optional[int] = typer.Option(
        None, "--score-batch-size", help="Override batch size."
    ),
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
    json: bool = typer.Option(
        False, "--json/--human", help="Output format (default: human)"
    ),
) -> None:
    try:
        cfg = load_cli_config(config)
        store = store_from_cfg(cfg)
        df = store.load()

        # Guard: if this round already exists in state.json, prompt unless --resume
        st_path = Path(cfg.campaign.workdir) / "state.json"
        if st_path.exists():
            try:
                st = CampaignState.load(st_path)
                exists = any(int(r.round_index) == int(round) for r in st.rounds)
            except Exception:
                exists = False
            if exists and not resume:
                resp = (
                    input(
                        f"[guard] Round r={int(round)} already recorded in {st_path.name}. "
                        "Overwrite this round entry and artifacts? (y/N): "
                    )
                    .strip()
                    .lower()
                )
                if resp not in ("y", "yes"):
                    print_stdout("Aborted.")
                    raise typer.Exit(code=ExitCodes.BAD_ARGS)
                resume = True

        req = RunRoundRequest(
            cfg=cfg,
            as_of_round=int(round),
            k_override=k,
            score_batch_size_override=score_batch_size,
            verbose=verbose,
            allow_resume=bool(resume),
        )
        with CampaignLock(Path(cfg.campaign.workdir)):
            res = run_round(store, df, req)
        summary = {
            "ok": res.ok,
            "run_id": res.run_id,
            "as_of_round": res.as_of_round,
            "trained_on": res.trained_on,
            "scored": res.scored,
            "top_k_requested": res.top_k_requested,
            "top_k_effective": res.top_k_effective,
            "events": res.events_path,
            "top_k_source": "cli_override" if k is not None else "yaml_default",
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
