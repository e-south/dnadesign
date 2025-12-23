"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/predict.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from ...predict import run_predict_ephemeral
from ...state import CampaignState
from ...utils import ExitCodes, OpalError, print_stdout
from ..registry import cli_command
from ._common import internal_error, load_cli_config, opal_error, store_from_cfg


@cli_command("predict", help="Ephemeral inference with a frozen model; no write-backs.")
def cmd_predict(
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
    model_path: Path = typer.Option(
        None,
        "--model-path",
        help="Path to model.joblib (or omit with --round/--config)",
    ),
    round: int = typer.Option(
        None,
        "--round",
        "-r",
        help="Round index to resolve model from state.json (default: latest)",
    ),
    input_path: Path = typer.Option(None, "--in", help="Optional input parquet/csv; defaults to records.parquet"),
    out_path: Path = typer.Option(None, "--out", help="Optional output parquet/csv; defaults to stdout CSV"),
):
    try:
        cfg = load_cli_config(config)
        store = store_from_cfg(cfg)
        # Resolve model_path if not provided
        if model_path is None:
            st_path = Path(cfg.campaign.workdir) / "state.json"
            if not st_path.exists():
                raise OpalError("Provide --model-path or run from a campaign with state.json.")
            st = CampaignState.load(st_path)
            rounds = sorted(st.rounds, key=lambda r: int(r.round_index))
            if not rounds:
                raise OpalError(f"No rounds found in {st_path}")
            entry = (
                next((r for r in rounds if int(r.round_index) == int(round)), None) if round is not None else rounds[-1]
            )
            if entry is None:
                raise OpalError(f"Round {round} not found in {st_path}")
            mp = Path(entry.model.get("artifact_path", "")) if entry.model else None
            if not mp or not mp.exists():
                mp = Path(entry.round_dir) / "model.joblib"
            model_path = mp
        df = (
            store.load()
            if input_path is None
            else (
                pd.read_parquet(input_path)
                if input_path.suffix.lower() in (".parquet", ".pq")
                else pd.read_csv(input_path)
            )
        )
        if cfg.data.x_column_name not in df.columns:
            raise OpalError(f"Input missing X column: {cfg.data.x_column_name}")
        preds = run_predict_ephemeral(store, df, model_path)
        if out_path:
            if out_path.suffix.lower() == ".csv":
                preds.to_csv(out_path, index=False)
            else:
                preds.to_parquet(out_path, index=False)
            print_stdout(f"Wrote predictions: {out_path}")
        else:
            print_stdout(preds.to_csv(index=False))
    except OpalError as e:
        opal_error("run", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("predict", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
