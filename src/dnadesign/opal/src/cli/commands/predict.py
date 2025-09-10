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

from ...config import load_config
from ...predict import run_predict_ephemeral
from ...utils import ExitCodes, OpalError, print_stdout
from ..registry import cli_command
from ._common import internal_error, resolve_config_path, store_from_cfg


@cli_command("predict", help="Ephemeral inference with a frozen model; no write-backs.")
def cmd_predict(
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
    model_path: Path = typer.Option(..., "--model-path"),
    input_path: Path = typer.Option(
        None, "--in", help="Optional input parquet/csv; defaults to records.parquet"
    ),
    out_path: Path = typer.Option(
        None, "--out", help="Optional output parquet/csv; defaults to stdout CSV"
    ),
):
    try:
        cfg = load_config(resolve_config_path(config))
        store = store_from_cfg(cfg)
        df = (
            store.load()
            if input_path is None
            else (
                pd.read_parquet(input_path)
                if input_path.suffix.lower() in (".parquet", ".pq")
                else pd.read_csv(input_path)
            )
        )
        if cfg.data.representation_column_name not in df.columns:
            raise OpalError(
                f"Input missing representation column: {cfg.data.representation_column_name}"
            )
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
        typer.echo(str(e), err=True)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("predict", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
