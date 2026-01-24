# ABOUTME: CLI command for running ephemeral predictions from saved models.
# ABOUTME: Resolves model artifacts and outputs prediction tables.
"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/predict.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json as _json
from pathlib import Path

import typer

from ...core.rounds import resolve_round_index
from ...core.utils import ExitCodes, OpalError, print_stdout
from ...runtime.predict import run_predict_ephemeral
from ...storage.parquet_io import read_parquet_df, write_parquet_df
from ...storage.state import CampaignState
from ..registry import cli_command
from ._common import (
    internal_error,
    load_cli_config,
    opal_error,
    resolve_config_path,
    resolve_json_path,
    resolve_table_path,
    store_from_cfg,
)


@cli_command("predict", help="Ephemeral inference with a frozen model; no write-backs.")
def cmd_predict(
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
    model_path: Path = typer.Option(
        None,
        "--model-path",
        help="Path to model.joblib (or omit with --round/--config)",
    ),
    model_name: str = typer.Option(
        None,
        "--model-name",
        help="Explicit model registry name (required if model_meta.json is missing).",
    ),
    model_params: Path = typer.Option(
        None,
        "--model-params",
        help="Optional JSON file (.json) with model params (used with --model-name).",
    ),
    round: str = typer.Option(
        None,
        "--round",
        "-r",
        help="Round index to resolve model from state.json (default: latest). Accepts 'latest'.",
    ),
    input_path: Path = typer.Option(None, "--in", help="Optional input parquet/csv; defaults to records.parquet"),
    out_path: Path = typer.Option(None, "--out", help="Optional output parquet/csv; defaults to stdout CSV"),
    id_col: str = typer.Option("id", "--id-col", help="ID column name in input table."),
    sequence_col: str = typer.Option("sequence", "--sequence-col", help="Sequence column name in input table."),
    generate_id_from_sequence: bool = typer.Option(
        False,
        "--generate-id-from-sequence",
        help="Generate deterministic ids from sequence when id column is missing.",
    ),
    assume_no_yops: bool = typer.Option(
        False,
        "--assume-no-yops",
        help="Skip Y-ops inversion even if training used Y-ops (use only if round_ctx.json is unavailable).",
    ),
):
    try:
        import pandas as pd

        cfg_path = resolve_config_path(config)
        cfg = load_cli_config(cfg_path)
        store = store_from_cfg(cfg)

        if model_path is not None and round is not None:
            raise OpalError("Use only one of --model-path or --round (they are mutually exclusive).")

        if model_path is not None:
            model_path = Path(model_path)
            if not model_path.exists():
                raise OpalError(f"--model-path not found: {model_path}")
            if model_path.is_dir():
                raise OpalError(f"--model-path must be a file, got directory: {model_path}")

        if input_path is not None:
            input_path = resolve_table_path(input_path, label="--in", must_exist=True)

        if out_path is not None:
            out_path = resolve_table_path(out_path, label="--out", must_exist=False)
        if model_params is not None:
            model_params = resolve_json_path(model_params, label="--model-params", must_exist=True)

        # Resolve model_path if not provided
        if model_path is None:
            st_path = Path(cfg.campaign.workdir) / "state.json"
            if not st_path.exists():
                raise OpalError("Provide --model-path or run from a campaign with state.json.")
            st = CampaignState.load(st_path)
            rounds = sorted(st.rounds, key=lambda r: int(r.round_index))
            round_values = [int(r.round_index) for r in rounds]
            round_sel = resolve_round_index(
                round,
                rounds=round_values,
                allow_none=False,
                empty_message=f"No rounds found in {st_path}",
                param_label="--round",
            )
            entry = next((r for r in rounds if int(r.round_index) == int(round_sel)), None)
            if entry is None:
                raise OpalError(f"Round {round} not found in {st_path}")
            mp = Path(entry.model.get("artifact_path", "")) if entry.model else None
            if not mp or not mp.exists():
                mp = Path(entry.round_dir) / "model" / "model.joblib"
            model_path = mp
        if model_path is None or not Path(model_path).exists():
            raise OpalError(f"Resolved model path not found: {model_path}")
        if Path(model_path).is_dir():
            raise OpalError(f"Resolved model path must be a file, got directory: {model_path}")
        df = (
            store.load()
            if input_path is None
            else (
                read_parquet_df(input_path)
                if input_path.suffix.lower() in (".parquet", ".pq")
                else pd.read_csv(input_path)
            )
        )
        if cfg.data.x_column_name not in df.columns:
            raise OpalError(f"Input missing X column: {cfg.data.x_column_name}")
        params_obj = None
        if model_params:
            params_obj = _json.loads(model_params.read_text())
            if not model_name:
                raise OpalError("Use --model-name with --model-params.")
        preds = run_predict_ephemeral(
            store,
            df,
            model_path,
            model_name=model_name,
            model_params=params_obj,
            id_column=id_col,
            sequence_column=sequence_col,
            generate_id_from_sequence=generate_id_from_sequence,
            assume_no_yops=assume_no_yops,
        )
        if out_path:
            if out_path.suffix.lower() == ".csv":
                df_out = preds.copy()
                df_out["y_pred_vec"] = df_out["y_pred_vec"].map(lambda v: _json.dumps(v))
                df_out.to_csv(out_path, index=False)
            else:
                write_parquet_df(out_path, preds, index=False)
            print_stdout(f"Wrote predictions: {out_path}")
        else:
            df_out = preds.copy()
            df_out["y_pred_vec"] = df_out["y_pred_vec"].map(lambda v: _json.dumps(v))
            print_stdout(df_out.to_csv(index=False))
    except OpalError as e:
        opal_error("predict", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("predict", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
