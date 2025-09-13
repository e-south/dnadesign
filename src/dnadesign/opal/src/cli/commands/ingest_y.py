"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/ingest_y.py

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import typer

from ...config import load_config
from ...ingest import run_ingest
from ...utils import ExitCodes, OpalError, print_stdout
from ..registry import cli_command
from ._common import internal_error, json_out, resolve_config_path, store_from_cfg


@cli_command(
    "ingest-y", help="Ingest tidy CSV → Y (preview + confirmation; strict checks)."
)
def cmd_ingest_y(
    config: Path = typer.Option(
        None, "--config", "-c", envvar="OPAL_CONFIG", help="campaign.yaml"
    ),
    round: int = typer.Option(
        ..., "--round", "-r", help="Round index to append to label history"
    ),
    csv: Path = typer.Option(..., "--csv", help="Tidy CSV/Parquet with raw reads"),
    transform: str = typer.Option(
        None, "--transform", help="Override YAML transform name (optional)"
    ),
    params: Path = typer.Option(
        None, "--params", help="JSON file with transform params (optional)"
    ),
    yes: bool = typer.Option(
        False, "--yes", "-y", help="Skip interactive prompt (non-TTY safe)"
    ),
):
    """
    Behavior:
      • Reads transform+params from YAML (overridable with flags).
      • Validates strict preflights (single timepoint, quartet completeness, etc).
      • Always prints a preview (counts + sample), then asks to proceed.
      • For NEW ids: requires essentials (sequence, bio_type, alphabet, X).
      • Idempotent per (id, round): if already labeled with a different Y → abort.
    """
    try:
        cfg = load_config(resolve_config_path(config))
        store = store_from_cfg(cfg)
        df = store.load()

        csv_df = (
            pd.read_parquet(csv)
            if csv.suffix.lower() in (".pq", ".parquet")
            else pd.read_csv(csv)
        )

        # choose effective transform name & params
        t_name = (transform or cfg.data.transforms_y.name).strip()
        t_params = cfg.data.transforms_y.params
        if params:
            import json as _json

            t_params = _json.loads(Path(params).read_text())
            typer.echo("[WARN] Transform params overridden via --params", err=True)
        if t_name != cfg.data.transforms_y.name:
            typer.echo(
                f"[WARN] CLI transform '{t_name}' != YAML '{cfg.data.transforms_y.name}'",
                err=True,
            )

        labels_df, preview = run_ingest(
            df,
            csv_df,
            transform_name=t_name,
            transform_params=t_params,
            y_expected_length=cfg.data.y_expected_length,
            setpoint_vector=cfg.objective.objective.params.get("setpoint_vector", [0,0,0,1]),
        )

        # PREVIEW: counts + head (no flag, always printed)
        json_out({"preview": preview.__dict__})

        # Prompt unless --yes
        if not yes:
            resp = (
                input(f"Proceed to write labels for round {round}? (y/N): ")
                .strip()
                .lower()
            )
            if resp not in ("y", "yes"):
                print_stdout("Aborted by user.")
                return

        # Ensure rows exist for NEW ids (essentials pulled from the CSV if present)
        df = store.ensure_rows_exist(df, labels_df["id"].tolist(), csv_df)

        # Append labels and save atomically
        df2 = store.append_labels_from_df(df, labels_df, r=round)
        store.save_atomic(df2)

        json_out(
            {
                "ok": True,
                "round": int(round),
                "labels_ingested": int(len(labels_df)),
                "y_column": cfg.data.label_source_column_name,
            }
        )

    except OpalError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("ingest-y", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
