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
    "ingest-y",
    help=(
        "Ingest tidy CSV → Y (preview + confirmation; strict checks). "
        "If the CSV lacks an 'id' column, OPAL resolves rows by 'sequence'. "
        "IDs are never inferred from other columns."
    ),
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
      • Uses transform from YAML (overridable via flags).
      • Prints a preview (counts + sample) before any write.
      • If CSV has no 'id', OPAL resolves by 'sequence':
          - If sequence exists → use that row's id.
          - If already labeled in this campaign → fail fast.
          - If not present → add a new row with essentials and a generated id.
      • History is immutable: per-(id, round) conflicts cause an error.
    """
    try:
        cfg = load_config(resolve_config_path(config))
        store = store_from_cfg(cfg)
        df = store.load()

        # Resolve CSV path with helpful fallbacks
        def _resolve_csv(p: Path) -> Path:
            if p.is_absolute() and p.exists():
                return p
            # try CWD
            q = Path.cwd() / p
            if q.exists():
                return q
            # try workdir
            wd = Path(cfg.campaign.workdir)
            q = wd / p
            if q.exists():
                return q
            # try common input locations
            candidates = [
                wd / "inputs" / p.name,
                wd / "inputs" / f"r{round}" / p.name,
                wd / "inputs" / f"r{round}" / str(p),
            ]
            for c in candidates:
                if c.exists():
                    return c
            # last resort: search by basename under inputs/
            matches = list((wd / "inputs").rglob(p.name))
            if len(matches) == 1:
                return matches[0]
            hint = (
                ("\n  suggestions:\n    - " + "\n    - ".join(map(str, matches[:10])))
                if matches
                else ""
            )
            raise OpalError(
                f"CSV not found: {p}.\n  cwd={Path.cwd()}\n  workdir={wd}{hint}"
            )

        csv_path = _resolve_csv(Path(csv))
        # Load with format sniffing
        csv_df = (
            pd.read_parquet(csv_path)
            if csv_path.suffix.lower() in (".pq", ".parquet")
            else pd.read_csv(csv_path)
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
            setpoint_vector=cfg.objective.objective.params.get(
                "setpoint_vector", [0, 0, 0, 1]
            ),
        )

        # PREVIEW
        json_out(
            {
                "preview": preview.__dict__,
                "sample": labels_df.head(5).to_dict(orient="records"),
                "source_csv": str(csv_path),
            }
        )

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

        # Ensure rows exist / resolve IDs (by sequence when id is absent)
        df = store.ensure_rows_exist(df, labels_df, csv_df)

        # Append labels and save atomically (fail if the sample already has ANY labels)
        df2 = store.append_labels_from_df(
            df,
            labels_df[["id", "y"]],
            r=round,
            fail_if_any_existing_labels=True,
        )
        store.save_atomic(df2)

        json_out(
            {
                "ok": True,
                "round": int(round),
                "labels_ingested": int(len(labels_df)),
                "y_column": cfg.data.y_column_name,
            }
        )
    except OpalError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("ingest-y", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
