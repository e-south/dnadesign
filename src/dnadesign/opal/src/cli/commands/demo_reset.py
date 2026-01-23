# ABOUTME: Resets the demo campaign to a clean slate for repeatable runs.
# ABOUTME: Prunes OPAL columns, removes outputs, and clears state.json.
"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/demo_reset.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

from ...core.utils import ExitCodes, OpalError, print_stdout
from ...storage.locks import CampaignLock
from ...storage.parquet_io import read_parquet_df
from ..formatting import bullet_list, kv_block
from ..registry import cli_command
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
from .prune_source import _classify_columns


@cli_command(
    "demo-reset",
    help="Reset the demo campaign to a clean slate (hidden).",
    hidden=True,
)
def cmd_demo_reset(
    config: Optional[Path] = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG", help="Path to campaign.yaml"),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip interactive confirmation."),
    backup: bool = typer.Option(
        False,
        "--backup/--no-backup",
        help="Backup records.parquet before pruning (default: no-backup).",
    ),
    json: bool = typer.Option(False, "--json/--human", help="Output format."),
) -> None:
    try:
        cfg_path = resolve_config_path(config)
        cfg = load_cli_config(cfg_path)
        store = store_from_cfg(cfg)
        if not json:
            print_config_context(cfg_path, cfg=cfg, records_path=store.records_path)

        if cfg.campaign.slug != "demo":
            raise OpalError("demo-reset is restricted to the demo campaign (slug must be 'demo').")

        rec_path = store.records_path
        if not rec_path.exists():
            raise OpalError(f"records.parquet not found: {rec_path}")

        workdir = Path(cfg.campaign.workdir)
        outputs_dir = workdir / "outputs"
        state_path = workdir / "state.json"

        df = read_parquet_df(rec_path)
        buckets = _classify_columns(
            columns=list(map(str, df.columns)),
            campaign_slug=cfg.campaign.slug,
            y_col=cfg.data.y_column_name,
            x_col=cfg.data.x_column_name,
            scope="any",
        )
        to_delete = sorted(set(buckets["opal_any"] + buckets["y"]))

        preview = {
            "config_path": str(cfg_path.resolve()),
            "workdir": str(workdir.resolve()),
            "records_path": str(rec_path.resolve()),
            "table_shape_before": f"{df.shape[0]} rows × {df.shape[1]} cols",
            "columns_to_prune": len(to_delete),
            "outputs_dir": str(outputs_dir.resolve()),
            "outputs_exists": outputs_dir.exists(),
            "state_path": str(state_path.resolve()),
            "state_exists": state_path.exists(),
            "backup_records": bool(backup),
        }

        if json:
            json_out({"preview": preview, "to_delete": to_delete})
            if not yes:
                raise typer.Exit(code=ExitCodes.OK)
        else:
            head = kv_block("[Preview] demo-reset", preview)
            del_list = bullet_list("Columns to prune from records.parquet", to_delete or ["(none)"])
            warning = "\n".join(
                [
                    "",
                    "⚠️  WARNING: demo-reset rewrites records.parquet and removes outputs/state.json.",
                    "   • Essentials and your X column are protected.",
                    "   • Use --backup if you want a copy of records.parquet.",
                ]
            )
            print_stdout("\n".join([head, "", del_list, warning]))

        if not yes and not json:
            if not prompt_confirm(
                "Proceed with demo-reset? This will remove outputs/ and state.json. (y/N): ",
                non_interactive_hint="No TTY available. Re-run with --yes to confirm demo-reset.",
            ):
                print_stdout("Aborted.")
                raise typer.Exit(code=ExitCodes.BAD_ARGS)

        outputs_existed = outputs_dir.exists()
        state_existed = state_path.exists()

        with CampaignLock(workdir):
            if backup and rec_path.exists():
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                backup_path = rec_path.with_name(f"{rec_path.name}.bak-{stamp}")
                shutil.copy2(rec_path, backup_path)

            if to_delete:
                df2 = df.drop(columns=to_delete)
                store.save_atomic(df2)

            if outputs_dir.exists():
                shutil.rmtree(outputs_dir)
            if state_path.exists():
                state_path.unlink()

        result = {
            "ok": True,
            "columns_pruned": len(to_delete),
            "outputs_removed": bool(outputs_existed),
            "state_removed": bool(state_existed),
        }
        if json:
            json_out(result)
        else:
            print_stdout(kv_block("demo-reset", result))
    except OpalError as e:
        opal_error("demo-reset", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("demo-reset", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
