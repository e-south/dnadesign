"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/campaign_reset.py

Resets a campaign to a clean slate for repeatable runs. Prunes OPAL columns,
removes outputs, and clears state.json.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
import sys
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
    resolve_config_path,
    store_from_cfg,
)
from .prune_source import _classify_columns


def _prompt_slug_confirm(slug: str) -> bool:
    if not sys.stdin.isatty():
        raise OpalError(
            "No TTY available. Re-run with --apply to confirm campaign-reset.",
            ExitCodes.BAD_ARGS,
        )
    resp = input(f"Type the campaign slug '{slug}' to confirm: ").strip()
    return resp == slug


@cli_command(
    "campaign-reset",
    help="Reset a campaign to a clean slate (hidden).",
    hidden=True,
)
def cmd_campaign_reset(
    config: Optional[Path] = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG", help="Path to campaign.yaml"),
    apply: bool = typer.Option(False, "--apply", help="Apply reset without interactive confirmation."),
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

        rec_path = store.records_path
        if not rec_path.exists():
            raise OpalError(f"records.parquet not found: {rec_path}")

        workdir = Path(cfg.campaign.workdir)
        outputs_dir = workdir / "outputs"
        notebooks_dir = workdir / "notebooks"
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
            "campaign_slug": cfg.campaign.slug,
            "columns_to_prune": len(to_delete),
            "outputs_dir": str(outputs_dir.resolve()),
            "outputs_exists": outputs_dir.exists(),
            "notebooks_dir": str(notebooks_dir.resolve()),
            "notebooks_exists": notebooks_dir.exists(),
            "notebooks_count": len(list(notebooks_dir.glob("*.py"))) if notebooks_dir.exists() else 0,
            "state_path": str(state_path.resolve()),
            "state_exists": state_path.exists(),
            "backup_records": bool(backup),
        }

        if json:
            json_out({"preview": preview, "to_delete": to_delete})
            if not apply:
                raise typer.Exit(code=ExitCodes.OK)
        else:
            head = kv_block("[Preview] campaign-reset", preview)
            del_list = bullet_list("Columns to prune from records.parquet", to_delete or ["(none)"])
            warning = "\n".join(
                [
                    "",
                    "⚠️  WARNING: campaign-reset rewrites records.parquet and removes outputs/notebooks/state.json.",
                    "   • Essentials and your X column are protected.",
                    "   • Use --backup if you want a copy of records.parquet.",
                ]
            )
            print_stdout("\n".join([head, "", del_list, warning]))

        if not apply and not json:
            confirmed = _prompt_slug_confirm(cfg.campaign.slug)
            if not confirmed:
                print_stdout("Aborted.")
                raise typer.Exit(code=ExitCodes.BAD_ARGS)

        outputs_existed = outputs_dir.exists()
        notebooks_existed = notebooks_dir.exists()
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
            if notebooks_dir.exists():
                shutil.rmtree(notebooks_dir)
            if state_path.exists():
                state_path.unlink()

        result = {
            "ok": True,
            "columns_pruned": len(to_delete),
            "outputs_removed": bool(outputs_existed),
            "notebooks_removed": bool(notebooks_existed),
            "state_removed": bool(state_existed),
        }
        if json:
            json_out(result)
        else:
            print_stdout(kv_block("campaign-reset", result))
    except OpalError as e:
        opal_error("campaign-reset", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("campaign-reset", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
