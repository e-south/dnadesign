"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/prune_source.py

Prune records.parquet of any OPAL namespace columns (opal__*), plus the configured
Y column. Designed to "reset" a campaign's source table prior to (re)starting
round 0, avoiding stray caches or edge cases.

Robust, assertive semantics (no silent fallbacks):
  • Shows an explicit preview of columns to be deleted.
  • Protects essential USR columns and the configured X column.
  • Requires explicit confirmation unless --yes is provided.
  • Writes atomically; optional backup of original file.

Module Author(s): Eric J. South (drafted per user request)
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import typer

from ...data_access import ESSENTIAL_COLS
from ...locks import CampaignLock
from ...utils import ExitCodes, OpalError, file_sha256, print_stdout
from ..formatting import bullet_list, kv_block
from ..registry import cli_command
from ._common import (
    internal_error,
    json_out,
    load_cli_config,
    opal_error,
    resolve_config_path,
    store_from_cfg,
)


def _classify_columns(
    *,
    columns: List[str],
    campaign_slug: str,
    y_col: str,
    x_col: str,
    scope: str = "any",  # "any" | "campaign"
) -> Dict[str, List[str]]:
    """
    Partition columns into buckets we may prune.
      - opal_any:  any column starting with 'opal__'
      - opal_this_campaign: only those for this campaign's slug
      - y: the configured Y column (even if not opal__-prefixed)
    The caller chooses which bucket(s) to actually delete via scope.
    """
    re_ns = re.compile(r"^opal__([a-z0-9_-]+)__(.+)$")
    opal_any: List[str] = []
    opal_this: List[str] = []

    for c in columns:
        if c.startswith("opal__"):
            opal_any.append(c)
            m = re_ns.match(c)
            if m and m.group(1) == campaign_slug:
                opal_this.append(c)

    # Deleting Y is always considered (if present)
    y_bucket: List[str] = [y_col] if y_col in columns else []

    # Guardrails: never propose to delete essentials or X
    protected = set(ESSENTIAL_COLS + [x_col])
    opal_any = [c for c in opal_any if c not in protected]
    opal_this = [c for c in opal_this if c not in protected]
    y_bucket = [c for c in y_bucket if c not in protected]

    return {
        "opal_any": sorted(opal_any),
        "opal_this_campaign": sorted(opal_this),
        "y": sorted(y_bucket),
    }


@cli_command(
    "prune-source",
    help=(
        "Remove OPAL namespace columns (opal__*) and the configured Y column "
        "from records.parquet. Shows a preview and asks for confirmation."
    ),
)
def cmd_prune_source(
    config: Optional[Path] = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG", help="Path to campaign.yaml"),
    scope: str = typer.Option(
        "any",
        "--scope",
        help="Which opal namespaces to prune: 'any' (default) or 'campaign' (this campaign's slug only).",
        case_sensitive=False,
    ),
    keep: List[str] = typer.Option(
        None,
        "--keep",
        "-k",
        help="Column name(s) to keep even if matched for deletion. May be passed multiple times.",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip interactive prompt."),
    backup: bool = typer.Option(
        True,
        "--backup/--no-backup",
        help="Write a copy of the original file next to records.parquet before pruning (default: on).",
    ),
    json: bool = typer.Option(False, "--json/--human", help="Output format."),
) -> None:
    """
    Typical use:
      opal prune-source -c path/to/campaign.yaml
      opal prune-source -c . --scope any --yes
      opal prune-source -c . --scope campaign --keep opal__othercampaign__label_hist
    """
    scope = (scope or "any").strip().lower()
    if scope not in ("any", "campaign"):
        raise typer.BadParameter("--scope must be 'any' or 'campaign'")

    try:
        cfg_path = resolve_config_path(config)
        cfg = load_cli_config(cfg_path)
        store = store_from_cfg(cfg)
        rec_path = store.records_path

        if not rec_path.exists():
            raise OpalError(f"records.parquet not found: {rec_path}")

        import pandas as pd

        df = pd.read_parquet(rec_path)

        buckets = _classify_columns(
            columns=list(map(str, df.columns)),
            campaign_slug=cfg.campaign.slug,
            y_col=cfg.data.y_column_name,
            x_col=cfg.data.x_column_name,
            scope=scope,
        )

        # Select which opal bucket to use based on scope
        opal_cols = buckets["opal_any"] if scope == "any" else buckets["opal_this_campaign"]
        y_cols = buckets["y"]

        # Apply explicit keep list (exact column names)
        keep_set = set(keep or [])
        to_delete = [c for c in sorted(set(opal_cols + y_cols)) if c not in keep_set]

        preview_info = {
            "config_path": str(cfg_path.resolve()),
            "workdir": str(Path(cfg.campaign.workdir).resolve()),
            "records_path": str(rec_path.resolve()),
            "table_shape_before": f"{df.shape[0]} rows × {df.shape[1]} cols",
            "campaign_slug": cfg.campaign.slug,
            "x_column_name": cfg.data.x_column_name,
            "y_column_name": cfg.data.y_column_name,
            "scope": scope,
            "opal_namespace_columns_found": len(opal_cols),
            "y_column_present": bool(len(y_cols) > 0),
            "to_delete_count": len(to_delete),
        }

        if json:
            json_out(
                {
                    "preview": preview_info,
                    "opal_namespace_columns": opal_cols,
                    "y_columns": y_cols,
                    "keep": sorted(list(keep_set)),
                    "to_delete": to_delete,
                }
            )
            if not yes:
                # In JSON mode, do not mutate unless --yes was passed.
                raise typer.Exit(code=ExitCodes.OK)
        else:
            # Human preview & warning banner
            head = kv_block("[Preview] prune-source", preview_info)
            opal_list = bullet_list("OPAL columns (matching scope)", opal_cols or [])
            y_list = bullet_list("Y column", (y_cols or ["(not present)"]))
            del_list = bullet_list("WILL DELETE", to_delete or ["(nothing)"])
            warning = "\n".join(
                [
                    "",
                    "⚠️  WARNING: This operation rewrites records.parquet.",
                    "   • Essentials and your X column are protected.",
                    "   • Consider keeping a backup (on by default; see --no-backup).",
                ]
            )
            print_stdout("\n".join([head, "", opal_list, "", y_list, "", del_list, warning]))

        if len(to_delete) == 0:
            if not json:
                print_stdout("\nNothing to prune. Exiting.")
            raise typer.Exit(code=ExitCodes.OK)

        if not yes and not json:
            resp = input("Proceed to DELETE the columns above and rewrite records.parquet? (y/N): ").strip().lower()  # noqa
            if resp not in ("y", "yes"):
                print_stdout("Aborted.")
                raise typer.Exit(code=ExitCodes.BAD_ARGS)

        # Perform mutation under a campaign lock to avoid concurrent writers
        with CampaignLock(Path(cfg.campaign.workdir)):
            sha_before = file_sha256(rec_path)

            # Optional backup (exact byte copy)
            backup_path: Optional[Path] = None
            if backup:
                ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                backup_path = rec_path.with_name(f"{rec_path.stem}.backup.{ts}{rec_path.suffix}")
                shutil.copy2(rec_path, backup_path)

            # Drop requested columns and write atomically
            df2 = df.drop(columns=[c for c in to_delete if c in df.columns], errors="ignore")
            # Assertive safety: USR essentials + X must remain
            missing_post = [c for c in (ESSENTIAL_COLS + [store.x_col]) if c not in df2.columns]
            if missing_post:
                raise OpalError(
                    "Post-prune safety check failed; protected columns are missing: "
                    f"{missing_post}. No write performed."
                )

            # Persist
            store.save_atomic(df2)
            sha_after = file_sha256(rec_path)

        out = {
            "ok": True,
            "records_path": str(rec_path.resolve()),
            "rows": int(df2.shape[0]),
            "columns_before": int(df.shape[1]),
            "columns_after": int(df2.shape[1]),
            "deleted_columns": to_delete,
            "deleted_count": int(len(to_delete)),
            "backup_path": (str(backup_path.resolve()) if backup and backup_path else None),
            "sha256_before": sha_before,
            "sha256_after": sha_after,
        }

        if json:
            json_out(out)
        else:
            body = kv_block(
                "[Committed] prune-source",
                {
                    "records": out["records_path"],
                    "rows": out["rows"],
                    "columns before": out["columns_before"],
                    "columns after": out["columns_after"],
                    "deleted": out["deleted_count"],
                    "backup": out["backup_path"] or "(none)",
                    "sha256 before": out["sha256_before"][:12],
                    "sha256 after": out["sha256_after"][:12],
                },
            )
            print_stdout("\n" + body)

    except typer.Exit:
        raise
    except OpalError as e:
        opal_error("prune-source", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("prune-source", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
