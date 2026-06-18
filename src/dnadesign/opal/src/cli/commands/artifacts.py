"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/opal/src/cli/commands/artifacts.py

CLI commands for manifest-authoritative artifact gardening.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ...core.utils import ExitCodes, OpalError, print_stdout
from ..formatting import kv_block
from ..registry import cli_group
from ._common import internal_error, json_error, json_out, opal_error

artifacts_app = typer.Typer(no_args_is_help=True, help="Audit and prune generated OPAL artifacts.")
cli_group("artifacts", help="Audit and prune generated OPAL artifacts.")(artifacts_app)


@artifacts_app.command("audit", help="Inventory generated artifacts and stale manifest-absent files.")
def cmd_artifacts_audit(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="campaign.yaml or campaign directory",
        envvar="OPAL_CONFIG",
    ),
    json: bool = typer.Option(False, "--json/--text", help="Output format."),
) -> None:
    try:
        from ...reporting.artifact_garden import build_artifact_garden_audit

        audit = build_artifact_garden_audit(config)
        if json:
            json_out(audit)
            return
        print_stdout(_render_audit_text(audit))
    except OpalError as e:
        if json:
            json_error("artifacts audit", e)
        else:
            opal_error("artifacts audit", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("artifacts audit", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)


@artifacts_app.command("prune", help="Prune stale artifacts; dry-run unless --apply is passed.")
def cmd_artifacts_prune(
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="campaign.yaml or campaign directory",
        envvar="OPAL_CONFIG",
    ),
    apply: bool = typer.Option(False, "--apply", help="Delete stale artifacts from the active prune plan."),
    json: bool = typer.Option(False, "--json/--text", help="Output format."),
) -> None:
    try:
        from ...reporting.artifact_garden import prune_stale_artifacts

        result = prune_stale_artifacts(config, apply=apply)
        if json:
            json_out(result)
            return
        print_stdout(_render_prune_text(result))
    except OpalError as e:
        if json:
            json_error("artifacts prune", e)
        else:
            opal_error("artifacts prune", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("artifacts prune", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)


def _render_audit_text(audit: dict) -> str:
    plan = audit.get("prune_plan") or {}
    return kv_block(
        "OPAL artifact audit",
        {
            "campaign": (audit.get("campaign") or {}).get("slug"),
            "root": audit.get("root"),
            "local_only": audit.get("local_only"),
            "active_manifests": len(audit.get("active_manifests") or []),
            "stale_artifacts": len(audit.get("stale_artifacts") or []),
            "stale_bytes": (audit.get("bytes") or {}).get("stale_artifacts"),
            "prune_requires_apply": plan.get("requires_apply"),
        },
    )


def _render_prune_text(result: dict) -> str:
    applied = bool(result.get("applied"))
    title = "OPAL artifact prune" if applied else "OPAL artifact prune dry-run"
    return kv_block(
        title,
        {
            "campaign": (result.get("campaign") or {}).get("slug"),
            "root": result.get("root"),
            "applied": applied,
            "stale_artifacts": len(result.get("stale_artifacts") or []),
            "deleted_count": result.get("deleted_count"),
            "bytes_deleted": result.get("bytes_deleted"),
            "next": None if applied else "re-run with --apply to delete stale artifacts",
        },
    )
