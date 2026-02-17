"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/opal/src/cli/commands/guide.py

CLI command group for guided workflow runbooks and state-aware next-step
recommendations.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from ...core.utils import ExitCodes, OpalError, print_stdout
from ..formatting.renderers.guide import (
    render_guide_markdown,
    render_guide_text,
    render_next_human,
)
from ..guidance import build_guidance_report, build_next_guidance
from ..registry import cli_group
from ._common import (
    internal_error,
    json_out,
    load_cli_config,
    opal_error,
    resolve_config_path,
    store_from_cfg,
)

guide_app = typer.Typer(
    no_args_is_help=False,
    help="Generate guided OPAL runbooks and next-step recommendations.",
)
cli_group("guide", help="Guided workflow helper commands.")(guide_app)


@guide_app.callback(invoke_without_command=True)
def cmd_guide(
    ctx: typer.Context,
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
    labels_as_of: int = typer.Option(0, "--labels-as-of", help="Labels cutoff used in generated runbook commands."),
    format: str = typer.Option("text", "--format", help="Runbook output format: text, markdown, or json."),
    out: Optional[Path] = typer.Option(None, "--out", help="Optional output file path."),
) -> None:
    if ctx.invoked_subcommand is not None:
        return
    try:
        fmt = str(format).strip().lower()
        if fmt not in {"text", "markdown", "json"}:
            raise OpalError("--format must be one of: text, markdown, json.")
        cfg_path = resolve_config_path(config)
        cfg = load_cli_config(cfg_path)
        report = build_guidance_report(cfg_path, cfg, labels_as_of=int(labels_as_of))
        if fmt == "json":
            if out is not None:
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(__import__("json").dumps(report, default=lambda o: o.__dict__, indent=2))
            json_out(report)
            return
        rendered = render_guide_markdown(report) if fmt == "markdown" else render_guide_text(report)
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(rendered)
        print_stdout(rendered)
    except OpalError as e:
        opal_error("guide", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("guide", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)


@guide_app.command("next", help="Inspect campaign state and print the recommended next command.")
def cmd_guide_next(
    config: Path = typer.Option(None, "--config", "-c", envvar="OPAL_CONFIG"),
    observed_round: Optional[int] = typer.Option(None, "--observed-round", help="Observed round to inspect."),
    labels_as_of: Optional[int] = typer.Option(None, "--labels-as-of", help="Labels cutoff round to inspect."),
    json: bool = typer.Option(False, "--json/--human", help="Output format."),
) -> None:
    try:
        cfg_path = resolve_config_path(config)
        cfg = load_cli_config(cfg_path)
        store = store_from_cfg(cfg)
        df = store.load()
        report = build_next_guidance(
            cfg_path,
            cfg,
            store,
            df,
            labels_as_of=labels_as_of,
            observed_round=observed_round,
        )
        if json:
            json_out(report)
        else:
            print_stdout(render_next_human(report))
    except OpalError as e:
        opal_error("guide next", e)
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        internal_error("guide next", e)
        raise typer.Exit(code=ExitCodes.INTERNAL_ERROR)
