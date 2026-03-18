"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/cli/commands_fit.py

Fit- and sweep-related cluster CLI command registration.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List, Optional

import typer
from rich.console import Console

from .resolution import (
    resolve_fit_method_params,
    resolve_workspace_context,
    resolve_workspace_value,
    runs_root_or_exit,
)


def register_fit_command(app: typer.Typer, *, console: Console) -> None:
    @app.command(
        "fit",
        help="Run one clustering method on X, attach minimal columns, and catalog a fit run.",
    )
    def cmd_fit(
        ctx: typer.Context,
        workspace: Optional[str] = typer.Option(None, help="Workspace directory or packaged workspace id."),
        results_root: Optional[str] = typer.Option(
            None,
            help="Standalone artifact root. Required unless --workspace is set.",
        ),
        dataset: Optional[str] = typer.Option(None, help="USR dataset name"),
        file: Optional[str] = typer.Option(None, help="Parquet/CSV path"),
        usr_root: Optional[str] = typer.Option(None, help="USR root directory"),
        name: Optional[str] = typer.Option(None, help="Run alias (slug). If omitted, auto-generated."),
        key_col: str = typer.Option("id", help="Key column"),
        x_col: Optional[str] = typer.Option(None, help="Vector column (list<float> or JSON array string)"),
        x_cols: Optional[str] = typer.Option(None, help="Comma-separated list of numeric columns"),
        method: str = typer.Option("leiden", help="Clustering method id", show_default=True),
        preset: Optional[str] = typer.Option(None, help="Preset name (kind: 'method') to pre-fill parameters"),
        method_param: List[str] = typer.Option(
            [],
            "--method-param",
            help="Method-specific parameter override as key=value. Repeatable.",
        ),
        silhouette: bool = typer.Option(False, help="Attach per-row silhouette quality as cluster__<NAME>__quality"),
        full_silhouette: bool = typer.Option(False, help="Compute silhouette on all rows (default samples to ≤20k)"),
        dedupe_policy: str = typer.Option(
            "error",
            help="Duplicate id policy: error|keep-first|keep-last",
            show_default=True,
        ),
        reuse: str = typer.Option("auto", help="Reuse policy: auto|require|never|reattach", show_default=True),
        force: bool = typer.Option(False, help="Force recompute (ignore reuse cache)", show_default=True),
        write: bool = typer.Option(False, help="Apply changes to the table"),
        yes: bool = typer.Option(
            False,
            "-y",
            "--allow-overwrite",
            help="Allow overwriting already-attached columns in USR/file writes",
        ),
        inplace: bool = typer.Option(False, help="Rewrite the input file in place (generic files only)"),
        out: Optional[str] = typer.Option(None, help="Output file path for generic files"),
    ) -> None:
        from ..execution import run_fit

        workspace_ctx = resolve_workspace_context(workspace, expected_section="fit")
        wp = workspace_ctx.params
        dataset = resolve_workspace_value(ctx, option_name="dataset", cli_value=dataset, config_params=wp)
        file = resolve_workspace_value(ctx, option_name="file", cli_value=file, config_params=wp)
        usr_root = resolve_workspace_value(ctx, option_name="usr_root", cli_value=usr_root, config_params=wp)
        name = resolve_workspace_value(ctx, option_name="name", cli_value=name, config_params=wp)
        key_col = resolve_workspace_value(ctx, option_name="key_col", cli_value=key_col, config_params=wp)
        x_col = resolve_workspace_value(ctx, option_name="x_col", cli_value=x_col, config_params=wp)
        if x_col:
            x_col = str(x_col).strip()
        x_cols = resolve_workspace_value(ctx, option_name="x_cols", cli_value=x_cols, config_params=wp)
        method = resolve_workspace_value(ctx, option_name="method", cli_value=method, config_params=wp)
        preset = resolve_workspace_value(ctx, option_name="preset", cli_value=preset, config_params=wp)
        raw_method_params = resolve_fit_method_params(wp, method_param, preset_name=preset)
        silhouette = bool(
            resolve_workspace_value(
                ctx,
                option_name="silhouette",
                cli_value=silhouette,
                config_params=wp,
            )
        )
        full_silhouette = bool(
            resolve_workspace_value(ctx, option_name="full_silhouette", cli_value=full_silhouette, config_params=wp)
        )
        dedupe_policy = resolve_workspace_value(
            ctx,
            option_name="dedupe_policy",
            cli_value=dedupe_policy,
            config_params=wp,
        )
        reuse = resolve_workspace_value(ctx, option_name="reuse", cli_value=reuse, config_params=wp)
        force = bool(resolve_workspace_value(ctx, option_name="force", cli_value=force, config_params=wp))
        write = bool(resolve_workspace_value(ctx, option_name="write", cli_value=write, config_params=wp))
        yes = bool(
            resolve_workspace_value(
                ctx,
                option_name="yes",
                cli_value=yes,
                config_params=wp,
                config_key="allow_overwrite",
            )
        )
        inplace = bool(resolve_workspace_value(ctx, option_name="inplace", cli_value=inplace, config_params=wp))
        out = resolve_workspace_value(ctx, option_name="out", cli_value=out, config_params=wp)
        root = runs_root_or_exit(
            console=console,
            workspace_root=workspace_ctx.results_root,
            results_root=results_root,
        )
        run_fit(
            dataset=dataset,
            file=file,
            usr_root=usr_root,
            name=name,
            key_col=key_col,
            x_col=x_col,
            x_cols=x_cols,
            method=method,
            preset=preset,
            method_params=raw_method_params,
            silhouette=silhouette,
            full_silhouette=full_silhouette,
            dedupe_policy=dedupe_policy,
            reuse=reuse,
            force=force,
            write=write,
            allow_overwrite=yes,
            inplace=inplace,
            out=out,
            root=root,
            workspace_id=workspace_ctx.workspace_id,
            console=console,
        )


def register_sweep_command(app: typer.Typer, *, console: Console) -> None:
    @app.command(
        "sweep",
        help="Run a method-scoped resolution sweep for methods that expose a resolution parameter.",
    )
    def cmd_sweep(
        ctx: typer.Context,
        workspace: Optional[str] = typer.Option(None, help="Workspace directory or packaged workspace id."),
        results_root: Optional[str] = typer.Option(
            None,
            help="Standalone artifact root. Required unless --workspace is set.",
        ),
        dataset: Optional[str] = typer.Option(None),
        file: Optional[str] = typer.Option(None),
        usr_root: Optional[str] = typer.Option(None),
        key_col: str = typer.Option("id"),
        x_col: Optional[str] = typer.Option(None),
        x_cols: Optional[str] = typer.Option(None),
        method: str = typer.Option(..., help="Clustering method id for the sweep"),
        preset: Optional[str] = typer.Option(None, help="Preset name (kind: 'method') to pre-fill parameters"),
        method_param: List[str] = typer.Option(
            [],
            "--method-param",
            help="Method-specific parameter override as key=value. Repeatable.",
        ),
        res_min: float = typer.Option(0.05),
        res_max: float = typer.Option(1.0),
        step: float = typer.Option(0.05),
        replicates: int = typer.Option(5),
        seeds: str = typer.Option("1,2,3,4,5"),
        out_dir: Optional[str] = typer.Option(
            None,
            help="Optional subdirectory under the chosen cluster artifact root. "
            "If omitted, cluster records the sweep under <results-root>/<alias>/sweeps/<run-slug>/.",
        ),
    ) -> None:
        from ..execution import run_sweep

        workspace_ctx = resolve_workspace_context(workspace, expected_section="fit")
        wp = workspace_ctx.params
        dataset = resolve_workspace_value(ctx, option_name="dataset", cli_value=dataset, config_params=wp)
        file = resolve_workspace_value(ctx, option_name="file", cli_value=file, config_params=wp)
        usr_root = resolve_workspace_value(ctx, option_name="usr_root", cli_value=usr_root, config_params=wp)
        key_col = resolve_workspace_value(ctx, option_name="key_col", cli_value=key_col, config_params=wp)
        x_col = resolve_workspace_value(ctx, option_name="x_col", cli_value=x_col, config_params=wp)
        if x_col:
            x_col = str(x_col).strip()
        x_cols = resolve_workspace_value(ctx, option_name="x_cols", cli_value=x_cols, config_params=wp)
        method = resolve_workspace_value(ctx, option_name="method", cli_value=method, config_params=wp)
        preset = resolve_workspace_value(ctx, option_name="preset", cli_value=preset, config_params=wp)
        raw_method_params = resolve_fit_method_params(wp, method_param, preset_name=preset)
        root = runs_root_or_exit(
            console=console,
            workspace_root=workspace_ctx.results_root,
            results_root=results_root,
        )
        run_sweep(
            dataset=dataset,
            file=file,
            usr_root=usr_root,
            key_col=key_col,
            x_col=x_col,
            x_cols=x_cols,
            method=method,
            preset=preset,
            method_params=raw_method_params,
            res_min=res_min,
            res_max=res_max,
            step=step,
            replicates=replicates,
            seeds=seeds,
            out_dir=out_dir,
            root=root,
            workspace_id=workspace_ctx.workspace_id,
            console=console,
        )


__all__ = ["register_fit_command", "register_sweep_command"]
