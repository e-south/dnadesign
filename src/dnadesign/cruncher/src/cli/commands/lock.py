"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/lock.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import typer

from dnadesign.cruncher.app.cache_readiness import lock_refresh_hint
from dnadesign.cruncher.app.lock_service import resolve_lock
from dnadesign.cruncher.cli.config_resolver import (
    ConfigResolutionError,
    resolve_config_path,
)
from dnadesign.cruncher.cli.paths import render_path
from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.utils.paths import resolve_catalog_root, resolve_lock_path, resolve_workspace_root


def lock(
    config: Path | None = typer.Argument(
        None,
        help="Path to cruncher config.yaml (resolved from workspace/CWD if omitted).",
        metavar="CONFIG",
    ),
    config_option: Path | None = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to cruncher config.yaml (overrides positional CONFIG).",
    ),
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)
    try:
        cfg = load_config(config_path)
    except (ValueError, FileNotFoundError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1)
    catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)
    lock_path = resolve_lock_path(config_path)
    names = {tf for group in cfg.regulator_sets for tf in group}
    if not names:
        typer.echo("Error: lock requires at least one TF target.", err=True)
        raise typer.Exit(code=1)
    try:
        resolve_lock(
            names=names,
            catalog_root=catalog_root,
            source_preference=cfg.catalog.source_preference,
            allow_ambiguous=cfg.catalog.allow_ambiguous,
            pwm_source=cfg.catalog.pwm_source,
            site_kinds=cfg.catalog.site_kinds,
            combine_sites=cfg.catalog.combine_sites,
            dataset_preference=cfg.catalog.dataset_preference,
            dataset_map=cfg.catalog.dataset_map,
            lock_path=lock_path,
        )
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        typer.echo(lock_refresh_hint(), err=True)
        raise typer.Exit(code=1)
    typer.echo(render_path(lock_path, base=resolve_workspace_root(config_path)))
