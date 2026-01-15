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
from dnadesign.cruncher.app.lock_service import resolve_lock
from dnadesign.cruncher.cli.config_resolver import (
    ConfigResolutionError,
    resolve_config_path,
)
from dnadesign.cruncher.cli.paths import render_path
from dnadesign.cruncher.config.load import load_config


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
    cfg = load_config(config_path)
    catalog_root = config_path.parent / cfg.motif_store.catalog_root
    lock_path = catalog_root / "locks" / f"{config_path.stem}.lock.json"
    names = {tf for group in cfg.regulator_sets for tf in group}
    try:
        resolve_lock(
            names=names,
            catalog_root=catalog_root,
            source_preference=cfg.motif_store.source_preference,
            allow_ambiguous=cfg.motif_store.allow_ambiguous,
            pwm_source=cfg.motif_store.pwm_source,
            site_kinds=cfg.motif_store.site_kinds,
            combine_sites=cfg.motif_store.combine_sites,
            dataset_preference=cfg.motif_store.dataset_preference,
            dataset_map=cfg.motif_store.dataset_map,
            lock_path=lock_path,
        )
    except ValueError as exc:
        typer.echo(f"Error: {exc}", err=True)
        typer.echo("Hint: run cruncher fetch motifs/sites before locking.", err=True)
        raise typer.Exit(code=1)
    typer.echo(render_path(lock_path, base=config_path.parent))
