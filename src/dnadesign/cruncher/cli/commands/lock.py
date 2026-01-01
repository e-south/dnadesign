"""Lock command."""

from __future__ import annotations

from pathlib import Path

import typer

from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.services.lock_service import resolve_lock

app = typer.Typer(no_args_is_help=True, help="Resolve TF names to exact motif IDs and write lockfile.")


@app.callback(invoke_without_command=True)
def main(config: Path = typer.Argument(..., help="Path to cruncher config.yaml.", metavar="CONFIG")) -> None:
    cfg = load_config(config)
    catalog_root = config.parent / cfg.motif_store.catalog_root
    lock_path = catalog_root / "locks" / f"{config.stem}.lock.json"
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
    typer.echo(str(lock_path))
