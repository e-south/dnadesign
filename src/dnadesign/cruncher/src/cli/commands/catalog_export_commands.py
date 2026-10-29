"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/cli/commands/catalog_export_commands.py

Catalog export command registration for motif artifacts and site tables.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.table import Table

from dnadesign.cruncher.app.motif_artifacts import build_manifest
from dnadesign.cruncher.cli.catalog_execution import (
    collect_site_export_rows,
    write_meme_artifacts,
    write_site_export,
)
from dnadesign.cruncher.cli.catalog_targets import _resolve_site_targets, _resolve_targets
from dnadesign.cruncher.cli.catalog_utils import (
    _resolve_export_format,
    _resolve_user_path,
)
from dnadesign.cruncher.cli.commands.catalog_common import console, load_config_or_exit
from dnadesign.cruncher.cli.config_resolver import ConfigResolutionError, resolve_config_path
from dnadesign.cruncher.utils.paths import resolve_catalog_root, resolve_workspace_root


def _resolve_motif_export_targets(
    *,
    cfg: object,
    config_path: Path,
    tfs: list[str],
    refs: list[str],
    set_index: int | None,
    source: str | None,
) -> tuple[list[object], object]:
    try:
        return _resolve_targets(
            cfg=cfg,
            config_path=config_path,
            tfs=tfs,
            refs=refs,
            set_index=set_index,
            source_filter=source,
        )
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        console.print("Hint: run cruncher fetch motifs/sites before catalog export-meme.")
        raise typer.Exit(code=1) from exc


def _resolve_site_export_targets(
    *,
    cfg: object,
    config_path: Path,
    tfs: list[str],
    refs: list[str],
    set_index: int | None,
    source: str | None,
) -> tuple[list[object], object]:
    try:
        return _resolve_site_targets(
            cfg=cfg,
            config_path=config_path,
            tfs=tfs,
            refs=refs,
            set_index=set_index,
            source_filter=source,
        )
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        console.print("Hint: run cruncher fetch sites --hydrate before export-sites.")
        raise typer.Exit(code=1) from exc


def _prepare_site_export_output_path(*, out_path: Path, overwrite: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_path.exists():
        return
    if out_path.is_dir():
        console.print(f"[red]Output path is a directory:[/] {out_path}")
        raise typer.Exit(code=1)
    if not overwrite:
        console.print(f"[red]Output file already exists:[/] {out_path}")
        raise typer.Exit(code=1)


def _portable_manifest_path(path: Path, *, start: Path) -> Path:
    return Path(os.path.relpath(path.resolve(), start.resolve()))


def _manifest_anchor(path: Path) -> Path:
    for parent in (path.parent, *path.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return path.parent


def _write_motif_manifest_file(
    *,
    out_dir: Path,
    producer: str,
    manifest_entries: list[dict[str, object]],
    config_path: Path,
    catalog_root: Path,
    background: str,
    pseudocount: float | None,
) -> Path:
    config_base = _manifest_anchor(config_path)
    manifest = build_manifest(
        producer=producer,
        entries=manifest_entries,
        config_path=_portable_manifest_path(config_path, start=config_base),
        catalog_root=_portable_manifest_path(catalog_root, start=config_base),
        background_policy=background,
        pseudocount=pseudocount,
    )
    manifest["created_at"] = datetime.now(timezone.utc).isoformat()
    manifest_path = out_dir / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest_path


def _print_meme_export_table(*, table_rows: list[dict[str, str]], manifest_path: Path) -> None:
    table = Table(title="MEME motif exports", header_style="bold")
    table.add_column("TF")
    table.add_column("Source")
    table.add_column("Motif ID")
    table.add_column("Artifact")
    for row in table_rows:
        table.add_row(row["tf_name"], row["source"], row["motif_id"], row["artifact"])
    console.print(table)
    console.print(f"[green]Wrote manifest:[/] {manifest_path}")


def _print_site_export_table(*, table_rows: list[dict[str, str]], out_path: Path, row_count: int) -> None:
    table = Table(title="Binding-site export", header_style="bold")
    table.add_column("TF")
    table.add_column("Source")
    table.add_column("Motif ID")
    table.add_column("Sites")
    for row in table_rows:
        table.add_row(row["tf_name"], row["source"], row["motif_id"], row["count"])
    console.print(table)
    console.print(f"[green]Wrote binding sites:[/] {out_path} ({row_count} rows)")


def register_catalog_export_commands(app: typer.Typer) -> None:
    @app.command("export-meme", help="Export cached motifs as minimal MEME files (one file per motif).")
    def export_meme(
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
        tf: list[str] = typer.Option([], "--tf", help="TF name to include (repeatable)."),
        ref: list[str] = typer.Option([], "--ref", help="Catalog reference (<source>:<motif_id>, repeatable)."),
        set_index: int | None = typer.Option(
            None,
            "--set",
            help="Regulator set index from config (1-based).",
        ),
        source: str | None = typer.Option(
            None,
            "--source",
            help="Limit TF resolution to a single source adapter.",
        ),
        out_dir: Path | None = typer.Option(
            None,
            "--out",
            "-o",
            help="Directory to write MEME files (defaults to <out_dir>/artifacts/meme).",
        ),
        producer: str = typer.Option(
            "cruncher",
            "--producer",
            help="Producer label written into the export manifest.",
        ),
        overwrite: bool = typer.Option(False, "--overwrite", help="Allow overwriting existing files."),
    ) -> None:
        try:
            config_path = resolve_config_path(config_option or config)
        except ConfigResolutionError as exc:
            console.print(str(exc))
            raise typer.Exit(code=1)
        cfg = load_config_or_exit(config_path)
        producer_clean = producer.strip()
        if not producer_clean:
            raise typer.BadParameter("--producer must be a non-empty string.")
        targets, _catalog = _resolve_motif_export_targets(
            cfg=cfg,
            config_path=config_path,
            tfs=tf,
            refs=ref,
            set_index=set_index,
            source=source,
        )
        catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)
        base_dir = resolve_workspace_root(config_path)
        default_out = base_dir / cfg.out_dir / "artifacts" / "meme"
        out_dir = _resolve_user_path(out_dir or default_out, base_dir=base_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            table_rows, manifest_entries = write_meme_artifacts(
                cfg=cfg,
                catalog_root=catalog_root,
                targets=targets,
                out_dir=out_dir,
                overwrite=overwrite,
            )
        except FileExistsError as exc:
            console.print(f"[red]Artifact already exists:[/] {exc}")
            raise typer.Exit(code=1)

        manifest = {
            "schema_version": "1.0",
            "producer": producer_clean,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config_path": str(_portable_manifest_path(config_path, start=_manifest_anchor(config_path))),
            "catalog_root": str(_portable_manifest_path(catalog_root, start=_manifest_anchor(config_path))),
            "artifacts": manifest_entries,
        }
        manifest_path = out_dir / "meme_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        _print_meme_export_table(table_rows=table_rows, manifest_path=manifest_path)

    @app.command("export-sites", help="Export cached binding sites as CSV or Parquet.")
    def export_sites(
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
        tf: list[str] = typer.Option([], "--tf", help="TF name to include (repeatable)."),
        ref: list[str] = typer.Option([], "--ref", help="Catalog reference (<source>:<motif_id>, repeatable)."),
        set_index: int | None = typer.Option(
            None,
            "--set",
            help="Regulator set index from config (1-based).",
        ),
        source: str | None = typer.Option(
            None,
            "--source",
            help="Limit TF resolution to a single source adapter.",
        ),
        out_path: Path = typer.Option(
            ...,
            "--out",
            "-o",
            help="Output file path (.csv or .parquet).",
        ),
        fmt: str | None = typer.Option(
            None,
            "--format",
            help="Output format: csv | parquet (inferred from --out if omitted).",
        ),
        overwrite: bool = typer.Option(False, "--overwrite", help="Allow overwriting existing file."),
    ) -> None:
        try:
            config_path = resolve_config_path(config_option or config)
        except ConfigResolutionError as exc:
            console.print(str(exc))
            raise typer.Exit(code=1)
        cfg = load_config_or_exit(config_path)
        targets, _catalog = _resolve_site_export_targets(
            cfg=cfg,
            config_path=config_path,
            tfs=tf,
            refs=ref,
            set_index=set_index,
            source=source,
        )
        out_path = _resolve_user_path(out_path, base_dir=resolve_workspace_root(config_path))
        _prepare_site_export_output_path(out_path=out_path, overwrite=overwrite)

        fmt = _resolve_export_format(out_path, fmt, label="--format")
        catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)

        try:
            rows, table_rows = collect_site_export_rows(catalog_root=catalog_root, targets=targets)
        except FileNotFoundError as exc:
            console.print(f"[red]Missing sites file:[/] {exc}")
            raise typer.Exit(code=1)
        except ValueError as exc:
            console.print(f"[red]{exc}[/]")
            console.print("Hint: re-fetch with `cruncher fetch sites --hydrate` or provide genome FASTA.")
            raise typer.Exit(code=1)

        if not rows:
            console.print("[red]No binding sites found for selected targets.[/]")
            raise typer.Exit(code=1)

        try:
            row_count = write_site_export(rows, out_path=out_path, fmt=fmt)
        except Exception as exc:
            console.print(f"[red]Failed to write export:[/] {exc}")
            raise typer.Exit(code=1)

        _print_site_export_table(table_rows=table_rows, out_path=out_path, row_count=row_count)
