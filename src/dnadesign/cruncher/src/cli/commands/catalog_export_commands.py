"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/catalog_export_commands.py

Catalog export command registration for DenseGen artifacts and site tables.

Author(s): Eric J. South
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
    write_densegen_artifacts,
    write_site_export,
)
from dnadesign.cruncher.cli.catalog_targets import _resolve_site_targets, _resolve_targets
from dnadesign.cruncher.cli.catalog_utils import (
    _remove_existing_artifacts,
    _require_densegen_inputs_path,
    _resolve_densegen_workspace,
    _resolve_export_format,
    _resolve_user_path,
)
from dnadesign.cruncher.cli.commands.catalog_common import console, load_config_or_exit
from dnadesign.cruncher.cli.config_resolver import ConfigResolutionError, resolve_config_path
from dnadesign.cruncher.utils.paths import resolve_catalog_root, resolve_workspace_root


def _resolve_densegen_artifact_out_dir(
    *,
    config_path: Path,
    densegen_workspace: str | None,
    out_dir: Path | None,
) -> Path:
    base_dir = resolve_workspace_root(config_path)
    if densegen_workspace:
        densegen_root = _resolve_densegen_workspace(densegen_workspace, config_path=config_path)
        inputs_root = densegen_root / "inputs"
        if out_dir is None:
            return inputs_root / "motif_artifacts"
        return _require_densegen_inputs_path(
            out_dir,
            inputs_root=inputs_root,
            label="--out",
            base_dir=base_dir,
        )
    if out_dir is None:
        raise typer.BadParameter("--out is required when --densegen-workspace is not set.")
    return out_dir


def _resolve_densegen_site_out_path(
    *,
    config_path: Path,
    densegen_workspace: str | None,
    out_path: Path | None,
) -> Path:
    base_dir = resolve_workspace_root(config_path)
    if densegen_workspace:
        densegen_root = _resolve_densegen_workspace(densegen_workspace, config_path=config_path)
        inputs_root = densegen_root / "inputs"
        if out_path is None:
            return inputs_root / "densegen_sites.parquet"
        return _require_densegen_inputs_path(
            out_path,
            inputs_root=inputs_root,
            label="--out",
            base_dir=base_dir,
        )
    if out_path is None:
        raise typer.BadParameter("--out is required when --densegen-workspace is not set.")
    return out_path


def _resolve_densegen_export_targets(
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
        console.print("Hint: run cruncher fetch motifs/sites before catalog export-densegen.")
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


def _write_densegen_manifest_file(
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


def _print_densegen_export_table(*, table_rows: list[dict[str, str]], manifest_path: Path) -> None:
    table = Table(title="DenseGen motif artifacts", header_style="bold")
    table.add_column("TF")
    table.add_column("Source")
    table.add_column("Motif ID")
    table.add_column("Artifact")
    for row in table_rows:
        table.add_row(row["tf_name"], row["source"], row["motif_id"], row["artifact"])
    console.print(table)
    console.print(f"[green]Wrote manifest:[/] {manifest_path}")


def _print_site_export_table(*, table_rows: list[dict[str, str]], out_path: Path, row_count: int) -> None:
    table = Table(title="DenseGen binding-site export", header_style="bold")
    table.add_column("TF")
    table.add_column("Source")
    table.add_column("Motif ID")
    table.add_column("Sites")
    for row in table_rows:
        table.add_row(row["tf_name"], row["source"], row["motif_id"], row["count"])
    console.print(table)
    console.print(f"[green]Wrote binding sites:[/] {out_path} ({row_count} rows)")


def register_catalog_export_commands(app: typer.Typer) -> None:
    @app.command("export-densegen", help="Export DenseGen motif artifacts (one file per motif).")
    def export_densegen(
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
        densegen_workspace: str | None = typer.Option(
            None,
            "--densegen-workspace",
            help="DenseGen workspace name or path (defaults --out to inputs/motif_artifacts).",
        ),
        out_dir: Path | None = typer.Option(
            None,
            "--out",
            "-o",
            help="Directory to write DenseGen motif artifacts.",
        ),
        background: str = typer.Option(
            "record",
            "--background",
            help="Background policy: record | uniform | matrix.",
        ),
        pseudocount: float | None = typer.Option(
            None,
            "--pseudocount",
            help="Optional pseudocount for log-odds (>= 0).",
        ),
        producer: str = typer.Option(
            "cruncher",
            "--producer",
            help="Producer label for DenseGen artifacts.",
        ),
        clean: bool = typer.Option(
            True,
            "--clean/--no-clean",
            help="Remove existing motif artifacts for selected TFs before export.",
        ),
        overwrite: bool = typer.Option(False, "--overwrite", help="Allow overwriting existing files."),
    ) -> None:
        try:
            config_path = resolve_config_path(config_option or config)
        except ConfigResolutionError as exc:
            console.print(str(exc))
            raise typer.Exit(code=1)
        cfg = load_config_or_exit(config_path)
        if background not in {"record", "uniform", "matrix"}:
            raise typer.BadParameter("--background must be 'record', 'uniform', or 'matrix'.")
        producer_clean = producer.strip()
        if not producer_clean:
            raise typer.BadParameter("--producer must be a non-empty string.")
        targets, _catalog = _resolve_densegen_export_targets(
            cfg=cfg,
            config_path=config_path,
            tfs=tf,
            refs=ref,
            set_index=set_index,
            source=source,
        )
        out_dir = _resolve_densegen_artifact_out_dir(
            config_path=config_path,
            densegen_workspace=densegen_workspace,
            out_dir=out_dir,
        )

        catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)
        out_dir = _resolve_user_path(out_dir, base_dir=resolve_workspace_root(config_path))
        out_dir.mkdir(parents=True, exist_ok=True)

        if clean:
            removed = _remove_existing_artifacts(out_dir, tf_names=[target.tf_name for target in targets])
            if removed:
                console.print(f"[dim]Removed {removed} existing artifact(s) for selected TFs.[/]")

        try:
            table_rows, manifest_entries = write_densegen_artifacts(
                catalog_root=catalog_root,
                targets=targets,
                out_dir=out_dir,
                background=background,
                pseudocount=pseudocount,
                producer=producer_clean,
                overwrite=overwrite,
            )
        except FileNotFoundError as exc:
            console.print(f"[red]Missing motif file:[/] {exc}")
            raise typer.Exit(code=1)
        except FileExistsError as exc:
            console.print(f"[red]Artifact already exists:[/] {exc}")
            raise typer.Exit(code=1)

        manifest_path = _write_densegen_manifest_file(
            out_dir=out_dir,
            producer=producer_clean,
            manifest_entries=manifest_entries,
            config_path=config_path,
            catalog_root=catalog_root,
            background=background,
            pseudocount=pseudocount,
        )
        _print_densegen_export_table(table_rows=table_rows, manifest_path=manifest_path)

    @app.command("export-sites", help="Export cached binding sites for DenseGen (CSV/Parquet).")
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
        densegen_workspace: str | None = typer.Option(
            None,
            "--densegen-workspace",
            help="DenseGen workspace name or path (defaults --out to inputs/densegen_sites.parquet).",
        ),
        out_path: Path | None = typer.Option(
            None,
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
        out_path = _resolve_densegen_site_out_path(
            config_path=config_path,
            densegen_workspace=densegen_workspace,
            out_path=out_path,
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
