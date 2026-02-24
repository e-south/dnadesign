"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/catalog_logo_commands.py

Catalog logo rendering command registration.

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.table import Table

from dnadesign.cruncher.artifacts.layout import logos_root, out_root
from dnadesign.cruncher.cli.catalog_execution import render_logo_outputs
from dnadesign.cruncher.cli.catalog_targets import _resolve_targets
from dnadesign.cruncher.cli.catalog_utils import (
    _build_logo_signature,
    _clear_logo_outputs,
    _matching_logo_dir,
    _write_logo_manifest,
)
from dnadesign.cruncher.cli.commands.catalog_common import console, load_config_or_exit
from dnadesign.cruncher.cli.config_resolver import ConfigResolutionError, resolve_config_path
from dnadesign.cruncher.cli.paths import render_path
from dnadesign.cruncher.utils.paths import resolve_catalog_root, resolve_workspace_root
from dnadesign.cruncher.viz.mpl import ensure_mpl_cache


def register_catalog_logo_commands(app: typer.Typer) -> None:
    @app.command("logos", help="Render PWM logos for selected TFs or motif refs.")
    def logos(
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
            "--out-dir",
            help="Directory to write logo PNGs (defaults to <out_dir>/plots).",
        ),
        bits_mode: str | None = typer.Option(
            None,
            "--bits-mode",
            help="Logo mode: information or probability (default: information).",
        ),
        dpi: int | None = typer.Option(
            None,
            "--dpi",
            help="DPI for logo output (default: 150).",
        ),
    ) -> None:
        try:
            config_path = resolve_config_path(config_option or config)
        except ConfigResolutionError as exc:
            console.print(str(exc))
            raise typer.Exit(code=1)
        cfg = load_config_or_exit(config_path)
        workspace_root = resolve_workspace_root(config_path)
        try:
            catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)
            ensure_mpl_cache(catalog_root)
            targets, catalog = _resolve_targets(
                cfg=cfg,
                config_path=config_path,
                tfs=tf,
                refs=ref,
                set_index=set_index,
                source_filter=source,
            )
            resolved_bits_mode = bits_mode or "information"
            resolved_dpi = dpi or 150
            if resolved_bits_mode not in {"information", "probability"}:
                raise typer.BadParameter("--bits-mode must be 'information' or 'probability'.")
            signature, signature_payload = _build_logo_signature(
                cfg=cfg,
                catalog_root=catalog_root,
                targets=targets,
                bits_mode=resolved_bits_mode,
                dpi=resolved_dpi,
            )
            out_base = out_dir or logos_root(out_root(config_path, cfg.out_dir))
            existing = _matching_logo_dir(out_base, signature)
            if existing is not None:
                console.print(f"Logos already rendered at {render_path(existing, base=workspace_root)}")
                return
            out_base.mkdir(parents=True, exist_ok=True)
            _clear_logo_outputs(out_base)
            table_rows, outputs = render_logo_outputs(
                cfg=cfg,
                catalog_root=catalog_root,
                targets=targets,
                catalog=catalog,
                bits_mode=resolved_bits_mode,
                dpi=resolved_dpi,
                out_base=out_base,
            )
            table = Table(title="Rendered PWM logos", header_style="bold")
            table.add_column("TF")
            table.add_column("Source")
            table.add_column("Motif ID")
            table.add_column("Length")
            table.add_column("Bits")
            table.add_column("Output")
            for row in table_rows:
                table.add_row(
                    row["tf_name"],
                    row["source"],
                    row["motif_id"],
                    row["length"],
                    row["bits"],
                    render_path(row["output"], base=workspace_root),
                )
            console.print(table)
            console.print(f"Logos saved to {render_path(out_base, base=workspace_root)}")
            _write_logo_manifest(
                out_base,
                {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "signature": signature,
                    "render": signature_payload["render"],
                    "pwm": signature_payload["pwm"],
                    "targets": signature_payload["targets"],
                    "outputs": outputs,
                },
            )
        except (ValueError, FileNotFoundError) as exc:
            console.print(f"Error: {exc}")
            console.print("Hint: run cruncher fetch motifs/sites before catalog logos.")
            raise typer.Exit(code=1)
