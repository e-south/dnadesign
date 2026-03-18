"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cluster/src/cli/commands_umap.py

UMAP-related cluster CLI command registration.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from typing import List, Optional

import typer
from rich.console import Console

from .umap_resolution import resolve_umap_command


def register_umap_command(app: typer.Typer, *, console: Console) -> None:
    @app.command(
        "umap",
        help="Compute UMAP, save coords under the fit run, and optionally render plots or attach overlays.",
    )
    def cmd_umap(
        ctx: typer.Context,
        workspace: Optional[str] = typer.Option(None, help="Workspace directory or packaged workspace id."),
        results_root: Optional[str] = typer.Option(
            None,
            help="Standalone artifact root. Required unless --workspace is set.",
        ),
        dataset: Optional[str] = typer.Option(None),
        file: Optional[str] = typer.Option(None),
        usr_root: Optional[str] = typer.Option(None),
        name: Optional[str] = typer.Option(None, help="Existing fit alias to associate UMAP with (uses same rows)."),
        key_col: str = typer.Option("id"),
        x_col: Optional[str] = typer.Option(None),
        x_cols: Optional[str] = typer.Option(None),
        neighbors: Optional[int] = typer.Option(None, help="Falls back to preset or 15"),
        min_dist: Optional[float] = typer.Option(None, help="Falls back to preset or 0.10"),
        metric: Optional[str] = typer.Option(None, help='Falls back to preset or "euclidean"'),
        random_state: Optional[int] = typer.Option(None, help="Falls back to preset or 42"),
        preset: Optional[str] = typer.Option(None, help="Preset (kind: 'umap' and optional 'plot')"),
        color_by: List[str] = typer.Option(["cluster"], help="Hue specs (repeatable). Includes 'highlight'."),
        highlight: Optional[str] = typer.Option(None, help="CSV/Parquet with ids to highlight (first column or 'id')."),
        highlight_topn: Optional[int] = typer.Option(
            None,
            help=(
                "Highlight Top-N rows from the primary table by ranking a numeric column "
                "(use with --highlight-topn-col)."
            ),
        ),
        highlight_topn_col: Optional[str] = typer.Option(
            None,
            help="Numeric column to rank for --highlight-topn (e.g., 'permuter__metric__llr_mean').",
        ),
        highlight_topn_asc: bool = typer.Option(
            False,
            "--highlight-topn-asc",
            help="If set, select the smallest N values (ascending) instead of largest.",
        ),
        highlight_hue_col: Optional[str] = typer.Option(
            None,
            help="Optional. If set, color highlights categorically by this column from the --highlight file "
            "(e.g., 'observed_round'). Integers are treated as categories.",
        ),
        alpha: Optional[float] = typer.Option(None, help="Point alpha (overrides workspace config or preset)."),
        size: Optional[float] = typer.Option(None, help="Point size (overrides workspace config or preset)."),
        dims: Optional[str] = typer.Option(None, help="Figure size 'W,H' (overrides workspace config or preset)."),
        font_scale: Optional[float] = typer.Option(
            None,
            help="Scale all plot fonts (1.0 = default). Overrides preset.plot.font_scale if set.",
        ),
        plots: bool = typer.Option(
            True,
            "--plots/--no-plots",
            help="Render UMAP PNG plots under the run workspace. Disable for faster large-scale coord generation.",
        ),
        opal_campaign: Optional[str] = typer.Option(
            None,
            help="Path to OPAL campaign dir or campaign name under dnadesign/opal/campaigns/",
        ),
        opal_run: Optional[str] = typer.Option(
            None,
            help="OPAL run selector: 'latest', 'round:<n>', or 'run_id:<rid>' "
            "(mutually exclusive with --opal-as-of-round).",
        ),
        opal_as_of_round: Optional[int] = typer.Option(None, help="Filter OPAL predictions to this round"),
        opal_fields: Optional[str] = typer.Option(
            None,
            help="Comma-separated OPAL prediction fields to join (e.g., pred__y_obj_scalar,obj__logic_fidelity,obj__effect_scaled).",  # noqa: E501
        ),
        derive_ratio: List[str] = typer.Option(
            [],
            help="Repeatable. Define a derived ratio column: '<new_col>:<numerator_col>:<denominator_col>'.",
        ),
        attach_coords: bool = typer.Option(False),
        write: bool = typer.Option(False),
        yes: bool = typer.Option(
            False,
            "-y",
            "--allow-overwrite",
            help="Allow overwriting attached coord columns",
        ),
        inplace: bool = typer.Option(False),
        out: Optional[str] = typer.Option(None),
    ) -> None:
        from ..execution import run_umap

        resolved = resolve_umap_command(
            ctx=ctx,
            console=console,
            workspace=workspace,
            results_root=results_root,
            dataset=dataset,
            file=file,
            usr_root=usr_root,
            name=name,
            key_col=key_col,
            x_col=x_col,
            x_cols=x_cols,
            neighbors=neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=random_state,
            preset=preset,
            color_by=list(color_by),
            highlight=highlight,
            highlight_topn=highlight_topn,
            highlight_topn_col=highlight_topn_col,
            highlight_topn_asc=highlight_topn_asc,
            highlight_hue_col=highlight_hue_col,
            alpha=alpha,
            size=size,
            dims=dims,
            font_scale=font_scale,
            plots=plots,
            opal_campaign=opal_campaign,
            opal_run=opal_run,
            opal_as_of_round=opal_as_of_round,
            opal_fields=opal_fields,
            derive_ratio=list(derive_ratio),
            attach_coords=attach_coords,
            write=write,
            allow_overwrite=yes,
            inplace=inplace,
            out=out,
        )
        run_umap(**resolved.run_kwargs)


__all__ = ["register_umap_command"]
