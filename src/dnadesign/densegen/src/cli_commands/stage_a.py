"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/cli_commands/stage_a.py

Stage-A CLI command registration for building TFBS pools.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import typer

from ..cli_commands.context import CliContext
from ..cli_render import stage_a_plan_table, stage_a_recap_tables
from ..cli_sampling import stage_a_plan_rows
from ..core.artifacts.candidates import build_candidate_artifact, find_candidate_files, prepare_candidates_dir
from ..core.artifacts.pool import build_pool_artifact
from ..core.motif_labels import input_motifs
from ..core.pipeline import default_deps
from ..core.run_paths import candidates_root
from ..core.seeding import derive_seed_map
from ..utils import logging_utils
from ..utils.logging_utils import setup_logging


def register_stage_a_commands(
    app: typer.Typer,
    *,
    context: CliContext,
    apply_stage_a_overrides: Callable[..., None],
    ensure_fimo_available: Callable[..., None],
    candidate_logging_enabled: Callable[..., bool],
    stage_a_sampling_rows: Callable[..., list[dict[str, object]]],
) -> None:
    console = context.console
    make_table = context.make_table

    @app.command("build-pool", help="Build Stage-A TFBS pools from inputs.")
    def stage_a_build_pool(
        ctx: typer.Context,
        out: str = typer.Option(
            "outputs/pools",
            "--out",
            help="Output directory (relative to run root; must be inside outputs/).",
        ),
        n_sites: Optional[int] = typer.Option(
            None,
            "--n-sites",
            help="Override Stage-A PWM sampling n_sites for all PWM inputs.",
        ),
        batch_size: Optional[int] = typer.Option(
            None,
            "--batch-size",
            help="Override Stage-A PWM mining batch_size for all PWM inputs.",
        ),
        max_seconds: Optional[float] = typer.Option(
            None,
            "--max-seconds",
            help="Override Stage-A PWM mining budget max_seconds for all PWM inputs.",
        ),
        input_name: Optional[list[str]] = typer.Option(
            None,
            "--input",
            "-i",
            help="Input name(s) to build (defaults to all inputs).",
        ),
        fresh: bool = typer.Option(
            False,
            "--fresh",
            help="Start from scratch and replace existing pool files.",
        ),
        show_motif_ids: bool = typer.Option(
            False,
            "--show-motif-ids",
            help="Show full motif IDs instead of TF names.",
        ),
        verbose: bool = typer.Option(
            False,
            "--verbose",
            help="Show verbose Stage-A recap columns.",
        ),
        config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
    ):
        cfg_path, is_default = context.resolve_config_path(ctx, config)
        loaded = context.load_config_or_exit(
            cfg_path,
            missing_message=context.default_config_missing_message if is_default else None,
        )
        cfg = loaded.root.densegen
        ensure_fimo_available(cfg, strict=True)
        run_root = context.run_root_for(loaded)
        log_cfg = cfg.logging
        log_dir = context.resolve_outputs_path_or_exit(
            loaded.path,
            run_root,
            Path(log_cfg.log_dir),
            label="logging.log_dir",
        )
        logfile = log_dir / f"{cfg.run.id}.stage_a.log"
        setup_logging(
            level=log_cfg.level,
            logfile=str(logfile),
            suppress_solver_stderr=bool(log_cfg.suppress_solver_stderr),
        )
        progress_style = str(log_cfg.progress_style)
        logging_utils.set_progress_style(progress_style)
        logging_utils.set_progress_enabled(progress_style in {"stream", "screen"})
        out_dir = context.resolve_outputs_path_or_exit(cfg_path, run_root, out, label="stage-a.out")
        out_dir.mkdir(parents=True, exist_ok=True)

        selected = {name for name in (input_name or [])}
        if selected:
            available = {inp.name for inp in cfg.inputs}
            missing = sorted(selected - available)
            if missing:
                raise typer.BadParameter(f"Unknown input name(s): {', '.join(missing)}")
        if n_sites is not None and n_sites <= 0:
            raise typer.BadParameter("--n-sites must be > 0")
        if batch_size is not None and batch_size <= 0:
            raise typer.BadParameter("--batch-size must be > 0")
        if max_seconds is not None and max_seconds <= 0:
            raise typer.BadParameter("--max-seconds must be > 0")
        apply_stage_a_overrides(
            cfg,
            selected=selected if selected else None,
            n_sites=n_sites,
            batch_size=batch_size,
            max_seconds=max_seconds,
        )

        seeds = derive_seed_map(int(cfg.runtime.random_seed), ["stage_a", "stage_b", "solver"])
        rng = np.random.default_rng(seeds["stage_a"])
        deps = default_deps()
        outputs_root = run_root / "outputs"
        outputs_root.mkdir(parents=True, exist_ok=True)
        candidate_logging = candidate_logging_enabled(cfg, selected=set(selected) if selected else None)
        candidates_dir = candidates_root(outputs_root, cfg.run.id)
        if candidate_logging:
            try:
                existed = prepare_candidates_dir(candidates_dir, overwrite=fresh)
            except Exception as exc:
                console.print(f"[bold red]{exc}[/]")
                raise typer.Exit(code=1)
            if existed and fresh:
                console.print(
                    f"[yellow]Cleared prior candidate artifacts at "
                    f"{context.display_path(candidates_dir, run_root, absolute=False)} to avoid mixing runs.[/]"
                )
            elif existed and not fresh:
                console.print(
                    "[yellow]Appending to existing candidate artifacts under "
                    f"{candidates_dir} (use --fresh to reset).[/]"
                )

        plan_rows = stage_a_plan_rows(
            cfg,
            cfg_path,
            selected if selected else None,
            show_motif_ids=show_motif_ids,
        )
        if plan_rows:
            console.print("[bold]Stage-A plan[/]")
            console.print(stage_a_plan_table(plan_rows))

        with context.suppress_pyarrow_sysctl_warnings():
            try:
                artifact, pool_data = build_pool_artifact(
                    cfg=cfg,
                    cfg_path=cfg_path,
                    deps=deps,
                    rng=rng,
                    outputs_root=outputs_root,
                    out_dir=out_dir,
                    overwrite=fresh,
                    selected_inputs=selected if selected else None,
                )
            except FileExistsError as exc:
                console.print(f"[bold red]{exc}[/]")
                raise typer.Exit(code=1)
            except Exception as exc:
                console.print(f"[bold red]Failed to build Stage-A pools:[/] {exc}")
                raise typer.Exit(code=1)
            if candidate_logging:
                candidate_files = find_candidate_files(candidates_dir)
                if candidate_files:
                    try:
                        build_candidate_artifact(
                            candidates_dir=candidates_dir,
                            cfg_path=cfg_path,
                            run_id=str(cfg.run.id),
                            run_root=run_root,
                            overwrite=True,
                        )
                    except Exception as exc:
                        console.print(f"[bold red]Failed to write candidate artifacts:[/] {exc}")
                        raise typer.Exit(code=1)
                else:
                    console.print(
                        f"[yellow]Candidate logging enabled but no candidate records found under {candidates_dir}.[/]"
                    )

        display_map_by_input: dict[str, dict[str, str]] = {}
        for inp in cfg.inputs:
            motifs = input_motifs(inp, cfg_path)
            display_map_by_input[inp.name] = {motif_id: name for motif_id, name in motifs}

        recap_rows = stage_a_sampling_rows(pool_data)
        if recap_rows:
            console.print("[bold]Stage-A sampling recap[/]")
            for title, table in stage_a_recap_tables(
                recap_rows,
                display_map_by_input=display_map_by_input,
                show_motif_ids=show_motif_ids,
                verbose=verbose,
            ):
                if title:
                    console.print(f"[bold]{title}[/]")
                console.print(table)
            legend_rows = [
                ("generated", "PWM candidates"),
                ("eligible_unique", "deduped by uniqueness.key"),
                ("retained", "carried forward after selection"),
                ("tier fill", "deepest diagnostic tier used"),
                ("selection", "Stage-A selection policy"),
                ("pool", "MMR pool size after rung slice ('*' = capped)"),
                ("overlap", "top ∩ diversified"),
                ("pairwise top/div", "weighted Hamming median (tfbs_core)"),
                ("score_norm top/div", "min/med/max"),
            ]
            if verbose:
                legend_rows.insert(1, ("has_hit", "FIMO hit present"))
                legend_rows.insert(2, ("eligible_raw", "best_hit_score > 0 among hits"))
                legend_rows.append(("tier target", "diagnostic tier target status"))
                legend_rows.append(("set_swaps", "diversified - overlap"))
            legend_table = make_table("Legend", "Meaning")
            for key, desc in legend_rows:
                legend_table.add_row(str(key), str(desc))
            console.print("[bold]Legend[/]")
            console.print(legend_table)
        console.print(
            f":sparkles: [bold green]Pool manifest written[/]: "
            f"{context.display_path(artifact.manifest_path, run_root, absolute=False)}"
        )
