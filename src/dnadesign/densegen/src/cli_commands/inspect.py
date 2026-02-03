"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/cli_commands/inspect.py

Inspect subcommands for the DenseGen CLI.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from ..config import load_config, resolve_outputs_scoped_path, resolve_relative_path, resolve_run_root
from ..core.artifacts.pool import pool_status_by_input
from ..core.event_log import load_events
from ..core.motif_labels import input_motifs
from ..core.pipeline import resolve_plan
from ..core.reporting import collect_report_data
from ..core.run_manifest import load_run_manifest
from ..core.run_paths import run_manifest_path, run_state_path
from ..core.run_state import load_run_state
from .context import CliContext


def _input_kind_label(input_type: str) -> str:
    if str(input_type).startswith("pwm_"):
        return "pwm"
    if str(input_type) == "sequence_pool":
        return "sequence"
    if str(input_type) == "background_pool":
        return "background"
    return str(input_type)


def _unique_preserve(values: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in values:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _print_inputs_summary(
    loaded,
    *,
    verbose: bool,
    absolute: bool,
    show_motif_ids: bool,
    context: CliContext,
) -> None:
    cfg = loaded.root.densegen
    run_root = context.run_root_for(loaded)
    statuses = pool_status_by_input(cfg, loaded.path, run_root)

    table = context.make_table("input", "kind", "motifs", "source", "stage-a pool")
    for inp in cfg.inputs:
        input_type = str(inp.type)
        kind = _input_kind_label(input_type)
        motifs = input_motifs(inp, loaded.path)
        if show_motif_ids:
            motif_labels = _unique_preserve([m_id for m_id, _name in motifs if m_id])
        else:
            motif_labels = _unique_preserve([_name for _m_id, _name in motifs if _name])
        motif_summary = "-"
        if motif_labels:
            if verbose or len(motif_labels) <= 6:
                motif_summary = f"{len(motif_labels)} ({','.join(motif_labels)})"
            else:
                motif_summary = f"{len(motif_labels)} motifs"

        source_label = "-"
        if hasattr(inp, "path"):
            resolved = resolve_relative_path(loaded.path, getattr(inp, "path"))
            source_label = context.display_path(resolved, run_root, absolute=absolute)
        elif hasattr(inp, "paths"):
            resolved = [resolve_relative_path(loaded.path, p) for p in getattr(inp, "paths") or []]
            parents = {p.parent for p in resolved} if resolved else set()
            if parents:
                root = parents.pop() if len(parents) == 1 else None
                if root is not None:
                    prefix = context.display_path(root, run_root, absolute=absolute)
                    if verbose:
                        names = ", ".join(sorted(p.name for p in resolved))
                        source_label = f"{prefix} ({len(resolved)} files): {names}"
                    else:
                        source_label = f"{prefix} ({len(resolved)} files)"
                else:
                    source_label = f"{len(resolved)} files (multiple dirs)"
            else:
                source_label = "0 files"
        elif hasattr(inp, "dataset"):
            root_path = resolve_relative_path(loaded.path, getattr(inp, "root"))
            root_label = context.display_path(root_path, run_root, absolute=absolute)
            source_label = f"{inp.dataset} (root={root_label})"
        status = statuses.get(inp.name)
        if status is None:
            status_label = "-"
        else:
            if status.state == "present":
                status_label = "present ✓"
            elif status.state == "stale":
                status_label = "stale !"
            else:
                status_label = "missing"
        table.add_row(str(inp.name), kind, motif_summary, source_label, status_label)
    context.console.print("[bold]Stage-A input sources[/]")
    context.console.print(table)
    context.console.print("Legend: motifs = TF display names; source is workspace-relative.")
    context.console.print("Legend: stage-a pool reflects whether the pool matches the current config.")
    context.console.print("Tip: run `dense stage-a build-pool --fresh` to rebuild pools.")


def register_inspect_commands(inspect_app: typer.Typer, *, context: CliContext) -> None:
    @inspect_app.command("run", help="Summarize a run manifest or list workspaces.")
    def inspect_run(
        ctx: typer.Context,
        run: Optional[Path] = typer.Option(None, "--run", "-r", help="Run directory (defaults to config run root)."),
        root: Optional[Path] = typer.Option(None, "--root", help="Workspaces root directory (lists workspaces)."),
        limit: int = typer.Option(0, "--limit", help="Limit workspaces displayed when using --root (0 = all)."),
        show_all: bool = typer.Option(
            False, "--all", help="Include directories without config.yaml when using --root."
        ),
        config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
        absolute: bool = typer.Option(False, "--absolute", help="Show absolute paths instead of workspace-relative."),
        verbose: bool = typer.Option(False, "--verbose", "-v", help="Show failure breakdown columns."),
        library: bool = typer.Option(False, "--library", help="Include offered-vs-used library summaries."),
        show_tfbs: bool = typer.Option(False, "--show-tfbs", help="Show TFBS sequences in library summaries."),
        show_motif_ids: bool = typer.Option(
            False,
            "--show-motif-ids",
            help="Show full motif IDs instead of TF display names in summaries.",
        ),
        events: bool = typer.Option(False, "--events", help="Show events summary (stalls/resamples)."),
    ):
        if root is not None and run is not None:
            context.console.print("[bold red]Choose either --root or --run, not both.[/]")
            raise typer.Exit(code=1)
        if root is not None:
            workspaces_root = root.resolve()
            if not workspaces_root.exists() or not workspaces_root.is_dir():
                context.console.print(
                    f"[bold red]Workspaces root not found:[/] "
                    f"{context.display_path(workspaces_root, Path.cwd(), absolute=absolute)}"
                )
                raise typer.Exit(code=1)
            context.console.print(_list_workspaces_table(context, workspaces_root, limit, show_all, absolute))
            return
        cfg_path = None
        loaded = None
        if run is None:
            cfg_path, is_default = context.resolve_config_path(ctx, config)
            loaded = context.load_config_or_exit(
                cfg_path,
                missing_message=context.default_config_missing_message if is_default else None,
                absolute=absolute,
                display_root=Path.cwd(),
            )
            run_root = context.run_root_for(loaded)
        else:
            run_root = run
            if library:
                cfg_path = run_root / "config.yaml"
                if not cfg_path.exists():
                    context.console.print(
                        f"[bold red]Config not found for --library:[/] "
                        f"{context.display_path(cfg_path, run_root, absolute=absolute)}. "
                        "Provide --config or run inspect run without --library."
                    )
                    raise typer.Exit(code=1)
                loaded = context.load_config_or_exit(cfg_path, absolute=absolute, display_root=run_root)
        manifest_path = run_manifest_path(run_root)
        if not manifest_path.exists():
            state_path = run_state_path(run_root)
            if state_path.exists():
                state = load_run_state(state_path)
                context.console.print("[yellow]Run manifest missing; showing checkpointed run_state.[/]")
                root_label = context.display_path(run_root, run_root, absolute=absolute)
                context.console.print(
                    f"[bold]Run:[/] {state.run_id}  [bold]Root:[/] {root_label}  "
                    f"[bold]Schema:[/] {state.schema_version}  [bold]Config:[/] {state.config_sha256[:8]}…"
                )
                table = context.make_table("input", "plan", "generated")
                for item in state.items:
                    table.add_row(item.input_name, item.plan_name, str(item.generated))
                context.console.print(table)
                context.console.print("[bold]Next steps[/]:")
                context.console.print(context.workspace_command("dense run", cfg_path=cfg_path, run_root=run_root))
                return

            missing_manifest = context.display_path(manifest_path, run_root, absolute=absolute)
            context.console.print(f"[bold red]Run manifest not found:[/] {missing_manifest}")
            entries = context.list_dir_entries(run_root, limit=8)
            if entries:
                context.console.print(f"[bold]Run root contents[/]: {', '.join(entries)}")
            context.console.print("[bold]Next steps[/]:")
            context.console.print(context.workspace_command("dense run", cfg_path=cfg_path, run_root=run_root))
            raise typer.Exit(code=1)

        manifest = load_run_manifest(manifest_path)
        schema_label = manifest.schema_version or "-"
        dense_arrays_label = manifest.dense_arrays_version or "-"
        dense_arrays_source = manifest.dense_arrays_version_source or "-"
        if dense_arrays_label != "-" and dense_arrays_source != "-":
            dense_arrays_label = f"{dense_arrays_label} ({dense_arrays_source})"
        root_label = context.display_path(run_root, run_root, absolute=absolute)
        context.console.print(
            f"[bold]Run:[/] {manifest.run_id}  [bold]Root:[/] {root_label}  "
            f"[bold]Schema:[/] {schema_label}  [bold]dense-arrays:[/] {dense_arrays_label}"
        )
        if verbose:
            table = context.make_table(
                "input",
                "plan",
                "generated",
                "dup_out",
                "dup_sol",
                "failed",
                "fail_tf",
                "fail_req",
                "fail_min",
                "fail_k",
                "resamples",
                "libraries",
                "stalls",
            )
        else:
            table = context.make_table(
                "input", "plan", "generated", "duplicates", "failed", "resamples", "libraries", "stalls"
            )
        for item in manifest.items:
            if verbose:
                table.add_row(
                    item.input_name,
                    item.plan_name,
                    str(item.generated),
                    str(item.duplicates_skipped),
                    str(item.duplicate_solutions),
                    str(item.failed_solutions),
                    str(item.failed_min_count_per_tf),
                    str(item.failed_required_regulators),
                    str(item.failed_min_count_by_regulator),
                    str(item.failed_min_required_regulators),
                    str(item.total_resamples),
                    str(item.libraries_built),
                    str(item.stall_events),
                )
            else:
                table.add_row(
                    item.input_name,
                    item.plan_name,
                    str(item.generated),
                    str(item.duplicates_skipped),
                    str(item.failed_solutions),
                    str(item.total_resamples),
                    str(item.libraries_built),
                    str(item.stall_events),
                )
        context.console.print(table)

        if events:
            events_path = run_root / "outputs" / "meta" / "events.jsonl"
            try:
                events_df = load_events(events_path, allow_missing=True)
            except Exception as exc:
                context.console.print(f"[bold red]Failed to read events log:[/] {exc}")
                raise typer.Exit(code=1)
            if events_df.empty or "event" not in events_df.columns:
                context.console.print("[yellow]No events found.[/]")
            else:
                event_summary = (
                    events_df.groupby("event")
                    .agg(count=("event", "size"), last_created_at=("created_at", "max"))
                    .reset_index()
                    .sort_values("count", ascending=False)
                )
                events_table = context.make_table("event", "count", "last_created_at")
                for _, row in event_summary.iterrows():
                    events_table.add_row(
                        str(row.get("event") or "-"),
                        str(int(row.get("count") or 0)),
                        str(row.get("last_created_at") or "-"),
                    )
                context.console.print(events_table)

        if library:
            if loaded is None or cfg_path is None:
                context.console.print("[bold red]Config is required for --library summaries.[/]")
                raise typer.Exit(code=1)
            with context.suppress_pyarrow_sysctl_warnings():
                try:
                    bundle = collect_report_data(loaded.root, cfg_path, include_combinatorics=False)
                except Exception as exc:
                    context.console.print(f"[bold red]Failed to build library summaries:[/] {exc}")
                    entries = context.list_dir_entries(run_root, limit=8)
                    if entries:
                        context.console.print(f"[bold]Run root contents[/]: {', '.join(entries)}")
                    context.console.print("[bold]Next steps[/]:")
                    context.console.print(context.workspace_command("dense run", cfg_path=cfg_path, run_root=run_root))
                    raise typer.Exit(code=1)

            offered_vs_used_tf = bundle.tables.get("offered_vs_used_tf")
            offered_vs_used_tfbs = bundle.tables.get("offered_vs_used_tfbs")
            if not isinstance(offered_vs_used_tf, pd.DataFrame) or offered_vs_used_tf.empty:
                if not isinstance(offered_vs_used_tfbs, pd.DataFrame) or offered_vs_used_tfbs.empty:
                    context.console.print("[yellow]No library usage summaries found (attempts missing).[/]")
                    return

            display_map: dict[str, str] = {}
            for inp in loaded.root.densegen.inputs:
                for motif_id, name in input_motifs(inp, cfg_path):
                    if motif_id and name and motif_id not in display_map:
                        display_map[motif_id] = name

            def _display_tf_label(label: str) -> str:
                if show_motif_ids:
                    return label
                return display_map.get(label, label)

            if isinstance(offered_vs_used_tf, pd.DataFrame) and not offered_vs_used_tf.empty:
                agg_tf = (
                    offered_vs_used_tf.groupby("tf")
                    .agg(
                        offered_instances=("offered_instances", "sum"),
                        offered_unique_tfbs=("offered_unique_tfbs", "sum"),
                        used_placements=("used_placements", "sum"),
                        used_unique_tfbs=("used_unique_tfbs", "sum"),
                        used_sequences=("used_sequences", "sum"),
                        total_sequences=("total_sequences", "sum"),
                    )
                    .reset_index()
                )
                agg_tf["utilization_any"] = agg_tf.apply(
                    lambda r: (r["used_sequences"] / r["total_sequences"]) if r["total_sequences"] else 0.0, axis=1
                )
                agg_tf["utilization_placements_per_offered"] = agg_tf.apply(
                    lambda r: (r["used_placements"] / r["offered_instances"]) if r["offered_instances"] else 0.0,
                    axis=1,
                )
                agg_tf = agg_tf.sort_values(["used_placements", "offered_instances"], ascending=False)
                tf_table = context.make_table("tf", "offered", "used", "util_any", "util_per_offered")
                for _, row in agg_tf.iterrows():
                    tf_table.add_row(
                        _display_tf_label(str(row.get("tf") or "-")),
                        str(int(row.get("offered_instances", 0))),
                        str(int(row.get("used_placements", 0))),
                        f"{float(row.get('utilization_any', 0.0)):.2f}",
                        f"{float(row.get('utilization_placements_per_offered', 0.0)):.2f}",
                    )
                context.console.print("[bold]TF usage summary (all libraries)[/]")
                context.console.print(tf_table)
                context.console.print(
                    "Legend: offered=instances across all libraries; util_any=used_sequences/total_sequences."
                )

            if isinstance(offered_vs_used_tfbs, pd.DataFrame) and not offered_vs_used_tfbs.empty:
                agg_tfbs = (
                    offered_vs_used_tfbs.groupby(["tf", "tfbs"])
                    .agg(
                        offered_instances=("offered_instances", "sum"),
                        used_placements=("used_placements", "sum"),
                        used_sequences=("used_sequences", "sum"),
                    )
                    .reset_index()
                )
                agg_tfbs = agg_tfbs.sort_values(["used_placements", "offered_instances"], ascending=False)
                if show_tfbs:
                    tfbs_table = context.make_table("tf", "tfbs", "offered", "used")
                    for _, row in agg_tfbs.iterrows():
                        tfbs_table.add_row(
                            _display_tf_label(str(row.get("tf") or "-")),
                            str(row.get("tfbs") or "-"),
                            str(int(row.get("offered_instances", 0))),
                            str(int(row.get("used_placements", 0))),
                        )
                    context.console.print("[bold]TFBS usage summary (all libraries)[/]")
                    context.console.print(tfbs_table)
                else:
                    unique_offered = int(len(agg_tfbs))
                    unique_used = int((agg_tfbs["used_placements"] > 0).sum())
                    usage_rate = float(unique_used) / float(unique_offered) if unique_offered else 0.0
                    used_vals = pd.to_numeric(
                        agg_tfbs.loc[agg_tfbs["used_placements"] > 0, "used_placements"], errors="coerce"
                    ).dropna()
                    if used_vals.empty:
                        min_used = med_used = max_used = "-"
                    else:
                        min_used = f"{float(used_vals.min()):.0f}"
                        med_used = f"{float(used_vals.median()):.0f}"
                        max_used = f"{float(used_vals.max()):.0f}"
                    summary_table = context.make_table(
                        "unique_offered",
                        "unique_used",
                        "usage_rate",
                        "used_min",
                        "used_median",
                        "used_max",
                    )
                    summary_table.add_row(
                        str(unique_offered),
                        str(unique_used),
                        f"{usage_rate:.2f}",
                        str(min_used),
                        str(med_used),
                        str(max_used),
                    )
                    context.console.print("[bold]TFBS usage summary (all libraries)[/]")
                    context.console.print(summary_table)

    @inspect_app.command("plan", help="Show the resolved per-constraint quota plan.")
    def inspect_plan(
        ctx: typer.Context,
        config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
    ):
        cfg_path, is_default = context.resolve_config_path(ctx, config)
        loaded = context.load_config_or_exit(
            cfg_path,
            missing_message=context.default_config_missing_message if is_default else None,
        )
        context.warn_full_pool_strategy(loaded)
        pl = resolve_plan(loaded)
        table = context.make_table("name", "quota", "has promoter_constraints")
        for item in pl:
            pcs = item.fixed_elements.promoter_constraints
            table.add_row(item.name, str(item.quota), "yes" if pcs else "no")
        context.console.print(table)

    @inspect_app.command("config", help="Describe resolved config, outputs, and pipeline settings.")
    def inspect_config(
        ctx: typer.Context,
        show_constraints: bool = typer.Option(False, help="Print full fixed elements per plan item."),
        probe_solver: bool = typer.Option(False, help="Probe the solver backend before reporting."),
        absolute: bool = typer.Option(False, "--absolute", help="Show absolute paths instead of workspace-relative."),
        config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
    ):
        cfg_path, is_default = context.resolve_config_path(ctx, config)
        loaded = context.load_config_or_exit(
            cfg_path,
            missing_message=context.default_config_missing_message if is_default else None,
        )
        root = loaded.root
        cfg = root.densegen
        run_root = context.run_root_for(loaded)

        if probe_solver:
            from ..adapters.optimizer import DenseArraysAdapter
            from ..core.pipeline import select_solver

            select_solver(
                cfg.solver.backend,
                DenseArraysAdapter(),
                strategy=str(cfg.solver.strategy),
            )

        context.console.print(f"[bold]Config[/]: {context.display_path(loaded.path, run_root, absolute=absolute)}")
        context.console.print(
            f"[bold]Run[/]: id={cfg.run.id} root={context.display_path(run_root, run_root, absolute=absolute)}"
        )
        effective_path = run_root / "outputs" / "meta" / "effective_config.json"
        if effective_path.exists():
            context.console.print(
                f"[bold]Effective config[/]: {context.display_path(effective_path, run_root, absolute=absolute)}"
            )
        context.console.print("See `dense inspect inputs` for resolved input sources.")

        plan_table = context.make_table(
            "name",
            "quota/fraction",
            "promoter_constraints",
            "side_biases",
            "regulator_groups",
            "group_min_required",
            "min_count_by_regulator",
        )
        for item in cfg.generation.resolve_plan():
            pcs = item.fixed_elements.promoter_constraints
            left = item.fixed_elements.side_biases.left if item.fixed_elements.side_biases else []
            right = item.fixed_elements.side_biases.right if item.fixed_elements.side_biases else []
            bias_count = len(set(left)) + len(set(right))
            groups = list(item.regulator_constraints.groups or [])
            group_count = len(groups)
            group_min_total = sum(int(group.min_required) for group in groups)
            min_count_regs = len(item.regulator_constraints.min_count_by_regulator or {})
            quota = str(item.quota) if item.quota is not None else f"{item.fraction:.3f}"
            plan_table.add_row(
                item.name,
                quota,
                str(len(pcs)),
                str(bias_count),
                str(group_count),
                str(group_min_total),
                str(min_count_regs),
            )
        context.console.print(plan_table)

        if show_constraints:
            for item in cfg.generation.resolve_plan():
                context.console.print(f"[bold]{item.name}[/]")
                pcs = item.fixed_elements.promoter_constraints
                if pcs:
                    for pc in pcs:
                        context.console.print(
                            f"  promoter: upstream={pc.upstream} downstream={pc.downstream} "
                            f"spacer_length={pc.spacer_length} upstream_pos={pc.upstream_pos} "
                            f"downstream_pos={pc.downstream_pos}"
                        )
                sb = item.fixed_elements.side_biases
                if sb:
                    if sb.left:
                        context.console.print(f"  side_biases.left: {', '.join(sb.left)}")
                    if sb.right:
                        context.console.print(f"  side_biases.right: {', '.join(sb.right)}")

        solver = context.make_table("backend", "strategy", "time_limit", "threads", "strands")
        time_limit = "-" if cfg.solver.time_limit is None else str(cfg.solver.time_limit)
        threads = "-" if cfg.solver.threads is None else str(cfg.solver.threads)
        backend_display = str(cfg.solver.backend)
        solver.add_row(backend_display, str(cfg.solver.strategy), time_limit, threads, str(cfg.solver.strands))
        context.console.print(solver)

        sampling = cfg.generation.sampling
        sampling_table = context.make_table("setting", "value")
        sampling_table.add_row("pool_strategy", str(sampling.pool_strategy))
        sampling_table.add_row("library_source", str(sampling.library_source))
        if sampling.library_source == "artifact":
            sampling_table.add_row("library_artifact_path", str(sampling.library_artifact_path))
        sampling_table.add_row("library_size", str(sampling.library_size))
        sampling_table.add_row("library_sampling_strategy", str(sampling.library_sampling_strategy))
        sampling_table.add_row("cover_all_regulators", str(sampling.cover_all_regulators))
        sampling_table.add_row("unique_binding_sites", str(sampling.unique_binding_sites))
        sampling_table.add_row("unique_binding_cores", str(sampling.unique_binding_cores))
        sampling_table.add_row("max_sites_per_regulator", str(sampling.max_sites_per_regulator))
        sampling_table.add_row("relax_on_exhaustion", str(sampling.relax_on_exhaustion))
        if sampling.pool_strategy == "iterative_subsample":
            sampling_table.add_row("iterative_max_libraries", str(sampling.iterative_max_libraries))
            sampling_table.add_row("iterative_min_new_solutions", str(sampling.iterative_min_new_solutions))
        sampling_table.add_row("arrays_generated_before_resample", str(cfg.runtime.arrays_generated_before_resample))
        sampling_table.add_row("max_consecutive_failures", str(cfg.runtime.max_consecutive_failures))
        context.console.print("[bold]Stage-B library sampling[/]")
        context.console.print(sampling_table)

        pad = cfg.postprocess.pad
        pad_gc = pad.gc
        if pad_gc.mode == "off":
            gc_label = "off"
        elif pad_gc.mode == "range":
            gc_label = f"range[{pad_gc.min:.2f}, {pad_gc.max:.2f}] min_pad_length={pad_gc.min_pad_length}"
        else:
            target_min = pad_gc.target - pad_gc.tolerance
            target_max = pad_gc.target + pad_gc.tolerance
            gc_label = (
                f"target={pad_gc.target:.2f}±{pad_gc.tolerance:.2f} "
                f"range=[{target_min:.2f}, {target_max:.2f}] "
                f"min_pad_length={pad_gc.min_pad_length}"
            )
        context.console.print(f"[bold]Pad[/]: mode={pad.mode} end={pad.end} gc={gc_label} max_tries={pad.max_tries}")
        log_dir = resolve_outputs_scoped_path(loaded.path, run_root, cfg.logging.log_dir, label="logging.log_dir")
        log_dir_label = context.display_path(log_dir, run_root, absolute=absolute)
        log_table = context.make_table("setting", "value")
        log_table.add_row("dir", log_dir_label)
        log_table.add_row("level", str(cfg.logging.level))
        log_table.add_row("progress_style", str(cfg.logging.progress_style))
        log_table.add_row("progress_every", str(cfg.logging.progress_every))
        log_table.add_row("progress_refresh_seconds", str(cfg.logging.progress_refresh_seconds))
        log_table.add_row("print_visual", str(cfg.logging.print_visual))
        log_table.add_row("show_tfbs", str(cfg.logging.show_tfbs))
        log_table.add_row("show_solutions", str(cfg.logging.show_solutions))
        log_table.add_row("suppress_solver_stderr", str(cfg.logging.suppress_solver_stderr))
        context.console.print("[bold]Logging[/]")
        context.console.print(log_table)

        if root.plots:
            out_dir = resolve_outputs_scoped_path(loaded.path, run_root, root.plots.out_dir, label="plots.out_dir")
            out_dir_label = context.display_path(out_dir, run_root, absolute=absolute)
            context.console.print(f"[bold]Plots[/]: source={root.plots.source} out_dir={out_dir_label}")
        else:
            context.console.print("[bold]Plots[/]: none")

    @inspect_app.command("inputs", help="Show resolved inputs and Stage-A pool status.")
    def inspect_inputs(
        ctx: typer.Context,
        verbose: bool = typer.Option(False, "--verbose", help="Show full source file lists."),
        absolute: bool = typer.Option(False, "--absolute", help="Show absolute paths instead of workspace-relative."),
        show_motif_ids: bool = typer.Option(False, "--show-motif-ids", help="Show full motif IDs instead of TF names."),
        config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
    ):
        cfg_path, is_default = context.resolve_config_path(ctx, config)
        loaded = context.load_config_or_exit(
            cfg_path,
            missing_message=context.default_config_missing_message if is_default else None,
        )
        run_root = context.run_root_for(loaded)
        cfg_label = context.display_path(loaded.path, run_root, absolute=absolute)
        run_root_label = context.display_path(run_root, run_root, absolute=absolute)
        context.console.print(f"[bold]Config[/]: {cfg_label}")
        context.console.print(f"[bold]Run[/]: id={loaded.root.densegen.run.id} root={run_root_label}")
        _print_inputs_summary(
            loaded,
            verbose=verbose,
            absolute=absolute,
            show_motif_ids=show_motif_ids,
            context=context,
        )


def _count_files(path: Path, pattern: str = "*") -> int:
    if not path.exists() or not path.is_dir():
        return 0
    return sum(1 for p in path.glob(pattern) if p.is_file())


def _list_workspaces_table(
    context: CliContext,
    workspaces_root: Path,
    limit: int,
    show_all: bool,
    absolute: bool,
):
    workspace_dirs = sorted([p for p in workspaces_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if limit and limit > 0:
        workspace_dirs = workspace_dirs[: int(limit)]

    table = context.make_table("workspace", "id", "config", "parquet", "plots", "logs", "status")
    for run_dir in workspace_dirs:
        cfg_path = run_dir / "config.yaml"
        if not show_all and not cfg_path.exists():
            continue

        run_id = run_dir.name
        status = "ok"
        parquet_count = "-"
        plots_count = _count_files(run_dir / "outputs" / "plots", pattern="*")
        logs_count = _count_files(run_dir / "outputs" / "logs", pattern="*")

        if cfg_path.exists():
            try:
                loaded = load_config(cfg_path)
                run_id = loaded.root.densegen.run.id
                run_root = resolve_run_root(cfg_path, loaded.root.densegen.run.root)
                if loaded.root.densegen.output.parquet is not None:
                    pq_dir = resolve_outputs_scoped_path(
                        cfg_path,
                        run_root,
                        loaded.root.densegen.output.parquet.path,
                        label="output.parquet.path",
                    )
                    parquet_count = _count_files(pq_dir, pattern="*.parquet")
                plots_count = _count_files(
                    resolve_outputs_scoped_path(
                        cfg_path,
                        run_root,
                        loaded.root.plots.out_dir if loaded.root.plots else "outputs/plots",
                        label="plots.out_dir",
                    ),
                    pattern="*",
                )
                logs_count = _count_files(
                    resolve_outputs_scoped_path(
                        cfg_path,
                        run_root,
                        loaded.root.densegen.logging.log_dir,
                        label="logging.log_dir",
                    ),
                    pattern="*",
                )
            except Exception:
                status = "invalid"
        else:
            status = "missing config.yaml"

        config_label = "-"
        if cfg_path.exists():
            config_label = context.display_path(cfg_path, workspaces_root, absolute=absolute)
        table.add_row(
            run_dir.name,
            run_id,
            config_label,
            str(parquet_count),
            str(plots_count),
            str(logs_count),
            status,
        )
    return table
