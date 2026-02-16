"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/cli_commands/stage_b.py

Stage-B CLI command registration for building library artifacts.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
import typer

from ..cli_commands.context import CliContext
from ..core.artifacts.library import load_library_artifact, write_library_artifact
from ..core.artifacts.pool import _hash_file, load_pool_data
from ..core.pipeline import resolve_plan
from ..core.pipeline.attempts import _load_existing_library_index, _load_failure_counts_from_attempts
from ..core.pipeline.outputs import _emit_event
from ..core.pipeline.plan_pools import PLAN_POOL_INPUT_TYPE, build_plan_pools
from ..core.pipeline.stage_b import assess_library_feasibility, build_library_for_plan
from ..core.seeding import derive_seed_map


def register_stage_b_commands(
    app: typer.Typer,
    *,
    context: CliContext,
    short_hash: Callable[[str], str],
) -> None:
    console = context.console
    make_table = context.make_table

    @app.command("build-libraries", help="Build Stage-B libraries from pools or inputs.")
    def stage_b_build_libraries(
        ctx: typer.Context,
        out: str = typer.Option(
            "outputs/libraries",
            "--out",
            help="Output directory (relative to run root; must be inside outputs/).",
        ),
        pool: Optional[Path] = typer.Option(
            None,
            "--pool",
            help=(
                "Pool directory from `stage-a build-pool` (defaults to outputs/pools for this workspace; "
                "must be inside outputs/)."
            ),
        ),
        input_name: Optional[list[str]] = typer.Option(
            None,
            "--input",
            "-i",
            help="Input name(s) to build (defaults to all inputs).",
        ),
        plan: Optional[list[str]] = typer.Option(
            None,
            "--plan",
            "-p",
            help="Plan item name(s) to build (defaults to all plans).",
        ),
        overwrite: bool = typer.Option(False, help="Overwrite existing library_builds.parquet."),
        append: bool = typer.Option(False, "--append", help="Append new libraries to existing artifacts."),
        show_motif_ids: bool = typer.Option(
            False,
            "--show-motif-ids",
            help="Show full motif IDs instead of TF display names in tables.",
        ),
        config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
    ):
        cfg_path, is_default = context.resolve_config_path(ctx, config)
        loaded = context.load_config_or_exit(
            cfg_path,
            missing_message=context.default_config_missing_message if is_default else None,
        )
        cfg = loaded.root.densegen
        run_root = context.run_root_for(loaded)
        out_dir = context.resolve_outputs_path_or_exit(cfg_path, run_root, out, label="stage-b.out")
        out_dir.mkdir(parents=True, exist_ok=True)
        if overwrite and append:
            console.print("[bold red]Choose either --append or --overwrite, not both.[/]")
            raise typer.Exit(code=1)
        builds_path = out_dir / "library_builds.parquet"
        members_path = out_dir / "library_members.parquet"
        manifest_path = out_dir / "library_manifest.json"
        artifacts_exist = builds_path.exists() or members_path.exists() or manifest_path.exists()
        if artifacts_exist and not (overwrite or append):
            out_label = context.display_path(out_dir, run_root, absolute=False)
            console.print(f"[bold red]Library artifacts already exist under[/] {out_label}.")
            console.print("[bold]Next steps[/]:")
            console.print("  - rerun with --append to add libraries")
            console.print("  - or --overwrite to replace existing library artifacts")
            raise typer.Exit(code=1)

        selected_inputs = {name for name in (input_name or [])}
        if selected_inputs:
            available = {inp.name for inp in cfg.inputs}
            missing = sorted(selected_inputs - available)
            if missing:
                raise typer.BadParameter(f"Unknown input name(s): {', '.join(missing)}")

        selected_plans = {name for name in (plan or [])}
        resolved_plan = resolve_plan(loaded)
        if selected_plans:
            available_plans = {p.name for p in resolved_plan}
            missing = sorted(selected_plans - available_plans)
            if missing:
                raise typer.BadParameter(f"Unknown plan name(s): {', '.join(missing)}")

        seeds = derive_seed_map(int(cfg.runtime.random_seed), ["stage_a", "stage_b", "solver"])
        rng = random.Random(seeds["stage_b"])
        np_rng = np.random.default_rng(seeds["stage_b"])
        sampling_cfg = cfg.generation.sampling
        outputs_root = run_root / "outputs"
        events_path = outputs_root / "meta" / "events.jsonl"
        failure_counts = _load_failure_counts_from_attempts(outputs_root)

        if pool is not None:
            pool_dir = context.resolve_outputs_path_or_exit(cfg_path, run_root, pool, label="stage-b.pool")
        else:
            pool_dir = run_root / "outputs" / "pools"
        if pool_dir.exists() and pool_dir.is_file():
            pool_label = context.display_path(pool_dir, run_root, absolute=False)
            raise typer.BadParameter(
                f"Pool path must be a directory from `stage-a build-pool`, not a file: {pool_label}"
            )
        if not pool_dir.exists() or not pool_dir.is_dir():
            pool_label = context.display_path(pool_dir, run_root, absolute=False)
            raise typer.BadParameter(f"Pool directory not found: {pool_label}")
        try:
            pool_artifact, pool_data = load_pool_data(pool_dir)
        except FileNotFoundError as exc:
            console.print(f"[bold red]{exc}[/]")
            entries = context.list_dir_entries(pool_dir, limit=10)
            if entries:
                console.print(f"[bold]Pool directory contents[/]: {', '.join(entries)}")
            console.print("[bold]Next steps[/]:")
            console.print(
                f"  - {context.workspace_command('dense stage-a build-pool', cfg_path=cfg_path, run_root=run_root)}"
            )
            console.print("  - ensure --pool points to the outputs/pools directory for this workspace")
            raise typer.Exit(code=1)

        config_hash = _hash_file(cfg_path)
        pool_manifest_path = pool_dir / "pool_manifest.json"
        pool_manifest_hash = _hash_file(pool_manifest_path)
        append_mode = append and artifacts_exist
        existing_build_rows: list[dict] = []
        existing_member_rows: list[dict] = []
        existing_libraries_total = 0
        existing_max_index = 0
        existing_created_at = None
        if append_mode:
            if not manifest_path.exists():
                console.print("[bold red]Library manifest not found; cannot append.[/]")
                console.print("[bold]Tip[/]: rerun with --overwrite to rebuild library artifacts.")
                raise typer.Exit(code=1)
            try:
                existing_artifact = load_library_artifact(out_dir)
            except Exception as exc:
                console.print(f"[bold red]Failed to read existing library manifest:[/] {exc}")
                console.print("[bold]Tip[/]: rerun with --overwrite to rebuild library artifacts.")
                raise typer.Exit(code=1)
            existing_config_hash = existing_artifact.config_hash
            existing_pool_hash = existing_artifact.pool_manifest_hash
            if not existing_config_hash or not existing_pool_hash:
                console.print("[bold red]Library manifest missing config/pool hashes; cannot append.[/]")
                console.print("[bold]Tip[/]: rerun with --overwrite to rebuild library artifacts.")
                raise typer.Exit(code=1)
            if existing_config_hash != config_hash or existing_pool_hash != pool_manifest_hash:
                console.print("[bold red]Library manifest does not match current config/pool; cannot append.[/]")
                if existing_config_hash != config_hash:
                    console.print(
                        "  - config hash mismatch "
                        f"(manifest={short_hash(existing_config_hash)}, current={short_hash(config_hash)})"
                    )
                if existing_pool_hash != pool_manifest_hash:
                    console.print(
                        "  - pool manifest hash mismatch "
                        f"(manifest={short_hash(existing_pool_hash)}, current={short_hash(pool_manifest_hash)})"
                    )
                console.print("[bold]Tip[/]: rerun with --overwrite to rebuild library artifacts.")
                raise typer.Exit(code=1)
            if not builds_path.exists() or not members_path.exists():
                console.print("[bold red]Library artifacts are incomplete; cannot append.[/]")
                console.print("[bold]Tip[/]: rerun with --overwrite to rebuild library artifacts.")
                raise typer.Exit(code=1)
            try:
                existing_build_rows = pd.read_parquet(builds_path).to_dict("records")
            except Exception as exc:
                console.print(f"[bold red]Failed to read existing library_builds.parquet:[/] {exc}")
                console.print("[bold]Tip[/]: rerun with --overwrite to rebuild library artifacts.")
                raise typer.Exit(code=1)
            try:
                existing_member_rows = pd.read_parquet(members_path).to_dict("records")
            except Exception as exc:
                console.print(f"[bold red]Failed to read existing library_members.parquet:[/] {exc}")
                console.print("[bold]Tip[/]: rerun with --overwrite to rebuild library artifacts.")
                raise typer.Exit(code=1)
            existing_libraries_total = len(existing_build_rows)
            existing_indices = [
                int(row.get("library_index") or 0)
                for row in existing_build_rows
                if row.get("library_index") is not None
            ]
            existing_max_index = max(existing_indices, default=0)
            existing_created_at = existing_artifact.created_at

        if append_mode:
            libraries_built = existing_max_index
        else:
            libraries_built = _load_existing_library_index(outputs_root) if outputs_root.exists() else 0

        plan_pools = build_plan_pools(plan_items=resolved_plan, pool_data=pool_data)
        plan_specs: list[tuple[object, object]] = []
        for plan_item in resolved_plan:
            spec = plan_pools.get(str(plan_item.name))
            if spec is None:
                continue
            if selected_plans and plan_item.name not in selected_plans:
                continue
            if selected_inputs and not set(selected_inputs).issubset(set(spec.include_inputs)):
                continue
            plan_specs.append((plan_item, spec))

        build_rows = []
        member_rows = []
        with context.suppress_pyarrow_sysctl_warnings():
            for plan_item, spec in plan_specs:
                pool = spec.pool
                try:
                    library, _parts, reg_labels, info = build_library_for_plan(
                        source_label=spec.pool_name,
                        plan_item=plan_item,
                        pool=pool,
                        sampling_cfg=sampling_cfg,
                        seq_len=int(cfg.generation.sequence_length),
                        min_count_per_tf=int(cfg.runtime.min_count_per_tf),
                        usage_counts={},
                        failure_counts=failure_counts if failure_counts else None,
                        rng=rng,
                        np_rng=np_rng,
                        library_index_start=libraries_built,
                    )
                except ValueError as exc:
                    console.print(f"[bold red]Stage-B sampling failed[/]: {exc}")
                    console.print(f"[bold]Context[/]: input={spec.pool_name} plan={plan_item.name}")
                    console.print("[bold]Next steps[/]:")
                    inspect_cmd = context.workspace_command(
                        "dense inspect inputs",
                        cfg_path=cfg_path,
                        run_root=run_root,
                    )
                    console.print("  - ensure regulator_constraints group members match Stage-A regulator labels")
                    console.print(f"  - inspect available regulators via {inspect_cmd}")
                    console.print("    or outputs/pools/pool_manifest.json")
                    raise typer.Exit(code=1)
                libraries_built = int(info.get("library_index", libraries_built))
                library_hash = str(info.get("library_hash") or "")
                achieved_len = int(info.get("achieved_length") or 0)
                pool_strategy = str(info.get("pool_strategy") or sampling_cfg.pool_strategy)
                sampling_strategy = str(info.get("library_sampling_strategy") or sampling_cfg.library_sampling_strategy)
                library_id = library_hash
                tfbs_id_by_index = info.get("tfbs_id_by_index") or []
                motif_id_by_index = info.get("motif_id_by_index") or []
                library_tfbs = list(library)
                library_tfs = list(reg_labels) if reg_labels else []
                _min_required_len, _min_breakdown, feasibility = assess_library_feasibility(
                    library_tfbs=library_tfbs,
                    library_tfs=library_tfs,
                    fixed_elements=plan_item.fixed_elements,
                    groups=list(plan_item.regulator_constraints.groups or []),
                    min_count_by_regulator=dict(plan_item.regulator_constraints.min_count_by_regulator or {}),
                    min_count_per_tf=int(cfg.runtime.min_count_per_tf),
                    sequence_length=int(cfg.generation.sequence_length),
                )
                row = {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "input_name": spec.pool_name,
                    "input_type": PLAN_POOL_INPUT_TYPE,
                    "plan_name": plan_item.name,
                    "library_index": int(info.get("library_index") or 0),
                    "library_id": library_id,
                    "library_hash": library_hash,
                    "library_tfbs": library_tfbs,
                    "library_tfs": library_tfs,
                    "unique_tf_count": len(set(library_tfs)),
                    "library_site_ids": list(info.get("site_id_by_index") or []),
                    "library_sources": list(info.get("source_by_index") or []),
                    "library_tfbs_ids": list(tfbs_id_by_index),
                    "library_motif_ids": list(motif_id_by_index),
                    "pool_strategy": pool_strategy,
                    "library_sampling_strategy": sampling_strategy,
                    "library_size": int(info.get("library_size") or len(library)),
                    "achieved_length": achieved_len,
                    "relaxed_cap": bool(info.get("relaxed_cap") or False),
                    "final_cap": info.get("final_cap"),
                    "iterative_max_libraries": int(info.get("iterative_max_libraries") or 0),
                    "iterative_min_new_solutions": int(info.get("iterative_min_new_solutions") or 0),
                    "required_regulators_selected": info.get("required_regulators_selected"),
                    "fixed_bp": int(feasibility["fixed_bp"]),
                    "min_required_bp": int(feasibility["min_required_bp"]),
                    "slack_bp": int(feasibility["slack_bp"]),
                    "infeasible": bool(feasibility["infeasible"]),
                    "sequence_length": int(feasibility["sequence_length"]),
                }
                build_rows.append(row)
                try:
                    _emit_event(
                        events_path,
                        event="LIBRARY_BUILT",
                        payload={
                            "input_name": spec.pool_name,
                            "plan_name": plan_item.name,
                            "library_index": int(info.get("library_index") or 0),
                            "library_hash": library_hash,
                            "library_size": int(info.get("library_size") or len(library)),
                        },
                    )
                    if info.get("sampling_weight_by_tf"):
                        _emit_event(
                            events_path,
                            event="LIBRARY_SAMPLING_PRESSURE",
                            payload={
                                "input_name": spec.pool_name,
                                "plan_name": plan_item.name,
                                "library_index": int(info.get("library_index") or 0),
                                "library_hash": library_hash,
                                "sampling_strategy": sampling_strategy,
                                "weight_by_tf": info.get("sampling_weight_by_tf"),
                                "weight_fraction_by_tf": info.get("sampling_weight_fraction_by_tf"),
                                "usage_count_by_tf": info.get("sampling_usage_count_by_tf"),
                                "failure_count_by_tf": info.get("sampling_failure_count_by_tf"),
                            },
                        )
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to write Stage-B events for {spec.pool_name}/{plan_item.name}: {exc}"
                    ) from exc
                for idx, tfbs in enumerate(list(library)):
                    member_rows.append(
                        {
                            "library_id": library_id,
                            "library_hash": library_hash,
                            "library_index": int(info.get("library_index") or 0),
                            "input_name": spec.pool_name,
                            "plan_name": plan_item.name,
                            "position": int(idx),
                            "tf": reg_labels[idx] if idx < len(reg_labels or []) else "",
                            "tfbs": tfbs,
                            "tfbs_id": tfbs_id_by_index[idx] if idx < len(tfbs_id_by_index) else None,
                            "motif_id": motif_id_by_index[idx] if idx < len(motif_id_by_index) else None,
                            "site_id": (info.get("site_id_by_index") or [None])[idx]
                            if idx < len(info.get("site_id_by_index") or [])
                            else None,
                            "source": (info.get("source_by_index") or [None])[idx]
                            if idx < len(info.get("source_by_index") or [])
                            else None,
                        }
                    )

            if not build_rows:
                console.print("[yellow]No libraries built (no matching inputs/plans).[/]")
                raise typer.Exit(code=1)

            if append_mode:
                build_rows_out = existing_build_rows + build_rows
                member_rows_out = existing_member_rows + member_rows
                libraries_total = existing_libraries_total + len(build_rows)
                created_at = existing_created_at
                updated_at = datetime.now(timezone.utc).isoformat()
                indices = [
                    int(row.get("library_index") or 0) for row in build_rows_out if row.get("library_index") is not None
                ]
                if len(indices) != len(set(indices)):
                    console.print("[bold red]Duplicate library_index detected after append.[/]")
                    console.print("[bold]Tip[/]: rerun with --overwrite to rebuild library artifacts.")
                    raise typer.Exit(code=1)
            else:
                build_rows_out = build_rows
                member_rows_out = member_rows
                libraries_total = len(build_rows)
                created_at = None
                updated_at = None
            write_overwrite = overwrite or append_mode
            try:
                artifact = write_library_artifact(
                    out_dir=out_dir,
                    builds=build_rows_out,
                    members=member_rows_out,
                    cfg_path=cfg_path,
                    run_id=str(cfg.run.id),
                    run_root=run_root,
                    overwrite=write_overwrite,
                    config_hash=config_hash,
                    pool_manifest_hash=pool_manifest_hash,
                    libraries_total=libraries_total,
                    created_at=created_at,
                    updated_at=updated_at,
                )
            except FileExistsError as exc:
                console.print(f"[bold red]{exc}[/]")
                if append_mode:
                    console.print("[bold]Tip[/]: rerun with --overwrite to rebuild library artifacts.")
                else:
                    console.print("[bold]Tip[/]: rerun with --append or --overwrite to replace existing artifacts.")
                raise typer.Exit(code=1)

        summary_df = pd.DataFrame(build_rows)

        def _format_min_median_max(series: pd.Series) -> str:
            values = pd.to_numeric(series, errors="coerce").dropna()
            if values.empty:
                return "-"
            return f"{values.min():.0f}/{values.median():.0f}/{values.max():.0f}"

        def _format_strategy(series: pd.Series) -> str:
            values = [str(val).strip() for val in series.tolist() if val and str(val).strip()]
            uniq = sorted(set(values))
            if not uniq:
                return "-"
            if len(uniq) == 1:
                return uniq[0]
            return "mixed"

        summary_table = make_table(
            "input",
            "plan",
            "libraries",
            "sites (min/med/max)",
            "TFs (min/med/max)",
            "bp total (min/med/max)",
            "sampling strategy",
        )
        for (input_name, plan_name), group in summary_df.groupby(["input_name", "plan_name"]):
            summary_table.add_row(
                str(input_name),
                str(plan_name),
                str(len(group)),
                _format_min_median_max(group["library_size"]),
                _format_min_median_max(group["unique_tf_count"]),
                _format_min_median_max(group["achieved_length"]),
                _format_strategy(group["library_sampling_strategy"]),
            )

        console.print("[bold]Stage-B libraries (solver inputs)[/]")
        console.print(summary_table)
        run_cmd = context.workspace_command("dense run", cfg_path=cfg_path, run_root=run_root)
        console.print(
            f"Stage-B builds solver libraries from Stage-A pools (cached for `{run_cmd}`). "
            "Sites/TFs/bp totals summarize min/median/max across libraries."
        )
        console.print(f"libraries built now: {len(build_rows)}; libraries total: {libraries_total}")
        console.print(
            f":sparkles: [bold green]Library builds written[/]: "
            f"{context.display_path(artifact.builds_path, run_root, absolute=False)}"
        )
        console.print(
            f":sparkles: [bold green]Library members written[/]: "
            f"{context.display_path(artifact.members_path, run_root, absolute=False)}"
        )
