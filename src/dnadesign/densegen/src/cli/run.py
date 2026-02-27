"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/cli/run.py

Run execution CLI command registration.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from copy import deepcopy
from pathlib import Path
from typing import Callable, Optional

import typer

from ..core.artifacts.pool import pool_status_by_input
from ..core.pipeline import resolve_plan, run_pipeline
from ..core.run_paths import has_existing_run_outputs, run_outputs_root, run_state_path
from ..core.run_state import load_run_state
from ..utils import logging_utils
from ..utils.logging_utils import install_native_stderr_filters, setup_logging
from ..utils.mpl_utils import ensure_mpl_cache_dir
from .context import CliContext

QUOTA_PLAN_INLINE_THRESHOLD = 8


def _resolve_progress_style(log_cfg) -> tuple[str, str | None]:
    requested = str(getattr(log_cfg, "progress_style", "stream"))
    effective, reason = logging_utils.resolve_progress_style(
        requested,
        stdout=sys.stdout,
        term=os.environ.get("TERM"),
    )
    setattr(log_cfg, "progress_style", effective)
    logging_utils.set_progress_style(effective)
    logging_utils.set_progress_enabled(effective in {"stream", "screen"})
    return effective, reason


def _active_stage_a_inputs(plan_items: list) -> set[str]:
    names: set[str] = set()
    for item in plan_items:
        include_inputs = list(getattr(item, "include_inputs", []) or [])
        for input_name in include_inputs:
            value = str(input_name).strip()
            if value:
                names.add(value)
    return names


def _format_quota_plan_message(plan_items: list) -> str:
    if len(plan_items) <= QUOTA_PLAN_INLINE_THRESHOLD:
        return ", ".join(f"{item.name}={item.quota}" for item in plan_items)

    quota_counts: dict[int, int] = {}
    quota_order: list[int] = []
    for item in plan_items:
        quota_value = int(getattr(item, "quota", 0))
        if quota_value not in quota_counts:
            quota_order.append(quota_value)
            quota_counts[quota_value] = 0
        quota_counts[quota_value] += 1

    quota_pattern = "; ".join(f"{quota_counts[value]} plans at {value} each" for value in quota_order)
    return f"{len(plan_items)} plans (quota pattern: {quota_pattern})"


def _model_to_dict(value) -> dict:
    if hasattr(value, "model_dump"):
        return value.model_dump(by_alias=True, exclude_none=False)
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict):
        return dict(value)
    raise TypeError("Expected config model or mapping.")


def _extract_quota_profile(cfg: dict) -> tuple[list[str], dict[str, int], int]:
    generation = cfg.get("generation")
    if not isinstance(generation, dict):
        raise ValueError("Missing generation block in effective config.")
    plan = generation.get("plan")
    if not isinstance(plan, list) or not plan:
        raise ValueError("generation.plan must be a non-empty list.")
    names: list[str] = []
    quotas: dict[str, int] = {}
    for item in plan:
        if not isinstance(item, dict):
            raise ValueError("generation.plan contains non-mapping entries.")
        name = str(item.get("name", "")).strip()
        if not name:
            raise ValueError("generation.plan item is missing name.")
        if "fraction" in item and item.get("fraction") is not None:
            raise ValueError("generation.plan fractions are not supported.")
        quota_raw = item.get("sequences")
        if quota_raw is None:
            raise ValueError("generation.plan items must define sequences.")
        quota_val = int(quota_raw)
        if quota_val <= 0:
            raise ValueError("generation.plan[].sequences must be positive.")
        names.append(name)
        quotas[name] = quota_val
    total = sum(quotas.values())
    if total <= 0:
        raise ValueError("generation.plan quotas must sum to > 0.")
    return names, quotas, total


def _strip_quota_fields(cfg: dict) -> dict:
    normalized = deepcopy(cfg)
    generation = normalized.get("generation")
    if not isinstance(generation, dict):
        return _canonicalize_structure(normalized)
    plan = generation.get("plan")
    if isinstance(plan, list):
        for item in plan:
            if not isinstance(item, dict):
                continue
            item.pop("sequences", None)
    return _canonicalize_structure(normalized)


def _canonicalize_structure(value):
    if isinstance(value, dict):
        return {k: _canonicalize_structure(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_canonicalize_structure(v) for v in value]
    if isinstance(value, list):
        return [_canonicalize_structure(v) for v in value]
    return value


def _validate_quota_increase_only(*, previous_cfg: dict, current_cfg: dict) -> None:
    prev_order, prev_quotas, prev_total = _extract_quota_profile(previous_cfg)
    curr_order, curr_quotas, curr_total = _extract_quota_profile(current_cfg)
    if prev_order != curr_order:
        raise ValueError("Plan names/order changed; quota-only resume is not allowed.")
    for name in prev_order:
        if curr_quotas[name] < prev_quotas[name]:
            raise ValueError(f"Plan quota decreased for '{name}' ({prev_quotas[name]} -> {curr_quotas[name]}).")
    if curr_total < prev_total:
        raise ValueError(f"Total plan quota decreased ({prev_total} -> {curr_total}).")
    prev_norm = _strip_quota_fields(previous_cfg)
    curr_norm = _strip_quota_fields(current_cfg)
    if prev_norm != curr_norm:
        raise ValueError("Config changed beyond plan quotas.")


def _load_previous_densegen_config(run_root: Path) -> dict:
    effective_path = run_root / "outputs" / "meta" / "effective_config.json"
    if not effective_path.exists():
        raise ValueError("Missing outputs/meta/effective_config.json; cannot validate quota-only config changes.")
    payload = json.loads(effective_path.read_text())
    previous_cfg = payload.get("config")
    if not isinstance(previous_cfg, dict):
        raise ValueError("effective_config.json is missing a valid 'config' object.")
    return previous_cfg


def _distribute_quota_extension(*, quotas: list[int], extension_rows: int) -> list[int]:
    if extension_rows <= 0:
        raise ValueError("--extend-quota must be > 0")
    if not quotas:
        raise ValueError("generation.plan must contain at least one item")
    total_quota = sum(int(value) for value in quotas)
    if total_quota <= 0:
        raise ValueError("generation.plan quotas must sum to > 0")
    additions = [0 for _ in quotas]
    remainders: list[tuple[int, int]] = []
    allocated = 0
    for index, quota_value in enumerate(quotas):
        scaled = int(quota_value) * int(extension_rows)
        add = scaled // total_quota
        remainder = scaled % total_quota
        additions[index] = add
        remainders.append((remainder, index))
        allocated += add
    remaining = int(extension_rows) - int(allocated)
    for _remainder, index in sorted(remainders, key=lambda pair: (-pair[0], pair[1]))[:remaining]:
        additions[index] += 1
    return [int(quota) + int(additions[index]) for index, quota in enumerate(quotas)]


def _apply_extend_quota(
    *,
    cfg,
    extension_rows: int,
    generated_by_plan: dict[str, int] | None = None,
) -> list[tuple[str, int, int]]:
    plan_items = list(cfg.generation.plan or [])
    existing = generated_by_plan or {}
    anchor_quotas = [max(int(item.sequences), int(existing.get(str(item.name), 0))) for item in plan_items]
    updated = _distribute_quota_extension(quotas=anchor_quotas, extension_rows=extension_rows)
    changes: list[tuple[str, int, int]] = []
    for item, old_quota, new_quota in zip(plan_items, anchor_quotas, updated):
        item.sequences = int(new_quota)
        changes.append((str(item.name), int(old_quota), int(new_quota)))
    return changes


def _load_generated_counts_by_plan(*, state_path: Path, run_id: str) -> dict[str, int]:
    if not state_path.exists():
        return {}
    state = load_run_state(state_path)
    if str(state.run_id) != str(run_id):
        return {}
    counts: dict[str, int] = {}
    for item in state.items:
        plan_name = str(item.plan_name)
        counts[plan_name] = counts.get(plan_name, 0) + int(item.generated)
    return counts


def _apply_resume_quota_floor(
    *,
    cfg,
    run_root: Path,
    generated_by_plan: dict[str, int],
) -> list[tuple[str, int, int]]:
    previous_quotas: dict[str, int] = {}
    try:
        previous_cfg = _load_previous_densegen_config(run_root)
        _names, previous_quotas, _total = _extract_quota_profile(previous_cfg)
    except ValueError:
        previous_quotas = {}
    changes: list[tuple[str, int, int]] = []
    for item in list(cfg.generation.plan or []):
        name = str(item.name)
        old_quota = int(item.sequences)
        floor_quota = max(
            old_quota,
            int(previous_quotas.get(name, 0)),
            int(generated_by_plan.get(name, 0)),
        )
        if floor_quota != old_quota:
            item.sequences = int(floor_quota)
            changes.append((name, old_quota, floor_quota))
    return changes


def _capture_usr_registry_snapshots(*, cfg, cfg_path: Path, run_root: Path, context: CliContext) -> dict[Path, str]:
    snapshots: dict[Path, str] = {}
    out_cfg = cfg.output
    if "usr" not in out_cfg.targets or out_cfg.usr is None:
        return snapshots
    usr_root = context.resolve_outputs_path_or_exit(
        cfg_path,
        run_root,
        Path(out_cfg.usr.root),
        label="output.usr.root",
    )
    registry_path = usr_root / "registry.yaml"
    if registry_path.exists() and registry_path.is_file():
        snapshots[registry_path] = registry_path.read_text()
    return snapshots


def _restore_usr_registry_snapshots(*, snapshots: dict[Path, str]) -> None:
    for registry_path, content in snapshots.items():
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        registry_path.write_text(content)


def _clear_outputs_preserving_notify(*, outputs_root: Path) -> None:
    notify_root = outputs_root / "notify"
    for child in outputs_root.iterdir():
        if child == notify_root:
            continue
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _resolve_resume_mode(
    *,
    fresh: bool,
    resume: bool,
    existing_outputs: bool,
    outputs_root: Path,
    cfg,
    cfg_path: Path,
    run_root: Path,
    context: CliContext,
    console,
) -> bool:
    if fresh and resume:
        console.print("[bold red]Choose either --fresh or --resume, not both.[/]")
        raise typer.Exit(code=1)

    if fresh:
        registry_snapshots = _capture_usr_registry_snapshots(
            cfg=cfg,
            cfg_path=cfg_path,
            run_root=run_root,
            context=context,
        )
        if outputs_root.exists():
            try:
                _clear_outputs_preserving_notify(outputs_root=outputs_root)
            except Exception as exc:
                console.print(f"[bold red]Failed to clear outputs:[/] {exc}")
                raise typer.Exit(code=1) from exc
            if registry_snapshots:
                _restore_usr_registry_snapshots(snapshots=registry_snapshots)
            console.print(
                ":broom: [bold yellow]Cleared outputs[/]: "
                f"{context.display_path(outputs_root, run_root, absolute=False)}"
            )
        else:
            console.print("[yellow]No outputs directory found; starting fresh.[/]")
        return False

    if resume:
        if not existing_outputs:
            console.print(
                f"[bold red]--resume requested but no outputs were found under[/] "
                f"{context.display_path(outputs_root, run_root, absolute=False)}. "
                "Run without --resume or use --fresh to reset the workspace."
            )
            raise typer.Exit(code=1)
        return True

    if existing_outputs:
        console.print(
            f"[yellow]Existing outputs found under[/] "
            f"{context.display_path(outputs_root, run_root, absolute=False)}. "
            "Resuming from existing outputs."
        )
        return True
    return False


def _validate_existing_state_for_resume(
    *,
    fresh: bool,
    state_path: Path,
    cfg,
    cfg_path: Path,
    run_root: Path,
    context: CliContext,
    console,
) -> bool:
    allow_config_mismatch = False
    if fresh or not state_path.exists():
        return allow_config_mismatch

    existing_state = load_run_state(state_path)
    config_sha = hashlib.sha256(cfg_path.read_bytes()).hexdigest()
    if existing_state.run_id and existing_state.run_id != str(cfg.run.id):
        console.print(
            "[bold red]Existing run_state.json was created with a different run_id. "
            "Remove run_state.json or stage a new run root to start fresh.[/]"
        )
        console.print("[bold]Next steps[/]:")
        fresh_cmd = context.workspace_command(
            "dense run --fresh",
            cfg_path=cfg_path,
            run_root=run_root,
        )
        reset_cmd = context.workspace_command(
            "dense campaign-reset",
            cfg_path=cfg_path,
            run_root=run_root,
        )
        console.print(f"  - {fresh_cmd}")
        console.print(f"  - {reset_cmd}")
        raise typer.Exit(code=1)

    if existing_state.config_sha256 and existing_state.config_sha256 != config_sha:
        try:
            previous_cfg = _load_previous_densegen_config(run_root)
            current_cfg = _model_to_dict(cfg)
            _validate_quota_increase_only(previous_cfg=previous_cfg, current_cfg=current_cfg)
        except ValueError as exc:
            console.print(
                "[bold red]Existing run_state.json was created with a different config "
                "and changes are not quota-only.[/]"
            )
            console.print(f"[bold red]Quota validation failed:[/] {exc}")
            console.print("[bold]Next steps[/]:")
            fresh_cmd = context.workspace_command(
                "dense run --fresh",
                cfg_path=cfg_path,
                run_root=run_root,
            )
            reset_cmd = context.workspace_command(
                "dense campaign-reset",
                cfg_path=cfg_path,
                run_root=run_root,
            )
            console.print(f"  - {fresh_cmd}")
            console.print(f"  - {reset_cmd}")
            raise typer.Exit(code=1) from exc
        allow_config_mismatch = True
        console.print("[yellow]Quota-only config change detected; resuming with existing outputs.[/]")

    if (
        not has_existing_run_outputs(run_root)
        and existing_state.items
        and sum(item.generated for item in existing_state.items) > 0
    ):
        console.print(
            "[bold red]run_state.json indicates prior progress, but no outputs were found. "
            "Restore outputs or delete run_state.json before resuming.[/]"
        )
        console.print("[bold]Next steps[/]:")
        reset_cmd = context.workspace_command(
            "dense campaign-reset",
            cfg_path=cfg_path,
            run_root=run_root,
        )
        console.print(f"  - {reset_cmd}")
        console.print("  - or restore outputs/ before resuming")
        raise typer.Exit(code=1)
    return allow_config_mismatch


def _print_run_next_steps(*, cfg_path: Path, run_root: Path, context: CliContext, console) -> None:
    console.print("[bold]Next steps[/]:")
    inspect_cmd = context.workspace_command(
        "dense inspect run --library",
        cfg_path=cfg_path,
        run_root=run_root,
    )
    list_plots_cmd = context.workspace_command(
        "dense ls-plots",
        cfg_path=cfg_path,
        run_root=run_root,
    )
    plot_cmd = context.workspace_command(
        "dense plot",
        cfg_path=cfg_path,
        run_root=run_root,
    )
    placement_cmd = context.workspace_command(
        "dense plot --only placement_map",
        cfg_path=cfg_path,
        run_root=run_root,
    )
    notebook_cmd = context.workspace_command(
        "dense notebook generate",
        cfg_path=cfg_path,
        run_root=run_root,
    )
    console.print(f"  - {inspect_cmd} (Stage-B library + solutions)")
    console.print(f"  - {list_plots_cmd} (list available plot ids for this workspace)")
    console.print(f"  - {plot_cmd} (render configured plot set)")
    console.print(f"  - {placement_cmd} (Stage-B placement plots)")
    console.print(f"  - {notebook_cmd} (generate workspace-scoped marimo run notebook)")


def _run_plots_if_configured(*, no_plot: bool, root, loaded, console) -> None:
    if no_plot or root.plots is None:
        return
    try:
        ensure_mpl_cache_dir()
    except Exception as exc:
        console.print(f"[bold red]Matplotlib cache setup failed:[/] {exc}")
        console.print(
            "[bold]Tip[/]: DenseGen defaults to a repo-local cache at .cache/matplotlib/densegen; "
            "set MPLCONFIGDIR to override."
        )
        raise typer.Exit(code=1) from exc
    install_native_stderr_filters(suppress_solver_messages=False)
    from ..viz.plotting import run_plots_from_config

    console.print("[bold]Generating plots...[/]")
    run_plots_from_config(root, loaded.path, source="run")
    console.print(":bar_chart: [bold green]Plots written.[/]")


def _handle_run_runtime_error(
    *,
    exc: RuntimeError,
    render_output_schema_hint: Callable[..., bool],
    context: CliContext,
    cfg_path: Path,
    run_root: Path,
    console,
) -> None:
    if render_output_schema_hint(exc):
        raise typer.Exit(code=1)
    message = str(exc)
    if "progress_style=screen requires" in message:
        console.print(f"[bold red]{message}[/]")
        console.print("[bold]Next steps[/]:")
        console.print("  - run in an interactive terminal with TERM=xterm-256color")
        console.print("  - or set densegen.logging.progress_style: stream")
        raise typer.Exit(code=1)
    if "Existing run_state.json was created with a different" in message:
        console.print(f"[bold red]{message}[/]")
        console.print("[bold]Next steps[/]:")
        fresh_cmd = context.workspace_command(
            "dense run --fresh",
            cfg_path=cfg_path,
            run_root=run_root,
        )
        reset_cmd = context.workspace_command(
            "dense campaign-reset",
            cfg_path=cfg_path,
            run_root=run_root,
        )
        console.print(f"  - {fresh_cmd}")
        console.print(f"  - {reset_cmd}")
        raise typer.Exit(code=1)
    if "run_state.json indicates prior progress" in message:
        console.print(f"[bold red]{message}[/]")
        console.print("[bold]Next steps[/]:")
        reset_cmd = context.workspace_command(
            "dense campaign-reset",
            cfg_path=cfg_path,
            run_root=run_root,
        )
        console.print(f"  - {reset_cmd}")
        console.print("  - or restore outputs/ before resuming")
        raise typer.Exit(code=1)
    if "Stage-A pools missing or stale" in message:
        console.print(f"[bold red]{message}[/]")
        console.print("[bold]Next steps[/]:")
        rebuild_cmd = context.workspace_command(
            "dense stage-a build-pool --fresh",
            cfg_path=cfg_path,
            run_root=run_root,
        )
        console.print(f"  - {rebuild_cmd}")
        console.print("  - or rerun with --fresh to rebuild outputs and pools")
        console.print(
            "  - Stage-B libraries are built during `uv run dense run`; "
            "no need to run `uv run dense stage-b build-libraries`"
        )
        raise typer.Exit(code=1)
    if "USR registry not found at " in message:
        console.print(f"[bold red]{message}[/]")
        console.print("[bold]Next steps[/]:")
        console.print("  - stage your run via `dense workspace init --output-mode usr|both` to seed registry.yaml")
        console.print("  - or create `<output.usr.root>/registry.yaml` before running `dense run`")
        raise typer.Exit(code=1)
    if "Exceeded max_consecutive_no_progress_resamples=" in message or "Exceeded max_failed_solutions=" in message:
        console.print(f"[bold red]{message}[/]")
        console.print("[bold]Next steps[/]:")
        inspect_cmd = context.workspace_command(
            "dense inspect run --events --library",
            cfg_path=cfg_path,
            run_root=run_root,
        )
        console.print(f"  - {inspect_cmd}")
        console.print(
            "  - increase densegen.runtime.no_progress_seconds_before_resample, "
            "max_consecutive_no_progress_resamples, "
            "max_failed_solutions_per_target, or max_failed_solutions"
        )
        console.print("  - or relax constraints / lower quota for the affected plan")
        raise typer.Exit(code=1)
    raise exc


def register_run_commands(
    app: typer.Typer,
    *,
    context: CliContext,
    render_missing_input_hint: Callable[..., None],
    render_output_schema_hint: Callable[..., bool],
    ensure_fimo_available: Callable[..., None],
) -> None:
    console = context.console

    @app.command(help="Run generation for the job. Optionally auto-run plots declared in YAML.")
    def run(
        ctx: typer.Context,
        no_plot: bool = typer.Option(False, help="Do not auto-run plots even if configured."),
        fresh: bool = typer.Option(False, "--fresh", help="Clear outputs and start a new run."),
        resume: bool = typer.Option(False, "--resume", help="Resume from existing outputs."),
        extend_quota: Optional[int] = typer.Option(
            None,
            "--extend-quota",
            help="Increase total plan quota by this many rows for this run without editing config.yaml.",
        ),
        log_file: Optional[Path] = typer.Option(
            None,
            help="Override logfile path (must be inside outputs/ under the run root).",
        ),
        show_tfbs: bool = typer.Option(False, "--show-tfbs", help="Show TFBS sequences in progress output."),
        show_solutions: bool = typer.Option(False, "--show-solutions", help="Show full solution sequences in output."),
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
        log_cfg = cfg.logging
        try:
            resolved_progress_style, progress_reason = _resolve_progress_style(log_cfg)
        except RuntimeError as exc:
            message = str(exc)
            if "progress_style=screen requires" in message:
                console.print(f"[bold red]{message}[/]")
                console.print("[bold]Next steps[/]:")
                console.print("  - run in an interactive terminal with TERM=xterm-256color")
                console.print("  - or set densegen.logging.progress_style: stream")
                raise typer.Exit(code=1)
            raise
        if progress_reason is not None:
            console.print(f"[dim]logging.progress_style=auto -> {resolved_progress_style} ({progress_reason})[/]")

        outputs_root = run_outputs_root(run_root)
        existing_outputs = has_existing_run_outputs(run_root)
        resume_run = _resolve_resume_mode(
            fresh=fresh,
            resume=resume,
            existing_outputs=existing_outputs,
            outputs_root=outputs_root,
            cfg=cfg,
            cfg_path=loaded.path,
            run_root=run_root,
            context=context,
            console=console,
        )

        state_path = run_state_path(run_root)
        generated_by_plan: dict[str, int] = _load_generated_counts_by_plan(
            state_path=state_path,
            run_id=str(cfg.run.id),
        )
        allow_config_mismatch = _validate_existing_state_for_resume(
            fresh=fresh,
            state_path=state_path,
            cfg=cfg,
            cfg_path=cfg_path,
            run_root=run_root,
            context=context,
            console=console,
        )

        if resume_run:
            quota_floor_changes = _apply_resume_quota_floor(
                cfg=cfg,
                run_root=run_root,
                generated_by_plan=generated_by_plan,
            )
            if quota_floor_changes:
                updates = ", ".join(f"{name}:{old}->{new}" for name, old, new in quota_floor_changes)
                console.print(f"[yellow]Resuming with prior effective quota floor[/]: {updates}")

        if extend_quota is not None:
            extend_value = int(extend_quota)
            if extend_value <= 0:
                console.print("[bold red]--extend-quota must be > 0[/]")
                raise typer.Exit(code=1)
            try:
                quota_changes = _apply_extend_quota(
                    cfg=cfg,
                    extension_rows=extend_value,
                    generated_by_plan=generated_by_plan,
                )
            except ValueError as exc:
                console.print(f"[bold red]{exc}[/]")
                raise typer.Exit(code=1)
            updates = ", ".join(f"{name}:{old}->{new}" for name, old, new in quota_changes)
            console.print(f"[yellow]Extended total quota by {extend_value} rows for this run[/]: {updates}")

        pl = resolve_plan(loaded)
        active_inputs = _active_stage_a_inputs(pl)
        if fresh:
            build_stage_a = True
        else:
            statuses = pool_status_by_input(cfg, cfg_path, run_root)
            rebuild_needed = any(
                status.state != "present" for name, status in statuses.items() if name in active_inputs
            )
            stale_unused = sorted(
                name for name, status in statuses.items() if name not in active_inputs and status.state != "present"
            )
            if stale_unused:
                console.print(f"[yellow]Ignoring stale Stage-A pools for unused inputs: {', '.join(stale_unused)}[/]")
            if rebuild_needed:
                console.print("[yellow]Stage-A pools missing or stale; rebuilding before run.[/]")
            build_stage_a = rebuild_needed

        if build_stage_a:
            ensure_fimo_available(cfg, strict=True)

        # Logging setup
        log_dir = context.resolve_outputs_path_or_exit(
            loaded.path,
            run_root,
            Path(log_cfg.log_dir),
            label="logging.log_dir",
        )
        default_logfile = log_dir / f"{cfg.run.id}.log"
        if log_file is not None:
            logfile = context.resolve_outputs_path_or_exit(loaded.path, run_root, log_file, label="logging.log_file")
        else:
            logfile = default_logfile
        setup_logging(
            level=log_cfg.level,
            logfile=str(logfile),
            suppress_solver_stderr=bool(log_cfg.suppress_solver_stderr),
        )

        # Plan & solver
        console.print("[bold]Quota plan[/]: " + _format_quota_plan_message(pl))
        try:
            summary = run_pipeline(
                loaded,
                resume=resume_run,
                build_stage_a=build_stage_a,
                show_tfbs=show_tfbs,
                show_solutions=show_solutions,
                allow_config_mismatch=allow_config_mismatch,
            )
        except FileNotFoundError as exc:
            render_missing_input_hint(cfg_path, loaded, exc)
            raise typer.Exit(code=1)
        except RuntimeError as exc:
            _handle_run_runtime_error(
                exc=exc,
                render_output_schema_hint=render_output_schema_hint,
                context=context,
                cfg_path=cfg_path,
                run_root=run_root,
                console=console,
            )

        if int(getattr(summary, "generated_this_run", 0)) == 0:
            console.print("[yellow]Quota is already met; no new sequences generated.[/]")

        console.print(":tada: [bold green]Run complete[/].")
        _print_run_next_steps(
            cfg_path=cfg_path,
            run_root=run_root,
            context=context,
            console=console,
        )
        _run_plots_if_configured(
            no_plot=no_plot,
            root=root,
            loaded=loaded,
            console=console,
        )

    @app.command("campaign-reset", help="Remove run outputs to reset a workspace.")
    def campaign_reset(
        ctx: typer.Context,
        config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
    ):
        cfg_path, is_default = context.resolve_config_path(ctx, config)
        loaded = context.load_config_or_exit(
            cfg_path,
            missing_message=context.default_config_missing_message if is_default else None,
        )
        run_root = context.run_root_for(loaded)
        outputs_root = run_outputs_root(run_root)
        if not outputs_root.exists():
            console.print(
                f"[bold yellow]No outputs found under[/] {context.display_path(outputs_root, run_root, absolute=False)}"
            )
            return
        if not outputs_root.is_dir():
            console.print(
                "[bold red]Outputs path is not a directory:[/] "
                f"{context.display_path(outputs_root, run_root, absolute=False)}"
            )
            raise typer.Exit(code=1)
        shutil.rmtree(outputs_root)
        console.print(
            ":broom: [bold green]Removed outputs under[/] "
            f"{context.display_path(outputs_root, run_root, absolute=False)}"
        )
