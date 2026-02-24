"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/study.py

CLI entrypoints for first-class Study workflows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from dnadesign.cruncher.app.progress import progress_output_enabled
from dnadesign.cruncher.cli.config_resolver import (
    CANDIDATE_CONFIG_FILENAMES,
    WorkspaceCandidate,
    discover_workspaces,
    resolve_invocation_cwd,
    workspace_search_roots,
)
from dnadesign.cruncher.cli.paths import render_path
from dnadesign.cruncher.study.discovery import discover_study_runs_for_workspace, discover_study_specs_for_workspace
from dnadesign.cruncher.study.load import load_study_spec

app = typer.Typer(no_args_is_help=True, help="Run and summarize parameter sweep studies.")
console = Console()


def run_study(*args, **kwargs):
    from dnadesign.cruncher.app.study_workflow import run_study as _run_study

    return _run_study(*args, **kwargs)


def summarize_study_run(*args, **kwargs):
    from dnadesign.cruncher.app.study_workflow import summarize_study_run as _summarize_study_run

    return _summarize_study_run(*args, **kwargs)


def study_show_payload(*args, **kwargs):
    from dnadesign.cruncher.app.study_workflow import study_show_payload as _study_show_payload

    return _study_show_payload(*args, **kwargs)


def compact_study_run(*args, **kwargs):
    from dnadesign.cruncher.app.study_compaction import compact_study_run as _compact_study_run

    return _compact_study_run(*args, **kwargs)


def _resolve_cli_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (resolve_invocation_cwd() / expanded).resolve()


def _progress_enabled() -> bool:
    return bool(progress_output_enabled() and console.is_terminal)


def _workspace_from_path(path: Path) -> WorkspaceCandidate:
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        if resolved.name not in CANDIDATE_CONFIG_FILENAMES:
            allowed = ", ".join(CANDIDATE_CONFIG_FILENAMES)
            raise ValueError(f"--workspace file must be one of: {allowed}. Got: {resolved.name}")
        root = resolved.parent
        config_path = resolved
    elif resolved.is_dir():
        configs = [resolved / name for name in CANDIDATE_CONFIG_FILENAMES if (resolved / name).is_file()]
        if len(configs) != 1:
            allowed = ", ".join(CANDIDATE_CONFIG_FILENAMES)
            if len(configs) == 0:
                raise ValueError(f"--workspace directory has no config file. Expected one of: {allowed}")
            rendered = ", ".join(path.name for path in configs)
            raise ValueError(f"--workspace directory has multiple config files: {rendered}")
        root = resolved
        config_path = configs[0].resolve()
    else:
        raise ValueError(f"--workspace path is neither a file nor a directory: {resolved}")

    return WorkspaceCandidate(
        name=root.name,
        root=root,
        config_path=config_path,
        catalog_path=(root / ".cruncher" / "catalog.json").resolve(),
    )


def _resolve_workspace(selector: str, workspaces: list[WorkspaceCandidate]) -> WorkspaceCandidate:
    token = str(selector).strip()
    if not token:
        raise ValueError("--workspace must be non-empty.")
    if token.isdigit():
        idx = int(token)
        if idx < 1 or idx > len(workspaces):
            raise ValueError(f"--workspace index {idx} is out of range (1..{len(workspaces)}).")
        return workspaces[idx - 1]

    named = [item for item in workspaces if item.name == token]
    if len(named) == 1:
        return named[0]
    if len(named) > 1:
        rendered = ", ".join(str(item.root) for item in named)
        raise ValueError(f"--workspace '{token}' is ambiguous. Matches: {rendered}")

    path = Path(token).expanduser()
    resolved = _resolve_cli_path(path)
    if resolved.exists():
        for item in workspaces:
            if item.root == resolved or item.config_path == resolved:
                return item
        return _workspace_from_path(resolved)

    if not workspaces:
        raise ValueError(
            f"--workspace '{token}' was not found and no discoverable workspaces exist. "
            "Pass an existing workspace path or set CRUNCHER_WORKSPACE_ROOTS."
        )
    available = ", ".join(item.name for item in workspaces)
    raise ValueError(f"--workspace '{token}' was not found. Available workspaces: {available}")


@app.command("list", help="List workspace-scoped Study specs and Study run outputs.")
def list_cmd(
    workspace: str | None = typer.Option(
        None,
        "--workspace",
        help="Workspace selector (name, index, or path). If omitted, list all discoverable workspaces.",
    ),
) -> None:
    try:
        workspaces = discover_workspaces()
        if workspace is not None:
            workspaces = [_resolve_workspace(workspace, workspaces)]
        elif not workspaces:
            roots = workspace_search_roots()
            roots_rendered = "\n".join(f"- {root}" for root in roots) if roots else "- (none)"
            console.print("No workspaces discovered.")
            console.print("Workspace discovery searched:")
            console.print(roots_rendered)
            raise typer.Exit(code=1)

        spec_rows = []
        run_rows = []
        for workspace in workspaces:
            rows = discover_study_specs_for_workspace(workspace_name=workspace.name, workspace_root=workspace.root)
            spec_rows.extend(rows)
            run_rows.extend(discover_study_runs_for_workspace(rows))

        console.print(
            "Layout: specs live in <workspace>/configs/studies/*.study.yaml; "
            "run artifacts live in <workspace>/outputs/studies/<study_name>/<study_id>."
        )

        if spec_rows:
            spec_table = Table(title="Study Specs", header_style="bold")
            spec_table.add_column("Workspace")
            spec_table.add_column("Study")
            spec_table.add_column("Spec")
            for row in sorted(spec_rows, key=lambda item: (item.workspace_name, item.study_name, str(item.spec_path))):
                spec_table.add_row(
                    row.workspace_name,
                    row.study_name,
                    render_path(row.spec_path),
                )
            console.print(spec_table)
        else:
            console.print("No Study specs discovered in known workspaces.")

        if run_rows:
            run_table = Table(title="Study Runs", header_style="bold")
            run_table.add_column("Workspace")
            run_table.add_column("Study")
            run_table.add_column("Study ID")
            run_table.add_column("Status")
            run_table.add_column("Run")
            for row in run_rows:
                run_table.add_row(
                    row.workspace_name,
                    row.study_name,
                    row.study_id,
                    row.status,
                    render_path(row.run_dir),
                )
            console.print(run_table)
        else:
            console.print("No Study runs discovered in known workspaces.")
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc


@app.command("clean", help="Delete Study output artifacts for a selected workspace/study run.")
def clean_cmd(
    workspace: str = typer.Option(..., "--workspace", help="Workspace selector (name, index, or path)."),
    study: str = typer.Option(..., "--study", help="Study name (matches <workspace>/<name>.study.yaml)."),
    study_id: str | None = typer.Option(None, "--id", help="Delete one Study run directory by deterministic study_id."),
    all_runs: bool = typer.Option(False, "--all", help="Delete all Study run directories for the selected Study."),
    confirm: bool = typer.Option(False, "--confirm", help="Execute deletion. Without this flag, command is dry-run."),
) -> None:
    try:
        if bool(study_id) == bool(all_runs):
            raise ValueError("Specify exactly one of --id or --all.")

        workspaces = discover_workspaces()

        selected_workspace = _resolve_workspace(workspace, workspaces)
        study_name = str(study).strip()
        if not study_name:
            raise ValueError("--study must be non-empty.")

        spec_rows = discover_study_specs_for_workspace(
            workspace_name=selected_workspace.name,
            workspace_root=selected_workspace.root,
        )
        matched_specs = [item for item in spec_rows if item.study_name == study_name]
        if not matched_specs:
            available = ", ".join(sorted({item.study_name for item in spec_rows})) or "(none)"
            raise FileNotFoundError(
                f"Study spec with study.name={study_name!r} not found for workspace '{selected_workspace.name}'. "
                f"Available studies: {available}"
            )
        spec = load_study_spec(matched_specs[0].spec_path)
        if spec.name != study_name:
            raise ValueError(
                f"Study name mismatch for spec {matched_specs[0].spec_path}: "
                f"expected {study_name!r}, found {spec.name!r}."
            )
        run_rows = [item for item in discover_study_runs_for_workspace(spec_rows) if item.study_name == study_name]
        if study_id:
            target_id = str(study_id).strip()
            run_rows = [item for item in run_rows if item.study_id == target_id]
            if not run_rows:
                raise ValueError(
                    f"No Study run found for workspace={selected_workspace.name!r}, study={study_name!r}, "
                    f"study_id={target_id!r}."
                )
        elif not run_rows:
            raise ValueError(f"No Study runs found for workspace={selected_workspace.name!r}, study={study_name!r}.")

        console.print(
            f"Study clean target -> workspace={selected_workspace.name} study={study_name} run_count={len(run_rows)}"
        )
        for row in run_rows:
            console.print(f"  run: {row.study_id} -> {render_path(row.run_dir)}")

        if not confirm:
            console.print("Dry run only. Re-run with --confirm to delete Study output artifacts.")
            return

        for row in run_rows:
            shutil.rmtree(row.run_dir)
        plural = "directory" if len(run_rows) == 1 else "directories"
        console.print(f"Deleted {len(run_rows)} Study run {plural}.")
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc


@app.command("run", help="Execute a Study spec end-to-end (trials, optional replays, and summary).")
def run_cmd(
    spec: Path = typer.Option(..., "--spec", help="Path to <workspace>/configs/studies/<name>.study.yaml."),
    resume: bool = typer.Option(False, "--resume", help="Resume from existing study manifest state."),
    force_overwrite: bool = typer.Option(
        False,
        "--force-overwrite",
        help="Delete and recreate the deterministic study run directory before execution.",
    ),
) -> None:
    resolved_spec = _resolve_cli_path(spec)
    progress_bar = _progress_enabled()

    try:
        # Validate schema/base_config contracts before importing heavy study workflow modules.
        load_study_spec(resolved_spec)
        study_run_dir = run_study(
            resolved_spec,
            resume=resume,
            force_overwrite=force_overwrite,
            progress_bar=progress_bar,
            quiet_logs=progress_bar,
        )
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1)
    payload = study_show_payload(study_run_dir)
    console.print(f"Study outputs -> {render_path(study_run_dir)}", soft_wrap=True)
    console.print(f"  status: {payload['status']}")
    console.print(f"  manifest: {render_path(payload['manifest_path'])}", soft_wrap=True)
    console.print(f"  tables: {len(payload['table_paths'])}")
    console.print(f"  plots: {len(payload['plot_paths'])}")


@app.command("summarize", help="Recompute aggregate Study tables and plots from completed trial runs.")
def summarize_cmd(
    run: Path = typer.Option(..., "--run", help="Path to a Study run directory."),
    allow_partial: bool = typer.Option(
        False,
        "--allow-partial",
        help="Allow summarize to include only successful trials when failures or missing artifacts exist.",
    ),
) -> None:
    run_dir = _resolve_cli_path(run)

    try:
        summary = summarize_study_run(run_dir, allow_partial=allow_partial)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1)
    payload = study_show_payload(run_dir)
    console.print(f"Study summary refreshed -> {render_path(run_dir)}", soft_wrap=True)
    console.print(f"  status: {payload['status']}")
    for path in payload["plot_paths"]:
        console.print(f"  plot: {render_path(path)}", soft_wrap=True)
    if allow_partial and summary.n_missing_total > 0:
        console.print(
            "  warning: summary used partial data "
            f"(n_missing_total={summary.n_missing_total}, "
            f"non_success={summary.n_missing_non_success}, "
            f"missing_run_dirs={summary.n_missing_run_dirs}, "
            f"missing_metric_artifacts={summary.n_missing_metric_artifacts}, "
            f"missing_mmr_tables={summary.n_missing_mmr_tables})"
        )
        if summary.exit_code_policy == "nonzero_if_any_error":
            console.print("Error: summary used partial data and exit_code_policy requires non-zero exit.")
            raise typer.Exit(code=1)


@app.command("compact", help="Prune transient trial artifacts from a Study run (dry-run by default).")
def compact_cmd(
    run: Path = typer.Option(..., "--run", help="Path to a Study run directory."),
    confirm: bool = typer.Option(
        False,
        "--confirm",
        help="Execute deletion. Without this flag, command reports what would be removed.",
    ),
) -> None:
    run_dir = _resolve_cli_path(run)

    try:
        summary = compact_study_run(run_dir, confirm=confirm)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1)
    console.print(f"Study compact target -> {render_path(run_dir)}")
    console.print(f"  trials considered: {summary.trial_count}")
    console.print(f"  candidate files: {summary.candidate_file_count}")
    console.print(f"  candidate bytes: {summary.candidate_bytes}")
    if not confirm:
        console.print("Dry run only. Re-run with --confirm to delete listed transient trial artifacts.")
        return
    console.print(f"  deleted files: {summary.deleted_file_count}")
    console.print(f"  deleted bytes: {summary.deleted_bytes}")


@app.command("show", help="Print Study run status and key artifact paths.")
def show_cmd(
    run: Path = typer.Option(..., "--run", help="Path to a Study run directory."),
) -> None:
    run_dir = _resolve_cli_path(run)

    try:
        payload = study_show_payload(run_dir)
    except (RuntimeError, ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1)
    console.print(f"Study -> {payload['study_name']} ({payload['study_id']})")
    console.print(
        "  status: "
        f"{payload['status']} "
        f"(success={payload['success_runs']} error={payload['error_runs']} pending={payload['pending_runs']})"
    )
    console.print(f"  manifest: {render_path(payload['manifest_path'])}", soft_wrap=True)
    console.print(f"  status file: {render_path(payload['status_path'])}", soft_wrap=True)
    for path in payload["table_paths"]:
        console.print(f"  table: {render_path(path)}", soft_wrap=True)
    for path in payload["plot_paths"]:
        console.print(f"  plot: {render_path(path)}", soft_wrap=True)
