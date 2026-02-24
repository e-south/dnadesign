"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/workspaces.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from dnadesign.cruncher.cli.config_resolver import (
    WorkspaceCandidate,
    discover_workspaces,
    resolve_config_path,
    resolve_invocation_cwd,
    workspace_search_roots,
)
from dnadesign.cruncher.cli.paths import render_path
from dnadesign.cruncher.study.discovery import discover_study_runs_for_workspace, discover_study_specs_for_workspace
from dnadesign.cruncher.utils.paths import resolve_workspace_root

app = typer.Typer(no_args_is_help=True, help="List discoverable cruncher workspaces.")
console = Console()


def _discover_transient_paths(
    root: Path,
    *,
    include_catalog_cache: bool = False,
) -> tuple[list[Path], list[Path]]:
    cache_dirs: list[Path] = []
    transient_files: list[Path] = []
    for path in root.rglob("*"):
        if not path.exists():
            continue
        if path.is_dir():
            if path.name == "__pycache__":
                cache_dirs.append(path)
                continue
            if include_catalog_cache and path.name == ".cruncher":
                cache_dirs.append(path)
            continue
        if path.is_file() and (path.name == ".DS_Store" or path.suffix == ".pyc"):
            transient_files.append(path)
    cache_dirs.sort()
    transient_files.sort()
    return cache_dirs, transient_files


def run_workspace_runbook(*args, **kwargs):
    from dnadesign.cruncher.workspaces.runbook import run_workspace_runbook as _run_workspace_runbook

    return _run_workspace_runbook(*args, **kwargs)


def _resolve_cli_path(path: Path) -> Path:
    expanded = path.expanduser()
    if expanded.is_absolute():
        return expanded.resolve()
    return (resolve_invocation_cwd() / expanded).resolve()


def _candidate_runbooks_in_dir(directory: Path) -> list[Path]:
    candidates: list[Path] = []
    workspace_runbook = directory / "configs" / "runbook.yaml"
    if workspace_runbook.is_file():
        candidates.append(workspace_runbook)
    if directory.name == "configs":
        direct = directory / "runbook.yaml"
        if direct.is_file():
            candidates.append(direct)
    return candidates


def _candidate_configs_in_dir(directory: Path) -> list[Path]:
    candidates: list[Path] = []
    workspace_config = directory / "configs" / "config.yaml"
    if workspace_config.is_file():
        candidates.append(workspace_config)
    if directory.name == "configs":
        direct = directory / "config.yaml"
        if direct.is_file():
            candidates.append(direct)
    return candidates


def _runbook_from_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        if resolved.name != "runbook.yaml" or resolved.parent.name != "configs":
            raise ValueError("--workspace file must be <workspace>/configs/runbook.yaml")
        return resolved
    if not resolved.is_dir():
        raise ValueError(f"--workspace path is neither a file nor a directory: {resolved}")

    direct = resolved / "configs" / "runbook.yaml"
    if direct.is_file():
        return direct.resolve()
    if resolved.name == "configs" and (resolved / "runbook.yaml").is_file():
        return (resolved / "runbook.yaml").resolve()
    raise ValueError("--workspace directory has no runbook file at <workspace>/configs/runbook.yaml")


def _discover_workspace_runbooks(cwd: Path | None = None) -> list[tuple[str, Path, Path]]:
    cwd_path = resolve_invocation_cwd(cwd)
    candidates: list[tuple[str, Path, Path]] = []
    seen_runbooks: set[Path] = set()
    for root in workspace_search_roots(cwd_path):
        if not root.is_dir():
            continue
        entries: list[Path] = [root]
        entries.extend(sorted(item for item in root.iterdir() if item.is_dir()))
        for entry in entries:
            if not entry.is_dir():
                continue
            runbooks = _candidate_runbooks_in_dir(entry)
            if len(runbooks) != 1:
                continue
            runbook_path = runbooks[0].resolve()
            if runbook_path in seen_runbooks:
                continue
            seen_runbooks.add(runbook_path)
            if runbook_path.parent.name == "configs":
                workspace_root = runbook_path.parent.parent.resolve()
            else:
                workspace_root = entry.resolve()
            candidates.append((workspace_root.name, workspace_root, runbook_path))
    candidates.sort(key=lambda item: (item[0], str(item[2])))
    return candidates


def _discover_workspaces_in_root(root: Path) -> list[WorkspaceCandidate]:
    resolved_root = root.expanduser().resolve()
    if not resolved_root.is_dir():
        return []
    candidates: list[WorkspaceCandidate] = []
    seen_configs: set[Path] = set()
    entries: list[Path] = [resolved_root]
    entries.extend(sorted(item for item in resolved_root.iterdir() if item.is_dir()))
    for entry in entries:
        if not entry.is_dir():
            continue
        configs = _candidate_configs_in_dir(entry)
        if len(configs) != 1:
            continue
        config_path = configs[0].resolve()
        if config_path in seen_configs:
            continue
        seen_configs.add(config_path)
        if config_path.parent.name == "configs":
            workspace_root = config_path.parent.parent.resolve()
        else:
            workspace_root = entry.resolve()
        candidates.append(
            WorkspaceCandidate(
                name=workspace_root.name,
                root=workspace_root,
                config_path=config_path,
                catalog_path=(workspace_root / ".cruncher" / "catalog.json").resolve(),
            )
        )
    candidates.sort(key=lambda item: (item.name, str(item.config_path)))
    return candidates


def _discover_workspace_runbooks_in_root(root: Path) -> list[tuple[str, Path, Path]]:
    resolved_root = root.expanduser().resolve()
    if not resolved_root.is_dir():
        return []
    candidates: list[tuple[str, Path, Path]] = []
    seen_runbooks: set[Path] = set()
    entries: list[Path] = [resolved_root]
    entries.extend(sorted(item for item in resolved_root.iterdir() if item.is_dir()))
    for entry in entries:
        if not entry.is_dir():
            continue
        runbooks = _candidate_runbooks_in_dir(entry)
        if len(runbooks) != 1:
            continue
        runbook_path = runbooks[0].resolve()
        if runbook_path in seen_runbooks:
            continue
        seen_runbooks.add(runbook_path)
        workspace_root = runbook_path.parent.parent.resolve()
        candidates.append((workspace_root.name, workspace_root, runbook_path))
    candidates.sort(key=lambda item: (item[0], str(item[2])))
    return candidates


def _resolve_workspace_runbook(selector: str, candidates: list[tuple[str, Path, Path]]) -> Path:
    token = str(selector).strip()
    if not token:
        raise ValueError("--workspace must be non-empty.")
    if token.isdigit():
        idx = int(token)
        if idx < 1 or idx > len(candidates):
            raise ValueError(f"--workspace index {idx} is out of range (1..{len(candidates)}).")
        return candidates[idx - 1][2]

    named = [item for item in candidates if item[0] == token]
    if len(named) == 1:
        return named[0][2]
    if len(named) > 1:
        rendered = ", ".join(str(item[1]) for item in named)
        raise ValueError(f"--workspace '{token}' is ambiguous. Matches: {rendered}")

    path = Path(token).expanduser()
    resolved = _resolve_cli_path(path)
    if resolved.exists():
        for _, workspace_root, runbook_path in candidates:
            if workspace_root == resolved or runbook_path == resolved:
                return runbook_path
        return _runbook_from_path(resolved)

    if not candidates:
        raise ValueError(
            f"--workspace '{token}' was not found and no discoverable workspaces exist. "
            "Pass an existing workspace path or set CRUNCHER_WORKSPACE_ROOTS."
        )
    available = ", ".join(item[0] for item in candidates)
    raise ValueError(f"--workspace '{token}' was not found. Available workspaces: {available}")


def _resolve_runbook_from_cwd_or_parents(cwd: Path) -> Path | None:
    for directory in (cwd, *cwd.parents):
        runbooks = _candidate_runbooks_in_dir(directory)
        if len(runbooks) == 1:
            return runbooks[0].resolve()
        if len(runbooks) > 1:
            rendered = ", ".join(str(path.resolve()) for path in runbooks)
            raise ValueError(f"Multiple runbooks found under {directory}: {rendered}")
    return None


def _is_workspace_layout(root: Path) -> bool:
    configs = root / "configs"
    if not configs.is_dir():
        return False
    return (configs / "config.yaml").is_file() or (configs / "runbook.yaml").is_file()


def _discover_reset_paths(root: Path) -> tuple[list[Path], list[Path]]:
    dirs: list[Path] = []
    files: list[Path] = []
    for name in ("outputs", ".cruncher"):
        candidate = root / name
        if candidate.is_dir():
            dirs.append(candidate)
        elif candidate.is_file():
            files.append(candidate)

    transient_dirs, transient_files = _discover_transient_paths(root, include_catalog_cache=False)
    dirs.extend(transient_dirs)
    files.extend(transient_files)

    dedup_dirs = sorted(
        set(path.resolve() for path in dirs),
        key=lambda path: (len(path.parts), str(path)),
        reverse=True,
    )
    dedup_files = sorted(set(path.resolve() for path in files), key=str)
    return dedup_dirs, dedup_files


def _discover_workspace_roots(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    roots: list[Path] = []
    for child in sorted(item.resolve() for item in root.iterdir() if item.is_dir()):
        if _is_workspace_layout(child):
            roots.append(child)
    return roots


def _workspace_from_path(path: Path) -> WorkspaceCandidate:
    resolved = path.expanduser().resolve()
    if resolved.is_file():
        if resolved.name != "config.yaml" or resolved.parent.name != "configs":
            raise ValueError("--workspace file must be <workspace>/configs/config.yaml")
        root = resolved.parent.parent
        config_path = resolved
    elif resolved.is_dir():
        direct = resolved / "configs" / "config.yaml"
        if direct.is_file():
            root = resolved
            config_path = direct.resolve()
        elif resolved.name == "configs" and (resolved / "config.yaml").is_file():
            root = resolved.parent
            config_path = (resolved / "config.yaml").resolve()
        else:
            raise ValueError("--workspace directory has no config file at <workspace>/configs/config.yaml")
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


@app.command("list", help="List discoverable workspaces and their configs.")
def list_workspaces(
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Optional workspace discovery root. Defaults to CRUNCHER_WORKSPACE_ROOTS/git-derived roots.",
    ),
) -> None:
    try:
        if root is None:
            config_workspaces = discover_workspaces()
            runbook_workspaces = _discover_workspace_runbooks()
            roots = workspace_search_roots()
            roots_rendered = "\n".join(f"- {root_path}" for root_path in roots) if roots else "- (none)"
        else:
            resolved_root = _resolve_cli_path(root)
            if not resolved_root.exists():
                raise FileNotFoundError(f"--root does not exist: {resolved_root}")
            if not resolved_root.is_dir():
                raise ValueError(f"--root must be a directory: {resolved_root}")
            config_workspaces = _discover_workspaces_in_root(resolved_root)
            runbook_workspaces = _discover_workspace_runbooks_in_root(resolved_root)
            roots_rendered = f"- {resolved_root}"
        if not config_workspaces and not runbook_workspaces:
            console.print("No workspaces discovered.")
            console.print("Workspace discovery searched:")
            console.print(roots_rendered)
            console.print(f"Hint: set CRUNCHER_WORKSPACE_ROOTS=/path/a{os.pathsep}/path/b to add roots.")
            raise typer.Exit(code=1)

        config_by_root = {item.root.resolve(): item for item in config_workspaces}
        runbook_by_root = {root.resolve(): runbook for _, root, runbook in runbook_workspaces}
        all_roots = sorted(set(config_by_root) | set(runbook_by_root), key=lambda path: path.name)

        table = Table(title="Workspaces", header_style="bold")
        table.add_column("Index", justify="right")
        table.add_column("Name")
        table.add_column("Kind")
        table.add_column("Config")
        table.add_column("Runbook")
        table.add_column("Root")
        table.add_column("Catalog (.cruncher)")
        table.add_column("Study Specs", justify="right")
        table.add_column("Study Runs", justify="right")

        for idx, workspace_root in enumerate(all_roots, start=1):
            config_workspace = config_by_root.get(workspace_root)
            runbook_path = runbook_by_root.get(workspace_root)
            has_config = config_workspace is not None
            has_runbook = runbook_path is not None
            if has_config and has_runbook:
                kind = "config+runbook"
            elif has_runbook:
                kind = "runbook-only"
            else:
                kind = "config-only"

            catalog_path = workspace_root / ".cruncher" / "catalog.json"
            catalog_flag = "yes" if catalog_path.exists() else "no"
            spec_rows = discover_study_specs_for_workspace(
                workspace_name=workspace_root.name,
                workspace_root=workspace_root,
            )
            run_rows = discover_study_runs_for_workspace(spec_rows)
            table.add_row(
                str(idx),
                workspace_root.name,
                kind,
                render_path(config_workspace.config_path) if config_workspace is not None else "-",
                render_path(runbook_path) if runbook_path is not None else "-",
                render_path(workspace_root),
                catalog_flag,
                str(len(spec_rows)),
                str(len(run_rows)),
            )
        console.print(table)
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc


@app.command("run", help="Execute a machine runbook (`configs/runbook.yaml`) in fail-fast order.")
def run_cmd(
    workspace: str | None = typer.Option(
        None,
        "--workspace",
        help="Workspace selector (name, index, or path). Optional when run from inside a workspace.",
    ),
    runbook: Path | None = typer.Option(
        None,
        "--runbook",
        help="Path to a runbook YAML file. Defaults to <workspace>/configs/runbook.yaml.",
    ),
    step: list[str] | None = typer.Option(
        None,
        "--step",
        help="Optional runbook step id(s) to run. Preserves runbook order.",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Validate and print resolved execution without running commands.",
    ),
) -> None:
    try:
        if runbook is not None:
            runbook_value = runbook.expanduser()
            if workspace is not None and not runbook_value.is_absolute():
                selected_runbook = _resolve_workspace_runbook(workspace, _discover_workspace_runbooks())
                workspace_root = selected_runbook.parent.parent
                runbook_path = (workspace_root / runbook_value).resolve()
                try:
                    runbook_path.relative_to(workspace_root.resolve())
                except ValueError as exc:
                    raise ValueError(
                        "Relative --runbook must resolve inside the selected workspace: "
                        f"workspace={workspace_root} runbook={runbook_path}"
                    ) from exc
            else:
                runbook_path = _resolve_cli_path(runbook)
        elif workspace is not None:
            runbook_path = _resolve_workspace_runbook(workspace, _discover_workspace_runbooks())
        else:
            cwd = resolve_invocation_cwd()
            runbook_path = _resolve_runbook_from_cwd_or_parents(cwd)
            if runbook_path is None:
                config_path = resolve_config_path(None, cwd=cwd, log=False)
                runbook_path = (resolve_workspace_root(config_path) / "configs" / "runbook.yaml").resolve()

        if not runbook_path.exists():
            raise FileNotFoundError(f"Workspace runbook not found: {runbook_path}")

        result = run_workspace_runbook(
            runbook_path,
            step_ids=step or None,
            dry_run=dry_run,
        )
        mode = "dry-run validated" if dry_run else "executed"
        console.print(f"Runbook {mode}: {render_path(result.runbook_path)}")
        console.print(f"Workspace root: {render_path(result.workspace_root)}")
        console.print(f"Steps: {', '.join(result.executed_step_ids)}")
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc


@app.command("reset", help="Reset generated workspace state (preserves inputs/config plans; dry-run by default).")
def reset(
    root: Path = typer.Option(
        Path("."),
        "--root",
        help="Workspace root to reset.",
    ),
    all_workspaces: bool = typer.Option(
        False,
        "--all-workspaces",
        help="Treat --root as a parent directory and reset every discovered workspace under it.",
    ),
    confirm: bool = typer.Option(
        False,
        "--confirm",
        help="Execute reset deletions. Without this flag, command is dry-run.",
    ),
) -> None:
    try:
        resolved_root = root.expanduser().resolve()
        if not resolved_root.exists():
            raise FileNotFoundError(f"Workspace root does not exist: {resolved_root}")
        if not resolved_root.is_dir():
            raise ValueError(f"Workspace root must be a directory: {resolved_root}")

        workspace_roots: list[Path]
        if all_workspaces:
            workspace_roots = _discover_workspace_roots(resolved_root)
            if not workspace_roots:
                raise ValueError(
                    f"No workspace roots found under {resolved_root}. "
                    "Expected child directories containing configs/runbook.yaml or configs/config.yaml."
                )
            console.print(f"Workspace reset root set: {render_path(resolved_root)}")
            console.print(f"Discovered {len(workspace_roots)} workspace roots to reset.")
            for workspace_root in workspace_roots:
                console.print(f"  workspace: {render_path(workspace_root)}")
        else:
            if not _is_workspace_layout(resolved_root):
                raise ValueError(
                    f"Workspace root must contain configs/runbook.yaml or configs/config.yaml: {resolved_root}"
                )
            workspace_roots = [resolved_root]
            console.print(f"Workspace reset root: {render_path(resolved_root)}")

        all_delete_dirs: list[Path] = []
        all_delete_files: list[Path] = []
        for workspace_root in workspace_roots:
            delete_dirs, delete_files = _discover_reset_paths(workspace_root)
            all_delete_dirs.extend(delete_dirs)
            all_delete_files.extend(delete_files)
        delete_dirs = sorted(
            set(path.resolve() for path in all_delete_dirs),
            key=lambda path: (len(path.parts), str(path)),
            reverse=True,
        )
        delete_files = sorted(set(path.resolve() for path in all_delete_files), key=str)
        total_targets = len(delete_dirs) + len(delete_files)
        console.print(
            "Reset scope preserves workspace inputs and configs "
            "(for example inputs/, configs/, runbook.md) and removes generated state."
        )
        console.print(
            f"Found {len(delete_dirs)} directories and {len(delete_files)} files to remove ({total_targets} total)."
        )
        for path in delete_dirs:
            console.print(f"  dir: {render_path(path)}")
        for path in delete_files:
            console.print(f"  file: {render_path(path)}")

        if not confirm:
            console.print("Dry run only. Re-run with --confirm to reset workspace state.")
            return

        for path in delete_dirs:
            if path.exists():
                shutil.rmtree(path)
        for path in delete_files:
            if path.exists():
                path.unlink()
        console.print(f"Reset complete. Removed {total_targets} generated artifacts.")
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc


@app.command("clean-transient", help="Remove transient cache files under a workspace root (dry-run by default).")
def clean_transient(
    root: Path = typer.Option(
        Path("."),
        "--root",
        help="Workspace root to scan for transient files.",
    ),
    confirm: bool = typer.Option(
        False,
        "--confirm",
        help="Execute deletions. Without this flag, command is dry-run.",
    ),
    include_catalog_cache: bool = typer.Option(
        False,
        "--include-catalog-cache",
        help="Also include .cruncher catalog cache directories in deletion candidates.",
    ),
) -> None:
    try:
        resolved_root = root.expanduser().resolve()
        if not resolved_root.exists():
            raise FileNotFoundError(f"Workspace root does not exist: {resolved_root}")
        if not resolved_root.is_dir():
            raise ValueError(f"Workspace root must be a directory: {resolved_root}")

        cache_dirs, transient_files = _discover_transient_paths(
            resolved_root,
            include_catalog_cache=include_catalog_cache,
        )
        total_targets = len(cache_dirs) + len(transient_files)
        console.print(f"Transient scan root: {render_path(resolved_root)}")
        console.print(
            f"Found {len(cache_dirs)} __pycache__ directories and {len(transient_files)} transient files "
            f"({total_targets} total)."
        )
        for path in cache_dirs:
            console.print(f"  dir: {render_path(path)}")
        for path in transient_files:
            console.print(f"  file: {render_path(path)}")

        if not confirm:
            console.print("Dry run only. Re-run with --confirm to delete transient artifacts.")
            return

        for path in cache_dirs:
            if path.exists():
                shutil.rmtree(path)
        for path in transient_files:
            if path.exists():
                path.unlink()
        console.print(f"Deleted {total_targets} transient artifacts.")
    except (ValueError, FileNotFoundError) as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1) from exc
