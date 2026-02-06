"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/cli/commands/cache.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Literal

import typer
from rich.console import Console
from rich.table import Table

from dnadesign.cruncher.app.catalog_service import catalog_stats, verify_cache
from dnadesign.cruncher.cli.config_resolver import (
    ConfigResolutionError,
    resolve_config_path,
)
from dnadesign.cruncher.config.load import load_config
from dnadesign.cruncher.utils.paths import resolve_catalog_root

app = typer.Typer(no_args_is_help=True, help="Inspect cache stats or verify cached artifacts.")
console = Console()


def _find_generated_cache_dirs(
    root: Path,
    *,
    include_pycache: bool,
    include_pytest_cache: bool,
) -> list[Path]:
    patterns: list[str] = []
    if include_pycache:
        patterns.append("__pycache__")
    if include_pytest_cache:
        patterns.append(".pytest_cache")
    if not patterns:
        return []
    blocked = {".git", ".venv"}
    found: list[Path] = []
    seen: set[Path] = set()
    for pattern in patterns:
        for path in root.rglob(pattern):
            if not path.is_dir():
                continue
            if any(part in blocked for part in path.parts):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            found.append(resolved)
    return sorted(found)


def _package_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _repo_root_from(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / ".git").exists():
            return candidate
    return start


def _resolve_clean_root(
    *,
    config_path: Path,
    scope: Literal["workspace", "package", "repo"],
    root_override: Path | None,
) -> Path:
    if root_override is not None:
        root = root_override.expanduser().resolve()
        if not root.exists() or not root.is_dir():
            raise ValueError(f"--root must point to an existing directory: {root}")
        return root
    if scope == "workspace":
        return config_path.parent.resolve()
    if scope == "package":
        return _package_root()
    return _repo_root_from(_package_root())


@app.command("stats", help="Show counts of cached motifs and site sets.")
def stats(
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
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)
    stats = catalog_stats(catalog_root)
    table = Table(title="Cache stats", header_style="bold")
    table.add_column("Metric")
    table.add_column("Value")
    table.add_row("entries", str(stats["entries"]))
    table.add_row("motifs", str(stats["motifs"]))
    table.add_row("site_sets", str(stats["site_sets"]))
    console.print(table)


@app.command("verify", help="Verify cached motif/site files are present on disk.")
def verify(
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
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    cfg = load_config(config_path)
    catalog_root = resolve_catalog_root(config_path, cfg.catalog.catalog_root)
    issues = verify_cache(catalog_root)
    if not issues:
        console.print("Cache verification OK.")
        return
    console.print("[red]Cache verification failed:[/red]")
    for issue in issues:
        console.print(f"- {issue}")
    raise typer.Exit(code=1)


@app.command("clean", help="Delete generated Python cache directories under the selected scope (dry-run by default).")
def clean(
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
    pycache: bool = typer.Option(True, "--pycache/--no-pycache", help="Include __pycache__ directories."),
    pytest_cache: bool = typer.Option(
        True,
        "--pytest-cache/--no-pytest-cache",
        help="Include .pytest_cache directories.",
    ),
    scope: Literal["workspace", "package", "repo"] = typer.Option(
        "package",
        "--scope",
        help="Scan scope for generated caches.",
    ),
    root: Path | None = typer.Option(
        None,
        "--root",
        help="Explicit scan root directory (overrides --scope).",
    ),
    apply: bool = typer.Option(False, "--apply", help="Delete matched directories."),
) -> None:
    try:
        config_path = resolve_config_path(config_option or config)
    except ConfigResolutionError as exc:
        console.print(str(exc))
        raise typer.Exit(code=1)
    try:
        root = _resolve_clean_root(config_path=config_path, scope=scope, root_override=root)
    except ValueError as exc:
        console.print(f"Error: {exc}")
        raise typer.Exit(code=1)
    matches = _find_generated_cache_dirs(
        root,
        include_pycache=pycache,
        include_pytest_cache=pytest_cache,
    )
    if not matches:
        console.print("No generated cache directories found.")
        return
    table = Table(title="Generated cache directories", header_style="bold")
    table.add_column("Path")
    for path in matches:
        try:
            shown = str(path.relative_to(root))
        except ValueError:
            shown = str(path)
        table.add_row(shown)
    console.print(table)
    if not apply:
        console.print("Dry-run only. Re-run with --apply to delete these directories.")
        return
    removed = 0
    for path in matches:
        shutil.rmtree(path, ignore_errors=False)
        removed += 1
    console.print(f"Deleted {removed} generated cache director{'y' if removed == 1 else 'ies'}.")
