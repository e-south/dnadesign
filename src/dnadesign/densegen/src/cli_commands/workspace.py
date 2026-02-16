"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/cli_commands/workspace.py

Workspace scaffolding CLI command registration.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Callable, Optional

import typer
import yaml

from ..cli_commands.context import CliContext
from ..config import LATEST_SCHEMA_VERSION, resolve_relative_path


def _repo_root_from(start: Path) -> Path | None:
    try:
        cursor = start.resolve()
    except Exception:
        cursor = start
    for root in [cursor, *cursor.parents]:
        if (root / "pyproject.toml").exists() or (root / ".git").exists():
            return root
    return None


def _default_workspace_root() -> Path:
    root, _ = _default_workspace_root_with_source()
    return root


def _default_workspace_root_with_source() -> tuple[Path, str]:
    env_root = os.environ.get("DENSEGEN_WORKSPACE_ROOT")
    if env_root:
        return Path(env_root).expanduser(), "env:DENSEGEN_WORKSPACE_ROOT"
    repo_root = _repo_root_from(Path(__file__).resolve())
    if repo_root is not None:
        return repo_root / "src" / "dnadesign" / "densegen" / "workspaces", "repo-default"
    raise RuntimeError("Unable to determine workspace root. Set DENSEGEN_WORKSPACE_ROOT or pass --root explicitly.")


def _workspace_source_root() -> Path:
    repo_root = _repo_root_from(Path(__file__).resolve())
    if repo_root is not None:
        return repo_root / "src" / "dnadesign" / "densegen" / "workspaces"
    return Path(__file__).resolve().parents[2] / "workspaces"


def _apply_output_mode(output: dict, *, run_id: str, output_mode: str) -> dict:
    mode = str(output_mode).strip().lower()
    if mode not in {"local", "usr", "both"}:
        raise ValueError("output_mode must be one of: local, usr, both.")

    out = dict(output or {})
    parquet_cfg = out.get("parquet")
    if not isinstance(parquet_cfg, dict):
        parquet_cfg = {}
    usr_cfg = out.get("usr")
    if not isinstance(usr_cfg, dict):
        usr_cfg = {}

    if mode in {"local", "both"}:
        parquet_cfg["path"] = "outputs/tables/records.parquet"
        out["parquet"] = parquet_cfg

    if mode in {"usr", "both"}:
        usr_cfg["root"] = "outputs/usr_datasets"
        if not str(usr_cfg.get("dataset", "")).strip():
            usr_cfg["dataset"] = run_id
        usr_cfg.setdefault("chunk_size", 128)
        usr_cfg.setdefault("allow_overwrite", False)
        out["usr"] = usr_cfg

    if mode == "local":
        out["targets"] = ["parquet"]
    elif mode == "usr":
        out["targets"] = ["usr"]
    else:
        out["targets"] = ["parquet", "usr"]
    return out


def _seed_usr_registry(*, run_dir: Path, output: dict, console, display_path: Callable[..., str]) -> None:
    targets = output.get("targets")
    if not isinstance(targets, list) or "usr" not in targets:
        return
    usr_cfg = output.get("usr")
    if not isinstance(usr_cfg, dict):
        return
    root_raw = usr_cfg.get("root")
    if not isinstance(root_raw, str) or not root_raw.strip():
        return
    usr_root = run_dir / Path(root_raw)
    registry_path = usr_root / "registry.yaml"
    if registry_path.exists():
        return
    repo_root = _repo_root_from(Path(__file__).resolve())
    seed_path = None
    if repo_root is not None:
        candidate = repo_root / "src" / "dnadesign" / "usr" / "datasets" / "registry.yaml"
        if candidate.exists() and candidate.is_file():
            seed_path = candidate
    usr_root.mkdir(parents=True, exist_ok=True)
    if seed_path is None:
        console.print(
            "[yellow]USR output selected but no registry seed file was found.[/] "
            "Create outputs/usr_datasets/registry.yaml before running `uv run dense run`."
        )
        return
    shutil.copy2(seed_path, registry_path)
    console.print(
        f":bookmark_tabs: [bold green]Seeded USR registry[/]: {display_path(registry_path, run_dir, absolute=False)}"
    )


def register_workspace_commands(
    app: typer.Typer,
    *,
    context: CliContext,
    resolve_workspace_source: Callable[..., object],
    sanitize_filename: Callable[[str], str],
    collect_relative_input_paths_from_raw: Callable[..., list[str]],
) -> None:
    console = context.console

    @app.command("where", help="Show effective workspace and source-workspace roots.")
    def workspace_where(
        fmt: str = typer.Option(
            "text",
            "--format",
            help="Output format: text or json.",
        ),
    ) -> None:
        try:
            root_path, source = _default_workspace_root_with_source()
        except RuntimeError as exc:
            console.print(f"[bold red]{exc}[/]")
            raise typer.Exit(code=1) from exc
        source_root = _workspace_source_root()
        payload = {
            "workspace_root": str(root_path),
            "workspace_root_source": source,
            "workspace_source_root": str(source_root),
        }
        fmt_norm = str(fmt).strip().lower()
        if fmt_norm == "json":
            typer.echo(json.dumps(payload, separators=(",", ":")))
            return
        if fmt_norm != "text":
            console.print("[bold red]format must be one of: text, json.[/]")
            raise typer.Exit(code=1)
        console.print(f"workspace_root: {payload['workspace_root']}")
        console.print(f"workspace_root_source: {payload['workspace_root_source']}")
        console.print(f"workspace_source_root: {payload['workspace_source_root']}")
        console.print("Tip: set DENSEGEN_WORKSPACE_ROOT to choose a custom workspace root directory.")

    @app.command("init", help="Stage a new workspace with config.yaml and standard subfolders.")
    def workspace_init(
        workspace_id: str = typer.Option(..., "--id", "-i", help="Workspace identifier (directory name)."),
        root: Optional[Path] = typer.Option(
            None,
            "--root",
            help="Workspace root directory (default: DENSEGEN_WORKSPACE_ROOT or workspaces).",
        ),
        source_workspace: Optional[str] = typer.Option(
            None,
            "--from-workspace",
            help="Packaged source workspace id (e.g., demo_tfbs_baseline).",
        ),
        source_config: Optional[Path] = typer.Option(
            None,
            "--from-config",
            help="Source workspace config YAML to copy.",
        ),
        copy_inputs: bool = typer.Option(False, help="Copy file-based inputs into workspace/inputs and rewrite paths."),
        output_mode: str = typer.Option(
            "local",
            "--output-mode",
            help="Output sink mode: local (parquet), usr, or both.",
        ),
    ):
        workspace_id_clean = sanitize_filename(workspace_id)
        if workspace_id_clean != workspace_id:
            console.print(f"[yellow]Sanitized workspace id:[/] {workspace_id} -> {workspace_id_clean}")
        if root is not None:
            root_path = root.expanduser()
        else:
            try:
                root_path = _default_workspace_root()
            except RuntimeError as exc:
                console.print(f"[bold red]{exc}[/]")
                raise typer.Exit(code=1) from exc
        if root_path.exists() and not root_path.is_dir():
            console.print(
                "[bold red]Workspace root is not a directory:[/] "
                f"{context.display_path(root_path, Path.cwd(), absolute=False)}"
            )
            raise typer.Exit(code=1)
        workspace_dir = (root_path / workspace_id_clean).resolve()
        if workspace_dir.exists():
            console.print(
                "[bold red]Workspace directory already exists:[/] "
                f"{context.display_path(workspace_dir, root_path.resolve(), absolute=False)}"
            )
            console.print("[yellow]Choose a new --id or remove the existing workspace directory, then retry.[/]")
            raise typer.Exit(code=1)

        with resolve_workspace_source(
            source_config=source_config,
            source_workspace=source_workspace,
        ) as (_source_dir, source_path):
            workspace_dir.mkdir(parents=True, exist_ok=False)
            (workspace_dir / "inputs").mkdir(parents=True, exist_ok=True)
            (workspace_dir / "outputs" / "logs").mkdir(parents=True, exist_ok=True)
            (workspace_dir / "outputs" / "meta").mkdir(parents=True, exist_ok=True)
            (workspace_dir / "outputs" / "pools").mkdir(parents=True, exist_ok=True)
            (workspace_dir / "outputs" / "libraries").mkdir(parents=True, exist_ok=True)
            (workspace_dir / "outputs" / "tables").mkdir(parents=True, exist_ok=True)
            (workspace_dir / "outputs" / "plots").mkdir(parents=True, exist_ok=True)

            raw = yaml.safe_load(source_path.read_text())
            if not isinstance(raw, dict):
                console.print("[bold red]Source config must be a YAML mapping.[/]")
                raise typer.Exit(code=1)

            dense = raw.setdefault("densegen", {})
            dense["schema_version"] = LATEST_SCHEMA_VERSION
            run_block = dense.get("run") or {}
            run_block["id"] = workspace_id_clean
            run_block["root"] = "."
            dense["run"] = run_block

            output = dense.get("output") or {}
            if not isinstance(output, dict):
                console.print("[bold red]Template output block must be a mapping.[/]")
                raise typer.Exit(code=1)
            try:
                output = _apply_output_mode(output, run_id=workspace_id_clean, output_mode=output_mode)
            except ValueError as exc:
                console.print(f"[bold red]{exc}[/]")
                raise typer.Exit(code=1)
            dense["output"] = output

            logging_cfg = dense.get("logging") or {}
            logging_cfg["log_dir"] = "outputs/logs"
            dense["logging"] = logging_cfg
            _seed_usr_registry(
                run_dir=workspace_dir,
                output=output,
                console=console,
                display_path=context.display_path,
            )

            if "plots" in raw and isinstance(raw.get("plots"), dict):
                raw["plots"]["out_dir"] = "outputs/plots"

            if copy_inputs:
                inputs_cfg = dense.get("inputs") or []
                for inp in inputs_cfg:
                    if not isinstance(inp, dict):
                        continue
                    if "path" in inp:
                        src = resolve_relative_path(source_path, inp["path"])
                        if not src.exists() or not src.is_file():
                            console.print(f"[bold red]Input file not found:[/] {src}")
                            raise typer.Exit(code=1)
                        dest = workspace_dir / "inputs" / src.name
                        if dest.exists():
                            console.print(f"[bold red]Input file already exists:[/] {dest}")
                            raise typer.Exit(code=1)
                        shutil.copy2(src, dest)
                        inp["path"] = str(Path("inputs") / src.name)
                    if "paths" in inp and isinstance(inp["paths"], list):
                        new_paths: list[str] = []
                        for path in inp["paths"]:
                            src = resolve_relative_path(source_path, path)
                            if not src.exists() or not src.is_file():
                                console.print(f"[bold red]Input file not found:[/] {src}")
                                raise typer.Exit(code=1)
                            dest = workspace_dir / "inputs" / src.name
                            if dest.exists():
                                console.print(f"[bold red]Input file already exists:[/] {dest}")
                                raise typer.Exit(code=1)
                            shutil.copy2(src, dest)
                            new_paths.append(str(Path("inputs") / src.name))
                        inp["paths"] = new_paths

            # Intentionally avoid copying auxiliary tools into the DenseGen workspace
            # to keep the workspace config-centric and low-cognitive-load.

            config_path = workspace_dir / "config.yaml"
            config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
            if not copy_inputs:
                rel_paths = collect_relative_input_paths_from_raw(dense)
                if rel_paths:
                    console.print(
                        "[yellow]Workspace uses file-based inputs with relative paths.[/]"
                        " They will resolve relative to the new workspace."
                    )
                    for rel_path in rel_paths[:6]:
                        console.print(f"  - {rel_path}")
                    console.print("[yellow]Tip[/]: re-run with --copy-inputs or update paths in config.yaml.")
            console.print(
                ":sparkles: [bold green]Workspace staged[/]: "
                f"{context.display_path(config_path, workspace_dir, absolute=False)}"
            )
