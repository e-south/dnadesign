"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/src/cli/main.py

Typer/Rich CLI entrypoint for DenseGen.

Commands:
  - validate-config : Validate YAML config (schema + sanity).
  - inspect inputs  : Show resolved inputs + Stage-A PWM sampling.
  - inspect plan    : Show resolved per-constraint quota plan.
  - inspect config  : Describe resolved config (inputs/outputs/solver).
  - inspect run     : Summarize run manifest or list workspaces.
  - workspace init  : Scaffold a new workspace with config.yaml + subfolders.
  - stage-a build-pool : Build Stage-A TFBS pools from inputs.
  - stage-b build-libraries : Build Stage-B libraries from pools/inputs.
  - run             : Execute generation pipeline; optionally auto-plot.
  - plot            : Generate plots from outputs using config YAML.
  - ls-plots         : List available plot names and descriptions.
  - notebook        : Generate/run workspace-scoped marimo notebooks.

Run:
  python -m dnadesign.densegen --help

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
import platform
import re
import shlex
import shutil
import sys
import tempfile
from functools import partial
from pathlib import Path
from typing import Callable, Iterator, Optional, Sequence

import typer
from rich.console import Console
from rich.traceback import install as rich_traceback

from ..config import resolve_relative_path, resolve_run_root
from ..core.run_paths import display_path
from ..utils.logging_utils import install_native_stderr_filters
from ..utils.rich_style import make_table
from .context import CliContext
from .setup import (
    DEFAULT_CONFIG_FILENAME,
)
from .setup import (
    ensure_fimo_available as _ensure_fimo_available_impl,
)
from .setup import (
    load_config_or_exit as _load_config_or_exit_impl,
)
from .setup import (
    resolve_config_path as _resolve_config_path_impl,
)
from .setup import (
    resolve_outputs_path_or_exit as _resolve_outputs_path_or_exit_impl,
)
from .stage_a_summary_rows import _format_tier_counts, _stage_a_sampling_rows

_STAGE_A_SUMMARY_EXPORTS = (_format_tier_counts,)


def _build_console() -> Console:
    if getattr(sys.stdout, "isatty", lambda: False)():
        return Console()
    width = shutil.get_terminal_size(fallback=(140, 40)).columns
    return Console(width=int(width))


rich_traceback(show_locals=False)
console = _build_console()
_PYARROW_SYSCTL_PATTERN = re.compile(r"sysctlbyname failed for 'hw\.")
log = logging.getLogger(__name__)
install_native_stderr_filters(suppress_solver_messages=False)

DEFAULT_CONFIG_MISSING_MESSAGE = (
    "No config file found. Pass -c/--config, set DENSEGEN_CONFIG_PATH, "
    "or run from a workspace directory with config.yaml."
)


@contextlib.contextmanager
def _suppress_pyarrow_sysctl_warnings() -> Iterator[None]:
    if platform.system() != "Darwin":
        yield
        return
    try:
        fd = sys.stderr.fileno()
    except Exception:
        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            yield
        text = buf.getvalue()
    else:
        saved = os.dup(fd)
        text = ""
        with tempfile.TemporaryFile(mode="w+b") as tmp:
            os.dup2(tmp.fileno(), fd)
            try:
                yield
            finally:
                os.dup2(saved, fd)
                os.close(saved)
                tmp.seek(0)
                text = tmp.read().decode("utf-8", errors="replace")
    if text:
        lines = [line for line in text.splitlines() if not _PYARROW_SYSCTL_PATTERN.search(line)]
        if lines:
            sys.stderr.write("\n".join(lines) + "\n")


def _candidate_logging_enabled(cfg, *, selected: set[str] | None = None) -> bool:
    for inp in cfg.inputs:
        if selected is not None and inp.name not in selected:
            continue
        sampling = getattr(inp, "sampling", None)
        if sampling is None:
            continue
        if getattr(sampling, "keep_all_candidates_debug", False):
            return True
    return False


def _apply_stage_a_overrides(
    cfg,
    *,
    selected: set[str] | None,
    n_sites: int | None,
    batch_size: int | None,
    max_seconds: float | None,
) -> None:
    if n_sites is None and batch_size is None and max_seconds is None:
        return
    for inp in cfg.inputs:
        if selected is not None and inp.name not in selected:
            continue
        if not str(getattr(inp, "type", "")).startswith("pwm_"):
            continue
        sampling = getattr(inp, "sampling", None)
        if sampling is None:
            continue
        inp.sampling = _with_stage_a_sampling_overrides(
            sampling=sampling,
            n_sites=n_sites,
            batch_size=batch_size,
            max_seconds=max_seconds,
        )
        overrides = getattr(inp, "overrides_by_motif_id", None)
        if isinstance(overrides, dict) and overrides:
            inp.overrides_by_motif_id = {
                motif_id: _with_stage_a_sampling_overrides(
                    sampling=override,
                    n_sites=n_sites,
                    batch_size=batch_size,
                    max_seconds=max_seconds,
                )
                for motif_id, override in overrides.items()
            }


def _with_stage_a_sampling_overrides(
    *,
    sampling,
    n_sites: int | None,
    batch_size: int | None,
    max_seconds: float | None,
):
    updates: dict[str, object] = {}
    if n_sites is not None:
        updates["n_sites"] = n_sites

    mining = sampling.mining
    mining_updates: dict[str, object] = {}
    if batch_size is not None:
        if mining is None:
            raise typer.BadParameter("Stage-A sampling mining config is required for overrides")
        mining_updates["batch_size"] = batch_size
    if max_seconds is not None:
        if mining is None or getattr(mining, "budget", None) is None:
            raise typer.BadParameter("Stage-A sampling mining.budget is required for overrides")
        mining_updates["budget"] = mining.budget.model_copy(update={"max_seconds": max_seconds})

    if mining_updates:
        updates["mining"] = mining.model_copy(update=mining_updates)
    if not updates:
        return sampling
    return sampling.model_copy(update=updates)


def _dense_command_launcher() -> str:
    if os.environ.get("PIXI_PROJECT_MANIFEST"):
        return "pixi run dense"
    return "uv run dense"


def _resolve_workspace_hint_command(command: str) -> str:
    stripped = command.strip()
    if stripped == "dense":
        return _dense_command_launcher()
    if stripped.startswith("dense "):
        tail = stripped[len("dense ") :]
        return f"{_dense_command_launcher()} {tail}"
    return stripped


def _workspace_hint_path(path: Path, *, base: Path) -> str:
    relative_label = display_path(path, base, absolute=False)
    if Path(relative_label).parts[:1] == ("..",):
        return display_path(path, base, absolute=True)
    return relative_label


def _workspace_command(command: str, *, cfg_path: Path | None = None, run_root: Path | None = None) -> str:
    resolved_command = _resolve_workspace_hint_command(command)
    root = run_root
    base = Path.cwd()
    uses_pixi_launcher = resolved_command.startswith("pixi run ")
    has_explicit_config_flag = " -c " in resolved_command or " --config " in resolved_command
    if uses_pixi_launcher and cfg_path is not None and not has_explicit_config_flag:
        cfg_label = _workspace_hint_path(cfg_path, base=base)
        return f"{resolved_command} -c {shlex.quote(cfg_label)}"
    if root is not None:
        try:
            root_resolved = root.resolve()
        except Exception:
            root_resolved = root
        if root_resolved == base.resolve():
            return resolved_command
        candidate = root / DEFAULT_CONFIG_FILENAME
        if candidate.exists():
            root_label = _workspace_hint_path(root, base=base)
            root_absolute_label = display_path(root, base, absolute=True)
            if cfg_path is not None and root_label == root_absolute_label:
                cfg_label = _workspace_hint_path(cfg_path, base=base)
                return f"{resolved_command} -c {shlex.quote(cfg_label)}"
            return f"cd {shlex.quote(root_label)} && {resolved_command}"
    if cfg_path is not None:
        cfg_label = _workspace_hint_path(cfg_path, base=base)
        return f"{resolved_command} -c {shlex.quote(cfg_label)}"
    return resolved_command


# ----------------- schema & helpers -----------------
def _sanitize_filename(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_") or "densegen"


def _infer_input_name(inputs_cfg: list) -> str:
    if not inputs_cfg:
        return "densegen"
    first = inputs_cfg[0]
    name = getattr(first, "name", None)
    if not name:
        if hasattr(first, "path"):
            name = Path(getattr(first, "path")).stem
        elif hasattr(first, "dataset"):
            name = getattr(first, "dataset")
        else:
            name = "densegen"
    return _sanitize_filename(str(name))


def _run_root_for(loaded) -> Path:
    return resolve_run_root(loaded.path, loaded.root.densegen.run.root)


def _count_files(path: Path, pattern: str = "*") -> int:
    if not path.exists() or not path.is_dir():
        return 0
    return sum(1 for p in path.glob(pattern) if p.is_file())


def _short_hash(val: str, *, n: int = 8) -> str:
    if not val:
        return "-"
    return val[:n]


def _display_path(path: Path, run_root: Path, absolute: bool) -> str:
    return display_path(path, run_root, absolute=absolute)


_resolve_config_path = partial(
    _resolve_config_path_impl,
    console=console,
    display_path=_display_path,
)
_load_config_or_exit = partial(
    _load_config_or_exit_impl,
    console=console,
    display_path=_display_path,
)
_resolve_outputs_path_or_exit = partial(
    _resolve_outputs_path_or_exit_impl,
    console=console,
)
_ensure_fimo_available = partial(
    _ensure_fimo_available_impl,
    console=console,
)


def _resolve_workspace_source(
    *,
    source_config: Optional[Path],
    source_workspace: Optional[str],
) -> Iterator[tuple[Path, Path]]:
    from .workspace_sources import resolve_workspace_source

    return resolve_workspace_source(
        source_config=source_config,
        source_workspace=source_workspace,
        console=console,
        display_path=_display_path,
        default_config_filename=DEFAULT_CONFIG_FILENAME,
    )


def _list_dir_entries(path: Path, *, limit: int = 10) -> list[str]:
    if not path.exists() or not path.is_dir():
        return []
    entries = sorted([p.name for p in path.iterdir()])
    if limit > 0:
        entries = entries[:limit]
    return entries


def _collect_missing_input_paths(loaded, cfg_path: Path) -> list[Path]:
    missing: list[Path] = []
    for inp in loaded.root.densegen.inputs:
        if hasattr(inp, "path"):
            raw = getattr(inp, "path")
            if isinstance(raw, str) and raw.strip():
                resolved = resolve_relative_path(cfg_path, raw)
                if not resolved.exists():
                    missing.append(resolved)
        if hasattr(inp, "paths"):
            raw_paths = getattr(inp, "paths")
            if isinstance(raw_paths, list):
                for raw in raw_paths:
                    if isinstance(raw, str) and raw.strip():
                        resolved = resolve_relative_path(cfg_path, raw)
                        if not resolved.exists():
                            missing.append(resolved)
    return missing


def _collect_relative_input_paths_from_raw(dense_cfg: dict) -> list[str]:
    rel_paths: list[str] = []
    inputs_cfg = dense_cfg.get("inputs") or []
    for inp in inputs_cfg:
        if not isinstance(inp, dict):
            continue
        raw_path = inp.get("path")
        if isinstance(raw_path, str) and raw_path.strip():
            if not Path(raw_path).is_absolute():
                rel_paths.append(raw_path)
        raw_paths = inp.get("paths")
        if isinstance(raw_paths, list):
            for path in raw_paths:
                if isinstance(path, str) and path.strip():
                    if not Path(path).is_absolute():
                        rel_paths.append(path)
    return rel_paths


def _render_missing_input_hint(cfg_path: Path, loaded, exc: Exception) -> None:
    console.print(f"[bold red]Input error:[/] {exc}")
    run_root = _run_root_for(loaded)
    missing = _collect_missing_input_paths(loaded, cfg_path)
    if missing:
        console.print("[bold]Missing inputs[/]:")
        for path in missing[:6]:
            console.print(f"  - {_display_path(path, run_root, absolute=False)}")
            parent = path.parent
            siblings = []
            if parent.exists() and parent.is_dir():
                for pattern in ("*.meme", "*.txt", "*.jaspar", "*.csv", "*.parquet", "*.json"):
                    siblings.extend([p.name for p in parent.glob(pattern)])
            siblings = sorted(set(siblings))[:6]
            if siblings:
                console.print(f"    siblings: {', '.join(siblings)}")
    else:
        console.print("[bold]No missing inputs detected in config.[/]")

    hints = []
    if (cfg_path.parent / "inputs").exists():
        workspace_init_cmd = _resolve_workspace_hint_command("dense workspace init --copy-inputs")
        hints.append(f"If this is a staged run dir, use `{workspace_init_cmd}` or copy files into run/inputs.")
    if hints:
        console.print("[bold]Next steps[/]:")
        for hint in hints:
            console.print(f"  - {hint}")


def _render_output_schema_hint(exc: Exception) -> bool:
    msg = str(exc)
    if "Existing Parquet schema does not match the current DenseGen schema" in msg:
        workspace_init_cmd = _resolve_workspace_hint_command("dense workspace init --copy-inputs")
        console.print(f"[bold red]Output schema mismatch:[/] {msg}")
        console.print("[bold]Next steps[/]:")
        console.print("  - Remove outputs/tables/records.parquet and outputs/meta/_densegen_ids.sqlite, or")
        console.print(f"  - Stage a fresh workspace with `{workspace_init_cmd}` and re-run.")
        return True
    if "Output sinks are out of sync before run" in msg:
        console.print(f"[bold red]Output sink mismatch:[/] {msg}")
        console.print("[bold]Next steps[/]:")
        console.print("  - Remove stale outputs so sinks align, or")
        console.print("  - Run with a single output target to rebuild from scratch.")
        return True
    return False


def _warn_pwm_sampling_configs(loaded, cfg_path: Path) -> None:
    warnings: list[str] = []
    for inp in loaded.root.densegen.inputs:
        src_type = getattr(inp, "type", "")
        sampling = getattr(inp, "sampling", None)
        if sampling is None:
            continue
        n_sites = getattr(sampling, "n_sites", None)
        # Width preflight (best-effort)
        if src_type in {"pwm_meme", "pwm_jaspar", "pwm_matrix_csv"} and isinstance(n_sites, int):
            widths = {}
            try:
                if src_type == "pwm_meme":
                    from .adapters.sources.pwm_meme import _parse_meme  # type: ignore

                    path = resolve_relative_path(cfg_path, getattr(inp, "path"))
                    if path.exists():
                        motifs = _parse_meme(path)
                        if getattr(inp, "motif_ids", None):
                            keep = set(getattr(inp, "motif_ids"))
                            motifs = [m for m in motifs if m.motif_id in keep]
                        widths = {m.motif_id: len(m.matrix) for m in motifs}
                elif src_type == "pwm_jaspar":
                    from .adapters.sources.pwm_jaspar import _parse_jaspar  # type: ignore

                    path = resolve_relative_path(cfg_path, getattr(inp, "path"))
                    if path.exists():
                        motifs = _parse_jaspar(path)
                        if getattr(inp, "motif_ids", None):
                            keep = set(getattr(inp, "motif_ids"))
                            motifs = [m for m in motifs if m.motif_id in keep]
                        widths = {m.motif_id: len(m.matrix) for m in motifs}
                elif src_type == "pwm_matrix_csv":
                    import pandas as pd

                    path = resolve_relative_path(cfg_path, getattr(inp, "path"))
                    if path.exists():
                        df = pd.read_csv(path)
                        widths = {getattr(inp, "motif_id", "motif"): len(df)}
            except Exception:
                widths = {}

            for motif_id, width in widths.items():
                if width <= 6 and n_sites > 200:
                    warnings.append(
                        f"{getattr(inp, 'name', src_type)}:{motif_id} width={width} with n_sites={n_sites} "
                        "may fail uniqueness; consider reducing n_sites or using length.policy=range."
                    )
    if warnings:
        console.print("[yellow]Stage-A PWM sampling warnings:[/]")
        for warn in warnings:
            console.print(f"  - {warn}")


def _warn_full_pool_strategy(loaded) -> None:
    sampling = loaded.root.densegen.generation.sampling
    if sampling.pool_strategy != "full":
        return
    ignored = [
        "library_size",
        "library_sampling_strategy",
        "cover_all_regulators",
        "max_sites_per_regulator",
        "relax_on_exhaustion",
    ]
    explicitly_set = set(getattr(sampling, "model_fields_set", set()))
    ignored_set = [name for name in ignored if name in explicitly_set]
    if not ignored_set:
        return
    console.print(
        "[yellow]Warning:[/] pool_strategy=full ignores explicitly set sampling keys: " + ", ".join(ignored_set) + "."
    )


# ----------------- Typer CLI -----------------
app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="DenseGen — Dense Array Generator (Typer/Rich CLI)",
    pretty_exceptions_show_locals=False,
)
inspect_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Inspect configs, inputs, and runs.",
    pretty_exceptions_show_locals=False,
)
stage_a_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Stage-A helpers (input TFBS pools).",
    pretty_exceptions_show_locals=False,
)
stage_b_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Stage-B helpers (library sampling).",
    pretty_exceptions_show_locals=False,
)
workspace_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Workspace scaffolding.",
    pretty_exceptions_show_locals=False,
)
notebook_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Workspace-scoped notebook workflows.",
    pretty_exceptions_show_locals=False,
)

app.add_typer(inspect_app, name="inspect")
app.add_typer(stage_a_app, name="stage-a")
app.add_typer(stage_b_app, name="stage-b")
app.add_typer(workspace_app, name="workspace")
app.add_typer(notebook_app, name="notebook")

cli_context = CliContext(
    console=console,
    make_table=make_table,
    display_path=_display_path,
    resolve_config_path=_resolve_config_path,
    load_config_or_exit=_load_config_or_exit,
    run_root_for=_run_root_for,
    list_dir_entries=_list_dir_entries,
    workspace_command=_workspace_command,
    suppress_pyarrow_sysctl_warnings=_suppress_pyarrow_sysctl_warnings,
    resolve_outputs_path_or_exit=_resolve_outputs_path_or_exit,
    warn_full_pool_strategy=_warn_full_pool_strategy,
    default_config_missing_message=DEFAULT_CONFIG_MISSING_MESSAGE,
)

_REGISTERED_TARGETS: set[str] = set()
_ALL_REGISTRATION_TARGETS: set[str] = {
    "inspect",
    "validate",
    "plots",
    "workspace",
    "stage_a",
    "stage_b",
    "notebook",
    "run",
}
_COMMAND_SCOPE_TARGETS: dict[str, set[str]] = {
    "inspect": {"inspect"},
    "validate-config": {"validate"},
    "plot": {"plots"},
    "ls-plots": {"plots"},
    "workspace": {"workspace"},
    "stage-a": {"stage_a"},
    "stage-b": {"stage_b"},
    "notebook": {"notebook"},
    "run": {"run"},
}


def _command_scope_from_argv(argv: Sequence[str]) -> str | None:
    idx = 0
    while idx < len(argv):
        token = str(argv[idx]).strip()
        if not token:
            idx += 1
            continue
        if token in {"-h", "--help"}:
            return None
        if token in {"-c", "--config"}:
            idx += 2
            continue
        if token.startswith("--config="):
            idx += 1
            continue
        if token.startswith("-"):
            idx += 1
            continue
        return token
    return None


def _registration_targets_for_scope(scope: str | None) -> set[str]:
    if scope is None:
        return set(_ALL_REGISTRATION_TARGETS)
    return set(_COMMAND_SCOPE_TARGETS.get(str(scope), _ALL_REGISTRATION_TARGETS))


def _register_inspect_commands() -> None:
    from .inspect import register_inspect_commands

    register_inspect_commands(inspect_app, context=cli_context)


def _register_validate_command() -> None:
    from .config import register_validate_command

    register_validate_command(
        app,
        context=cli_context,
        warn_pwm_sampling_configs=_warn_pwm_sampling_configs,
        ensure_fimo_available=_ensure_fimo_available,
    )


def _register_plot_commands() -> None:
    from .plots import register_plot_commands

    register_plot_commands(app, context=cli_context)


def _register_workspace_commands() -> None:
    from .workspace import register_workspace_commands

    register_workspace_commands(
        workspace_app,
        context=cli_context,
        resolve_workspace_source=_resolve_workspace_source,
        sanitize_filename=_sanitize_filename,
        collect_relative_input_paths_from_raw=_collect_relative_input_paths_from_raw,
    )


def _register_stage_a_commands() -> None:
    from .stage_a import register_stage_a_commands

    register_stage_a_commands(
        stage_a_app,
        context=cli_context,
        apply_stage_a_overrides=_apply_stage_a_overrides,
        ensure_fimo_available=_ensure_fimo_available,
        candidate_logging_enabled=_candidate_logging_enabled,
        stage_a_sampling_rows=_stage_a_sampling_rows,
    )


def _register_stage_b_commands() -> None:
    from .stage_b import register_stage_b_commands

    register_stage_b_commands(stage_b_app, context=cli_context, short_hash=_short_hash)


def _register_notebook_commands() -> None:
    from .notebook import register_notebook_commands

    register_notebook_commands(notebook_app, context=cli_context)


def _register_run_commands() -> None:
    from .run import register_run_commands

    register_run_commands(
        app,
        context=cli_context,
        render_missing_input_hint=_render_missing_input_hint,
        render_output_schema_hint=_render_output_schema_hint,
        ensure_fimo_available=_ensure_fimo_available,
    )


_REGISTER_TARGET_TO_CALLABLE: dict[str, Callable[[], None]] = {
    "inspect": _register_inspect_commands,
    "validate": _register_validate_command,
    "plots": _register_plot_commands,
    "workspace": _register_workspace_commands,
    "stage_a": _register_stage_a_commands,
    "stage_b": _register_stage_b_commands,
    "notebook": _register_notebook_commands,
    "run": _register_run_commands,
}


def _ensure_commands_registered(scope: str | None) -> None:
    targets = sorted(_registration_targets_for_scope(scope))
    for target in targets:
        if target in _REGISTERED_TARGETS:
            continue
        register = _REGISTER_TARGET_TO_CALLABLE.get(target)
        if register is None:
            raise RuntimeError(f"Unknown DenseGen CLI registration target: {target}")
        register()
        _REGISTERED_TARGETS.add(target)


def _patch_typer_testing_get_command(
    patched_get_command: Callable,
    *,
    typer_testing_module=None,
) -> None:
    testing = typer_testing_module
    if testing is None:
        import typer.testing as testing
    if not hasattr(testing, "_get_command"):
        raise RuntimeError("typer.testing._get_command hook is unavailable.")
    testing._get_command = patched_get_command


def _patch_typer_get_command_for_lazy_registration() -> None:
    patch_flag = "__densegen_lazy_patch__"
    original = typer.main.get_command
    if getattr(original, patch_flag, False):
        return

    def _patched_get_command(typer_instance):
        if typer_instance is app:
            scope = _command_scope_from_argv(sys.argv[1:])
            _ensure_commands_registered(scope=scope)
        return original(typer_instance)

    setattr(_patched_get_command, patch_flag, True)
    typer.main.get_command = _patched_get_command
    _patch_typer_testing_get_command(_patched_get_command)


_patch_typer_get_command_for_lazy_registration()


@app.callback()
def _root(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to config YAML (defaults to ./config.yaml in the current directory).",
    ),
):
    ctx.obj = {"config_path": config}


if __name__ == "__main__":
    app()
