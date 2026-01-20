"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/cli.py

Typer/Rich CLI entrypoint for DenseGen.

Commands:
  - validate : Validate YAML config (schema + sanity).
  - plan     : Show resolved per-constraint quota plan.
  - stage    : Scaffold a new workspace with config.yaml + subfolders.
  - run      : Execute generation pipeline; optionally auto-plot.
  - plot     : Generate plots from outputs using config YAML.
  - ls-plots : List available plot names and descriptions.
  - summarize : Print an outputs/meta/run_manifest.json summary table.
  - report   : Generate audit-grade report tables for a run.

Run:
  python -m dnadesign.densegen.src.cli --help

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import contextlib
import io
import os
import platform
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterator, Optional

import pandas as pd
import typer
import yaml
from rich.console import Console
from rich.table import Table
from rich.traceback import install as rich_traceback

from .config import (
    LATEST_SCHEMA_VERSION,
    ConfigError,
    load_config,
    resolve_relative_path,
    resolve_run_root,
    resolve_run_scoped_path,
)
from .core.pipeline import resolve_plan, run_pipeline
from .core.reporting import collect_report_data, write_report
from .core.run_manifest import load_run_manifest
from .core.run_paths import run_manifest_path, run_state_path
from .core.run_state import load_run_state
from .utils.logging_utils import install_native_stderr_filters, setup_logging

rich_traceback(show_locals=False)
console = Console()
_PYARROW_SYSCTL_PATTERN = re.compile(r"sysctlbyname failed for 'hw\.")


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


# ----------------- local path helpers -----------------
def _densegen_root_from(file_path: Path) -> Path:
    return file_path.resolve().parent.parent


DENSEGEN_ROOT = _densegen_root_from(Path(__file__))
DEFAULT_WORKSPACES_ROOT = DENSEGEN_ROOT / "workspaces"


def _default_config_path() -> Path:
    # Prefer a realistic, self-contained MEME demo config inside the package tree.
    return DENSEGEN_ROOT / "workspaces" / "demo_meme_two_tf" / "config.yaml"


def _default_template_path() -> Path:
    return DENSEGEN_ROOT / "workspaces" / "demo_meme_two_tf" / "config.yaml"


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


def _resolve_config_path(ctx: typer.Context, override: Optional[Path]) -> Path:
    if override is not None:
        return override
    if ctx.obj and "config_path" in ctx.obj:
        return Path(ctx.obj["config_path"])
    return _default_config_path()


def _load_config_or_exit(cfg_path: Path):
    try:
        return load_config(cfg_path)
    except FileNotFoundError:
        console.print(f"[bold red]Config file not found:[/] {cfg_path}")
        raise typer.Exit(code=1)
    except ConfigError as e:
        console.print(f"[bold red]Config error:[/] {e}")
        raise typer.Exit(code=1)


def _run_root_for(loaded) -> Path:
    return resolve_run_root(loaded.path, loaded.root.densegen.run.root)


def _count_files(path: Path, pattern: str = "*") -> int:
    if not path.exists() or not path.is_dir():
        return 0
    return sum(1 for p in path.glob(pattern) if p.is_file())


def _ensure_mpl_cache_dir() -> None:
    if os.environ.get("MPLCONFIGDIR"):
        return
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    target = cache_root / "densegen" / "matplotlib"
    try:
        target.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(target)
    except Exception:
        tmp = Path(os.getenv("TMPDIR") or "/tmp") / "densegen-matplotlib"
        tmp.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(tmp)


def _short_hash(val: str, *, n: int = 8) -> str:
    if not val:
        return "-"
    return val[:n]


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


def _render_missing_input_hint(cfg_path: Path, loaded, exc: Exception) -> None:
    console.print(f"[bold red]Input error:[/] {exc}")
    missing = _collect_missing_input_paths(loaded, cfg_path)
    if missing:
        console.print("[bold]Missing inputs[/]:")
        for path in missing[:6]:
            console.print(f"  - {path}")
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
        hints.append("If this is a staged run dir, use `dense stage --copy-inputs` or copy files into run/inputs.")
    missing_str = " ".join(str(p) for p in missing)
    demo_paths = (
        "cruncher/workspaces/demo_basics_two_tf",
        "cruncher/workspaces/demo_campaigns_multi_tf",
    )
    if any(path in missing_str for path in demo_paths):
        hints.append(
            "To regenerate Cruncher demo motifs: "
            "cruncher fetch motifs --source demo_local_meme --tf lexA --tf cpxR --update -c <CONFIG>; "
            "cruncher lock -c <CONFIG>; cruncher parse -c <CONFIG>; cruncher sample --no-auto-opt -c <CONFIG>"
        )
    if hints:
        console.print("[bold]Next steps[/]:")
        for hint in hints:
            console.print(f"  - {hint}")


def _warn_pwm_sampling_configs(loaded, cfg_path: Path) -> None:
    warnings: list[str] = []
    for inp in loaded.root.densegen.inputs:
        src_type = getattr(inp, "type", "")
        sampling = getattr(inp, "sampling", None)
        if sampling is None:
            continue
        scoring_backend = getattr(sampling, "scoring_backend", "densegen")
        n_sites = getattr(sampling, "n_sites", None)
        oversample = getattr(sampling, "oversample_factor", None)
        max_candidates = getattr(sampling, "max_candidates", None)
        score_threshold = getattr(sampling, "score_threshold", None)
        score_percentile = getattr(sampling, "score_percentile", None)
        if scoring_backend == "fimo" and (score_threshold is not None or score_percentile is not None):
            warnings.append(
                f"{getattr(inp, 'name', src_type)}: scoring_backend=fimo ignores score_threshold/score_percentile."
            )
        if isinstance(n_sites, int) and isinstance(oversample, int) and max_candidates is not None:
            requested = n_sites * oversample
            if requested > int(max_candidates):
                warnings.append(
                    f"{getattr(inp, 'name', src_type)}: requested candidates ({requested}) exceeds max_candidates "
                    f"({max_candidates}); sampling will be capped."
                )
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
                        "may fail uniqueness; consider reducing n_sites or using length_policy=range."
                    )
    if warnings:
        console.print("[yellow]PWM sampling warnings:[/]")
        for warn in warnings:
            console.print(f"  - {warn}")


def _warn_full_pool_strategy(loaded) -> None:
    sampling = loaded.root.densegen.generation.sampling
    if sampling.pool_strategy != "full":
        return
    ignored = [
        "library_size",
        "subsample_over_length_budget_by",
        "library_sampling_strategy",
        "cover_all_regulators",
        "max_sites_per_regulator",
        "relax_on_exhaustion",
        "allow_incomplete_coverage",
        "iterative_max_libraries",
        "iterative_min_new_solutions",
    ]
    console.print(
        "[yellow]Warning:[/] pool_strategy=full uses the entire input library; " + ", ".join(ignored) + " are ignored."
    )


def _list_workspaces_table(workspaces_root: Path, *, limit: int, show_all: bool) -> Table:
    workspace_dirs = sorted([p for p in workspaces_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if limit and limit > 0:
        workspace_dirs = workspace_dirs[: int(limit)]

    table = Table("workspace", "id", "config", "parquet", "plots", "logs", "status")
    for run_dir in workspace_dirs:
        cfg_path = run_dir / "config.yaml"
        if not show_all and not cfg_path.exists():
            continue

        run_id = run_dir.name
        status = "ok"
        parquet_count = "-"
        plots_count = _count_files(run_dir / "outputs", pattern="*")
        logs_count = _count_files(run_dir / "logs", pattern="*")

        if cfg_path.exists():
            try:
                loaded = load_config(cfg_path)
                run_id = loaded.root.densegen.run.id
                run_root = resolve_run_root(cfg_path, loaded.root.densegen.run.root)
                if loaded.root.densegen.output.parquet is not None:
                    pq_dir = resolve_run_scoped_path(
                        cfg_path,
                        run_root,
                        loaded.root.densegen.output.parquet.path,
                        label="output.parquet.path",
                    )
                    parquet_count = _count_files(pq_dir, pattern="*.parquet")
                plots_count = _count_files(
                    resolve_run_scoped_path(
                        cfg_path,
                        run_root,
                        loaded.root.plots.out_dir if loaded.root.plots else "outputs",
                        label="plots.out_dir",
                    ),
                    pattern="*",
                )
                logs_count = _count_files(
                    resolve_run_scoped_path(
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

        table.add_row(
            run_dir.name,
            run_id,
            str(cfg_path) if cfg_path.exists() else "-",
            str(parquet_count),
            str(plots_count),
            str(logs_count),
            status,
        )
    return table


# ----------------- Typer CLI -----------------
app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="DenseGen — Dense Array Generator (Typer/Rich CLI)",
)


@app.callback()
def _root(
    ctx: typer.Context,
    config: Path = typer.Option(
        _default_config_path(),
        "--config",
        "-c",
        help="Path to config YAML (can also be passed per command).",
    ),
):
    ctx.obj = {"config_path": config}


@app.command(help="Validate the config YAML (schema + sanity).")
def validate(
    ctx: typer.Context,
    probe_solver: bool = typer.Option(False, help="Also probe the solver backend."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
):
    cfg_path = _resolve_config_path(ctx, config)
    loaded = _load_config_or_exit(cfg_path)
    _warn_pwm_sampling_configs(loaded, cfg_path)
    _warn_full_pool_strategy(loaded)
    if probe_solver:
        from .adapters.optimizer import DenseArraysAdapter
        from .core.pipeline import select_solver_strict

        solver_cfg = loaded.root.densegen.solver
        select_solver_strict(
            solver_cfg.backend,
            DenseArraysAdapter(),
            strategy=str(solver_cfg.strategy),
        )
    console.print(":white_check_mark: [bold green]Config is valid.[/]")


@app.command("ls-plots", help="List available plot names and descriptions.")
def ls_plots():
    from .viz.plot_registry import PLOT_SPECS

    table = Table("plot", "description")
    for name, meta in PLOT_SPECS.items():
        table.add_row(name, meta["description"])
    console.print(table)


@app.command(help="Stage a new workspace with config.yaml and standard subfolders.")
def stage(
    run_id: str = typer.Option(..., "--id", "-i", help="Run identifier (directory name)."),
    root: Path = typer.Option(DEFAULT_WORKSPACES_ROOT, "--root", help="Workspaces root directory."),
    template: Optional[Path] = typer.Option(None, "--template", help="Template config YAML to copy."),
    copy_inputs: bool = typer.Option(False, help="Copy file-based inputs into workspace/inputs and rewrite paths."),
):
    run_id_clean = _sanitize_filename(run_id)
    if run_id_clean != run_id:
        console.print(f"[yellow]Sanitized run id:[/] {run_id} -> {run_id_clean}")
    run_dir = (root / run_id_clean).resolve()
    if run_dir.exists():
        console.print(f"[bold red]Run directory already exists:[/] {run_dir}")
        raise typer.Exit(code=1)

    template_path = template or _default_template_path()
    if not template_path.exists():
        console.print(f"[bold red]Template config not found:[/] {template_path}")
        raise typer.Exit(code=1)

    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
    (run_dir / "outputs" / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "outputs" / "meta").mkdir(parents=True, exist_ok=True)

    raw = yaml.safe_load(template_path.read_text())
    if not isinstance(raw, dict):
        console.print("[bold red]Template config must be a YAML mapping.[/]")
        raise typer.Exit(code=1)

    dense = raw.setdefault("densegen", {})
    dense["schema_version"] = LATEST_SCHEMA_VERSION
    run_block = dense.get("run") or {}
    run_block["id"] = run_id_clean
    run_block["root"] = "."
    dense["run"] = run_block

    output = dense.get("output") or {}
    if "parquet" in output and isinstance(output.get("parquet"), dict):
        output["parquet"]["path"] = "outputs/dense_arrays.parquet"
    if "usr" in output and isinstance(output.get("usr"), dict):
        output["usr"]["root"] = "outputs/usr"
    dense["output"] = output

    logging_cfg = dense.get("logging") or {}
    logging_cfg["log_dir"] = "outputs/logs"
    dense["logging"] = logging_cfg

    if "plots" in raw and isinstance(raw.get("plots"), dict):
        raw["plots"]["out_dir"] = "outputs"

    if copy_inputs:
        inputs_cfg = dense.get("inputs") or []
        for inp in inputs_cfg:
            if not isinstance(inp, dict):
                continue
            if "path" in inp:
                src = resolve_relative_path(template_path, inp["path"])
                if not src.exists() or not src.is_file():
                    console.print(f"[bold red]Input file not found:[/] {src}")
                    raise typer.Exit(code=1)
                dest = run_dir / "inputs" / src.name
                if dest.exists():
                    console.print(f"[bold red]Input file already exists:[/] {dest}")
                    raise typer.Exit(code=1)
                shutil.copy2(src, dest)
                inp["path"] = str(Path("inputs") / src.name)
            if "paths" in inp and isinstance(inp["paths"], list):
                new_paths: list[str] = []
                for path in inp["paths"]:
                    src = resolve_relative_path(template_path, path)
                    if not src.exists() or not src.is_file():
                        console.print(f"[bold red]Input file not found:[/] {src}")
                        raise typer.Exit(code=1)
                    dest = run_dir / "inputs" / src.name
                    if dest.exists():
                        console.print(f"[bold red]Input file already exists:[/] {dest}")
                        raise typer.Exit(code=1)
                    shutil.copy2(src, dest)
                    new_paths.append(str(Path("inputs") / src.name))
                inp["paths"] = new_paths

    config_path = run_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    console.print(f":sparkles: [bold green]Workspace staged[/]: {config_path}")


@app.command(help="Summarize a run manifest.")
def summarize(
    ctx: typer.Context,
    run: Optional[Path] = typer.Option(None, "--run", "-r", help="Run directory (defaults to config run root)."),
    root: Optional[Path] = typer.Option(None, "--root", help="Workspaces root directory (lists workspaces)."),
    limit: int = typer.Option(0, "--limit", help="Limit workspaces displayed when using --root (0 = all)."),
    show_all: bool = typer.Option(False, "--all", help="Include directories without config.yaml when using --root."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show failure breakdown columns."),
    library: bool = typer.Option(False, "--library", help="Include offered-vs-used library summaries."),
    top: int = typer.Option(10, "--top", help="Rows to show for library summaries."),
    by_library: bool = typer.Option(True, "--by-library/--no-by-library", help="Group library summaries per build."),
    top_per_tf: Optional[int] = typer.Option(None, "--top-per-tf", help="Limit TFBS rows per TF when summarizing."),
    show_library_hash: bool = typer.Option(
        True,
        "--show-library-hash/--short-library-hash",
        help="Show full library hash (or short hash if disabled).",
    ),
):
    if root is not None and run is not None:
        console.print("[bold red]Choose either --root or --run, not both.[/]")
        raise typer.Exit(code=1)
    if root is not None:
        workspaces_root = root.resolve()
        if not workspaces_root.exists() or not workspaces_root.is_dir():
            console.print(f"[bold red]Workspaces root not found:[/] {workspaces_root}")
            raise typer.Exit(code=1)
        console.print(_list_workspaces_table(workspaces_root, limit=limit, show_all=show_all))
        return
    cfg_path = None
    loaded = None
    if run is None:
        cfg_path = _resolve_config_path(ctx, config)
        loaded = _load_config_or_exit(cfg_path)
        run_root = _run_root_for(loaded)
    else:
        run_root = run
        if library:
            cfg_path = run_root / "config.yaml"
            if not cfg_path.exists():
                console.print(
                    f"[bold red]Config not found for --library:[/] {cfg_path}. "
                    "Provide --config or run summarize without --library."
                )
                raise typer.Exit(code=1)
            loaded = _load_config_or_exit(cfg_path)
    manifest_path = run_manifest_path(run_root)
    if not manifest_path.exists():
        state_path = run_state_path(run_root)
        if state_path.exists():
            state = load_run_state(state_path)
            console.print("[yellow]Run manifest missing; showing checkpointed run_state.[/]")
            console.print(
                f"[bold]Run:[/] {state.run_id}  [bold]Root:[/] {state.run_root}  "
                f"[bold]Schema:[/] {state.schema_version}  [bold]Config:[/] {state.config_sha256[:8]}…"
            )
            table = Table("input", "plan", "generated")
            for item in state.items:
                table.add_row(item.input_name, item.plan_name, str(item.generated))
            console.print(table)
            console.print("[bold]Next steps[/]:")
            console.print(f"  - dense run -c {cfg_path or run_root / 'config.yaml'}")
            return

        console.print(f"[bold red]Run manifest not found:[/] {manifest_path}")
        entries = _list_dir_entries(run_root, limit=8)
        if entries:
            console.print(f"[bold]Run root contents[/]: {', '.join(entries)}")
        console.print("[bold]Next steps[/]:")
        console.print(f"  - dense run -c {cfg_path or run_root / 'config.yaml'}")
        raise typer.Exit(code=1)

    manifest = load_run_manifest(manifest_path)
    schema_label = manifest.schema_version or "-"
    dense_arrays_label = manifest.dense_arrays_version or "-"
    dense_arrays_source = manifest.dense_arrays_version_source or "-"
    if dense_arrays_label != "-" and dense_arrays_source != "-":
        dense_arrays_label = f"{dense_arrays_label} ({dense_arrays_source})"
    console.print(
        f"[bold]Run:[/] {manifest.run_id}  [bold]Root:[/] {manifest.run_root}  "
        f"[bold]Schema:[/] {schema_label}  [bold]dense-arrays:[/] {dense_arrays_label}"
    )
    if verbose:
        table = Table(
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
        table = Table("input", "plan", "generated", "duplicates", "failed", "resamples", "libraries", "stalls")
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
    console.print(table)

    if library:
        if loaded is None or cfg_path is None:
            console.print("[bold red]Config is required for --library summaries.[/]")
            raise typer.Exit(code=1)
        with _suppress_pyarrow_sysctl_warnings():
            try:
                bundle = collect_report_data(loaded.root, cfg_path, include_combinatorics=False)
            except Exception as exc:
                console.print(f"[bold red]Failed to build library summaries:[/] {exc}")
                entries = _list_dir_entries(run_root, limit=8)
                if entries:
                    console.print(f"[bold]Run root contents[/]: {', '.join(entries)}")
                console.print("[bold]Next steps[/]:")
                console.print(f"  - dense run -c {cfg_path}")
                raise typer.Exit(code=1)

        offered_vs_used_tf = bundle.tables.get("offered_vs_used_tf")
        offered_vs_used_tfbs = bundle.tables.get("offered_vs_used_tfbs")
        library_summary = bundle.tables.get("library_summary", pd.DataFrame())

        library_hashes = set()
        if isinstance(library_summary, pd.DataFrame) and not library_summary.empty:
            library_hashes = set(library_summary.get("library_hash", []))
        elif isinstance(offered_vs_used_tf, pd.DataFrame):
            library_hashes = set(offered_vs_used_tf.get("library_hash", []))

        lib_table = Table(
            "library_index",
            "library_hash",
            "input",
            "plan",
            "size",
            "achieved/target",
            "outputs",
        )
        sampling_cfg = loaded.root.densegen.generation.sampling
        target_len = loaded.root.densegen.generation.sequence_length + int(sampling_cfg.subsample_over_length_budget_by)

        if isinstance(library_summary, pd.DataFrame) and not library_summary.empty:
            for _, row in library_summary.sort_values("library_index").iterrows():
                lib_hash = str(row.get("library_hash") or "")
                lib_hash_disp = lib_hash if show_library_hash else _short_hash(lib_hash)
                achieved = row.get("total_bp")
                achieved_label = f"{int(achieved)}/{target_len}" if achieved is not None else f"-/{target_len}"
                lib_table.add_row(
                    str(int(row.get("library_index") or 0)),
                    lib_hash_disp or "-",
                    str(row.get("input_name") or "-"),
                    str(row.get("plan_name") or "-"),
                    str(int(row.get("size") or 0)),
                    achieved_label,
                    str(int(row.get("outputs") or 0)),
                )
        elif library_hashes:
            for lib_hash in sorted(library_hashes):
                lib_hash_disp = lib_hash if show_library_hash else _short_hash(lib_hash)
                lib_table.add_row("-", lib_hash_disp, "-", "-", "-", f"-/{target_len}", "0")
        else:
            console.print("[yellow]No library attempts found (outputs/attempts.parquet missing).[/]")
            entries = _list_dir_entries(run_root, limit=8)
            if entries:
                console.print(f"[bold]Run root contents[/]: {', '.join(entries)}")
            console.print("[bold]Next steps[/]:")
            console.print(f"  - dense run -c {cfg_path}")
        console.print("[bold]Library build summary[/]")
        console.print(lib_table)

        if not isinstance(offered_vs_used_tf, pd.DataFrame) or offered_vs_used_tf.empty:
            console.print("[yellow]No offered/used TF data found (attempts missing).[/]")
            return

        def _fmt_hash(val: str) -> str:
            return val if show_library_hash else _short_hash(val)

        def _render_tf_tables(lib_hash: str) -> None:
            sub_tf = offered_vs_used_tf[offered_vs_used_tf["library_hash"] == lib_hash]
            if sub_tf.empty:
                return
            tf_table = Table("library", "tf", "offered", "used", "utilization_any")
            top_rows = sub_tf.sort_values("used_placements", ascending=False).head(top)
            for _, row in top_rows.iterrows():
                tf_table.add_row(
                    _fmt_hash(lib_hash),
                    str(row.get("tf")),
                    str(int(row.get("offered_instances", 0))),
                    str(int(row.get("used_placements", 0))),
                    f"{float(row.get('utilization_any', 0.0)):.2f}",
                )
            console.print(f"[bold]Top {top} TFs by used placements (per library)[/]")
            console.print(tf_table)

        def _render_tfbs_tables(lib_hash: str) -> None:
            if not isinstance(offered_vs_used_tfbs, pd.DataFrame) or offered_vs_used_tfbs.empty:
                return
            sub_tfbs = offered_vs_used_tfbs[offered_vs_used_tfbs["library_hash"] == lib_hash]
            if sub_tfbs.empty:
                return
            if top_per_tf:
                top_tf = offered_vs_used_tf[offered_vs_used_tf["library_hash"] == lib_hash]
                top_tf = top_tf.sort_values("used_placements", ascending=False).head(top)
                top_tf_names = set(top_tf["tf"].tolist())
                tfbs_table = Table("library", "tf", "tfbs", "offered", "used", "note")
                for tf, group in sub_tfbs.groupby("tf"):
                    if tf not in top_tf_names:
                        continue
                    group_sorted = group.sort_values("used_placements", ascending=False)
                    head = group_sorted.head(top_per_tf)
                    omitted = max(0, len(group_sorted) - len(head))
                    for i, (_, row) in enumerate(head.iterrows()):
                        note = ""
                        if i == len(head) - 1 and omitted > 0:
                            note = f"(+{omitted} more)"
                        tfbs_table.add_row(
                            _fmt_hash(lib_hash),
                            str(tf),
                            str(row.get("tfbs")),
                            str(int(row.get("offered_instances", 0))),
                            str(int(row.get("used_placements", 0))),
                            note,
                        )
                console.print(f"[bold]Top TFBS per TF (per library; top={top} TFs, top_per_tf={top_per_tf})[/]")
                console.print(tfbs_table)
            else:
                tfbs_table = Table("library", "tf", "tfbs", "offered", "used")
                top_tfbs = sub_tfbs.sort_values("used_placements", ascending=False).head(top)
                for _, row in top_tfbs.iterrows():
                    tfbs_table.add_row(
                        _fmt_hash(lib_hash),
                        str(row.get("tf")),
                        str(row.get("tfbs")),
                        str(int(row.get("offered_instances", 0))),
                        str(int(row.get("used_placements", 0))),
                    )
                console.print(f"[bold]Top {top} TFBS by used placements (per library)[/]")
                console.print(tfbs_table)

        if by_library and isinstance(library_summary, pd.DataFrame) and not library_summary.empty:
            for _, row in library_summary.sort_values("library_index").iterrows():
                lib_hash = str(row.get("library_hash") or "")
                lib_index = int(row.get("library_index") or 0)
                console.print(f"[bold]Library {lib_index}[/]: {_fmt_hash(lib_hash)}")
                _render_tf_tables(lib_hash)
                _render_tfbs_tables(lib_hash)
        elif by_library and library_hashes:
            for lib_hash in sorted(library_hashes):
                console.print(f"[bold]Library[/]: {_fmt_hash(lib_hash)}")
                _render_tf_tables(lib_hash)
                _render_tfbs_tables(lib_hash)
        else:
            tf_table = Table("library_hash", "tf", "offered", "used", "utilization_any")
            top_rows = offered_vs_used_tf.sort_values("used_placements", ascending=False).head(top)
            for _, row in top_rows.iterrows():
                lib_hash = str(row.get("library_hash") or "")
                tf_table.add_row(
                    _fmt_hash(lib_hash),
                    str(row.get("tf")),
                    str(int(row.get("offered_instances", 0))),
                    str(int(row.get("used_placements", 0))),
                    f"{float(row.get('utilization_any', 0.0)):.2f}",
                )
            console.print(f"[bold]Top {top} TFs by used placements (all libraries)[/]")
            console.print(tf_table)

            if isinstance(offered_vs_used_tfbs, pd.DataFrame) and not offered_vs_used_tfbs.empty:
                tfbs_table = Table("library_hash", "tf", "tfbs", "offered", "used")
                top_tfbs = offered_vs_used_tfbs.sort_values("used_placements", ascending=False).head(top)
                for _, row in top_tfbs.iterrows():
                    lib_hash = str(row.get("library_hash") or "")
                    tfbs_table.add_row(
                        _fmt_hash(lib_hash),
                        str(row.get("tf")),
                        str(row.get("tfbs")),
                        str(int(row.get("offered_instances", 0))),
                        str(int(row.get("used_placements", 0))),
                    )
                console.print(f"[bold]Top {top} TFBS by used placements (all libraries)[/]")
                console.print(tfbs_table)


@app.command(help="Generate audit-grade report summary for a run.")
def report(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
    out: str = typer.Option("outputs", "--out", help="Output directory (relative to run root)."),
):
    cfg_path = _resolve_config_path(ctx, config)
    loaded = _load_config_or_exit(cfg_path)
    try:
        with _suppress_pyarrow_sysctl_warnings():
            write_report(
                loaded.root,
                cfg_path,
                out_dir=out,
            )
    except FileNotFoundError as exc:
        console.print(f"[bold red]Report failed:[/] {exc}")
        run_root = _run_root_for(loaded)
        entries = _list_dir_entries(run_root, limit=8)
        if entries:
            console.print(f"[bold]Run root contents[/]: {', '.join(entries)}")
        console.print("[bold]Next steps[/]:")
        console.print(f"  - dense run -c {cfg_path}")
        raise typer.Exit(code=1)
    run_root = _run_root_for(loaded)
    out_dir = resolve_run_scoped_path(cfg_path, run_root, out, label="report.out")
    console.print(f":sparkles: [bold green]Report written[/]: {out_dir}")
    console.print("[bold]Outputs[/]: report.json, report.md")


@app.command(help="Show the resolved per-constraint quota plan.")
def plan(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
):
    cfg_path = _resolve_config_path(ctx, config)
    loaded = _load_config_or_exit(cfg_path)
    _warn_full_pool_strategy(loaded)
    pl = resolve_plan(loaded)
    table = Table("name", "quota", "has promoter_constraints")
    for item in pl:
        pcs = item.fixed_elements.promoter_constraints
        table.add_row(item.name, str(item.quota), "yes" if pcs else "no")
    console.print(table)


@app.command(help="Describe resolved config, inputs, outputs, and solver details.")
def describe(
    ctx: typer.Context,
    show_constraints: bool = typer.Option(False, help="Print full fixed elements per plan item."),
    probe_solver: bool = typer.Option(False, help="Probe the solver backend before reporting."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
):
    cfg_path = _resolve_config_path(ctx, config)
    loaded = _load_config_or_exit(cfg_path)
    root = loaded.root
    cfg = root.densegen
    run_root = _run_root_for(loaded)

    if probe_solver:
        from .adapters.optimizer import DenseArraysAdapter
        from .core.pipeline import select_solver_strict

        select_solver_strict(cfg.solver.backend, DenseArraysAdapter(), strategy=str(cfg.solver.strategy))

    console.print(f"[bold]Config[/]: {cfg_path}")
    console.print(f"[bold]Run[/]: id={cfg.run.id} root={run_root}")

    inputs = Table("name", "type", "source")
    for inp in cfg.inputs:
        if hasattr(inp, "path"):
            src = str(resolve_relative_path(loaded.path, inp.path))
        elif hasattr(inp, "paths"):
            resolved = [str(resolve_relative_path(loaded.path, p)) for p in getattr(inp, "paths") or []]
            src = f"{len(resolved)} files"
            if resolved:
                src = f"{len(resolved)} files ({resolved[0]})"
        elif hasattr(inp, "dataset"):
            src = f"{inp.dataset} (root={resolve_relative_path(loaded.path, inp.root)})"
        else:
            src = "-"
        inputs.add_row(inp.name, inp.type, src)
    console.print(inputs)

    # Alignment (8): make two-stage sampling explicit in CLI describe output.
    pwm_inputs = [
        inp
        for inp in cfg.inputs
        if getattr(inp, "type", "")
        in {
            "pwm_meme",
            "pwm_meme_set",
            "pwm_jaspar",
            "pwm_matrix_csv",
            "pwm_artifact",
            "pwm_artifact_set",
        }
    ]
    if pwm_inputs:
        pwm_table = Table(
            "name",
            "motifs",
            "n_sites",
            "strategy",
            "backend",
            "score",
            "selection",
            "bins",
            "bgfile",
            "oversample",
            "max_candidates",
            "max_seconds",
            "length",
        )
        for inp in pwm_inputs:
            sampling = getattr(inp, "sampling", None)
            if sampling is None:
                continue
            if inp.type == "pwm_matrix_csv":
                motif_label = str(getattr(inp, "motif_id", "-"))
            elif inp.type in {"pwm_meme", "pwm_meme_set", "pwm_jaspar"}:
                motif_ids = getattr(inp, "motif_ids", None) or []
                motif_label = ", ".join(motif_ids) if motif_ids else "all"
                if inp.type == "pwm_meme_set":
                    file_count = len(getattr(inp, "paths", []) or [])
                    motif_label = f"{motif_label} ({file_count} files)"
            elif inp.type == "pwm_artifact_set":
                motif_label = f"{len(getattr(inp, 'paths', []) or [])} artifacts"
            else:
                motif_label = "from artifact"
            backend = getattr(sampling, "scoring_backend", "densegen")
            score_label = "-"
            if backend == "fimo" and sampling.pvalue_threshold is not None:
                comparator = ">=" if sampling.strategy == "background" else "<="
                score_label = f"pvalue{comparator}{sampling.pvalue_threshold}"
            elif sampling.score_threshold is not None:
                score_label = f"threshold={sampling.score_threshold}"
            elif sampling.score_percentile is not None:
                score_label = f"percentile={sampling.score_percentile}"
            selection_label = "-" if backend != "fimo" else (getattr(sampling, "selection_policy", None) or "-")
            bins_label = "-"
            if backend == "fimo":
                bins_label = "canonical"
                if getattr(sampling, "pvalue_bins", None) is not None:
                    bins_label = "custom"
                bin_ids = getattr(sampling, "pvalue_bin_ids", None)
                if bin_ids:
                    bins_label = f"{bins_label} pick={bin_ids}"
            bgfile_label = getattr(sampling, "bgfile", None) or "-"
            length_label = str(sampling.length_policy)
            if sampling.length_policy == "range" and sampling.length_range is not None:
                length_label = f"range({sampling.length_range[0]}..{sampling.length_range[1]})"
            pwm_table.add_row(
                inp.name,
                motif_label,
                str(sampling.n_sites),
                str(sampling.strategy),
                str(backend),
                score_label,
                str(selection_label),
                str(bins_label),
                str(bgfile_label),
                str(sampling.oversample_factor),
                str(sampling.max_candidates) if sampling.max_candidates is not None else "-",
                str(sampling.max_seconds) if sampling.max_seconds is not None else "-",
                length_label,
            )
        console.print("[bold]Input-stage PWM sampling[/]")
        console.print(pwm_table)
        console.print(
            "  -> Produces the realized TFBS pool (input_tfbs_count), captured in inputs_manifest.json after runs."
        )

    plan_table = Table(
        "name",
        "quota/fraction",
        "promoter_constraints",
        "side_biases",
        "required_regulators",
        "min_required_regulators",
        "min_count_by_regulator",
    )
    for item in cfg.generation.resolve_plan():
        pcs = item.fixed_elements.promoter_constraints
        left = item.fixed_elements.side_biases.left if item.fixed_elements.side_biases else []
        right = item.fixed_elements.side_biases.right if item.fixed_elements.side_biases else []
        bias_count = len(set(left)) + len(set(right))
        req_count = len(item.required_regulators or [])
        min_req = item.min_required_regulators if item.min_required_regulators is not None else "-"
        min_count_regs = len(item.min_count_by_regulator or {})
        quota = str(item.quota) if item.quota is not None else f"{item.fraction:.3f}"
        plan_table.add_row(
            item.name,
            quota,
            str(len(pcs)),
            str(bias_count),
            str(req_count),
            str(min_req),
            str(min_count_regs),
        )
    console.print(plan_table)

    if show_constraints:
        for item in cfg.generation.resolve_plan():
            console.print(f"[bold]{item.name}[/]")
            pcs = item.fixed_elements.promoter_constraints
            if pcs:
                for pc in pcs:
                    console.print(
                        f"  promoter: upstream={pc.upstream} downstream={pc.downstream} "
                        f"spacer_length={pc.spacer_length} upstream_pos={pc.upstream_pos} "
                        f"downstream_pos={pc.downstream_pos}"
                    )
            sb = item.fixed_elements.side_biases
            if sb:
                if sb.left:
                    console.print(f"  side_biases.left: {', '.join(sb.left)}")
                if sb.right:
                    console.print(f"  side_biases.right: {', '.join(sb.right)}")
            if item.required_regulators:
                console.print(f"  required_regulators: {', '.join(item.required_regulators)}")

    outputs = Table("target", "path")
    for target in cfg.output.targets:
        if target == "parquet":
            parquet_path = resolve_run_scoped_path(
                loaded.path,
                run_root,
                cfg.output.parquet.path,
                label="output.parquet.path",
            )
            outputs.add_row(
                "parquet",
                str(parquet_path),
            )
        elif target == "usr":
            usr_root = resolve_run_scoped_path(loaded.path, run_root, cfg.output.usr.root, label="output.usr.root")
            outputs.add_row("usr", f"{cfg.output.usr.dataset} (root={usr_root})")
        else:
            outputs.add_row(target, "-")
    console.print(outputs)

    solver = Table("backend", "strategy", "options", "strands")
    backend_display = str(cfg.solver.backend) if cfg.solver.backend is not None else "-"
    solver.add_row(backend_display, str(cfg.solver.strategy), str(len(cfg.solver.options)), str(cfg.solver.strands))
    console.print(solver)

    sampling = cfg.generation.sampling
    sampling_table = Table("setting", "value")
    target_length = cfg.generation.sequence_length + int(sampling.subsample_over_length_budget_by)
    sampling_table.add_row("pool_strategy", str(sampling.pool_strategy))
    sampling_table.add_row("library_size", str(sampling.library_size))
    sampling_table.add_row("library_sampling_strategy", str(sampling.library_sampling_strategy))
    sampling_table.add_row(
        "subsample_over_length_budget_by",
        f"{sampling.subsample_over_length_budget_by} (target={target_length} bp)",
    )
    sampling_table.add_row("cover_all_regulators", str(sampling.cover_all_regulators))
    sampling_table.add_row("unique_binding_sites", str(sampling.unique_binding_sites))
    sampling_table.add_row("max_sites_per_regulator", str(sampling.max_sites_per_regulator))
    sampling_table.add_row("relax_on_exhaustion", str(sampling.relax_on_exhaustion))
    sampling_table.add_row("allow_incomplete_coverage", str(sampling.allow_incomplete_coverage))
    sampling_table.add_row("iterative_max_libraries", str(sampling.iterative_max_libraries))
    sampling_table.add_row("iterative_min_new_solutions", str(sampling.iterative_min_new_solutions))
    sampling_table.add_row("arrays_generated_before_resample", str(cfg.runtime.arrays_generated_before_resample))
    sampling_table.add_row("max_resample_attempts", str(cfg.runtime.max_resample_attempts))
    sampling_table.add_row("max_total_resamples", str(cfg.runtime.max_total_resamples))
    console.print("[bold]Solver-stage library sampling[/]")
    console.print(sampling_table)

    gap = cfg.postprocess.gap_fill
    console.print(
        "[bold]Gap fill[/]: "
        f"mode={gap.mode} end={gap.end} gc=[{gap.gc_min:.2f}, {gap.gc_max:.2f}] "
        f"max_tries={gap.max_tries}"
    )
    log_dir = resolve_run_scoped_path(loaded.path, run_root, cfg.logging.log_dir, label="logging.log_dir")
    console.print(f"[bold]Logging[/]: dir={log_dir} level={cfg.logging.level}")

    if root.plots:
        out_dir = resolve_run_scoped_path(loaded.path, run_root, root.plots.out_dir, label="plots.out_dir")
        console.print(f"[bold]Plots[/]: source={root.plots.source} out_dir={out_dir}")
    else:
        console.print("[bold]Plots[/]: none")


@app.command(help="Run generation for the job. Optionally auto-run plots declared in YAML.")
def run(
    ctx: typer.Context,
    no_plot: bool = typer.Option(False, help="Do not auto-run plots even if configured."),
    log_file: Optional[Path] = typer.Option(None, help="Override logfile path."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
):
    cfg_path = _resolve_config_path(ctx, config)
    loaded = _load_config_or_exit(cfg_path)
    root = loaded.root
    cfg = root.densegen
    run_root = _run_root_for(loaded)

    # Logging setup
    log_cfg = cfg.logging
    log_dir = resolve_run_scoped_path(loaded.path, run_root, Path(log_cfg.log_dir), label="logging.log_dir")
    default_logfile = log_dir / f"{cfg.run.id}.log"
    if log_file is not None:
        logfile = resolve_run_scoped_path(loaded.path, run_root, log_file, label="logging.log_file")
    else:
        logfile = default_logfile
    setup_logging(
        level=log_cfg.level,
        logfile=str(logfile),
        suppress_solver_stderr=bool(log_cfg.suppress_solver_stderr),
    )

    # Plan & solver
    pl = resolve_plan(loaded)
    console.print("[bold]Quota plan[/]: " + ", ".join(f"{p.name}={p.quota}" for p in pl))
    try:
        run_pipeline(loaded)
    except FileNotFoundError as exc:
        _render_missing_input_hint(cfg_path, loaded, exc)
        raise typer.Exit(code=1)

    console.print(":tada: [bold green]Run complete[/].")
    console.print("[bold]Next steps[/]:")
    console.print(f"  - dense summarize --library -c {cfg_path}")
    console.print(f"  - dense report -c {cfg_path}")

    # Auto-plot if configured
    if not no_plot and root.plots:
        _ensure_mpl_cache_dir()
        install_native_stderr_filters()
        from .viz.plotting import run_plots_from_config

        console.print("[bold]Generating plots...[/]")
        run_plots_from_config(root, loaded.path)
        console.print(":bar_chart: [bold green]Plots written.[/]")


@app.command(help="Generate plots from outputs according to YAML. Use --only to select plots.")
def plot(
    ctx: typer.Context,
    only: Optional[str] = typer.Option(None, help="Comma-separated plot names (subset of available plots)."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
):
    cfg_path = _resolve_config_path(ctx, config)
    loaded = _load_config_or_exit(cfg_path)
    _ensure_mpl_cache_dir()
    install_native_stderr_filters()
    from .viz.plotting import run_plots_from_config

    run_plots_from_config(loaded.root, loaded.path, only=only)
    console.print(":bar_chart: [bold green]Plots written.[/]")


if __name__ == "__main__":
    app()
