"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/cli.py

Typer/Rich CLI entrypoint for DenseGen.

Commands:
  - validate : Validate YAML config (schema + sanity).
  - plan     : Show resolved per-constraint quota plan.
  - stage    : Scaffold a new run directory with config.yaml + subfolders.
  - run      : Execute generation pipeline; optionally auto-plot.
  - plot     : Generate plots from outputs using config YAML.
  - ls-plots : List available plot names and descriptions.
  - summarize : Print a run_manifest.json summary table.

Run:
  python -m dnadesign.densegen.src.cli --help

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table
from rich.traceback import install as rich_traceback

from .config import (
    ConfigError,
    load_config,
    resolve_relative_path,
    resolve_run_root,
    resolve_run_scoped_path,
)
from .core.pipeline import resolve_plan, run_pipeline
from .core.run_manifest import load_run_manifest
from .utils.logging_utils import setup_logging

rich_traceback(show_locals=False)
console = Console()


# ----------------- local path helpers -----------------
def _densegen_root_from(file_path: Path) -> Path:
    return file_path.resolve().parent.parent


DENSEGEN_ROOT = _densegen_root_from(Path(__file__))
DEFAULT_RUNS_ROOT = DENSEGEN_ROOT / "runs"


def _default_config_path() -> Path:
    # Prefer a small, run-scoped example config inside the package tree.
    return DENSEGEN_ROOT / "runs" / "demo" / "config.yaml"


def _default_template_path() -> Path:
    return DENSEGEN_ROOT / "runs" / "demo" / "config.yaml"


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


def _list_runs_table(runs_root: Path, *, limit: int, show_all: bool) -> Table:
    run_dirs = sorted([p for p in runs_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if limit and limit > 0:
        run_dirs = run_dirs[: int(limit)]

    table = Table("run", "id", "config", "parquet", "plots", "logs", "status")
    for run_dir in run_dirs:
        cfg_path = run_dir / "config.yaml"
        if not show_all and not cfg_path.exists():
            continue

        run_id = run_dir.name
        status = "ok"
        parquet_count = "-"
        plots_count = _count_files(run_dir / "plots", pattern="*")
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
                        loaded.root.plots.out_dir if loaded.root.plots else "plots",
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


@app.command(help="Stage a new run directory with config.yaml and standard subfolders.")
def stage(
    run_id: str = typer.Option(..., "--id", "-i", help="Run identifier (directory name)."),
    root: Path = typer.Option(DEFAULT_RUNS_ROOT, "--root", help="Runs root directory."),
    template: Optional[Path] = typer.Option(None, "--template", help="Template config YAML to copy."),
    copy_inputs: bool = typer.Option(False, help="Copy file-based inputs into run/inputs and rewrite paths."),
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
    (run_dir / "outputs" / "parquet").mkdir(parents=True, exist_ok=True)
    (run_dir / "outputs" / "usr").mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)

    raw = yaml.safe_load(template_path.read_text())
    if not isinstance(raw, dict):
        console.print("[bold red]Template config must be a YAML mapping.[/]")
        raise typer.Exit(code=1)

    dense = raw.setdefault("densegen", {})
    dense["schema_version"] = "2.1"
    run_block = dense.get("run") or {}
    run_block["id"] = run_id_clean
    run_block["root"] = "."
    dense["run"] = run_block

    output = dense.get("output") or {}
    if "parquet" in output and isinstance(output.get("parquet"), dict):
        output["parquet"]["path"] = "outputs/parquet"
    if "usr" in output and isinstance(output.get("usr"), dict):
        output["usr"]["root"] = "outputs/usr"
    dense["output"] = output

    logging_cfg = dense.get("logging") or {}
    logging_cfg["log_dir"] = "logs"
    dense["logging"] = logging_cfg

    if "plots" in raw and isinstance(raw.get("plots"), dict):
        raw["plots"]["out_dir"] = "plots"

    if copy_inputs:
        inputs_cfg = dense.get("inputs") or []
        for inp in inputs_cfg:
            if not isinstance(inp, dict) or "path" not in inp:
                continue
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

    config_path = run_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
    console.print(f":sparkles: [bold green]Run staged[/]: {config_path}")


@app.command(help="Summarize a run manifest.")
def summarize(
    ctx: typer.Context,
    run: Optional[Path] = typer.Option(None, "--run", "-r", help="Run directory (defaults to config run root)."),
    root: Optional[Path] = typer.Option(None, "--root", help="Runs root directory (lists runs)."),
    limit: int = typer.Option(0, "--limit", help="Limit runs displayed when using --root (0 = all)."),
    show_all: bool = typer.Option(False, "--all", help="Include directories without config.yaml when using --root."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show failure breakdown columns."),
):
    if root is not None and run is not None:
        console.print("[bold red]Choose either --root or --run, not both.[/]")
        raise typer.Exit(code=1)
    if root is not None:
        runs_root = root.resolve()
        if not runs_root.exists() or not runs_root.is_dir():
            console.print(f"[bold red]Runs root not found:[/] {runs_root}")
            raise typer.Exit(code=1)
        console.print(_list_runs_table(runs_root, limit=limit, show_all=show_all))
        return
    if run is None:
        cfg_path = _resolve_config_path(ctx, config)
        loaded = _load_config_or_exit(cfg_path)
        run_root = _run_root_for(loaded)
    else:
        run_root = run
    manifest_path = run_root / "run_manifest.json"
    if not manifest_path.exists():
        console.print(f"[bold red]Run manifest not found:[/] {manifest_path}")
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


@app.command(help="Show the resolved per-constraint quota plan.")
def plan(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
):
    cfg_path = _resolve_config_path(ctx, config)
    loaded = _load_config_or_exit(cfg_path)
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
        elif hasattr(inp, "dataset"):
            src = f"{inp.dataset} (root={resolve_relative_path(loaded.path, inp.root)})"
        else:
            src = "-"
        inputs.add_row(inp.name, inp.type, src)
    console.print(inputs)

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
    run_pipeline(loaded)

    console.print(":tada: [bold green]Run complete[/].")

    # Auto-plot if configured
    if not no_plot and root.plots:
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
    from .viz.plotting import run_plots_from_config

    run_plots_from_config(loaded.root, loaded.path, only=only)
    console.print(":bar_chart: [bold green]Plots written.[/]")


if __name__ == "__main__":
    app()
