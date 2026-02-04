"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/cli.py

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
  - report          : Generate audit-grade report tables for a run.

Run:
  python -m dnadesign.densegen.src.cli --help

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
import random
import re
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import typer
import yaml
from rich.console import Console
from rich.traceback import install as rich_traceback

from .adapters.sources.stage_a.stage_a_summary import PWMSamplingSummary
from .cli_commands.context import CliContext
from .cli_commands.inspect import register_inspect_commands
from .cli_commands.report import register_report_command
from .cli_render import stage_a_plan_table, stage_a_recap_tables
from .cli_sampling import format_selection_label, stage_a_plan_rows
from .config import (
    LATEST_SCHEMA_VERSION,
    ConfigError,
    load_config,
    resolve_outputs_scoped_path,
    resolve_relative_path,
    resolve_run_root,
)
from .core.artifacts.candidates import build_candidate_artifact, find_candidate_files, prepare_candidates_dir
from .core.artifacts.library import load_library_artifact, write_library_artifact
from .core.artifacts.pool import PoolData, _hash_file, build_pool_artifact, load_pool_data
from .core.motif_labels import input_motifs
from .core.pipeline import default_deps, resolve_plan, run_pipeline
from .core.pipeline.attempts import _load_existing_library_index, _load_failure_counts_from_attempts
from .core.pipeline.outputs import _emit_event
from .core.pipeline.plan_pools import PLAN_POOL_INPUT_TYPE, build_plan_pools
from .core.pipeline.stage_b import assess_library_feasibility, build_library_for_plan
from .core.run_paths import (
    candidates_root,
    display_path,
    has_existing_run_outputs,
    run_outputs_root,
)
from .core.seeding import derive_seed_map
from .integrations.meme_suite import require_executable
from .utils import logging_utils
from .utils.logging_utils import install_native_stderr_filters, setup_logging
from .utils.mpl_utils import ensure_mpl_cache_dir
from .utils.rich_style import make_table


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

DEFAULT_CONFIG_FILENAME = "config.yaml"
DEFAULT_CONFIG_MISSING_MESSAGE = (
    "No config found. cd into a workspace containing config.yaml, or pass -c path/to/config.yaml."
)
PACKAGED_TEMPLATES: dict[str, str] = {
    "demo_meme_three_tfs": "workspaces/demo_meme_three_tfs",
}


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
def _list_packaged_template_ids() -> list[str]:
    return sorted(PACKAGED_TEMPLATES.keys())


@contextlib.contextmanager
def _resolve_template_dir(
    *,
    template: Optional[Path],
    template_id: Optional[str],
) -> Iterator[tuple[Path, Path]]:
    if template and template_id:
        console.print("[bold red]Choose either --template or --template-id, not both.[/]")
        raise typer.Exit(code=1)
    if template_id:
        rel_dir = PACKAGED_TEMPLATES.get(template_id)
        if not rel_dir:
            available = ", ".join(_list_packaged_template_ids()) or "-"
            console.print(f"[bold red]Unknown template id:[/] {template_id}")
            console.print(f"[bold]Available template ids:[/] {available}")
            raise typer.Exit(code=1)
        package_root = resources.files("dnadesign.densegen")
        template_dir = package_root.joinpath(rel_dir)
        if not template_dir.exists():
            console.print(f"[bold red]Packaged template not found:[/] {rel_dir}")
            raise typer.Exit(code=1)
        with resources.as_file(template_dir) as resolved:
            config_path = Path(resolved) / DEFAULT_CONFIG_FILENAME
            if not config_path.exists():
                console.print(
                    f"[bold red]Template config not found:[/] {_display_path(config_path, Path.cwd(), absolute=False)}"
                )
                raise typer.Exit(code=1)
            yield Path(resolved), config_path
        return
    if template is None:
        console.print("[bold red]No template provided.[/] Use --template-id or --template.")
        raise typer.Exit(code=1)
    template_path = template.expanduser().resolve()
    if not template_path.exists():
        console.print(
            f"[bold red]Template config not found:[/] {_display_path(template_path, Path.cwd(), absolute=False)}"
        )
        raise typer.Exit(code=1)
    if not template_path.is_file():
        console.print(
            f"[bold red]Template path is not a file:[/] {_display_path(template_path, Path.cwd(), absolute=False)}"
        )
        raise typer.Exit(code=1)
    yield template_path.parent, template_path


def _input_uses_fimo(input_cfg) -> bool:
    if not str(getattr(input_cfg, "type", "")).startswith("pwm_"):
        return False
    return True


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
        sampling_updates: dict = {}
        budget_updates: dict = {}
        if n_sites is not None:
            sampling_updates["n_sites"] = n_sites
        mining_updates: dict = {}
        if batch_size is not None:
            mining_updates["batch_size"] = batch_size
        if max_seconds is not None:
            budget_updates["max_seconds"] = max_seconds
        if mining_updates:
            mining = sampling.mining
            if mining is None:
                raise typer.BadParameter("Stage-A sampling mining config is required for overrides")
            sampling_updates["mining"] = mining.model_copy(update=mining_updates)
        if budget_updates:
            mining = sampling_updates.get("mining", sampling.mining)
            if mining is None or getattr(mining, "budget", None) is None:
                raise typer.BadParameter("Stage-A sampling mining.budget is required for overrides")
            budget = mining.budget
            mining_updates = dict(getattr(mining, "model_dump", lambda **_: {})()) or {}
            mining_updates["budget"] = budget.model_copy(update=budget_updates)
            sampling_updates["mining"] = mining.model_copy(update=mining_updates)
        if sampling_updates:
            inp.sampling = sampling.model_copy(update=sampling_updates)
        overrides = getattr(inp, "overrides_by_motif_id", None)
        if isinstance(overrides, dict) and overrides:
            new_overrides = {}
            for motif_id, override in overrides.items():
                override_updates: dict = {}
                override_budget_updates: dict = {}
                if n_sites is not None:
                    override_updates["n_sites"] = n_sites
                if mining_updates:
                    mining = override.mining
                    if mining is None:
                        raise typer.BadParameter("Stage-A sampling mining config is required for overrides")
                    override_updates["mining"] = mining.model_copy(update=mining_updates)
                if override_budget_updates:
                    mining = override_updates.get("mining", override.mining)
                    if mining is None or getattr(mining, "budget", None) is None:
                        raise typer.BadParameter("Stage-A sampling mining.budget is required for overrides")
                    budget = mining.budget
                    mining_updates = dict(getattr(mining, "model_dump", lambda **_: {})()) or {}
                    mining_updates["budget"] = budget.model_copy(update=override_budget_updates)
                    override_updates["mining"] = mining.model_copy(update=mining_updates)
                if override_updates:
                    override = override.model_copy(update=override_updates)
                new_overrides[motif_id] = override
            inp.overrides_by_motif_id = new_overrides


def _ensure_fimo_available(cfg, *, strict: bool = True) -> None:
    if not any(_input_uses_fimo(inp) for inp in cfg.inputs):
        return
    try:
        require_executable("fimo", tool_path=None)
    except FileNotFoundError as exc:
        msg = f"FIMO is required for this config but was not found. {exc}"
        if strict:
            console.print(f"[bold red]{msg}[/]")
            raise typer.Exit(code=1)
        log.warning(msg)


def _default_config_path() -> Path:
    return Path.cwd() / DEFAULT_CONFIG_FILENAME


def _find_config_in_parents(start: Path) -> Path | None:
    try:
        cursor = start.resolve()
    except Exception:
        cursor = start
    for root in [cursor, *cursor.parents]:
        candidate = root / DEFAULT_CONFIG_FILENAME
        if candidate.exists():
            return candidate
    return None


def _workspace_search_roots() -> list[Path]:
    roots: list[Path] = []
    env_root = os.environ.get("DENSEGEN_WORKSPACE_ROOT")
    if env_root:
        roots.append(Path(env_root))
    pixi_root = os.environ.get("PIXI_PROJECT_ROOT")
    if pixi_root:
        roots.append(Path(pixi_root))
    if not roots:
        repo_root = _repo_root_from(Path(__file__).resolve())
        if repo_root is not None:
            roots.append(repo_root)
    roots.append(Path.cwd())
    seen: set[str] = set()
    unique: list[Path] = []
    for root in roots:
        try:
            key = str(root.resolve())
        except Exception:
            key = str(root)
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return unique


def _repo_root_from(start: Path) -> Path | None:
    try:
        cursor = start.resolve()
    except Exception:
        cursor = start
    for root in [cursor, *cursor.parents]:
        if (root / "pyproject.toml").exists() or (root / ".git").exists():
            return root
    return None


def _auto_config_path() -> tuple[Path | None, list[Path]]:
    candidates: list[Path] = []
    for root in _workspace_search_roots():
        for base in (
            root / "src" / "dnadesign" / "densegen" / "workspaces",
            root / "workspaces",
        ):
            if not base.exists():
                continue
            for path in sorted(base.glob(f"*/{DEFAULT_CONFIG_FILENAME}")):
                candidates.append(path)
    unique: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        try:
            key = str(path.resolve())
        except Exception:
            key = str(path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    if len(unique) == 1:
        return unique[0], []
    return None, unique


def _workspace_command(command: str, *, cfg_path: Path | None = None, run_root: Path | None = None) -> str:
    root = run_root or (cfg_path.parent if cfg_path is not None else None)
    base = Path.cwd()
    if root is not None:
        try:
            root_resolved = root.resolve()
        except Exception:
            root_resolved = root
        if root_resolved == base.resolve():
            return command
        candidate = root / DEFAULT_CONFIG_FILENAME
        if candidate.exists():
            root_label = display_path(root, base, absolute=False)
            return f"cd {root_label} && {command}"
    if cfg_path is not None:
        cfg_label = display_path(cfg_path, base, absolute=False)
        return f"{command} -c {cfg_label}"
    return command


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


def _resolve_config_path(ctx: typer.Context, override: Optional[Path]) -> tuple[Path, bool]:
    if override is not None:
        return Path(override), False
    if ctx.obj:
        ctx_path = ctx.obj.get("config_path")
        if ctx_path is not None:
            return Path(ctx_path), False
    env_path = os.environ.get("DENSEGEN_CONFIG_PATH")
    if env_path:
        return Path(env_path), False
    default_path = _default_config_path()
    if default_path.exists():
        return default_path, True
    parent_path = _find_config_in_parents(Path.cwd())
    if parent_path is not None:
        return parent_path, False
    auto_path, candidates = _auto_config_path()
    if auto_path is not None:
        console.print(
            f"[bold yellow]Config not found in cwd; using[/] "
            f"{_display_path(auto_path, Path.cwd(), absolute=False)} (auto-detected). "
            "Pass -c to select a different workspace."
        )
        return auto_path, False
    if candidates:
        console.print("[bold red]Multiple workspace configs found; use -c to select one.[/]")
        for path in candidates:
            console.print(f" - {_display_path(path, Path.cwd(), absolute=False)}")
        raise typer.Exit(code=1)
    return default_path, True


def _load_config_or_exit(
    cfg_path: Path,
    *,
    missing_message: str | None = None,
    absolute: bool = False,
    display_root: Path | None = None,
):
    try:
        return load_config(cfg_path)
    except FileNotFoundError:
        if missing_message:
            console.print(f"[bold red]{missing_message}[/]")
        else:
            root = display_root or Path.cwd()
            console.print(f"[bold red]Config file not found:[/] {_display_path(cfg_path, root, absolute=absolute)}")
        raise typer.Exit(code=1)
    except ConfigError as e:
        console.print(f"[bold red]Config error:[/] {e}")
        raise typer.Exit(code=1)


def _resolve_outputs_path_or_exit(
    cfg_path: Path,
    run_root: Path,
    value: str | os.PathLike,
    *,
    label: str,
) -> Path:
    try:
        return resolve_outputs_scoped_path(cfg_path, run_root, value, label=label)
    except ConfigError as exc:
        console.print(f"[bold red]{exc}[/]")
        raise typer.Exit(code=1)


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


def _display_path(path: Path, run_root: Path, *, absolute: bool) -> str:
    return display_path(path, run_root, absolute=absolute)


def _format_sampling_ratio(value: int, target: int | None) -> str:
    if target is None or target <= 0:
        return str(int(value))
    return f"{int(value)}/{int(target)}"


def _format_count(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{int(value):,}"


def _format_ratio(count: int | None, total: int | None) -> str:
    if count is None:
        return "-"
    if total is None or int(total) <= 0:
        return _format_count(count)
    pct = 100.0 * float(count) / float(total)
    return f"{_format_count(count)} ({pct:.0f}%)"


def _format_sampling_lengths(
    *,
    min_len: int | None,
    median_len: float | None,
    mean_len: float | None,
    max_len: int | None,
    count: int | None,
) -> str:
    if count is None:
        return "-"
    if min_len is None or median_len is None or mean_len is None or max_len is None:
        return f"{int(count)}/-/-/-/-"
    return f"{int(count)}/{int(min_len)}/{median_len:.1f}/{mean_len:.1f}/{int(max_len)}"


def _format_score_stats(
    *,
    min_score: float | None,
    median_score: float | None,
    mean_score: float | None,
    max_score: float | None,
) -> str:
    if min_score is None or median_score is None or mean_score is None or max_score is None:
        return "-"
    return f"{min_score:.2f}/{median_score:.2f}/{mean_score:.2f}/{max_score:.2f}"


def _format_diversity_value(value: float | None, *, show_sign: bool = False) -> str:
    if value is None:
        return "-"
    if show_sign:
        return f"{float(value):+.2f}"
    return f"{float(value):.2f}"


def _format_score_norm_summary(summary) -> str:
    if summary is None:
        return "-"
    top = getattr(summary, "top_candidates", None)
    diversified = getattr(summary, "diversified_candidates", None)
    if top is None or diversified is None:
        return "-"
    return (
        f"top {float(top.min):.2f}/{float(top.median):.2f}/{float(top.max):.2f} | "
        f"div {float(diversified.min):.2f}/{float(diversified.median):.2f}/{float(diversified.max):.2f}"
    )


def _format_score_norm_triplet(summary, *, label: str) -> str:
    if summary is None:
        return "-"
    block = getattr(summary, label, None)
    if block is None:
        return "-"
    return f"{float(block.min):.2f}/{float(block.median):.2f}/{float(block.max):.2f}"


def _format_tier_counts(eligible: list[int] | None, retained: list[int] | None) -> str:
    if not eligible or not retained:
        raise ValueError("Stage-A tier counts are required.")
    if len(eligible) != len(retained):
        raise ValueError("Stage-A tier counts length mismatch.")
    parts = []
    for idx in range(len(eligible)):
        parts.append(f"t{idx} {int(eligible[idx])}/{int(retained[idx])}")
    return " | ".join(parts)


def _format_tier_fraction_label(fraction: float) -> str:
    return f"{float(fraction) * 100:.3f}%"


def _stage_a_sampling_rows(
    pool_data: dict[str, PoolData],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pool in pool_data.values():
        summaries = pool.summaries or []
        if summaries:
            for summary in summaries:
                if not isinstance(summary, PWMSamplingSummary):
                    continue
                input_name = summary.input_name or pool.name
                if summary.generated is None:
                    raise ValueError("Stage-A summary missing generated count.")
                if not summary.regulator:
                    raise ValueError("Stage-A summary missing regulator.")
                regulator = summary.regulator
                if summary.candidates_with_hit is None:
                    raise ValueError("Stage-A summary missing candidates_with_hit.")
                if summary.eligible_raw is None:
                    raise ValueError("Stage-A summary missing eligible_raw.")
                if summary.eligible_unique is None:
                    raise ValueError("Stage-A summary missing eligible_unique.")
                if summary.retained is None:
                    raise ValueError("Stage-A summary missing retained count.")
                if summary.eligible_tier_counts is None or summary.retained_tier_counts is None:
                    raise ValueError("Stage-A summary missing tier counts.")
                if len(summary.eligible_tier_counts) != len(summary.retained_tier_counts):
                    raise ValueError("Stage-A summary tier counts length mismatch.")
                if summary.tier_fractions is None:
                    raise ValueError("Stage-A summary missing tier fractions.")
                tier_fractions = list(summary.tier_fractions)
                if len(summary.retained_tier_counts) not in {len(tier_fractions), len(tier_fractions) + 1}:
                    raise ValueError("Stage-A summary tier fraction/count length mismatch.")
                if summary.selection_policy is None:
                    raise ValueError("Stage-A summary missing selection policy.")
                if summary.diversity is None:
                    raise ValueError("Stage-A diversity summary missing.")
                candidates = _format_count(summary.generated)
                has_hit = _format_ratio(summary.candidates_with_hit, summary.generated)
                eligible_raw = _format_ratio(summary.eligible_raw, summary.generated)
                eligible_unique = _format_ratio(summary.eligible_unique, summary.eligible_raw)
                retained = _format_count(summary.retained)
                tier_target = "-"
                if summary.tier_target_fraction is not None:
                    frac = float(summary.tier_target_fraction)
                    frac_label = f"{frac:.3%}"
                    if summary.tier_target_met is True:
                        tier_target = f"{frac_label} met"
                    elif summary.tier_target_met is False:
                        tier_target = f"{frac_label} unmet"
                selection_label = format_selection_label(
                    policy=str(summary.selection_policy),
                    alpha=summary.selection_alpha,
                    relevance_norm=summary.selection_relevance_norm or "minmax_raw_score",
                )
                tier_counts = _format_tier_counts(summary.eligible_tier_counts, summary.retained_tier_counts)
                tier_fill = "-"
                if summary.retained_tier_counts:
                    tier_labels = [_format_tier_fraction_label(frac) for frac in tier_fractions]
                    if len(summary.retained_tier_counts) == len(tier_fractions) + 1:
                        tier_labels.append("rest")
                    last_idx = None
                    for idx, val in enumerate(summary.retained_tier_counts):
                        if int(val) > 0:
                            last_idx = idx
                    if last_idx is not None:
                        tier_fill = tier_labels[last_idx]
                length_label = _format_sampling_lengths(
                    min_len=summary.retained_len_min,
                    median_len=summary.retained_len_median,
                    mean_len=summary.retained_len_mean,
                    max_len=summary.retained_len_max,
                    count=summary.retained,
                )
                score_label = _format_score_stats(
                    min_score=summary.retained_score_min,
                    median_score=summary.retained_score_median,
                    mean_score=summary.retained_score_mean,
                    max_score=summary.retained_score_max,
                )
                diversity = summary.diversity
                core_hamming = diversity.core_hamming
                pairwise = core_hamming.pairwise
                if pairwise is None:
                    raise ValueError("Stage-A diversity missing pairwise summary.")
                top_pairwise = pairwise.top_candidates
                diversified_pairwise = pairwise.diversified_candidates
                if int(diversified_pairwise.n_pairs) <= 0 or int(top_pairwise.n_pairs) <= 0:
                    pairwise_top_label = "n/a"
                    pairwise_div_label = "n/a"
                else:
                    pairwise_top_label = _format_diversity_value(top_pairwise.median)
                    pairwise_div_label = _format_diversity_value(diversified_pairwise.median)
                score_block = diversity.score_quantiles
                if score_block.top_candidates is None or score_block.diversified_candidates is None:
                    raise ValueError("Stage-A diversity missing top/diversified score quantiles.")
                score_norm_top = _format_score_norm_triplet(diversity.score_norm_summary, label="top_candidates")
                score_norm_div = _format_score_norm_triplet(
                    diversity.score_norm_summary, label="diversified_candidates"
                )
                if diversity.set_overlap_fraction is None or diversity.set_overlap_swaps is None:
                    raise ValueError("Stage-A diversity missing overlap stats.")
                diversity_overlap = f"{float(diversity.set_overlap_fraction) * 100:.1f}%"
                diversity_swaps = str(int(diversity.set_overlap_swaps))
                if diversity.candidate_pool_size is None:
                    raise ValueError("Stage-A diversity missing pool size.")
                pool_label = str(int(diversity.candidate_pool_size))
                pool_source = "-"
                if summary.selection_pool_capped:
                    pool_label = f"{pool_label}*"
                    if summary.selection_pool_cap_value is not None:
                        pool_source = f"cap={int(summary.selection_pool_cap_value)}"
                elif summary.selection_pool_rung_fraction_used is not None:
                    pool_source = f"rung={float(summary.selection_pool_rung_fraction_used) * 100:.3f}%"
                diversity_pool = pool_label
                rows.append(
                    {
                        "input_name": str(input_name),
                        "regulator": str(regulator),
                        "generated": candidates,
                        "has_hit": has_hit,
                        "eligible_raw": eligible_raw,
                        "eligible_unique": eligible_unique,
                        "retained": retained,
                        "tier_fill": tier_fill,
                        "tier_counts": tier_counts,
                        "tier_target": tier_target,
                        "selection": selection_label,
                        "score": score_label,
                        "length": length_label,
                        "pairwise_top": pairwise_top_label,
                        "pairwise_div": pairwise_div_label,
                        "score_norm_top": score_norm_top,
                        "score_norm_div": score_norm_div,
                        "set_overlap": diversity_overlap,
                        "set_swaps": diversity_swaps,
                        "diversity_pool": diversity_pool,
                        "diversity_pool_source": pool_source,
                        "tier0_score": summary.tier0_score,
                        "tier1_score": summary.tier1_score,
                        "tier2_score": summary.tier2_score,
                    }
                )
            continue
        total = len(pool.sequences)
        lengths = [len(seq) for seq in pool.sequences]
        if lengths:
            arr = np.asarray(lengths, dtype=float)
            length_label = _format_sampling_lengths(
                min_len=int(arr.min()),
                median_len=float(np.median(arr)),
                mean_len=float(arr.mean()),
                max_len=int(arr.max()),
                count=int(total),
            )
        else:
            length_label = _format_sampling_lengths(
                min_len=None,
                median_len=None,
                mean_len=None,
                max_len=None,
                count=int(total),
            )
        rows.append(
            {
                "input_name": str(pool.name),
                "regulator": "-",
                "generated": _format_count(total),
                "has_hit": _format_ratio(total, total),
                "eligible_raw": _format_ratio(total, total),
                "eligible_unique": _format_ratio(total, total),
                "retained": _format_count(total),
                "tier_fill": "-",
                "tier_counts": "-",
                "tier_target": "-",
                "selection": "-",
                "score": "-",
                "length": length_label,
                "pairwise_top": "-",
                "pairwise_div": "-",
                "score_norm_top": "-",
                "score_norm_div": "-",
                "set_overlap": "-",
                "set_swaps": "-",
                "diversity_pool": "-",
                "diversity_pool_source": "-",
                "tier0_score": None,
                "tier1_score": None,
                "tier2_score": None,
            }
        )
    rows.sort(key=lambda row: (row["input_name"], row["regulator"]))
    return rows


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
        hints.append(
            "If this is a staged run dir, use `dense workspace init --copy-inputs` or copy files into run/inputs."
        )
    if hints:
        console.print("[bold]Next steps[/]:")
        for hint in hints:
            console.print(f"  - {hint}")


def _render_output_schema_hint(exc: Exception) -> bool:
    msg = str(exc)
    if "Existing Parquet schema does not match the current DenseGen schema" in msg:
        console.print(f"[bold red]Output schema mismatch:[/] {msg}")
        console.print("[bold]Next steps[/]:")
        console.print("  - Remove outputs/tables/dense_arrays.parquet and outputs/meta/_densegen_ids.sqlite, or")
        console.print("  - Stage a fresh workspace with `dense workspace init --copy-inputs` and re-run.")
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
    console.print(
        "[yellow]Warning:[/] pool_strategy=full uses the entire input library; " + ", ".join(ignored) + " are ignored."
    )


# ----------------- Typer CLI -----------------
app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="DenseGen — Dense Array Generator (Typer/Rich CLI)",
)
inspect_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Inspect configs, inputs, and runs.")
stage_a_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Stage-A helpers (input TFBS pools).")
stage_b_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Stage-B helpers (library sampling).")
workspace_app = typer.Typer(add_completion=False, no_args_is_help=True, help="Workspace scaffolding.")

app.add_typer(inspect_app, name="inspect")
app.add_typer(stage_a_app, name="stage-a")
app.add_typer(stage_b_app, name="stage-b")
app.add_typer(workspace_app, name="workspace")

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
register_inspect_commands(inspect_app, context=cli_context)
register_report_command(app, context=cli_context)


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


@app.command("validate-config", help="Validate the config YAML (schema + sanity).")
def validate_config(
    ctx: typer.Context,
    probe_solver: bool = typer.Option(False, help="Also probe the solver backend."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
):
    cfg_path, is_default = _resolve_config_path(ctx, config)
    loaded = _load_config_or_exit(
        cfg_path,
        missing_message=DEFAULT_CONFIG_MISSING_MESSAGE if is_default else None,
    )
    _warn_pwm_sampling_configs(loaded, cfg_path)
    _warn_full_pool_strategy(loaded)
    explicit_cfg = bool(
        config or (ctx.obj and ctx.obj.get("config_path") is not None) or os.environ.get("DENSEGEN_CONFIG_PATH")
    )
    _ensure_fimo_available(loaded.root.densegen, strict=explicit_cfg)
    if probe_solver:
        from .adapters.optimizer import DenseArraysAdapter
        from .core.pipeline import select_solver

        solver_cfg = loaded.root.densegen.solver
        select_solver(
            solver_cfg.backend,
            DenseArraysAdapter(),
            strategy=str(solver_cfg.strategy),
        )
    console.print(":white_check_mark: [bold green]Config is valid.[/]")


@app.command("ls-plots", help="List available plot names and descriptions.")
def ls_plots():
    from .viz.plot_registry import PLOT_SPECS

    table = make_table("plot", "description")
    for name, meta in PLOT_SPECS.items():
        table.add_row(name, meta["description"])
    console.print(table)


@workspace_app.command("init", help="Stage a new workspace with config.yaml and standard subfolders.")
def workspace_init(
    run_id: str = typer.Option(..., "--id", "-i", help="Run identifier (directory name)."),
    root: Path = typer.Option(
        Path("."),
        "--root",
        help="Workspace root directory (default: current directory).",
    ),
    template_id: Optional[str] = typer.Option(
        None,
        "--template-id",
        help="Packaged template id (use to avoid repo-root paths).",
    ),
    template: Optional[Path] = typer.Option(None, "--template", help="Template config YAML to copy."),
    copy_inputs: bool = typer.Option(False, help="Copy file-based inputs into workspace/inputs and rewrite paths."),
):
    run_id_clean = _sanitize_filename(run_id)
    if run_id_clean != run_id:
        console.print(f"[yellow]Sanitized run id:[/] {run_id} -> {run_id_clean}")
    root_path = root.expanduser()
    if root_path.exists() and not root_path.is_dir():
        console.print(
            f"[bold red]Workspace root is not a directory:[/] {_display_path(root_path, Path.cwd(), absolute=False)}"
        )
        raise typer.Exit(code=1)
    run_dir = (root_path / run_id_clean).resolve()
    if run_dir.exists():
        console.print(
            f"[bold red]Run directory already exists:[/] {_display_path(run_dir, root_path.resolve(), absolute=False)}"
        )
        raise typer.Exit(code=1)

    with _resolve_template_dir(template=template, template_id=template_id) as (_template_dir, template_path):
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "inputs").mkdir(parents=True, exist_ok=True)
        (run_dir / "outputs" / "logs").mkdir(parents=True, exist_ok=True)
        (run_dir / "outputs" / "meta").mkdir(parents=True, exist_ok=True)
        (run_dir / "outputs" / "pools").mkdir(parents=True, exist_ok=True)
        (run_dir / "outputs" / "libraries").mkdir(parents=True, exist_ok=True)
        (run_dir / "outputs" / "tables").mkdir(parents=True, exist_ok=True)
        (run_dir / "outputs" / "plots").mkdir(parents=True, exist_ok=True)
        (run_dir / "outputs" / "report").mkdir(parents=True, exist_ok=True)

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
            output["parquet"]["path"] = "outputs/tables/dense_arrays.parquet"
        if "usr" in output and isinstance(output.get("usr"), dict):
            output["usr"]["root"] = "outputs/usr"
        dense["output"] = output

        logging_cfg = dense.get("logging") or {}
        logging_cfg["log_dir"] = "outputs/logs"
        dense["logging"] = logging_cfg

        if "plots" in raw and isinstance(raw.get("plots"), dict):
            raw["plots"]["out_dir"] = "outputs/plots"

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

        # Intentionally avoid copying auxiliary tools into the DenseGen workspace
        # to keep the workspace config-centric and low-cognitive-load.

        config_path = run_dir / "config.yaml"
        config_path.write_text(yaml.safe_dump(raw, sort_keys=False))
        if not copy_inputs:
            rel_paths = _collect_relative_input_paths_from_raw(dense)
            if rel_paths:
                console.print(
                    "[yellow]Workspace uses file-based inputs with relative paths.[/]"
                    " They will resolve relative to the new workspace."
                )
                for rel_path in rel_paths[:6]:
                    console.print(f"  - {rel_path}")
                console.print("[yellow]Tip[/]: re-run with --copy-inputs or update paths in config.yaml.")
        console.print(
            f":sparkles: [bold green]Workspace staged[/]: {_display_path(config_path, run_dir, absolute=False)}"
        )


@stage_a_app.command("build-pool", help="Build Stage-A TFBS pools from inputs.")
def stage_a_build_pool(
    ctx: typer.Context,
    out: str = typer.Option(
        "outputs/pools",
        "--out",
        help="Output directory (relative to run root; must be inside outputs/).",
    ),
    n_sites: Optional[int] = typer.Option(
        None,
        "--n-sites",
        help="Override Stage-A PWM sampling n_sites for all PWM inputs.",
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size",
        help="Override Stage-A PWM mining batch_size for all PWM inputs.",
    ),
    max_seconds: Optional[float] = typer.Option(
        None,
        "--max-seconds",
        help="Override Stage-A PWM mining budget max_seconds for all PWM inputs.",
    ),
    input_name: Optional[list[str]] = typer.Option(
        None,
        "--input",
        "-i",
        help="Input name(s) to build (defaults to all inputs).",
    ),
    fresh: bool = typer.Option(
        False,
        "--fresh",
        help="Start from scratch and replace existing pool files.",
    ),
    show_motif_ids: bool = typer.Option(
        False,
        "--show-motif-ids",
        help="Show full motif IDs instead of TF names.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        help="Show verbose Stage-A recap columns.",
    ),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
):
    cfg_path, is_default = _resolve_config_path(ctx, config)
    loaded = _load_config_or_exit(
        cfg_path,
        missing_message=DEFAULT_CONFIG_MISSING_MESSAGE if is_default else None,
    )
    cfg = loaded.root.densegen
    _ensure_fimo_available(cfg, strict=True)
    run_root = _run_root_for(loaded)
    log_cfg = cfg.logging
    log_dir = _resolve_outputs_path_or_exit(
        loaded.path,
        run_root,
        Path(log_cfg.log_dir),
        label="logging.log_dir",
    )
    logfile = log_dir / f"{cfg.run.id}.stage_a.log"
    setup_logging(
        level=log_cfg.level,
        logfile=str(logfile),
        suppress_solver_stderr=bool(log_cfg.suppress_solver_stderr),
    )
    progress_style = str(log_cfg.progress_style)
    logging_utils.set_progress_style(progress_style)
    logging_utils.set_progress_enabled(progress_style in {"stream", "screen"})
    out_dir = _resolve_outputs_path_or_exit(cfg_path, run_root, out, label="stage-a.out")
    out_dir.mkdir(parents=True, exist_ok=True)

    selected = {name for name in (input_name or [])}
    if selected:
        available = {inp.name for inp in cfg.inputs}
        missing = sorted(selected - available)
        if missing:
            raise typer.BadParameter(f"Unknown input name(s): {', '.join(missing)}")
    if n_sites is not None and n_sites <= 0:
        raise typer.BadParameter("--n-sites must be > 0")
    if batch_size is not None and batch_size <= 0:
        raise typer.BadParameter("--batch-size must be > 0")
    if max_seconds is not None and max_seconds <= 0:
        raise typer.BadParameter("--max-seconds must be > 0")
    _apply_stage_a_overrides(
        cfg,
        selected=selected if selected else None,
        n_sites=n_sites,
        batch_size=batch_size,
        max_seconds=max_seconds,
    )

    seeds = derive_seed_map(int(cfg.runtime.random_seed), ["stage_a", "stage_b", "solver"])
    rng = np.random.default_rng(seeds["stage_a"])
    deps = default_deps()
    outputs_root = run_root / "outputs"
    outputs_root.mkdir(parents=True, exist_ok=True)
    candidate_logging = _candidate_logging_enabled(cfg, selected=set(selected) if selected else None)
    candidates_dir = candidates_root(outputs_root, cfg.run.id)
    if candidate_logging:
        try:
            existed = prepare_candidates_dir(candidates_dir, overwrite=fresh)
        except Exception as exc:
            console.print(f"[bold red]{exc}[/]")
            raise typer.Exit(code=1)
        if existed and fresh:
            console.print(
                f"[yellow]Cleared prior candidate artifacts at "
                f"{_display_path(candidates_dir, run_root, absolute=False)} to avoid mixing runs.[/]"
            )
        elif existed and not fresh:
            console.print(
                f"[yellow]Appending to existing candidate artifacts under {candidates_dir} (use --fresh to reset).[/]"
            )

    plan_rows = stage_a_plan_rows(
        cfg,
        cfg_path,
        selected if selected else None,
        show_motif_ids=show_motif_ids,
    )
    if plan_rows:
        console.print("[bold]Stage-A plan[/]")
        console.print(stage_a_plan_table(plan_rows))

    with _suppress_pyarrow_sysctl_warnings():
        try:
            artifact, pool_data = build_pool_artifact(
                cfg=cfg,
                cfg_path=cfg_path,
                deps=deps,
                rng=rng,
                outputs_root=outputs_root,
                out_dir=out_dir,
                overwrite=fresh,
                selected_inputs=selected if selected else None,
            )
        except FileExistsError as exc:
            console.print(f"[bold red]{exc}[/]")
            raise typer.Exit(code=1)
        except Exception as exc:
            console.print(f"[bold red]Failed to build Stage-A pools:[/] {exc}")
            raise typer.Exit(code=1)
        if candidate_logging:
            candidate_files = find_candidate_files(candidates_dir)
            if candidate_files:
                try:
                    build_candidate_artifact(
                        candidates_dir=candidates_dir,
                        cfg_path=cfg_path,
                        run_id=str(cfg.run.id),
                        run_root=run_root,
                        overwrite=True,
                    )
                except Exception as exc:
                    console.print(f"[bold red]Failed to write candidate artifacts:[/] {exc}")
                    raise typer.Exit(code=1)
            else:
                console.print(
                    f"[yellow]Candidate logging enabled but no candidate records found under {candidates_dir}.[/]"
                )

    display_map_by_input: dict[str, dict[str, str]] = {}
    for inp in cfg.inputs:
        motifs = input_motifs(inp, cfg_path)
        display_map_by_input[inp.name] = {motif_id: name for motif_id, name in motifs}

    recap_rows = _stage_a_sampling_rows(pool_data)
    if recap_rows:
        console.print("[bold]Stage-A sampling recap[/]")
        for title, table in stage_a_recap_tables(
            recap_rows,
            display_map_by_input=display_map_by_input,
            show_motif_ids=show_motif_ids,
            verbose=verbose,
        ):
            if title:
                console.print(f"[bold]{title}[/]")
            console.print(table)
        legend_rows = [
            ("generated", "PWM candidates"),
            ("eligible_unique", "deduped by uniqueness.key"),
            ("retained", "carried forward after selection"),
            ("tier fill", "deepest diagnostic tier used"),
            ("selection", "Stage-A selection policy"),
            ("pool", "MMR pool size after rung slice ('*' = capped)"),
            ("overlap", "top ∩ diversified"),
            ("pairwise top/div", "weighted Hamming median (tfbs_core)"),
            ("score_norm top/div", "min/med/max"),
        ]
        if verbose:
            legend_rows.insert(1, ("has_hit", "FIMO hit present"))
            legend_rows.insert(2, ("eligible_raw", "best_hit_score > 0 among hits"))
            legend_rows.append(("tier target", "diagnostic tier target status"))
            legend_rows.append(("set_swaps", "diversified - overlap"))
        legend_table = make_table("Legend", "Meaning")
        for key, desc in legend_rows:
            legend_table.add_row(str(key), str(desc))
        console.print("[bold]Legend[/]")
        console.print(legend_table)
    console.print(
        f":sparkles: [bold green]Pool manifest written[/]: "
        f"{_display_path(artifact.manifest_path, run_root, absolute=False)}"
    )


@stage_b_app.command("build-libraries", help="Build Stage-B libraries from pools or inputs.")
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
    cfg_path, is_default = _resolve_config_path(ctx, config)
    loaded = _load_config_or_exit(
        cfg_path,
        missing_message=DEFAULT_CONFIG_MISSING_MESSAGE if is_default else None,
    )
    cfg = loaded.root.densegen
    run_root = _run_root_for(loaded)
    out_dir = _resolve_outputs_path_or_exit(cfg_path, run_root, out, label="stage-b.out")
    out_dir.mkdir(parents=True, exist_ok=True)
    if overwrite and append:
        console.print("[bold red]Choose either --append or --overwrite, not both.[/]")
        raise typer.Exit(code=1)
    builds_path = out_dir / "library_builds.parquet"
    members_path = out_dir / "library_members.parquet"
    manifest_path = out_dir / "library_manifest.json"
    artifacts_exist = builds_path.exists() or members_path.exists() or manifest_path.exists()
    if artifacts_exist and not (overwrite or append):
        out_label = _display_path(out_dir, run_root, absolute=False)
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
        pool_dir = _resolve_outputs_path_or_exit(cfg_path, run_root, pool, label="stage-b.pool")
    else:
        pool_dir = run_root / "outputs" / "pools"
    if pool_dir.exists() and pool_dir.is_file():
        pool_label = _display_path(pool_dir, run_root, absolute=False)
        raise typer.BadParameter(f"Pool path must be a directory from `stage-a build-pool`, not a file: {pool_label}")
    if not pool_dir.exists() or not pool_dir.is_dir():
        pool_label = _display_path(pool_dir, run_root, absolute=False)
        raise typer.BadParameter(f"Pool directory not found: {pool_label}")
    try:
        pool_artifact, pool_data = load_pool_data(pool_dir)
    except FileNotFoundError as exc:
        console.print(f"[bold red]{exc}[/]")
        entries = _list_dir_entries(pool_dir, limit=10)
        if entries:
            console.print(f"[bold]Pool directory contents[/]: {', '.join(entries)}")
        console.print("[bold]Next steps[/]:")
        console.print(f"  - {_workspace_command('dense stage-a build-pool', cfg_path=cfg_path, run_root=run_root)}")
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
                    f"(manifest={_short_hash(existing_config_hash)}, current={_short_hash(config_hash)})"
                )
            if existing_pool_hash != pool_manifest_hash:
                console.print(
                    "  - pool manifest hash mismatch "
                    f"(manifest={_short_hash(existing_pool_hash)}, current={_short_hash(pool_manifest_hash)})"
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
            int(row.get("library_index") or 0) for row in existing_build_rows if row.get("library_index") is not None
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
    with _suppress_pyarrow_sysctl_warnings():
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
                console.print("  - ensure regulator_constraints group members match Stage-A regulator labels")
                console.print("  - inspect available regulators via dense inspect inputs")
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
    console.print(
        "Stage-B builds solver libraries from Stage-A pools (cached for `dense run`). "
        "Sites/TFs/bp totals summarize min/median/max across libraries."
    )
    console.print(f"libraries built now: {len(build_rows)}; libraries total: {libraries_total}")
    console.print(
        f":sparkles: [bold green]Library builds written[/]: "
        f"{_display_path(artifact.builds_path, run_root, absolute=False)}"
    )
    console.print(
        f":sparkles: [bold green]Library members written[/]: "
        f"{_display_path(artifact.members_path, run_root, absolute=False)}"
    )


@app.command(help="Run generation for the job. Optionally auto-run plots declared in YAML.")
def run(
    ctx: typer.Context,
    no_plot: bool = typer.Option(False, help="Do not auto-run plots even if configured."),
    fresh: bool = typer.Option(False, "--fresh", help="Clear outputs and start a new run."),
    resume: bool = typer.Option(False, "--resume", help="Resume from existing outputs."),
    rebuild_stage_a: bool = typer.Option(
        False,
        "--rebuild-stage-a",
        help="Rebuild Stage-A pools before running (required if pools are missing or stale).",
    ),
    log_file: Optional[Path] = typer.Option(
        None,
        help="Override logfile path (must be inside outputs/ under the run root).",
    ),
    show_tfbs: bool = typer.Option(False, "--show-tfbs", help="Show TFBS sequences in progress output."),
    show_solutions: bool = typer.Option(False, "--show-solutions", help="Show full solution sequences in output."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
):
    cfg_path, is_default = _resolve_config_path(ctx, config)
    loaded = _load_config_or_exit(
        cfg_path,
        missing_message=DEFAULT_CONFIG_MISSING_MESSAGE if is_default else None,
    )
    root = loaded.root
    cfg = root.densegen
    run_root = _run_root_for(loaded)
    if rebuild_stage_a:
        _ensure_fimo_available(cfg, strict=True)

    if fresh and resume:
        console.print("[bold red]Choose either --fresh or --resume, not both.[/]")
        raise typer.Exit(code=1)
    outputs_root = run_outputs_root(run_root)
    existing_outputs = has_existing_run_outputs(run_root)
    if fresh:
        if outputs_root.exists():
            try:
                shutil.rmtree(outputs_root)
            except Exception as exc:
                console.print(f"[bold red]Failed to clear outputs:[/] {exc}")
                raise typer.Exit(code=1)
            console.print(
                f":broom: [bold yellow]Cleared outputs[/]: {_display_path(outputs_root, run_root, absolute=False)}"
            )
        else:
            console.print("[yellow]No outputs directory found; starting fresh.[/]")
        resume_run = False
    elif resume:
        if not existing_outputs:
            console.print(
                f"[bold red]--resume requested but no outputs were found under[/] "
                f"{_display_path(outputs_root, run_root, absolute=False)}. "
                "Run without --resume or use --fresh to reset the workspace."
            )
            raise typer.Exit(code=1)
        resume_run = True
    else:
        if existing_outputs:
            console.print(
                f"[bold red]Existing outputs found under[/] "
                f"{_display_path(outputs_root, run_root, absolute=False)}. "
                "Use --resume to continue or --fresh to clear outputs."
            )
            raise typer.Exit(code=1)
        resume_run = False

    # Logging setup
    log_cfg = cfg.logging
    log_dir = _resolve_outputs_path_or_exit(
        loaded.path,
        run_root,
        Path(log_cfg.log_dir),
        label="logging.log_dir",
    )
    default_logfile = log_dir / f"{cfg.run.id}.log"
    if log_file is not None:
        logfile = _resolve_outputs_path_or_exit(loaded.path, run_root, log_file, label="logging.log_file")
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
        run_pipeline(
            loaded,
            resume=resume_run,
            build_stage_a=rebuild_stage_a,
            show_tfbs=show_tfbs,
            show_solutions=show_solutions,
        )
    except FileNotFoundError as exc:
        _render_missing_input_hint(cfg_path, loaded, exc)
        raise typer.Exit(code=1)
    except RuntimeError as exc:
        if _render_output_schema_hint(exc):
            raise typer.Exit(code=1)
        message = str(exc)
        if "Stage-A pools missing or stale" in message:
            console.print(f"[bold red]{message}[/]")
            console.print("[bold]Next steps[/]:")
            rebuild_cmd = _workspace_command(
                "dense stage-a build-pool --fresh",
                cfg_path=cfg_path,
                run_root=run_root,
            )
            console.print(f"  - {rebuild_cmd}")
            console.print("  - or rerun with --rebuild-stage-a to bootstrap pools")
            console.print(
                "  - Stage-B libraries are built during dense run; no need to run dense stage-b build-libraries"
            )
            raise typer.Exit(code=1)
        raise

    console.print(":tada: [bold green]Run complete[/].")
    console.print("[bold]Next steps[/]:")
    console.print(f"  - {_workspace_command('dense inspect run --library', cfg_path=cfg_path, run_root=run_root)}")
    console.print(f"  - {_workspace_command('dense report', cfg_path=cfg_path, run_root=run_root)}")

    # Auto-plot if configured
    if not no_plot and root.plots:
        try:
            ensure_mpl_cache_dir(run_root / "outputs" / ".mpl-cache")
        except Exception as exc:
            console.print(f"[bold red]Matplotlib cache setup failed:[/] {exc}")
            console.print("[bold]Tip[/]: set MPLCONFIGDIR=outputs/.mpl-cache inside the workspace.")
            raise typer.Exit(code=1)
        install_native_stderr_filters(suppress_solver_messages=False)
        from .viz.plotting import run_plots_from_config

        console.print("[bold]Generating plots...[/]")
        run_plots_from_config(root, loaded.path, source="run")
        console.print(":bar_chart: [bold green]Plots written.[/]")


@app.command("campaign-reset", hidden=True, help="Remove run outputs to reset a workspace.")
def campaign_reset(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
):
    cfg_path, is_default = _resolve_config_path(ctx, config)
    loaded = _load_config_or_exit(
        cfg_path,
        missing_message=DEFAULT_CONFIG_MISSING_MESSAGE if is_default else None,
    )
    run_root = resolve_run_root(loaded.path, loaded.root.densegen.run.root)
    outputs_root = run_outputs_root(run_root)
    if not outputs_root.exists():
        console.print(f"[bold yellow]No outputs found under[/] {_display_path(outputs_root, run_root, absolute=False)}")
        return
    if not outputs_root.is_dir():
        console.print(
            f"[bold red]Outputs path is not a directory:[/] {_display_path(outputs_root, run_root, absolute=False)}"
        )
        raise typer.Exit(code=1)
    shutil.rmtree(outputs_root)
    console.print(
        f":broom: [bold green]Removed outputs under[/] {_display_path(outputs_root, run_root, absolute=False)}"
    )


@app.command(help="Generate plots from outputs according to YAML. Use --only to select plots.")
def plot(
    ctx: typer.Context,
    only: Optional[str] = typer.Option(None, help="Comma-separated plot names (subset of available plots)."),
    absolute: bool = typer.Option(False, "--absolute", help="Show absolute paths instead of workspace-relative."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
):
    cfg_path, is_default = _resolve_config_path(ctx, config)
    loaded = _load_config_or_exit(
        cfg_path,
        missing_message=DEFAULT_CONFIG_MISSING_MESSAGE if is_default else None,
    )
    run_root = resolve_run_root(loaded.path, loaded.root.densegen.run.root)
    try:
        ensure_mpl_cache_dir(run_root / "outputs" / ".mpl-cache")
    except Exception as exc:
        console.print(f"[bold red]Matplotlib cache setup failed:[/] {exc}")
        console.print("[bold]Tip[/]: set MPLCONFIGDIR=outputs/.mpl-cache inside the workspace.")
        raise typer.Exit(code=1)
    install_native_stderr_filters(suppress_solver_messages=False)
    from .viz.plotting import run_plots_from_config

    run_plots_from_config(loaded.root, loaded.path, only=only, source="plot", absolute=absolute)
    console.print(":bar_chart: [bold green]Plots written.[/]")


if __name__ == "__main__":
    app()
