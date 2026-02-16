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
  - notebook        : Generate/run workspace-scoped marimo notebooks.

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
import re
import shlex
import shutil
import sys
import tempfile
from functools import partial
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import typer
from rich.console import Console
from rich.traceback import install as rich_traceback

from .cli_commands.config import register_validate_command
from .cli_commands.context import CliContext
from .cli_commands.inspect import register_inspect_commands
from .cli_commands.notebook import register_notebook_commands
from .cli_commands.plots import register_plot_commands
from .cli_commands.run import register_run_commands
from .cli_commands.stage_a import register_stage_a_commands
from .cli_commands.stage_b import register_stage_b_commands
from .cli_commands.workspace import register_workspace_commands
from .cli_commands.workspace_sources import resolve_workspace_source as _resolve_workspace_source_impl
from .cli_sampling import format_selection_label
from .cli_setup import (
    DEFAULT_CONFIG_FILENAME,
)
from .cli_setup import (
    ensure_fimo_available as _ensure_fimo_available_impl,
)
from .cli_setup import (
    load_config_or_exit as _load_config_or_exit_impl,
)
from .cli_setup import (
    resolve_config_path as _resolve_config_path_impl,
)
from .cli_setup import (
    resolve_outputs_path_or_exit as _resolve_outputs_path_or_exit_impl,
)
from .config import resolve_relative_path, resolve_run_root
from .core.artifacts.pool import PoolData
from .core.run_paths import display_path
from .core.stage_a.stage_a_summary import PWMSamplingSummary
from .utils.logging_utils import install_native_stderr_filters
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


def _workspace_command(command: str, *, cfg_path: Path | None = None, run_root: Path | None = None) -> str:
    resolved_command = _resolve_workspace_hint_command(command)
    root = run_root
    base = Path.cwd()
    if root is not None:
        try:
            root_resolved = root.resolve()
        except Exception:
            root_resolved = root
        if root_resolved == base.resolve():
            return resolved_command
        candidate = root / DEFAULT_CONFIG_FILENAME
        if candidate.exists():
            root_label = display_path(root, base, absolute=False)
            return f"cd {shlex.quote(root_label)} && {resolved_command}"
    if cfg_path is not None:
        cfg_label = display_path(cfg_path, base, absolute=False)
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
    return _resolve_workspace_source_impl(
        source_config=source_config,
        source_workspace=source_workspace,
        console=console,
        display_path=_display_path,
        default_config_filename=DEFAULT_CONFIG_FILENAME,
    )


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


def _stage_a_row_from_pwm_summary(*, summary: PWMSamplingSummary, pool_name: str) -> dict[str, object]:
    input_name = summary.input_name or pool_name
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
    score_norm_div = _format_score_norm_triplet(diversity.score_norm_summary, label="diversified_candidates")
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
    return {
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


def _stage_a_row_from_sequence_pool(*, pool: PoolData) -> dict[str, object]:
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
    return {
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
                rows.append(_stage_a_row_from_pwm_summary(summary=summary, pool_name=str(pool.name)))
            continue
        rows.append(_stage_a_row_from_sequence_pool(pool=pool))
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
    console.print(
        "[yellow]Warning:[/] pool_strategy=full uses the entire input library; " + ", ".join(ignored) + " are ignored."
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
register_inspect_commands(inspect_app, context=cli_context)
register_validate_command(
    app,
    context=cli_context,
    warn_pwm_sampling_configs=_warn_pwm_sampling_configs,
    ensure_fimo_available=_ensure_fimo_available,
)
register_plot_commands(app, context=cli_context)
register_workspace_commands(
    workspace_app,
    context=cli_context,
    resolve_workspace_source=_resolve_workspace_source,
    sanitize_filename=_sanitize_filename,
    collect_relative_input_paths_from_raw=_collect_relative_input_paths_from_raw,
)
register_stage_a_commands(
    stage_a_app,
    context=cli_context,
    apply_stage_a_overrides=_apply_stage_a_overrides,
    ensure_fimo_available=_ensure_fimo_available,
    candidate_logging_enabled=_candidate_logging_enabled,
    stage_a_sampling_rows=_stage_a_sampling_rows,
)
register_stage_b_commands(stage_b_app, context=cli_context, short_hash=_short_hash)
register_notebook_commands(notebook_app, context=cli_context)
register_run_commands(
    app,
    context=cli_context,
    render_missing_input_hint=_render_missing_input_hint,
    render_output_schema_hint=_render_output_schema_hint,
    ensure_fimo_available=_ensure_fimo_available,
)


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
