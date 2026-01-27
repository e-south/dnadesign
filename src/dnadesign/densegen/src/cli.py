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
import json
import logging
import os
import platform
import random
import re
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import typer
import yaml
from rich.console import Console
from rich.table import Table
from rich.traceback import install as rich_traceback

from dnadesign.cruncher.io.parsers.meme import parse_meme_file

from .adapters.sources.pwm_jaspar import _parse_jaspar
from .adapters.sources.pwm_sampling import PWMSamplingSummary
from .config import (
    LATEST_SCHEMA_VERSION,
    ConfigError,
    PWMMiningConfig,
    load_config,
    resolve_outputs_scoped_path,
    resolve_relative_path,
    resolve_run_root,
)
from .core.artifacts.candidates import build_candidate_artifact, find_candidate_files, prepare_candidates_dir
from .core.artifacts.library import write_library_artifact
from .core.artifacts.pool import (
    POOL_MODE_SEQUENCE,
    POOL_MODE_TFBS,
    PoolData,
    build_pool_artifact,
    load_pool_artifact,
    pool_status_by_input,
)
from .core.pipeline import default_deps, resolve_plan, run_pipeline
from .core.pipeline.attempts import _load_existing_library_index, _load_failure_counts_from_attempts
from .core.pipeline.outputs import _emit_event
from .core.pipeline.stage_b import build_library_for_plan
from .core.reporting import collect_report_data, write_report
from .core.run_manifest import load_run_manifest
from .core.run_paths import (
    candidates_root,
    has_existing_run_outputs,
    run_manifest_path,
    run_outputs_root,
    run_state_path,
)
from .core.run_state import load_run_state
from .core.seeding import derive_seed_map
from .integrations.meme_suite import require_executable
from .utils import logging_utils
from .utils.logging_utils import install_native_stderr_filters, setup_logging
from .utils.mpl_utils import ensure_mpl_cache_dir

rich_traceback(show_locals=False)
console = Console()
_PYARROW_SYSCTL_PATTERN = re.compile(r"sysctlbyname failed for 'hw\.")
log = logging.getLogger(__name__)
install_native_stderr_filters(suppress_solver_messages=False)

DEFAULT_CONFIG_FILENAME = "config.yaml"
DEFAULT_CONFIG_MISSING_MESSAGE = (
    "No config found. cd into a workspace containing config.yaml, or pass -c path/to/config.yaml."
)
PACKAGED_TEMPLATES: dict[str, str] = {
    "demo_meme_two_tf": "workspaces/demo_meme_two_tf",
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
                console.print(f"[bold red]Template config not found:[/] {config_path}")
                raise typer.Exit(code=1)
            yield Path(resolved), config_path
        return
    if template is None:
        console.print("[bold red]No template provided.[/] Use --template-id or --template.")
        raise typer.Exit(code=1)
    template_path = template.expanduser().resolve()
    if not template_path.exists():
        console.print(f"[bold red]Template config not found:[/] {template_path}")
        raise typer.Exit(code=1)
    if not template_path.is_file():
        console.print(f"[bold red]Template path is not a file:[/] {template_path}")
        raise typer.Exit(code=1)
    yield template_path.parent, template_path


def _input_uses_fimo(input_cfg) -> bool:
    sampling = getattr(input_cfg, "sampling", None)
    backend = str(getattr(sampling, "scoring_backend", "densegen")).lower() if sampling is not None else ""
    if backend == "fimo":
        return True
    overrides = getattr(input_cfg, "overrides_by_motif_id", None)
    if isinstance(overrides, dict):
        for override in overrides.values():
            try:
                override_backend = str(override.get("scoring_backend", "")).lower()
            except Exception:
                continue
            if override_backend == "fimo":
                return True
    return False


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
    oversample_factor: int | None,
    batch_size: int | None,
    max_seconds: float | None,
) -> None:
    if n_sites is None and oversample_factor is None and batch_size is None and max_seconds is None:
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
        if n_sites is not None:
            sampling_updates["n_sites"] = n_sites
        if oversample_factor is not None:
            sampling_updates["oversample_factor"] = oversample_factor
        mining_updates: dict = {}
        if batch_size is not None:
            mining_updates["batch_size"] = batch_size
        if max_seconds is not None:
            mining_updates["max_seconds"] = max_seconds
        if mining_updates:
            mining = sampling.mining or PWMMiningConfig()
            sampling_updates["mining"] = mining.model_copy(update=mining_updates)
        if sampling_updates:
            inp.sampling = sampling.model_copy(update=sampling_updates)
        overrides = getattr(inp, "overrides_by_motif_id", None)
        if isinstance(overrides, dict) and overrides:
            new_overrides = {}
            for motif_id, override in overrides.items():
                override_updates: dict = {}
                if n_sites is not None:
                    override_updates["n_sites"] = n_sites
                if oversample_factor is not None:
                    override_updates["oversample_factor"] = oversample_factor
                if mining_updates:
                    mining = override.mining or PWMMiningConfig()
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


def _workspace_command(command: str, *, cfg_path: Path | None = None, run_root: Path | None = None) -> str:
    root = run_root or (cfg_path.parent if cfg_path is not None else None)
    if root is not None:
        try:
            root_resolved = root.resolve()
        except Exception:
            root_resolved = root
        if root_resolved == Path.cwd().resolve():
            return command
        candidate = root / DEFAULT_CONFIG_FILENAME
        if candidate.exists():
            return f"cd {root} && {command}"
    if cfg_path is not None:
        return f"{command} -c {cfg_path}"
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
    return _default_config_path(), True


def _load_config_or_exit(cfg_path: Path, *, missing_message: str | None = None):
    try:
        return load_config(cfg_path)
    except FileNotFoundError:
        if missing_message:
            console.print(f"[bold red]{missing_message}[/]")
        else:
            console.print(f"[bold red]Config file not found:[/] {cfg_path}")
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
    if absolute:
        return str(path)
    try:
        return str(path.relative_to(run_root))
    except ValueError:
        return os.path.relpath(path, run_root)


def _input_kind_label(input_type: str) -> str:
    labels = {
        "binding_sites": "Binding sites table",
        "sequence_library": "Sequence library",
        "pwm_meme": "PWM MEME",
        "pwm_meme_set": "PWM MEME set",
        "pwm_jaspar": "PWM JASPAR",
        "pwm_matrix_csv": "PWM matrix CSV",
        "pwm_artifact": "PWM artifacts",
        "pwm_artifact_set": "PWM artifacts",
    }
    return labels.get(input_type, input_type.replace("_", " "))


def _motif_display_name(motif_id: str, tf_name: str | None) -> str:
    if tf_name and str(tf_name).strip():
        return str(tf_name).strip()
    if "_" in motif_id:
        return motif_id.split("_", 1)[0]
    return motif_id


def _unique_preserve(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for val in values:
        if val in seen:
            continue
        seen.add(val)
        out.append(val)
    return out


def _artifact_motif_metadata(path: Path) -> tuple[str, str | None]:
    raw = json.loads(path.read_text())
    motif_id = raw.get("motif_id")
    if not motif_id or not str(motif_id).strip():
        raise ValueError(f"PWM artifact missing motif_id: {path}")
    tf_name = raw.get("tf_name")
    tf_name = str(tf_name).strip() if tf_name and str(tf_name).strip() else None
    return str(motif_id).strip(), tf_name


def _meme_motif_ids(path: Path, motif_ids: list[str] | None) -> list[str]:
    result = parse_meme_file(path)
    motifs = _filter_meme_motifs(result.motifs, motif_ids)
    labels: list[str] = []
    for motif in motifs:
        label = (
            getattr(motif, "motif_id", None)
            or getattr(motif, "motif_name", None)
            or getattr(motif, "motif_label", None)
        )
        if label:
            labels.append(str(label))
    return labels


def _input_motifs(
    inp,
    cfg_path: Path,
) -> list[tuple[str, str]]:
    input_type = str(inp.type)
    motifs: list[tuple[str, str]] = []
    if input_type == "pwm_meme":
        path = resolve_relative_path(cfg_path, getattr(inp, "path"))
        for motif_id in _meme_motif_ids(path, getattr(inp, "motif_ids", None)):
            motifs.append((motif_id, _motif_display_name(motif_id, None)))
    elif input_type == "pwm_meme_set":
        for raw in getattr(inp, "paths", []) or []:
            path = resolve_relative_path(cfg_path, raw)
            for motif_id in _meme_motif_ids(path, getattr(inp, "motif_ids", None)):
                motifs.append((motif_id, _motif_display_name(motif_id, None)))
    elif input_type == "pwm_jaspar":
        path = resolve_relative_path(cfg_path, getattr(inp, "path"))
        for motif in _jaspar_motif_labels(path, getattr(inp, "motif_ids", None)):
            motifs.append((motif, _motif_display_name(motif, None)))
    elif input_type == "pwm_matrix_csv":
        motif_id = getattr(inp, "motif_id", None)
        if motif_id:
            motif_id = str(motif_id)
            motifs.append((motif_id, _motif_display_name(motif_id, None)))
    elif input_type == "pwm_artifact":
        path = resolve_relative_path(cfg_path, getattr(inp, "path"))
        motif_id, tf_name = _artifact_motif_metadata(path)
        motifs.append((motif_id, _motif_display_name(motif_id, tf_name)))
    elif input_type == "pwm_artifact_set":
        for raw in getattr(inp, "paths", []) or []:
            path = resolve_relative_path(cfg_path, raw)
            motif_id, tf_name = _artifact_motif_metadata(path)
            motifs.append((motif_id, _motif_display_name(motif_id, tf_name)))
    return motifs


def _print_inputs_summary(
    loaded,
    *,
    verbose: bool,
    absolute: bool,
    show_motif_ids: bool,
) -> None:
    cfg = loaded.root.densegen
    run_root = _run_root_for(loaded)
    statuses = pool_status_by_input(cfg, loaded.path, run_root)

    table = Table("input", "kind", "motifs", "source", "stage-a pool")
    for inp in cfg.inputs:
        input_type = str(inp.type)
        kind = _input_kind_label(input_type)
        motifs = _input_motifs(inp, loaded.path)
        if show_motif_ids:
            motif_labels = _unique_preserve([m_id for m_id, _name in motifs if m_id])
        else:
            motif_labels = _unique_preserve([_name for _m_id, _name in motifs if _name])
        motif_summary = "-"
        if motif_labels:
            if verbose or len(motif_labels) <= 6:
                motif_summary = f"{len(motif_labels)} ({','.join(motif_labels)})"
            else:
                motif_summary = f"{len(motif_labels)} motifs"

        source_label = "-"
        if hasattr(inp, "path"):
            resolved = resolve_relative_path(loaded.path, getattr(inp, "path"))
            source_label = _display_path(resolved, run_root, absolute=absolute)
        elif hasattr(inp, "paths"):
            resolved = [resolve_relative_path(loaded.path, p) for p in getattr(inp, "paths") or []]
            parents = {p.parent for p in resolved} if resolved else set()
            if parents:
                root = parents.pop() if len(parents) == 1 else None
                if root is not None:
                    prefix = _display_path(root, run_root, absolute=absolute)
                    if verbose:
                        names = ", ".join(sorted(p.name for p in resolved))
                        source_label = f"{prefix} ({len(resolved)} files): {names}"
                    else:
                        source_label = f"{prefix} ({len(resolved)} files)"
                else:
                    source_label = f"{len(resolved)} files (multiple dirs)"
            else:
                source_label = "0 files"
        elif hasattr(inp, "dataset"):
            root_path = resolve_relative_path(loaded.path, getattr(inp, "root"))
            root_label = _display_path(root_path, run_root, absolute=absolute)
            source_label = f"{inp.dataset} (root={root_label})"
        status = statuses.get(inp.name)
        if status is None:
            status_label = "-"
        else:
            if status.state == "present":
                status_label = "present ✓"
            elif status.state == "stale":
                status_label = "stale !"
            else:
                status_label = "missing"
        table.add_row(str(inp.name), kind, motif_summary, source_label, status_label)
    console.print("[bold]Stage-A input sources[/]")
    console.print(table)
    console.print("Legend: motifs = TF display names; source is workspace-relative.")
    console.print("Legend: stage-a pool reflects whether the pool matches the current config.")
    console.print("Tip: run `dense stage-a build-pool --fresh` to rebuild pools.")


def _format_sampling_ratio(value: int, target: int | None) -> str:
    if target is None or target <= 0:
        return str(int(value))
    return f"{int(value)}/{int(target)}"


def _format_count(value: int | None) -> str:
    if value is None:
        return "-"
    return f"{int(value):,}"


def _format_eligible(eligible: int | None, generated: int | None) -> str:
    if eligible is None:
        return "-"
    if generated is None or generated <= 0:
        return _format_count(eligible)
    pct = 100.0 * float(eligible) / float(generated)
    return f"{_format_count(eligible)} ({pct:.0f}%)"


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


def _format_tier_counts(eligible: list[int] | None, retained: list[int] | None) -> str:
    if not eligible or not retained or len(eligible) != len(retained):
        return "-"
    parts = []
    for idx in range(len(eligible)):
        parts.append(f"t{idx} {int(eligible[idx])}/{int(retained[idx])}")
    return " | ".join(parts)


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
                regulator = summary.regulator or "-"
                candidates = _format_count(summary.generated)
                eligible = _format_eligible(summary.eligible, summary.generated)
                retained = _format_count(summary.retained)
                tier_counts = _format_tier_counts(summary.eligible_tier_counts, summary.retained_tier_counts)
                tier_fill = "-"
                if summary.retained_tier_counts:
                    last_idx = None
                    for idx, val in enumerate(summary.retained_tier_counts):
                        if int(val) > 0:
                            last_idx = idx
                    if last_idx is not None:
                        tier_fill = {0: "0.1%", 1: "1%", 2: "9%", 3: "rest"}.get(last_idx, "-")
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
                rows.append(
                    {
                        "input_name": str(input_name),
                        "regulator": str(regulator),
                        "generated": candidates,
                        "eligible": eligible,
                        "retained": retained,
                        "tier_fill": tier_fill,
                        "tier_counts": tier_counts,
                        "score": score_label,
                        "length": length_label,
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
                "eligible": _format_eligible(total, total),
                "retained": _format_count(total),
                "tier_fill": "-",
                "tier_counts": "-",
                "score": "-",
                "length": length_label,
                "tier0_score": None,
                "tier1_score": None,
                "tier2_score": None,
            }
        )
    rows.sort(key=lambda row: (row["input_name"], row["regulator"]))
    return rows


def _filter_meme_motifs(motifs, motif_ids: list[str] | None) -> list:
    if not motif_ids:
        return list(motifs)
    keep = {m.strip().lower() for m in motif_ids if m}
    filtered = []
    for motif in motifs:
        cand = {
            getattr(motif, "motif_id", None),
            getattr(motif, "motif_name", None),
            getattr(motif, "motif_label", None),
        }
        cand = {str(x).strip().lower() for x in cand if x}
        if cand & keep:
            filtered.append(motif)
    return filtered


def _meme_motif_labels(path: Path, motif_ids: list[str] | None) -> list[str]:
    result = parse_meme_file(path)
    motifs = _filter_meme_motifs(result.motifs, motif_ids)
    labels = []
    for motif in motifs:
        label = (
            getattr(motif, "motif_id", None)
            or getattr(motif, "motif_name", None)
            or getattr(motif, "motif_label", None)
        )
        if label:
            labels.append(str(label))
    return labels


def _jaspar_motif_labels(path: Path, motif_ids: list[str] | None) -> list[str]:
    motifs = _parse_jaspar(path)
    if motif_ids:
        keep = {m for m in motif_ids if m}
        motifs = [m for m in motifs if m.motif_id in keep]
    return [m.motif_id for m in motifs if m.motif_id]


def _stage_a_plan_rows(
    cfg,
    cfg_path: Path,
    selected_inputs: set[str] | None,
    *,
    show_motif_ids: bool,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for inp in cfg.inputs:
        if selected_inputs and inp.name not in selected_inputs:
            continue
        input_type = str(inp.type)
        if not input_type.startswith("pwm_"):
            continue
        sampling = getattr(inp, "sampling", None)
        backend = str(getattr(sampling, "scoring_backend", "fimo")).lower() if sampling else "fimo"
        base_n_sites = getattr(sampling, "n_sites", None) if sampling else None
        base_oversample = getattr(sampling, "oversample_factor", None) if sampling else None
        length_policy = str(getattr(sampling, "length_policy", "-")) if sampling else "-"
        length_range = getattr(sampling, "length_range", None) if sampling else None
        length_label = length_policy
        if length_policy == "range" and length_range:
            length_label = f"range({length_range[0]}..{length_range[1]})"

        motifs = _input_motifs(inp, cfg_path)
        overrides = getattr(inp, "overrides_by_motif_id", None) if input_type == "pwm_artifact_set" else None
        for motif_id, display_name in motifs:
            reg_backend = backend
            reg_n_sites = base_n_sites
            reg_oversample = base_oversample
            if overrides and motif_id in overrides:
                override = overrides.get(motif_id) or {}
                if "scoring_backend" in override:
                    reg_backend = str(override.get("scoring_backend", reg_backend)).lower()
                if "n_sites" in override:
                    reg_n_sites = override.get("n_sites")
                if "oversample_factor" in override:
                    reg_oversample = override.get("oversample_factor")
            label = motif_id if show_motif_ids else display_name
            candidates = "-"
            if reg_n_sites is not None and reg_oversample is not None:
                candidates = f"{int(reg_n_sites) * int(reg_oversample)}"
            rows.append(
                {
                    "input": str(inp.name),
                    "tf": str(label),
                    "retain": str(reg_n_sites) if reg_n_sites is not None else "-",
                    "candidates": candidates,
                    "eligibility": "best_hit_score>0",
                    "backend": str(reg_backend),
                    "length": length_label,
                }
            )
    rows.sort(key=lambda row: (row["input"], row["tf"]))
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
                        "may fail uniqueness; consider reducing n_sites or using length_policy=range."
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


def _list_workspaces_table(
    workspaces_root: Path,
    *,
    limit: int,
    show_all: bool,
    absolute: bool,
) -> Table:
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
        plots_count = _count_files(run_dir / "outputs" / "plots", pattern="*")
        logs_count = _count_files(run_dir / "outputs" / "logs", pattern="*")

        if cfg_path.exists():
            try:
                loaded = load_config(cfg_path)
                run_id = loaded.root.densegen.run.id
                run_root = resolve_run_root(cfg_path, loaded.root.densegen.run.root)
                if loaded.root.densegen.output.parquet is not None:
                    pq_dir = resolve_outputs_scoped_path(
                        cfg_path,
                        run_root,
                        loaded.root.densegen.output.parquet.path,
                        label="output.parquet.path",
                    )
                    parquet_count = _count_files(pq_dir, pattern="*.parquet")
                plots_count = _count_files(
                    resolve_outputs_scoped_path(
                        cfg_path,
                        run_root,
                        loaded.root.plots.out_dir if loaded.root.plots else "outputs/plots",
                        label="plots.out_dir",
                    ),
                    pattern="*",
                )
                logs_count = _count_files(
                    resolve_outputs_scoped_path(
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

        config_label = "-"
        if cfg_path.exists():
            config_label = _display_path(cfg_path, workspaces_root, absolute=absolute)
        table.add_row(
            run_dir.name,
            run_id,
            config_label,
            str(parquet_count),
            str(plots_count),
            str(logs_count),
            status,
        )
    return table


def _read_events(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows: list[dict] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


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
    _ensure_fimo_available(loaded.root.densegen, strict=True)
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

    table = Table("plot", "description")
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
        console.print(f"[bold red]Workspace root is not a directory:[/] {root_path}")
        raise typer.Exit(code=1)
    run_dir = (root_path / run_id_clean).resolve()
    if run_dir.exists():
        console.print(f"[bold red]Run directory already exists:[/] {run_dir}")
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
        console.print(f":sparkles: [bold green]Workspace staged[/]: {config_path}")


@inspect_app.command("run", help="Summarize a run manifest or list workspaces.")
def inspect_run(
    ctx: typer.Context,
    run: Optional[Path] = typer.Option(None, "--run", "-r", help="Run directory (defaults to config run root)."),
    root: Optional[Path] = typer.Option(None, "--root", help="Workspaces root directory (lists workspaces)."),
    limit: int = typer.Option(0, "--limit", help="Limit workspaces displayed when using --root (0 = all)."),
    show_all: bool = typer.Option(False, "--all", help="Include directories without config.yaml when using --root."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
    absolute: bool = typer.Option(False, "--absolute", help="Show absolute paths instead of workspace-relative."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Show failure breakdown columns."),
    library: bool = typer.Option(False, "--library", help="Include offered-vs-used library summaries."),
    library_limit: int = typer.Option(10, "--library-limit", help="Limit libraries shown in summaries (0 = all)."),
    top: int = typer.Option(10, "--top", help="Rows to show for library summaries."),
    by_library: bool = typer.Option(True, "--by-library/--no-by-library", help="Group library summaries per build."),
    top_per_tf: Optional[int] = typer.Option(None, "--top-per-tf", help="Limit TFBS rows per TF when summarizing."),
    show_library_hash: bool = typer.Option(
        True,
        "--show-library-hash/--short-library-hash",
        help="Show full library hash (or short hash if disabled).",
    ),
    events: bool = typer.Option(False, "--events", help="Show events summary (stalls/resamples)."),
):
    if root is not None and run is not None:
        console.print("[bold red]Choose either --root or --run, not both.[/]")
        raise typer.Exit(code=1)
    if library_limit < 0:
        console.print("[bold red]--library-limit must be >= 0.[/]")
        raise typer.Exit(code=1)
    if root is not None:
        workspaces_root = root.resolve()
        if not workspaces_root.exists() or not workspaces_root.is_dir():
            console.print(f"[bold red]Workspaces root not found:[/] {workspaces_root}")
            raise typer.Exit(code=1)
        console.print(_list_workspaces_table(workspaces_root, limit=limit, show_all=show_all, absolute=absolute))
        return
    cfg_path = None
    loaded = None
    if run is None:
        cfg_path, is_default = _resolve_config_path(ctx, config)
        loaded = _load_config_or_exit(
            cfg_path,
            missing_message=DEFAULT_CONFIG_MISSING_MESSAGE if is_default else None,
        )
        run_root = _run_root_for(loaded)
    else:
        run_root = run
        if library:
            cfg_path = run_root / "config.yaml"
            if not cfg_path.exists():
                console.print(
                    f"[bold red]Config not found for --library:[/] "
                    f"{_display_path(cfg_path, run_root, absolute=absolute)}. "
                    "Provide --config or run inspect run without --library."
                )
                raise typer.Exit(code=1)
            loaded = _load_config_or_exit(cfg_path)
    manifest_path = run_manifest_path(run_root)
    if not manifest_path.exists():
        state_path = run_state_path(run_root)
        if state_path.exists():
            state = load_run_state(state_path)
            console.print("[yellow]Run manifest missing; showing checkpointed run_state.[/]")
            root_label = _display_path(run_root, run_root, absolute=absolute)
            console.print(
                f"[bold]Run:[/] {state.run_id}  [bold]Root:[/] {root_label}  "
                f"[bold]Schema:[/] {state.schema_version}  [bold]Config:[/] {state.config_sha256[:8]}…"
            )
            table = Table("input", "plan", "generated")
            for item in state.items:
                table.add_row(item.input_name, item.plan_name, str(item.generated))
            console.print(table)
            console.print("[bold]Next steps[/]:")
            console.print(f"  - {_workspace_command('dense run', cfg_path=cfg_path, run_root=run_root)}")
            return

        console.print(
            f"[bold red]Run manifest not found:[/] {_display_path(manifest_path, run_root, absolute=absolute)}"
        )
        entries = _list_dir_entries(run_root, limit=8)
        if entries:
            console.print(f"[bold]Run root contents[/]: {', '.join(entries)}")
        console.print("[bold]Next steps[/]:")
        console.print(f"  - {_workspace_command('dense run', cfg_path=cfg_path, run_root=run_root)}")
        raise typer.Exit(code=1)

    manifest = load_run_manifest(manifest_path)
    schema_label = manifest.schema_version or "-"
    dense_arrays_label = manifest.dense_arrays_version or "-"
    dense_arrays_source = manifest.dense_arrays_version_source or "-"
    if dense_arrays_label != "-" and dense_arrays_source != "-":
        dense_arrays_label = f"{dense_arrays_label} ({dense_arrays_source})"
    root_label = _display_path(run_root, run_root, absolute=absolute)
    console.print(
        f"[bold]Run:[/] {manifest.run_id}  [bold]Root:[/] {root_label}  "
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

    if events:
        events_path = run_root / "outputs" / "meta" / "events.jsonl"
        rows = _read_events(events_path)
        if not rows:
            console.print("[yellow]No events found.[/]")
        else:
            counts: dict[str, int] = {}
            last_seen: dict[str, str] = {}
            for entry in rows:
                name = str(entry.get("event") or "unknown")
                counts[name] = counts.get(name, 0) + 1
                created = str(entry.get("created_at") or "")
                if created:
                    prev = last_seen.get(name)
                    if prev is None or created > prev:
                        last_seen[name] = created
            events_table = Table("event", "count", "last_created_at")
            for name, count in sorted(counts.items(), key=lambda kv: kv[1], reverse=True):
                events_table.add_row(name, str(count), last_seen.get(name, "-"))
            console.print(events_table)

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
                console.print(f"  - {_workspace_command('dense run', cfg_path=cfg_path, run_root=run_root)}")
                raise typer.Exit(code=1)

        offered_vs_used_tf = bundle.tables.get("offered_vs_used_tf")
        offered_vs_used_tfbs = bundle.tables.get("offered_vs_used_tfbs")
        library_summary = bundle.tables.get("library_summary", pd.DataFrame())

        library_hashes = set()
        if isinstance(library_summary, pd.DataFrame) and not library_summary.empty:
            library_hashes = set(library_summary.get("library_hash", []))
        elif isinstance(offered_vs_used_tf, pd.DataFrame):
            library_hashes = set(offered_vs_used_tf.get("library_hash", []))

        total_libraries = 0
        display_count = 0
        truncated_libraries = False
        display_library_summary = library_summary
        display_hashes: list[str] = []
        if isinstance(library_summary, pd.DataFrame) and not library_summary.empty:
            display_library_summary = library_summary.sort_values("library_index")
            total_libraries = len(display_library_summary)
            if library_limit > 0 and total_libraries > library_limit:
                display_library_summary = display_library_summary.head(library_limit)
                truncated_libraries = True
            display_count = len(display_library_summary)
            display_hashes = [str(val) for val in display_library_summary.get("library_hash", [])]
        elif library_hashes:
            display_hashes = sorted(library_hashes)
            total_libraries = len(display_hashes)
            if library_limit > 0 and total_libraries > library_limit:
                display_hashes = display_hashes[:library_limit]
                truncated_libraries = True
            display_count = len(display_hashes)

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

        if isinstance(display_library_summary, pd.DataFrame) and not display_library_summary.empty:
            for _, row in display_library_summary.iterrows():
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
        elif display_hashes:
            for lib_hash in display_hashes:
                lib_hash_disp = lib_hash if show_library_hash else _short_hash(lib_hash)
                lib_table.add_row("-", lib_hash_disp, "-", "-", "-", f"-/{target_len}", "0")
        else:
            console.print("[yellow]No library attempts found (outputs/tables/attempts.parquet missing).[/]")
            entries = _list_dir_entries(run_root, limit=8)
            if entries:
                console.print(f"[bold]Run root contents[/]: {', '.join(entries)}")
            console.print("[bold]Next steps[/]:")
            console.print(f"  - {_workspace_command('dense run', cfg_path=cfg_path, run_root=run_root)}")
        console.print("[bold]Library build summary[/]")
        console.print(lib_table)
        if truncated_libraries:
            console.print(
                f"[yellow]Showing {display_count} of {total_libraries} libraries. Use --library-limit 0 to show all.[/]"
            )

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

        if by_library and isinstance(display_library_summary, pd.DataFrame) and not display_library_summary.empty:
            for _, row in display_library_summary.iterrows():
                lib_hash = str(row.get("library_hash") or "")
                lib_index = int(row.get("library_index") or 0)
                console.print(f"[bold]Library {lib_index}[/]: {_fmt_hash(lib_hash)}")
                _render_tf_tables(lib_hash)
                _render_tfbs_tables(lib_hash)
        elif by_library and display_hashes:
            for lib_hash in display_hashes:
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
    run: Optional[Path] = typer.Option(None, "--run", "-r", help="Run directory (defaults to config run root)."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
    out: str = typer.Option(
        "outputs/report",
        "--out",
        help="Output directory (relative to run root; must be inside outputs/).",
    ),
    absolute: bool = typer.Option(False, "--absolute", help="Show absolute paths instead of workspace-relative."),
    plots: str = typer.Option(
        "none",
        "--plots",
        help="Include plot links in the report: none or include (requires outputs/plots/plot_manifest.json).",
    ),
    strict: bool = typer.Option(
        False,
        "--strict",
        "--fail-on-missing",
        help="Fail if core report inputs are missing.",
    ),
    format: str = typer.Option(
        "all",
        "--format",
        "-f",
        help="Report format: json, md, html, or all (comma-separated allowed).",
    ),
):
    if run is not None and config is not None:
        console.print("[bold red]Choose either --run or --config, not both.[/]")
        raise typer.Exit(code=1)
    if run is not None:
        cfg_path = Path(run) / "config.yaml"
        if not cfg_path.exists():
            console.print(f"[bold red]Config not found under run:[/] {cfg_path}")
            raise typer.Exit(code=1)
        loaded = _load_config_or_exit(cfg_path)
    else:
        cfg_path, is_default = _resolve_config_path(ctx, config)
        loaded = _load_config_or_exit(
            cfg_path,
            missing_message=DEFAULT_CONFIG_MISSING_MESSAGE if is_default else None,
        )
    raw_formats = {f.strip().lower() for f in format.split(",") if f.strip()}
    if not raw_formats:
        raw_formats = {"all"}
    allowed_formats = {"json", "md", "html", "all"}
    unknown = sorted(raw_formats - allowed_formats)
    if unknown:
        console.print(f"[bold red]Unknown report format(s):[/] {', '.join(unknown)}")
        console.print("Allowed: json, md, html, all.")
        raise typer.Exit(code=1)
    plots_mode = str(plots or "none").strip().lower()
    if plots_mode not in {"none", "include"}:
        console.print("[bold red]--plots must be one of: none, include.[/]")
        raise typer.Exit(code=1)
    include_plots = plots_mode == "include"
    formats_used = {"json", "md", "html"} if "all" in raw_formats else raw_formats
    run_root = _run_root_for(loaded)
    out_dir = _resolve_outputs_path_or_exit(cfg_path, run_root, out, label="report.out")
    try:
        with _suppress_pyarrow_sysctl_warnings():
            write_report(
                loaded.root,
                cfg_path,
                out_dir=out_dir,
                include_plots=include_plots,
                strict=strict,
                formats=raw_formats,
            )
    except (FileNotFoundError, ValueError) as exc:
        console.print(f"[bold red]Report failed:[/] {exc}")
        entries = _list_dir_entries(run_root, limit=8)
        if entries:
            console.print(f"[bold]Run root contents[/]: {', '.join(entries)}")
        console.print("[bold]Next steps[/]:")
        if "plot_manifest" in str(exc):
            console.print(f"  - {_workspace_command('dense plot', cfg_path=cfg_path, run_root=run_root)}")
        else:
            console.print(f"  - {_workspace_command('dense run', cfg_path=cfg_path, run_root=run_root)}")
        raise typer.Exit(code=1)
    console.print(f":sparkles: [bold green]Report written[/]: {_display_path(out_dir, run_root, absolute=absolute)}")
    outputs = []
    if "json" in formats_used:
        outputs.append("report.json")
    if "md" in formats_used:
        outputs.append("report.md")
    if "html" in formats_used:
        outputs.append("report.html")
    console.print(f"[bold]Outputs[/]: {', '.join(outputs) if outputs else '-'}")


@inspect_app.command("plan", help="Show the resolved per-constraint quota plan.")
def inspect_plan(
    ctx: typer.Context,
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
):
    cfg_path, is_default = _resolve_config_path(ctx, config)
    loaded = _load_config_or_exit(
        cfg_path,
        missing_message=DEFAULT_CONFIG_MISSING_MESSAGE if is_default else None,
    )
    _warn_full_pool_strategy(loaded)
    pl = resolve_plan(loaded)
    table = Table("name", "quota", "has promoter_constraints")
    for item in pl:
        pcs = item.fixed_elements.promoter_constraints
        table.add_row(item.name, str(item.quota), "yes" if pcs else "no")
    console.print(table)


@inspect_app.command("config", help="Describe resolved config, outputs, and pipeline settings.")
def inspect_config(
    ctx: typer.Context,
    show_constraints: bool = typer.Option(False, help="Print full fixed elements per plan item."),
    probe_solver: bool = typer.Option(False, help="Probe the solver backend before reporting."),
    absolute: bool = typer.Option(False, "--absolute", help="Show absolute paths instead of workspace-relative."),
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

    if probe_solver:
        from .adapters.optimizer import DenseArraysAdapter
        from .core.pipeline import select_solver

        select_solver(
            cfg.solver.backend,
            DenseArraysAdapter(),
            strategy=str(cfg.solver.strategy),
        )

    console.print(f"[bold]Config[/]: {_display_path(loaded.path, run_root, absolute=absolute)}")
    console.print(f"[bold]Run[/]: id={cfg.run.id} root={_display_path(run_root, run_root, absolute=absolute)}")
    effective_path = run_root / "outputs" / "meta" / "effective_config.json"
    if effective_path.exists():
        console.print(f"[bold]Effective config[/]: {_display_path(effective_path, run_root, absolute=absolute)}")
    console.print("See `dense inspect inputs` for resolved input sources.")

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

    pwm_inputs = [inp for inp in cfg.inputs if str(getattr(inp, "type", "")).startswith("pwm_")]
    if pwm_inputs:
        stage_a_table = Table("input", "backend", "n_sites", "oversample", "candidates", "mining", "length")
        for inp in pwm_inputs:
            sampling = getattr(inp, "sampling", None)
            if sampling is None:
                continue
            backend = str(getattr(sampling, "scoring_backend", "fimo")).lower()
            n_sites = getattr(sampling, "n_sites", None)
            oversample = getattr(sampling, "oversample_factor", None)
            candidates = "-"
            if n_sites is not None and oversample is not None:
                candidates = str(int(n_sites) * int(oversample))
            mining_cfg = getattr(sampling, "mining", None)
            mining_label = "-"
            if mining_cfg is not None:
                parts = [f"batch={mining_cfg.batch_size}"]
                if mining_cfg.max_seconds is not None:
                    parts.append(f"max_seconds={mining_cfg.max_seconds}")
                if mining_cfg.log_every_batches is not None:
                    parts.append(f"log_every={mining_cfg.log_every_batches}")
                mining_label = ", ".join(parts)
            length_label = str(sampling.length_policy)
            if sampling.length_policy == "range" and sampling.length_range is not None:
                length_label = f"range({sampling.length_range[0]}..{sampling.length_range[1]})"
            stage_a_table.add_row(
                str(inp.name),
                backend,
                str(n_sites) if n_sites is not None else "-",
                str(oversample) if oversample is not None else "-",
                candidates,
                mining_label,
                length_label,
            )
        console.print("[bold]Stage-A sampling[/]")
        console.print(stage_a_table)
        console.print("Legend: candidates = n_sites * oversample_factor; eligibility = best_hit_score > 0.")

    outputs = Table("target", "path")
    for target in cfg.output.targets:
        if target == "parquet":
            parquet_path = resolve_outputs_scoped_path(
                loaded.path,
                run_root,
                cfg.output.parquet.path,
                label="output.parquet.path",
            )
            outputs.add_row(
                "parquet",
                _display_path(parquet_path, run_root, absolute=absolute),
            )
        elif target == "usr":
            usr_root = resolve_outputs_scoped_path(loaded.path, run_root, cfg.output.usr.root, label="output.usr.root")
            usr_root_label = _display_path(usr_root, run_root, absolute=absolute)
            outputs.add_row("usr", f"{cfg.output.usr.dataset} (root={usr_root_label})")
        else:
            outputs.add_row(target, "-")
    console.print(outputs)

    solver = Table("backend", "strategy", "time_limit_s", "threads", "strands")
    backend_display = str(cfg.solver.backend) if cfg.solver.backend is not None else "-"
    time_limit = "-" if cfg.solver.time_limit_seconds is None else str(cfg.solver.time_limit_seconds)
    threads = "-" if cfg.solver.threads is None else str(cfg.solver.threads)
    solver.add_row(backend_display, str(cfg.solver.strategy), time_limit, threads, str(cfg.solver.strands))
    console.print(solver)

    sampling = cfg.generation.sampling
    sampling_table = Table("setting", "value")
    target_length = cfg.generation.sequence_length + int(sampling.subsample_over_length_budget_by)
    sampling_table.add_row("pool_strategy", str(sampling.pool_strategy))
    sampling_table.add_row("library_source", str(sampling.library_source))
    if sampling.library_source == "artifact":
        sampling_table.add_row("library_artifact_path", str(sampling.library_artifact_path))
    sampling_table.add_row("library_size", str(sampling.library_size))
    sampling_table.add_row("library_sampling_strategy", str(sampling.library_sampling_strategy))
    sampling_table.add_row(
        "subsample_over_length_budget_by",
        f"{sampling.subsample_over_length_budget_by} (target={target_length} bp)",
    )
    sampling_table.add_row("cover_all_regulators", str(sampling.cover_all_regulators))
    sampling_table.add_row("unique_binding_sites", str(sampling.unique_binding_sites))
    sampling_table.add_row("unique_binding_cores", str(sampling.unique_binding_cores))
    sampling_table.add_row("max_sites_per_regulator", str(sampling.max_sites_per_regulator))
    sampling_table.add_row("relax_on_exhaustion", str(sampling.relax_on_exhaustion))
    sampling_table.add_row("allow_incomplete_coverage", str(sampling.allow_incomplete_coverage))
    sampling_table.add_row("iterative_max_libraries", str(sampling.iterative_max_libraries))
    sampling_table.add_row("iterative_min_new_solutions", str(sampling.iterative_min_new_solutions))
    sampling_table.add_row("arrays_generated_before_resample", str(cfg.runtime.arrays_generated_before_resample))
    sampling_table.add_row("max_resample_attempts", str(cfg.runtime.max_resample_attempts))
    sampling_table.add_row("max_total_resamples", str(cfg.runtime.max_total_resamples))
    console.print("[bold]Stage-B library sampling[/]")
    console.print(sampling_table)

    pad = cfg.postprocess.pad
    pad_gc = pad.gc
    if pad_gc.mode == "off":
        gc_label = "off"
    elif pad_gc.mode == "range":
        gc_label = f"range[{pad_gc.min:.2f}, {pad_gc.max:.2f}] min_pad_length={pad_gc.min_pad_length}"
    else:
        target_min = pad_gc.target - pad_gc.tolerance
        target_max = pad_gc.target + pad_gc.tolerance
        gc_label = (
            f"target={pad_gc.target:.2f}±{pad_gc.tolerance:.2f} "
            f"range=[{target_min:.2f}, {target_max:.2f}] "
            f"min_pad_length={pad_gc.min_pad_length}"
        )
    console.print(f"[bold]Pad[/]: mode={pad.mode} end={pad.end} gc={gc_label} max_tries={pad.max_tries}")
    log_dir = resolve_outputs_scoped_path(loaded.path, run_root, cfg.logging.log_dir, label="logging.log_dir")
    log_dir_label = _display_path(log_dir, run_root, absolute=absolute)
    log_table = Table("setting", "value")
    log_table.add_row("dir", log_dir_label)
    log_table.add_row("level", str(cfg.logging.level))
    log_table.add_row("progress_style", str(cfg.logging.progress_style))
    log_table.add_row("progress_every", str(cfg.logging.progress_every))
    log_table.add_row("progress_refresh_seconds", str(cfg.logging.progress_refresh_seconds))
    log_table.add_row("print_visual", str(cfg.logging.print_visual))
    log_table.add_row("show_tfbs", str(cfg.logging.show_tfbs))
    log_table.add_row("show_solutions", str(cfg.logging.show_solutions))
    log_table.add_row("suppress_solver_stderr", str(cfg.logging.suppress_solver_stderr))
    console.print("[bold]Logging[/]")
    console.print(log_table)

    if root.plots:
        out_dir = resolve_outputs_scoped_path(loaded.path, run_root, root.plots.out_dir, label="plots.out_dir")
        out_dir_label = _display_path(out_dir, run_root, absolute=absolute)
        console.print(f"[bold]Plots[/]: source={root.plots.source} out_dir={out_dir_label}")
    else:
        console.print("[bold]Plots[/]: none")


@inspect_app.command("inputs", help="Show resolved inputs and Stage-A pool status.")
def inspect_inputs(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", help="Show full source file lists."),
    absolute: bool = typer.Option(False, "--absolute", help="Show absolute paths instead of workspace-relative."),
    show_motif_ids: bool = typer.Option(False, "--show-motif-ids", help="Show full motif IDs instead of TF names."),
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to config YAML."),
):
    cfg_path, is_default = _resolve_config_path(ctx, config)
    loaded = _load_config_or_exit(
        cfg_path,
        missing_message=DEFAULT_CONFIG_MISSING_MESSAGE if is_default else None,
    )
    run_root = _run_root_for(loaded)
    cfg_label = _display_path(loaded.path, run_root, absolute=absolute)
    run_root_label = _display_path(run_root, run_root, absolute=absolute)
    console.print(f"[bold]Config[/]: {cfg_label}")
    console.print(f"[bold]Run[/]: id={loaded.root.densegen.run.id} root={run_root_label}")
    _print_inputs_summary(loaded, verbose=verbose, absolute=absolute, show_motif_ids=show_motif_ids)


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
    oversample_factor: Optional[int] = typer.Option(
        None,
        "--oversample-factor",
        help="Override Stage-A PWM sampling oversample_factor for all PWM inputs.",
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size",
        help="Override Stage-A PWM mining batch_size for all PWM inputs.",
    ),
    max_seconds: Optional[float] = typer.Option(
        None,
        "--max-seconds",
        help="Override Stage-A PWM mining max_seconds for all PWM inputs.",
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
    logging_utils.set_progress_enabled(str(log_cfg.progress_style) == "stream")
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
    if oversample_factor is not None and oversample_factor <= 0:
        raise typer.BadParameter("--oversample-factor must be > 0")
    if batch_size is not None and batch_size <= 0:
        raise typer.BadParameter("--batch-size must be > 0")
    if max_seconds is not None and max_seconds <= 0:
        raise typer.BadParameter("--max-seconds must be > 0")
    _apply_stage_a_overrides(
        cfg,
        selected=selected if selected else None,
        n_sites=n_sites,
        oversample_factor=oversample_factor,
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

    plan_rows = _stage_a_plan_rows(
        cfg,
        cfg_path,
        selected if selected else None,
        show_motif_ids=show_motif_ids,
    )
    if plan_rows:
        plan_table = Table()
        plan_table.add_column("input", overflow="fold")
        plan_table.add_column("TF", overflow="fold")
        plan_table.add_column("retain")
        plan_table.add_column("candidates")
        plan_table.add_column("eligibility")
        plan_table.add_column("backend")
        plan_table.add_column("length")
        for row in plan_rows:
            plan_table.add_row(
                row["input"],
                row["tf"],
                row["retain"],
                row["candidates"],
                row["eligibility"],
                row["backend"],
                row["length"],
            )
        console.print("[bold]Stage-A plan[/]")
        console.print(plan_table)

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
        motifs = _input_motifs(inp, cfg_path)
        display_map_by_input[inp.name] = {motif_id: name for motif_id, name in motifs}

    recap_rows = _stage_a_sampling_rows(pool_data)
    if recap_rows:
        console.print("[bold]Stage-A sampling recap[/]")
        grouped: dict[str, list[dict[str, object]]] = {}
        for row in recap_rows:
            grouped.setdefault(str(row["input_name"]), []).append(row)
        for input_name in sorted(grouped):
            recap_table = Table()
            recap_table.add_column("TF", overflow="fold")
            recap_table.add_column("generated")
            recap_table.add_column("eligible")
            recap_table.add_column("retained")
            recap_table.add_column("tier fill")
            recap_table.add_column("score(min/med/avg/max)")
            recap_table.add_column("len(n/min/med/avg/max)")
            for row in sorted(grouped[input_name], key=lambda item: str(item["regulator"])):
                reg_label = str(row["regulator"])
                if not show_motif_ids:
                    reg_label = display_map_by_input.get(input_name, {}).get(reg_label, reg_label)
                recap_table.add_row(
                    reg_label,
                    str(row["generated"]),
                    str(row["eligible"]),
                    str(row["retained"]),
                    str(row["tier_fill"]),
                    str(row["score"]),
                    str(row["length"]),
                )
            console.print(f"[bold]Input: {input_name}[/]")
            console.print(recap_table)

            tier_rows = [
                row
                for row in grouped[input_name]
                if row.get("tier0_score") is not None
                or row.get("tier1_score") is not None
                or row.get("tier2_score") is not None
            ]
            if tier_rows:
                boundary_table = Table("TF", "tier0.1% score", "tier1% score", "tier9% score")
                for row in sorted(tier_rows, key=lambda item: str(item["regulator"])):
                    reg_label = str(row["regulator"])
                    if not show_motif_ids:
                        reg_label = display_map_by_input.get(input_name, {}).get(reg_label, reg_label)
                    t0 = row.get("tier0_score")
                    t1 = row.get("tier1_score")
                    t2 = row.get("tier2_score")
                    boundary_table.add_row(
                        reg_label,
                        f"{float(t0):.2f}" if t0 is not None else "-",
                        f"{float(t1):.2f}" if t1 is not None else "-",
                        f"{float(t2):.2f}" if t2 is not None else "-",
                    )
                console.print(boundary_table)
        console.print(
            "Legend: generated=PWM candidates; eligible=best_hit_score>0 with hit; "
            "retained=top-N by score after dedupe; tier fill=deepest diagnostic tier used."
        )
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
    show_hash: bool = typer.Option(False, "--show-hash", help="Show full library hash in the table."),
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
    libraries_built = _load_existing_library_index(outputs_root) if outputs_root.exists() else 0

    if pool is not None:
        pool_dir = _resolve_outputs_path_or_exit(cfg_path, run_root, pool, label="stage-b.pool")
    else:
        pool_dir = run_root / "outputs" / "pools"
    if pool_dir.exists() and pool_dir.is_file():
        raise typer.BadParameter(f"Pool path must be a directory from `stage-a build-pool`, not a file: {pool_dir}")
    if not pool_dir.exists() or not pool_dir.is_dir():
        raise typer.BadParameter(f"Pool directory not found: {pool_dir}")
    try:
        pool_artifact = load_pool_artifact(pool_dir)
    except FileNotFoundError as exc:
        console.print(f"[bold red]{exc}[/]")
        entries = _list_dir_entries(pool_dir, limit=10)
        if entries:
            console.print(f"[bold]Pool directory contents[/]: {', '.join(entries)}")
        console.print("[bold]Next steps[/]:")
        console.print(f"  - {_workspace_command('dense stage-a build-pool', cfg_path=cfg_path, run_root=run_root)}")
        console.print("  - ensure --pool points to the outputs/pools directory for this workspace")
        raise typer.Exit(code=1)

    display_map_by_input: dict[str, dict[str, str]] = {}
    for inp in cfg.inputs:
        motifs = _input_motifs(inp, cfg_path)
        display_map_by_input[inp.name] = {motif_id: name for motif_id, name in motifs}

    build_rows = []
    member_rows = []
    headers = ["input", "plan", "build", "sites", "TF counts", "bp total/budget", "sampling strategy"]
    if show_hash:
        headers.insert(3, "hash")
    table = Table(*headers)
    with _suppress_pyarrow_sysctl_warnings():
        for inp in cfg.inputs:
            if selected_inputs and inp.name not in selected_inputs:
                continue
            entry = pool_artifact.entry_for(inp.name)
            pool_path = pool_dir / entry.pool_path
            if not pool_path.exists():
                raise typer.BadParameter(f"Pool file not found for input {inp.name}: {pool_path}")
            df = pd.read_parquet(pool_path)
            if entry.pool_mode == POOL_MODE_TFBS:
                meta_df = df
                data_entries = df["tfbs"].tolist() if "tfbs" in df.columns else []
            elif entry.pool_mode == POOL_MODE_SEQUENCE:
                meta_df = None
                data_entries = df["sequence"].tolist()
            else:
                raise typer.BadParameter(f"Unsupported pool_mode for input {inp.name}: {entry.pool_mode}")
            pool = PoolData(
                name=inp.name,
                input_type=str(inp.type),
                pool_mode=entry.pool_mode,
                df=meta_df,
                sequences=list(data_entries),
                pool_path=pool_path,
            )

            for plan_item in resolved_plan:
                if selected_plans and plan_item.name not in selected_plans:
                    continue
                try:
                    library, _parts, reg_labels, info = build_library_for_plan(
                        source_label=inp.name,
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
                    console.print(f"[bold]Context[/]: input={inp.name} plan={plan_item.name}")
                    console.print("[bold]Next steps[/]:")
                    console.print("  - ensure required_regulators match Stage-A regulator labels")
                    console.print("  - inspect available regulators via dense inspect inputs")
                    console.print("    or outputs/pools/pool_manifest.json")
                    raise typer.Exit(code=1)
                libraries_built = int(info.get("library_index", libraries_built))
                library_hash = str(info.get("library_hash") or "")
                target_len = int(info.get("target_length") or 0)
                achieved_len = int(info.get("achieved_length") or 0)
                pool_strategy = str(info.get("pool_strategy") or sampling_cfg.pool_strategy)
                sampling_strategy = str(info.get("library_sampling_strategy") or sampling_cfg.library_sampling_strategy)
                library_id = library_hash
                tfbs_id_by_index = info.get("tfbs_id_by_index") or []
                motif_id_by_index = info.get("motif_id_by_index") or []
                row = {
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "input_name": inp.name,
                    "input_type": inp.type,
                    "plan_name": plan_item.name,
                    "library_index": int(info.get("library_index") or 0),
                    "library_id": library_id,
                    "library_hash": library_hash,
                    "library_tfbs": list(library),
                    "library_tfs": list(reg_labels) if reg_labels else [],
                    "library_site_ids": list(info.get("site_id_by_index") or []),
                    "library_sources": list(info.get("source_by_index") or []),
                    "library_tfbs_ids": list(tfbs_id_by_index),
                    "library_motif_ids": list(motif_id_by_index),
                    "pool_strategy": pool_strategy,
                    "library_sampling_strategy": sampling_strategy,
                    "library_size": int(info.get("library_size") or len(library)),
                    "target_length": target_len,
                    "achieved_length": achieved_len,
                    "relaxed_cap": bool(info.get("relaxed_cap") or False),
                    "final_cap": info.get("final_cap"),
                    "iterative_max_libraries": int(info.get("iterative_max_libraries") or 0),
                    "iterative_min_new_solutions": int(info.get("iterative_min_new_solutions") or 0),
                    "required_regulators_selected": info.get("required_regulators_selected"),
                }
                build_rows.append(row)
                try:
                    _emit_event(
                        events_path,
                        event="LIBRARY_BUILT",
                        payload={
                            "input_name": inp.name,
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
                                "input_name": inp.name,
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
                        f"Failed to write Stage-B events for {inp.name}/{plan_item.name}: {exc}"
                    ) from exc
                for idx, tfbs in enumerate(list(library)):
                    member_rows.append(
                        {
                            "library_id": library_id,
                            "library_hash": library_hash,
                            "library_index": int(info.get("library_index") or 0),
                            "input_name": inp.name,
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
                tf_counts = Counter(reg_labels or [])
                tf_counts_label = "-"
                if tf_counts:
                    parts = []
                    for tf, count in sorted(tf_counts.items(), key=lambda kv: kv[0]):
                        label = tf
                        if not show_motif_ids:
                            label = display_map_by_input.get(inp.name, {}).get(tf, tf)
                        parts.append(f"{label}={int(count)}")
                    tf_counts_label = " ".join(parts)
                over_budget = achieved_len - target_len
                over_label = f"{over_budget:+d}" if target_len > 0 else "n/a"
                bp_label = f"{achieved_len} / {target_len} ({over_label})" if target_len > 0 else f"{achieved_len} / -"

                row_items = [
                    inp.name,
                    plan_item.name,
                    str(row["library_index"]),
                ]
                if show_hash:
                    row_items.append(str(library_hash))
                row_items.extend(
                    [
                        str(len(library)),
                        tf_counts_label,
                        bp_label,
                        str(sampling_strategy),
                    ]
                )
                table.add_row(*row_items)

        if not build_rows:
            console.print("[yellow]No libraries built (no matching inputs/plans).[/]")
            raise typer.Exit(code=1)

        try:
            artifact = write_library_artifact(
                out_dir=out_dir,
                builds=build_rows,
                members=member_rows,
                cfg_path=cfg_path,
                run_id=str(cfg.run.id),
                run_root=run_root,
                overwrite=overwrite,
            )
        except FileExistsError as exc:
            console.print(f"[bold red]{exc}[/]")
            console.print("[bold]Tip[/]: rerun with --overwrite to replace existing library artifacts.")
            raise typer.Exit(code=1)
    console.print("[bold]Stage-B libraries (solver inputs)[/]")
    console.print(table)
    console.print(
        "Stage-B builds solver libraries from Stage-A pools (cached for `dense run`). "
        "bp budget = sequence_length + subsample_over_length_budget_by."
    )
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

    run_plots_from_config(loaded.root, loaded.path, only=only, source="plot")
    console.print(":bar_chart: [bold green]Plots written.[/]")


if __name__ == "__main__":
    app()
