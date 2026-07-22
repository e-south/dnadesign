"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/permuter/src/cli/plot.py

CLI wiring for plot Permuter CLI.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import inspect
import json
import logging
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from rich.console import Console

from dnadesign.permuter.src.cli.output import emit_json
from dnadesign.permuter.src.contracts.metrics import (
    observed_metric_column,
    observed_metric_ids,
)
from dnadesign.permuter.src.core.config import ScopeConfig
from dnadesign.permuter.src.core.paths import normalize_data_path
from dnadesign.permuter.src.core.storage import (
    append_record_md,
    ensure_output_dir,
    read_parquet,
    read_ref_fasta,
    read_ref_protein_fasta,
)
from dnadesign.permuter.src.plots.aa_category_effects import plot as plot_cat
from dnadesign.permuter.src.plots.hairpin_length_vs_metric import plot as plot_hlvm
from dnadesign.permuter.src.plots.metric_by_mutation_count import plot as plot_mmc
from dnadesign.permuter.src.plots.mutation_summary import emit_aa_mutation_llr_summary
from dnadesign.permuter.src.plots.position_scatter_and_heatmap import plot as plot_psh
from dnadesign.permuter.src.plots.ranked_variants import plot as plot_ranked
from dnadesign.permuter.src.plots.registry import (
    assert_supported_plot_id,
    plot_description_payload,
    plot_registry_payload,
    supported_plot_ids,
)
from dnadesign.permuter.src.plots.synergy_scatter import plot as plot_syn
from dnadesign.permuter.src.workspaces.datasets import resolve_workspace_dataset_path

console = Console()
_LOG = logging.getLogger("permuter.plot")


def _call_plot(func, *, plot_name: str, **kwargs) -> None:
    """
    Call a plot function with only the parameters it declares.
    Unknown kwargs that are None are dropped. Unknown kwargs that are set (not None)
    raise a TypeError with guidance (assertive; no silent fallbacks).
    """
    sig = inspect.signature(func)
    allowed = set(sig.parameters.keys())
    accepted = {}
    rejected = {}
    for k, v in kwargs.items():
        if k in allowed:
            if v is not None:
                accepted[k] = v
        else:
            if v is not None:
                rejected[k] = v
    if rejected:
        bad = ", ".join(sorted(rejected))
        allowed_list = ", ".join(sorted(allowed))
        raise TypeError(f"{plot_name}: unsupported option(s): {bad}. Supported parameters are: {allowed_list}")
    return func(**accepted)


def _normalize_for_plots(df: pd.DataFrame, metric_id: str, log=_LOG) -> pd.DataFrame:
    """
    Assert the study-agnostic observed metric exists and provide only
    convenience aliases needed by downstream plots.

    Plot-specific contracts remain in the plot modules. Ordinary DMS plots
    should not be blocked by interaction-only columns such as expected score or
    namespaced epistasis.
    """
    df2 = df.copy()
    req = [observed_metric_column(metric_id)]
    missing = [c for c in req if c not in df2.columns]
    if missing:
        raise ValueError(
            "Dataset missing required canonical column(s) for plotting with "
            f"metric_id={metric_id}: {missing}\n"
            "Run 'permuter evaluate' after generation to populate observed metrics."
        )
    # Unprefixed convenience columns for plot code
    if "mut_count" not in df2.columns and "permuter__mut_count" in df2.columns:
        df2["mut_count"] = df2["permuter__mut_count"].astype(int)
    if "aa_combo_str" not in df2.columns and "permuter__aa_combo_str" in df2.columns:
        df2["aa_combo_str"] = df2["permuter__aa_combo_str"].astype(str)
    return df2


def _derive_records_from_workspace(
    workspace_hint: str, ref: Optional[str], out: Optional[Path]
) -> Tuple[Path, ScopeConfig, Path]:
    resolved = resolve_workspace_dataset_path(workspace_hint=workspace_hint, ref=ref, out=out)
    return resolved.records, resolved.config, resolved.config_path


def plot(
    data: Optional[Path],
    workspace: Optional[str],
    ref: Optional[str],
    out: Optional[Path],
    which: Optional[List[str]],
    metric_id: Optional[str] = None,
    width: Optional[float] = None,
    height: Optional[float] = None,
    font_scale: Optional[float] = None,
    emit_summaries: Optional[bool] = None,
    list_plots: bool = False,
    describe: Optional[str] = None,
    as_json: bool = False,
) -> dict[str, object]:
    if list_plots and describe:
        raise ValueError("Use either --list or --describe, not both.")
    if list_plots:
        payload = plot_registry_payload()
        if as_json:
            emit_json(payload)
        else:
            for item in payload["plots"]:
                console.print(f"{item['id']}: {item['summary']}")
        return payload
    if describe:
        payload = plot_description_payload(describe)
        if as_json:
            emit_json(payload)
        else:
            plot_meta = payload["plot"]
            console.print(f"[bold]{plot_meta['id']}[/bold]")
            console.print(str(plot_meta["summary"]))
            console.print("Requires: " + ", ".join(str(item) for item in plot_meta["requires"]))
        return payload

    # Resolve dataset path and load workspace config if provided.
    cfg: Optional[ScopeConfig] = None
    config_path: Optional[Path] = None
    if data is not None:
        records = normalize_data_path(data)
    elif workspace:
        records, cfg, config_path = _derive_records_from_workspace(workspace, ref, out)
    else:
        raise ValueError("Provide either --data (file or dataset dir) or --workspace/--ref.")

    df = read_parquet(records)
    plots_dir = records.parent / "plots"
    ensure_output_dir(plots_dir)
    ref_dna = read_ref_fasta(records.parent)
    ref_seq = ref_dna[1] if ref_dna else None
    ref_aa = read_ref_protein_fasta(records.parent)
    ref_aa_seq = ref_aa[1] if ref_aa else None
    scope_name = str(df.get("permuter__scope", pd.Series(["scope"])).iloc[0])

    # Hard requirements shared by both built-in plots
    required = ["sequence", "permuter__modifications", "permuter__round"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns for plotting: {missing}")

    # Defaults from workspace config if available.
    yaml_which = None
    yaml_metric = None
    yaml_width = None
    yaml_height = None
    yaml_font = None
    yaml_strip_every = None
    yaml_emit = None
    yaml_ranked_annot_top = None
    yaml_ranked_summary_top_n = None
    yaml_ranked_export_top_k = None
    yaml_ranked_xtick_every = None
    yaml_sizes_map = {}
    if cfg and cfg.scope.plot:
        yaml_which = list(cfg.scope.plot.which or [])
        yaml_metric = cfg.scope.plot.metric_id
        if cfg.scope.plot.size:
            yaml_width = cfg.scope.plot.size.width
            yaml_height = cfg.scope.plot.size.height
        yaml_font = cfg.scope.plot.font_scale
        yaml_strip_every = getattr(cfg.scope.plot, "strip_every", None)
        yaml_emit = getattr(cfg.scope.plot, "emit_summaries", True)
        yaml_ranked_annot_top = getattr(cfg.scope.plot, "ranked_annotate_top", None)
        yaml_ranked_summary_top_n = getattr(cfg.scope.plot, "ranked_summary_top_n", None)
        yaml_ranked_export_top_k = getattr(cfg.scope.plot, "ranked_export_top_k", None)
        yaml_ranked_xtick_every = getattr(cfg.scope.plot, "ranked_xtick_every", None)
        yaml_sizes_map = dict(getattr(cfg.scope.plot, "sizes", {}) or {})

    which = list(which or yaml_which or ["position_scatter_and_heatmap"])
    which = [assert_supported_plot_id(name) for name in which]
    metric_id = metric_id or yaml_metric
    width = width or yaml_width
    height = height or yaml_height
    font_scale = font_scale or yaml_font
    figsize_global = (width, height) if (width and height) else None
    strip_every = yaml_strip_every
    emit_summaries = emit_summaries if emit_summaries is not None else (yaml_emit if yaml_emit is not None else True)

    # Discover present metric ids once
    obs_cols = [c for c in df.columns if c.startswith("permuter__observed__")]
    present_ids = observed_metric_ids(df.columns)

    # If no metric-id was given, infer when there is exactly one id.
    if not metric_id:
        ids = observed_metric_ids(df.columns)
        if len(ids) == 1:
            metric_id = ids[0]
        else:
            raise ValueError(
                "Multiple metrics present; choose one with --metric-id or set plot.metric_id in the workspace config.\n"
                f"Found: {ids or '<none>'}"
            )
    # Verify the requested metric exists and suggest fixes
    if metric_id not in present_ids:
        hint = ""
        # No metric columns at all → suggest 'evaluate'
        if not obs_cols:
            if cfg and config_path:
                ref_arg = f" --ref {ref}" if ref else ""
                hint = (
                    f"\nHint: this dataset has no observed metric columns yet. "
                    f"Append them with:\n"
                    f"  permuter evaluate --workspace {config_path}{ref_arg}\n"
                    f"or a quick smoke test:\n"
                    f"  permuter evaluate --workspace {config_path}{ref_arg}\n"
                )
            else:
                hint = (
                    "\nHint: this dataset has no observed metric columns. Append them with:\n"
                    "  permuter evaluate --data <dataset_dir> --with <id>:<evaluator>:<metric>"
                )
        raise ValueError(
            f"Metric id '{metric_id}' not found in dataset.\nAvailable metric ids: {present_ids or '<none>'}.{hint}"
        )

    # Build an informative subtitle when we know the scope config for this metric id.
    subtitle = ""
    if cfg and cfg.scope.evaluate and cfg.scope.evaluate.metrics:
        for m in cfg.scope.evaluate.metrics:
            if str(m.id) == str(metric_id):
                red = (m.params or {}).get("reduction", None)
                red_txt = f", reduction={red}" if red else ""
                subtitle = f"metric={m.id} • evaluator={m.evaluator}.{m.metric}{red_txt}"
                break

    # Prepare canonical columns shared by all plots (once):
    # - requires permuter__observed__{metric_id}
    # - provides unprefixed 'mut_count' / 'aa_combo_str' aliases for ranked plots
    try:
        df = _normalize_for_plots(df, metric_id)
        _LOG.info(
            "plot: normalized canonical columns for metric_id=%s",
            metric_id,
        )
    except Exception as e:
        raise ValueError(f"Unable to prepare canonical columns for plotting (metric_id={metric_id}). {e}") from e

    artifacts: list[dict[str, object]] = []
    for name in which:
        # Compute figsize for this plot with explicit precedence:
        # CLI > plot.sizes[name] > plot.size > internal default
        if figsize_global:
            figsize = figsize_global
        else:
            ps = yaml_sizes_map.get(name)
            if ps and ps.width and ps.height:
                figsize = (float(ps.width), float(ps.height))
            elif yaml_width and yaml_height:
                figsize = (float(yaml_width), float(yaml_height))
            else:
                figsize = None
        if name == "position_scatter_and_heatmap":
            output_path = plots_dir / f"{name}__{metric_id}.pdf"
            _LOG.info(
                "plot: %s → %s (metric_id=%s, figsize=%s, font_scale=%s)",
                name,
                output_path,
                metric_id or "<auto>",
                str(figsize) if figsize else "auto",
                str(font_scale) if font_scale else "1.0",
            )
            plot_psh(
                elite_df=df.head(0),
                all_df=df,
                output_path=output_path,
                scope_name=scope_name,
                ref_sequence=ref_seq,
                ref_aa_sequence=ref_aa_seq,
                metric_id=metric_id,
                evaluators=subtitle,
                figsize=figsize,
                font_scale=font_scale,
                ref_strip_every=strip_every,
            )
            artifacts.append(_artifact_entry(name, output_path, figsize=figsize, font_scale=font_scale))
            if not as_json:
                console.print(f"[green]✔[/green] {name} → {output_path}")
        elif name == "ranked_variants":
            output_path = plots_dir / f"{name}__{metric_id}.png"
            _LOG.info(
                "plot: %s → %s (metric_id=%s, figsize=%s, font_scale=%s)",
                name,
                output_path,
                metric_id or "<auto>",
                str(figsize) if figsize else "auto",
                str(font_scale) if font_scale else "1.0",
            )
            yaml_ranked_jitter = getattr(cfg.scope.plot, "ranked_jitter", None) if cfg and cfg.scope.plot else None
            yaml_ranked_point_size = (
                getattr(cfg.scope.plot, "ranked_point_size", None) if cfg and cfg.scope.plot else None
            )
            yaml_ranked_alpha = getattr(cfg.scope.plot, "ranked_alpha", None) if cfg and cfg.scope.plot else None
            yaml_ranked_cmap = getattr(cfg.scope.plot, "ranked_cmap", None) if cfg and cfg.scope.plot else None
            _call_plot(
                plot_ranked,
                plot_name="ranked_variants",
                elite_df=df.head(0),
                all_df=df,
                output_path=output_path,
                scope_name=scope_name,
                ref_sequence=ref_seq,
                metric_id=metric_id,
                evaluators=subtitle,
                figsize=figsize,
                font_scale=font_scale,
                ranked_jitter=yaml_ranked_jitter,
                ranked_point_size=yaml_ranked_point_size,
                ranked_alpha=yaml_ranked_alpha,
                ranked_cmap=yaml_ranked_cmap,
                ranked_annotate_top=yaml_ranked_annot_top,
                ranked_summary_top_n=yaml_ranked_summary_top_n,
                ranked_export_top_k=yaml_ranked_export_top_k,
                ranked_xtick_every=yaml_ranked_xtick_every,
            )
            artifacts.append(_artifact_entry(name, output_path, figsize=figsize, font_scale=font_scale))
            if not as_json:
                console.print(f"[green]✔[/green] {name} → {output_path}")
        elif name == "synergy_scatter":
            output_path = plots_dir / f"{name}__{metric_id}.png"
            _LOG.info(
                "plot: %s → %s (metric_id=%s, figsize=%s, font_scale=%s)",
                name,
                output_path,
                metric_id or "<auto>",
                str(figsize) if figsize else "auto",
                str(font_scale) if font_scale else "1.0",
            )
            plot_syn(
                elite_df=df.head(0),
                all_df=df,
                output_path=output_path,
                scope_name=scope_name,
                ref_sequence=ref_seq,
                metric_id=metric_id,
                evaluators=subtitle,
                figsize=figsize,
                font_scale=font_scale,
            )
            artifacts.append(_artifact_entry(name, output_path, figsize=figsize, font_scale=font_scale))
            if not as_json:
                console.print(f"[green]✔[/green] {name} → {output_path}")
        elif name == "metric_by_mutation_count":
            output_path = plots_dir / f"{name}__{metric_id}.png"
            _LOG.info(
                "plot: %s → %s (metric_id=%s, figsize=%s, font_scale=%s)",
                name,
                output_path,
                metric_id or "<auto>",
                str(figsize) if figsize else "auto",
                str(font_scale) if font_scale else "1.0",
            )
            plot_mmc(
                elite_df=df.head(0),
                all_df=df,
                output_path=output_path,
                scope_name=scope_name,
                ref_sequence=ref_seq,
                metric_id=metric_id,
                evaluators=subtitle,
                figsize=figsize,
                font_scale=font_scale,
            )
            artifacts.append(_artifact_entry(name, output_path, figsize=figsize, font_scale=font_scale))
            if not as_json:
                console.print(f"[green]✔[/green] {name} → {output_path}")

        elif name == "aa_category_effects":
            output_path = plots_dir / f"{name}__{metric_id}.png"
            _LOG.info(
                "plot: %s → %s (metric_id=%s, figsize=%s, font_scale=%s)",
                name,
                output_path,
                metric_id or "<auto>",
                "auto",
                str(font_scale) if font_scale else "1.0",
            )
            plot_cat(
                elite_df=df.head(0),
                all_df=df,
                output_path=output_path,
                scope_name=scope_name,
                ref_sequence=ref_seq,
                metric_id=metric_id,
                evaluators=subtitle,
                figsize=None,
                font_scale=font_scale,
            )
            artifacts.append(_artifact_entry(name, output_path, figsize=None, font_scale=font_scale))
            if not as_json:
                console.print(f"[green]✔[/green] {name} → {output_path}")
        elif name == "hairpin_length_vs_metric":
            output_path = plots_dir / f"{name}__{metric_id}.png"
            _LOG.info(
                "plot: %s → %s (metric_id=%s, figsize=%s, font_scale=%s)",
                name,
                output_path,
                metric_id or "<auto>",
                str(figsize) if figsize else "auto",
                str(font_scale) if font_scale else "1.0",
            )
            plot_hlvm(
                elite_df=df.head(0),
                all_df=df,
                output_path=output_path,
                scope_name=scope_name,
                ref_sequence=ref_seq,
                metric_id=metric_id,
                evaluators=subtitle,
                figsize=figsize,
                font_scale=font_scale,
            )
            artifacts.append(_artifact_entry(name, output_path, figsize=figsize, font_scale=font_scale))
            if not as_json:
                console.print(f"[green]✔[/green] {name} → {output_path}")
        else:
            raise ValueError(f"Unknown plot {name!r}. Supported plots: {', '.join(supported_plot_ids())}")

    # ---- Decoupled analysis summaries (optional, once per invocation) ----
    summaries: list[dict[str, object]] = []
    if emit_summaries and metric_id:
        try:
            out_csv = emit_aa_mutation_llr_summary(
                df,
                dataset_dir=plots_dir.parent,
                metric_id=str(metric_id),
                top_k=20,
                strict_llr_only=True,
            )
            if out_csv:
                summaries.append(
                    _artifact_entry(
                        "aa_mutation_llr_summary",
                        Path(out_csv),
                        figsize=None,
                        font_scale=None,
                    )
                )
                if not as_json:
                    console.print(f"[green]✔[/green] AA mutation summary → {out_csv}")
            else:
                _LOG.info("AA mutation summary not emitted (not applicable).")
        except Exception as e:
            # Summaries are optional; avoid breaking plots while still surfacing issues.
            _LOG.error("AA mutation summary failed: %s", e)
    # Journal once per call
    try:
        cmd = shlex.join(sys.argv)
    except Exception:
        cmd = " ".join(sys.argv)
    append_record_md(records.parent, "plot", cmd)
    manifest_path = _write_manifest(
        records=records,
        metric_id=str(metric_id),
        which=which,
        artifacts=artifacts,
        summaries=summaries,
        params={
            "width": width,
            "height": height,
            "font_scale": font_scale,
            "emit_summaries": emit_summaries,
        },
    )
    summary: dict[str, object] = {
        "schema": "permuter.plot.v1",
        "records": records,
        "dataset_dir": records.parent,
        "metric_id": str(metric_id),
        "plots_dir": plots_dir,
        "manifest": manifest_path,
        "artifacts": artifacts,
        "summaries": summaries,
    }
    if as_json:
        emit_json(summary)
    return summary


def _artifact_entry(
    plot_id: str,
    path: Path,
    *,
    figsize: Optional[Tuple[float, float]],
    font_scale: Optional[float],
) -> dict[str, object]:
    stat = path.stat() if path.exists() else None
    return {
        "id": plot_id,
        "path": path,
        "format": path.suffix.removeprefix("."),
        "size_bytes": stat.st_size if stat else None,
        "mtime_ns": stat.st_mtime_ns if stat else None,
        "figsize": list(figsize) if figsize else None,
        "font_scale": font_scale,
    }


def _write_manifest(
    *,
    records: Path,
    metric_id: str,
    which: list[str],
    artifacts: list[dict[str, object]],
    summaries: list[dict[str, object]],
    params: dict[str, object],
) -> Path:
    stat = records.stat()
    path = records.parent / "plots" / "manifest.json"
    payload = {
        "schema": "permuter.plot_manifest.v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "records": str(records),
        "records_size_bytes": stat.st_size,
        "records_mtime_ns": stat.st_mtime_ns,
        "metric_id": metric_id,
        "which": which,
        "params": params,
        "artifacts": [_jsonable_dict(item) for item in artifacts],
        "summaries": [_jsonable_dict(item) for item in summaries],
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)
    return path


def _jsonable_dict(payload: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in payload.items():
        if isinstance(value, Path):
            out[key] = str(value)
        elif isinstance(value, tuple):
            out[key] = list(value)
        else:
            out[key] = value
    return out
