"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/cli/plot.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import inspect
import logging
import shlex
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import yaml
from rich.console import Console

from dnadesign.permuter.src.core.config import JobConfig
from dnadesign.permuter.src.core.paths import (
    normalize_data_path,
    resolve,
    resolve_job_hint,
)
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
from dnadesign.permuter.src.plots.synergy_scatter import plot as plot_syn

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
        raise TypeError(
            f"{plot_name}: unsupported option(s): {bad}. "
            f"Supported parameters are: {allowed_list}"
        )
    return func(**accepted)


def _normalize_for_plots(df: pd.DataFrame, metric_id: str, log=_LOG) -> pd.DataFrame:
    """
    Assert canonical columns exist (already produced by run→evaluate) and
    provide only convenience aliases needed by the plots.
    """
    df2 = df.copy()
    req = [
        f"permuter__observed__{metric_id}",
        f"permuter__expected__{metric_id}",
        "epistasis",
    ]
    missing = [c for c in req if c not in df2.columns]
    if missing:
        raise ValueError(
            "Dataset missing required canonical column(s) for plotting with "
            f"metric_id={metric_id}: {missing}\n"
            "Run 'permuter evaluate' after generation to populate observed and epistasis."
        )
    # Unprefixed convenience columns for plot code
    if "mut_count" not in df2.columns and "permuter__mut_count" in df2.columns:
        df2["mut_count"] = df2["permuter__mut_count"].astype(int)
    if "aa_combo_str" not in df2.columns and "permuter__aa_combo_str" in df2.columns:
        df2["aa_combo_str"] = df2["permuter__aa_combo_str"].astype(str)
    return df2


def _pick_reference(
    df, name_col: str, seq_col: str, desired: Optional[str]
) -> tuple[str, str]:
    if desired:
        sub = df[df[name_col] == desired]
        if sub.empty:
            raise ValueError(f"Reference '{desired}' not found in '{name_col}'")
        if len(sub) > 1:
            raise ValueError(f"Reference '{desired}' not unique in CSV")
        row = sub.iloc[0]
        return str(row[name_col]), str(row[seq_col])
    if len(df) == 1:
        row = df.iloc[0]
        return str(row[name_col]), str(row[seq_col])
    raise ValueError("--ref is required because the refs CSV has multiple rows")


def _derive_records_from_job(
    job_hint: str, ref: Optional[str], out: Optional[Path]
) -> Tuple[Path, JobConfig, Path]:
    job_path = resolve_job_hint(job_hint)
    data = yaml.safe_load(job_path.read_text(encoding="utf-8"))
    cfg = JobConfig.model_validate(data)
    # Resolve base to get output_root/refs without binding to a ref yet
    jp0 = resolve(
        job_yaml=job_path,
        refs=cfg.job.input.refs,
        output_dir=cfg.job.output.dir,
        ref_name="__PENDING__",
        out_override=out,
    )
    # 1) flat-job
    flat_job = (jp0.output_root / "records.parquet").resolve()
    if flat_job.exists():
        return flat_job, cfg, job_path

    # Need ref for nested/flat_jobref
    df_refs = pd.read_csv(jp0.refs_csv, dtype=str)
    desired = ref or getattr(cfg.job.input, "reference_sequence", None)
    ref_name, _ = _pick_reference(
        df_refs, cfg.job.input.name_col, cfg.job.input.seq_col, desired
    )
    # 2) nested
    jp = resolve(
        job_yaml=job_path,
        refs=cfg.job.input.refs,
        output_dir=cfg.job.output.dir,
        ref_name=ref_name,
        out_override=out,
    )
    nested = jp.records_parquet
    if nested.exists():
        return nested, cfg, job_path
    # 3) flat-jobref
    flat_jobref = (
        jp.output_root.parent / f"{jp.output_root.name}__{ref_name}" / "records.parquet"
    ).resolve()
    if flat_jobref.exists():
        return flat_jobref, cfg, job_path
    # fallback (caller will error clearly)
    return nested, cfg, job_path


def plot(
    data: Optional[Path],
    job: Optional[str],
    ref: Optional[str],
    out: Optional[Path],
    which: Optional[List[str]],
    metric_id: Optional[str] = None,
    width: Optional[float] = None,
    height: Optional[float] = None,
    font_scale: Optional[float] = None,
    emit_summaries: Optional[bool] = None,
):
    # Resolve dataset path and load job cfg if provided
    cfg: Optional[JobConfig] = None
    job_path: Optional[Path] = None
    if data is not None:
        records = normalize_data_path(data)
    elif job:
        records, cfg, job_path = _derive_records_from_job(job, ref, out)
    else:
        raise ValueError("Provide either --data (file or dataset dir) or --job/--ref.")

    df = read_parquet(records)
    plots_dir = records.parent / "plots"
    ensure_output_dir(plots_dir)
    ref_dna = read_ref_fasta(records.parent)
    ref_seq = ref_dna[1] if ref_dna else None
    ref_aa = read_ref_protein_fasta(records.parent)
    ref_aa_seq = ref_aa[1] if ref_aa else None
    job_name = str(df.get("permuter__job", pd.Series(["job"])).iloc[0])

    # Hard requirements shared by both built-in plots
    required = ["sequence", "permuter__modifications", "permuter__round"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset missing required columns for plotting: {missing}")

    # Defaults from YAML if available
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
    if cfg and cfg.job.plot:
        yaml_which = list(cfg.job.plot.which or [])
        yaml_metric = cfg.job.plot.metric_id
        if cfg.job.plot.size:
            yaml_width = cfg.job.plot.size.width
            yaml_height = cfg.job.plot.size.height
        yaml_font = cfg.job.plot.font_scale
        yaml_strip_every = getattr(cfg.job.plot, "strip_every", None)
        yaml_emit = getattr(cfg.job.plot, "emit_summaries", True)
        yaml_ranked_annot_top = getattr(cfg.job.plot, "ranked_annotate_top", None)
        yaml_ranked_summary_top_n = getattr(cfg.job.plot, "ranked_summary_top_n", None)
        yaml_ranked_export_top_k = getattr(cfg.job.plot, "ranked_export_top_k", None)
        yaml_ranked_xtick_every = getattr(cfg.job.plot, "ranked_xtick_every", None)
        yaml_sizes_map = dict(getattr(cfg.job.plot, "sizes", {}) or {})

    which = list(which or yaml_which or ["position_scatter_and_heatmap"])
    metric_id = metric_id or yaml_metric
    width = width or yaml_width
    height = height or yaml_height
    font_scale = font_scale or yaml_font
    figsize_global = (width, height) if (width and height) else None
    strip_every = yaml_strip_every
    emit_summaries = (
        emit_summaries
        if emit_summaries is not None
        else (yaml_emit if yaml_emit is not None else True)
    )

    # Discover present metric ids once
    obs_cols = [c for c in df.columns if c.startswith("permuter__observed__")]
    present_ids = sorted(
        {
            c.split("permuter__observed__", 1)[1].lstrip("_").split("__", 1)[0]
            for c in obs_cols
        }
    )

    # If no metric-id was given, infer when there is exactly one id.
    if not metric_id:
        ids = sorted(
            {c.split("permuter__observed__", 1)[1].lstrip("_") for c in obs_cols}
        )
        if len(ids) == 1:
            metric_id = ids[0]
        else:
            raise ValueError(
                "Multiple metrics present; choose one with --metric-id or set job.plot.metric_id.\n"
                f"Found: {ids or '<none>'}"
            )
    # Verify the requested metric exists and suggest fixes
    if metric_id not in present_ids:
        hint = ""
        # No metric columns at all → suggest 'evaluate'
        if not obs_cols:
            if cfg and job_path:
                ref_arg = f" --ref {ref}" if ref else ""
                hint = (
                    f"\nHint: this dataset has no observed metric columns yet. "
                    f"Append them with:\n"
                    f"  permuter evaluate --job {job_path}{ref_arg}\n"
                    f"or a quick smoke test:\n"
                    f"  permuter evaluate --job {job_path}{ref_arg}\n"
                )
            else:
                hint = (
                    "\nHint: this dataset has no observed metric columns. Append them with:\n"
                    "  permuter evaluate --data <dataset_dir> --with <id>:<evaluator>:<metric>"
                )
        raise ValueError(
            f"Metric id '{metric_id}' not found in dataset.\n"
            f"Available metric ids: {present_ids or '<none>'}.{hint}"
        )

    # Build an informative subtitle when we know the job config for this metric id.
    subtitle = ""
    if cfg and cfg.job.evaluate and cfg.job.evaluate.metrics:
        for m in cfg.job.evaluate.metrics:
            if str(m.id) == str(metric_id):
                red = (m.params or {}).get("reduction", None)
                red_txt = f", reduction={red}" if red else ""
                subtitle = (
                    f"metric={m.id} • evaluator={m.evaluator}.{m.metric}{red_txt}"
                )
                break

    # Prepare canonical columns for all plots (once):
    # - writes permuter__observed__{metric_id} / permuter__expected__{metric_id}
    # - attaches 'epistasis' (observed - expected)
    # - provides unprefixed 'mut_count' / 'aa_combo_str' aliases for ranked plots
    try:
        df = _normalize_for_plots(df, metric_id)
        _LOG.info(
            "plot: normalized canonical columns for metric_id=%s (observed/expected/epistasis ready)",
            metric_id,
        )
    except Exception as e:
        raise ValueError(
            f"Unable to prepare canonical columns for plotting (metric_id={metric_id}). {e}"
        ) from e

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
        if name in ("position_scatter_and_heatmap", "position_scatter"):
            out = plots_dir / f"{name}__{metric_id}.pdf"
            _LOG.info(
                "plot: %s → %s (metric_id=%s, figsize=%s, font_scale=%s)",
                name,
                out,
                metric_id or "<auto>",
                str(figsize) if figsize else "auto",
                str(font_scale) if font_scale else "1.0",
            )
            plot_psh(
                elite_df=df.head(0),
                all_df=df,
                output_path=out,
                job_name=job_name,
                ref_sequence=ref_seq,
                ref_aa_sequence=ref_aa_seq,
                metric_id=metric_id,
                evaluators=subtitle,
                figsize=figsize,
                font_scale=font_scale,
                ref_strip_every=strip_every,
            )
            console.print(f"[green]✔[/green] {name} → {out}")
        elif name == "ranked_variants":
            out = plots_dir / f"{name}__{metric_id}.png"
            _LOG.info(
                "plot: %s → %s (metric_id=%s, figsize=%s, font_scale=%s)",
                name,
                out,
                metric_id or "<auto>",
                str(figsize) if figsize else "auto",
                str(font_scale) if font_scale else "1.0",
            )
            yaml_ranked_jitter = (
                getattr(cfg.job.plot, "ranked_jitter", None)
                if cfg and cfg.job.plot
                else None
            )
            yaml_ranked_point_size = (
                getattr(cfg.job.plot, "ranked_point_size", None)
                if cfg and cfg.job.plot
                else None
            )
            yaml_ranked_alpha = (
                getattr(cfg.job.plot, "ranked_alpha", None)
                if cfg and cfg.job.plot
                else None
            )
            yaml_ranked_cmap = (
                getattr(cfg.job.plot, "ranked_cmap", None)
                if cfg and cfg.job.plot
                else None
            )
            _call_plot(
                plot_ranked,
                plot_name="ranked_variants",
                elite_df=df.head(0),
                all_df=df,
                output_path=out,
                job_name=job_name,
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
            console.print(f"[green]✔[/green] {name} → {out}")
        elif name == "synergy_scatter":
            out = plots_dir / f"{name}__{metric_id}.png"
            _LOG.info(
                "plot: %s → %s (metric_id=%s, figsize=%s, font_scale=%s)",
                name,
                out,
                metric_id or "<auto>",
                str(figsize) if figsize else "auto",
                str(font_scale) if font_scale else "1.0",
            )
            plot_syn(
                elite_df=df.head(0),
                all_df=df,
                output_path=out,
                job_name=job_name,
                ref_sequence=ref_seq,
                metric_id=metric_id,
                evaluators=subtitle,
                figsize=figsize,
                font_scale=font_scale,
            )
            console.print(f"[green]✔[/green] {name} → {out}")
        elif name == "metric_by_mutation_count":
            out = plots_dir / f"{name}__{metric_id}.png"
            _LOG.info(
                "plot: %s → %s (metric_id=%s, figsize=%s, font_scale=%s)",
                name,
                out,
                metric_id or "<auto>",
                str(figsize) if figsize else "auto",
                str(font_scale) if font_scale else "1.0",
            )
            plot_mmc(
                elite_df=df.head(0),
                all_df=df,
                output_path=out,
                job_name=job_name,
                ref_sequence=ref_seq,
                metric_id=metric_id,
                evaluators=subtitle,
                figsize=figsize,
                font_scale=font_scale,
            )
            console.print(f"[green]✔[/green] {name} → {out}")

        elif name == "aa_category_effects":
            out = plots_dir / f"{name}__{metric_id}.png"
            _LOG.info(
                "plot: %s → %s (metric_id=%s, figsize=%s, font_scale=%s)",
                name,
                out,
                metric_id or "<auto>",
                "auto",
                str(font_scale) if font_scale else "1.0",
            )
            plot_cat(
                elite_df=df.head(0),
                all_df=df,
                output_path=out,
                job_name=job_name,
                ref_sequence=ref_seq,
                metric_id=metric_id,
                evaluators=subtitle,
                figsize=None,
                font_scale=font_scale,
            )
            console.print(f"[green]✔[/green] {name} → {out}")
        elif name == "hairpin_length_vs_metric":
            out = plots_dir / f"{name}__{metric_id}.png"
            _LOG.info(
                "plot: %s → %s (metric_id=%s, figsize=%s, font_scale=%s)",
                name,
                out,
                metric_id or "<auto>",
                str(figsize) if figsize else "auto",
                str(font_scale) if font_scale else "1.0",
            )
            plot_hlvm(
                elite_df=df.head(0),
                all_df=df,
                output_path=out,
                job_name=job_name,
                ref_sequence=ref_seq,
                metric_id=metric_id,
                evaluators=subtitle,
                figsize=figsize,
                font_scale=font_scale,
            )
            console.print(f"[green]✔[/green] {name} → {out}")
        else:
            raise ValueError(f"Unknown plot '{name}'")

    # ---- Decoupled analysis summaries (optional, once per invocation) ----
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
