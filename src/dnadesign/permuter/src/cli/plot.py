"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/cli/plot.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

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
from dnadesign.permuter.src.plots.metric_by_mutation_count import plot as plot_mmc
from dnadesign.permuter.src.plots.mutation_summary import emit_aa_mutation_llr_summary
from dnadesign.permuter.src.plots.position_scatter_and_heatmap import plot as plot_psh

console = Console()
_LOG = logging.getLogger("permuter.plot")


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
    ref_name, _ = _pick_reference(
        df_refs, cfg.job.input.name_col, cfg.job.input.seq_col, ref
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
    if cfg and cfg.job.plot:
        yaml_which = list(cfg.job.plot.which or [])
        yaml_metric = cfg.job.plot.metric_id
        if cfg.job.plot.size:
            yaml_width = cfg.job.plot.size.width
            yaml_height = cfg.job.plot.size.height
        yaml_font = cfg.job.plot.font_scale
        yaml_strip_every = getattr(cfg.job.plot, "strip_every", None)
        yaml_emit = getattr(cfg.job.plot, "emit_summaries", True)

    which = list(which or yaml_which or ["position_scatter_and_heatmap"])
    metric_id = metric_id or yaml_metric
    width = width or yaml_width
    height = height or yaml_height
    font_scale = font_scale or yaml_font
    figsize = (width, height) if (width and height) else None
    strip_every = yaml_strip_every
    emit_summaries = (
        emit_summaries
        if emit_summaries is not None
        else (yaml_emit if yaml_emit is not None else True)
    )

    # Verify the requested metric exists and suggest fixes
    metric_cols = [c for c in df.columns if c.startswith("permuter__metric__")]
    present_ids = sorted(
        {
            c.split("permuter__metric__", 1)[1].lstrip("_").split("__", 1)[0]
            for c in metric_cols
        }
    )
    if metric_id not in present_ids:
        hint = ""
        if metric_id.endswith("_mean") and metric_id[:-5] in present_ids:
            hint = (
                f"\nHint: dataset has '{metric_id[:-5]}'. "
                f"Either set --metric-id {metric_id[:-5]} "
                f"or re-evaluate with --with {metric_id}:evo2_llr:log_likelihood_ratio."
            )
        raise ValueError(
            f"Metric id '{metric_id}' not found in dataset.\n"
            f"Available metric ids: {present_ids or '<none>'}.{hint}"
        )

    # If no metric-id was given, infer one when there is exactly a single metric column.
    if not metric_id:
        metric_cols = [c for c in df.columns if c.startswith("permuter__metric__")]
        ids = sorted(
            {c.split("permuter__metric__", 1)[1].lstrip("_") for c in metric_cols}
        )
        if len(ids) == 1:
            metric_id = ids[0]
        else:
            raise ValueError(
                "Multiple metrics present; choose one with --metric-id or set job.plot.metric_id.\n"
                f"Found: {ids or '<none>'}"
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

    for name in which:
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
