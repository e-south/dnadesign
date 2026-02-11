"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/core/reporting_render.py

Report rendering helpers for DenseGen runs.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from ..config import RootConfig, resolve_outputs_scoped_path, resolve_run_root
from .reporting_data import ReportBundle, collect_report_data


def _safe_filename(text: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text) or "densegen"


def _markdown_table(df: pd.DataFrame, *, columns: list[str] | None = None, max_rows: int = 10) -> str:
    if df is None or df.empty:
        return ""
    cols = columns or list(df.columns)
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return ""
    sub = df[cols].head(max_rows)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in sub.iterrows():
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def _load_plot_manifest(manifest_path: Path, *, report_root: Path) -> dict[str, list[str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"plot_manifest not found: {manifest_path}")
    payload = json.loads(manifest_path.read_text())
    plots: dict[str, list[str]] = {}
    for entry in payload.get("plots", []):
        name = str(entry.get("name") or "").strip()
        rel = str(entry.get("path") or "").strip()
        if not name or not rel:
            raise ValueError("plot_manifest entries must include name and path.")
        plot_path = manifest_path.parent / rel
        if not plot_path.exists():
            raise FileNotFoundError(f"Plot listed in plot_manifest is missing: {plot_path}")
        rel_to_report = os.path.relpath(plot_path, report_root)
        plots.setdefault(name, []).append(rel_to_report)
    return plots


def write_report(
    root_cfg: RootConfig,
    cfg_path: Path,
    *,
    out_dir: str | Path = "outputs/report",
    include_combinatorics: bool = False,
    include_plots: bool = False,
    strict: bool = False,
    formats: set[str] | None = None,
) -> ReportBundle:
    run_root = resolve_run_root(cfg_path, root_cfg.densegen.run.root)
    out_path = resolve_outputs_scoped_path(cfg_path, run_root, str(out_dir), label="report.out")
    out_path.mkdir(parents=True, exist_ok=True)
    bundle = collect_report_data(
        root_cfg,
        cfg_path,
        include_combinatorics=include_combinatorics,
        strict=strict,
    )
    composition = bundle.tables.get("composition")
    if composition is not None and not composition.empty:
        bundle.run_report["composition_rows"] = int(len(composition))
    if include_plots:
        plot_manifest = run_root / "outputs" / "plots" / "plot_manifest.json"
        plots = _load_plot_manifest(plot_manifest, report_root=out_path)
        bundle.plots = plots
        bundle.run_report["plot_manifest"] = str(plot_manifest.relative_to(run_root))
    formats = {f.lower() for f in (formats or {"json", "md"})}
    if "all" in formats:
        formats = {"json", "md", "html"}
    if "json" in formats:
        report_path = out_path / "report.json"
        report_path.write_text(json.dumps(bundle.run_report, indent=2, sort_keys=True))
    if "md" in formats:
        report_md = out_path / "report.md"
        _write_report_md(report_md, bundle)
    if "html" in formats:
        report_html = out_path / "report.html"
        _write_report_html(report_html, bundle)
    return bundle


def _render_report_md(bundle: ReportBundle) -> str:
    report = bundle.run_report
    coverage = report.get("coverage") or {}
    usage_stats = report.get("usage_stats") or {}
    tf_cov = coverage.get("tf_coverage")
    tfbs_cov = coverage.get("tfbs_coverage")
    tf_cov_label = "-"
    tfbs_cov_label = "-"
    if tf_cov is not None:
        tf_cov_label = f"{float(tf_cov):.1%} ({coverage.get('used_tf_count', 0)}/{coverage.get('library_tf_count', 0)})"
    if tfbs_cov is not None:
        tfbs_cov_label = (
            f"{float(tfbs_cov):.1%} ({coverage.get('used_tfbs_count', 0)}/{coverage.get('library_tfbs_count', 0)})"
        )
    tf_usage = usage_stats.get("tf") or {}
    tfbs_usage = usage_stats.get("tfbs") or {}
    tf_usage_label = "-"
    tfbs_usage_label = "-"
    if tf_usage.get("min") is not None:
        tf_usage_label = f"{tf_usage.get('min'):.0f}/{tf_usage.get('median'):.0f}/{tf_usage.get('max'):.0f}"
    if tfbs_usage.get("min") is not None:
        tfbs_usage_label = f"{tfbs_usage.get('min'):.0f}/{tfbs_usage.get('median'):.0f}/{tfbs_usage.get('max'):.0f}"
    output_records_path = report.get("outputs_path")
    if output_records_path:
        output_records_line = f"- {output_records_path}"
    else:
        output_records_line = f"- {report.get('output_source')} (output records source)"
    lines = [
        "# DenseGen Report",
        "",
        f"- Run root: {report.get('run_root')}",
        f"- Schema: {report.get('schema_version')}",
        f"- Output rows: {report.get('output_rows')}",
        f"- Unique sequences: {report.get('output_unique_sequences')}",
        f"- Libraries in outputs: {report.get('libraries_in_outputs')}",
        f"- Diversity (unique TFs): {report.get('diversity_unique_tfs')}",
        f"- Diversity (unique TFBS): {report.get('diversity_unique_tfbs')}",
        f"- Diversity entropy (TFBS): {report.get('diversity_entropy_tfbs')}",
        f"- Coverage (TFs used / offered): {tf_cov_label}",
        f"- Coverage (TFBS used / offered): {tfbs_cov_label}",
        f"- TF usage (min/median/max): {tf_usage_label}",
        f"- TFBS usage (min/median/max): {tfbs_usage_label}",
        f"- Warnings: {len(report.get('warnings') or [])}",
        "",
        "## Outputs",
        output_records_line,
        "- outputs/tables/attempts.parquet",
        "- outputs/tables/solutions.parquet",
        "- outputs/tables/composition.parquet",
        "- outputs/libraries/library_builds.parquet",
        "- outputs/libraries/library_members.parquet",
        "- outputs/pools/pool_manifest.json",
        "- outputs/meta/effective_config.json",
        "- outputs/meta/events.jsonl",
        "- outputs/pools/candidates/candidates.parquet (when candidate logging is enabled)",
        "- outputs/pools/candidates/candidates_summary.parquet (when candidate logging is enabled)",
        "- outputs/plots/ (visual artifacts; run `dense plot` to populate)",
        "- outputs/plots/plot_manifest.json (plot index for reports)",
    ]
    warnings = report.get("warnings") or []
    if warnings:
        lines.extend(["", "## Notes"])
        for warning in warnings:
            lines.append(f"- {warning}")
    stage_a_tiers = bundle.tables.get("stage_a_tiers")
    if stage_a_tiers is not None and not stage_a_tiers.empty:
        lines.extend(["", "## Stage-A tiers"])
        for (input_name, tf), sub in stage_a_tiers.groupby(["input_name", "tf"]):
            sub = sub.sort_values("tier")
            parts = []
            for _, row in sub.iterrows():
                tier = int(row.get("tier") or 0)
                count = int(row.get("count") or 0)
                parts.append(f"tier{tier}:{count}")
            lines.append(f"- {input_name}/{tf}: " + " ".join(parts))
    stage_a_score_summary = bundle.tables.get("stage_a_score_summary")
    if stage_a_score_summary is not None and not stage_a_score_summary.empty:
        lines.extend(["", "## Stage-A score summary (per TF)"])
        lines.append(
            _markdown_table(
                stage_a_score_summary,
                columns=["input_name", "tf", "metric", "count", "min", "p10", "p50", "p90", "max"],
                max_rows=20,
            )
        )
    candidates_summary = bundle.tables.get("candidates_summary")
    if candidates_summary is not None and not candidates_summary.empty:
        lines.extend(["", "## Candidate mining summary"])
        lines.append(
            _markdown_table(
                candidates_summary,
                columns=[
                    "input_name",
                    "motif_label",
                    "motif_id",
                    "scoring_backend",
                    "total_candidates",
                    "accepted",
                    "selected",
                    "rejected",
                ],
                max_rows=20,
            )
        )
    plan_summary = bundle.tables.get("plan_summary")
    if plan_summary is not None and not plan_summary.empty:
        summary = plan_summary.copy()
        for col in ("coverage_required_rate", "coverage_all_tfs_rate"):
            if col in summary.columns:
                summary[col] = summary[col].apply(
                    lambda v: f"{float(v):.1%}" if v is not None and not pd.isna(v) else "-"
                )
        lines.extend(["", "## Plan coverage summary"])
        lines.append(
            _markdown_table(
                summary,
                columns=[
                    "input_name",
                    "plan_name",
                    "outputs",
                    "unique_solutions",
                    "coverage_required_rate",
                    "coverage_all_tfs_rate",
                    "avg_used_tf_count",
                ],
                max_rows=20,
            )
        )
    library_summary = bundle.tables.get("library_summary")
    if library_summary is not None and not library_summary.empty:
        lines.extend(["", "## Library summary (aggregate)"])
        lines.append(
            _markdown_table(
                library_summary,
                columns=[
                    "input_name",
                    "plan_name",
                    "libraries",
                    "library_size_min",
                    "library_size_median",
                    "library_size_max",
                    "total_bp_min",
                    "total_bp_median",
                    "total_bp_max",
                ],
                max_rows=20,
            )
        )
    events = bundle.tables.get("events")
    if events is not None and not events.empty and "event" in events.columns:
        event_summary = (
            events.groupby("event")
            .agg(count=("event", "size"), last_created_at=("created_at", "max"))
            .reset_index()
            .sort_values("count", ascending=False)
        )
        lines.extend(["", "## Events summary"])
        lines.append(_markdown_table(event_summary, columns=["event", "count", "last_created_at"], max_rows=10))
    solutions = bundle.tables.get("solutions")
    if solutions is not None and not solutions.empty:
        preview = solutions.copy()
        if "sequence" in preview.columns:
            preview["sequence_len"] = preview["sequence"].apply(lambda s: len(s) if isinstance(s, str) else 0)
            preview["sequence_preview"] = preview["sequence"].apply(
                lambda s: (s[:24] + "…") if isinstance(s, str) and len(s) > 25 else s
            )
        lines.extend(["", "## Solutions (sample)"])
        lines.append(
            _markdown_table(
                preview,
                columns=[
                    "solution_id",
                    "attempt_id",
                    "input_name",
                    "plan_name",
                    "sampling_library_hash",
                    "sequence_len",
                    "sequence_preview",
                ],
                max_rows=10,
            )
        )
    else:
        lines.extend(["", "## Solutions"])
        lines.append("- No solutions found yet. Review attempts/events and adjust constraints or runtime settings.")
    composition = bundle.tables.get("composition")
    if composition is not None and not composition.empty:
        comp = composition.copy()
        if "library_hash" in comp.columns:
            comp["library_hash"] = comp["library_hash"].apply(lambda v: str(v)[:8])
        lines.extend(["", "## Composition (sample)"])
        lines.append(
            _markdown_table(
                comp,
                columns=[
                    "solution_id",
                    "attempt_id",
                    "placement_index",
                    "tf",
                    "tfbs",
                    "motif_id",
                    "tfbs_id",
                    "orientation",
                    "offset",
                    "library_hash",
                ],
                max_rows=12,
            )
        )
        comp_rows = report.get("composition_rows")
        if comp_rows is not None:
            lines.append(f"- Full composition rows: {comp_rows}")
    if bundle.plots:
        lines.extend(["", "## Report plots"])
        for plot_name, paths in bundle.plots.items():
            lines.append(f"- {plot_name}:")
            for rel_path in paths:
                lines.append(f"  - {rel_path}")
    return "\n".join(lines) + "\n"


def _write_report_md(path: Path, bundle: ReportBundle) -> None:
    path.write_text(_render_report_md(bundle))


def _write_report_html(path: Path, bundle: ReportBundle) -> None:
    md = _render_report_md(bundle)
    body = md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    img_sections: list[str] = []
    if bundle.plots:
        for plot_name, paths in bundle.plots.items():
            for rel_path in paths:
                img_sections.append(
                    f'<div><h3>{plot_name}</h3><img src="{rel_path}" '
                    'style="max-width:100%;height:auto;border:1px solid #ddd;"/></div>'
                )
    img_html = "\n".join(img_sections)
    html = "\n".join(
        [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            '<meta charset="utf-8"/>',
            "<title>DenseGen Report</title>",
            "<style>body{font-family:ui-monospace,Menlo,Monaco,Consolas,'Liberation Mono','Courier New',monospace;"
            "padding:24px;background:#fafafa;color:#111;}pre{white-space:pre-wrap;}</style>",
            "</head>",
            "<body>",
            "<pre>",
            body,
            "</pre>",
            img_html,
            "</body>",
            "</html>",
        ]
    )
    path.write_text(html)
