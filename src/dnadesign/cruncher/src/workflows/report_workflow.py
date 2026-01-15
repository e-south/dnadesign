"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/workflows/report_workflow.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict

from dnadesign.cruncher.config.schema_v2 import CruncherConfig
from dnadesign.cruncher.services.run_service import get_run, update_run_index_from_manifest
from dnadesign.cruncher.utils.analysis_layout import analysis_root, summary_path
from dnadesign.cruncher.utils.artifacts import append_artifacts, artifact_entry, normalize_artifacts
from dnadesign.cruncher.utils.diagnostics import summarize_sampling_diagnostics
from dnadesign.cruncher.utils.elites import find_elites_parquet
from dnadesign.cruncher.utils.hashing import sha256_path
from dnadesign.cruncher.utils.manifest import load_manifest, write_manifest
from dnadesign.cruncher.utils.mpl import ensure_mpl_cache
from dnadesign.cruncher.utils.parquet import read_parquet
from dnadesign.cruncher.utils.run_layout import (
    report_dir,
    sequences_path,
    trace_path,
)

logger = logging.getLogger(__name__)


def _safe_metric(value: float | None, label: str, warnings: list[str]) -> float | None:
    if value is None:
        return None
    if not math.isfinite(value):
        warnings.append(f"{label} was not finite; omitted from report.")
        return None
    return float(value)


def run_report(cfg: CruncherConfig, config_path: Path, run_name: str) -> None:
    ensure_mpl_cache(config_path.parent / cfg.motif_store.catalog_root)
    run = get_run(cfg, config_path, run_name)
    run_dir = run.run_dir

    manifest = load_manifest(run_dir)
    if manifest.get("stage") != "sample":
        raise ValueError(f"Run '{run_name}' is not a sample run (stage={manifest.get('stage')})")

    lock_path_raw = manifest.get("lockfile_path")
    if not lock_path_raw:
        raise ValueError("Run manifest missing lockfile_path; re-run `cruncher sample`.")
    lock_path = Path(lock_path_raw)
    if not lock_path.exists():
        raise FileNotFoundError(f"Lockfile not found: {lock_path}")
    expected_sha = manifest.get("lockfile_sha256")
    if expected_sha:
        actual_sha = sha256_path(lock_path)
        if actual_sha != expected_sha:
            raise ValueError(f"Lockfile checksum mismatch for {lock_path}")

    trace_file = trace_path(run_dir)
    seq_path = sequences_path(run_dir)
    if not trace_file.exists():
        raise FileNotFoundError(
            f"Missing artifacts/trace.nc in {run_dir}. Re-run `cruncher sample` with sample.output.trace.save=true."
        )
    if not seq_path.exists():
        raise FileNotFoundError(
            f"Missing artifacts/sequences.parquet in {run_dir}. "
            "Re-run `cruncher sample` with sample.output.save_sequences=true."
        )

    elite_path = find_elites_parquet(run_dir)

    seq_df = read_parquet(seq_path)
    elites_df = read_parquet(elite_path)

    tf_names = [m["tf_name"] for m in manifest.get("motifs", [])]

    import arviz as az

    idata = az.from_netcdf(trace_file)
    warnings: list[str] = []
    rhat = None
    ess = None
    diagnostics_summary = None
    try:
        diagnostics_summary = summarize_sampling_diagnostics(
            trace_idata=idata,
            sequences_df=seq_df,
            elites_df=elites_df,
            tf_names=tf_names,
            optimizer=manifest.get("optimizer", {}),
            optimizer_stats=manifest.get("optimizer_stats", {}),
            optimizer_kind=(manifest.get("optimizer", {}) or {}).get("kind"),
            sample_meta=None,
        )
        warnings = diagnostics_summary.get("warnings", []) if isinstance(diagnostics_summary, dict) else []
        trace_metrics = diagnostics_summary.get("metrics", {}).get("trace", {}) if diagnostics_summary else {}
        rhat = _safe_metric(trace_metrics.get("rhat"), "R-hat", warnings)
        ess = _safe_metric(trace_metrics.get("ess"), "ESS", warnings)
    except Exception as exc:
        warnings.append(f"Diagnostics summary failed: {exc}")

    report: Dict[str, Any] = {
        "run": run_name,
        "run_dir": str(run_dir.resolve()),
        "stage": manifest.get("stage"),
        "tf_names": tf_names,
        "sequence_length": manifest.get("sequence_length"),
        "n_sequences": int(len(seq_df)),
        "n_elites": int(len(elites_df)),
        "rhat": rhat,
        "ess": ess,
        "optimizer_stats": manifest.get("optimizer_stats", {}),
        "artifacts": manifest.get("artifacts", []),
    }
    if diagnostics_summary:
        report["diagnostics"] = diagnostics_summary
    if warnings:
        report["diagnostics_warnings"] = warnings

    report_root = report_dir(run_dir)
    report_root.mkdir(parents=True, exist_ok=True)
    report_path = report_root / "report.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False))

    latest_analysis_id = None
    latest_analysis_dir = analysis_root(run_dir)
    summary_file = summary_path(latest_analysis_dir)
    if summary_file.exists():
        try:
            summary_payload = json.loads(summary_file.read_text())
        except Exception as exc:
            raise ValueError(f"Invalid analysis summary JSON at {summary_file}: {exc}") from exc
        if isinstance(summary_payload, dict):
            latest_analysis_id = summary_payload.get("analysis_id")

    analysis_links: list[str] = []
    if latest_analysis_id:
        prefix = "analysis/"
        latest_top = {
            "plots",
            "tables",
            "notebooks",
            "meta",
        }
        artifacts = normalize_artifacts(manifest.get("artifacts"))
        for item in artifacts:
            path = str(item.get("path", ""))
            if not path.startswith(prefix) or path.startswith("analysis/_archive/"):
                continue
            rel = path[len(prefix) :]
            head = rel.split("/", 1)[0] if rel else ""
            if head not in latest_top:
                continue
            label = item.get("label") or Path(path).name
            analysis_links.append(f"- {label}: {path}")

    rhat_text = f"{report['rhat']:.3f}" if report["rhat"] is not None else "n/a"
    ess_text = f"{report['ess']:.1f}" if report["ess"] is not None else "n/a"
    md_lines = [
        f"# Cruncher Report: {run_name}",
        "",
        f"- Stage: {report['stage']}",
        f"- TFs: {', '.join(tf_names)}",
        f"- Sequence length: {report['sequence_length']}",
        f"- Sequences sampled: {report['n_sequences']}",
        f"- Elites: {report['n_elites']}",
        f"- R-hat (score): {rhat_text}",
        f"- ESS (score): {ess_text}",
    ]
    if diagnostics_summary:
        diag_status = diagnostics_summary.get("status", "n/a")
        md_lines.insert(7, f"- Diagnostics status: {diag_status}")
    if warnings:
        md_lines.extend(["", "## Diagnostics notes", ""])
        md_lines.extend([f"- {item}" for item in warnings])
    if latest_analysis_id:
        report["analysis_id"] = latest_analysis_id
        md_lines.extend(["", f"## Latest analysis: {latest_analysis_id}", ""])
        if analysis_links:
            md_lines.extend(analysis_links)
        else:
            md_lines.append("- No analysis artifacts recorded.")
    else:
        md_lines.extend(["", "## Latest analysis", "", "- No analysis runs found."])

    md_lines.extend(
        [
            "",
            "## What to look at",
            "",
            "- Summary: analysis/meta/summary.json",
            "- Plots: analysis/plots/",
            "- Tables: analysis/tables/",
        ]
    )
    md_path = report_root / "report.md"
    md_path.write_text("\n".join(md_lines))

    report_artifacts = [
        artifact_entry(report_path, run_dir, kind="json", label="Run report (JSON)", stage="report"),
        artifact_entry(md_path, run_dir, kind="text", label="Run report (Markdown)", stage="report"),
    ]
    append_artifacts(manifest, report_artifacts)
    write_manifest(run_dir, manifest)
    update_run_index_from_manifest(
        config_path,
        run_dir,
        manifest,
        catalog_root=cfg.motif_store.catalog_root,
    )

    logger.info("Wrote report → %s", report_path.relative_to(run_dir.parent))
    logger.info("Wrote report → %s", md_path.relative_to(run_dir.parent))
