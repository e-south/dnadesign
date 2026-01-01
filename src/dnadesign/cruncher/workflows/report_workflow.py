"""Generate a concise run report from a completed sample run."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict

import arviz as az
import pandas as pd

from dnadesign.cruncher.config.schema_v2 import CruncherConfig
from dnadesign.cruncher.services.run_service import update_run_index_from_manifest
from dnadesign.cruncher.utils.hashing import sha256_path
from dnadesign.cruncher.utils.manifest import load_manifest, write_manifest

logger = logging.getLogger(__name__)


def run_report(cfg: CruncherConfig, config_path: Path, batch_name: str) -> None:
    base_out = config_path.parent / cfg.out_dir
    run_dir = base_out / batch_name
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    manifest = load_manifest(run_dir)
    if manifest.get("stage") != "sample":
        raise ValueError(f"Run '{batch_name}' is not a sample run (stage={manifest.get('stage')})")

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

    trace_path = run_dir / "trace.nc"
    seq_path = run_dir / "sequences.parquet"
    if not trace_path.exists():
        raise FileNotFoundError(f"Missing trace.nc in {run_dir}. Re-run `cruncher sample` with sample.save_trace=true.")
    if not seq_path.exists():
        raise FileNotFoundError(
            f"Missing sequences.parquet in {run_dir}. Re-run `cruncher sample` with sample.save_sequences=true."
        )

    elite_files = list(run_dir.glob("cruncher_elites_*/*.parquet"))
    if not elite_files:
        raise FileNotFoundError(f"Missing elites parquet in {run_dir}")
    elite_path = max(elite_files, key=lambda p: p.stat().st_mtime)

    seq_df = pd.read_parquet(seq_path)
    elites_df = pd.read_parquet(elite_path)

    idata = az.from_netcdf(trace_path)
    rhat = az.rhat(idata, var_names=["score"])["score"].item()
    ess = az.ess(idata, var_names=["score"])["score"].item()

    tf_names = [m["tf_name"] for m in manifest.get("motifs", [])]
    report: Dict[str, Any] = {
        "run": batch_name,
        "run_dir": str(run_dir.resolve()),
        "stage": manifest.get("stage"),
        "tf_names": tf_names,
        "sequence_length": manifest.get("sequence_length"),
        "n_sequences": int(len(seq_df)),
        "n_elites": int(len(elites_df)),
        "rhat": float(rhat),
        "ess": float(ess),
        "optimizer_stats": manifest.get("optimizer_stats", {}),
        "artifacts": manifest.get("artifacts", []),
    }

    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))

    md_lines = [
        f"# Cruncher Report: {batch_name}",
        "",
        f"- Stage: {report['stage']}",
        f"- TFs: {', '.join(tf_names)}",
        f"- Sequence length: {report['sequence_length']}",
        f"- Sequences sampled: {report['n_sequences']}",
        f"- Elites: {report['n_elites']}",
        f"- R-hat (score): {report['rhat']:.3f}",
        f"- ESS (score): {report['ess']:.1f}",
    ]
    md_path = run_dir / "report.md"
    md_path.write_text("\n".join(md_lines))

    artifacts = list(manifest.get("artifacts", []))
    for item in ("report.json", "report.md"):
        if item not in artifacts:
            artifacts.append(item)
    manifest["artifacts"] = artifacts
    write_manifest(run_dir, manifest)
    update_run_index_from_manifest(config_path, run_dir, manifest)

    logger.info("Wrote report → %s", report_path.relative_to(run_dir.parent))
    logger.info("Wrote report → %s", md_path.relative_to(run_dir.parent))
