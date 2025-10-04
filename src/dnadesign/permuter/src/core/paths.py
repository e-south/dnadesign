"""
--------------------------------------------------------------------------------
<dnadesign project>
src/dnadesign/permuter/src/core/paths.py

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class JobPaths:
    job_yaml: Path
    job_dir: Path
    refs_csv: Path
    output_root: Path
    dataset_dir: Path
    records_parquet: Path
    ref_fa: Path
    plots_dir: Path


def _expand(s: str, *, job_dir: Path) -> Path:
    # expand ~ and $VARS and ${JOB_DIR}
    s = s or ""
    s = s.replace("${JOB_DIR}", str(job_dir))
    s = os.path.expandvars(s)
    p = Path(os.path.expanduser(s))
    return p if p.is_absolute() else (job_dir / p)


def resolve(
    job_yaml: Path,
    *,
    refs: str,
    output_dir: str,
    ref_name: str,
    out_override: Path | None,
) -> JobPaths:
    job_yaml = job_yaml.resolve()
    job_dir = job_yaml.parent

    refs_csv = _expand(refs, job_dir=job_dir)
    if not refs_csv.exists():
        raise FileNotFoundError(f"Refs CSV not found: {refs_csv}")

    output_root = _expand(output_dir, job_dir=job_dir)
    # If --out provided, treat it as *root*; dataset lives under <out>/<ref_name>
    dataset_dir = (
        (out_override / ref_name) if out_override else (output_root / ref_name)
    )

    records_parquet = dataset_dir / "records.parquet"
    ref_fa = dataset_dir / "REF.fa"
    plots_dir = dataset_dir / "plots"

    return JobPaths(
        job_yaml=job_yaml,
        job_dir=job_dir,
        refs_csv=refs_csv,
        output_root=output_root,
        dataset_dir=dataset_dir,
        records_parquet=records_parquet,
        ref_fa=ref_fa,
        plots_dir=plots_dir,
    )
