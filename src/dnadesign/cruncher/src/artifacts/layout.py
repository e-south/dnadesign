"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/artifacts/layout.py

Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from dnadesign.cruncher.core.labels import format_regulator_slug
from dnadesign.cruncher.utils.paths import resolve_workspace_root

# Run-level IA:
# - meta/: run metadata/state files
# - provenance/: resolved lock/parse inputs pinned for reproducibility
# - optimize/: sampler outputs split into tables/ + state/
# - analysis/: analysis outputs (managed by analysis/layout.py)
# - plots/: all plot outputs (flat filenames)
# - export/: downstream contract exports
RUN_META_DIR = "meta"
RUN_ARTIFACTS_DIR = "artifacts"
RUN_LIVE_DIR = "live"
RUN_PROVENANCE_DIR = "provenance"
RUN_OPTIMIZE_DIR = "optimize"
RUN_OPTIMIZE_TABLES_DIR = "tables"
RUN_OPTIMIZE_META_DIR = "state"
RUN_OUTPUT_DIR = "analysis"
RUN_PLOTS_DIR = "plots"
RUN_EXPORT_DIR = "export"
# Logos now live under `plots/*`, so preserve the top-level `plots/`.
LOGOS_ROOT_DIR = RUN_PLOTS_DIR


def out_root(config_path: Path, out_dir: str | Path) -> Path:
    return resolve_workspace_root(config_path) / Path(out_dir)


def stage_root(out_root_path: Path, stage: str) -> Path:
    stage_name = str(stage).strip().lower()
    if stage_name == "parse":
        return out_root_path.parent / ".cruncher" / "parse"
    return out_root_path


def runs_root(out_root_path: Path) -> Path:
    return out_root_path


def run_group_label(
    tfs: Iterable[str],
    set_index: int | None,
    *,
    include_set_index: bool = False,
) -> str:
    label = format_regulator_slug(list(tfs))
    if include_set_index and set_index is not None:
        return f"set{set_index}_{label}"
    return label


def run_group_dir(
    out_root_path: Path,
    stage: str,
    tfs: Iterable[str],
    set_index: int | None,
) -> Path:
    return runs_root(stage_root(out_root_path, stage))


def build_run_dir(
    *,
    config_path: Path,
    out_dir: str | Path,
    stage: str,
    tfs: Iterable[str],
    set_index: int | None,
    include_set_index: bool = False,
) -> Path:
    root = run_group_dir(out_root(config_path, out_dir), stage, tfs, set_index)
    if include_set_index:
        label = run_group_label(tfs, set_index, include_set_index=include_set_index)
        return root / label
    return root


def ensure_run_dirs(
    run_dir: Path,
    *,
    meta: bool = True,
    artifacts: bool = False,
    live: bool = False,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    if meta:
        run_meta_dir(run_dir).mkdir(parents=True, exist_ok=True)
    if meta or artifacts or live:
        run_provenance_dir(run_dir).mkdir(parents=True, exist_ok=True)
    if artifacts or live:
        run_optimize_dir(run_dir).mkdir(parents=True, exist_ok=True)
        run_optimize_tables_dir(run_dir).mkdir(parents=True, exist_ok=True)
        run_optimize_meta_dir(run_dir).mkdir(parents=True, exist_ok=True)
    if artifacts:
        run_output_dir(run_dir).mkdir(parents=True, exist_ok=True)
        run_plots_dir(run_dir).mkdir(parents=True, exist_ok=True)
        run_export_dir(run_dir).mkdir(parents=True, exist_ok=True)


def run_meta_dir(run_dir: Path) -> Path:
    return run_dir / RUN_META_DIR


def run_provenance_dir(run_dir: Path) -> Path:
    return run_dir / RUN_PROVENANCE_DIR


def run_optimize_dir(run_dir: Path) -> Path:
    return run_dir / RUN_OPTIMIZE_DIR


def run_optimize_tables_dir(run_dir: Path) -> Path:
    return run_optimize_dir(run_dir) / RUN_OPTIMIZE_TABLES_DIR


def run_optimize_meta_dir(run_dir: Path) -> Path:
    return run_optimize_dir(run_dir) / RUN_OPTIMIZE_META_DIR


def run_output_dir(run_dir: Path) -> Path:
    return run_dir / RUN_OUTPUT_DIR


def run_plots_dir(run_dir: Path) -> Path:
    return run_dir / RUN_PLOTS_DIR


def run_plots_analysis_dir(run_dir: Path) -> Path:
    return run_plots_dir(run_dir)


def run_plots_logos_dir(run_dir: Path) -> Path:
    return run_plots_dir(run_dir)


def run_export_dir(run_dir: Path) -> Path:
    return run_dir / RUN_EXPORT_DIR


def run_export_sequences_dir(run_dir: Path) -> Path:
    return run_export_dir(run_dir)


def run_export_sequences_manifest_path(run_dir: Path) -> Path:
    return run_export_sequences_dir(run_dir) / "export_manifest.json"


def run_export_sequences_table_path(run_dir: Path, *, table_name: str, fmt: str) -> Path:
    normalized_fmt = str(fmt).strip().lower()
    if normalized_fmt not in {"parquet", "csv"}:
        raise ValueError(f"Unsupported export table format: {fmt!r}")
    if not table_name.strip():
        raise ValueError("table_name must be non-empty")
    return run_export_sequences_dir(run_dir) / f"table__{table_name}.{normalized_fmt}"


def run_export_table_path(run_dir: Path, *, table_name: str, fmt: str) -> Path:
    normalized_fmt = str(fmt).strip().lower()
    if normalized_fmt not in {"parquet", "csv"}:
        raise ValueError(f"Unsupported export table format: {fmt!r}")
    if not table_name.strip():
        raise ValueError("table_name must be non-empty")
    return run_export_dir(run_dir) / f"table__{table_name}.{normalized_fmt}"


def manifest_path(run_dir: Path) -> Path:
    return run_meta_dir(run_dir) / "run_manifest.json"


def status_path(run_dir: Path) -> Path:
    return run_meta_dir(run_dir) / "run_status.json"


def config_used_path(run_dir: Path) -> Path:
    return run_meta_dir(run_dir) / "config_used.yaml"


def live_metrics_path(run_dir: Path) -> Path:
    return run_optimize_meta_dir(run_dir) / "metrics.jsonl"


def trace_path(run_dir: Path) -> Path:
    return run_optimize_meta_dir(run_dir) / "trace.nc"


def sequences_path(run_dir: Path) -> Path:
    return run_optimize_tables_dir(run_dir) / "sequences.parquet"


def random_baseline_path(run_dir: Path) -> Path:
    return run_optimize_tables_dir(run_dir) / "random_baseline.parquet"


def random_baseline_hits_path(run_dir: Path) -> Path:
    return run_optimize_tables_dir(run_dir) / "random_baseline_hits.parquet"


def elites_path(run_dir: Path) -> Path:
    return run_optimize_tables_dir(run_dir) / "elites.parquet"


def elites_mmr_meta_path(run_dir: Path) -> Path:
    return run_optimize_tables_dir(run_dir) / "elites_mmr_meta.parquet"


def elites_hits_path(run_dir: Path) -> Path:
    return run_optimize_tables_dir(run_dir) / "elites_hits.parquet"


def elites_json_path(run_dir: Path) -> Path:
    return run_optimize_meta_dir(run_dir) / "elites.json"


def elites_yaml_path(run_dir: Path) -> Path:
    return run_optimize_meta_dir(run_dir) / "elites.yaml"


def lockfile_snapshot_path(run_dir: Path) -> Path:
    return run_provenance_dir(run_dir) / "lockfile.json"


def parse_manifest_path(run_dir: Path) -> Path:
    return run_provenance_dir(run_dir) / "parse_manifest.json"


def pwm_summary_path(run_dir: Path) -> Path:
    return run_provenance_dir(run_dir) / "pwm_summary.json"


def logos_root(out_root_path: Path) -> Path:
    return out_root_path / RUN_PLOTS_DIR
