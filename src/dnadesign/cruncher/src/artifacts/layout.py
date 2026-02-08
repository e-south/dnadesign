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

# Run metadata/artifact files are written directly under each run directory.
RUN_META_DIR = ""
RUN_ARTIFACTS_DIR = ""
RUN_LIVE_DIR = ""
RUNS_ROOT_DIR = "runs"
RUN_INPUT_DIR = "input"
RUN_OPTIMIZE_DIR = "optimize"
RUN_OUTPUT_DIR = "output"
RUN_PLOTS_DIR = "plots"
LOGOS_ROOT_DIR = "logos"
CAMPAIGN_ROOT_DIR = "campaign"


def out_root(config_path: Path, out_dir: str | Path) -> Path:
    return config_path.parent / Path(out_dir)


def stage_root(out_root_path: Path, stage: str) -> Path:
    return out_root_path / stage


def runs_root(out_root_path: Path) -> Path:
    return out_root_path / RUNS_ROOT_DIR


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
    return runs_root(out_root_path)


def build_run_dir(
    *,
    config_path: Path,
    out_dir: str | Path,
    stage: str,
    tfs: Iterable[str],
    set_index: int | None,
    include_set_index: bool = False,
    slot: str = "latest",
) -> Path:
    root = run_group_dir(out_root(config_path, out_dir), stage, tfs, set_index)
    if include_set_index:
        label = run_group_label(tfs, set_index, include_set_index=include_set_index)
        return root / label / slot
    return root / slot


def ensure_run_dirs(
    run_dir: Path,
    *,
    meta: bool = True,
    artifacts: bool = False,
    live: bool = False,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    run_input_dir(run_dir).mkdir(parents=True, exist_ok=True)
    run_optimize_dir(run_dir).mkdir(parents=True, exist_ok=True)
    run_output_dir(run_dir).mkdir(parents=True, exist_ok=True)
    run_plots_dir(run_dir).mkdir(parents=True, exist_ok=True)


def run_input_dir(run_dir: Path) -> Path:
    return run_dir / RUN_INPUT_DIR


def run_optimize_dir(run_dir: Path) -> Path:
    return run_dir / RUN_OPTIMIZE_DIR


def run_output_dir(run_dir: Path) -> Path:
    return run_dir / RUN_OUTPUT_DIR


def run_plots_dir(run_dir: Path) -> Path:
    return run_dir / RUN_PLOTS_DIR


def manifest_path(run_dir: Path) -> Path:
    return run_dir / "run_manifest.json"


def status_path(run_dir: Path) -> Path:
    return run_dir / "run_status.json"


def config_used_path(run_dir: Path) -> Path:
    return run_dir / "config_used.yaml"


def live_metrics_path(run_dir: Path) -> Path:
    return run_optimize_dir(run_dir) / "metrics.jsonl"


def trace_path(run_dir: Path) -> Path:
    return run_optimize_dir(run_dir) / "trace.nc"


def sequences_path(run_dir: Path) -> Path:
    return run_optimize_dir(run_dir) / "sequences.parquet"


def random_baseline_path(run_dir: Path) -> Path:
    return run_optimize_dir(run_dir) / "random_baseline.parquet"


def random_baseline_hits_path(run_dir: Path) -> Path:
    return run_optimize_dir(run_dir) / "random_baseline_hits.parquet"


def elites_path(run_dir: Path) -> Path:
    return run_optimize_dir(run_dir) / "elites.parquet"


def elites_mmr_meta_path(run_dir: Path) -> Path:
    return run_optimize_dir(run_dir) / "elites_mmr_meta.parquet"


def elites_hits_path(run_dir: Path) -> Path:
    return run_optimize_dir(run_dir) / "elites_hits.parquet"


def elites_json_path(run_dir: Path) -> Path:
    return run_optimize_dir(run_dir) / "elites.json"


def elites_yaml_path(run_dir: Path) -> Path:
    return run_optimize_dir(run_dir) / "elites.yaml"


def lockfile_snapshot_path(run_dir: Path) -> Path:
    return run_input_dir(run_dir) / "lockfile.json"


def parse_manifest_path(run_dir: Path) -> Path:
    return run_input_dir(run_dir) / "parse_manifest.json"


def pwm_summary_path(run_dir: Path) -> Path:
    return run_input_dir(run_dir) / "pwm_summary.json"


def logos_root(out_root_path: Path) -> Path:
    return out_root_path / LOGOS_ROOT_DIR


def logos_dir_for_run(out_root_path: Path, stage: str, run_name: str) -> Path:
    return logos_root(out_root_path) / stage / run_name


def campaign_name_slug(name: str) -> str:
    return format_regulator_slug([name], max_len=80)


def campaign_stage_root(
    *,
    config_path: Path,
    out_dir: str | Path,
    campaign_name: str,
) -> Path:
    return out_root(config_path, out_dir) / CAMPAIGN_ROOT_DIR / campaign_name_slug(campaign_name)


def campaign_slot_dir(
    *,
    config_path: Path,
    out_dir: str | Path,
    campaign_name: str,
    slot: str = "latest",
) -> Path:
    return campaign_stage_root(config_path=config_path, out_dir=out_dir, campaign_name=campaign_name) / slot
