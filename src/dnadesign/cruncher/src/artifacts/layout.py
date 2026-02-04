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

from dnadesign.cruncher.core.labels import build_run_name, format_regulator_slug

RUN_META_DIR = "meta"
RUN_ARTIFACTS_DIR = "artifacts"
RUN_LIVE_DIR = "live"
LOGOS_ROOT_DIR = "logos"


def out_root(config_path: Path, out_dir: str | Path) -> Path:
    return config_path.parent / Path(out_dir)


def stage_root(out_root_path: Path, stage: str) -> Path:
    return out_root_path / stage


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
    return stage_root(out_root_path, stage)


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
    return root / build_run_name(
        stage,
        list(tfs),
        set_index=set_index,
        include_stage=False,
        include_label=True,
        include_set_index=include_set_index,
    )


def ensure_run_dirs(
    run_dir: Path,
    *,
    meta: bool = True,
    artifacts: bool = False,
    live: bool = False,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    if meta:
        (run_dir / RUN_META_DIR).mkdir(parents=True, exist_ok=True)
    if artifacts:
        (run_dir / RUN_ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
    if live:
        (run_dir / RUN_LIVE_DIR).mkdir(parents=True, exist_ok=True)


def manifest_path(run_dir: Path) -> Path:
    return run_dir / RUN_META_DIR / "run_manifest.json"


def status_path(run_dir: Path) -> Path:
    return run_dir / RUN_META_DIR / "run_status.json"


def config_used_path(run_dir: Path) -> Path:
    return run_dir / RUN_META_DIR / "config_used.yaml"


def live_metrics_path(run_dir: Path) -> Path:
    return run_dir / RUN_LIVE_DIR / "metrics.jsonl"


def trace_path(run_dir: Path) -> Path:
    return run_dir / RUN_ARTIFACTS_DIR / "trace.nc"


def sequences_path(run_dir: Path) -> Path:
    return run_dir / RUN_ARTIFACTS_DIR / "sequences.parquet"


def elites_path(run_dir: Path) -> Path:
    return run_dir / RUN_ARTIFACTS_DIR / "elites.parquet"


def elites_top_score_path(run_dir: Path) -> Path:
    return run_dir / RUN_ARTIFACTS_DIR / "elites_top_score.parquet"


def elites_mmr_path(run_dir: Path) -> Path:
    return run_dir / RUN_ARTIFACTS_DIR / "elites_mmr.parquet"


def elites_mmr_meta_path(run_dir: Path) -> Path:
    return run_dir / RUN_ARTIFACTS_DIR / "elites_mmr_meta.parquet"


def elites_json_path(run_dir: Path) -> Path:
    return run_dir / RUN_ARTIFACTS_DIR / "elites.json"


def elites_yaml_path(run_dir: Path) -> Path:
    return run_dir / RUN_ARTIFACTS_DIR / "elites.yaml"


def logos_root(out_root_path: Path) -> Path:
    return out_root_path / LOGOS_ROOT_DIR


def logos_dir_for_run(out_root_path: Path, stage: str, run_name: str) -> Path:
    return logos_root(out_root_path) / stage / run_name
