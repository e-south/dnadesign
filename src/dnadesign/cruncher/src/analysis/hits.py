"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/analysis/hits.py

Load and validate analysis hit and occurrence metadata.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from dnadesign.cruncher.analysis.parquet import read_parquet
from dnadesign.cruncher.artifacts.manifest import load_manifest

REQUIRED_HITS_COLUMNS = {
    "best_core_offset",
    "best_core_seq",
    "best_score_norm",
    "best_score_raw",
    "best_score_scaled",
    "best_start",
    "best_strand",
    "best_window_seq",
    "core_def_hash",
    "core_width",
    "draw_idx",
    "elite_id",
    "pwm_hash",
    "pwm_ref",
    "pwm_width",
    "rank",
    "tf",
    "tiebreak_rule",
}

REQUIRED_BASELINE_HITS_COLUMNS = {
    "baseline_id",
    "tf",
    "best_start",
    "best_core_offset",
    "best_strand",
    "best_window_seq",
    "best_core_seq",
    "best_score_raw",
    "best_score_scaled",
    "best_score_norm",
    "tiebreak_rule",
    "pwm_ref",
    "pwm_hash",
    "pwm_width",
    "core_width",
    "core_def_hash",
}

REQUIRED_ELITE_OCCURRENCES_COLUMNS = {
    "elite_id",
    "objective_id",
    "tf",
    "occurrence_rank",
    "start",
    "end",
    "strand",
    "raw_score",
    "scaled_score",
    "normalized_score",
    "selected",
}

REQUIRED_BASELINE_OCCURRENCES_COLUMNS = {
    "baseline_id",
    "objective_id",
    "tf",
    "occurrence_rank",
    "start",
    "end",
    "strand",
    "raw_score",
    "scaled_score",
    "normalized_score",
    "selected",
}


def validate_elites_hits_df(df: pd.DataFrame) -> None:
    missing = [col for col in sorted(REQUIRED_HITS_COLUMNS) if col not in df.columns]
    if missing:
        raise ValueError(f"elites_hits.parquet missing required columns: {missing}")


def representative_hit_contract_enabled(objective_payload: dict[str, object]) -> bool:
    if "representative_hit_contract" in objective_payload:
        return bool(objective_payload.get("representative_hit_contract", True))
    return bool(objective_payload.get("legacy_hit_contract", True))


def require_representative_hit_contract(run_dir: Path, *, context: str) -> None:
    manifest = load_manifest(run_dir)
    objective_payload = manifest.get("objective")
    if not isinstance(objective_payload, dict):
        return
    if representative_hit_contract_enabled(objective_payload):
        return
    raise ValueError(
        f"{context} is unsupported for occurrence-aware sample runs because the run manifest declares "
        "representative_hit_contract=false. Use occurrence-aware artifacts instead."
    )


def require_legacy_hit_contract(run_dir: Path, *, context: str) -> None:
    require_representative_hit_contract(run_dir, context=context)


def load_elites_hits(path: Path) -> pd.DataFrame:
    try:
        run_dir = path.parent.parent.parent
        require_representative_hit_contract(run_dir, context="Loading elites_hits.parquet")
    except FileNotFoundError:
        pass
    df = read_parquet(path)
    validate_elites_hits_df(df)
    return df


def validate_baseline_hits_df(df: pd.DataFrame) -> None:
    missing = [col for col in sorted(REQUIRED_BASELINE_HITS_COLUMNS) if col not in df.columns]
    if missing:
        raise ValueError(f"random_baseline_hits.parquet missing required columns: {missing}")


def load_baseline_hits(path: Path) -> pd.DataFrame:
    try:
        run_dir = path.parent.parent.parent
        require_representative_hit_contract(run_dir, context="Loading random_baseline_hits.parquet")
    except FileNotFoundError:
        pass
    df = read_parquet(path)
    validate_baseline_hits_df(df)
    return df


def validate_elite_occurrences_df(df: pd.DataFrame) -> None:
    missing = [col for col in sorted(REQUIRED_ELITE_OCCURRENCES_COLUMNS) if col not in df.columns]
    if missing:
        raise ValueError(f"elites_occurrences.parquet missing required columns: {missing}")


def validate_baseline_occurrences_df(df: pd.DataFrame) -> None:
    missing = [col for col in sorted(REQUIRED_BASELINE_OCCURRENCES_COLUMNS) if col not in df.columns]
    if missing:
        raise ValueError(f"random_baseline_occurrences.parquet missing required columns: {missing}")


def load_elite_occurrences(path: Path) -> pd.DataFrame:
    df = read_parquet(path)
    validate_elite_occurrences_df(df)
    return df


def load_baseline_occurrences(path: Path) -> pd.DataFrame:
    df = read_parquet(path)
    validate_baseline_occurrences_df(df)
    return df
