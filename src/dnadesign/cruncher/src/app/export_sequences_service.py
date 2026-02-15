"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/export_sequences_service.py

Export sequence-oriented run artifacts for downstream operators.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import pandas as pd

from dnadesign.cruncher.analysis.consensus import pwm_consensus
from dnadesign.cruncher.analysis.hits import load_elites_hits
from dnadesign.cruncher.analysis.parquet import read_parquet, write_parquet
from dnadesign.cruncher.app.analyze.metadata import load_pwms_from_config
from dnadesign.cruncher.app.run_service import get_run, list_runs
from dnadesign.cruncher.artifacts.atomic_write import atomic_write_json, atomic_write_text
from dnadesign.cruncher.artifacts.entries import append_artifacts, artifact_entry
from dnadesign.cruncher.artifacts.layout import (
    config_used_path,
    elites_hits_path,
    elites_path,
    manifest_path,
    run_export_sequences_manifest_path,
    run_export_sequences_table_path,
)
from dnadesign.cruncher.artifacts.manifest import load_manifest, write_manifest
from dnadesign.cruncher.config.schema_v3 import CruncherConfig

_ELITES_REQUIRED_COLUMNS = {
    "id",
    "rank",
    "sequence",
}

_WINDOW_COLUMNS = [
    "elite_id",
    "elite_rank",
    "elite_sequence",
    "tf",
    "best_start",
    "best_end",
    "best_strand",
    "best_window_seq",
    "best_core_seq",
    "best_score_raw",
    "best_score_scaled",
    "best_score_norm",
    "pwm_ref",
    "pwm_hash",
    "pwm_width",
    "core_width",
]

_BISPECIFIC_COLUMNS = [
    "elite_id",
    "elite_rank",
    "elite_sequence",
    "tf_a",
    "tf_b",
    "window_a_seq",
    "window_b_seq",
    "core_a_seq",
    "core_b_seq",
    "start_a",
    "end_a",
    "start_b",
    "end_b",
    "strand_a",
    "strand_b",
    "score_norm_a",
    "score_norm_b",
    "pair_min_score_norm",
    "pair_mean_score_norm",
    "overlap_bp",
    "gap_bp",
    "pair_span_start",
    "pair_span_end",
    "pair_span_width",
]

_MULTISPECIFIC_COLUMNS = [
    "elite_id",
    "elite_rank",
    "elite_sequence",
    "combo_size",
    "tf_combo_key",
    "tf_members_json",
    "member_windows_json",
    "combo_min_score_norm",
    "combo_mean_score_norm",
    "combo_min_score_scaled",
    "combo_mean_score_scaled",
    "combo_span_start",
    "combo_span_end",
    "combo_span_width",
    "combo_overlap_total_bp",
    "forward_count",
    "reverse_count",
    "member_count",
]


@dataclass(frozen=True)
class SequenceExportResult:
    run_name: str
    run_dir: Path
    output_dir: Path
    manifest_path: Path
    files: dict[str, Path]
    row_counts: dict[str, int]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_table_format(table_format: str) -> str:
    normalized = str(table_format).strip().lower()
    if normalized not in {"parquet", "csv"}:
        raise ValueError(f"table_format must be 'parquet' or 'csv', got {table_format!r}.")
    return normalized


def _write_table(df: pd.DataFrame, path: Path, *, table_format: str) -> None:
    if table_format == "parquet":
        write_parquet(df, path)
        return
    if table_format == "csv":
        atomic_write_text(path, df.to_csv(index=False))
        return
    raise ValueError(f"Unsupported table format: {table_format!r}")


def _required_columns(df: pd.DataFrame, required: set[str], *, context: str) -> None:
    missing = sorted(column for column in required if column not in df.columns)
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")


def _require_non_null(df: pd.DataFrame, columns: list[str], *, context: str) -> None:
    for column in columns:
        if df[column].isna().any():
            raise ValueError(f"{context} contains null values in column '{column}'.")


def _require_finite(df: pd.DataFrame, columns: list[str], *, context: str) -> None:
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"{context} contains non-numeric values in column '{column}'.")
        if ((values == float("inf")) | (values == float("-inf"))).any():
            raise ValueError(f"{context} contains non-finite values in column '{column}'.")


def _split_ref(ref: str, *, tf_name: str) -> tuple[str, str]:
    raw = str(ref).strip()
    if ":" not in raw:
        raise ValueError(f"Invalid pwm_ref for TF '{tf_name}': {ref!r}; expected '<source>:<motif_id>'.")
    source, motif_id = raw.split(":", 1)
    if not source or not motif_id:
        raise ValueError(f"Invalid pwm_ref for TF '{tf_name}': {ref!r}; expected '<source>:<motif_id>'.")
    return source, motif_id


def _interval_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, min(end_a, end_b) - max(start_a, start_b))


def _interval_gap(start_a: int, end_a: int, start_b: int, end_b: int) -> int:
    return max(0, max(start_a, start_b) - min(end_a, end_b))


def _pairwise_overlap_total(intervals: list[tuple[int, int]]) -> int:
    total = 0
    for (start_a, end_a), (start_b, end_b) in combinations(intervals, 2):
        total += _interval_overlap(start_a, end_a, start_b, end_b)
    return int(total)


def _validate_hits_contract(hits_df: pd.DataFrame) -> None:
    duplicate = hits_df.groupby(["elite_id", "tf"]).size()
    duplicate = duplicate[duplicate > 1]
    if not duplicate.empty:
        labels = [f"({elite_id},{tf})x{int(count)}" for (elite_id, tf), count in duplicate.items()]
        raise ValueError(f"elites_hits.parquet contains duplicate elite/tf rows: {labels}")


def _build_window_rows(elites_df: pd.DataFrame, hits_df: pd.DataFrame) -> pd.DataFrame:
    _required_columns(elites_df, _ELITES_REQUIRED_COLUMNS, context="elites.parquet")
    _validate_hits_contract(hits_df)

    elite_df = elites_df.loc[:, ["id", "rank", "sequence"]].copy()
    elite_df["id"] = elite_df["id"].astype(str)
    elite_df["sequence"] = elite_df["sequence"].astype(str)
    elite_df["rank"] = pd.to_numeric(elite_df["rank"], errors="coerce")
    if elite_df["rank"].isna().any():
        raise ValueError("elites.parquet contains non-numeric rank values.")
    elite_df["rank"] = elite_df["rank"].astype(int)
    elite_df = elite_df.rename(
        columns={
            "id": "elite_id_from_elites",
            "rank": "elite_rank_from_elites",
            "sequence": "elite_sequence",
        }
    )

    merged = hits_df.merge(
        elite_df,
        left_on="elite_id",
        right_on="elite_id_from_elites",
        how="left",
        validate="many_to_one",
    )
    if merged["elite_id_from_elites"].isna().any():
        missing_elites = sorted(str(x) for x in merged.loc[merged["elite_id_from_elites"].isna(), "elite_id"].unique())
        raise ValueError(f"elites_hits.parquet references elite IDs missing from elites.parquet: {missing_elites}")

    merged["best_start"] = pd.to_numeric(merged["best_start"], errors="coerce")
    merged["pwm_width"] = pd.to_numeric(merged["pwm_width"], errors="coerce")
    if merged["best_start"].isna().any() or merged["pwm_width"].isna().any():
        raise ValueError("elites_hits.parquet best_start/pwm_width must be numeric.")
    merged["best_start"] = merged["best_start"].astype(int)
    merged["pwm_width"] = merged["pwm_width"].astype(int)
    if (merged["best_start"] < 0).any():
        raise ValueError("elites_hits.parquet best_start must be >= 0.")
    if (merged["pwm_width"] < 1).any():
        raise ValueError("elites_hits.parquet pwm_width must be >= 1.")
    merged["best_end"] = merged["best_start"] + merged["pwm_width"]
    merged["elite_rank_from_elites"] = merged["elite_rank_from_elites"].astype(int)
    hit_ranks = pd.to_numeric(merged["rank"], errors="coerce")
    if hit_ranks.isna().any():
        raise ValueError("elites_hits.parquet rank must be numeric.")
    if (hit_ranks.astype(int) != merged["elite_rank_from_elites"]).any():
        raise ValueError("elites_hits.parquet rank values do not match elites.parquet rank values.")

    window_df = pd.DataFrame(
        {
            "elite_id": merged["elite_id"].astype(str),
            "elite_rank": merged["elite_rank_from_elites"].astype(int),
            "elite_sequence": merged["elite_sequence"].astype(str),
            "tf": merged["tf"].astype(str),
            "best_start": merged["best_start"].astype(int),
            "best_end": merged["best_end"].astype(int),
            "best_strand": merged["best_strand"].astype(str),
            "best_window_seq": merged["best_window_seq"].astype(str),
            "best_core_seq": merged["best_core_seq"].astype(str),
            "best_score_raw": pd.to_numeric(merged["best_score_raw"], errors="coerce").astype(float),
            "best_score_scaled": pd.to_numeric(merged["best_score_scaled"], errors="coerce").astype(float),
            "best_score_norm": pd.to_numeric(merged["best_score_norm"], errors="coerce").astype(float),
            "pwm_ref": merged["pwm_ref"].astype(str),
            "pwm_hash": merged["pwm_hash"].astype(str),
            "pwm_width": merged["pwm_width"].astype(int),
            "core_width": pd.to_numeric(merged["core_width"], errors="coerce").fillna(merged["pwm_width"]).astype(int),
        }
    )
    _require_non_null(
        window_df,
        columns=[
            "elite_id",
            "tf",
            "best_strand",
            "best_window_seq",
            "best_core_seq",
            "pwm_ref",
            "pwm_hash",
        ],
        context="elites_hits.parquet",
    )
    _require_finite(
        window_df,
        columns=["best_score_raw", "best_score_scaled", "best_score_norm"],
        context="elites_hits.parquet",
    )
    if (window_df["core_width"] < 1).any():
        raise ValueError("elites_hits.parquet core_width must be >= 1.")
    invalid_strand = sorted(set(window_df["best_strand"]) - {"+", "-"})
    if invalid_strand:
        raise ValueError(f"elites_hits.parquet best_strand must be '+' or '-', found invalid values: {invalid_strand}")
    seq_lengths = window_df["elite_sequence"].astype(str).str.len()
    if (window_df["best_end"] > seq_lengths).any():
        offenders = window_df.loc[window_df["best_end"] > seq_lengths, "elite_id"].astype(str).unique().tolist()
        raise ValueError(f"elites_hits.parquet contains out-of-bounds windows for elite IDs: {sorted(offenders)}")
    window_lengths = window_df["best_window_seq"].astype(str).str.len()
    if (window_lengths != window_df["pwm_width"]).any():
        raise ValueError("elites_hits.parquet best_window_seq length must match pwm_width.")
    core_lengths = window_df["best_core_seq"].astype(str).str.len()
    if (core_lengths != window_df["core_width"]).any():
        raise ValueError("elites_hits.parquet best_core_seq length must match core_width.")
    window_df = window_df.sort_values(["elite_rank", "elite_id", "tf"]).reset_index(drop=True)
    return window_df


def _build_consensus_rows(window_df: pd.DataFrame, pwms: dict[str, object]) -> tuple[pd.DataFrame, list[str]]:
    tf_names = sorted(window_df["tf"].astype(str).unique().tolist())
    if not tf_names:
        raise ValueError("No TF rows found in elites_hits.parquet for sequence export.")
    rows: list[dict[str, object]] = []
    for tf_name in tf_names:
        tf_df = window_df[window_df["tf"] == tf_name]
        refs = sorted(set(tf_df["pwm_ref"].astype(str)))
        hashes = sorted(set(tf_df["pwm_hash"].astype(str)))
        widths = sorted(int(v) for v in tf_df["pwm_width"].astype(int).unique().tolist())
        if len(refs) != 1:
            raise ValueError(f"Expected one pwm_ref for TF '{tf_name}', found {refs}.")
        if len(hashes) != 1:
            raise ValueError(f"Expected one pwm_hash for TF '{tf_name}', found {hashes}.")
        if len(widths) != 1:
            raise ValueError(f"Expected one pwm_width for TF '{tf_name}', found {widths}.")
        pwm = pwms.get(tf_name)
        if pwm is None:
            raise ValueError(f"Missing PWM for TF '{tf_name}' in config_used.yaml.")
        consensus = pwm_consensus(pwm)
        pwm_width = int(widths[0])
        if len(consensus) != int(pwm.length):
            raise ValueError(
                f"Consensus width mismatch for TF '{tf_name}': consensus={len(consensus)} pwm_length={pwm.length}"
            )
        if int(pwm.length) != pwm_width:
            raise ValueError(
                f"PWM width mismatch for TF '{tf_name}': config_used width={pwm.length} hits width={pwm_width}"
            )
        source, motif_id = _split_ref(refs[0], tf_name=tf_name)
        rows.append(
            {
                "tf": tf_name,
                "motif_source": source,
                "motif_id": motif_id,
                "pwm_ref": refs[0],
                "pwm_hash": hashes[0],
                "pwm_width": pwm_width,
                "consensus_sequence": consensus,
                "consensus_width": len(consensus),
            }
        )
    consensus_df = pd.DataFrame(rows).sort_values("tf").reset_index(drop=True)
    return consensus_df, tf_names


def _build_bispecific_rows(window_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (_, elite_id, elite_sequence), group in window_df.groupby(
        ["elite_rank", "elite_id", "elite_sequence"], sort=True
    ):
        rank = int(group.iloc[0]["elite_rank"])
        ordered = group.sort_values("tf").to_dict(orient="records")
        for left, right in combinations(ordered, 2):
            start_a = int(left["best_start"])
            end_a = int(left["best_end"])
            start_b = int(right["best_start"])
            end_b = int(right["best_end"])
            overlap_bp = _interval_overlap(start_a, end_a, start_b, end_b)
            gap_bp = _interval_gap(start_a, end_a, start_b, end_b)
            span_start = min(start_a, start_b)
            span_end = max(end_a, end_b)
            score_norm_a = float(left["best_score_norm"])
            score_norm_b = float(right["best_score_norm"])
            rows.append(
                {
                    "elite_id": str(elite_id),
                    "elite_rank": rank,
                    "elite_sequence": str(elite_sequence),
                    "tf_a": str(left["tf"]),
                    "tf_b": str(right["tf"]),
                    "window_a_seq": str(left["best_window_seq"]),
                    "window_b_seq": str(right["best_window_seq"]),
                    "core_a_seq": str(left["best_core_seq"]),
                    "core_b_seq": str(right["best_core_seq"]),
                    "start_a": start_a,
                    "end_a": end_a,
                    "start_b": start_b,
                    "end_b": end_b,
                    "strand_a": str(left["best_strand"]),
                    "strand_b": str(right["best_strand"]),
                    "score_norm_a": score_norm_a,
                    "score_norm_b": score_norm_b,
                    "pair_min_score_norm": min(score_norm_a, score_norm_b),
                    "pair_mean_score_norm": (score_norm_a + score_norm_b) / 2.0,
                    "overlap_bp": int(overlap_bp),
                    "gap_bp": int(gap_bp),
                    "pair_span_start": int(span_start),
                    "pair_span_end": int(span_end),
                    "pair_span_width": int(span_end - span_start),
                }
            )
    bispecific_df = pd.DataFrame(rows, columns=_BISPECIFIC_COLUMNS)
    if not bispecific_df.empty:
        bispecific_df = bispecific_df.sort_values(["elite_rank", "elite_id", "tf_a", "tf_b"]).reset_index(drop=True)
    return bispecific_df


def _resolve_max_combo_size(tf_count: int, max_combo_size: int | None) -> int:
    if tf_count < 2:
        return 1
    if max_combo_size is None:
        return tf_count
    if max_combo_size < 2:
        raise ValueError("max_combo_size must be >= 2 when provided.")
    return min(max_combo_size, tf_count)


def _build_multispecific_rows(window_df: pd.DataFrame, *, max_combo_size: int | None) -> tuple[pd.DataFrame, int]:
    rows: list[dict[str, object]] = []
    tf_count = int(window_df["tf"].nunique()) if not window_df.empty else 0
    max_combo = _resolve_max_combo_size(tf_count, max_combo_size)
    for (_, elite_id, elite_sequence), group in window_df.groupby(
        ["elite_rank", "elite_id", "elite_sequence"], sort=True
    ):
        rank = int(group.iloc[0]["elite_rank"])
        ordered = group.sort_values("tf").to_dict(orient="records")
        for combo_size in range(3, max_combo + 1):
            for members in combinations(ordered, combo_size):
                tf_members = [str(member["tf"]) for member in members]
                starts = [int(member["best_start"]) for member in members]
                ends = [int(member["best_end"]) for member in members]
                scores_norm = [float(member["best_score_norm"]) for member in members]
                scores_scaled = [float(member["best_score_scaled"]) for member in members]
                strands = [str(member["best_strand"]) for member in members]
                member_windows = [
                    {
                        "tf": str(member["tf"]),
                        "start": int(member["best_start"]),
                        "end": int(member["best_end"]),
                        "strand": str(member["best_strand"]),
                        "window_seq": str(member["best_window_seq"]),
                        "core_seq": str(member["best_core_seq"]),
                        "score_norm": float(member["best_score_norm"]),
                        "score_scaled": float(member["best_score_scaled"]),
                        "pwm_ref": str(member["pwm_ref"]),
                        "pwm_hash": str(member["pwm_hash"]),
                    }
                    for member in members
                ]
                intervals = list(zip(starts, ends))
                span_start = min(starts)
                span_end = max(ends)
                rows.append(
                    {
                        "elite_id": str(elite_id),
                        "elite_rank": rank,
                        "elite_sequence": str(elite_sequence),
                        "combo_size": combo_size,
                        "tf_combo_key": "|".join(tf_members),
                        "tf_members_json": json.dumps(tf_members, sort_keys=False),
                        "member_windows_json": json.dumps(member_windows, sort_keys=True),
                        "combo_min_score_norm": float(min(scores_norm)),
                        "combo_mean_score_norm": float(sum(scores_norm) / len(scores_norm)),
                        "combo_min_score_scaled": float(min(scores_scaled)),
                        "combo_mean_score_scaled": float(sum(scores_scaled) / len(scores_scaled)),
                        "combo_span_start": int(span_start),
                        "combo_span_end": int(span_end),
                        "combo_span_width": int(span_end - span_start),
                        "combo_overlap_total_bp": _pairwise_overlap_total(intervals),
                        "forward_count": sum(1 for strand in strands if strand == "+"),
                        "reverse_count": sum(1 for strand in strands if strand == "-"),
                        "member_count": len(members),
                    }
                )
    multispecific_df = pd.DataFrame(rows, columns=_MULTISPECIFIC_COLUMNS)
    if not multispecific_df.empty:
        multispecific_df = multispecific_df.sort_values(
            ["elite_rank", "elite_id", "combo_size", "tf_combo_key"]
        ).reset_index(drop=True)
    return multispecific_df, max_combo


def _relative_to_run(path: Path, run_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(run_dir.resolve()))
    except ValueError as exc:
        raise ValueError(f"Export artifact path escapes run_dir: {path} (run_dir={run_dir})") from exc


def _append_export_artifacts(
    run_dir: Path,
    *,
    files: dict[str, Path],
    manifest_output: Path,
) -> None:
    run_manifest = load_manifest(run_dir)
    artifact_rows = [
        artifact_entry(
            files["monospecific_consensus_sites"],
            run_dir,
            kind="table",
            label="Sequence export: monospecific consensus sites",
            stage="export",
        ),
        artifact_entry(
            files["monospecific_elite_windows"],
            run_dir,
            kind="table",
            label="Sequence export: monospecific elite windows",
            stage="export",
        ),
        artifact_entry(
            files["bispecific_elite_windows"],
            run_dir,
            kind="table",
            label="Sequence export: bispecific elite windows",
            stage="export",
        ),
        artifact_entry(
            files["multispecific_elite_windows"],
            run_dir,
            kind="table",
            label="Sequence export: multispecific elite windows",
            stage="export",
        ),
        artifact_entry(
            manifest_output,
            run_dir,
            kind="manifest",
            label="Sequence export manifest",
            stage="export",
        ),
    ]
    append_artifacts(run_manifest, artifact_rows)
    write_manifest(run_dir, run_manifest)


def export_sequences_for_run(
    run_dir: Path,
    *,
    run_name: str,
    table_format: str = "parquet",
    max_combo_size: int | None = None,
) -> SequenceExportResult:
    normalized_format = _resolve_table_format(table_format)
    if max_combo_size is not None and max_combo_size < 2:
        raise ValueError("max_combo_size must be >= 2 when provided.")
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
    if not manifest_path(run_dir).exists():
        raise FileNotFoundError(f"Missing run_manifest.json in run directory: {run_dir}")

    elites_file = elites_path(run_dir)
    hits_file = elites_hits_path(run_dir)
    if not elites_file.exists():
        raise FileNotFoundError(f"Missing elites.parquet in run directory: {elites_file}")
    if not hits_file.exists():
        raise FileNotFoundError(f"Missing elites_hits.parquet in run directory: {hits_file}")

    elites_df = read_parquet(elites_file)
    hits_df = load_elites_hits(hits_file)
    if elites_df.empty:
        raise ValueError("elites.parquet is empty; run `cruncher sample` before export.")
    if hits_df.empty:
        raise ValueError("elites_hits.parquet is empty; run `cruncher sample` before export.")

    pwms, _ = load_pwms_from_config(run_dir)
    windows_df = _build_window_rows(elites_df, hits_df)
    consensus_df, tf_names = _build_consensus_rows(windows_df, pwms)
    bispecific_df = _build_bispecific_rows(windows_df)
    multispecific_df, resolved_max_combo_size = _build_multispecific_rows(windows_df, max_combo_size=max_combo_size)

    files = {
        "monospecific_consensus_sites": run_export_sequences_table_path(
            run_dir, table_name="monospecific_consensus_sites", fmt=normalized_format
        ),
        "monospecific_elite_windows": run_export_sequences_table_path(
            run_dir, table_name="monospecific_elite_windows", fmt=normalized_format
        ),
        "bispecific_elite_windows": run_export_sequences_table_path(
            run_dir, table_name="bispecific_elite_windows", fmt=normalized_format
        ),
        "multispecific_elite_windows": run_export_sequences_table_path(
            run_dir, table_name="multispecific_elite_windows", fmt=normalized_format
        ),
    }
    for key, path in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if key == "monospecific_consensus_sites":
            _write_table(consensus_df, path, table_format=normalized_format)
        elif key == "monospecific_elite_windows":
            _write_table(windows_df.loc[:, _WINDOW_COLUMNS], path, table_format=normalized_format)
        elif key == "bispecific_elite_windows":
            _write_table(bispecific_df.loc[:, _BISPECIFIC_COLUMNS], path, table_format=normalized_format)
        else:
            _write_table(multispecific_df.loc[:, _MULTISPECIFIC_COLUMNS], path, table_format=normalized_format)

    row_counts = {
        "monospecific_consensus_sites": int(len(consensus_df)),
        "monospecific_elite_windows": int(len(windows_df)),
        "bispecific_elite_windows": int(len(bispecific_df)),
        "multispecific_elite_windows": int(len(multispecific_df)),
    }
    manifest_output = run_export_sequences_manifest_path(run_dir)
    max_multispecific_group_size: int | None = int(resolved_max_combo_size) if resolved_max_combo_size >= 3 else None
    manifest_payload = {
        "schema_version": 2,
        "kind": "sequence_export_v2",
        "created_at": _utc_now(),
        "run_name": run_name,
        "run_dir": str(run_dir.resolve()),
        "table_format": normalized_format,
        "tf_names": tf_names,
        "specificity": {
            "monospecific": {"group_size": 1},
            "bispecific": {"group_size": 2},
            "multispecific": {
                "min_group_size": 3,
                "max_group_size": max_multispecific_group_size,
            },
        },
        "inputs": {
            "elites_path": _relative_to_run(elites_file, run_dir),
            "elites_hits_path": _relative_to_run(hits_file, run_dir),
            "config_used_path": _relative_to_run(config_used_path(run_dir), run_dir),
        },
        "files": {name: _relative_to_run(path, run_dir) for name, path in files.items()},
        "row_counts": row_counts,
    }
    atomic_write_json(manifest_output, manifest_payload, sort_keys=False)
    _append_export_artifacts(run_dir, files=files, manifest_output=manifest_output)

    return SequenceExportResult(
        run_name=run_name,
        run_dir=run_dir,
        output_dir=manifest_output.parent,
        manifest_path=manifest_output,
        files=files,
        row_counts=row_counts,
    )


def _is_exportable_sample_run(run: object) -> bool:
    status = str(getattr(run, "status", "") or "").strip().lower()
    if status in {"failed", "aborted", "running"}:
        return False
    run_dir = getattr(run, "run_dir", None)
    if not isinstance(run_dir, Path):
        return False
    return manifest_path(run_dir).exists()


def _resolve_run_names(
    cfg: CruncherConfig,
    config_path: Path,
    *,
    runs_override: list[str] | None,
    use_latest: bool,
) -> list[str]:
    if runs_override:
        return runs_override
    if not use_latest:
        raise ValueError("Either --run or --latest is required for sequence export.")
    runs = list_runs(cfg, config_path, stage="sample")
    if not runs:
        raise ValueError("No sample runs found for export. Run `cruncher sample` first.")
    for run in runs:
        if _is_exportable_sample_run(run):
            return [str(run.name)]
    raise ValueError("No completed sample runs are exportable. Re-run sampling with `cruncher sample -c <CONFIG>`.")


def run_export_sequences(
    cfg: CruncherConfig,
    config_path: Path,
    *,
    runs_override: list[str] | None = None,
    use_latest: bool = False,
    table_format: str = "parquet",
    max_combo_size: int | None = None,
) -> list[SequenceExportResult]:
    run_names = _resolve_run_names(
        cfg,
        config_path,
        runs_override=runs_override,
        use_latest=use_latest,
    )
    results: list[SequenceExportResult] = []
    for run_name in run_names:
        run = get_run(cfg, config_path, run_name)
        if str(run.stage) != "sample":
            raise ValueError(f"Run '{run_name}' is not a sample run (stage={run.stage}).")
        result = export_sequences_for_run(
            run.run_dir,
            run_name=run.name,
            table_format=table_format,
            max_combo_size=max_combo_size,
        )
        results.append(result)
    return results
