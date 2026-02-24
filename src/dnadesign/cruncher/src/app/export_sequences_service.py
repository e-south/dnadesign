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
    run_export_dir,
    run_export_sequences_manifest_path,
    run_export_sequences_table_path,
    run_export_table_path,
)
from dnadesign.cruncher.artifacts.manifest import load_manifest, write_manifest
from dnadesign.cruncher.config.schema_v3 import CruncherConfig

_ELITES_REQUIRED_COLUMNS = {
    "id",
    "rank",
    "sequence",
}

_ELITES_EXPORT_OPTIONAL_COLUMNS = [
    "norm_sum",
    "min_norm",
    "sum_norm",
    "combined_score_final",
    "combined_score_scaled",
    "combined_score_raw",
    "chain",
    "chain_1based",
    "draw_idx",
    "draw_in_phase",
    "canonical_sequence",
    "meta_type",
    "meta_source",
    "meta_date",
]

_ELITES_EXPORT_BASE_COLUMNS = [
    "elite_id",
    "elite_rank",
    "elite_sequence",
    "sequence_length",
    "window_count",
    "regulator_ids_csv",
    "window_members_json",
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


def _build_elites_export_rows(elites_df: pd.DataFrame, window_df: pd.DataFrame) -> pd.DataFrame:
    _required_columns(elites_df, _ELITES_REQUIRED_COLUMNS, context="elites.parquet")
    export_df = elites_df.copy()
    if "combined_score_final" not in export_df.columns:
        raise ValueError("elites.parquet missing required column: combined_score_final")
    export_df["id"] = export_df["id"].astype(str)
    export_df["sequence"] = export_df["sequence"].astype(str)
    export_df["rank"] = pd.to_numeric(export_df["rank"], errors="coerce")
    if export_df["rank"].isna().any():
        raise ValueError("elites.parquet contains non-numeric rank values.")
    export_df["rank"] = export_df["rank"].astype(int)
    export_df["combined_score_final"] = pd.to_numeric(export_df["combined_score_final"], errors="coerce")
    if export_df["combined_score_final"].isna().any():
        raise ValueError("elites.parquet contains non-numeric combined_score_final values.")
    inf_mask = (export_df["combined_score_final"] == float("inf")) | (
        export_df["combined_score_final"] == float("-inf")
    )
    if inf_mask.any():
        raise ValueError("elites.parquet contains non-finite combined_score_final values.")
    export_df = export_df.rename(
        columns={
            "id": "elite_id",
            "rank": "elite_rank",
            "sequence": "elite_sequence",
        }
    )
    export_df["sequence_length"] = export_df["elite_sequence"].astype(str).str.len().astype(int)

    window_key_map: dict[tuple[str, int, str], tuple[str, str, int]] = {}
    for (elite_id, elite_rank, elite_sequence), group in window_df.groupby(
        ["elite_id", "elite_rank", "elite_sequence"], sort=True
    ):
        members: list[dict[str, object]] = []
        for item in group.sort_values(["tf", "best_start", "best_end"]).to_dict(orient="records"):
            members.append(
                {
                    "regulator_id": str(item["tf"]),
                    "offset_start": int(item["best_start"]),
                    "offset_end": int(item["best_end"]),
                    "strand": str(item["best_strand"]),
                    "window_kmer": str(item["best_window_seq"]),
                    "core_kmer": str(item["best_core_seq"]),
                    "score_name": "best_score_norm",
                    "score": float(item["best_score_norm"]),
                    "score_scaled_name": "best_score_scaled",
                    "score_scaled": float(item["best_score_scaled"]),
                    "score_raw_name": "best_score_raw",
                    "score_raw": float(item["best_score_raw"]),
                    "pwm_ref": str(item["pwm_ref"]),
                    "pwm_hash": str(item["pwm_hash"]),
                    "pwm_width": int(item["pwm_width"]),
                    "core_width": int(item["core_width"]),
                }
            )
        regulators = sorted({str(member["regulator_id"]) for member in members})
        window_key_map[(str(elite_id), int(elite_rank), str(elite_sequence))] = (
            json.dumps(members, sort_keys=True),
            ",".join(regulators),
            int(len(members)),
        )

    members_col: list[str] = []
    regulators_col: list[str] = []
    count_col: list[int] = []
    for row in export_df.to_dict(orient="records"):
        key = (str(row["elite_id"]), int(row["elite_rank"]), str(row["elite_sequence"]))
        payload = window_key_map.get(key)
        if payload is None:
            raise ValueError(
                "Missing elite window metadata for export row: "
                f"elite_id={row['elite_id']!r} elite_rank={row['elite_rank']!r}"
            )
        members_col.append(payload[0])
        regulators_col.append(payload[1])
        count_col.append(payload[2])
    export_df["window_members_json"] = members_col
    export_df["regulator_ids_csv"] = regulators_col
    export_df["window_count"] = count_col

    optional_columns = [column for column in _ELITES_EXPORT_OPTIONAL_COLUMNS if column in export_df.columns]
    score_columns = sorted(column for column in export_df.columns if column.startswith(("score_", "norm_")))
    selected = list(_ELITES_EXPORT_BASE_COLUMNS)
    for column in optional_columns + score_columns:
        if column not in selected:
            selected.append(column)
    return export_df.loc[:, selected].sort_values(["elite_rank", "elite_id"]).reset_index(drop=True)


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


def _relative_to_run(path: Path, run_dir: Path) -> str:
    try:
        return str(path.resolve().relative_to(run_dir.resolve()))
    except ValueError as exc:
        raise ValueError(f"Export artifact path escapes run_dir: {path} (run_dir={run_dir})") from exc


def _cleanup_export_tables(export_dir: Path, *, keep_files: set[str]) -> None:
    if not export_dir.exists():
        return
    for path in export_dir.glob("table__*.*"):
        if path.name in keep_files:
            continue
        if path.suffix.lower() not in {".csv", ".parquet"}:
            continue
        path.unlink()


def _append_export_artifacts(
    run_dir: Path,
    *,
    files: dict[str, Path],
    manifest_output: Path,
) -> None:
    run_manifest = load_manifest(run_dir)
    artifact_rows = [
        artifact_entry(
            files["consensus_sites"],
            run_dir,
            kind="table",
            label="Sequence export: consensus sites",
            stage="export",
        ),
        artifact_entry(
            files["elites"],
            run_dir,
            kind="table",
            label="Sequence export: elites table",
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
    table_format: str = "csv",
) -> SequenceExportResult:
    normalized_format = _resolve_table_format(table_format)
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
    elites_export_df = _build_elites_export_rows(elites_df, windows_df)
    consensus_df, tf_names = _build_consensus_rows(windows_df, pwms)

    files = {
        "consensus_sites": run_export_sequences_table_path(
            run_dir, table_name="consensus_sites", fmt=normalized_format
        ),
        "elites": run_export_table_path(run_dir, table_name="elites", fmt="csv"),
    }
    _cleanup_export_tables(run_export_dir(run_dir), keep_files={path.name for path in files.values()})
    for key, path in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if key == "consensus_sites":
            _write_table(consensus_df, path, table_format=normalized_format)
        else:
            _write_table(elites_export_df, path, table_format="csv")

    row_counts = {
        "consensus_sites": int(len(consensus_df)),
        "elites": int(len(elites_export_df)),
    }
    manifest_output = run_export_sequences_manifest_path(run_dir)
    manifest_payload = {
        "schema_version": 3,
        "kind": "sequence_export_v3",
        "created_at": _utc_now(),
        "run_name": run_name,
        "run_dir": str(run_dir.resolve()),
        "table_format": normalized_format,
        "tf_names": tf_names,
        "export_tables": ["consensus_sites", "elites"],
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
        output_dir=run_export_dir(run_dir),
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
    table_format: str = "csv",
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
        )
        results.append(result)
    return results
