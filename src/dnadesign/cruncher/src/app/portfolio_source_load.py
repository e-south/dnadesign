"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/cruncher/src/app/portfolio_source_load.py

Helpers for loading one Portfolio source run into typed aggregation rows.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import pandas as pd

from dnadesign.cruncher.analysis.layout import analysis_root, summary_path
from dnadesign.cruncher.analysis.parquet import read_parquet
from dnadesign.cruncher.app.portfolio_preflight import _resolve_source_label
from dnadesign.cruncher.app.portfolio_studies import _load_source_study_summary
from dnadesign.cruncher.artifacts.layout import run_export_sequences_manifest_path
from dnadesign.cruncher.artifacts.manifest import load_manifest
from dnadesign.cruncher.portfolio.manifest import PortfolioSourceRun
from dnadesign.cruncher.portfolio.schema_models import PortfolioSource

PortfolioSourceLoadEventCallback = Callable[[str, dict[str, object]], None]
RunStudyFn = Callable[..., Path]


def _ensure_required_columns(df: pd.DataFrame, required: list[str], *, context: str) -> None:
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")


def _stable_hash16(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _load_analysis_summary(run_dir: Path) -> dict[str, object]:
    path = summary_path(analysis_root(run_dir))
    if not path.exists():
        raise FileNotFoundError(
            f"Missing analysis summary for portfolio source run: {path}. "
            "Run `cruncher analyze --summary --run <run>` first."
        )
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Analysis summary must be a JSON object: {path}")
    return payload


def _load_export_elites_windows_and_consensus(
    source: PortfolioSource,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    manifest_file = run_export_sequences_manifest_path(source.run_dir)
    if not manifest_file.exists():
        raise FileNotFoundError(
            f"Missing export_manifest.json for portfolio source run: {manifest_file}. "
            "Run `cruncher export sequences --run <run>` first."
        )
    payload = json.loads(manifest_file.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"Export manifest must be a JSON object: {manifest_file}")
    files = payload.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"Export manifest missing files mapping: {manifest_file}")
    elites_rel = files.get("elites")
    if not isinstance(elites_rel, str) or not elites_rel.strip():
        raise ValueError(f"Export manifest missing elites file: {manifest_file}")
    consensus_rel = files.get("consensus_sites")
    if not isinstance(consensus_rel, str) or not consensus_rel.strip():
        raise ValueError(f"Export manifest missing consensus_sites file: {manifest_file}")

    table_path = (source.run_dir / elites_rel).resolve()
    try:
        table_path.relative_to(source.run_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"Export table path escapes run directory: {table_path}") from exc
    if not table_path.exists():
        raise FileNotFoundError(f"Export table listed in manifest does not exist: {table_path}")
    consensus_path = (source.run_dir / consensus_rel).resolve()
    try:
        consensus_path.relative_to(source.run_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"Export table path escapes run directory: {consensus_path}") from exc
    if not consensus_path.exists():
        raise FileNotFoundError(f"Export table listed in manifest does not exist: {consensus_path}")

    if table_path.suffix == ".parquet":
        export_df = read_parquet(table_path)
    elif table_path.suffix == ".csv":
        export_df = pd.read_csv(table_path)
    else:
        raise ValueError(f"Unsupported elites table format: {table_path.suffix}")
    if consensus_path.suffix == ".parquet":
        consensus_df = read_parquet(consensus_path)
    elif consensus_path.suffix == ".csv":
        consensus_df = pd.read_csv(consensus_path)
    else:
        raise ValueError(f"Unsupported consensus table format: {consensus_path.suffix}")

    required_export_columns = [
        "elite_id",
        "elite_rank",
        "elite_sequence",
        "window_members_json",
        "combined_score_final",
    ]
    missing_export_columns = [name for name in required_export_columns if name not in export_df.columns]
    if missing_export_columns:
        raise ValueError(
            f"Portfolio source elites table ({table_path}) missing required columns: {missing_export_columns}. "
            "nudge: rerun `cruncher workspaces run --workspace <source_workspace> --runbook configs/runbook.yaml "
            "--step export_sequences_latest` for this source and retry the portfolio run."
        )
    export_df = export_df.copy()
    export_df["elite_id"] = export_df["elite_id"].astype(str)
    export_df["elite_sequence"] = export_df["elite_sequence"].astype(str)
    export_df["elite_rank"] = pd.to_numeric(export_df["elite_rank"], errors="coerce")
    if export_df["elite_rank"].isna().any():
        raise ValueError(f"Portfolio source elites table elite_rank contains non-numeric values: {table_path}")
    export_df["elite_rank"] = export_df["elite_rank"].astype(int)
    export_df["combined_score_final"] = pd.to_numeric(export_df["combined_score_final"], errors="coerce")
    if export_df["combined_score_final"].isna().any():
        raise ValueError(
            f"Portfolio source elites table combined_score_final contains non-numeric values: {table_path}"
        )
    if export_df["elite_id"].duplicated().any():
        raise ValueError(f"Portfolio source elites table contains duplicate elite_id values: {table_path}")
    if export_df["elite_rank"].duplicated().any():
        raise ValueError(f"Portfolio source elites table contains duplicate elite_rank values: {table_path}")

    elites_export_df = export_df.loc[:, ["elite_id", "elite_rank", "elite_sequence", "combined_score_final"]].copy()
    elites_export_df = elites_export_df.sort_values(["elite_rank", "elite_id"]).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for elite_row in export_df.to_dict(orient="records"):
        elite_id = str(elite_row["elite_id"])
        elite_rank = int(elite_row["elite_rank"])
        elite_sequence = str(elite_row["elite_sequence"])
        members_raw = elite_row.get("window_members_json")
        if not isinstance(members_raw, str) or not members_raw.strip():
            raise ValueError(
                f"Portfolio source elites table contains empty window_members_json: {table_path} elite_id={elite_id!r}"
            )
        try:
            members = json.loads(members_raw)
        except Exception as exc:
            raise ValueError(
                "Portfolio source elites table window_members_json must be valid JSON: "
                f"{table_path} elite_id={elite_id!r}"
            ) from exc
        if not isinstance(members, list) or not members:
            raise ValueError(
                "Portfolio source elites table window_members_json must be a non-empty list: "
                f"{table_path} elite_id={elite_id!r}"
            )
        for member in members:
            if not isinstance(member, dict):
                raise ValueError(
                    "Portfolio source elites table window_members_json items must be objects: "
                    f"{table_path} elite_id={elite_id!r}"
                )
            tf_name = member.get("regulator_id")
            best_start = member.get("offset_start")
            best_end = member.get("offset_end")
            best_strand = member.get("strand")
            best_window_seq = member.get("window_kmer")
            best_core_seq = member.get("core_kmer")
            best_score_norm = member.get("score")
            score_name = member.get("score_name")
            if score_name != "best_score_norm":
                raise ValueError(
                    "Portfolio source elites table window_members_json score_name must be 'best_score_norm': "
                    f"{table_path} elite_id={elite_id!r} score_name={score_name!r}"
                )
            rows.append(
                {
                    "elite_id": elite_id,
                    "elite_rank": elite_rank,
                    "elite_sequence": elite_sequence,
                    "tf": str(tf_name),
                    "best_start": int(best_start),
                    "best_end": int(best_end),
                    "best_strand": str(best_strand),
                    "best_window_seq": str(best_window_seq),
                    "best_core_seq": str(best_core_seq),
                    "best_score_norm": float(best_score_norm),
                }
            )
    window_df = pd.DataFrame(rows)
    _ensure_required_columns(
        window_df,
        [
            "elite_id",
            "elite_rank",
            "elite_sequence",
            "tf",
            "best_start",
            "best_end",
            "best_strand",
            "best_window_seq",
            "best_core_seq",
            "best_score_norm",
        ],
        context=f"Portfolio source window records ({table_path})",
    )
    if window_df.empty:
        raise ValueError(f"Portfolio source window records are empty after parsing: {table_path}")
    required_consensus_columns = [
        "tf",
        "motif_source",
        "motif_id",
        "pwm_ref",
        "pwm_hash",
        "pwm_width",
        "consensus_sequence",
        "consensus_width",
    ]
    missing_consensus_columns = [name for name in required_consensus_columns if name not in consensus_df.columns]
    if missing_consensus_columns:
        raise ValueError(
            f"Portfolio source consensus table ({consensus_path}) missing required columns: "
            f"{missing_consensus_columns}. nudge: rerun `cruncher workspaces run --workspace <source_workspace> "
            "--runbook configs/runbook.yaml --step export_sequences_latest` for this source and retry the "
            "portfolio run."
        )
    consensus_df = consensus_df.loc[:, required_consensus_columns].copy()
    if consensus_df.empty:
        raise ValueError(f"Portfolio source consensus table is empty: {consensus_path}")
    consensus_df["tf"] = consensus_df["tf"].astype(str)
    consensus_df["motif_source"] = consensus_df["motif_source"].astype(str)
    consensus_df["motif_id"] = consensus_df["motif_id"].astype(str)
    consensus_df["pwm_ref"] = consensus_df["pwm_ref"].astype(str)
    consensus_df["pwm_hash"] = consensus_df["pwm_hash"].astype(str)
    consensus_df["consensus_sequence"] = consensus_df["consensus_sequence"].astype(str)
    consensus_df["pwm_width"] = pd.to_numeric(consensus_df["pwm_width"], errors="coerce")
    consensus_df["consensus_width"] = pd.to_numeric(consensus_df["consensus_width"], errors="coerce")
    if consensus_df["pwm_width"].isna().any() or consensus_df["consensus_width"].isna().any():
        raise ValueError(f"Portfolio source consensus table contains non-numeric widths: {consensus_path}")
    consensus_df["pwm_width"] = consensus_df["pwm_width"].astype(int)
    consensus_df["consensus_width"] = consensus_df["consensus_width"].astype(int)
    if consensus_df["tf"].duplicated().any():
        raise ValueError(f"Portfolio source consensus table contains duplicate tf rows: {consensus_path}")
    consensus_df = consensus_df.sort_values("tf").reset_index(drop=True)
    return elites_export_df, window_df, consensus_df


def _mean_pairwise_hamming_bp(sequences: list[str]) -> float | None:
    if len(sequences) < 2:
        return None
    total = 0
    pairs = 0
    for idx, left in enumerate(sequences):
        for jdx in range(idx + 1, len(sequences)):
            right = sequences[jdx]
            mismatch = sum(1 for a, b in zip(left, right, strict=False) if a != b)
            total += mismatch + abs(len(left) - len(right))
            pairs += 1
    if pairs == 0:
        return None
    return float(total / pairs)


def _load_source_rows_with_study_runner(
    source: PortfolioSource,
    *,
    studies_enabled: bool,
    run_study_fn: RunStudyFn,
    on_event: PortfolioSourceLoadEventCallback | None = None,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    dict[str, object],
    PortfolioSourceRun,
    dict[str, object] | None,
]:
    if not source.run_dir.exists():
        raise FileNotFoundError(
            f"Portfolio source run_dir does not exist after preparation: id={source.id!r} run_dir={source.run_dir}"
        )
    if not source.run_dir.is_dir():
        raise ValueError(f"Portfolio source run_dir must be a directory: {source.run_dir}")

    run_manifest = load_manifest(source.run_dir)
    run_stage = str(run_manifest.get("stage") or "").strip().lower()
    if run_stage != "sample":
        raise ValueError(
            f"Portfolio source run must be a sample run: id={source.id!r} run_dir={source.run_dir} stage={run_stage!r}"
        )
    top_k_raw = run_manifest.get("top_k")
    if not isinstance(top_k_raw, int) or top_k_raw < 1:
        raise ValueError(
            "Manifest top_k must be a positive integer for portfolio aggregation: "
            f"id={source.id!r} run_dir={source.run_dir}"
        )
    source_top_k = int(top_k_raw)

    summary_payload = _load_analysis_summary(source.run_dir)

    export_elites_df, windows_df, consensus_df = _load_export_elites_windows_and_consensus(source)
    export_elites_rows = int(len(export_elites_df))
    if export_elites_rows > source_top_k:
        raise ValueError(
            "Manifest top_k must be >= export elites row count for portfolio source: "
            f"id={source.id!r} top_k={source_top_k} export_elites_rows={export_elites_rows}"
        )
    selected = export_elites_df.nsmallest(source_top_k, "elite_rank").copy()
    if selected.empty:
        raise ValueError(f"Portfolio source selected zero export elites for top_k={source_top_k}.")
    selected_top_k = int(len(selected))
    expected_ranks = list(range(1, selected_top_k + 1))
    selected_ranks = sorted(selected["elite_rank"].astype(int).tolist())
    if selected_ranks != expected_ranks:
        raise ValueError(
            "Portfolio source export elites rank contract violated: "
            f"id={source.id!r} expected={expected_ranks[:5]}..{expected_ranks[-1]} actual={selected_ranks[:5]}"
        )

    windows_df = windows_df.copy()
    windows_df["elite_id"] = windows_df["elite_id"].astype(str)
    windows_df["elite_rank"] = pd.to_numeric(windows_df["elite_rank"], errors="coerce")
    if windows_df["elite_rank"].isna().any():
        raise ValueError("Portfolio source windows table contains non-numeric elite_rank values.")
    windows_df["elite_rank"] = windows_df["elite_rank"].astype(int)
    windows_df["best_score_norm"] = pd.to_numeric(windows_df["best_score_norm"], errors="coerce")
    if windows_df["best_score_norm"].isna().any():
        raise ValueError("Portfolio source windows table contains non-numeric best_score_norm values.")

    selected_ids = set(selected["elite_id"].astype(str).tolist())
    selected_windows = windows_df[windows_df["elite_id"].isin(selected_ids)].copy()
    key_counts = selected_windows.groupby(["elite_id", "tf"]).size()
    duplicate = key_counts[key_counts > 1]
    if not duplicate.empty:
        labels = [f"({elite_id},{tf})x{int(count)}" for (elite_id, tf), count in duplicate.items()]
        raise ValueError(
            "Portfolio source windows table has duplicate elite/tf rows for selected elites: " + ", ".join(labels)
        )
    expected_tf_names = sorted(set(selected_windows["tf"].astype(str).tolist()))
    if not expected_tf_names:
        raise ValueError(f"Portfolio source windows table has no selected TF rows: id={source.id!r}")
    summary_tf_names = summary_payload.get("tf_names")
    if isinstance(summary_tf_names, list):
        summary_tf_set = sorted(str(item) for item in summary_tf_names)
        if summary_tf_set != expected_tf_names:
            raise ValueError(
                "Portfolio source TF mismatch between analysis summary and export windows: "
                f"id={source.id!r} summary_tf={summary_tf_set} windows_tf={expected_tf_names}"
            )

    source_windows_rows: list[dict[str, object]] = []
    source_elite_rows: list[dict[str, object]] = []
    source_consensus_rows: list[dict[str, object]] = []
    min_score_values: list[float] = []
    sequence_values: list[str] = []
    seen_elite_hashes: set[str] = set()
    seen_window_hashes: set[str] = set()

    for row in selected.sort_values("elite_rank").to_dict(orient="records"):
        elite_id = str(row["elite_id"])
        elite_windows = selected_windows[selected_windows["elite_id"] == elite_id].copy()
        if elite_windows.empty:
            raise ValueError(f"Portfolio source windows table has no rows for selected elite_id={elite_id!r}.")

        elite_windows = elite_windows.sort_values(["tf", "best_start", "best_end"]).reset_index(drop=True)
        score_values = elite_windows["best_score_norm"].astype(float).tolist()
        min_score = float(min(score_values))
        mean_score = float(sum(score_values) / len(score_values))
        tf_names = sorted({str(tf) for tf in elite_windows["tf"].astype(str).tolist()})
        if tf_names != expected_tf_names:
            raise ValueError(
                "Portfolio source selected elite does not cover the expected TF set: "
                f"id={source.id!r} elite_id={elite_id!r} tf={tf_names} expected={expected_tf_names}"
            )
        sequence = str(row["elite_sequence"])
        elite_rank = int(row["elite_rank"])
        elite_hash_id = _stable_hash16(source.id, source.run_dir, elite_id, elite_rank, sequence)
        if elite_hash_id in seen_elite_hashes:
            raise ValueError(f"Portfolio elite hash collision detected: id={source.id!r} hash={elite_hash_id}")
        seen_elite_hashes.add(elite_hash_id)
        tf_names_csv = ",".join(tf_names)
        sequence_length = int(len(sequence))

        source_elite_rows.append(
            {
                "source_id": str(source.id),
                "source_label": _resolve_source_label(source),
                "workspace_name": source.workspace.name,
                "workspace_path": str(source.workspace),
                "run_name": source.run_dir.name,
                "run_dir": str(source.run_dir),
                "source_top_k": source_top_k,
                "elite_hash_id": elite_hash_id,
                "elite_id": elite_id,
                "elite_rank": elite_rank,
                "sequence": sequence,
                "sequence_length": sequence_length,
                "combined_score_final": float(row["combined_score_final"]),
                "min_best_score_norm": min_score,
                "mean_best_score_norm": mean_score,
                "tf_count": int(len(tf_names)),
                "tf_names_csv": tf_names_csv,
            }
        )

        for window in elite_windows.to_dict(orient="records"):
            tf_name = str(window["tf"])
            best_start = int(window["best_start"])
            best_end = int(window["best_end"])
            best_strand = str(window["best_strand"])
            best_window_seq = str(window["best_window_seq"])
            best_core_seq = str(window["best_core_seq"])
            best_score_norm = float(window["best_score_norm"])
            window_hash_id = _stable_hash16(
                elite_hash_id,
                tf_name,
                best_start,
                best_end,
                best_strand,
                best_window_seq,
                best_core_seq,
            )
            if window_hash_id in seen_window_hashes:
                raise ValueError(f"Portfolio window hash collision detected: id={source.id!r} hash={window_hash_id}")
            seen_window_hashes.add(window_hash_id)
            source_windows_rows.append(
                {
                    "source_id": str(source.id),
                    "source_label": _resolve_source_label(source),
                    "workspace_name": source.workspace.name,
                    "workspace_path": str(source.workspace),
                    "run_name": source.run_dir.name,
                    "run_dir": str(source.run_dir),
                    "source_top_k": source_top_k,
                    "elite_hash_id": elite_hash_id,
                    "elite_id": elite_id,
                    "elite_rank": elite_rank,
                    "sequence": sequence,
                    "sequence_length": sequence_length,
                    "combined_score_final": float(row["combined_score_final"]),
                    "min_best_score_norm": min_score,
                    "mean_best_score_norm": mean_score,
                    "tf_count": int(len(tf_names)),
                    "tf_names_csv": tf_names_csv,
                    "window_hash_id": window_hash_id,
                    "tf": tf_name,
                    "best_start": best_start,
                    "best_end": best_end,
                    "best_strand": best_strand,
                    "best_window_seq": best_window_seq,
                    "best_core_seq": best_core_seq,
                    "best_score_norm": best_score_norm,
                }
            )
        min_score_values.append(min_score)
        sequence_values.append(sequence)

    source_run = PortfolioSourceRun(
        source_id=str(source.id),
        source_label=_resolve_source_label(source),
        workspace_name=source.workspace.name,
        workspace_path=str(source.workspace),
        run_dir=str(source.run_dir),
        run_name=source.run_dir.name,
        source_top_k=source_top_k,
        selected_elites=len(source_elite_rows),
    )

    source_summary_row = {
        "source_id": str(source.id),
        "source_label": _resolve_source_label(source),
        "workspace_name": source.workspace.name,
        "workspace_path": str(source.workspace),
        "run_name": source.run_dir.name,
        "run_dir": str(source.run_dir),
        "source_top_k": source_top_k,
        "n_selected_elites": int(len(source_elite_rows)),
        "selected_rank_max": int(max(item["elite_rank"] for item in source_elite_rows)),
        "mean_min_best_score_norm": float(sum(min_score_values) / len(min_score_values)),
        "median_min_best_score_norm": float(pd.Series(min_score_values).median()),
        "mean_pairwise_hamming_bp": _mean_pairwise_hamming_bp(sequence_values),
        "analysis_id": summary_payload.get("analysis_id"),
        "analysis_best_score_final": (summary_payload.get("objective_components") or {}).get("best_score_final"),
    }
    study_summary_row = _load_source_study_summary(
        source,
        studies_enabled=studies_enabled,
        run_study_fn=run_study_fn,
        on_event=on_event,
    )
    for consensus in consensus_df.to_dict(orient="records"):
        source_consensus_rows.append(
            {
                "source_id": str(source.id),
                "source_label": _resolve_source_label(source),
                "workspace_name": source.workspace.name,
                "workspace_path": str(source.workspace),
                "run_name": source.run_dir.name,
                "run_dir": str(source.run_dir),
                "tf": str(consensus["tf"]),
                "motif_source": str(consensus["motif_source"]),
                "motif_id": str(consensus["motif_id"]),
                "pwm_ref": str(consensus["pwm_ref"]),
                "pwm_hash": str(consensus["pwm_hash"]),
                "pwm_width": int(consensus["pwm_width"]),
                "consensus_sequence": str(consensus["consensus_sequence"]),
                "consensus_width": int(consensus["consensus_width"]),
            }
        )
    return (
        source_windows_rows,
        source_elite_rows,
        source_consensus_rows,
        source_summary_row,
        source_run,
        study_summary_row,
    )
