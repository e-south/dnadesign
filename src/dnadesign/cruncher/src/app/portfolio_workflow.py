"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/src/app/portfolio_workflow.py

Orchestrate cross-workspace Portfolio aggregation outputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
import shutil
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Literal

import pandas as pd

from dnadesign.cruncher.analysis.layout import analysis_root, summary_path
from dnadesign.cruncher.analysis.parquet import read_parquet
from dnadesign.cruncher.app.portfolio_materialization import (
    _materialize_portfolio_outputs as _materialize_portfolio_outputs_helper,
)
from dnadesign.cruncher.app.portfolio_materialization import (
    _select_portfolio_showcase_elites as _select_portfolio_showcase_elites_helper,
)
from dnadesign.cruncher.app.portfolio_materialization import (
    _write_tradeoff_plot as _write_tradeoff_plot_helper,
)
from dnadesign.cruncher.app.portfolio_preflight import (
    _collect_source_readiness,
    _raise_aggregate_only_preflight,
    _render_prepare_runbook_command,
    _render_prepare_runbook_path,
    _requires_full_runbook_prepare,
    _resolve_source_label,
)
from dnadesign.cruncher.app.portfolio_preflight import (
    _preflight_source_readiness as _preflight_source_readiness_helper,
)
from dnadesign.cruncher.app.portfolio_studies import (
    _ensure_required_source_studies,
    _ensure_required_source_studies_for_sources,
    _load_source_sequence_length_rows,
    _load_source_study_summary,
)
from dnadesign.cruncher.artifacts.layout import (
    run_export_sequences_manifest_path,
)
from dnadesign.cruncher.artifacts.manifest import load_manifest
from dnadesign.cruncher.portfolio.layout import (
    portfolio_logs_dir,
    portfolio_manifest_path,
    portfolio_meta_dir,
    portfolio_plot_glob,
    portfolio_plots_dir,
    portfolio_status_path,
    portfolio_tables_dir,
    resolve_portfolio_run_dir,
)
from dnadesign.cruncher.portfolio.layout import (
    portfolio_plot_path as _portfolio_plot_path,
)
from dnadesign.cruncher.portfolio.load import load_portfolio_spec
from dnadesign.cruncher.portfolio.manifest import (
    PortfolioManifestV1,
    PortfolioPreparedSource,
    PortfolioSourceRun,
    PortfolioStatusV1,
    load_portfolio_manifest,
    load_portfolio_status,
    utc_now_iso,
    write_portfolio_manifest,
    write_portfolio_status,
)
from dnadesign.cruncher.portfolio.schema_models import PortfolioSource, PortfolioSpec
from dnadesign.cruncher.utils.hashing import sha256_path
from dnadesign.cruncher.viz.mpl import ensure_mpl_cache
from dnadesign.cruncher.workspaces.runbook import run_workspace_runbook

PrepareReadyPolicy = Literal["rerun", "skip"]
PortfolioEventCallback = Callable[[str, dict[str, object]], None]
_preflight_source_readiness = _preflight_source_readiness_helper


def _portfolio_id(spec: PortfolioSpec) -> str:
    payload = json.dumps(spec.model_dump(mode="json"), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return digest[:12]


def run_study(*args, **kwargs):
    from dnadesign.cruncher.app.study_workflow import run_study as _run_study

    return _run_study(*args, **kwargs)


def _ensure_required_columns(df: pd.DataFrame, required: list[str], *, context: str) -> None:
    missing = [name for name in required if name not in df.columns]
    if missing:
        raise ValueError(f"{context} missing required columns: {missing}")


def _stable_hash16(*parts: object) -> str:
    payload = "|".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _emit_event(on_event: PortfolioEventCallback | None, name: str, **payload: object) -> None:
    if on_event is None:
        return
    on_event(name, dict(payload))


def _remove_finder_metadata(root: Path) -> None:
    if not root.exists():
        return
    for path in root.rglob(".DS_Store"):
        if path.is_file():
            path.unlink()


def portfolio_preflight_payload(spec_path: Path) -> dict[str, object]:
    resolved_spec = spec_path.expanduser().resolve()
    spec = load_portfolio_spec(resolved_spec)
    readiness = _collect_source_readiness(spec)
    ready_ids = [source_id for source_id, record in readiness.items() if bool(record.get("ready"))]
    unready_ids = [source_id for source_id, record in readiness.items() if not bool(record.get("ready"))]
    return {
        "spec_path": str(resolved_spec),
        "execution_mode": spec.execution.mode,
        "source_count": len(spec.sources),
        "ready_source_ids": ready_ids,
        "unready_source_ids": unready_ids,
        "sources": list(readiness.values()),
    }


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


def _load_export_elites_and_windows(source: PortfolioSource) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    rel = files.get("elites")
    if not isinstance(rel, str) or not rel.strip():
        raise ValueError(f"Export manifest missing elites file: {manifest_file}")

    table_path = (source.run_dir / rel).resolve()
    try:
        table_path.relative_to(source.run_dir.resolve())
    except ValueError as exc:
        raise ValueError(f"Export table path escapes run directory: {table_path}") from exc
    if not table_path.exists():
        raise FileNotFoundError(f"Export table listed in manifest does not exist: {table_path}")

    if table_path.suffix == ".parquet":
        export_df = read_parquet(table_path)
    elif table_path.suffix == ".csv":
        export_df = pd.read_csv(table_path)
    else:
        raise ValueError(f"Unsupported elites table format: {table_path.suffix}")

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
    return elites_export_df, window_df


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


def _load_source_rows(
    source: PortfolioSource,
    *,
    on_event: PortfolioEventCallback | None = None,
) -> tuple[
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

    export_elites_df, windows_df = _load_export_elites_and_windows(source)
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
    study_summary_row = _load_source_study_summary(source, run_study_fn=run_study, on_event=on_event)
    return source_windows_rows, source_elite_rows, source_summary_row, source_run, study_summary_row


_materialize_portfolio_outputs = _materialize_portfolio_outputs_helper
_select_portfolio_showcase_elites = _select_portfolio_showcase_elites_helper
_write_tradeoff_plot = _write_tradeoff_plot_helper
portfolio_plot_path = _portfolio_plot_path


def _prepare_source_log_path(run_dir: Path, source_id: str) -> Path:
    return portfolio_logs_dir(run_dir) / f"prepare__{source_id}.log"


def _prepare_source(
    source: PortfolioSource,
    *,
    readiness: dict[str, dict[str, object]],
    prepare_ready_policy: PrepareReadyPolicy,
    prepare_log_path: Path | None = None,
) -> PortfolioPreparedSource:
    if source.prepare is None:
        raise ValueError(
            "portfolio.execution.mode=prepare_then_aggregate requires prepare for every source: "
            f"missing source={source.id!r}"
        )
    source_id = str(source.id)
    source_readiness = readiness.get(source_id)
    is_ready = bool(source_readiness and source_readiness.get("ready"))
    if is_ready and prepare_ready_policy == "skip":
        return PortfolioPreparedSource(
            source_id=source_id,
            runbook_path=str(source.prepare.runbook),
            step_ids=[],
        )
    try:
        result = run_workspace_runbook(
            source.prepare.runbook,
            step_ids=source.prepare.step_ids,
            dry_run=False,
            output_log_path=prepare_log_path,
        )
    except RuntimeError as exc:
        nudge_cmd = _render_prepare_runbook_command(source, include_steps=True)
        lines = [
            "Portfolio source preparation failed.",
            f"source={source_id} workspace={source.workspace.name}",
            f"runbook={_render_prepare_runbook_path(source)}",
            f"step_ids={list(source.prepare.step_ids)}",
        ]
        readiness_issues = list(source_readiness.get("issues", [])) if source_readiness else []
        if readiness_issues:
            lines.append("preflight issues:")
            for issue in readiness_issues:
                lines.append(f"  - {issue}")
        lines.append(
            "nudge: include all steps needed for source readiness "
            "(usually sample_run, analyze_summary, export_sequences_latest)."
        )
        lines.append(f"nudge: {nudge_cmd}")
        if _requires_full_runbook_prepare(readiness_issues) and source.prepare.step_ids:
            lines.append(
                f"nudge: full runbook required: {_render_prepare_runbook_command(source, include_steps=False)}"
            )
        if prepare_log_path is not None:
            lines.append(f"log: {prepare_log_path}")
        lines.append(f"cause: {exc}")
        raise ValueError("\n".join(lines)) from exc
    return PortfolioPreparedSource(
        source_id=source_id,
        runbook_path=str(result.runbook_path),
        step_ids=list(result.executed_step_ids),
    )


def run_portfolio(
    spec_path: Path,
    *,
    force_overwrite: bool = False,
    prepare_ready_policy: PrepareReadyPolicy = "rerun",
    on_event: PortfolioEventCallback | None = None,
) -> Path:
    if prepare_ready_policy not in {"rerun", "skip"}:
        raise ValueError(f"Invalid prepare_ready_policy: {prepare_ready_policy!r}")
    resolved_spec = spec_path.expanduser().resolve()
    _emit_event(on_event, "portfolio_started", spec_path=str(resolved_spec))
    spec = load_portfolio_spec(resolved_spec)
    readiness = _collect_source_readiness(spec)
    _emit_event(
        on_event,
        "preflight_completed",
        ready_source_ids=[key for key, value in readiness.items() if bool(value.get("ready"))],
        unready_source_ids=[key for key, value in readiness.items() if not bool(value.get("ready"))],
    )
    if spec.execution.mode == "aggregate_only":
        _raise_aggregate_only_preflight(spec, readiness)

    workspace_root = resolved_spec.parent
    if resolved_spec.parent.name == "configs":
        workspace_root = resolved_spec.parent.parent
    portfolio_id = _portfolio_id(spec)
    run_dir = resolve_portfolio_run_dir(workspace_root, spec.name, portfolio_id)
    _remove_finder_metadata(workspace_root / "outputs")

    if run_dir.exists():
        if force_overwrite:
            shutil.rmtree(run_dir)
        else:
            raise ValueError(f"Portfolio run directory already exists: {run_dir}. Use --force-overwrite.")

    portfolio_meta_dir(run_dir).mkdir(parents=True, exist_ok=True)
    portfolio_logs_dir(run_dir).mkdir(parents=True, exist_ok=True)
    portfolio_tables_dir(run_dir).mkdir(parents=True, exist_ok=True)
    portfolio_plots_dir(run_dir).mkdir(parents=True, exist_ok=True)
    ensure_mpl_cache(workspace_root / ".cruncher")

    status = PortfolioStatusV1(
        portfolio_name=spec.name,
        portfolio_id=portfolio_id,
        status="running",
        n_sources=len(spec.sources),
        n_selected_elites=0,
        warnings=[],
        started_at=utc_now_iso(),
        updated_at=utc_now_iso(),
    )
    write_portfolio_status(portfolio_status_path(run_dir), status)

    table_paths: list[Path] = []
    plot_paths: list[Path] = []
    source_runs: list[PortfolioSourceRun] = []
    prepared_sources: list[PortfolioPreparedSource] = []
    elite_summary_df = pd.DataFrame()

    try:
        if spec.execution.mode == "prepare_then_aggregate":
            ensured_study_runs: dict[tuple[str, str], Path] = {}
        else:
            ensured_study_runs = _ensure_required_source_studies(spec, run_study_fn=run_study, on_event=on_event)
        all_window_rows: list[dict[str, object]] = []
        all_elite_rows: list[dict[str, object]] = []
        source_summary_rows: list[dict[str, object]] = []
        study_summary_rows: list[dict[str, object]] = []
        sequence_length_rows: list[dict[str, object]] = []
        if spec.execution.mode == "prepare_then_aggregate":
            _emit_event(on_event, "prepare_phase_started", source_count=len(spec.sources))
            max_workers = min(int(spec.execution.max_parallel_sources), len(spec.sources))
            prepared_by_id: dict[str, PortfolioPreparedSource] = {}
            pending_by_id: dict[str, tuple[Future[PortfolioPreparedSource], Path]] = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                for source in spec.sources:
                    source_id = str(source.id)
                    source_readiness = readiness.get(source_id)
                    is_ready = bool(source_readiness and source_readiness.get("ready"))
                    if is_ready and prepare_ready_policy == "skip":
                        _emit_event(
                            on_event,
                            "prepare_source_skipped",
                            source_id=source_id,
                            reason="source already ready",
                        )
                        if source.prepare is None:
                            raise ValueError(
                                "portfolio.execution.mode=prepare_then_aggregate requires prepare for every source: "
                                f"missing source={source.id!r}"
                            )
                        prepared_by_id[source_id] = PortfolioPreparedSource(
                            source_id=source_id,
                            runbook_path=str(source.prepare.runbook),
                            step_ids=[],
                        )
                        continue
                    log_path = _prepare_source_log_path(run_dir, source_id)
                    _emit_event(
                        on_event,
                        "prepare_source_started",
                        source_id=source_id,
                        runbook=str(source.prepare.runbook if source.prepare is not None else ""),
                        log_path=str(log_path),
                    )
                    future = executor.submit(
                        _prepare_source,
                        source,
                        readiness=readiness,
                        prepare_ready_policy=prepare_ready_policy,
                        prepare_log_path=log_path,
                    )
                    pending_by_id[source_id] = (future, log_path)

                _emit_event(on_event, "aggregate_phase_started", source_count=len(spec.sources))
                for source in spec.sources:
                    source_id = str(source.id)
                    prepared = prepared_by_id.get(source_id)
                    if prepared is None:
                        future, log_path = pending_by_id[source_id]
                        try:
                            prepared = future.result()
                        except Exception:
                            for queued, _ in pending_by_id.values():
                                queued.cancel()
                            raise
                        _emit_event(
                            on_event,
                            "prepare_source_completed",
                            source_id=source_id,
                            executed_steps=list(prepared.step_ids),
                            log_path=str(log_path),
                        )
                        prepared_by_id[source_id] = prepared
                    prepared_sources.append(prepared)
                    ensured_study_runs.update(
                        _ensure_required_source_studies_for_sources(
                            spec,
                            [source],
                            run_study_fn=run_study,
                            on_event=on_event,
                        )
                    )
                    _emit_event(on_event, "aggregate_source_started", source_id=source_id)
                    (
                        source_windows_rows,
                        source_elite_rows,
                        source_summary_row,
                        source_run,
                        source_study_summary,
                    ) = _load_source_rows(source, on_event=on_event)
                    all_window_rows.extend(source_windows_rows)
                    all_elite_rows.extend(source_elite_rows)
                    source_summary_rows.append(source_summary_row)
                    source_runs.append(source_run)
                    _emit_event(
                        on_event,
                        "aggregate_source_completed",
                        source_id=source_id,
                        selected_elites=int(source_summary_row["n_selected_elites"]),
                    )
                    if source_study_summary is not None:
                        study_summary_rows.append(source_study_summary)
                    if spec.studies.sequence_length_table.enabled:
                        sequence_length_rows.extend(
                            _load_source_sequence_length_rows(
                                source,
                                study_spec=spec.studies.sequence_length_table.study_spec,
                                top_n_lengths=int(spec.studies.sequence_length_table.top_n_lengths),
                                ensured_study_runs=ensured_study_runs,
                                run_study_fn=run_study,
                                on_event=on_event,
                            )
                        )
                    table_paths, plot_paths, elite_summary_df = _materialize_portfolio_outputs(
                        run_dir=run_dir,
                        spec=spec,
                        all_window_rows=all_window_rows,
                        all_elite_rows=all_elite_rows,
                        source_summary_rows=source_summary_rows,
                        study_summary_rows=study_summary_rows,
                        sequence_length_rows=sequence_length_rows,
                    )
                    _emit_event(
                        on_event,
                        "aggregate_source_outputs_updated",
                        source_id=source_id,
                        completed_sources=len(source_runs),
                        table_count=len(table_paths),
                        plot_count=len(plot_paths),
                    )
            _emit_event(on_event, "prepare_phase_completed", prepared_count=len(prepared_sources))
            _emit_event(on_event, "aggregate_phase_completed", source_count=len(spec.sources))
        else:
            _emit_event(on_event, "aggregate_phase_started", source_count=len(spec.sources))
            for source in spec.sources:
                source_id = str(source.id)
                _emit_event(on_event, "aggregate_source_started", source_id=source_id)
                (
                    source_windows_rows,
                    source_elite_rows,
                    source_summary_row,
                    source_run,
                    source_study_summary,
                ) = _load_source_rows(source, on_event=on_event)
                all_window_rows.extend(source_windows_rows)
                all_elite_rows.extend(source_elite_rows)
                source_summary_rows.append(source_summary_row)
                source_runs.append(source_run)
                _emit_event(
                    on_event,
                    "aggregate_source_completed",
                    source_id=source_id,
                    selected_elites=int(source_summary_row["n_selected_elites"]),
                )
                if source_study_summary is not None:
                    study_summary_rows.append(source_study_summary)
                if spec.studies.sequence_length_table.enabled:
                    sequence_length_rows.extend(
                        _load_source_sequence_length_rows(
                            source,
                            study_spec=spec.studies.sequence_length_table.study_spec,
                            top_n_lengths=int(spec.studies.sequence_length_table.top_n_lengths),
                            ensured_study_runs=ensured_study_runs,
                            run_study_fn=run_study,
                            on_event=on_event,
                        )
                    )
                table_paths, plot_paths, elite_summary_df = _materialize_portfolio_outputs(
                    run_dir=run_dir,
                    spec=spec,
                    all_window_rows=all_window_rows,
                    all_elite_rows=all_elite_rows,
                    source_summary_rows=source_summary_rows,
                    study_summary_rows=study_summary_rows,
                    sequence_length_rows=sequence_length_rows,
                )
                _emit_event(
                    on_event,
                    "aggregate_source_outputs_updated",
                    source_id=source_id,
                    completed_sources=len(source_runs),
                    table_count=len(table_paths),
                    plot_count=len(plot_paths),
                )
            _emit_event(on_event, "aggregate_phase_completed", source_count=len(spec.sources))

        manifest = PortfolioManifestV1(
            portfolio_name=spec.name,
            portfolio_id=portfolio_id,
            spec_path=str(resolved_spec),
            spec_sha256=sha256_path(resolved_spec),
            created_at=utc_now_iso(),
            execution_mode=spec.execution.mode,
            source_runs=source_runs,
            prepared_sources=prepared_sources,
            table_paths=[str(path.resolve()) for path in table_paths],
            plot_paths=[str(path.resolve()) for path in plot_paths],
        )
        write_portfolio_manifest(portfolio_manifest_path(run_dir), manifest)

        status.status = "completed"
        status.n_sources = len(source_runs)
        status.n_selected_elites = int(len(elite_summary_df))
        status.updated_at = utc_now_iso()
        status.finished_at = utc_now_iso()
        write_portfolio_status(portfolio_status_path(run_dir), status)
        _emit_event(on_event, "portfolio_completed", run_dir=str(run_dir))
        return run_dir
    except Exception:
        status.status = "failed"
        status.updated_at = utc_now_iso()
        status.finished_at = utc_now_iso()
        write_portfolio_status(portfolio_status_path(run_dir), status)
        _emit_event(on_event, "portfolio_failed")
        raise


def portfolio_show_payload(portfolio_run_dir: Path) -> dict[str, object]:
    run_dir = portfolio_run_dir.expanduser().resolve()
    manifest = load_portfolio_manifest(portfolio_manifest_path(run_dir))
    status = load_portfolio_status(portfolio_status_path(run_dir))
    table_paths = sorted(portfolio_tables_dir(run_dir).glob("table__*"))
    plot_paths = sorted(portfolio_plots_dir(run_dir).glob(portfolio_plot_glob(run_dir)))
    source_runs = [
        {
            "source_id": item.source_id,
            "source_label": item.source_label,
            "source_top_k": int(item.source_top_k),
            "selected_elites": int(item.selected_elites),
            "workspace_name": item.workspace_name,
            "run_name": item.run_name,
        }
        for item in manifest.source_runs
    ]
    return {
        "portfolio_name": manifest.portfolio_name,
        "portfolio_id": manifest.portfolio_id,
        "status": status.status,
        "n_sources": status.n_sources,
        "n_selected_elites": status.n_selected_elites,
        "source_runs": source_runs,
        "manifest_path": str(portfolio_manifest_path(run_dir)),
        "status_path": str(portfolio_status_path(run_dir)),
        "table_paths": [str(path) for path in table_paths],
        "plot_paths": [str(path) for path in plot_paths],
    }
