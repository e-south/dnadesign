"""Study-owned DenseGen axis OPAL probe package."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from .artifacts import ProbeArtifactLayout, RunSpec
from .constants import (
    CAMPAIGNS,
    CANDIDATE_RECORDS,
    DENSEGEN_SIDECAR,
    FORBIDDEN_EXACT_COLUMNS,
    FORBIDDEN_PREFIXES,
    SFXI_INTENSITY_COLUMNS,
    SFXI_STATE_COLUMNS,
)
from .paths import _resolve_repo_path
from .records_manifest import records_manifest_path, records_manifest_payload, records_manifest_problems


def _required_source_columns(path: Path, requested: Sequence[str]) -> list[str]:
    import pyarrow.parquet as pq

    schema_names = set(pq.ParquetFile(path).schema_arrow.names)
    return [column for column in requested if column in schema_names]


def _guard_forbidden_columns(columns: Sequence[str]) -> None:
    forbidden = [
        column
        for column in columns
        if column.startswith(FORBIDDEN_PREFIXES) or column in FORBIDDEN_EXACT_COLUMNS or column.startswith("opal__")
    ]
    if forbidden:
        raise ValueError(f"oracle source requested forbidden columns: {forbidden}")


def _load_candidate_inputs(repo_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidate_path = _resolve_repo_path(repo_root, CANDIDATE_RECORDS)
    sidecar_path = _resolve_repo_path(repo_root, DENSEGEN_SIDECAR)
    candidate_columns = [
        "id",
        "sequence",
        "densegen__used_tfbs_detail",
        "densegen__plan",
        "densegen__required_regulators",
        "densegen__sampling_library_hash",
        "opal_candidate__design_family",
    ]
    sidecar_columns = [
        "id",
        "densegen__used_tfbs_detail",
        "densegen__plan",
        "densegen__required_regulators",
        "densegen__sampling_library_hash",
    ]
    _guard_forbidden_columns(candidate_columns)
    _guard_forbidden_columns(sidecar_columns)
    candidate_columns = _required_source_columns(candidate_path, candidate_columns)
    sidecar_columns = _required_source_columns(sidecar_path, sidecar_columns)
    if "id" not in candidate_columns or "sequence" not in candidate_columns:
        raise ValueError(f"candidate records missing required id/sequence columns: {candidate_path}")
    if "id" not in sidecar_columns or "densegen__used_tfbs_detail" not in sidecar_columns:
        raise ValueError(f"DenseGen sidecar missing required id/detail columns: {sidecar_path}")
    return (
        pd.read_parquet(candidate_path, columns=candidate_columns),
        pd.read_parquet(sidecar_path, columns=sidecar_columns),
    )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)


def _stable_id_hash(ids: Sequence[str]) -> str:
    payload = "\n".join(sorted(map(str, ids))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_records_subset(src: Path, dst: Path, *, ids: Sequence[str]) -> None:
    requested_ids = sorted(set(map(str, ids)))
    if not requested_ids:
        raise ValueError("records subset requires at least one id")
    dst.parent.mkdir(parents=True, exist_ok=True)
    expected_hash = _stable_id_hash(requested_ids)
    if dst.exists():
        problems = records_manifest_problems(dst, src)
        manifest_path = records_manifest_path(dst)
        if problems:
            raise RuntimeError(
                "scratch records.parquet already exists without a matching source manifest. "
                f"Use a fresh --run-root or remove the stale scratch dataset: {', '.join(problems)}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("copy_mode") != "subset" or manifest.get("subset_ids_sha256") != expected_hash:
            raise RuntimeError(
                "scratch records.parquet already exists for a different subset. "
                "Use a fresh --run-root or remove the stale scratch dataset."
            )
        return

    import pyarrow as pa
    import pyarrow.parquet as pq

    source = pq.ParquetFile(src)
    wanted = set(requested_ids)
    found_ids: set[str] = set()
    row_count = 0
    tmp = dst.with_suffix(".tmp.parquet")
    writer: pq.ParquetWriter | None = None
    failed = False
    try:
        writer = pq.ParquetWriter(tmp, source.schema_arrow)
        for batch in source.iter_batches(batch_size=512):
            ids = [str(value) for value in batch.column("id").to_pylist()]
            mask = [value in wanted for value in ids]
            if not any(mask):
                continue
            table = pa.Table.from_batches([batch]).filter(pa.array(mask))
            kept_ids = [str(value) for value in table.column("id").to_pylist()]
            found_ids.update(kept_ids)
            row_count += int(table.num_rows)
            writer.write_table(table)
    except Exception:
        failed = True
        raise
    finally:
        if writer is not None:
            writer.close()
        if failed and tmp.exists():
            tmp.unlink()

    missing = sorted(wanted - found_ids)
    if missing:
        if tmp.exists():
            tmp.unlink()
        preview = ", ".join(missing[:5])
        suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} total)"
        raise RuntimeError(f"records subset is missing requested id(s): {preview}{suffix}")
    tmp.replace(dst)
    _write_json(
        records_manifest_path(dst),
        records_manifest_payload(
            src,
            dst,
            copy_mode="subset",
            row_count=int(row_count),
            subset_id_count=len(requested_ids),
            subset_ids_sha256=expected_hash,
        ),
    )


def _make_training_input(labels: pd.DataFrame, train_ids: Sequence[str]) -> pd.DataFrame:
    requested_ids = set(map(str, train_ids))
    selected = labels.loc[labels["id"].astype(str).isin(requested_ids)].copy()
    found_ids = set(selected["id"].astype(str).tolist())
    missing_ids = sorted(requested_ids - found_ids)
    if missing_ids:
        preview = ", ".join(missing_ids[:5])
        suffix = "" if len(missing_ids) <= 5 else f", ... ({len(missing_ids)} total)"
        raise ValueError(f"missing label rows for train id(s): {preview}{suffix}")
    required = ["id", "sequence", *SFXI_STATE_COLUMNS, *SFXI_INTENSITY_COLUMNS, "intensity_log2_offset_delta"]
    missing = [column for column in required if column not in selected.columns]
    if missing:
        raise ValueError(f"label frame missing OPAL ingest column(s): {missing}")
    return selected[required].sort_values("id").reset_index(drop=True)


def _write_campaign_config(repo_root: Path, run: RunSpec, run_root: Path) -> None:
    source_config = _resolve_repo_path(repo_root, CAMPAIGNS[run.campaign_key]["source_config"])
    layout = ProbeArtifactLayout(run_root)
    cfg = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"campaign config must be a mapping: {source_config}")
    slug = f"opal_axis_probe_v0_{run.run_key}"
    dataset = layout.split_dataset(run.split_id)
    dataset_dir = layout.split_dataset_dir(run.split_id)
    sidecar_rel = run.sidecar_path.relative_to(dataset_dir)
    cfg["campaign"]["name"] = f"{cfg['campaign']['name']} [{run.run_key}]"
    cfg["campaign"]["slug"] = slug
    cfg["campaign"]["workdir"] = str(run.workdir)
    cfg["data"]["location"] = {
        "kind": "usr",
        "path": str(layout.scratch_usr_dir),
        "dataset": dataset,
    }
    cfg["data"]["y_column_name"] = f"opal__{slug}__y"
    cfg["labels"] = {
        "source": {
            "kind": "usr_sidecar",
            "dataset": dataset,
            "path": str(sidecar_rel),
        },
        "y_space": "sfxi_vec8",
        "id_column": "id",
        "round_column": "observed_round",
        "batch_column": "batch_id",
        "dedup_policy": "latest_by_round",
    }
    selection = cfg.setdefault("selection", {})
    selection_params = selection.setdefault("params", {})
    selection_params["top_k"] = int(run.selection_k)
    selection_params["tie_handling"] = "ordinal"
    if run.max_x_matrix_gib is not None:
        cfg.setdefault("safety", {})["max_x_matrix_gib"] = float(run.max_x_matrix_gib)
    if run.score_batch_size is not None:
        cfg.setdefault("scoring", {})["score_batch_size"] = int(run.score_batch_size)
    cfg["writeback"] = {"prediction_records": "ledger_only"}
    cfg["plot_config"] = "plots.yaml"
    run.config_path.parent.mkdir(parents=True, exist_ok=True)
    run.config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    _write_campaign_plot_config(run)


def _write_campaign_plot_config(run: RunSpec) -> None:
    target_vec8 = list(CAMPAIGNS[run.campaign_key]["target_vec8"])
    plot_config = {
        "plot_defaults": {
            "output": {
                "format": "png",
                "dpi": 180,
                "save_data": True,
            }
        },
        "plots": [
            {
                "name": "score_selected_over_rounds",
                "kind": "metric_over_rounds",
                "tags": ["rounds", "dogfood"],
                "params": {
                    "metric": "pred__score_selected",
                    "cohort": ["selected", "top_k", "all_pool"],
                    "top_k": int(run.selection_k),
                    "summaries": ["mean", "median", "q25", "q75"],
                    "highlight_round": "latest",
                    "title": f"{run.run_key}: score over rounds",
                },
            },
            {
                "name": "score_vs_rank_by_round",
                "kind": "scatter_score_vs_rank",
                "tags": ["rounds", "dogfood", "selection"],
                "params": {
                    "rank_mode": "competition",
                    "alpha": 0.45,
                    "rasterize_at": 20000,
                    "figsize_in": [9, 5],
                },
            },
            {
                "name": "score_threshold_over_rounds",
                "kind": "percent_high_activity_over_rounds",
                "tags": ["rounds", "dogfood", "selection", "threshold"],
                "params": {
                    "metric": "pred__score_selected",
                    "threshold": 0.8,
                    "mode": "both",
                    "hue": "logic_fidelity",
                    "highlight_round": "latest",
                    "rasterize_at": 20000,
                },
            },
            {
                "name": "feature_importance_heatmap",
                "kind": "feature_importance_heatmap",
                "tags": ["rounds", "dogfood", "model"],
                "params": {
                    "top_n": 128,
                    "sort": "max_importance",
                    "cluster": False,
                    "title": f"{run.run_key}: feature importance over rounds",
                },
            },
            {
                "name": "selected_vec8_summary",
                "kind": "vector_summary_heatmap",
                "tags": ["rounds", "dogfood", "vector"],
                "params": {
                    "vector_field": "pred__y_hat_model",
                    "cohort": "selected",
                    "include_reference_vector": True,
                    "reference_vector": target_vec8,
                    "reference_label": "target vec8",
                    "channel_labels": [*SFXI_STATE_COLUMNS, *SFXI_INTENSITY_COLUMNS],
                    "title": f"{run.run_key}: selected vec8 summary",
                },
            },
            {
                "name": "fold_change_vs_logic_fidelity_latest",
                "kind": "fold_change_vs_logic_fidelity",
                "round_selector": "latest",
                "tags": ["sfxi", "dogfood", "selection", "overlay", "single-round"],
                "params": {
                    "y_axis": "score",
                    "hue": "effect_scaled",
                    "size_by": "logic_fidelity",
                    "alpha": 0.35,
                    "rasterize_at": 20000,
                },
            },
            {
                "name": "sfxi_logic_fidelity_closeness_latest",
                "kind": "sfxi_logic_fidelity_closeness",
                "round_selector": "latest",
                "tags": ["sfxi", "dogfood", "labels", "overlay", "single-round"],
                "params": {
                    "on_violin_invalid": "line",
                    "violin_min_points": 3,
                },
            },
            {
                "name": "sfxi_factorial_effects_latest",
                "kind": "sfxi_factorial_effects",
                "round_selector": "latest",
                "tags": ["sfxi", "dogfood", "labels", "overlay", "single-round"],
                "params": {
                    "size_by": "obj__effect_scaled",
                    "include_labels": True,
                    "rasterize_at": 20000,
                },
            },
            {
                "name": "sfxi_setpoint_sweep_latest",
                "kind": "sfxi_setpoint_sweep",
                "round_selector": "latest",
                "tags": ["sfxi", "dogfood", "labels", "single-round"],
                "params": {"min_n": 5},
            },
            {
                "name": "sfxi_support_diagnostics_latest",
                "kind": "sfxi_support_diagnostics",
                "round_selector": "latest",
                "tags": ["sfxi", "dogfood", "labels", "overlay", "single-round"],
                "params": {
                    "y_axis": "score",
                    "hue": "effect_scaled",
                    "batch_size": 2048,
                },
            },
            {
                "name": "sfxi_uncertainty_latest",
                "kind": "sfxi_uncertainty",
                "round_selector": "latest",
                "tags": ["sfxi", "dogfood", "model", "single-round"],
                "params": {
                    "kind": "score",
                    "y_axis": "score",
                    "hue": "logic_fidelity",
                },
            },
            {
                "name": "sfxi_intensity_scaling_latest",
                "kind": "sfxi_intensity_scaling",
                "round_selector": "latest",
                "tags": ["sfxi", "dogfood", "labels", "single-round"],
                "params": {
                    "min_n": 5,
                    "include_pool": True,
                },
            },
        ],
    }
    path = run.config_path.parent / "plots.yaml"
    path.write_text(yaml.safe_dump(plot_config, sort_keys=False), encoding="utf-8")


def _run_command(command: Sequence[str], *, cwd: Path, machine_readable: bool = False) -> None:
    stream = sys.stderr if machine_readable else sys.stdout
    print("+ " + " ".join(map(str, command)), file=stream, flush=True)
    if machine_readable:
        rendered = list(map(str, command))
        try:
            sys.stderr.fileno()
        except (AttributeError, OSError, ValueError):
            completed = subprocess.run(rendered, cwd=cwd, check=True, capture_output=True, text=True)
            if completed.stdout:
                print(completed.stdout, end="", file=sys.stderr)
            if completed.stderr:
                print(completed.stderr, end="", file=sys.stderr)
        else:
            subprocess.run(rendered, cwd=cwd, check=True, stdout=sys.stderr, stderr=sys.stderr)
    else:
        subprocess.run(list(map(str, command)), cwd=cwd, check=True)
