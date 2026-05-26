"""Study-owned DenseGen axis OPAL probe package."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import yaml

from .active_targets import active_target_spec, with_active_target_columns
from .artifacts import ProbeArtifactLayout, RunSpec
from .constants import (
    ACTIVE_LABEL_FAMILY_ID,
    CAMPAIGNS,
    CANDIDATE_RECORDS,
    DENSEGEN_SIDECAR,
    FORBIDDEN_EXACT_COLUMNS,
    FORBIDDEN_PREFIXES,
    ORACLE_ID,
    STUDY_ID,
)
from .paths import _resolve_repo_path


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


def _write_candidate_scope(path: Path, *, ids: Sequence[str]) -> None:
    requested_ids = sorted(set(map(str, ids)))
    if not requested_ids:
        raise ValueError("candidate scope requires at least one id")
    _write_parquet(path, pd.DataFrame({"id": requested_ids}))


def _write_records_reference(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"source records.parquet not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        raise RuntimeError(
            "scratch records.parquet already exists and is not the expected shared-records symlink. "
            "Use a fresh --run-root or remove the stale scratch dataset."
        )
    rel_src = Path(os.path.relpath(src.resolve(), start=dst.parent.resolve()))
    try:
        dst.symlink_to(rel_src)
    except OSError:
        dst.symlink_to(src.resolve())


def _make_training_input(labels: pd.DataFrame, train_ids: Sequence[str]) -> pd.DataFrame:
    return _make_training_input_for_target(
        labels, train_ids, label_family_id=ACTIVE_LABEL_FAMILY_ID, campaign_key="cipro"
    )


def _make_training_input_for_run(labels: pd.DataFrame, train_ids: Sequence[str], run: RunSpec) -> pd.DataFrame:
    return _make_training_input_for_target(
        labels,
        train_ids,
        label_family_id=run.label_family_id,
        campaign_key=run.campaign_key,
    )


def _make_training_input_for_target(
    labels: pd.DataFrame,
    train_ids: Sequence[str],
    *,
    label_family_id: str,
    campaign_key: str,
) -> pd.DataFrame:
    requested_ids = set(map(str, train_ids))
    selected = (
        with_active_target_columns(labels, label_family_id)
        .loc[lambda frame: frame["id"].astype(str).isin(requested_ids)]
        .copy()
    )
    found_ids = set(selected["id"].astype(str).tolist())
    missing_ids = sorted(requested_ids - found_ids)
    if missing_ids:
        preview = ", ".join(missing_ids[:5])
        suffix = "" if len(missing_ids) <= 5 else f", ... ({len(missing_ids)} total)"
        raise ValueError(f"missing label rows for train id(s): {preview}{suffix}")
    required = list(active_target_spec(label_family_id, campaign_key).label_input_columns)
    missing = [column for column in required if column not in selected.columns]
    if missing:
        raise ValueError(f"label frame missing OPAL ingest column(s): {missing}")
    return selected[required].sort_values("id").reset_index(drop=True)


def _write_campaign_config(repo_root: Path, run: RunSpec, run_root: Path) -> None:
    source_config = _resolve_repo_path(repo_root, CAMPAIGNS[run.campaign_key]["source_config"])
    layout = ProbeArtifactLayout(run_root.resolve())
    cfg = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"campaign config must be a mapping: {source_config}")
    target = active_target_spec(run.label_family_id, run.campaign_key)
    slug = f"opal_densegen_probe_v1_{run.run_key}"
    dataset = layout.split_dataset(run.split_id)
    dataset_dir = layout.split_dataset_dir(run.split_id)
    sidecar_rel = run.sidecar_path.resolve().relative_to(dataset_dir.resolve())
    cfg["campaign"]["name"] = f"DenseGen control probe {run.run_key}"
    cfg["campaign"]["description"] = (
        f"Study-owned DenseGen control probe for {run.label_family_id}; "
        "synthetic labels are not measured SFXI assay values."
    )
    cfg["campaign"]["slug"] = slug
    cfg["campaign"]["workdir"] = str(run.workdir.resolve())
    campaign_metadata = dict(cfg["campaign"].get("metadata") or {})
    campaign_metadata.update(
        {
            "study_id": STUDY_ID,
            "campaign_context": "study_probe",
            "label_family_id": run.label_family_id,
            "label_oracle_kind": "positive" if run.oracle_id == ORACLE_ID else "null",
            "label_oracle_id": run.oracle_id,
            "label_split_id": run.split_id,
            "seed": run.seed,
            "target": run.campaign_key,
            "target_label": target.target_display,
            "target_class": run.target_class,
            "target_channel": target.target_channel,
            "target_description": target.target_description,
            "label_family_label": target.label_family_display,
            "probe_family": "opal_densegen_learnability_probe",
            "probe_target": run.campaign_key,
            "probe_target_label": target.target_display,
            "probe_target_class": run.target_class,
            "probe_target_channel": target.target_channel,
            "probe_oracle_kind": "positive" if run.oracle_id == ORACLE_ID else "null",
            "probe_oracle_id": run.oracle_id,
            "probe_label_family_id": run.label_family_id,
            "probe_seed": run.seed,
            "probe_split_id": run.split_id,
            "probe_run_key": run.run_key,
        }
    )
    cfg["campaign"]["metadata"] = campaign_metadata
    cfg["data"]["location"] = {
        "kind": "usr",
        "path": str(layout.scratch_usr_dir.resolve()),
        "dataset": dataset,
    }
    cfg["data"]["candidate_scope"] = {
        "kind": "id_list",
        "path": str(layout.split_candidate_scope_path(run.split_id).resolve()),
        "id_column": "id",
    }
    cfg["data"]["y_column_name"] = f"opal__{slug}__y"
    cfg["data"]["y_expected_length"] = int(target.y_expected_length)
    cfg["labels"] = {
        "source": {
            "kind": "usr_sidecar",
            "dataset": dataset,
            "path": str(sidecar_rel),
        },
        "y_space": target.y_space,
        "id_column": "id",
        "round_column": "observed_round",
        "batch_column": "batch_id",
        "dedup_policy": "latest_by_round",
    }
    cfg["transforms_y"] = {
        "name": target.transforms_y["name"],
        "params": dict(target.transforms_y.get("params") or {}),
    }
    cfg["objectives"] = [{"name": item["name"], "params": dict(item.get("params") or {})} for item in target.objectives]
    cfg.setdefault("training", {}).pop("y_ops", None)
    selection = cfg.setdefault("selection", {})
    selection_params = selection.setdefault("params", {})
    selection_params["top_k"] = int(run.selection_k)
    selection_params["score_ref"] = target.score_ref
    selection_params["objective_mode"] = target.objective_mode
    selection_params["tie_handling"] = "ordinal"
    if run.max_x_matrix_gib is not None:
        cfg.setdefault("safety", {})["max_x_matrix_gib"] = float(run.max_x_matrix_gib)
    if run.score_batch_size is not None:
        cfg.setdefault("scoring", {})["score_batch_size"] = int(run.score_batch_size)
    cfg["writeback"] = {"prediction_records": "ledger_only"}
    cfg["plot_config"] = "plots.yaml"
    run.config_path.parent.mkdir(parents=True, exist_ok=True)
    run.config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    write_campaign_plot_config(run)


def write_campaign_plot_config(run: RunSpec) -> None:
    target = active_target_spec(run.label_family_id, run.campaign_key)
    score_params = {
        "metric_label": target.score_label,
        "legend_metric_label": target.score_short_label,
        "metric_expression": target.score_expression,
        "collection_visual_label": target.collection_visual_label,
        "y_axis": dict(target.score_axis),
    }
    plots = [
        {
            "name": "score_selected_over_rounds",
            "kind": "metric_over_rounds",
            "round_selector": "all",
            "tags": ["rounds", "dogfood"],
            "params": {
                "metric": "pred__score_selected",
                "cohort": "selected",
                "summaries": ["mean", "count"],
                "band": "iqr",
                "title": "Selected objective score over rounds",
                "surface_label": f"Selected objective score: {target.score_title_label}",
                "caption": (
                    f"Selected-candidate mean {target.score_short_label} by OPAL round. {target.score_expression}."
                ),
                "review_purpose": (
                    "Check whether the active learner improves the configured objective for selected candidates "
                    "without treating objective-scale score as a biological effect size."
                ),
                **score_params,
            },
        },
        {
            "name": "score_vs_rank_over_rounds",
            "kind": "scatter_score_vs_rank",
            "round_selector": "all",
            "tags": ["rounds", "dogfood", "selection"],
            "params": {
                "score_field": "pred__score_selected",
                "rank_mode": "competition",
                "alpha": 0.45,
                "multi_round_alpha": 0.24,
                "round_cmap": "round_progression",
                "rasterize_at": 20000,
                "title": "Objective score vs selection rank",
                "surface_label": f"Objective score vs selection rank: {target.score_title_label}",
                **score_params,
            },
        },
        {
            "name": "score_threshold_over_rounds",
            "kind": "percent_high_activity_over_rounds",
            "round_selector": "all",
            "tags": ["rounds", "dogfood", "selection", "threshold"],
            "params": {
                "metric": "pred__score_selected",
                "threshold_quantile": 0.9,
                "mode": "line",
                "title": "Objective score enrichment above fixed P90",
                "surface_label": f"Objective score enrichment above fixed P90: {target.score_title_label}",
                **score_params,
            },
        },
        {
            "name": "feature_importance_heatmap",
            "kind": "feature_importance_heatmap",
            "round_selector": "all",
            "tags": ["rounds", "dogfood", "model"],
            "params": {
                "order_policy": "sort_index",
                "cluster": False,
                "rasterized": True,
                "figsize_in": [14.0, 4.4],
                "max_xticks": 16,
                "contrast_gamma": 0.55,
                "cmap": "opal_importance",
                "colorbar_label": "rf_feature_importance",
                "title": "RF feature importance by round",
            },
        },
        {
            "name": "feature_importance_bars",
            "kind": "feature_importance_bars",
            "round_selector": "all",
            "tags": ["rounds", "dogfood", "model"],
            "params": {
                "order_policy": "sort_index",
                "alpha": 0.40,
                "figsize_in": [14.0, 4.4],
                "bar_width": 1.05,
                "cmap": "round_progression",
                "max_xticks": 16,
                "title": "Random forest feature importance by round",
            },
        },
    ]
    vector_params: dict[str, Any] = {
        "vector_field": "pred__y_hat_model",
        "cohort": "selected",
        "include_reference_vector": bool(target.reference_vector),
        "channel_labels": list(target.channel_labels),
        "cmap": "opal_seafoam",
        "value_label": "Mean predicted value",
        "title": f"Selected predicted {target.label_family_display} vector",
        "figsize_in": [10.8, 5.2],
        "font_size": 13,
        "channel_axis_label": "",
    }
    if target.reference_vector:
        vector_params.update(
            {
                "reference_vector": list(target.reference_vector),
                "reference_label": "Target vector",
                "reference_mse_panel": True,
                "reference_mse_title": "Target-vector loss",
                "reference_mse_metric_label": ("Target-vector MSE\n$d^{-1}\\sum_c(\\bar{\\hat{y}}_c - t_c)^2$"),
                "reference_mse_legend_label": "target-vector MSE",
                "reference_mse_expression": ("MSE = d^-1 sum_c((mean selected y_hat_c - target_c)^2); lower is better"),
                "reference_mse_scale_class": "densegen_plan_logic4_reference_mse",
                "reference_mse_y_limits": [0.0, 0.25],
                "reference_mse_include_zero_tick": True,
            }
        )
    plots.append(
        {
            "name": "selected_target_vector_summary",
            "kind": "vector_summary_heatmap",
            "round_selector": "all",
            "tags": ["rounds", "dogfood", "vector", "generic"],
            "params": vector_params,
        }
    )
    plot_config = {
        "plot_defaults": {
            "output": {
                "format": "png",
                "dpi": 300,
                "save_data": True,
            }
        },
        "plots": plots,
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
