"""Study-owned DenseGen axis OPAL probe package."""

from __future__ import annotations

import json
import shutil
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
    SCRATCH_DATASET,
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


def _clone_records_file(src: Path, dst: Path, *, copy_mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        problems = records_manifest_problems(dst, src)
        if problems:
            raise RuntimeError(
                "scratch records.parquet already exists without a matching source manifest. "
                f"Use a fresh --run-root or remove the stale scratch dataset: {', '.join(problems)}"
            )
        return
    if copy_mode == "clone":
        result = subprocess.run(["cp", "-c", str(src), str(dst)], check=False, capture_output=True, text=True)
        if result.returncode != 0:
            message = (result.stderr or result.stdout or "").strip()
            raise RuntimeError(
                "failed to create APFS clone for scratch records.parquet. "
                "Re-run with --copy-mode full if a full 5 GB scratch copy is acceptable. "
                f"cp -c error: {message}"
            )
    elif copy_mode == "full":
        shutil.copy2(src, dst)
    else:
        raise ValueError("--copy-mode must be clone or full")
    _write_json(records_manifest_path(dst), records_manifest_payload(src, dst))


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
    sidecar_rel = run.sidecar_path.relative_to(layout.scratch_dataset_dir)
    cfg["campaign"]["name"] = f"{cfg['campaign']['name']} [{run.run_key}]"
    cfg["campaign"]["slug"] = slug
    cfg["campaign"]["workdir"] = str(run.workdir)
    cfg["data"]["location"] = {
        "kind": "usr",
        "path": str(layout.scratch_usr_dir),
        "dataset": SCRATCH_DATASET,
    }
    cfg["data"]["y_column_name"] = f"opal__{slug}__y"
    cfg["labels"] = {
        "source": {
            "kind": "usr_sidecar",
            "dataset": SCRATCH_DATASET,
            "path": str(sidecar_rel),
        },
        "y_space": "sfxi_vec8",
        "id_column": "id",
        "round_column": "observed_round",
        "batch_column": "batch_id",
        "dedup_policy": "latest_by_round",
    }
    cfg["writeback"] = {"prediction_records": "ledger_only"}
    run.config_path.parent.mkdir(parents=True, exist_ok=True)
    run.config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")


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
