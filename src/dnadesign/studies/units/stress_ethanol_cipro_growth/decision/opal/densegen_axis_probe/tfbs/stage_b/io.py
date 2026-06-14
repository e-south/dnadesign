"""Stage B TFBS filesystem IO contracts."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


def read_stage_b_json(path: Path) -> dict[str, Any]:
    """Read a required JSON manifest and fail if it is not an object."""

    if not path.exists():
        raise FileNotFoundError(f"required Stage B source manifest not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manifest must be a JSON object: {path}")
    return payload


def write_stage_b_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write a deterministic Stage B JSON artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_stage_b_parquet(path: Path, frame: pd.DataFrame) -> None:
    """Write a compressed Stage B parquet artifact."""

    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False, compression="zstd")


def read_stage_b_label_table(path: Path) -> pd.DataFrame:
    """Read a required positive or null label table."""

    if not path.exists():
        raise FileNotFoundError(f"Stage B label table not found: {path}")
    return pd.read_parquet(path)


def read_stage_b_candidate_identity(path: Path) -> pd.DataFrame:
    """Read and validate the candidate identity columns required by Stage B."""

    if not path.exists():
        raise FileNotFoundError(f"Stage B source records not found: {path}")
    frame = pd.read_parquet(path, columns=["id", "sequence"])
    missing = [column for column in ("id", "sequence") if column not in frame.columns]
    if missing:
        raise ValueError(f"Stage B source records missing identity column(s): {missing}")
    if frame["id"].isna().any() or frame["sequence"].isna().any():
        raise ValueError("Stage B source records identity columns must not contain nulls")
    out = frame.loc[:, ["id", "sequence"]].copy()
    out["id"] = out["id"].astype(str)
    out["sequence"] = out["sequence"].astype(str)
    if out["id"].duplicated().any():
        duplicates = out.loc[out["id"].duplicated(), "id"].head(10).tolist()
        raise ValueError(f"Stage B source records contain duplicate ids: {duplicates}")
    return out


def write_stage_b_initial_label_input(
    path: Path,
    frame: pd.DataFrame,
    *,
    label_name: str,
    initial_ids: Sequence[str],
    candidate_identity: pd.DataFrame,
) -> None:
    """Write the round-zero label input for one campaign."""

    required = ["id", label_name]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"label table missing Stage B input column(s): {missing}")
    if "sequence" not in frame.columns:
        frame = frame.merge(candidate_identity, on="id", how="left", validate="one_to_one")
    if "sequence" not in frame.columns:
        raise ValueError("Stage B label input requires sequence from either label table or source records")
    if frame["sequence"].isna().any():
        missing_sequence_ids = frame.loc[frame["sequence"].isna(), "id"].astype(str).head(10).tolist()
        raise ValueError(f"source records missing sequence for label id(s): {missing_sequence_ids}")
    required = ["id", "sequence", label_name]
    wanted = set(map(str, initial_ids))
    selected = frame.loc[frame["id"].astype(str).isin(wanted), required].copy()
    found = set(selected["id"].astype(str).tolist())
    missing_ids = sorted(wanted - found)
    if missing_ids:
        raise ValueError(f"label table missing initial id(s) for {label_name}: {missing_ids[:10]}")
    order = {candidate_id: index for index, candidate_id in enumerate(initial_ids)}
    selected["__order__"] = selected["id"].astype(str).map(order)
    selected = selected.sort_values("__order__").drop(columns=["__order__"])
    write_stage_b_parquet(path, selected)


def write_stage_b_candidate_scope(path: Path, ids: Sequence[str]) -> None:
    """Write the ordered candidate-scope id list used by OPAL."""

    unique_ids = sorted(set(map(str, ids)))
    if not unique_ids:
        raise ValueError("Stage B candidate scope requires at least one id")
    if len(unique_ids) != len(ids):
        raise ValueError("Stage B candidate scope requires unique ids")
    write_stage_b_parquet(path, pd.DataFrame({"id": unique_ids}))


def write_stage_b_records_reference(src: Path, dst: Path) -> None:
    """Create a stable records reference without copying the large source table."""

    if not src.exists():
        raise FileNotFoundError(f"Stage B source records.parquet not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        if dst.resolve() == src.resolve():
            return
        raise RuntimeError(f"Stage B records reference already exists and points elsewhere: {dst}")
    rel_src = Path(os.path.relpath(src.resolve(), start=dst.parent.resolve()))
    try:
        dst.symlink_to(rel_src)
    except OSError:
        dst.symlink_to(src.resolve())


def prepare_stage_b_out_dir(
    out_dir: Path,
    *,
    replace: bool,
    refresh_existing_execution_state: bool = False,
) -> None:
    """Prepare a Stage B output directory without silently overwriting run state."""

    if replace and out_dir.exists():
        shutil.rmtree(out_dir)
    if not out_dir.exists():
        out_dir.mkdir(parents=True)
        return
    mutable_outputs = list(out_dir.glob("campaigns/*/outputs")) + list(
        out_dir.glob("scratch_usr/*/_opal/*/observed_labels.parquet")
    )
    if mutable_outputs:
        if refresh_existing_execution_state:
            return
        preview = ", ".join(str(path) for path in mutable_outputs[:3])
        raise RuntimeError(
            "Stage B config generation refuses to overwrite execution state without replace_out_dir=True "
            "or refresh_existing_execution_state=True "
            f"(sample={preview})"
        )
