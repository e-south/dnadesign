"""
Stage-A TFBS pool artifacts.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from ...utils.logging_utils import install_native_stderr_filters
from .ids import hash_tfbs_id

POOL_SCHEMA_VERSION = "1.0"
POOL_MODE_TFBS = "tfbs"
POOL_MODE_SEQUENCE = "sequence"
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _sanitize_filename(name: str) -> str:
    cleaned = _SAFE_FILENAME_RE.sub("_", str(name).strip())
    return cleaned or "densegen"


@dataclass(frozen=True)
class PoolInputEntry:
    name: str
    input_type: str
    pool_path: Path
    rows: int
    columns: list[str]
    pool_mode: str


@dataclass(frozen=True)
class PoolData:
    name: str
    input_type: str
    pool_mode: str
    df: pd.DataFrame | None
    sequences: list[str]
    pool_path: Path


@dataclass(frozen=True)
class TFBSPoolArtifact:
    manifest_path: Path
    inputs: dict[str, PoolInputEntry]
    schema_version: str
    run_id: str
    run_root: str
    config_path: str

    @classmethod
    def load(cls, manifest_path: Path) -> "TFBSPoolArtifact":
        payload = json.loads(manifest_path.read_text())
        entries = {}
        for item in payload.get("inputs", []):
            entry = PoolInputEntry(
                name=str(item.get("name")),
                input_type=str(item.get("type")),
                pool_path=Path(item.get("pool_path")),
                rows=int(item.get("rows", 0)),
                columns=list(item.get("columns") or []),
                pool_mode=str(item.get("pool_mode") or POOL_MODE_TFBS),
            )
            entries[entry.name] = entry
        return cls(
            manifest_path=manifest_path,
            inputs=entries,
            schema_version=str(payload.get("schema_version")),
            run_id=str(payload.get("run_id")),
            run_root=str(payload.get("run_root")),
            config_path=str(payload.get("config_path")),
        )

    def entry_for(self, input_name: str) -> PoolInputEntry:
        if input_name not in self.inputs:
            raise KeyError(f"Pool manifest missing input: {input_name}")
        return self.inputs[input_name]


def _pool_manifest_path(out_dir: Path) -> Path:
    return out_dir / "pool_manifest.json"


def load_pool_artifact(out_dir: Path) -> TFBSPoolArtifact:
    manifest_path = _pool_manifest_path(out_dir)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Pool manifest not found: {manifest_path}")
    return TFBSPoolArtifact.load(manifest_path)


def _resolve_pool_mode(df: pd.DataFrame) -> str:
    if "tf" in df.columns and "tfbs" in df.columns:
        return POOL_MODE_TFBS
    if "sequence" in df.columns:
        return POOL_MODE_SEQUENCE
    raise ValueError("Pool dataframe must contain tf/tfbs columns or a sequence column.")


def _ensure_tfbs_ids(df: pd.DataFrame) -> None:
    missing = [col for col in ("motif_id", "tfbs_id") if col not in df.columns]
    if missing:
        raise ValueError(f"TFBS pool missing required columns: {', '.join(missing)}")


def _build_sequence_pool(sequences: Iterable[str]) -> pd.DataFrame:
    seqs = [str(s) for s in sequences]
    df = pd.DataFrame({"sequence": seqs})
    df["tfbs_id"] = [
        hash_tfbs_id(
            motif_id=None,
            sequence=seq,
            scoring_backend="sequence_library",
        )
        for seq in seqs
    ]
    return df


def build_pool_artifact(
    *,
    cfg,
    cfg_path: Path,
    deps,
    rng,
    outputs_root: Path,
    out_dir: Path,
    overwrite: bool = False,
    selected_inputs: set[str] | None = None,
) -> tuple[TFBSPoolArtifact, dict[str, PoolData]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    install_native_stderr_filters()
    pool_entries: dict[str, PoolInputEntry] = {}
    pool_data: dict[str, PoolData] = {}
    used_names: dict[str, int] = {}
    rows: list[tuple[str, str, str, Path]] = []

    for inp in cfg.inputs:
        if selected_inputs and inp.name not in selected_inputs:
            continue
        src = deps.source_factory(inp, cfg_path)
        data_entries, meta_df = src.load_data(rng=rng, outputs_root=outputs_root)
        if meta_df is None:
            df = _build_sequence_pool(data_entries)
        else:
            df = meta_df.copy()
        df.insert(0, "input_name", inp.name)

        pool_mode = _resolve_pool_mode(df)
        if pool_mode == POOL_MODE_TFBS:
            _ensure_tfbs_ids(df)

        base = _sanitize_filename(inp.name)
        count = used_names.get(base, 0)
        used_names[base] = count + 1
        suffix = f"{base}__{count}" if count else base
        filename = f"{suffix}__pool.parquet"
        dest = out_dir / filename
        if dest.exists() and not overwrite:
            raise FileExistsError(f"Pool already exists: {dest}")
        df.to_parquet(dest, index=False)

        entry = PoolInputEntry(
            name=inp.name,
            input_type=str(inp.type),
            pool_path=Path(filename),
            rows=int(len(df)),
            columns=list(df.columns),
            pool_mode=pool_mode,
        )
        pool_entries[inp.name] = entry
        sequences: list[str]
        if pool_mode == POOL_MODE_SEQUENCE:
            sequences = df["sequence"].tolist()
            pool_df = None
        else:
            sequences = df["tfbs"].tolist() if "tfbs" in df.columns else []
            pool_df = df
        pool_data[inp.name] = PoolData(
            name=inp.name,
            input_type=str(inp.type),
            pool_mode=pool_mode,
            df=pool_df,
            sequences=sequences,
            pool_path=dest,
        )
        rows.append((inp.name, str(inp.type), str(len(df)), dest))

    if not rows:
        raise ValueError("No pools built (no matching inputs).")

    manifest = {
        "schema_version": POOL_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": cfg.run.id,
        "run_root": str(cfg.run.root),
        "config_path": str(cfg_path),
        "inputs": [
            {
                "name": entry.name,
                "type": entry.input_type,
                "pool_path": entry.pool_path.name,
                "rows": entry.rows,
                "columns": entry.columns,
                "pool_mode": entry.pool_mode,
            }
            for entry in pool_entries.values()
        ],
    }
    manifest_path = _pool_manifest_path(out_dir)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))

    artifact = TFBSPoolArtifact(
        manifest_path=manifest_path,
        inputs=pool_entries,
        schema_version=POOL_SCHEMA_VERSION,
        run_id=str(cfg.run.id),
        run_root=str(cfg.run.root),
        config_path=str(cfg_path),
    )
    return artifact, pool_data
