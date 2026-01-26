"""
Stage-A TFBS pool artifacts.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd

from ...adapters.sources.pwm_sampling import FIMO_REPORT_THRESH, PWMSamplingSummary
from ...config import resolve_relative_path
from ...utils.logging_utils import install_native_stderr_filters
from .ids import hash_tfbs_id

POOL_SCHEMA_VERSION = "1.3"
POOL_MODE_TFBS = "tfbs"
POOL_MODE_SEQUENCE = "sequence"
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _sanitize_filename(name: str) -> str:
    cleaned = _SAFE_FILENAME_RE.sub("_", str(name).strip())
    return cleaned or "densegen"


def _hash_pool_config(cfg) -> str:
    payload = [inp.model_dump(mode="json") for inp in sorted(cfg.inputs, key=lambda item: item.name)]
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _resolve_input_paths(cfg_path: Path, source_cfg) -> list[Path]:
    paths: list[Path] = []
    if hasattr(source_cfg, "path"):
        paths.append(resolve_relative_path(cfg_path, getattr(source_cfg, "path")))
    if hasattr(source_cfg, "paths"):
        for path in getattr(source_cfg, "paths") or []:
            paths.append(resolve_relative_path(cfg_path, path))
    return paths


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_fingerprints(fingerprints: list[dict]) -> list[dict]:
    return sorted((dict(fp) for fp in fingerprints), key=lambda fp: str(fp.get("path", "")))


def _resolve_input_fingerprints(cfg_path: Path, source_cfg) -> list[dict]:
    fingerprints: list[dict] = []
    for path in _resolve_input_paths(cfg_path, source_cfg):
        if not path.exists():
            raise FileNotFoundError(f"Input file missing: {path}")
        if not path.is_file():
            raise ValueError(f"Input path is not a file: {path}")
        stat = path.stat()
        fingerprints.append(
            {
                "path": str(path),
                "size": int(stat.st_size),
                "mtime": float(stat.st_mtime),
                "sha256": _hash_file(path),
            }
        )
    return _normalize_fingerprints(fingerprints)


@dataclass(frozen=True)
class PoolInputEntry:
    name: str
    input_type: str
    pool_path: Path
    rows: int
    columns: list[str]
    pool_mode: str
    fingerprints: list[dict] | None = None
    stage_a_sampling: dict | None = None


@dataclass(frozen=True)
class PoolData:
    name: str
    input_type: str
    pool_mode: str
    df: pd.DataFrame | None
    sequences: list[str]
    pool_path: Path
    summaries: list[object] | None = None


@dataclass(frozen=True)
class TFBSPoolArtifact:
    manifest_path: Path
    inputs: dict[str, PoolInputEntry]
    schema_version: str
    run_id: str
    run_root: str
    config_path: str
    config_hash: str | None = None

    @classmethod
    def load(cls, manifest_path: Path) -> "TFBSPoolArtifact":
        payload = json.loads(manifest_path.read_text())
        entries = {}
        for item in payload.get("inputs", []):
            fingerprints = item.get("fingerprints")
            if fingerprints is not None:
                fingerprints = [dict(fp) for fp in fingerprints]
            stage_a_sampling = item.get("stage_a_sampling")
            if stage_a_sampling is not None:
                stage_a_sampling = dict(stage_a_sampling)
            entry = PoolInputEntry(
                name=str(item.get("name")),
                input_type=str(item.get("type")),
                pool_path=Path(item.get("pool_path")),
                rows=int(item.get("rows", 0)),
                columns=list(item.get("columns") or []),
                pool_mode=str(item.get("pool_mode") or POOL_MODE_TFBS),
                fingerprints=fingerprints,
                stage_a_sampling=stage_a_sampling,
            )
            entries[entry.name] = entry
        return cls(
            manifest_path=manifest_path,
            inputs=entries,
            schema_version=str(payload.get("schema_version")),
            run_id=str(payload.get("run_id")),
            run_root=str(payload.get("run_root")),
            config_path=str(payload.get("config_path")),
            config_hash=payload.get("config_hash"),
        )

    def entry_for(self, input_name: str) -> PoolInputEntry:
        if input_name not in self.inputs:
            raise KeyError(f"Pool manifest missing input: {input_name}")
        return self.inputs[input_name]


def _build_stage_a_sampling_manifest(summaries: list[object] | None) -> dict | None:
    if not summaries:
        return None
    pwm_summaries = [s for s in summaries if isinstance(s, PWMSamplingSummary)]
    if not pwm_summaries:
        return None
    fimo_summaries = [s for s in pwm_summaries if s.backend == "fimo"]
    if not fimo_summaries:
        return None
    eligible_score_hist = []
    for summary in fimo_summaries:
        if summary.eligible_score_hist_edges is None or summary.eligible_score_hist_counts is None:
            raise ValueError("Stage-A sampling summaries missing eligible score histogram.")
        if summary.eligible_score_hist_edges:
            if len(summary.eligible_score_hist_counts) != len(summary.eligible_score_hist_edges) - 1:
                raise ValueError("Stage-A eligible score histogram length mismatch.")
        eligible_score_hist.append(
            {
                "regulator": summary.regulator,
                "edges": [float(v) for v in summary.eligible_score_hist_edges],
                "counts": [int(v) for v in summary.eligible_score_hist_counts],
                "tier0_score": summary.tier0_score,
                "tier1_score": summary.tier1_score,
            }
        )
    return {
        "backend": "fimo",
        "tier_scheme": "pct_1_9_90",
        "eligibility_rule": "best_hit_score > 0 (and has at least one FIMO hit)",
        "retention_rule": "top_n_sites_by_best_hit_score",
        "fimo_thresh": FIMO_REPORT_THRESH,
        "eligible_score_hist": eligible_score_hist,
    }


def _pool_manifest_path(out_dir: Path) -> Path:
    return out_dir / "pool_manifest.json"


def load_pool_artifact(out_dir: Path) -> TFBSPoolArtifact:
    manifest_path = _pool_manifest_path(out_dir)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Pool manifest not found: {manifest_path}")
    return TFBSPoolArtifact.load(manifest_path)


def _pool_filename_prefix(path: Path) -> str:
    name = path.name
    suffix = "__pool.parquet"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return path.stem


def _seed_used_names_from_entries(entries: dict[str, PoolInputEntry]) -> dict[str, int]:
    used: dict[str, int] = {}
    for entry in entries.values():
        prefix = _pool_filename_prefix(entry.pool_path)
        base = prefix
        count = None
        if "__" in prefix:
            maybe_base, maybe_num = prefix.rsplit("__", 1)
            if maybe_num.isdigit():
                base = maybe_base
                count = int(maybe_num)
        current = used.get(base, 0)
        if count is None:
            used[base] = max(current, 1)
        else:
            used[base] = max(current, count + 1)
    return used


def load_pool_data(out_dir: Path) -> tuple[TFBSPoolArtifact, dict[str, PoolData]]:
    artifact = load_pool_artifact(out_dir)
    pool_data: dict[str, PoolData] = {}
    for entry in artifact.inputs.values():
        pool_path = out_dir / entry.pool_path
        if not pool_path.exists():
            raise FileNotFoundError(f"Pool file listed in manifest is missing: {pool_path}")
        df = pd.read_parquet(pool_path)
        pool_mode = entry.pool_mode or _resolve_pool_mode(df)
        if pool_mode == POOL_MODE_TFBS:
            _ensure_tfbs_ids(df)
        sequences: list[str]
        if pool_mode == POOL_MODE_SEQUENCE:
            sequences = df["sequence"].tolist() if "sequence" in df.columns else []
            pool_df = None
        else:
            sequences = df["tfbs"].tolist() if "tfbs" in df.columns else []
            pool_df = df
        pool_data[entry.name] = PoolData(
            name=entry.name,
            input_type=entry.input_type,
            pool_mode=pool_mode,
            df=pool_df,
            sequences=sequences,
            pool_path=pool_path,
            summaries=None,
        )
    return artifact, pool_data


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
    install_native_stderr_filters(suppress_solver_messages=False)
    pool_entries: dict[str, PoolInputEntry] = {}
    pool_data: dict[str, PoolData] = {}
    used_names: dict[str, int] = {}
    rows: list[tuple[str, str, str, Path]] = []
    existing_entries: dict[str, PoolInputEntry] = {}
    preserved_entries: dict[str, PoolInputEntry] = {}
    config_hash = _hash_pool_config(cfg)
    fingerprints_by_input = {inp.name: _resolve_input_fingerprints(cfg_path, inp) for inp in cfg.inputs}

    if not overwrite:
        manifest_path = _pool_manifest_path(out_dir)
        if manifest_path.exists():
            existing_artifact = TFBSPoolArtifact.load(manifest_path)
            existing_config_hash = existing_artifact.config_hash
            if not existing_config_hash:
                raise ValueError("Pool manifest missing config hash. Use --fresh to rebuild pools.")
            if existing_config_hash != config_hash:
                raise ValueError("Pool config changed. Use --fresh to rebuild pools.")
            existing_entries = existing_artifact.inputs
            used_names = _seed_used_names_from_entries(existing_entries)
            current_inputs = {inp.name for inp in cfg.inputs}
            stale = sorted(set(existing_entries) - current_inputs)
            if stale:
                raise ValueError(
                    "Pool manifest contains inputs not present in the config: "
                    f"{', '.join(stale)}. Use --fresh to rebuild pools."
                )
            if selected_inputs:
                preserved_entries = {
                    name: entry for name, entry in existing_entries.items() if name not in selected_inputs
                }
            for inp in cfg.inputs:
                entry = existing_entries.get(inp.name)
                if entry is None:
                    continue
                existing_fingerprints = entry.fingerprints
                if existing_fingerprints is None:
                    raise ValueError(
                        f"Pool manifest missing input fingerprints for '{inp.name}'. Use --fresh to rebuild pools."
                    )
                current_fingerprints = fingerprints_by_input.get(inp.name, [])
                if _normalize_fingerprints(existing_fingerprints) != _normalize_fingerprints(current_fingerprints):
                    raise ValueError(f"Input files changed for '{inp.name}'. Use --fresh to rebuild pools.")

    for inp in cfg.inputs:
        if selected_inputs and inp.name not in selected_inputs:
            continue
        existing_entry = existing_entries.get(inp.name)
        existing_df = None
        if existing_entry is not None:
            pool_path = out_dir / existing_entry.pool_path
            if not pool_path.exists():
                raise FileNotFoundError(f"Pool file listed in manifest is missing: {pool_path}")
            existing_df = pd.read_parquet(pool_path)
        src = deps.source_factory(inp, cfg_path)
        data_entries, meta_df, summaries = src.load_data(
            rng=rng,
            outputs_root=outputs_root,
            run_id=str(cfg.run.id),
        )
        if meta_df is None:
            df = _build_sequence_pool(data_entries)
        else:
            df = meta_df.copy()
        df.insert(0, "input_name", inp.name)

        pool_mode = _resolve_pool_mode(df)
        if pool_mode == POOL_MODE_TFBS:
            _ensure_tfbs_ids(df)

        if existing_df is not None:
            if set(existing_df.columns) != set(df.columns):
                raise ValueError(f"Pool schema changed for input '{inp.name}'. Use --fresh to rebuild pools.")
            if existing_entry is None:
                raise ValueError(f"Pool manifest missing entry for input '{inp.name}'.")
            if existing_entry.pool_mode and existing_entry.pool_mode != pool_mode:
                raise ValueError(
                    f"Pool mode mismatch for input '{inp.name}': "
                    f"{existing_entry.pool_mode} vs {pool_mode}. Use --fresh to rebuild."
                )
            df = df[existing_df.columns]
            merge_key = "tfbs_id" if "tfbs_id" in df.columns else "sequence"
            combined = pd.concat([existing_df, df], ignore_index=True)
            combined = combined.drop_duplicates(subset=[merge_key], keep="first")
            df = combined
            dest = out_dir / existing_entry.pool_path
        else:
            base = _sanitize_filename(inp.name)
            count = used_names.get(base, 0)
            used_names[base] = count + 1
            suffix = f"{base}__{count}" if count else base
            filename = f"{suffix}__pool.parquet"
            dest = out_dir / filename
            if dest.exists() and not overwrite:
                raise FileExistsError(f"Pool already exists: {dest}")
        df.to_parquet(dest, index=False)

        stage_a_sampling = _build_stage_a_sampling_manifest(summaries)
        entry = PoolInputEntry(
            name=inp.name,
            input_type=str(inp.type),
            pool_path=Path(dest.name),
            rows=int(len(df)),
            columns=list(df.columns),
            pool_mode=pool_mode,
            fingerprints=fingerprints_by_input.get(inp.name, []),
            stage_a_sampling=stage_a_sampling,
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
            summaries=summaries,
        )
        rows.append((inp.name, str(inp.type), str(len(df)), dest))

    if not rows and not preserved_entries:
        raise ValueError("No pools built (no matching inputs).")

    if preserved_entries:
        for name, entry in preserved_entries.items():
            pool_entries.setdefault(name, entry)

    manifest = {
        "schema_version": POOL_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": cfg.run.id,
        "run_root": str(cfg.run.root),
        "config_path": str(cfg_path),
        "config_hash": config_hash,
        "inputs": [
            {
                "name": entry.name,
                "type": entry.input_type,
                "pool_path": entry.pool_path.name,
                "rows": entry.rows,
                "columns": entry.columns,
                "pool_mode": entry.pool_mode,
                "fingerprints": entry.fingerprints or [],
                "stage_a_sampling": entry.stage_a_sampling,
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
