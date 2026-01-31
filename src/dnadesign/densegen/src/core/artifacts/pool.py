"""
--------------------------------------------------------------------------------
<dnadesign project>
dnadesign/densegen/core/artifacts/pool.py

Stage-A TFBS pool artifacts.

Module Author(s): Eric J. South
Dunlop Lab
--------------------------------------------------------------------------------
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

from ...adapters.sources.stage_a_summary import PWMSamplingSummary
from ...config import resolve_relative_path
from ...core.stage_a_constants import FIMO_REPORT_THRESH
from ...utils.logging_utils import install_native_stderr_filters
from .ids import hash_tfbs_id

POOL_SCHEMA_VERSION = "1.6"
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


def _tier_scheme_label(fractions: Iterable[float]) -> str:
    values = []
    for frac in fractions:
        pct = float(frac) * 100.0
        if pct.is_integer():
            values.append(str(int(pct)))
        else:
            values.append(str(round(pct, 3)).rstrip("0").rstrip("."))
    label = "_".join(values) if values else "custom"
    return f"pct_{label}"


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
class PoolInputStatus:
    name: str
    state: str
    reason: str | None = None


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
        schema_version = str(payload.get("schema_version"))
        if schema_version != POOL_SCHEMA_VERSION:
            raise ValueError(
                f"Pool manifest schema_version={schema_version} is unsupported "
                f"(expected {POOL_SCHEMA_VERSION}). Rebuild pools with --fresh."
            )
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
            schema_version=schema_version,
            run_id=str(payload.get("run_id")),
            run_root=str(payload.get("run_root")),
            config_path=str(payload.get("config_path")),
            config_hash=payload.get("config_hash"),
        )

    def entry_for(self, input_name: str) -> PoolInputEntry:
        if input_name not in self.inputs:
            raise KeyError(f"Pool manifest missing input: {input_name}")
        return self.inputs[input_name]


def pool_status_by_input(cfg, cfg_path: Path, run_root: Path) -> dict[str, PoolInputStatus]:
    pool_dir = run_root / "outputs" / "pools"
    manifest_path = _pool_manifest_path(pool_dir)
    statuses: dict[str, PoolInputStatus] = {}
    inputs = list(cfg.inputs)
    if not manifest_path.exists():
        for inp in inputs:
            statuses[inp.name] = PoolInputStatus(name=inp.name, state="missing", reason="manifest_missing")
        return statuses

    try:
        artifact = TFBSPoolArtifact.load(manifest_path)
    except Exception as exc:
        for inp in inputs:
            statuses[inp.name] = PoolInputStatus(name=inp.name, state="stale", reason=f"manifest_invalid:{exc}")
        return statuses

    config_hash = _hash_pool_config(cfg)
    if not artifact.config_hash or artifact.config_hash != config_hash:
        for inp in inputs:
            statuses[inp.name] = PoolInputStatus(name=inp.name, state="stale", reason="config_changed")
        return statuses

    current_inputs = {inp.name for inp in inputs}
    stale_manifest = sorted(set(artifact.inputs) - current_inputs)
    if stale_manifest:
        for inp in inputs:
            statuses[inp.name] = PoolInputStatus(name=inp.name, state="stale", reason="inputs_changed")
        return statuses

    fingerprints_by_input: dict[str, list[dict]] = {}
    for inp in inputs:
        try:
            fingerprints_by_input[inp.name] = _resolve_input_fingerprints(cfg_path, inp)
        except Exception as exc:
            statuses[inp.name] = PoolInputStatus(name=inp.name, state="stale", reason=f"inputs_unreadable:{exc}")

    for inp in inputs:
        if inp.name in statuses:
            continue
        entry = artifact.inputs.get(inp.name)
        if entry is None:
            statuses[inp.name] = PoolInputStatus(name=inp.name, state="stale", reason="manifest_missing_entry")
            continue
        pool_path = pool_dir / entry.pool_path
        if not pool_path.exists():
            statuses[inp.name] = PoolInputStatus(name=inp.name, state="stale", reason="pool_file_missing")
            continue
        existing_fps = entry.fingerprints
        if existing_fps is None:
            statuses[inp.name] = PoolInputStatus(name=inp.name, state="stale", reason="fingerprints_missing")
            continue
        current_fps = fingerprints_by_input.get(inp.name, [])
        if _normalize_fingerprints(existing_fps) != _normalize_fingerprints(current_fps):
            statuses[inp.name] = PoolInputStatus(name=inp.name, state="stale", reason="inputs_changed")
            continue
        statuses[inp.name] = PoolInputStatus(name=inp.name, state="present", reason=None)
    return statuses


def _build_stage_a_sampling_manifest(
    summaries: list[object] | None,
    *,
    bgfile: str | None = None,
    bgfile_by_regulator: dict[str, str] | None = None,
) -> dict | None:
    if not summaries:
        return None
    pwm_summaries = [s for s in summaries if isinstance(s, PWMSamplingSummary)]
    if not pwm_summaries:
        return None
    fimo_summaries = [s for s in pwm_summaries if s.backend == "fimo"]
    if not fimo_summaries:
        return None
    eligible_score_hist = []
    bgfile_by_regulator = bgfile_by_regulator or {}
    reg_bgfiles = []
    uniqueness_values = {summary.uniqueness_key for summary in fimo_summaries}
    selection_policies = {summary.selection_policy for summary in fimo_summaries}
    selection_alphas = {summary.selection_alpha for summary in fimo_summaries}
    selection_similarity = {summary.selection_similarity for summary in fimo_summaries}
    selection_relevance_norm = {summary.selection_relevance_norm for summary in fimo_summaries}
    selection_pool_min_score_norm = {summary.selection_pool_min_score_norm_used for summary in fimo_summaries}
    selection_pool_cap_value = {summary.selection_pool_cap_value for summary in fimo_summaries}
    tier_target_fraction = {summary.tier_target_fraction for summary in fimo_summaries}
    tier_fractions_values: list[tuple[float, float, float]] = []
    tier_fractions_sources: list[str] = []
    for summary in fimo_summaries:
        if summary.eligible_score_hist_edges is None or summary.eligible_score_hist_counts is None:
            raise ValueError("Stage-A sampling summaries missing eligible score histogram.")
        if summary.tier_fractions is None:
            raise ValueError("Stage-A sampling summaries missing tier fractions.")
        if summary.tier_fractions_source is None:
            raise ValueError("Stage-A sampling summaries missing tier fractions source.")
        if summary.pwm_consensus_iupac is None:
            raise ValueError("Stage-A sampling summaries missing pwm_consensus_iupac.")
        if summary.pwm_consensus_score is None:
            raise ValueError("Stage-A sampling summaries missing pwm_consensus_score.")
        if summary.pwm_theoretical_max_score is None:
            raise ValueError("Stage-A sampling summaries missing pwm_theoretical_max_score.")
        tier_fractions_values.append(tuple(float(v) for v in summary.tier_fractions))
        tier_fractions_sources.append(str(summary.tier_fractions_source))
        if summary.eligible_score_hist_edges:
            if len(summary.eligible_score_hist_counts) != len(summary.eligible_score_hist_edges) - 1:
                raise ValueError("Stage-A eligible score histogram length mismatch.")
        if summary.candidates_with_hit is None or summary.eligible_raw is None:
            raise ValueError("Stage-A sampling summaries missing yield counters.")
        reg_bgfile = bgfile_by_regulator.get(summary.regulator, bgfile)
        reg_bgfiles.append(reg_bgfile)
        if summary.diversity is None:
            raise ValueError("Stage-A sampling summaries missing diversity metrics.")
        diversity_block = summary.diversity
        if hasattr(diversity_block, "to_dict"):
            diversity_block = diversity_block.to_dict()
        eligible_score_hist.append(
            {
                "regulator": summary.regulator,
                "pwm_consensus": summary.pwm_consensus,
                "pwm_consensus_iupac": summary.pwm_consensus_iupac,
                "pwm_consensus_score": summary.pwm_consensus_score,
                "pwm_theoretical_max_score": summary.pwm_theoretical_max_score,
                "edges": [float(v) for v in summary.eligible_score_hist_edges],
                "counts": [int(v) for v in summary.eligible_score_hist_counts],
                "tier0_score": summary.tier0_score,
                "tier1_score": summary.tier1_score,
                "tier2_score": summary.tier2_score,
                "tier_fractions": list(summary.tier_fractions),
                "tier_fractions_source": summary.tier_fractions_source,
                "bgfile": reg_bgfile,
                "background_source": "bgfile" if reg_bgfile else "motif_background",
                "generated": int(summary.generated),
                "candidates_with_hit": int(summary.candidates_with_hit),
                "eligible_raw": int(summary.eligible_raw),
                "eligible_unique": int(summary.eligible_unique),
                "retained": int(summary.retained),
                "target_tier_fraction": summary.tier_target_fraction,
                "required_unique_for_target_tier": summary.tier_target_required_unique,
                "target_tier_met": summary.tier_target_met,
                "selection_policy": summary.selection_policy,
                "selection_alpha": summary.selection_alpha,
                "selection_similarity": summary.selection_similarity,
                "selection_relevance_norm": summary.selection_relevance_norm,
                "selection_pool_size_final": summary.selection_pool_size_final,
                "selection_pool_rung_fraction_used": summary.selection_pool_rung_fraction_used,
                "selection_pool_min_score_norm_used": summary.selection_pool_min_score_norm_used,
                "selection_pool_capped": summary.selection_pool_capped,
                "selection_pool_cap_value": summary.selection_pool_cap_value,
                "selection_score_norm_max_raw": summary.selection_score_norm_max_raw,
                "selection_score_norm_clipped": summary.selection_score_norm_clipped,
                "collapsed_by_core_identity": summary.collapsed_by_core_identity,
                "diversity_nearest_distance_mean": summary.diversity_nearest_distance_mean,
                "diversity_nearest_distance_min": summary.diversity_nearest_distance_min,
                "diversity_nearest_similarity_mean": summary.diversity_nearest_similarity_mean,
                "eligible_score_norm_by_tier": summary.eligible_score_norm_by_tier,
                "diversity": diversity_block,
                "max_observed_score": summary.max_observed_score,
                "mining_audit": summary.mining_audit,
                "padding_audit": summary.padding_audit,
            }
        )
    unique_bgfiles = {val for val in reg_bgfiles}
    if len(unique_bgfiles) == 1:
        base_bgfile = reg_bgfiles[0]
        background_source = "bgfile" if base_bgfile else "motif_background"
    else:
        base_bgfile = None
        background_source = "mixed"
    if len(uniqueness_values) == 1:
        uniqueness_key = next(iter(uniqueness_values))
    else:
        uniqueness_key = "mixed"
    selection_policy = next(iter(selection_policies)) if len(selection_policies) == 1 else "mixed"
    selection_alpha_value = next(iter(selection_alphas)) if len(selection_alphas) == 1 else None
    selection_similarity_value = next(iter(selection_similarity)) if len(selection_similarity) == 1 else None
    selection_relevance_norm_value = (
        next(iter(selection_relevance_norm)) if len(selection_relevance_norm) == 1 else None
    )
    selection_pool_min_score_norm_value = (
        next(iter(selection_pool_min_score_norm)) if len(selection_pool_min_score_norm) == 1 else None
    )
    selection_pool_cap_value_value = (
        next(iter(selection_pool_cap_value)) if len(selection_pool_cap_value) == 1 else None
    )
    tier_target_fraction_value = next(iter(tier_target_fraction)) if len(tier_target_fraction) == 1 else None
    tier_fractions_value: list[float] | None = None
    tier_scheme_label = "mixed"
    if tier_fractions_values:
        unique_fractions = {vals for vals in tier_fractions_values}
        if len(unique_fractions) == 1:
            tier_fractions_value = list(next(iter(unique_fractions)))
            tier_scheme_label = _tier_scheme_label(tier_fractions_value)
    tier_fractions_source_value = None
    unique_sources = {src for src in tier_fractions_sources}
    if len(unique_sources) == 1:
        tier_fractions_source_value = next(iter(unique_sources))
    elif unique_sources:
        tier_fractions_source_value = "mixed"
    return {
        "backend": "fimo",
        "tier_scheme": tier_scheme_label,
        "tier_fractions": tier_fractions_value,
        "tier_fractions_source": tier_fractions_source_value,
        "eligibility_rule": "best_hit_score > 0 (and has at least one FIMO hit)",
        "retention_rule": "top_n_sites_by_best_hit_score",
        "fimo_thresh": FIMO_REPORT_THRESH,
        "uniqueness_key": uniqueness_key,
        "selection_policy": selection_policy,
        "selection_alpha": selection_alpha_value,
        "selection_similarity": selection_similarity_value,
        "selection_relevance_norm": selection_relevance_norm_value,
        "selection_pool_min_score_norm": selection_pool_min_score_norm_value,
        "selection_pool_cap_value": selection_pool_cap_value_value,
        "target_tier_fraction": tier_target_fraction_value,
        "bgfile": base_bgfile,
        "background_source": background_source,
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

        sampling_cfg = getattr(inp, "sampling", None)
        base_bgfile = getattr(sampling_cfg, "bgfile", None) if sampling_cfg is not None else None
        overrides = getattr(inp, "overrides_by_motif_id", None) or {}
        bgfile_by_regulator: dict[str, str] = {}
        for motif_id, override in overrides.items():
            override_bgfile = getattr(override, "bgfile", None)
            if override_bgfile is not None:
                bgfile_by_regulator[str(motif_id)] = str(override_bgfile)
        stage_a_sampling = _build_stage_a_sampling_manifest(
            summaries,
            bgfile=str(base_bgfile) if base_bgfile is not None else None,
            bgfile_by_regulator=bgfile_by_regulator or None,
        )
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
