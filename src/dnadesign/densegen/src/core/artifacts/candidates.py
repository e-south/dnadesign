"""
Stage-A candidate site artifacts.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from ...utils.logging_utils import install_native_stderr_filters

CANDIDATE_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class CandidateArtifact:
    manifest_path: Path
    candidates_path: Path
    summary_path: Path
    schema_version: str
    run_id: str
    run_root: str
    config_path: str


def _manifest_path(out_dir: Path) -> Path:
    return out_dir / "candidates_manifest.json"


def _summary_path(out_dir: Path) -> Path:
    return out_dir / "candidates_summary.parquet"


def _candidates_path(out_dir: Path) -> Path:
    return out_dir / "candidates.parquet"


def find_candidate_files(candidates_dir: Path) -> list[Path]:
    if not candidates_dir.exists():
        return []
    return sorted(candidates_dir.rglob("candidates__*.parquet"))


def prepare_candidates_dir(candidates_dir: Path, *, overwrite: bool = True) -> bool:
    existed = candidates_dir.exists()
    if existed:
        if not overwrite:
            raise FileExistsError(f"Candidate artifacts already exist in {candidates_dir}")
        shutil.rmtree(candidates_dir)
    candidates_dir.mkdir(parents=True, exist_ok=True)
    return existed


def build_candidate_artifact(
    *,
    candidates_dir: Path,
    cfg_path: Path,
    run_id: str,
    run_root: Path,
    overwrite: bool = False,
) -> CandidateArtifact:
    install_native_stderr_filters(suppress_solver_messages=False)
    candidates_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = _manifest_path(candidates_dir)
    candidates_path = _candidates_path(candidates_dir)
    summary_path = _summary_path(candidates_dir)

    if not overwrite:
        if candidates_path.exists() or summary_path.exists() or manifest_path.exists():
            raise FileExistsError(f"Candidate artifacts already exist in {candidates_dir}")

    files = find_candidate_files(candidates_dir)
    if not files:
        raise FileNotFoundError(f"No candidate parquet files found under {candidates_dir}")

    frames = []
    for path in files:
        try:
            frames.append(pd.read_parquet(path))
        except Exception as exc:
            raise RuntimeError(f"Failed to read candidate parquet: {path}") from exc
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    required = {"input_name", "motif_id", "motif_label", "scoring_backend", "sequence", "run_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(
            "Candidate artifacts missing required columns "
            f"{missing}. Clear outputs/pools/candidates and re-run with keep_all_candidates_debug."
        )
    df = df[df["run_id"] == str(run_id)].copy()
    if df.empty:
        raise ValueError(
            "No candidate records match the current run_id. "
            "Clear outputs/pools/candidates and re-run with keep_all_candidates_debug."
        )
    df.to_parquet(candidates_path, index=False)

    summary = pd.DataFrame(
        columns=[
            "input_name",
            "motif_id",
            "motif_label",
            "scoring_backend",
            "total_candidates",
            "accepted",
            "selected",
            "rejected",
        ]
    )
    if not df.empty:
        grouped = df.groupby(["input_name", "motif_id", "motif_label", "scoring_backend"], dropna=False)
        rows = []
        for (input_name, motif_id, motif_label, scoring_backend), sub in grouped:
            total = int(len(sub))
            accepted = int(sub["accepted"].sum()) if "accepted" in sub.columns else 0
            selected = int(sub["selected"].sum()) if "selected" in sub.columns else 0
            rejected = total - accepted
            rows.append(
                {
                    "input_name": str(input_name),
                    "motif_id": str(motif_id),
                    "motif_label": str(motif_label),
                    "scoring_backend": str(scoring_backend),
                    "total_candidates": total,
                    "accepted": accepted,
                    "selected": selected,
                    "rejected": rejected,
                }
            )
        summary = pd.DataFrame(rows)
    summary.to_parquet(summary_path, index=False)

    manifest = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_id": str(run_id),
        "run_root": str(run_root),
        "config_path": str(cfg_path),
        "candidates_path": str(candidates_path),
        "summary_path": str(summary_path),
        "source_files": [str(path) for path in files],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return CandidateArtifact(
        manifest_path=manifest_path,
        candidates_path=candidates_path,
        summary_path=summary_path,
        schema_version=CANDIDATE_SCHEMA_VERSION,
        run_id=str(run_id),
        run_root=str(run_root),
        config_path=str(cfg_path),
    )
