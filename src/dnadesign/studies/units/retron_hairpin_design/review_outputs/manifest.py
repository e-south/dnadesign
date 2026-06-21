"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/review_outputs/manifest.py

Review manifest writer for Retron hairpin review outputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

from .plan import TetoReviewPlan
from .sequence_evidence import SequenceEvidenceSummary
from .sequence_index import SequenceReviewFrame


def write_review_manifest(
    *,
    plan: TetoReviewPlan,
    review_root: Path,
    materialized_root: Path,
    frames: Sequence[SequenceReviewFrame],
    pwm_svg: Path,
    pwm_png: Path,
    video_path: Path,
    video_manifest_path: Path,
    sequence_evidence: SequenceEvidenceSummary,
) -> Path:
    manifest_path = review_root / "reviews" / "review_manifest.json"
    sequence_index_path = materialized_root / "manifest" / "indexes" / "sequence_index.tsv"
    manifest = {
        "contract": "retron_hairpin_review_output_manifest_v1",
        "deliverable_plan_id": plan.deliverable_plan_id,
        "design_set_ref": _repo_relative(plan.design_set_path, plan_root=plan.plan_path),
        "deliverable_plan_ref": _repo_relative(plan.plan_path, plan_root=plan.plan_path),
        "materialized_sequence_rows": len(frames),
        "clone_handoff_verified_count": len(frames),
        "source_indexes": {
            "sequence_index": f"{materialized_root.name}/manifest/indexes/sequence_index.tsv",
            "sequence_index_sha256": _sha256(sequence_index_path),
        },
        "pwm_triptych": {
            "svg": _relative_to(pwm_svg, review_root),
            "png": _relative_to(pwm_png, review_root),
            "payload_trim_ids": [panel.payload_trim_id for panel in plan.pwm_panels],
            "meme_pwm_source": _repo_relative(plan.meme_pwm_path, plan_root=plan.plan_path),
        },
        "sequence_montage": {
            "mp4": _relative_to(video_path, review_root),
            "manifest": _relative_to(video_manifest_path, review_root),
            "frame_count": len(frames),
            "still_count": len(frames),
        },
        "sequence_evidence": sequence_evidence.as_manifest(),
        "clone_handoff": {
            "verified_count": len(frames),
            "required_fields": [
                "genbank",
                "reverse_complement_genbank",
                "forward_fasta",
                "reverse_complement_fasta",
                "features_csv",
            ],
        },
        "reader_boundary": {
            "status": "experiment_time_only",
            "note": "Reader owns SPOP and viability math; this manifest records pre-assay review outputs only.",
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_to(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _repo_relative(path: Path, *, plan_root: Path) -> str:
    for parent in plan_root.parents:
        if (parent / "pyproject.toml").is_file():
            try:
                return path.resolve().relative_to(parent).as_posix()
            except ValueError:
                break
    return path.as_posix()


__all__ = ["write_review_manifest"]
