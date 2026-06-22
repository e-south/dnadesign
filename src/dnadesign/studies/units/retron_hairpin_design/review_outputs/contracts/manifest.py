"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/review_outputs/contracts/manifest.py

Review manifest writer for Retron hairpin review outputs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

from ..handoff.benchling import BenchlingGenbankExport
from ..handoff.contract import SEQUENCE_HANDOFF_MANIFEST_KEY, SEQUENCE_HANDOFF_REQUIRED_FIELDS
from ..handoff.index import HandoffIndex
from ..sequence.evidence import SequenceEvidenceSummary
from ..sequence.index import SequenceReviewFrame
from .plan import TetoReviewPlan


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
    handoff_index: HandoffIndex,
    benchling_import: BenchlingGenbankExport,
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
        "handoff_verified_count": len(frames),
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
            "review_variant_ids": dict(plan.review_variant_ids),
        },
        "sequence_evidence": sequence_evidence.as_manifest(),
        SEQUENCE_HANDOFF_MANIFEST_KEY: {
            "verified_count": len(frames),
            "index_tsv": _relative_to(handoff_index.tsv_path, review_root),
            "index_markdown": _relative_to(handoff_index.markdown_path, review_root),
            "required_fields": list(SEQUENCE_HANDOFF_REQUIRED_FIELDS),
        },
        "benchling_genbank_import": {
            "orientation": plan.benchling_import.orientation,
            "included_payload_trim_ids": list(plan.benchling_import.included_payload_trim_ids),
            "assigned_retron_ids": dict(plan.benchling_import.assigned_retron_ids),
            "source_precedent_ids": dict(plan.benchling_import.source_precedent_ids),
            "verified_count": len(benchling_import.files),
            "expected_count": plan.benchling_import.expected_count,
            "directory": _relative_to(benchling_import.directory, review_root),
            "index_tsv": _relative_to(benchling_import.index_tsv, review_root),
            "files": [_relative_to(path, review_root) for path in benchling_import.files],
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
