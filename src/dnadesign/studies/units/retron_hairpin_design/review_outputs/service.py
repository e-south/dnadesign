"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/review_outputs/service.py

Service facade for Retron hairpin review-output generation.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..compiler.exceptions import RetronMsdCompilerError
from .contracts.manifest import write_review_manifest
from .contracts.plan import load_retron_review_plan
from .handoff.benchling import write_benchling_genbank_import
from .handoff.index import write_handoff_index
from .pwm.triptych import render_pwm_triptych
from .sequence.evidence import verify_sequence_evidence
from .sequence.index import load_validated_sequence_frames
from .video.montage import VideoWriter, write_sequence_montage


@dataclass(frozen=True)
class ReviewOutputResult:
    deliverable_plan_id: str
    review_root: Path
    pwm_triptych_svg: Path
    pwm_triptych_png: Path
    sequence_montage_mp4: Path
    sequence_montage_manifest: Path
    handoff_tsv: Path
    handoff_markdown: Path
    benchling_genbank_dir: Path
    benchling_genbank_index: Path
    benchling_genbank_count: int
    review_manifest_path: Path
    sequence_row_count: int
    handoff_verified_count: int
    reverse_complement_verified_count: int


def generate_retron_hairpin_review_outputs(
    *,
    deliverable_plan_path: Path,
    materialized_root: Path | None = None,
    out_dir: Path | None = None,
    repo_root: Path | None = None,
    video_writer: VideoWriter | None = None,
) -> ReviewOutputResult:
    resolved_repo_root = repo_root.resolve() if repo_root is not None else _find_repo_root(deliverable_plan_path)
    plan = load_retron_review_plan(deliverable_plan_path, repo_root=resolved_repo_root)
    resolved_root = (
        materialized_root.expanduser().resolve() if materialized_root is not None else plan.preferred_materialized_root
    )
    review_root = out_dir.expanduser().resolve() if out_dir is not None else plan.preferred_generated_root
    frames = load_validated_sequence_frames(resolved_root, plan=plan)
    sequence_evidence = verify_sequence_evidence(frames, materialized_root=resolved_root)
    handoff_index = write_handoff_index(
        frames,
        review_root=review_root,
        materialized_root=resolved_root,
        deliverable_plan_id=plan.deliverable_plan_id,
    )
    benchling_import = write_benchling_genbank_import(
        frames,
        review_root=review_root,
        materialized_root=resolved_root,
        deliverable_plan_id=plan.deliverable_plan_id,
        benchling_plan=plan.benchling_import,
    )
    pwm_svg, pwm_png = render_pwm_triptych(plan, out_dir=review_root)
    video_path, video_manifest_path = write_sequence_montage(
        frames,
        out_dir=review_root,
        deliverable_plan_id=plan.deliverable_plan_id,
        materialized_root=resolved_root,
        review_variant_ids=plan.review_variant_ids,
        video_writer=video_writer,
    )
    review_manifest_path = write_review_manifest(
        plan=plan,
        review_root=review_root,
        materialized_root=resolved_root,
        frames=frames,
        pwm_svg=pwm_svg,
        pwm_png=pwm_png,
        video_path=video_path,
        video_manifest_path=video_manifest_path,
        handoff_index=handoff_index,
        benchling_import=benchling_import,
        sequence_evidence=sequence_evidence,
    )
    return ReviewOutputResult(
        deliverable_plan_id=plan.deliverable_plan_id,
        review_root=review_root,
        pwm_triptych_svg=pwm_svg,
        pwm_triptych_png=pwm_png,
        sequence_montage_mp4=video_path,
        sequence_montage_manifest=video_manifest_path,
        handoff_tsv=handoff_index.tsv_path,
        handoff_markdown=handoff_index.markdown_path,
        benchling_genbank_dir=benchling_import.directory,
        benchling_genbank_index=benchling_import.index_tsv,
        benchling_genbank_count=len(benchling_import.files),
        review_manifest_path=review_manifest_path,
        sequence_row_count=len(frames),
        handoff_verified_count=len(frames),
        reverse_complement_verified_count=sequence_evidence.reverse_complement_verified_count,
    )


def _find_repo_root(path: Path) -> Path:
    current = path.expanduser().resolve()
    for candidate in (current, *current.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    raise RetronMsdCompilerError(f"Could not resolve repository root from Retron review plan path: {path}")


__all__ = [
    "ReviewOutputResult",
    "generate_retron_hairpin_review_outputs",
]
