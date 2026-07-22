"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/communication_visuals/proposal_backbone_cycle.py

ChimeraX movie cycling fitted ProteinMPNN proposals over the Eco1 RT reference.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from numbers import Real
from pathlib import Path
from typing import Any

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    SECTION_DESIGNS_AND_FOLD_TRIAGE,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.manifest import (
    file_hashes,
    make_deliverable_row,
)

from .catalog import (
    COMMUNICATION_ROLE,
    PROPOSAL_BACKBONE_CYCLE_FRAME_DIRECTORY_NAME,
    PROPOSAL_BACKBONE_CYCLE_LOG_FILE_NAME,
    PROPOSAL_BACKBONE_CYCLE_MOVIE_FILE_NAME,
    PROPOSAL_BACKBONE_CYCLE_MOVIE_ID,
    PROPOSAL_BACKBONE_CYCLE_RENDER_MANIFEST_FILE_NAME,
    PROPOSAL_BACKBONE_CYCLE_SCRIPT_FILE_NAME,
    PROPOSAL_BACKBONE_CYCLE_SCRIPT_ID,
)
from .movie_runtime import MovieRenderSpec, materialize_chimerax_movie
from .pose import (
    CHIMERAX_MOVIE_FRAME_RATE,
    CHIMERAX_MOVIE_HEIGHT,
    CHIMERAX_MOVIE_WIDTH,
    CHIMERAX_START_ORIENTATION_OFFSET_DEGREES,
)
from .proposal_backbone_script import (
    POLICY_LABELS as _POLICY_LABELS,
)
from .proposal_backbone_script import (
    PROPOSAL_ENCODED_FRAMES_PER_RAW_FRAME,
    ProposalBackboneChapter,
    ProposalBackboneScene,
    build_proposal_backbone_chapters,
    proposal_backbone_raw_frame_count,
    write_proposal_backbone_cycle_script,
)
from .proposal_backbone_script import (
    PROPOSAL_RAW_FRAMES_PER_FULL_ROTATION as _PROPOSAL_RAW_FRAMES_PER_FULL_ROTATION,
)
from .proposal_backbone_script import (
    SCENE_ORDER as _SCENE_ORDER,
)
from .structure_set import FoldcheckStructureSet

CANONICAL_RT_LENGTH = 320
PROPOSAL_RAW_FRAMES_PER_FULL_ROTATION = _PROPOSAL_RAW_FRAMES_PER_FULL_ROTATION
_RENDER_SPEC = MovieRenderSpec(
    schema_id="eco1_rt.communication_proposal_backbone_cycle_render",
    schema_version=6,
    renderer="ChimeraX centered retained proposal-backbone 16:9 PNG saves",
    output_key="proposal_backbone_cycle_movie",
    frame_width=CHIMERAX_MOVIE_WIDTH,
    frame_height=CHIMERAX_MOVIE_HEIGHT,
    frame_rate=CHIMERAX_MOVIE_FRAME_RATE,
    frames_per_scene=1,
    hold_frames_per_scene=1,
    timeout_seconds=3600,
)


def write_proposal_backbone_cycle(
    *,
    panel_root: Path,
    triage_rows: list[dict[str, Any]],
    triage_table_path: Path,
    structure_set: FoldcheckStructureSet,
    foldcheck_full_structure_set_path: Path,
    reference_backbone_path: Path,
    render_requested: bool,
) -> list[dict[str, Any]]:
    """Write the retained-proposal backbone script and optional requested movie."""

    panel_root.mkdir(parents=True, exist_ok=True)
    scenes = build_proposal_backbone_scenes(triage_rows=triage_rows, structure_set=structure_set)
    script_path = panel_root / PROPOSAL_BACKBONE_CYCLE_SCRIPT_FILE_NAME
    movie_path = panel_root / PROPOSAL_BACKBONE_CYCLE_MOVIE_FILE_NAME
    render_manifest_path = panel_root / PROPOSAL_BACKBONE_CYCLE_RENDER_MANIFEST_FILE_NAME
    frame_directory = panel_root / PROPOSAL_BACKBONE_CYCLE_FRAME_DIRECTORY_NAME
    write_proposal_backbone_cycle_script(
        script_path=script_path,
        reference_backbone_path=reference_backbone_path,
        scenes=scenes,
        frame_directory=frame_directory,
    )
    source_paths = {
        "candidate_triage_table": triage_table_path,
        "foldcheck_full_structure_set": foldcheck_full_structure_set_path,
        "cryoem_reference_backbone": reference_backbone_path,
    }
    render_status, render_reason = materialize_chimerax_movie(
        script_path=script_path,
        movie_path=movie_path,
        frame_directory=frame_directory,
        render_manifest_path=render_manifest_path,
        log_path=panel_root / PROPOSAL_BACKBONE_CYCLE_LOG_FILE_NAME,
        source_paths=source_paths,
        render_requested=render_requested,
        spec=_RENDER_SPEC,
        expected_raw_frame_count=proposal_backbone_raw_frame_count(scenes),
        encoding_metadata={
            "encoded_frames_per_raw_capture": PROPOSAL_ENCODED_FRAMES_PER_RAW_FRAME,
            "mapped_residue_count": 309,
            "starting_orientation_offset_degrees": CHIMERAX_START_ORIENTATION_OFFSET_DEGREES,
            "layout": "centered_local_geometry_retained_stream",
            "maximum_open_atomic_models": 2,
        },
    )
    input_hashes = file_hashes({**source_paths, "movie_script": script_path})
    source_tables = [
        "generation_policies_v3/selection/candidate_triage_table.parquet",
        "generation_policies_v3/foldcheck_review/foldcheck_full_structure_set.yaml",
        "generation_policies_v3/foldcheck_review/structures/full_fold_set/*.pdb",
    ]
    chapter_counts: dict[str, int] = {}
    for scene in scenes:
        chapter_counts[scene.chapter_label] = chapter_counts.get(scene.chapter_label, 0) + 1
    excluded_count = len(triage_rows) - len(scenes)
    mutation_counts = [scene.mutation_count for scene in scenes]
    identity_values = [scene.wt_sequence_identity_percent for scene in scenes]
    script_row = make_deliverable_row(
        deliverable_id=PROPOSAL_BACKBONE_CYCLE_SCRIPT_ID,
        section=SECTION_DESIGNS_AND_FOLD_TRIAGE,
        artifact_kind="chimerax_script",
        status="rendered",
        path=script_path,
        source_tables=source_tables,
        input_hashes=input_hashes,
        alt_text=(
            "ChimeraX script that overlays the ColabFold model for every ProteinMPNN-generated sequence on one "
            "RT reference backbone."
        ),
        description=(
            "Opens, aligns, captures, and closes one candidate at a time over the shared 309-residue C-alpha map."
        ),
        interpretation_limit="The script controls presentation and does not alter local RMSD or panel selection.",
        title="ColabFold model-cycle movie script",
        role="operator_review",
    )
    movie_row = make_deliverable_row(
        deliverable_id=PROPOSAL_BACKBONE_CYCLE_MOVIE_ID,
        section=SECTION_DESIGNS_AND_FOLD_TRIAGE,
        artifact_kind="video",
        status=render_status if movie_path.exists() else _missing_output_status(render_status),
        path=movie_path,
        source_tables=[*source_tables, PROPOSAL_BACKBONE_CYCLE_SCRIPT_FILE_NAME],
        input_hashes=input_hashes,
        alt_text=(
            f"Rapid centered ChimeraX sequence covering all {len(scenes):,} ColabFold models retained by the "
            "declared local-geometry review. Each dark blue-gray candidate cartoon and its complete modeled side "
            "chains rotate over one off-white cryo-EM reference cartoon. Frame labels report full-length WT "
            "identity and substitution count; policy chapters distinguish distal, peripheral, and combined redesigns."
        ),
        description=(
            "Shows every local-geometry-retained model without subsampling. Each model is opened, aligned over the "
            "shared mapped residue span, captured, and closed before the next model is loaded. The candidate layer "
            "shows its complete modeled side chains, not mutation-only highlights. Every frame reports full-length "
            "WT sequence identity and substitution count."
        ),
        interpretation_limit=(
            "Rapid overlap is qualitative. Quantitative structural retention comes from recorded local C-alpha "
            "RMSD, and the movie does not predict function."
        ),
        title="Local-geometry-retained ColabFold models align to the RT reference",
        role=COMMUNICATION_ROLE,
        render_mode="wide_visual",
        skip_reason="" if movie_path.exists() else render_reason,
        evidence_summary={
            "source_candidate_count": len(triage_rows),
            "rendered_candidate_count": len(scenes),
            "local_geometry_retained_count": len(scenes),
            "excluded_local_geometry_threshold_count": excluded_count,
            "chapter_counts": chapter_counts,
            "mapped_residue_count": 309,
            "mutation_count_range": [min(mutation_counts), max(mutation_counts)],
            "wt_sequence_identity_percent_range": [min(identity_values), max(identity_values)],
        },
    )
    return [script_row, movie_row]


def build_proposal_backbone_scenes(
    *,
    triage_rows: list[dict[str, Any]],
    structure_set: FoldcheckStructureSet,
) -> tuple[ProposalBackboneScene, ...]:
    """Join retained triage rows to models and order plain-language chapters."""

    rows_by_id: dict[str, dict[str, Any]] = {}
    for row in triage_rows:
        candidate_id = str(row.get("candidate_id") or "").strip()
        if not candidate_id:
            raise ValueError("Proposal triage row is missing candidate_id")
        if candidate_id in rows_by_id:
            raise ValueError(f"Proposal triage rows contain duplicate candidate_id: {candidate_id}")
        rows_by_id[candidate_id] = row
    structure_by_id = structure_set.candidate_by_id
    if rows_by_id.keys() != structure_by_id.keys():
        missing_structures = sorted(rows_by_id.keys() - structure_by_id.keys())
        missing_triage = sorted(structure_by_id.keys() - rows_by_id.keys())
        raise ValueError(
            "Proposal triage and structure-set candidate IDs differ: "
            f"missing structures={missing_structures[:5]!r}; missing triage={missing_triage[:5]!r}"
        )

    grouped: dict[str, list[str]] = {policy_id: [] for policy_id in _SCENE_ORDER}
    for candidate_id, row in rows_by_id.items():
        policy_id = str(row.get("policy_id") or "")
        if policy_id not in _POLICY_LABELS:
            raise ValueError(f"Unsupported proposal generation policy for {candidate_id}: {policy_id!r}")
        gate_status = str(row.get("local_structure_gate_status") or "")
        if gate_status not in {"passed", "threshold_exceeded"}:
            raise ValueError(f"Unknown local-geometry status for {candidate_id}: {gate_status!r}")
        mutation_count = _required_mutation_count(row, candidate_id=candidate_id)
        structure_identity = structure_by_id[candidate_id].full_sequence_identity_percent
        expected_identity = 100.0 * (CANONICAL_RT_LENGTH - mutation_count) / CANONICAL_RT_LENGTH
        if abs(structure_identity - expected_identity) > 1e-9:
            raise ValueError(
                "WT identity and mutation count disagree for "
                f"{candidate_id}: structure manifest={structure_identity}, expected={expected_identity}"
            )
        if gate_status == "passed":
            grouped[policy_id].append(candidate_id)

    scenes: list[ProposalBackboneScene] = []
    for policy_id in _SCENE_ORDER:
        candidate_ids = sorted(grouped[policy_id])
        if not candidate_ids:
            continue
        chapter_label = _POLICY_LABELS[policy_id]
        for position, candidate_id in enumerate(candidate_ids, start=1):
            row = rows_by_id[candidate_id]
            scenes.append(
                ProposalBackboneScene(
                    candidate_id=candidate_id,
                    structure_path=structure_by_id[candidate_id].path,
                    policy_id=policy_id,
                    chapter_label=chapter_label,
                    chapter_position=position,
                    chapter_size=len(candidate_ids),
                    mutation_count=_required_mutation_count(row, candidate_id=candidate_id),
                    wt_sequence_identity_percent=structure_by_id[candidate_id].full_sequence_identity_percent,
                )
            )
    return tuple(scenes)


def _required_mutation_count(row: dict[str, Any], *, candidate_id: str) -> int:
    value = row.get("mutation_count_total")
    if isinstance(value, bool) or not isinstance(value, Real) or not float(value).is_integer():
        raise ValueError(f"Proposal triage row {candidate_id} requires integer mutation_count_total")
    mutation_count = int(value)
    if not 0 <= mutation_count <= CANONICAL_RT_LENGTH:
        raise ValueError(f"Proposal triage row {candidate_id} has out-of-range mutation_count_total: {value}")
    sequence_distance = row.get("sequence_distance_to_wt")
    if sequence_distance != mutation_count:
        raise ValueError(
            f"Proposal triage row {candidate_id} has inconsistent sequence_distance_to_wt: "
            f"{sequence_distance!r} versus {mutation_count} substitutions"
        )
    return mutation_count


def _missing_output_status(render_status: str) -> str:
    return render_status if render_status != "rendered" else "errored"


__all__ = [
    "PROPOSAL_RAW_FRAMES_PER_FULL_ROTATION",
    "ProposalBackboneChapter",
    "ProposalBackboneScene",
    "build_proposal_backbone_chapters",
    "build_proposal_backbone_scenes",
    "proposal_backbone_raw_frame_count",
    "write_proposal_backbone_cycle",
    "write_proposal_backbone_cycle_script",
]
