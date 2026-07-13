"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/operations/materialization/review_deliverables/communication_visuals/structure_story.py

Materialize the browser and ChimeraX Eco1 structure-story artifacts.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.constants import (
    SECTION_CONSTRAINT_EVIDENCE,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.manifest import (
    file_hashes,
    make_deliverable_row,
)

from ..structure_browser_common import (
    reference_residue_number_by_canonical,
    reference_selection_coordinate_basis,
)
from .catalog import (
    COMMUNICATION_ROLE,
    STRUCTURE_STORY_BROWSER_ID,
    STRUCTURE_STORY_MOVIE_FILE_NAME,
    STRUCTURE_STORY_MOVIE_ID,
    STRUCTURE_STORY_RENDER_MANIFEST_FILE_NAME,
    STRUCTURE_STORY_SCRIPT_FILE_NAME,
    STRUCTURE_STORY_SCRIPT_ID,
)
from .chimerax_story import (
    CHIMERAX_FRAME_DIRECTORY_NAME,
    materialize_chimerax_outputs,
    write_chimerax_script,
)
from .structure_browser_story import build_structure_browser_payload
from .structure_scenes import structure_scene_specs

BROWSER_MANIFEST_FILE_NAME = "structure_story_browser_manifest.yaml"


def write_structure_story(
    *,
    panel_root: Path,
    reference_structure_path: Path,
    reference_structure_format: str,
    mask_residues: list[dict[str, Any]],
    policy_position_rows: list[dict[str, Any]],
    mask_set_path: Path,
    policy_positions_path: Path,
    render_chimerax: bool,
) -> list[dict[str, Any]]:
    """Write interactive structure scenes plus the optional rotation movie."""

    panel_root.mkdir(parents=True, exist_ok=True)
    if not reference_structure_path.exists():
        raise FileNotFoundError(reference_structure_path)
    reference_number_by_canonical = reference_residue_number_by_canonical(
        mask_residues,
        reference_structure_format=reference_structure_format,
    )
    coordinate_basis = reference_selection_coordinate_basis(reference_structure_format=reference_structure_format)
    scene_specs = structure_scene_specs(policy_position_rows)

    browser_manifest_path = panel_root / BROWSER_MANIFEST_FILE_NAME
    browser_payload = build_structure_browser_payload(
        reference_structure_path=reference_structure_path,
        reference_structure_format=reference_structure_format,
        manifest_root=browser_manifest_path.parent,
        reference_number_by_canonical=reference_number_by_canonical,
        selection_coordinate_basis=coordinate_basis,
        scene_specs=scene_specs,
        mask_set_path=mask_set_path,
        policy_positions_path=policy_positions_path,
    )
    browser_manifest_path.write_text(yaml.safe_dump(browser_payload, sort_keys=False), encoding="utf-8")

    script_path = panel_root / STRUCTURE_STORY_SCRIPT_FILE_NAME
    movie_path = panel_root / STRUCTURE_STORY_MOVIE_FILE_NAME
    frame_directory = panel_root / CHIMERAX_FRAME_DIRECTORY_NAME
    render_manifest_path = panel_root / STRUCTURE_STORY_RENDER_MANIFEST_FILE_NAME
    write_chimerax_script(
        script_path=script_path,
        reference_structure_path=reference_structure_path,
        frame_directory=frame_directory,
        reference_number_by_canonical=reference_number_by_canonical,
        scene_specs=scene_specs,
    )
    render_status, render_reason = materialize_chimerax_outputs(
        script_path=script_path,
        reference_structure_path=reference_structure_path,
        movie_path=movie_path,
        frame_directory=frame_directory,
        render_manifest_path=render_manifest_path,
        render_chimerax=render_chimerax,
    )
    source_paths = {
        "reference_structure": reference_structure_path,
        "mask_set": mask_set_path,
        "generation_policy_positions": policy_positions_path,
    }
    return [
        make_deliverable_row(
            deliverable_id=STRUCTURE_STORY_BROWSER_ID,
            section=SECTION_CONSTRAINT_EVIDENCE,
            artifact_kind="structure_browser_manifest",
            status="rendered",
            path=browser_manifest_path,
            source_tables=[
                "mask_set.yaml",
                "generation_policies_v3/generation_policy_positions.parquet",
                "foldcheck_review/structures/ec86kit_protomer1_all_atom_reference.pdb",
            ],
            input_hashes=file_hashes(source_paths),
            alt_text=(
                "Interactive Ec86 RT-DNA-RNA structure story with a translucent protein surface and separate "
                "views for NAxxH/YADD/VTG contexts, direct contacts, conserved positions, primer-recognition context, "
                "the protected union, and the three design spaces."
            ),
            description=(
                "Uses one retained-complex structure and one camera memory key so mask evidence and design spaces "
                "can be inspected without conflating overlapping residue categories."
            ),
            interpretation_limit=(
                "The scenes show declared residue sets on the reference structure. They do not measure activity "
                "or electrostatic binding energy."
            ),
            title="The retained complex separates protected evidence from design space",
            role=COMMUNICATION_ROLE,
            method_summary=(
                "Each view resets to the same RT-DNA-RNA reference. The protein uses a translucent VDW surface; "
                "DNA and RNA retain gold and salmon nucleic-acid ribbons with matching ladder rungs."
            ),
            evidence_summary={"scene_count": len(browser_payload["structures"])},
        ),
        make_deliverable_row(
            deliverable_id=STRUCTURE_STORY_SCRIPT_ID,
            section=SECTION_CONSTRAINT_EVIDENCE,
            artifact_kind="chimerax_script",
            status="rendered",
            path=script_path,
            source_tables=[
                "mask_set.yaml",
                "generation_policies_v3/generation_policy_positions.parquet",
                "foldcheck_review/structures/ec86kit_protomer1_all_atom_reference.pdb",
            ],
            input_hashes=file_hashes({**source_paths, "structure_story_script": script_path}),
            alt_text="ChimeraX command script for the Eco1 RT protected-evidence and design-space movie.",
            description="Provides a reproducible command path for the protected-evidence rotation movie.",
            interpretation_limit="The script controls presentation only and does not alter study evidence.",
            title="ChimeraX can reproduce the structure story",
            role="operator_review",
        ),
        _optional_render_row(
            deliverable_id=STRUCTURE_STORY_MOVIE_ID,
            artifact_kind="video",
            path=movie_path,
            status=render_status if movie_path.exists() else _missing_output_status(render_status),
            skip_reason="" if movie_path.exists() else render_reason,
            source_paths={**source_paths, "structure_story_script": script_path},
            title="Protected evidence and design spaces are revealed on one structure",
            alt_text=(
                "ChimeraX rotation movie that gives each motif, contact, conservation, primer-recognition, protected-"
                "union, and design-space category a full five-second turn. A translucent protein surface reveals the "
                "gray cartoon, and only the active category's side chains are shown."
            ),
            description=(
                "Presents the mask logic as ordered full-turn views from one structure and controlled camera path."
            ),
            interpretation_limit=(
                "The movie communicates residue-set definitions. It does not show molecular dynamics or function."
            ),
        ),
    ]


def _missing_output_status(render_status: str) -> str:
    return render_status if render_status != "rendered" else "errored"


def _optional_render_row(
    *,
    deliverable_id: str,
    artifact_kind: str,
    path: Path,
    status: str,
    skip_reason: str,
    source_paths: dict[str, Path],
    title: str,
    alt_text: str,
    description: str,
    interpretation_limit: str,
) -> dict[str, Any]:
    return make_deliverable_row(
        deliverable_id=deliverable_id,
        section=SECTION_CONSTRAINT_EVIDENCE,
        artifact_kind=artifact_kind,
        status=status,
        path=path,
        source_tables=[
            "mask_set.yaml",
            "generation_policies_v3/generation_policy_positions.parquet",
            "foldcheck_review/structures/ec86kit_protomer1_all_atom_reference.pdb",
            STRUCTURE_STORY_SCRIPT_FILE_NAME,
        ],
        input_hashes=file_hashes(source_paths),
        alt_text=alt_text,
        description=description,
        interpretation_limit=interpretation_limit,
        title=title,
        role=COMMUNICATION_ROLE,
        render_mode="wide_visual",
        skip_reason=skip_reason,
    )
