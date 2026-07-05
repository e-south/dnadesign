"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/review_deliverables/test_visual_content.py

Eco1 review-deliverable visual content tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

import yaml

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables import (
    design_class_mask_annotations,
    materialize_review_deliverables,
)
from dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.design_class_masks import (
    _MASK_EMPTY_STATE_ALPHA,
    _MASK_MATRIX_ZORDER,
    write_design_class_mask_overview,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.chimerax_assertions import (
    assert_chimerax_context_scripts,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.fixtures import (
    write_deliverable_inputs,
    write_rt_annotation_context_sources,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.notebook_assertions import (
    assert_review_notebook_contract,
)
from dnadesign.studies.units.eco1_rt_repack.tests.materialization.review_deliverables.visual_content_assertions import (
    assert_linked_fold_and_esmc_content,
    assert_mask_and_msa_content,
    assert_selection_content,
)

from .proteinmpnn_visual_assertions import assert_proteinmpnn_visual_content

_RT_CONTEXT_MODULE = (
    "dnadesign.studies.units.eco1_rt_repack.operations.materialization.review_deliverables.rt_annotation_context"
)
load_rt_annotation_context = import_module(_RT_CONTEXT_MODULE).load_rt_annotation_context


def test_review_deliverable_visual_content_is_plain_and_linked(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)

    result = materialize_review_deliverables(repo_root=Path.cwd(), output_root=tmp_path)

    manifest = yaml.safe_load(result.manifest_path.read_text(encoding="utf-8"))
    deliverables = {entry["deliverable_id"]: entry for entry in manifest["deliverables"]}
    assert_mask_and_msa_content(result.manifest_path, deliverables)
    assert_proteinmpnn_visual_content(result.manifest_path, deliverables)
    assert_linked_fold_and_esmc_content(result.manifest_path, deliverables)
    assert_selection_content(deliverables)
    assert_chimerax_context_scripts(
        manifest_path=result.manifest_path,
        deliverables=deliverables,
        forbidden_path_text=str(tmp_path),
    )
    assert_review_notebook_contract(result.notebook_path.read_text(encoding="utf-8"))


def test_design_class_mask_overview_renders_rt_context_spans_from_ontology(tmp_path: Path) -> None:
    write_deliverable_inputs(tmp_path)
    annotation_tracks_path, manual_authority_path = write_rt_annotation_context_sources(tmp_path)
    rt_annotation_context = load_rt_annotation_context(
        annotation_tracks_path=annotation_tracks_path,
        manual_mask_authority_source_path=manual_authority_path,
    )

    deliverable = write_design_class_mask_overview(
        panel_root=tmp_path / "review_deliverables" / "mask_structure_context",
        baseline_mask_set_path=tmp_path / "mask_set.yaml",
        design_classes_root=tmp_path / "design_classes",
        rt_annotation_context=rt_annotation_context,
    )

    svg_path = tmp_path / "review_deliverables" / "mask_structure_context" / "design_class_mask_overview.svg"
    svg_text = svg_path.read_text(encoding="utf-8")
    assert "RT1" in svg_text
    assert "RT2" in svg_text
    assert "Region X local context" in svg_text
    assert "Catalytic YADD local context" in svg_text
    assert "NAxxH" in svg_text
    assert "YADD" in svg_text
    assert "rt_annotation_tracks" in deliverable["input_hashes"]
    assert "manual_mask_authority_source" in deliverable["input_hashes"]
    assert "docs/studies/eco1_rt_repack/workbench/ontology/rt-annotation-tracks.yaml" in deliverable["source_tables"]
    assert "docs/studies/eco1_rt_repack/workbench/ontology/manual-mask-authority.yaml" in deliverable["source_tables"]


def test_design_class_rt_context_spans_stay_behind_mask_features() -> None:
    assert design_class_mask_annotations.MASK_ANNOTATION_SPAN_ZORDER < _MASK_MATRIX_ZORDER
    assert _MASK_EMPTY_STATE_ALPHA == 0.0
