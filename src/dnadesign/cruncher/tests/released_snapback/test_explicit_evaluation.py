"""
--------------------------------------------------------------------------------
<cruncher project>
src/dnadesign/cruncher/tests/released_snapback/test_explicit_evaluation.py

Focused tests for released explicit evaluation seams.

Module Author(s): Codex
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path

from dnadesign.cruncher.nickases.models import NickaseCatalog, NickaseCatalogEntry
from dnadesign.cruncher.release_enzymes.models import ReleaseEnzymeCatalog, ReleaseEnzymeEntry
from dnadesign.cruncher.snapback.released_explicit_evaluation import (
    AmbiguousPrecursorOriginError,
    build_released_explicit_report,
    infer_precursor_coordinate_offset,
)
from dnadesign.cruncher.snapback.released_models import (
    SingleNickReleasedSnapbackSpec,
)


def _spec(*, precursor_top_strand: str, release_variant_id: str = "Re.Exact") -> SingleNickReleasedSnapbackSpec:
    return SingleNickReleasedSnapbackSpec.model_validate(
        {
            "released_snapback": {
                "schema_version": 1,
                "kind": "single_nick_released_snapback_v1",
                "name": "demo_released",
            },
            "input": {"precursor_top_strand": precursor_top_strand},
            "nick_stage": {
                "nickase_variant_id": "Nx.Exact7",
                "catalog": {"preset": "local"},
            },
            "release_stage": {
                "release_variant_id": release_variant_id,
                "catalog": {"preset": "local_release"},
                "retained_side": "upstream",
                "stage_order": "nick_then_release",
            },
            "final_target": {"nick_boundary_from_left": 0, "paired_bp": 3, "cap_nt": 3},
        }
    )


def _nick_entry() -> NickaseCatalogEntry:
    return NickaseCatalogEntry(
        id="Nx.Exact7",
        specificity_id="Nx.Exact7",
        motif_top_5to3="AACGTTG",
        top_cut_offset=0,
    )


def _release_entry() -> ReleaseEnzymeEntry:
    return ReleaseEnzymeEntry(
        variant_id="Re.Exact",
        display_name="Re.Exact",
        recognition_sequence="CCAA",
        top_cut_offset=1,
        bottom_cut_offset=0,
        class_label="other_ds_re",
        commercial_confidence="primary_vendor_current",
        source_catalog_id="local_release",
    )


def test_infer_precursor_coordinate_offset_returns_single_anchor_offset() -> None:
    spec = _spec(precursor_top_strand="TTAACGTTGTTCCAA")

    offset = infer_precursor_coordinate_offset(spec, nick_entry=_nick_entry())

    assert offset == 2


def test_infer_precursor_coordinate_offset_raises_on_ambiguous_primary_anchors() -> None:
    spec = _spec(precursor_top_strand="AACGTTGAACGTTGTTCCAA")

    try:
        infer_precursor_coordinate_offset(spec, nick_entry=_nick_entry())
    except AmbiguousPrecursorOriginError as exc:
        assert exc.variant_id == "Nx.Exact7"
        assert exc.target_boundary == 0
        assert exc.offsets == [0, 7]
    else:
        raise AssertionError("Expected ambiguous released-product precursor origin.")


def test_build_released_explicit_report_rejects_unknown_release_variant() -> None:
    spec = _spec(precursor_top_strand="AACGTTGTTCCAA", release_variant_id="Re.Missing")

    report = build_released_explicit_report(
        spec,
        spec_path=Path("/tmp/demo.released.snapback.yaml"),
        workspace_root=Path("/tmp/workspace"),
        nick_catalog=NickaseCatalog(preset_id="local", preset_ids=["local"], entries=[_nick_entry()]),
        release_catalog=ReleaseEnzymeCatalog(
            preset_id="local_release",
            preset_ids=["local_release"],
            entries=[_release_entry()],
        ),
        nick_catalog_source="local",
        release_catalog_source="local_release",
    )

    assert report.status == "invalid_catalog"
    assert report.issues[0].code == "UNKNOWN_RELEASE_VARIANT_ID"
