"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/compiler/test_msd_unit.py

Single-unit MSD sequence compiler tests.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from Bio.Seq import Seq

from dnadesign.studies.units.retron_hairpin_design.catalog.compiler_spec import load_msd_compiler_spec
from dnadesign.studies.units.retron_hairpin_design.compiler.msd_unit import compile_msd_design_unit

from ..support.compiler_fixtures import SNAPBACK_FOLDBACK, TETO_PAYLOAD
from ..support.paths import repo_root_from


def test_retron_msd_compiled_unit_api_exposes_parts_without_materialization() -> None:
    repo_root = repo_root_from(__file__)
    resolved = load_msd_compiler_spec(
        repo_root / "docs/studies/retron_hairpin_design/compiler/inputs/msd_design_177_194_cap_sources_spec.yaml",
        study_dir=repo_root / "docs/studies/retron_hairpin_design",
    )
    record = next(item for item in resolved.catalog.records if item.construct_id == "pES-retron-179")

    unit = compile_msd_design_unit(
        record,
        payload_sequences=resolved.payload_sequences,
        cap_sequences=resolved.cap_sequences,
    )

    expected_payload = TETO_PAYLOAD.upper()
    expected_cap = SNAPBACK_FOLDBACK.upper()
    expected_payload_rc = str(Seq(expected_payload).reverse_complement()).upper()
    assert unit.sequence_5to3 == (
        "GTCAGAAAAAA" + "AGTG" + expected_payload + expected_cap + expected_payload_rc + "CAAT" + "ACAGTAACTCAGA"
    )
    assert [segment.role for segment in unit.segments] == [
        "flank_5p_prefix",
        "stem_base_left",
        "payload_primary",
        "snapback_foldback_geometry",
        "payload_complement",
        "stem_base_right",
        "flank_3p_suffix",
    ]
    assert unit.segment_sequence("payload_complement") == expected_payload_rc
    assert unit.segment_span("stem_base_left") == (11, 15)
    assert unit.segment_span("payload_primary") == (15, 34)
    assert unit.segment_span("snapback_foldback_geometry") == (34, 43)
    assert unit.segment_span("payload_complement") == (43, 62)
    assert unit.segment_span("stem_base_right") == (62, 66)
    assert unit.provenance["cap_id"] == "C172"
    assert unit.provenance["snapback_topology_source"] == "de033 released-product 0/3/3 foldback geometry"
    assert record.scar_nick.left_base == "AGTG"
    assert record.scar_nick.right_base == "CAAT"
    assert record.scar_nick.route_status == "resolved"
    assert record.scar_nick.nick_orientation == "bottom"
    assert record.scar_nick.nickase == "Nb.BtsI"
