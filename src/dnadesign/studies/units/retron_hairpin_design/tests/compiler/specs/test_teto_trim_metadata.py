"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/retron_hairpin_design/tests/compiler/specs/test_teto_trim_metadata.py

Tests for tetO payload-trim metadata in typed Retron MSD compiler specs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import csv
from pathlib import Path

from dnadesign.studies.units.retron_hairpin_design.catalog.compiler_spec import load_msd_compiler_spec
from dnadesign.studies.units.retron_hairpin_design.compiler.catalog_bundle import write_msd_design_catalog

from ...support.registry import write_minimal_retron_msd_registry


def test_retron_msd_spec_preserves_teto_trim_and_design_metadata_in_reference_index(tmp_path: Path) -> None:
    study_dir = write_minimal_retron_msd_registry(tmp_path)
    spec_path = tmp_path / "compiler_spec.yaml"
    out_dir = tmp_path / "compiled"
    spec_path.write_text(
        """
contract: retron_msd_compiler_spec_v1
schema_version: 1
designs:
  - construct_id: pES-tetr-d033-w02-17
    payload_id: TetR_w02_17
    cap_id: C172
    left_base: AGTG
    right_base: CAAT
    literal_stem_base_source_id: scar_nick_literal_de033_mxmm_v1
    nick_orientation: bottom
    nickase: Nb.BtsI
    variant_role: rescue_candidate
    scaffold_context: de033_selected
    payload_trim_id: TetR_w02_17
    cap_selector_id: de033_snapback_cap_rank_1
    stem_base_selector_id: de033_scar_nick_stem_base_rank_1
    rt_mode: wt_eco1
    decision_group: target_trim_rescue
    control_id: pES-tetr-d033-w00-19
    rationale: Window [2,17) PWM-edge trim in the DE033-compatible scaffold target.
payload_sequences:
  TetR_w02_17:
    sequence: CCTATCAGTGATAGA
    display_name: msd[teto_w02_17]
    parent_payload_id: TetR_w00_19
    payload_trim_id: TetR_w02_17
    trim_class: conservative
    trim_5p_nt: 2
    trim_3p_nt: 2
    retained_parent_span_0: {start: 2, end: 17}
    pwm_source_ref: cruncher:westmann_tetr_mitomi:tetR
    selection_basis: tetO_pwm_information_content
    protected_positions_or_reason: Retains the central tetO operator motif while removing low-information edges.
cap_sequences:
  C172: GAGAGACTC
""",
        encoding="utf-8",
    )

    resolved = load_msd_compiler_spec(spec_path, study_dir=study_dir)
    record = resolved.catalog.records[0]

    assert record.payload_or_target.payload_trim_id == "TetR_w02_17"
    assert record.payload_or_target.parent_payload_id == "TetR_w00_19"
    assert record.payload_or_target.trim_class == "conservative"
    assert record.payload_or_target.retained_parent_span_0 is not None
    assert record.payload_or_target.retained_parent_span_0.start == 2
    assert record.variant_metadata is not None
    assert record.variant_metadata.variant_role == "rescue_candidate"
    assert record.variant_metadata.scaffold_context == "de033_selected"
    assert record.variant_metadata.rt_mode == "wt_eco1"

    write_msd_design_catalog(resolved.catalog, out_dir=out_dir)
    rows = list(
        csv.DictReader((out_dir / "reference_index.tsv").read_text(encoding="utf-8").splitlines(), delimiter="\t")
    )
    assert rows[0]["payload_trim_id"] == "TetR_w02_17"
    assert rows[0]["payload_trim_class"] == "conservative"
    assert rows[0]["variant_role"] == "rescue_candidate"
    assert rows[0]["scaffold_context"] == "de033_selected"
    assert rows[0]["rt_mode"] == "wt_eco1"
    assert rows[0]["decision_group"] == "target_trim_rescue"
