"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/eco1_rt_repack/tests/materialization/manual_mask_authority/test_motif_sequence_validation.py

Manual motif sequence validation tests for Eco1 RT repack.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import pytest

from dnadesign.studies.units.eco1_rt_repack.operations.materialization.manual_mask_authority.pipeline import (
    _validate_feature_sequence,
)


def test_manual_motif_sequence_validation_rejects_coordinate_drift() -> None:
    residue_by_position = {
        105: {"wt_aa": "N"},
        106: {"wt_aa": "A"},
        107: {"wt_aa": "T"},
        108: {"wt_aa": "P"},
        109: {"wt_aa": "Q"},
        195: {"wt_aa": "Y"},
        196: {"wt_aa": "A"},
        197: {"wt_aa": "D"},
        198: {"wt_aa": "E"},
        243: {"wt_aa": "V"},
        244: {"wt_aa": "S"},
        245: {"wt_aa": "G"},
    }

    with pytest.raises(ValueError, match="EC86 NAxxH"):
        _validate_feature_sequence(
            feature_id="retron_x_naxxh",
            authority_type="retron_x_motif_anchor",
            canonical_positions=[105, 106, 107, 108, 109],
            residue_by_position=residue_by_position,
        )
    with pytest.raises(ValueError, match="EC86 YADD"):
        _validate_feature_sequence(
            feature_id="catalytic_yadd",
            authority_type="catalytic_core_motif_anchor",
            canonical_positions=[195, 196, 197, 198],
            residue_by_position=residue_by_position,
        )
    with pytest.raises(ValueError, match="EC86 VTG"):
        _validate_feature_sequence(
            feature_id="retron_y_vtg",
            authority_type="retron_y_motif_anchor",
            canonical_positions=[243, 244, 245],
            residue_by_position=residue_by_position,
        )


def test_manual_motif_sequence_validation_accepts_ec86_motifs() -> None:
    residue_by_position = {
        105: {"wt_aa": "N"},
        106: {"wt_aa": "A"},
        107: {"wt_aa": "T"},
        108: {"wt_aa": "P"},
        109: {"wt_aa": "H"},
        195: {"wt_aa": "Y"},
        196: {"wt_aa": "A"},
        197: {"wt_aa": "D"},
        198: {"wt_aa": "D"},
        243: {"wt_aa": "V"},
        244: {"wt_aa": "T"},
        245: {"wt_aa": "G"},
    }

    _validate_feature_sequence(
        feature_id="retron_x_naxxh",
        authority_type="retron_x_motif_anchor",
        canonical_positions=[105, 106, 107, 108, 109],
        residue_by_position=residue_by_position,
    )
    _validate_feature_sequence(
        feature_id="catalytic_yadd",
        authority_type="catalytic_core_motif_anchor",
        canonical_positions=[195, 196, 197, 198],
        residue_by_position=residue_by_position,
    )
    _validate_feature_sequence(
        feature_id="retron_y_vtg",
        authority_type="retron_y_motif_anchor",
        canonical_positions=[243, 244, 245],
        residue_by_position=residue_by_position,
    )
