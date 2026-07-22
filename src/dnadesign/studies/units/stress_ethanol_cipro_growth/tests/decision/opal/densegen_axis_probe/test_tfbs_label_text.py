"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/studies/units/stress_ethanol_cipro_growth/tests/decision/opal/densegen_axis_probe/test_tfbs_label_text.py

Regression tests for TFBS label text studies units stress ethanol cipro.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from .probe_modules import probe_object

tfbs_label_display = probe_object("tfbs.label_text", "tfbs_label_display")
tfbs_label_compact_title = probe_object("tfbs.label_text", "tfbs_label_compact_title")
tfbs_control_display_label = probe_object("tfbs.label_text", "tfbs_control_display_label")
tfbs_control_pair_label = probe_object("tfbs.label_text", "tfbs_control_pair_label")
tfbs_label_dropdown_title = probe_object("tfbs.label_text", "tfbs_label_dropdown_title")
tfbs_label_expression = probe_object("tfbs.label_text", "tfbs_label_expression")
tfbs_label_title = probe_object("tfbs.label_text", "tfbs_label_title")


def test_tfbs_label_text_prettifies_literal_labels_without_losing_math() -> None:
    assert tfbs_label_display("lexA_count_fraction") == "LexA count fraction"
    assert tfbs_label_compact_title("lexA_count_fraction") == "LexA count-fraction"
    assert tfbs_label_expression("lexA_count_fraction") == "LexA count / 3"
    assert tfbs_label_title("lexA_count_fraction") == "LexA count fraction (LexA count / 3)"
    assert tfbs_label_dropdown_title("lexA_count_fraction") == "LexA count fraction (count / 3)"
    assert tfbs_label_title("cpxR_or_baeR_count_fraction") == ("CpxR or BaeR count fraction ((CpxR + BaeR) count / 3)")
    assert tfbs_label_dropdown_title("cpxR_or_baeR_count_fraction") == (
        "CpxR or BaeR count fraction (combined count / 3)"
    )
    assert tfbs_label_title("lexA_present") == "LexA presence"
    assert tfbs_label_title("lexA_in_slot0") == "LexA in leftmost slot"


def test_tfbs_control_text_distinguishes_count_fixed_slot_controls() -> None:
    assert tfbs_control_display_label("matched_label_permutation_negative_control") == "row-shuffled control"
    assert (
        tfbs_control_display_label("count_fixed_shuffled_slot_negative_control", label_name="lexA_in_slot0")
        == "slot-shuffled control"
    )
    assert (
        tfbs_control_pair_label("count_fixed_shuffled_slot_negative_control", label_name="lexA_in_slot0")
        == "Sequence-matched metadata vs slot-shuffled control"
    )
