"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/latentdna/tests/test_study_neutral_defaults.py

Study labels and styling are supplied through workspace configuration.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from dnadesign.latentdna.src.metadata.axes import AxisStyle, axis_color_map
from dnadesign.latentdna.src.plots.axes import axis_category_label
from dnadesign.latentdna.src.presentation.visual_style import reference_annotation_label


def test_reference_identifiers_are_not_reinterpreted_as_study_windows():
    assert reference_annotation_label("sample_core60_context1kb_rc") == "sample_core60_context1kb_rc"
    assert reference_annotation_label("spyp") == "spyp"


def test_explicit_category_styles_preserve_scientific_labels_and_colors():
    style = AxisStyle(
        axis_id="family",
        column="family",
        display_labels={"a": "Binding sites A"},
        category_colors={"a": "#E69F00"},
        compact_display_labels={"a": "A"},
    )
    assert axis_category_label("a", column="family", axis_styles={"family": style}) == "Binding sites A"
    assert axis_category_label("a", column="family", axis_styles={"family": style}, compact=True) == "A"
    assert axis_color_map(style, ["a"], fallback_palette=["#000000"])["a"] == "#E69F00"
