"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_feature_annotation_labels.py

Contract and rendering checks for producer-neutral feature annotation labels.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from dataclasses import replace
from xml.etree import ElementTree as ET

import matplotlib.pyplot as plt
import pytest

from dnadesign.baserender import ContractError, Feature, LayoutError, Record, RenderingError, Span, render_record_svg


def _record(annotation, *, source="caller"):
    return Record(
        id="annotated",
        alphabet="DNA",
        sequence="ACGT" * 15,
        features=(
            Feature(
                "site",
                "kmer",
                Span(20, 26, "fwd"),
                "ACGTAC",
                tags=("part:binding",),
                attrs={"source": source, "display_label": "not the selected text"},
                render={"annotation_label": annotation},
            ),
        ),
    )


@pytest.mark.parametrize("placement", ["near_feature", "inline"])
def test_annotation_text_and_placement_are_explicit_and_producer_independent(placement):
    annotation = {"text": "selected part", "placement": placement}
    style = {"legend": True, "legend_mode": "inline"}
    for source in ("caller", "densegen_promoter", "usr_genbank"):
        result = render_record_svg(_record(annotation, source=source), style_overrides=style)
        texts = [node.text for node in ET.fromstring(result.svg).iter() if node.tag.endswith("}text")]
        assert texts.count("selected part") == 1
        assert "not the selected text" not in texts
        assert "Binding" not in texts


@pytest.mark.parametrize(
    "annotation",
    [
        "Selected part",
        {"text": "", "placement": "near_feature"},
        {"text": "Selected part"},
        {"text": "Selected part", "placement": "floating"},
        {"text": "Selected part", "placement": "inline", "font": 12},
    ],
)
def test_invalid_annotation_declarations_fail_at_record_admission(annotation):
    with pytest.raises(ContractError, match="annotation_label"):
        _record(annotation).validate()


@pytest.mark.parametrize("placement", ["near_feature", "inline"])
def test_required_annotation_cannot_silently_disappear(placement):
    record = _record({"text": "unfittable " * 200, "placement": placement})
    before = plt.get_fignums()
    try:
        with pytest.raises(LayoutError, match="site.*annotation.*fit"):
            render_record_svg(record, style_overrides={"legend": True, "legend_mode": "inline"})
        assert plt.get_fignums() == before
    finally:
        for number in set(plt.get_fignums()) - set(before):
            plt.close(number)


def test_bad_annotation_color_is_rejected_before_figure_allocation():
    record = _record({"text": "Selected part", "placement": "near_feature"})
    feature = record.features[0]
    record = replace(record, features=(replace(feature, attrs={**feature.attrs, "annotation_color": "invalid"}),))
    before = plt.get_fignums()
    try:
        with pytest.raises(RenderingError, match="site.*annotation_color"):
            render_record_svg(record)
        assert plt.get_fignums() == before
    finally:
        for number in set(plt.get_fignums()) - set(before):
            plt.close(number)


def test_source_metadata_alone_does_not_request_an_annotation():
    record = _record({"text": "Selected part", "placement": "near_feature"}, source="densegen_promoter")
    record = replace(record, features=(replace(record.features[0], render={}),))
    result = render_record_svg(record, style_overrides={"legend": False})
    assert "not the selected text" not in result.svg
