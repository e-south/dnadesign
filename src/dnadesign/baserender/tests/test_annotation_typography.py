"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_annotation_typography.py

Keep site, span and illustration caption sizes on one optional annotation role.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

import json
import re
from dataclasses import replace
from xml.etree import ElementTree as ET

import pytest

import dnadesign.baserender as br

from .test_native_jobs import _payload, _rows
from .test_vector_output import example


def _record():
    record = example(100)
    features = tuple(
        br.Feature(
            name,
            "kmer",
            br.Span(start, start + 6, "fwd"),
            record.sequence[start : start + 6],
            render={"annotation_label": {"text": text, "placement": "near_feature"}},
        )
        for name, start, text in (("left", 20, "−35 site"), ("right", 50, "−10 site"))
    )
    illustration = replace(
        record.effects[0],
        target={
            "bindings": [
                {"anchor_id": anchor, "feature_id": feature.id, "start": feature.span.start, "end": feature.span.end}
                for anchor, feature in zip(("upstream", "downstream"), features)
            ]
        },
    )
    return replace(
        record,
        features=features,
        effects=(
            illustration,
            br.Effect(
                "span_link",
                {"from_feature_id": "left", "to_feature_id": "right"},
                {"label": "24 bp", "shrink_label_to_fit": False},
            ),
        ),
    )


def _size(node):
    if node.get("font-size") is not None:
        return float(node.get("font-size"))
    return float(re.search(r"font-size:\s*([0-9.]+)px", node.get("style")).group(1))


@pytest.mark.parametrize(
    "settings,expected", [({"size_pt": 18}, 18), ({"sequence_scale": 1}, 14), ({"sequence_scale": 1.5}, 21), ({}, 14)]
)
def test_shared_annotation_role_resolves_all_sizes_together(settings, expected):
    style = {
        "legend": False,
        "font_size_seq": 14,
        "font_size_feature_label": 14,
        "font_size_annotation_label": 13,
        "font_size_span_link_label": 17,
    }
    result = br.render_record_svg(
        _record(),
        style_overrides=style,
        captions={"rnap_sigma70": br.IllustrationCaption("RNAP", size_pt=11)},
        annotation_typography=br.AnnotationTypography(**settings),
    )
    texts = {n.text: n for n in ET.fromstring(result.svg).iter("{http://www.w3.org/2000/svg}text")}
    assert [_size(texts[key]) for key in ("−35 site", "−10 site", "24 bp", "RNAP")] == [expected] * 4
    assert result.diagnostics.annotation_font_size_pt == expected
    assert result.diagnostics.span_links[0].font_size_pt == expected
    assert style["font_size_annotation_label"] == 13


def test_independent_annotation_sizes_remain_available_when_group_is_omitted():
    style = {"legend": False, "font_size_annotation_label": 13, "font_size_span_link_label": 17}
    kwargs = {"style_overrides": style, "captions": {"rnap_sigma70": br.IllustrationCaption("RNAP", size_pt=11)}}
    result = br.render_record_svg(_record(), **kwargs)
    assert result.svg == br.render_record_svg(_record(), annotation_typography=None, **kwargs).svg
    texts = {n.text: n for n in ET.fromstring(result.svg).iter("{http://www.w3.org/2000/svg}text")}
    assert [_size(texts[key]) for key in ("−35 site", "24 bp", "RNAP")] == [13, 17, 11]
    assert "annotation_font_size_pt" not in result.diagnostics.to_mapping()


@pytest.mark.parametrize(
    "settings",
    [
        {"size_pt": 5},
        {"size_pt": True},
        {"size_pt": float("nan")},
        {"sequence_scale": 0},
        {"sequence_scale": float("inf")},
        {"sequence_scale": "1.2"},
        {"size_pt": 18, "sequence_scale": 1},
    ],
)
def test_bad_annotation_typography_fails_at_construction(settings):
    with pytest.raises(br.RenderingError, match="annotation_typography"):
        br.AnnotationTypography(**settings)


def test_shared_sizes_are_admitted_before_drawing(monkeypatch):
    import matplotlib.pyplot as plt

    monkeypatch.setattr(plt, "figure", lambda *a, **kw: pytest.fail("late typography admission"))
    with pytest.raises(br.RenderingError, match="annotation_typography.*6"):
        br.render_record_svg(
            _record(),
            style_overrides={"font_size_seq": 6},
            annotation_typography=br.AnnotationTypography(sequence_scale=0.5),
        )


def test_job_and_public_api_share_annotation_typography_and_receipts(tmp_path):
    payload = _payload(tmp_path)
    payload["render"]["native_svg"]["annotation_typography"] = {"sequence_scale": 1.25}
    job = br.validate_job(payload, caller_root=tmp_path)
    assert job.render.native_svg.annotation_typography.to_mapping() == {"sequence_scale": 1.25}
    br.run_job(job)
    for row in _rows():
        expected = br.render_record_svg(
            br.adapt_record(
                row, adapter_kind="generic_features", adapter_columns=payload["input"]["adapter"]["columns"]
            ),
            style_overrides=payload["render"]["style"]["overrides"],
            layout=br.SequenceLayout.from_mapping(payload["render"]["options"]["layout"]),
            captions={"rnap_sigma70": br.IllustrationCaption("RNAP", family="Arial", size_pt=18)},
            annotation_typography=br.AnnotationTypography(sequence_scale=1.25),
        )
        assert (tmp_path / "bundle/images" / f"{row['id']}.svg").read_text() == expected.svg
    manifest = json.loads((tmp_path / "bundle/manifest.json").read_text())
    assert manifest["render_spec"]["native_svg"]["annotation_typography"] == {"sequence_scale": 1.25}
    assert (
        manifest["output_metrics"]["images"]["records"][0]["resolved_illustrations"]["br-vector-0"]["caption"][
            "size_pt"
        ]
        == 17.5
    )


@pytest.mark.parametrize(
    "setting", [{"typo": 1}, {"sequence_scale": 0}, {"size_pt": 18, "sequence_scale": 1}, {"sequence_scale": 0.01}]
)
def test_bad_group_settings_fail_in_job_admission(tmp_path, setting):
    payload = _payload(tmp_path)
    payload["render"]["native_svg"]["annotation_typography"] = setting
    with pytest.raises(br.SchemaError, match="annotation_typography"):
        br.validate_job(payload, caller_root=tmp_path)
    assert not (tmp_path / "bundle").exists()
