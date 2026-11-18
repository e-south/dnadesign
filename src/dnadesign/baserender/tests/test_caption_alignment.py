"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/tests/test_caption_alignment.py

Keep caption edge anchoring consistent in the API, measured bounds and native jobs.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from xml.etree import ElementTree as ET

import pytest

import dnadesign.baserender as br

from .test_native_jobs import _payload
from .test_vector_output import example


@pytest.mark.parametrize("align", ["start", "middle", "end"])
def test_caption_anchor_keeps_requested_edge_as_type_size_changes(align):
    placements = []
    for size in (12, 24):
        result = br.render_record_svg(
            example(100),
            style_overrides={"legend": False},
            captions={
                "rnap_sigma70": br.IllustrationCaption(
                    "RNAP", size_pt=size, anchor_fraction=(1, 0.5), offset_px=(20, 0), text_anchor=align
                )
            },
        )
        placement = result.diagnostics.illustrations[0]
        text = ET.fromstring(result.svg).find(".//{http://www.w3.org/2000/svg}text[@data-baserender-caption]")
        assert text.get("text-anchor") == align
        x = placement.caption_origin_pt[0]
        left, _, right, _ = placement.caption_bounds_pt
        if align == "start":
            assert left >= x
        elif align == "end":
            assert right <= x
        else:
            assert left < x < right
        placements.append(placement)
    assert placements[0].bounds_pt == placements[1].bounds_pt
    assert placements[0].caption_origin_pt == placements[1].caption_origin_pt


def test_job_caption_alignment_uses_the_public_caption_contract(tmp_path):
    payload = _payload(tmp_path)
    payload["render"]["native_svg"]["captions"]["rnap_sigma70"]["text_anchor"] = "start"
    job = br.validate_job(payload, caller_root=tmp_path)
    assert job.render.native_svg.captions["rnap_sigma70"].text_anchor == "start"
    assert job.render.native_svg.to_mapping()["captions"]["rnap_sigma70"]["text_anchor"] == "start"
    br.run_job(job)
    for svg in (tmp_path / "bundle/images").glob("*.svg"):
        caption = ET.fromstring(svg.read_text()).find(".//{http://www.w3.org/2000/svg}text[@data-baserender-caption]")
        assert caption.get("text-anchor") == "start"


@pytest.mark.parametrize("align", ["left", "", None, True, []])
def test_caption_alignment_rejects_unknown_values(align):
    with pytest.raises(br.RenderingError, match="text_anchor"):
        br.IllustrationCaption("RNAP", text_anchor=align).validate()
