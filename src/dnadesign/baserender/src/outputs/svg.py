"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/src/outputs/svg.py

Compose native SVG at reserved illustration slots in Matplotlib painter order.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

from dataclasses import dataclass
from io import StringIO
from xml.etree import ElementTree as ET

import matplotlib
from matplotlib.artist import Artist

from ..core import RenderingError
from ..core.vector import IllustrationCaption, SvgArtwork
from ..render.svg_assets import SVG, namespace_selection
from ..render.svg_caption import CaptionGeometry


class SvgSlot(Artist):
    """Internal output marker; never exposed as a usable raster figure."""

    def __init__(self, slot_id):
        super().__init__()
        self.slot_id = slot_id
        self.set_zorder(2.7)

    def draw(self, renderer):
        renderer.open_group("vector_illustration", gid=self.slot_id)
        renderer.close_group("vector_illustration")
        self.stale = False


@dataclass(frozen=True)
class SvgPlacement:
    slot_id: str
    artwork: SvgArtwork
    fragment: object
    bounds_pt: tuple[float, float, float, float]
    caption: IllustrationCaption | None
    caption_geometry: CaptionGeometry | None


def compose_svg(figure, placements):
    stream = StringIO()
    with matplotlib.rc_context({"svg.hashsalt": "dnadesign.baserender", "svg.fonttype": "none"}):
        figure.savefig(stream, format="svg", transparent=True, bbox_inches=None, pad_inches=0, metadata={"Date": None})
    root = ET.fromstring(stream.getvalue())
    for placement in placements:
        matches = [
            (parent, i)
            for parent in root.iter()
            for i, child in enumerate(parent)
            if child.get("id") == placement.slot_id
        ]
        if len(matches) != 1:
            raise RenderingError("native SVG illustration slot was not serialized exactly once")
        parent, index = matches[0]
        group = ET.Element(f"{SVG}g", {"id": placement.slot_id})
        fragment = namespace_selection(placement.fragment, placement.slot_id)
        x0, y0, x1, y1 = placement.bounds_pt
        fragment.attrib.update(
            {
                "x": f"{x0:.12g}",
                "y": f"{y0:.12g}",
                "width": f"{x1 - x0:.12g}",
                "height": f"{y1 - y0:.12g}",
                "viewBox": " ".join(f"{n:.12g}" for n in placement.artwork.view_box),
                "preserveAspectRatio": "xMidYMid meet",
                "overflow": "visible",
            }
        )
        group.append(fragment)
        caption = placement.caption
        if caption is not None and caption.visible:
            if placement.caption_geometry is None:
                raise RenderingError("visible SVG caption has no admitted placement")
            cx, cy = placement.caption_geometry.origin_pt
            text = ET.SubElement(
                group,
                f"{SVG}text",
                {
                    "data-baserender-caption": placement.slot_id,
                    "x": f"{cx:.12g}",
                    "y": f"{cy:.12g}",
                    "font-family": caption.family,
                    "font-size": f"{caption.size_pt:.12g}",
                    "text-anchor": caption.text_anchor,
                    "fill": "#000000",
                    "{http://www.w3.org/XML/1998/namespace}space": "preserve",
                },
            )
            text.text = caption.text
        parent[index] = group
    if root.findall(f".//{SVG}image"):
        raise RenderingError("native SVG output contains raster artwork; select supported vector content")
    return ET.tostring(root, encoding="unicode") + "\n"
