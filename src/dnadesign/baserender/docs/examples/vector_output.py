"""
--------------------------------------------------------------------------------
dnadesign
src/dnadesign/baserender/docs/examples/vector_output.py

Build neutral 60/100-base SVG proofs using only the public BaseRender API.

Module Author(s): Eric J. South
--------------------------------------------------------------------------------
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET

from dnadesign.baserender import Effect, Feature, IllustrationCaption, Record, Span, render_record_svg


def example(length):
    sequence = ("ACGT" * 25)[:length]
    return Record(
        id=f"vector-{length}",
        alphabet="DNA",
        sequence=sequence,
        features=(
            Feature("left", "kmer", Span(4, 10, "fwd"), sequence[4:10]),
            Feature("right", "kmer", Span(28, 34, "fwd"), sequence[28:34]),
        ),
        effects=(
            Effect(
                "anchored_illustration",
                {
                    "bindings": [
                        {"anchor_id": "upstream", "feature_id": "left", "start": 4, "end": 10},
                        {"anchor_id": "downstream", "feature_id": "right", "start": 28, "end": 34},
                    ]
                },
                {"asset_id": "rnap_sigma70", "width_px": 300},
            ),
            Effect("span_link", {"from_feature_id": "left", "to_feature_id": "right"}, {"label": "18 bp"}),
        ),
        meta={"fixed_content_top_extent_px": 260},
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rendered = {}
    records = []
    for length in (60, 100):
        for size in (12, 18):
            name = f"{length}bp-caption-{size}.svg"
            result = render_record_svg(
                example(length),
                style_overrides={"legend": False, "show_pair_rungs": False, "font_label": "Arial"},
                captions={
                    "rnap_sigma70": IllustrationCaption(
                        "RNAP", size_pt=size, family="Arial", anchor_fraction=(0.83827, 0.55688)
                    )
                },
            )
            root = ET.fromstring(result.svg)
            rendered[name] = result.svg
            records.append(
                {
                    "file": name,
                    "sha256": hashlib.sha256(result.svg.encode()).hexdigest(),
                    "width_pt": result.width_pt,
                    "height_pt": result.height_pt,
                    "caption_size_pt": size,
                    "base_pitch_pt": result.diagnostics.base_pitch_pt,
                    "image_count": len(root.findall(".//{http://www.w3.org/2000/svg}image")),
                    "source_sha256": result.diagnostics.illustrations[0].source_sha256,
                }
            )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, payload in rendered.items():
        (args.output_dir / name).write_text(payload)
    (args.output_dir / "receipt.json").write_text(json.dumps(records, indent=2) + "\n")
    cards = "".join(
        f'<article><h2>{name}</h2><object type="image/svg+xml" data="{name}"></object>'
        f'<p><a href="{name}">SVG source</a></p></article>'
        for name in rendered
    )
    (args.output_dir / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>BaseRender vector proof</title>'
        "<style>body{font:16px Arial;margin:24px;background:#e5ece8}article{background:white;padding:16px;"
        "margin:24px 0}object{width:100%;height:auto;aspect-ratio:4/1}h2{font-size:18px}</style>"
        "<h1>Native SVG: 60/100-base proof</h1><p>Identical body geometry; independent 12/18-point captions. "
        "No source artwork was changed. The white card is review chrome, not a masking layer.</p>" + cards
    )
    print(args.output_dir / "index.html")


if __name__ == "__main__":
    main()
