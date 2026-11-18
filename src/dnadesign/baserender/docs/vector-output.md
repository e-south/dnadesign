---
doc_id: baserender-native-vector-output
title: Native SVG sequence output
owner: dnadesign-maintainers
status: experimental
last_verified: 2026-11-17
---

# Native SVG sequence output

Use `render_record_svg` when sequence artwork must remain vector and illustration
captions must remain independently editable text. This is a source-checkout API;
no package release is implied. The existing figure and fitted-panel APIs retain
their existing raster-illustration behavior.

## First use

From the DNADesign repository with its environment and Arial available:

```sh
uv run --locked python src/dnadesign/baserender/docs/examples/vector_output.py \
  --output-dir src/dnadesign/baserender/outputs/vector-proof-example
```

Choose a fresh directory. The public-API-only example creates four neutral SVGs:
60/100-base records with identical body geometry and independent 12/18-point
captions, an artifact/source index and an HTML comparison. It preserves the
18-base gap between declared features. It does not select scientific data.

The operation returns `SvgRenderResult(svg, width_pt, height_pt, diagnostics)` in
memory and closes its internal figure. It never writes source files or fits
content to a box. The example is responsible for publishing its new review files.

## Source and caption contracts

Import from `dnadesign.baserender`:

- `render_record_svg(record, style_preset=..., style_overrides=...,
  illustrations=..., captions=...)` supports sequence-row records.
- `SvgArtwork(path, sha256, element_ids, view_box, anchors)` selects caller-owned
  geometry. `path` is a `Path`; the hash pins exact bytes; IDs are a distinct
  tuple. `view_box` is `(x, y, width, height)` in source SVG coordinates. Named
  anchors are `(x_fraction, y_fraction_from_top)` within that frame.
- `SvgIllustration(artwork, caption=None)` supplies artwork for an effect's
  `asset_id`. Caller-owned asset IDs need not be packaged. No scientific name or
  producer-specific branch is needed by drawing code.
- `IllustrationCaption(text, family=..., size_pt=..., anchor_fraction=...,
  offset_px=..., visible=...)` controls text separately from illustration scale.
  Fractions start at the illustration's top left; positive offsets move right
  and down. The generic default anchor is bottom-center. Packaged artwork owns
  its particular default caption position in its manifest.

`illustrations` and `captions` are keyed by anchored-effect `asset_id`. The RNAP
manifest selects its existing SVG body IDs and source hash, excluding the source
caption. Bindings reference explicit `Feature.id` values; missing bindings fail
rather than silently dropping a connector. Nothing edits that SVG. Each instance namespaces IDs and retained
definition references and preserves ancestor transforms and source painter order.
The internal SVG slots and effect parameter `svg_asset` are implementation details;
callers use the public result and diagnostics, not artists or generated IDs.

Diagnostics report base pitch, forward/reverse strand positions, logical feature bounds
and illustration anchors/bounds in points, with origin at the canvas top left.
Bounds are `(x0, y0, x1, y1)`. Illustration diagnostics include source SHA-256
and the font file resolved for the caption. Native output requires
`figure_scale=1`; explicit font sizes keep their physical size.

### Nucleotide rectangles

Layout and drawing share nucleotide ink measurements. Letters stay aligned to
the sequence's column grid; the colored rectangle is optically centered around
the visible run. Capitals use their own ink bounds rather than a descender-based
text baseline. Larger feature lettering reserves column pitch and box height
without reducing its requested point size. Too-small explicit pitch fails.
Minimum spacing uses a font/alphabet reference so changing sequence letters
does not change the grid or rectangle height between examples.

`font_size_seq` and `font_size_feature_label` control the two roles separately.
`kmer.pad_y_px` sets minimum vertical inset; `kmer.min_glyph_gap_px` sets minimum
space between neighboring glyphs. Both default to one pixel and reject invalid
units. `kmer.box_height_cells` is a minimum based on both sequence and feature
font cell heights. Diagnostics expose
`nucleotide_box_bounds_pt` for actual colored rectangles separately from
`feature_bounds_pt`, which retains genomic span bounds used by spacer indicators.

For a 12-point sequence with 18-point feature lettering, repeat the example with
`--sequence-font-size 12 --feature-font-size 18 --font-family Arial`. The review
uses point-sized viewports and scrolling, not per-record fitting.

## Admission and current qualification

Source hashes, selections, definitions and fonts are checked before allocating
a figure. Vector selections reject raster images, active content, external
references, DTDs, cycles and unsupported stylesheets. Unused definitions are
pruned. Sources are bounded to 64 MiB and 128 nesting levels. Raster tinting and
trajectory panels are explicitly unsupported on this path. There is no pixel
substitution or silent fit. Retain a separate supported raster input when using
the existing raster figure path.

SVG references font families; fonts are not embedded. A Matplotlib-resolved font
can still be unavailable to another viewer. Select a font installed in the
target viewer and inspect the intended output. The first proof uses explicit
Arial in Chrome and Inkscape. Portable font policy, explicit canvas admission,
neutral label placement, PNG/frame output and CLI/job integration remain in the
[execution plan](../../../../docs/exec-plans/active/baserender-vector-api.md).
Native PowerPoint behavior and real consumer migration are not qualified by this
neutral proof.
