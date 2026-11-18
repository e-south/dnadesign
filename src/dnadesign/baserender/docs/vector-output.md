---
doc_id: baserender-native-vector-output
title: Native SVG sequence output
owner: dnadesign-maintainers
status: experimental
last_verified: 2026-11-18
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
  illustrations=..., captions=..., layout=...)` supports sequence-row records.
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
and the font file resolved for the caption. They also return `caption_origin_pt`
(the text-anchor midpoint and baseline) and `caption_bounds_pt` (visible ink).
Hidden captions return neither bound nor origin. Native output requires
`figure_scale=1`; explicit font sizes keep their physical size.

### Artwork and caption bounds

Native artwork keeps its calculated anchor position. A declared illustration
frame outside the canvas fails with its asset ID and attempted bounds; the
existing raster figure route retains its clamping behavior. Change the explicit
grid origin, canvas, reserved top extent or illustration width to resolve the
failure. The illustration frame comes from the selected source's declared
`view_box`; callers must supply a frame containing the selected geometry.
BaseRender preserves those SVG paths and transforms rather than estimating them
through a second SVG geometry engine.

Captions are single-line literal text. Dollar signs remain literal, repeated
spaces are preserved, and control characters fail admission. Missing visible caption
glyphs fail instead of producing replacement characters. The measured caption
origin is used directly by serialization. Out-of-canvas captions fail before
figure allocation; they are not moved, clipped or reduced automatically.

### Physical canvas and grid

`SequenceLayout` declares physical units independently of output DPI:

```python
from dnadesign.baserender import SequenceLayout, render_record_svg

layout = SequenceLayout(
    base_pitch_pt=10,
    canvas_size_pt=(1200, 400),
    sequence_origin_pt=(80, 200),
)
result = render_record_svg(record, style_overrides={"legend": False}, layout=layout)
```

The origin is the left edge of base column zero on the forward strand's grid
reference line, measured rightward/downward from the top left. It is independent
of a particular glyph's ink or baseline. Diagnostics return `sequence_origin_pt`
in that same frame. Without a new origin, expanding the canvas preserves the
existing top-left grid position. Features, nucleotide rectangles and motif lanes
move with the grid; type and pitch are not fitted to the new canvas. Natural
canvas sizing also includes spacer lines and labels, for both feature references
and explicit spans.

`top_extent_pt` and `bottom_extent_pt` reserve minimum space about the midpoint
between the two strand-reference lines (or the single line).
`extra_bottom_padding_pt` reserves additional footer space. These values preserve
the measured content rather than cropping it. The canvas must contain the
natural surface, including its configured margins. Insufficient pitch or bounds,
invalid units, arithmetic overflow and unknown mapping fields fail before figure
allocation. `SequenceLayout.from_mapping(...)` admits JSON/YAML-shaped settings;
`to_mapping()` returns a detached, serializable mapping.

The figure API uses the same settings through
`renderer_options={"layout": layout}`. Image jobs use `render.options.layout`;
see the [job syntax](reference.md#physical-sequence-layout). Existing metadata
keys for cell scale, fixed extents/radius and video padding remain supported when
no typed layout is supplied. Declaring both authorities fails with the conflicting
keys instead of choosing one. Explicit layout requires `figure_scale=1`.
Video jobs with layout are currently rejected because the legacy video compositor
does not preserve this contract; native frame parity remains in the execution plan.

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

### Spacer labels

Native output and explicit `SequenceLayout` rendering preserve spacer-label
sizes, including fractional points. A label that cannot fit its declared span
fails before allocation instead of shrinking. Explicit
`shrink_label_to_fit=True` conflicts with that contract; remove the flag or set
it to `False`, then choose the size/pitch/clearance explicitly. The existing
ordinary figure route retains its fitting behavior.

Spacer drawing and feature-label collision checks consume one measured plan.
`diagnostics.span_links` returns `SvgSpanLinkPlacement` values with the source
effect index, rendered text, point size, endpoint coordinates and reserved label
bounds (including clearance). Unlabelled spacers return no font or label bounds;
their line/tick bounds still undergo canvas admission. Span targets accept
read-only mappings and reject fractional, boolean or out-of-range base coordinates.

### Bottom legends

Repeat the first-use command with `--font-family Arial --legend-font-size 12.5`
to include measured bottom legends in all four proofs. `--legend-height-px`
controls their reserved footer (60 px in the example).

Native SVG and explicit `SequenceLayout` output preserve `legend_font_size`,
including fractional points. `uniform_display_font_size=True` uses the largest
configured display role instead. Legends wrap at that size within the declared
`legend_height_px` footer; insufficient width or height fails before allocation.
Increase the canvas width/footer reservation or choose another size explicitly.
Native swatch dimensions and gaps also retain their configured values.

`diagnostics.bottom_legend` contains `SvgLegendEntryPlacement` values: tag, exact
label, point size, swatch bounds, text origin and reserved entry bounds. Coordinates
use the same top-left point frame as the other diagnostics. Row heights account
for DPI; swatches and text share the row center. Labels are literal text, with
optional newline breaks, rather than math expressions. Fonts, glyph coverage,
colors and units are checked before drawing.

The ordinary figure route retains font fitting and uses spare footer clearance
when needed; even that route fails if no complete, nonoverlapping legend fits.
The default presentation preset now reserves 40 px instead of 16 px so its
11-point legend fits at 180 dpi. This intentionally increases the natural height
of figures using that bottom legend; figures without it are unchanged.

## Final raster output

Keep the SVG result as the composition and rasterize only when an output needs
pixels. This reuses its measured point geometry without repeating record layout:

```python
from pathlib import Path
from dnadesign.baserender import rasterize_svg

png = rasterize_svg(result, dpi=144)
output = Path("src/dnadesign/baserender/outputs/native-png/sequence.png")
output.parent.mkdir(parents=True, exist_ok=True)
with output.open("xb") as handle:
    handle.write(png.png)
```

This native conversion requires an **existing Inkscape CLI** on `PATH`, or an
explicit `inkscape=Path(...)`. It does not install a dependency or try another
renderer. The ordinary Matplotlib figure API remains independent of Inkscape.
Repeat the first-use example with `--raster-dpi 144` to generate SVG and PNG
pairs; `--inkscape` selects an explicit executable for that example.

`PngRenderResult` reports `png`, `width_px`, `height_px`, `dpi`,
`source_svg_sha256`, the same `diagnostics` object and a `RasterizerInfo` value.
The latter records the resolved executable path, that file's SHA-256 and the
reported Inkscape version. An executable can be a platform wrapper; its digest
identifies the selected file, not every library used by the application.

The converter exports the page at the selected DPI, preserving alpha. It does
not crop, fit, recolor, or regenerate the artwork. Pixel dimensions are nearest
integer samples of the serialized point page; half-pixel ties depend on the
backend. Use a common canvas with integral pixel dimensions for a frame series.

`timeout_s` defaults to 30 per external invocation and `max_pixels` to 64 million.
Invalid units/budgets, incomplete or changed font identities, unsafe SVG content,
missing tools, process failures/timeouts and corrupt or wrongly sized PNGs fail
with `RenderingError`. No partial PNG result is returned. The caller owns file
publication; jobs retain their existing staged-publication boundary. Native
image-job selection and frame integration are not yet connected to this API.

## Admission and current qualification

Source hashes, selections, definitions and fonts are checked before allocating
a figure. Vector selections reject raster images, active content, external
references, DTDs, cycles and unsupported stylesheets. Unused definitions are
pruned. Sources are bounded to 64 MiB and 128 nesting levels. Raster tinting and
trajectory panels are explicitly unsupported on this path. There is no pixel
substitution or silent fit. Retain a separate supported raster input when using
the existing raster figure path.

Diagnostics declare `text_font_policy="reference"` and return `fonts` by declared
role (`sequence`, `label`, `caption:<asset_id>`). Each `ResolvedFont` records the
requested/resolved family, face style, local file and SHA-256. The SVG references
fonts and does not embed font files; the local file paths stay in diagnostics.
Caption text references the canonical resolved family. Text already present in
caller-selected artwork retains its source SVG styling.

A resolved producer font can still be unavailable to another viewer. Select a
font installed in the target viewer and inspect the intended output; this policy
does not claim portable text rendering from Matplotlib lookup alone. The example
uses explicit Arial for editable labels in Chrome and Inkscape and records font
identities alongside the SVGs. Feature annotations use the explicit neutral
declaration in the [record contract](reference.md#neutral-record). Portable font
qualification, native frames and native SVG CLI/job integration remain in the
[execution plan](../../../../docs/exec-plans/active/baserender-vector-api.md).
Native PowerPoint behavior and real consumer migration are not qualified by this
neutral proof.
