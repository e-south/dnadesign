---
doc_id: baserender-vector-assets-and-layout-proposal
title: Vector artwork and explicit sequence layout
owner: dnadesign-maintainers
status: proposed
last_verified: 2026-11-17
---

## Vector artwork and explicit sequence layout

**Decision requested:** extend BaseRender's vector artwork, layout and output
contracts before adding consumer-specific rendering workarounds. This is a
proposal, not an implemented API or permission to modify caller artwork.

### Observed problems

The packaged RNAP SVG already contains vector body geometry and separate
`rnap_label` text. The selected drawable closure contains no embedded images.
However, its asset manifest selects `rnap_sigma70.png`, and
[the illustration loader](../../../../../src/dnadesign/baserender/src/render/effects/anchored_illustration.py)
uses `mpimg.imread` followed by `ax.imshow`. The manifest's `source_svg` is not
consumed. Saving that figure as SVG therefore embeds a PNG; an SVG output suffix
does not establish that the illustration stayed vector.

[The fixed-panel API](../../../../../src/dnadesign/baserender/src/public/api.py)
also renders RGBA, forces a white background and fits/crops it into a canvas.
This is appropriate for an explicitly requested fitted raster panel. It cannot
establish fixed physical type size simply by fixing the output dimensions.
The current public figure API avoids that fit, but callers still have to
coordinate output, layout settings and measurements themselves.

Some annotation placement in
[sequence_rows.py](../../../../../src/dnadesign/baserender/src/render/sequence_rows.py)
is enabled by producer-name attributes. Neutral rendering requests should
declare label placement directly. Public `Style`, `resolve_style` and
`compute_layout` already exist; reuse those capabilities rather than building
a second layout engine or requiring consumers to inspect Matplotlib artists.

### Boundary

BaseRender owns reusable drawing, vector assets, anchors, captions, typography,
layout measurement and output conversion. Producers own sequence facts,
placements, classifications and inferred order. Consumers own record selection,
page placement and playback timing. Consumer paths and study-specific names do
not enter built-in render logic.

SVG remains the source for vector artwork through composition. PNG/video frames
are explicit final raster outputs. Raster assets remain usable when declared as
such; a vector-only request fails if its selected drawable closure requires a
raster image or unsupported reference. Never substitute a PNG silently.

### Interface alternatives

| Alternative | Consequence |
| --- | --- |
| Add rasterizing SVG support to the current image loader | Accepts SVG filenames but still embeds pixels through `imshow`; does not meet the vector contract. Reject. |
| Add more consumer post-processing around exported figures | Replicates asset, caption, placement and measurement policy outside its owner. Retain existing consumer code as migration evidence only. |
| Compose declared vector assets at BaseRender's SVG output boundary | Reuses measured sequence geometry and existing effects, preserves SVG definitions/transforms, and localizes format handling. Preferred first slice. |

Do not replace the whole renderer. First separate asset loading, measured
illustration placement and output emission inside the existing module hierarchy.
Evaluate SVG-native composition using the current measured Matplotlib geometry;
do not implement a general SVG-to-Matplotlib parser or a new scene framework.

### Proposed public contract

Extend the existing public facade with a narrow vector-render operation returning
an artifact and documented diagnostics. Names below are illustrative, not final:

```python
result = render_record_svg(
    record,
    style=style,
    layout=layout,          # physical type scale, base pitch, strand anchor, bounds
    illustrations=assets,  # explicit SVG selections, hashes and named anchors
)
```

The asset descriptor selects SVG bytes or a pinned source, exact body/caption
IDs and named anchors. Preserve closed definitions and painter order, namespace
IDs and reject unsupported external references. A caption is independently
editable vector text with its own font, size, offset and visibility. The caption
must not inherit the protein artwork's scaling as its only typography control.

Expose physical font sizes, base pitch, strand anchor, illustration clearance,
transparent background and explicit `preserve-type` versus `fit` behavior through
the supported contract. Declare font, canvas and diagnostic coordinate units
explicitly; output conversion must not introduce an implicit second type scale. Keep layout controls out of scientific record metadata
when they are caller presentation settings. Preserve-type overflow fails with
required/available bounds; it never shrinks text or drops annotations silently.

Return glyph/strand anchors, content bounds, resolved font identities and
illustration/caption bounds through diagnostics using the existing layout
authority. A caller should not scan private attributes, inspect artist GIDs or
remeasure exported pixels to learn where the renderer placed its own content.
SVG, PNG and frame outputs must consume the same measured placement.

### Performance contract

A read-only in-process probe of one annotated 60-base record measured about
176–180 ms for subsequent figure creation and 141 ms for SVG export on the
inspection host. The first figure took 299 ms. Each exported SVG still embedded
one PNG. These are three local observations, excluding process startup and video
encoding; they are not a general benchmark or a real-time qualification.

Keep parsing/assets/font measurements reusable across an explicit batch, with
bounded caches keyed by source content and resolved style. Measure existing
caches before adding another. Separate preparation, drawing, serialization and
final rasterization timings; test 60- and 100-base records, crowded tracks and
repeated states. Agree numeric performance budgets after the vector tracer;
do not promise speed by reducing fidelity or recomputing unchanged frames.

### Delivery and acceptance

1. Prove one SVG illustration and separate caption through native SVG output.
   Reuse the packaged editable source; do not trace a PNG or modify caller files.
   Verify its selected closure is vector and exported SVG contains no raster
   image for that illustration. Preserve source transforms, gradients and anchors.
2. Admit explicit layout/label controls and public diagnostics. Prove fixed type,
   pitch and strand position across both sequence lengths and crowded examples.
   Verify missing IDs/fonts/assets, unsupported references and overflow fail
   before output publication. Existing raster API behavior remains explicit.
3. Qualify PNG and representative frames from the same vector composition, then
   measure batch performance. Run BaseRender's package, public-boundary, style
   and docs checks. Inspect representative outputs visually as well as structurally.
4. Migrate actual consumers to public calls and remove displaced rendering,
   artist inspection and SVG-composition logic from them. Retain source selection,
   scientific adaptation, page placement and timing at their existing owners.

No video framework, producer search, scientific reduction, plugin discovery or
cross-project synchronization belongs in this change. Implement the first slice
and review its output before expanding the API surface.
