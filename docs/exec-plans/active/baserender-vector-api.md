---
doc_id: baserender-vector-api-execution
title: BaseRender vector output and pragmatic API hardening
owner: dnadesign-maintainers
status: active
last_verified: 2026-11-18
---

## Exec plan: BaseRender vector output and pragmatic API hardening

**Status:** active
**Owner:** dnadesign-maintainers
**Created:** 2026-11-18
**Last updated:** 2026-11-18

### Purpose / Big Picture

Make BaseRender the reusable owner of vector artwork, independent captions and
predictable typography/layout. Success lets actual consumers select records and
place outputs without duplicating drawing, SVG repair or geometry inspection.
That reduces visual drift and makes future aesthetic changes local to one API.

In scope: BaseRender's public API/CLI, configuration, assets, layout, drawing,
SVG/raster/frame publication, performance and supporting tests/docs, plus bounded
consumer migration. Out of scope: scientific searches/reductions, manuscript or
author-SVG edits, a presentation engine, a monorepo-wide rewrite, package release
or remote publication. Rasterization remains an explicit final output operation.

### Progress

- [x] (2026-11-18 02:06Z) Capture the proposal, known source findings, focused baseline and next-executor contract; implementation has not started.
- [x] (2026-11-18 02:34Z) A — Baseline package suite, catalog, demo admission and public usage reviewed; prioritized findings below retain open work.
- [x] (2026-11-18 02:34Z) B — Native SVG artwork and independent vector captions proved through the public API on 60/100-base examples.
- [ ] (2026-11-18 05:15Z) C — In progress: spacer drawing/collision geometry now unified with native preserve-type checks; bottom-legend fitting remains before C closes.
- [ ] (2026-11-18 02:06Z) D — Qualify shared geometry through SVG, PNG, representative frames and CLI/job admission.
- [ ] (2026-11-18 02:06Z) E — Measure and improve batch preparation against fidelity-preserving baselines.
- [ ] (2026-11-18 02:06Z) F — Migrate the two real sequence uses in separate verified batches.
- [ ] (2026-11-18 02:06Z) G — Remove displaced consumer logic and close the bounded package hardening review.

### Surprises & Discoveries

The packaged RNAP SVG has editable body geometry and separate `rnap_label` text.
Its drawable closure has no images, but the manifest chooses PNG and the loader
uses `imread`/`imshow`. SVG export therefore embeds pixels. Some unused SVG
definitions contain raster material: selection/definition closure matters more
than the filename or a blanket scan of every unused definition.

The fitted-panel path forces white RGBA and rescales it; a fixed canvas does not
fix physical typography. Public `Style`, `resolve_style` and `compute_layout`
already exist and should supply the new path's geometry authority. Some label
placement is currently gated by producer-name attributes.

Read-only observations identified large responsibility concentrations in
`render/sequence_rows.py`, `outputs/video.py`, `public/api.py`, `render/layout.py`
and `config/style_v1.py`. Size alone is not a defect or a reason to split them.
Before implementation, the complete package suite passed on `4bb0793da`.
Catalog output and demo-job validation also passed. Review covered API/job
admission, configuration, the asset loader, sequence layout/drawing, image/video
output routes, public/resource tests and documentation. The resulting findings
are bounded to these observed paths, not a claim that every renderer or hostile
SVG feature has been exhaustively audited.

| Priority / criterion | Evidence and status | Owning slice |
| --- | --- | --- |
| P1: vector requests preserve actual geometry | [Illustration loader](../../../src/dnadesign/baserender/src/render/effects/anchored_illustration.py) selected pixels; the new native API reads pinned SVG selections without decoding PNG. Closed for the native API; legacy APIs remain explicit raster paths. | B |
| P1: fixed typography and transparent composition | [Fitted panel](../../../src/dnadesign/baserender/src/public/api.py) forces white and rescales. Native SVG avoids that route; fixed-canvas admission and PNG/frame parity are still open. | C/D |
| P2: neutral placement policy | [Sequence labels](../../../src/dnadesign/baserender/src/render/sequence_rows.py) now use explicit feature annotation declarations. DenseGen/GenBank adapters supply the policy; provenance attributes no longer control placement. | C, closed |
| P2: exact placement is observable | Native illustration frames now fail on overflow instead of clamping; captions share admitted origins with serialization and report ink bounds. Existing raster behavior stays explicit. | C, closed |
| P2: viewer font identity | Native diagnostics declare referenced text and hash the resolved declared fonts. Captions reject missing glyphs and use resolved family names. Arial proofs render in Chrome/Inkscape; target font availability still needs consumer qualification. | C policy closed; D/F qualification |
| P2: repeated illustrations have unique IDs | A two-instance test exposed colliding footprint IDs. Unique footprint IDs and definition namespaces now pass structural tests. | B, closed |
| P2: video honors admitted layout | [Legacy video](../../../src/dnadesign/baserender/src/outputs/video.py) receives no renderer options, mutates layout metadata and applies its own fitting (including a 0.96 content scale). Explicit layout with video now fails at admission. D must qualify a shared native frame path before enabling it. | D, open |
| P2: native labels preserve requested type | Spacers now use one plan and preserve fractional native sizes or fail. Nonuniform bottom legends still need explicit preserve-type admission; ordinary fitting remains supported. | C, bottom legend open |

### Decision Log

- Preserve selected SVG artwork through composition, with explicit caption
  text/style and anchors. A vector request must not silently substitute pixels.
- Reuse existing layout/style machinery. Prefer owner-local SVG composition
  over a general SVG-to-Matplotlib parser or a new public scene framework.
- Expand the supported vector surface, migrate real consumers, then contract
  displaced code. Existing raster/figure APIs remain valid where appropriate;
  do not remove them merely because the new vector path exists.
- Keep generic layout/labels producer-neutral. Scientific mappings, inferred
  order, classifications and data selection remain with their current owners.
- Refactor only responsibilities exposed by findings or touched by a slice.
  Separate behavior-preserving extraction from new behavior and use red/green
  evidence for contract changes. Do not create pass-through modules or generic
  utilities solely to reduce line counts.
- Treat performance as measured work: preserve correctness, fonts and source
  fidelity while separating preparation, drawing, serialization and raster cost.

### Outcomes & Retrospective

The implementation rationale is the
[vector-assets proposal](../../dev/plans/tools/baserender/vector-assets-and-layout.md).
Slices A/B now provide `render_record_svg`, explicit source/caption records and
measured diagnostics. Exact body IDs and a SHA-256 were added to the packaged
manifest; original SVG bytes are unchanged. SVG emission composes at internal
slots in painter order, without a new general renderer or SVG-to-path parser.
Legacy figure output remains supported. Public-API-only neutral proofs and
repeat commands live in the [native output guide](../../../src/dnadesign/baserender/docs/vector-output.md).

The proof at `src/dnadesign/baserender/outputs/vector-proof-20261118-03/` has
four SVGs, a source/artifact index, a Chrome screenshot and an Inkscape-rendered
PNG. Its `receipt.json` reports zero raster images in every SVG. The earlier
`vector-proof-20261118-01/` retains the observed font fallback. These are ignored
review artifacts, reproducible by the checked-in example; no scientific asset,
consumer selection, manuscript or release changed. C–G remain required.

Qualification: 659 package tests passed, including 28 new native-output and
negative-path tests; Ruff check/format, architecture boundaries, documentation
checks, catalog and demo-job admission passed. Before runtime edits, all 631
baseline tests passed. Red evidence includes the missing public API, duplicate
footprint IDs, a corrupt-pixel preflight regression and a silently omitted
missing-feature connector. The latter two now fail before figure allocation.
The compact-docs allowlist deliberately adds only the focused native-output guide.
Harness endpoints are architecture-invariants and knowledge-integrity, verified
with existing checks rather than a new execution framework.
The commit detector flagged the new SVG SHA-256 as high entropy. One exact
false-positive entry was added after matching it to the artwork bytes; detector
settings and unrelated baseline entries are unchanged.

C's first implementation addresses the reported nucleotide/rectangle mismatch.
[Shared nucleotide geometry](../../../src/dnadesign/baserender/src/render/nucleotide_geometry.py)
now owns ink metrics, minimum pitch/height and optical box placement. Layout and
drawing use that authority; genomic column correspondence and requested font
sizes remain intact. A 12-point sequence with 18-point feature lettering reserves its
required space. Explicit insufficient pitch fails rather than shrinking text.
Public diagnostics distinguish genomic spans from painted nucleotide rectangles.

The review at `src/dnadesign/baserender/outputs/vector-alignment-20261118-03/comparison.html`
compares the retained B proof with corrected SVGs at natural point size. The
default example's left/right margins changed from 1.63/3.56 px to 2.59/2.59 px,
and bottom/top from 8.67/1.19 px to 4.93/4.93 px at 180 dpi. The larger-font
stress case also fits without overlap. Font/alphabet reference dimensions keep
the grid and rectangle height stable when sequence letters change, including
DNA case and ambiguity-code variants. Chrome and Inkscape inspection agree.
Twenty-seven additional tests cover centering, both strands/fonts, preserved
glyph scale and columns, invalid spacing units, tight-pitch rejection and
diagnostic-to-painted-bound agreement. The package suite now passes 686 tests.
Strict bottom-legend fitting remains open in C; spacer, artwork/caption overflow and the
referenced-font policy are qualified below. This is not completion of C–G or real
consumer qualification.

The CLI/API admission review found that `job validate` accepted negative glyph
spacing, unknown style fields and nonexistent presets that rendering rejected.
Job parsing now resolves the existing style contract before returning a job.
Seven failing public-path/mapping/CLI checks reproduced the gap; all eight new
admission tests now pass, including valid explicit geometry with no published
bundle. The complete package suite passes 694 tests. Preset lookup and rendering
remain owned by the existing style module; no duplicate validator was added.

C now also admits `Feature.render.annotation_label` with explicit text and
`near_feature`/`inline` placement. DenseGen/GenBank adapters translate their labels
into that neutral declaration; the renderer no longer branches on producer
names or trims scientific prefixes. Invalid nested fields fail at record
admission. Declared annotations retain exact text and font size; insufficient
space raises instead of silently omitting them. Shared label measurements now
reserve vertical room in layout. This exposed and fixed an older 24-point example
that silently lost both site labels. Nine new public contract checks plus the
adapter regression pass; the complete package suite passes 703 tests.

The label responsibilities are now separated into
[`render/feature_labels.py`](../../../src/dnadesign/baserender/src/render/feature_labels.py).
It produces immutable label placements before figure allocation; the sequence
renderer only paints that plan. Both label modes account for reserved overlay
space, and inline labels also avoid already planned near-feature labels. This
removes about 500 lines from the sequence-row orchestrator without adding a
public controller or wrapper. Two negative tests exposed allocation leaks from
bad annotation colors and unfittable inline text; both now fail before allocation.
The package suite passes 705 tests. A 60/100-base forward/reverse proof at
`src/dnadesign/baserender/outputs/annotation-contract-20261118-02/index.html`
preserves 14-point sequence text and 24-point annotations in Chrome. Its natural
point-sized view scrolls for the longer sequence rather than shrinking type.

`SequenceLayout` now owns explicit point-based pitch, canvas, forward grid origin,
minimum content extents and footer clearance. Strict mapping admission and frozen
values live in configuration; measured canvas placement lives in
[`render/layout_frame.py`](../../../src/dnadesign/baserender/src/render/layout_frame.py).
Both reuse the existing layout engine. Native SVG, figure rendering and image
jobs share the contract; registered option normalization also serves CLI job
validation without constructing a renderer or loading NumPy/Matplotlib. Conflicting
legacy metadata, bad units, insufficient pitch/canvas and non-finite arithmetic
fail before allocation. No fitting or font reduction is introduced by this path.

The neutral review at `src/dnadesign/baserender/outputs/layout-contract-20261118-01/`
contains public-API 60/100-base SVGs and a real CLI image job. Both CLI outputs
have exact `0 0 1200 400` viewBoxes; the API examples pin the same (80, 200)-point
grid origin and 10-point pitch. Browser inspection preserves the requested type.
Twenty-nine new tests exercise both public surfaces, 72/144-dpi geometry,
mapping isolation, empty/small-feature glyph cases and negative admission.
Image jobs still use the existing figure-based illustration output; D must
integrate native SVG composition and qualify raster/frame parity. Video layout
is explicitly rejected until that work passes; forwarding options into the
current compositor alone would still trigger competing metadata and hidden fit.

Sequence rows now separate `prepare` from `draw_prepared` internally. The
prepared value binds the admitted record, style, palette and measured geometry;
native output can inspect that plan before allocating a figure. The previous
private `render_with_layout` tuple route is removed. Four 60/100-base SVGs with
12/18-point captions remain byte-identical across this preparatory refactor;
their comparison is retained in `outputs/vector-placement-20261118-01/` under
BaseRender. All 734 package tests still pass.

Native illustration placement now rejects horizontal overflow, keeping the
calculated anchors. Captions are measured before figure allocation using the
resolved font's literal text advances and ink bounds. Serialization uses that
same admitted origin. Diagnostics include caption origins/bounds and frozen
`ResolvedFont` identities under an explicit `reference` text-font policy. Missing
glyphs, unsupported control characters, extreme units and collapsing SVG aspect
ratios fail with context. The original raster clamping route remains supported.
The checked-in example now uses `SequenceLayout`, pins its DPI and records font
and caption geometry. Native SVG still contains no raster images. The review at
`outputs/vector-placement-20261118-03/` under BaseRender includes a transparent
Inkscape PNG and Chrome inspection of the 60/100-base examples. Caption and body
geometry agree across independently sized captions; font-file embedding and
arbitrary source-SVG font interpretation are not claimed.
Eighteen new placement/font tests pass, including explicit recovery through a
larger typed canvas. The full package suite passes 752 tests. The browser review
resets per-card horizontal scroll before comparing caption sizes; automatic
`scrollIntoView` had panned one review card by 16 CSS pixels, while its SVG bounds
were unchanged. The aligned screenshot records the actual common frame.

Spacer geometry now owns font selection, gap and occupied bounds once. Drawing
and annotation collision checks consume that plan, removing about 100 lines of
duplicated calculation. Native/explicit-layout preparation preserves fractional
point sizes, rejects conflicting shrink requests and fails on insufficient span
or canvas bounds, including unlabelled brackets. Ordinary figure fitting stays
supported. Native diagnostics report spacer placement by source effect index.
The contract review also found that accepted read-only span mappings failed in
the drawer and that explicit spans bypassed coordinate admission; both are fixed.
Fifteen new tests cover those cases and actual drawn-text containment on both
lanes. A valid Yiu diagram exposed a natural-canvas omission for explicit-span
links; natural measurement now includes all spacer extents, reusing the same
primitive geometry. Both output routes expand naturally, while a fixed canvas
rejects content that cannot fit. An initial high-track test mistook valid natural
expansion for clipping; that hypothesis was corrected before qualification.
All 767 package tests pass, including the Yiu job. Evidence and the matched-font
native SVG regression comparison live in
`outputs/span-type-contract-20261118-01/` under BaseRender.

F must also pin the producer source revision alongside its distribution version:
the current talk preparer checks `dnadesign-tools` version alone, which cannot
distinguish changes within an editable checkout. Keep the existing v2 comparison
as retained evidence until both caller migrations verify.

### Context and Orientation

Work inside `src/dnadesign/baserender/`, following its `AGENTS.md` and the
[repo change gate](../../repo-change-gate.md). Public imports are
`dnadesign.baserender`; consumers must not import sibling internals. Keep private
consumer paths/identities out of this public package. Use neutral 60/100-base
fixtures here; qualify the real pinned records in their consuming workspace.

The inspected runtime baseline is `ff88f42ba95995c9c4db5bc6b0d1f857f5057bf0`
(`dnadesign-tools` 0.4.1); the proposal is local commit `2b729d719` on
`plan/baserender-vector-api`. A rechecks the working revision and dirty state
before recording implementation evidence. This plan does not assert that an API
exists in an installed distribution.

| Current surface | Contract/responsibility to inspect |
| --- | --- |
| `src/public/api.py`, `sequence_panel.py` | Small public entrypoints, explicit sizing and artifact/diagnostic results; preserve lazy imports. |
| `src/config/`, `src/core/` | Boundary parsing, versioned asset/layout/label records, units, supported fields and failures. |
| `src/render/layout.py` | One authority for pitch, strands, feature boxes, caption/illustration bounds. |
| `src/render/feature_labels.py` | Producer-neutral annotation planning, collision checks and painting of admitted placements. |
| `src/render/effects/anchored_illustration.py`, `assets/overlays/` | Separate source resolution, geometry and format-specific emission; preserve SVG definitions and painter order. |
| `src/render/sequence_rows.py` | Neutral label placement and focused drawing helpers; remove producer-name policy from core after adapter migration. |
| `src/outputs/images.py`, `video.py` | Shared placement, final conversion, deterministic output and create-only publication. |
| `tests/`, `docs/`, catalog/CLI | Public boundaries, packaged resources, failure cases and one runnable first-use path. |

### Plan of Work

A establishes the fixed review baseline. B is the first end-to-end tracer. C/D
expand its contract and outputs without breaking current users. E qualifies the
batch path before wider migration. F moves each consumer independently. G removes
only proven replacements and closes audit findings. Each dependent slice starts
after its predecessor's acceptance; independent documentation/audit work can
continue when a specific render capability is blocked.

### Concrete Steps

1. **A — Baseline and findings.** Freeze source revision and diff scope, run the
   BaseRender suite, catalog and one current API/job path. Inventory active
   callers, config versions and known source defects. Audit API/config through
   assets/layout/drawing/output/tests/docs against ownership, contracts, DRY
   knowledge and failure behavior. Produce ranked findings with evidence, failed
   criteria, remediation slice and deferred coverage. Separate observed defects
   from unverified risks; do not claim a package audit from file sizes alone.
2. **B — Native vector tracer.** Define the smallest asset/caption and output
   contract needed by a public SVG operation. Reuse the bundled SVG with exact
   body/caption selections and named anchors; support explicit caller-owned,
   hash-pinned selections without copying their files into the package. Separate
   loading, measured placement and SVG emission within existing tool boundaries.
   Add tests that currently fail because the illustration exports as PNG, then
   prove vector paths plus independently styled text at 60 and 100 bases.
3. **C — Layout and label ergonomics.** Expose font roles, base pitch, strand
   anchor, clearance, bounds, transparency and preserve-type/fit policy with
   explicit units. Return measured anchors/bounds and resolved font identities.
   Replace producer-name triggers with neutral placement declarations through
   the owning integrations. Unknown fields, missing fonts/IDs and overflow must
   fail with actionable bounds/context; never shrink or omit labels silently.
   Extract only the touched drawing/config/API responsibilities with stable tests.
4. **D/E — Outputs and performance.** Route SVG, PNG and representative frame
   output through the same measured composition. Keep rasterization at the final
   boundary, preserve alpha, and admit new settings through the existing CLI/job
   contract rather than another DSL. Benchmark cold/warm and batch cases before
   profiling/optimizing measured bottlenecks. Reuse bounded asset/font/layout
   preparation where justified, keyed by content and resolved settings; test
   invalidation. Do not add unbounded caches, a video engine or a new system
   dependency without a documented need. Qualify packaging, catalog and docs.
5. **F — Migrate in two batches.** First qualify the graph-plus-sequence use,
   then the compact library use with their existing pinned selections. Both use
   the same producer style/layout authority. Preserve actual coordinates, source
   identities, spacer semantics, font minima and neighboring artwork. Record
   source hashes, repeat commands and visual/native-output limits per batch.
   Keep record adaptation, page composition and timing at the consumer.
6. **G — Contract and review.** Only after both batches pass, remove displaced
   consumer asset handling, typography/layout workarounds, artist/GID inspection
   and duplicate SVG composition. Keep legitimate page-placement code and all
   scientific provenance. Retire obsolete routes/config selections with explicit
   migration guidance; retain still-valid public raster/figure APIs. Freeze the
   final diff and review standards and specification axes separately. Close or
   explicitly defer findings, align owner docs, and preserve recoverable sources.

#### Completed executor contract — A then B

**Goal:** one public BaseRender call preserves an SVG illustration and a separate,
independently styled caption on neutral 60/100-base examples. No consumer repair
is needed to restore vector artwork or separate its caption.

**Allowed:** BaseRender source/contracts/tests/docs and fresh disposable review
outputs, after A records the full baseline. B may introduce minimal public types
and SVG output plumbing; C's full layout migration, optimization, video encoding,
consumer rewrites, original SVG edits and publishing remain outside this slice.

**Done:** the selected vector fixture exports without an embedded raster image;
caption family/size/offset/visibility vary independently of body scale; anchors,
transforms, gradients and painter order remain correct. Existing supported paths
still pass. Missing/unsupported sources fail before publication. Both neutral
examples use only public imports and receive diagnostics sufficient for placement.

**Verify:** red/green structural and failure tests, focused public/import/resource
checks, then the BaseRender suite and static/docs gates. Inspect SVG in a browser
and Inkscape, plus a faithful raster preview, at intended placement sizes. If the
chosen backend cannot preserve vector semantics, record the failed criterion and
revise that bounded approach; do not substitute a raster or rewrite the renderer.

**Handoff:** commit/diff identity, exact commands and test results, source/style
hashes, SVG plus preview, measurements and a short list of unresolved findings.
The next executor updates this plan rather than creating a competing roadmap.

#### Next executor contract — C

Remove implicit native bottom-legend font shrinking through a measured plan and
explicit insufficient-space errors. Preserve the new spacer, caption, layout and
neutral annotation contracts, the A/B proofs and supported legacy raster fitting.
Qualify crowded/reverse-strand cases and keep font-policy
claims limited to declared referenced fonts; consumer viewers need their own
inspection. Then audit C against its original criteria before starting D.

### Validation and Acceptance

| Area | Required proof |
| --- | --- |
| Vector/caption fidelity | Selected fixture has paths and editable text; no raster image for vector-declared artwork. Check transform/gradient/clip definitions, repeated-instance ID collisions and independent caption sizing. |
| Placement | Fixed pitch/type/strand anchors across 60/100-base and crowded/reverse-strand cases; outputs share one geometry, no white masking, clipping or hidden rescaling. Real consumers preserve their saved exemplar minima and adjacent arrows. |
| Fail-fast admission | Missing/duplicate selections, changed hashes, unknown fields, bad units, missing fonts, malformed SVG, active content, unpinned external references and insufficient bounds fail clearly before final publication. Unused definitions are pruned by selection closure. |
| Ownership | No scientific producer names in neutral drawing policy; no private paths, sibling-internal imports, copied analysis logic or writes into author assets. Test public lazy-import and package-resource contracts. |
| Performance | Stage-separated cold/warm/batch measurements with input/style/output identities; compare median/tail and memory. Set numeric budgets from the tracer, not the earlier three-observation probe. Demonstrate gains without reducing fidelity or repeatedly rendering identical states. |
| Migration/recovery | Both caller batches verify before deletion. Retain prior qualified artifacts/recipes; each slice is a reviewable commit. No automatic source upgrade or writeback; new schema/default behavior is explicit. |

Run from the owning repository, recording counts, skips and failures:

```sh
uv run --locked pytest -q src/dnadesign/baserender/tests
uv run --locked ruff check src/dnadesign/baserender pyproject.toml
uv run --locked ruff format --check src/dnadesign/baserender
uv run --locked python -m dnadesign.devtools.architecture.boundaries --repo-root .
uv run --locked python -m dnadesign.devtools.docs.checks --repo-root .
uv run --locked baserender catalog --json
uv run --locked baserender job validate --workspace demo_densegen_render --workspace-root src/dnadesign/baserender/workspaces
```

Consumer migration uses its own regression suite and visual review; those results
do not qualify native presentation playback automatically. Broaden monorepo
checks only for actual cross-tool changes or merge-depth validation. If a source
or dependency is unavailable, keep the failure evidence and block only dependent
work; no historical artifact substitution or system installation is implied.
