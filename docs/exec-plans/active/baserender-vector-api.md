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
- [ ] (2026-11-18 02:06Z) A — Complete the package baseline and prioritized maintainability audit before runtime edits.
- [ ] (2026-11-18 02:06Z) B — Prove native SVG artwork and independent vector captions through a public call.
- [ ] (2026-11-18 02:06Z) C — Expose explicit layout/label controls and documented diagnostics.
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
The full package audit is pending. Focused public-playback, sequence-panel-layout
and runtime/public-API tests, catalog output and demo-job validation passed during
planning; this is not a full-suite claim.

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

Planning only. The implementation rationale is the
[vector-assets proposal](../../dev/plans/tools/baserender/vector-assets-and-layout.md).
No executable, asset, source-selection or release change is made by this plan.
Update this section with actual results, deviations and residual risks per slice.
No blocking user question is currently required; public naming and backend details
are executor decisions constrained by the tracer and acceptance criteria below.

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

#### Next executor contract — A then B

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
