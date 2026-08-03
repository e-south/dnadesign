---
doc_id: baserender-junction-integration
title: junction review integration
owner: dnadesign-maintainers
status: active
last_verified: 2026-08-02
---

# junction review integration

**Type:** route
**Plane:** downstream-tool
**Owner-boundary:** baserender
**Entry artifact:** verified junction `views/three_way_junction_review.v1.json` plus an explicit BaseRender `RenderJobV4` job
**Exit artifact:** create-only BaseRender review bundle containing per-target images and its render manifest
**Use when:** you want one optional four-panel review image per selected target
**Input:** `views/three_way_junction_review.v1.json` from a verified junction bundle
**Output:** a separate private BaseRender bundle
**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-02

Use BaseRender to make optional review images from a junction design.
junction plans and verifies the sequences. BaseRender reads the saved review
records and draws them; it does not recompute the design or change the source
bundle.

## What the image shows

The shared `dnadesign.contracts.visual.ThreeWayJunctionReviewV1` model defines
each JSON row. Its exact `contract_kind` is
`three_way_junction_review_v1`; unknown fields fail validation.

Each image has four panels:

1. target coordinates, fragment domains, and selected toeholds;
2. the selected toehold-to-barcode assignments;
3. strand roles and declared recovery primers;
4. search counts, checks, and the fact that thermodynamic screening was not run.

For targets with at most five junctions, the junction panel shows bounded
sequence previews with the total nucleotide length and a 12-character SHA-256
prefix. Larger targets show a bounded set of junction identifiers instead. The
recovery panel also uses bounded sequence previews. An ellipsis means the label
is only a review aid; use the review JSON and `orders/oligos.tsv` for complete
sequences.

The contract keeps primer mechanics separate:

- `binding_sequence_5to3` is the target-binding portion;
- `five_prime_extension_5to3` is an uninterpreted extension and may be empty;
- `order_sequence_5to3` must equal extension plus binding sequence;
- `target_binding_span` records the checked terminal target match.

The contract checks every barcode-bearing and complement sequence against the
target domains and adjacent junctions. It also records the extension-aware top
and bottom recovery products and requires them to be reverse complements.
Every check names either a `target` or an `assembly_group` as its subject.

The same file supports target-specific and universal recovery. A primer may
carry a reviewed 5-prime extension for later Type IIS work, but BaseRender does
not infer an enzyme, cut geometry, or cloning plan.

`thermodynamic_screening` is restricted to `not_run`. String-distance search
receipts are not thermodynamic or experimental validation.

## Verify the design first

BaseRender validates the review records, but it does not read the junction
manifest or establish where the JSON came from. Verify the source bundle before
rendering:

```bash
uv run junction verify review-root/verified-design --format json
```

## Create a separate review bundle

Keep the job file, verified source bundle, and review destination under one
review directory. Save the job as `review-root/review.job.yaml`:

```text
review-root/
├── review.job.yaml
├── verified-design/
│   └── views/
│       └── three_way_junction_review.v1.json
└── reviews/
```

Paths are resolved relative to the job file. Point the input at the review JSON
in the junction bundle, and choose a new output directory beside, not inside,
the source bundle:

```yaml
version: 4
contract:
  kind: three_way_junction_review_render_v1
bundle:
  path: reviews/design-v1
input:
  kind: json
  path: verified-design/views/three_way_junction_review.v1.json
  adapter:
    kind: three_way_junction_review_v1
  alphabet: DNA
render:
  renderer: three_way_junction_review
  style:
    preset: null
    overrides: {}
outputs:
  - kind: images
    dir: images
    fmt: svg
run:
  strict: true
  fail_on_skips: true
```

Run through the stable API or CLI:

```python
import dnadesign.baserender as baserender

job = baserender.validate_job("review-root/review.job.yaml")
report = baserender.run_job(job)
```

```bash
uv run baserender job validate review-root/review.job.yaml
uv run baserender job run review-root/review.job.yaml
```

The review records contain complete DNA sequences. BaseRender therefore creates
directories with mode `0700` and files with `0600`. It installs `bundle.path`
atomically and fails if that path already exists. The input JSON and the
junction bundle remain unchanged.

The JSON file has one row per target, so `dir: images` writes one target-named
image per row. Filename collisions receive deterministic numeric suffixes.
BaseRender rejects review JSON or an optional selection CSV above 64 MiB. It
also accepts at most 2,000 review rows, 2,000 selection rows, and 10,000,000
target bases. These checks run on the complete source before `input.limit` or a
selection narrows the rendered rows. Filtering cannot make an oversized source
acceptable, and there is no direct BaseRender route for a source above these
limits.

This adapter writes per-target images only. It rejects video and
`outputs[].path`. Do not edit or split the verified source file in place.

## Literature and visual design

The review terms follow the three-way-junction and pooled-recovery concepts
described by Robinson *et al.* in
[the Sidewinder paper](https://doi.org/10.1038/s41586-025-10006-0) and
[the pooled extension](https://doi.org/10.64898/2026.05.01.722326).
The four-panel layout is original. It organizes target geometry, assignments,
strand and recovery declarations, and software checks without copying the
papers' figures or claiming their experimental results.

For the method mapping and implementation limits, use junction's
[method reference](../../../junction/docs/reference/method-v1.md) and
[sources and scope](../../../junction/docs/reference/sources.md).

## Code boundaries

- Tools that write these records may import `dnadesign.contracts.visual`.
- Code that renders them uses `dnadesign.baserender`; it does not import
  the private `dnadesign.baserender.src.*` package.
- Study names, objectives, rankings, and campaign state do not belong in this
  contract.
- The review image is advisory. It is not part of junction plan identity,
  offline verification, ordering, or laboratory validation.
