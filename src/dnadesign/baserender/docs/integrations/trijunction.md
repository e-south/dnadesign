---
doc_id: baserender-trijunction-integration
title: TriJunction review integration
owner: dnadesign-maintainers
status: active
last_verified: 2026-08-02
---

# TriJunction review integration

**Type:** runbook
**Plane:** downstream-tool
**Owner-boundary:** TriJunction owns the verified design bundle; BaseRender owns the separate review bundle
**Entry artifact:** verified `views/three_way_junction_review.v1.json` from a TriJunction bundle
**Exit artifact:** create-only private BaseRender bundle with one optional QA image per selected target
**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-02

TriJunction owns sequence planning, search evidence, strand composition,
recovery-primer declarations, and verification. BaseRender consumes the
verified bundle's strict, study-neutral review records and produces optional
four-panel QA images. It does not recompute the design or change its verified
source bundle.

## Contract and renderer

The shared `dnadesign.contracts.visual.ThreeWayJunctionReviewV1` model defines
the file boundary. Its exact `contract_kind` is
`three_way_junction_review_v1`; unknown fields fail validation.

The four panels have stable meanings:

1. target coordinates, fragment domains, and selected toeholds;
2. explicit toehold-to-barcode assignments;
3. strand roles and declared recovery primers;
4. search receipts, checks, and the unresolved thermodynamic boundary.

The fixed-width junction and recovery panels display bounded sequence previews
with total nucleotide length and a 12-hex-character SHA-256 prefix. The
ellipsis is explicit: rendered labels are review aids, not exact ordering
evidence. Complete toehold, barcode, primer-binding, extension, and order
sequences remain in the typed review record and TriJunction order evidence.

The contract keeps primer mechanics separate:

- `binding_sequence_5to3` is the target-binding portion;
- `five_prime_extension_5to3` is an uninterpreted extension and may be empty;
- `order_sequence_5to3` must equal extension plus binding sequence;
- `target_binding_span` proves the terminal target match.

The same contract proves every barcode-bearing and complement sequence from
the target domains and adjacent junction assignments. It also carries the
extension-aware top and bottom recovery products and requires them to be exact
reverse complements. Checks identify their structural subject explicitly as
the current `target` or `pool`; renderer code never assigns scope by parsing a
free-text detail.

This representation supports target-specific and universal recovery without
making cloning assumptions. A later Type IIS workflow can declare a reviewed
5-prime extension, but BaseRender does not infer an enzyme, cut geometry, or
cloning plan from that sequence.

`thermodynamic_screening` is restricted to `not_run`. String-distance search
receipts are not thermodynamic or experimental validation.

## Create a separate review bundle

Keep the job file, verified source bundle, and review destinations under one
explicit review root. Save the job as `review-root/target-01.review.job.yaml`
and run BaseRender from `review-root/`:

```text
review-root/
├── target-01.review.job.yaml
├── verified-design/
│   └── views/
│       └── three_way_junction_review.v1.json
└── reviews/
```

Paths in the job are resolved relative to the job file. Point the input at the
canonical review array in the published TriJunction bundle, and choose a new
output directory beside—not inside—that verified source bundle:

```yaml
version: 4
contract:
  kind: three_way_junction_review_render_v1
bundle:
  path: reviews/target-01-v1
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

job = baserender.validate_job("target-01.review.job.yaml")
report = baserender.run_job(job)
```

```bash
uv run baserender job validate target-01.review.job.yaml
uv run baserender job run target-01.review.job.yaml
```

The review records contain complete DNA sequences, so BaseRender creates this
review bundle with owner-only access: directories use mode `0700` and files use
mode `0600`. It creates `bundle.path` atomically and fails when that destination
already exists. The input JSON and its verified TriJunction bundle remain
unchanged. Other BaseRender contract kinds keep their declared publication
permissions.

The JSON file is an array with one row per target. `dir: images` therefore
writes one stable target-named image per row. Sanitized or case-insensitive
filename collisions receive deterministic numeric suffixes. Before full source
capture, BaseRender rejects review JSON larger than 64 MiB. Before record
materialization, it also enforces at most 2,000 review rows and 10,000,000 total
target bases. `input.limit` and selection only control which records are
rendered; neither is a source-capture guard. Do not edit or split the verified
source file in place.

## Literature and visual boundary

The review vocabulary follows the three-way-junction and pooled-recovery
concepts described by Robinson *et al.* in
[the Sidewinder paper](https://doi.org/10.1038/s41586-025-10006-0) and
[the pooled extension](https://doi.org/10.64898/2026.05.01.722326).
The BaseRender panel is an original QA composition: it is informed by the
papers' assembly stages but does not reproduce their figures or claim their
experimental results.

For the method mapping and implementation limits, use TriJunction's
[method reference](../../../trijunction/docs/reference/method.md) and
[source boundary](../../../trijunction/docs/reference/sources.md).

## Ownership rules

- Producers may import `dnadesign.contracts.visual` to publish the shared
  contract.
- Consumers use `dnadesign.baserender`; they do not import
  `dnadesign.baserender.src.*`.
- Study names, objectives, rankings, and campaign state do not belong in this
  contract.
- The review image is advisory. It is not part of TriJunction plan identity,
  offline verification, ordering, or laboratory validation.
