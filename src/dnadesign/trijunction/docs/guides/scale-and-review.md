# Scale and Quality Review

**Type:** guide
**Audience:** operators preflighting larger requests or reviewing a plan
**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-02

There is no single supported target count. Capacity depends on target length,
physical-pool grouping, and search budgets. A request that parses can still be
too expensive or too constrained to design, so run `preflight` on the exact
request before publication.

## Preserve the Physical-Pool Boundary

| Intended work | Request shape | What preflight proves |
| --- | --- | --- |
| One roughly 1 kb target | One target in one pool | The exact geometry, search policy, and order ceiling fit. |
| One roughly 10 kb target | One target in one pool | The larger locus set remains inside the declared and implementation resource envelopes. |
| 100 roughly 1 kb targets in one physical pool | All targets share one `pool_id` | The tested example jointly plans 600 junctions and 1,600 order rows. The exact request still controls search depth. |
| 1,000 roughly 1 kb targets in one physical pool | All targets share one `pool_id` | The schema represents the intended topology, but the v1 test rejects its search state before allocation. This shape is not currently a successful pooled design. |
| 100 or 1,000 targets in independent physical pools | Distinct `pool_id` values | Each pool is designed independently while request-wide aggregate work is still bounded. |

The executable scale scenarios use exact 1,000-base and 10,000-base targets,
100 jointly designed 1,000-base targets, and 1,000 independently designed
1,000-base targets. The multi-target success cases exercise more than one
toehold path and barcode subset; they are bounded load and topology checks,
not claims that a minimal search budget is experimentally sufficient. A
single shared pool of 1,000 1,000-base targets is also exercised and is
currently rejected before allocation by the v1 resource envelope. That
refusal is an explicit supported outcome, not successful pooled design or a
reason to weaken the physical-pool boundary.

Maintainers can rerun the exact matrix with:

```bash
uv run pytest -q src/dnadesign/trijunction/tests/scenarios/test_scale_dogfood.py
```

Use one shared `pool_id` only when the oligos will coexist in one physical
pool. If a shared-pool request exceeds an envelope, lowering explicit budgets
or changing the intended physical design are reviewable choices. Artificially
splitting that pool would remove cross-target string checks and is not an
equivalent workaround.

Independent pools may be split across requests when the aggregate request
envelope is exceeded. Because each pool receives its own deterministic search
stream, that split does not weaken within-pool string checks. Keep target IDs
globally unique within each request and retain every request with its bundle.

## Preflight Before Publication

```bash
uv run trijunction preflight request.yaml --format json
uv run trijunction plan request.yaml --format json
```

`preflight` runs the complete design path without a durable write. It rejects
unsafe work before candidate pools or large search state are materialized.
`plan` returns the full in-memory evidence only after the same checks pass.

The v1 contract has several independent limits:

- request files are at most 16 MiB;
- declared search fields have integer ceilings;
- each physical pool has CPU, memory, and sequence-state envelopes; and
- the complete request has aggregate limits for pools, targets, input bases,
  loci, candidates, dynamic-programming work, caches, and sampled state.

The failure message names the exceeded dimension, requested amount, limit, and
the explicit adjustment surface. See the [contract
reference](../reference/contracts.md#declared-and-derived-resource-ceilings)
for the stable public ceilings.

## Constraint Exhaustion Is Not a Resource Failure

The barcode substring parameters `barcode_toehold_k` and `barcode_pair_k` are
fixed declarations. If they do not permit the requested candidate pool within
`barcode_generation_attempts`, TriJunction fails. It does not automatically
increase either value or weaken another constraint.

That policy intentionally differs from the pooled Sidewinder paper, whose
described software starts from length-derived `q` and `k` values and alternates
increasing `k` and `q` until at least `5|T|` candidates exist, or until the
process can no longer meet that threshold. In TriJunction, an operator may
review the failure and submit a new request with different explicit values;
the original request remains reproducible.

## Review Evidence in Layers

1. Read `checks.json` for the compact invariant results.
2. Inspect pool search receipts and target reconstruction in `plan.json`.
3. Review every complete primer and fragment sequence in
   `orders/oligos.tsv` against the owning synthesis and experimental process.
4. Treat `thermodynamic_screening: not_run` as a required unresolved boundary,
   not as a passed thermodynamic check.

### Optional BaseRender projection

Visual review is a separate, read-only consumer concern. Every verified bundle
contains `views/three_way_junction_review.v1.json`, a canonical array with one
strict review record per target. BaseRender may consume that file to create an
original literature-inspired four-panel QA image covering target tiling,
selected junctions, strand and recovery geometry, string-search evidence, and
unresolved checks. Follow the runnable
[TriJunction review integration](../../../baserender/docs/integrations/trijunction.md)
to keep the source bundle and render output separate.

The JSON projection is part of the verified TriJunction bundle. Rendered
images are not: keep them in a new create-only BaseRender bundle beside the
source design. BaseRender must not copy figures from the cited papers, mutate
the TriJunction bundle, become required for `preflight`/`plan`/`build`/`verify`,
or imply that thermodynamic or laboratory validation ran. For large pools,
write one image per target or select a bounded review subset instead of making
one unreadable contact sheet.
