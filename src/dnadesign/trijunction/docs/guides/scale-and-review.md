# Scale and Quality Review

**Type:** guide
**Audience:** users checking larger requests or reviewing a plan
**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-02

The request schema allows at most 100,000 targets, but target count alone does
not determine whether a request fits. Capacity also depends on target length,
physical-pool grouping, and search settings. A request can be valid YAML and
still require too much work or yield too few candidates. Run `preflight` on the
exact request before writing a bundle.

## Keep Targets Together When Their Oligos Will Mix

| Intended work | Request shape | What preflight checks |
| --- | --- | --- |
| One roughly 1 kb target | One target in one pool | The exact geometry, search policy, and order ceiling fit. |
| One roughly 10 kb target | One target in one pool | The larger locus set remains within the request and software limits. |
| 100 roughly 1 kb targets in one physical pool | All targets share one `pool_id` | The tested example jointly plans 600 junctions and 1,600 order rows. The exact request still controls search depth. |
| 1,000 roughly 1 kb targets in one physical pool | All targets share one `pool_id` | The request format can describe the pool, but v1 rejects the planned work before creating large search data. This is not a successful pooled design. |
| 100 or 1,000 targets in independent physical pools | Distinct `pool_id` values | Each pool is designed independently while request-wide aggregate work is still bounded. |

The scale tests cover exact 1,000-base and 10,000-base targets, 100 jointly
designed 1,000-base targets, and 1,000 independently designed 1,000-base
targets. The successful multi-target cases evaluate several toehold paths and
barcode subsets. They test software load and pool grouping; they do not show
that the search settings are experimentally sufficient. The tests also submit
one shared pool of 1,000 1,000-base targets. V1 rejects it before creating its
large search state. That rejection is expected and tested.

Maintainers can rerun the exact matrix with:

```bash
uv run pytest -q src/dnadesign/trijunction/tests/scenarios/test_scale_dogfood.py
```

Use one shared `pool_id` only when the oligos will mix in one physical pool. If
that request exceeds a limit, review the search settings or change the physical
design. Splitting the pool only to make the software pass would remove the
cross-target sequence checks.

Independent pools may be split across requests when their combined request is
too large. Each pool has its own deterministic search, so the split preserves
the checks within that pool. Keep target IDs unique within each request and
retain every request with its bundle.

## Preflight Before Publication

```bash
uv run trijunction preflight request.yaml --format json
uv run trijunction plan request.yaml --format json
```

`preflight` runs the complete design without writing a bundle. It stops before
creating candidate pools or large search data when a request exceeds a limit.
`plan` returns the full in-memory result only after the same checks pass.
Preflight also estimates the largest possible bundle files before search. If
an estimate is too large, it fails without writing files. Reduce the request,
or split only physical pools that are genuinely independent.

The v1 contract has several independent limits:

- request files are at most 16 MiB;
- declared search fields have integer ceilings;
- each physical pool has CPU, memory, and sequence-state limits; and
- the complete request has aggregate limits for pools, targets, input bases,
  loci, candidates, dynamic-programming work, caches, and sampled state.

Resource failures stop with an actionable message that identifies the kind of
work that exceeded a limit. See the [contract
reference](../reference/contracts.md#declared-and-derived-resource-ceilings)
for the stable public ceilings.

## When the Barcode Search Finds Too Few Candidates

The barcode substring parameters `barcode_toehold_k` and `barcode_pair_k` are
fixed declarations. If they do not permit the requested candidate pool within
`barcode_generation_attempts`, TriJunction fails. It does not automatically
increase either value or weaken another constraint.

This differs from the pooled Sidewinder paper. Its described software derives
starting `q` and `k` values from sequence lengths, then alternates increasing
them until it finds at least `5|T|` candidates or can no longer meet that
threshold. TriJunction leaves both values unchanged. You can review the failure
and submit a new request with different values while preserving the original
request.

## Review a Bundle

1. Read `checks.json` for the compact invariant results.
2. Inspect pool search receipts and target reconstruction in `plan.json`.
3. Review every complete primer and fragment sequence in
   `orders/oligos.tsv` against the owning synthesis and experimental process.
4. Treat `thermodynamic_screening: not_run` as an unresolved item, not as a
   passed thermodynamic check.

### Create optional review images

Every verified bundle contains
`views/three_way_junction_review.v1.json`, with one review record per target.
BaseRender can turn each record into an original four-panel QA image showing
target tiling, selected junctions, strand and recovery geometry, search
results, and unresolved checks. Follow the runnable
[TriJunction review integration](../../../baserender/docs/integrations/trijunction.md)
to keep the source bundle and render output separate.

The JSON review records are part of the verified TriJunction bundle; rendered
images are not. Save images in a new BaseRender bundle beside the source
design. Rendering does not change the TriJunction bundle, copy figures from the
papers, or add thermodynamic or laboratory evidence. For large pools, write
one image per target or choose a bounded subset instead of making an unreadable
contact sheet. BaseRender checks the complete source before applying a
selection, so the source must still fit its 64 MiB, 2,000-row, and
10-million-base limits. See the integration runbook for the full constraints.
