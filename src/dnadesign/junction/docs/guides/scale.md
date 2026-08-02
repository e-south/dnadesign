# Scale

**Type:** guide
**Audience:** users evaluating larger requests
**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-02

Target count alone does not determine cost. Work also depends on target length,
the number of loci in each assembly group, candidate width, sequence lengths,
and search budgets.

## Understand the grouping cost

Targets in one assembly group share a joint toehold and barcode search. Adding
a target can change existing assignments and increases pairwise work. Targets
with different `assembly_group_id` values are searched independently, but the
request-wide workload is still bounded.

Use separate groups only when the underlying assemblies are genuinely
independent. Splitting one intended joint assembly to make a request pass would
omit the cross-target checks.

## Use the right command

```bash
uv run junction preflight request.yaml --format json
```

`preflight` performs the complete design search and returns a short summary. It
is not a cheap static estimate. Before the expensive stages, the planner does
predict locus counts, derived artifact sizes, and guarded search work; requests
outside those envelopes fail before candidate materialization.

Use `plan` instead when you need full no-file JSON. Use `build` directly when
you are ready to publish. Running all three repeats the complete search.

## Tested software scenarios

The repository has deterministic scenarios for:

- one exact 1,000-base target, published and replay-verified;
- one exact 10,000-base target, published and replay-verified;
- 100 exact 1,000-base targets searched in one assembly group and planned;
- 1,000 exact 1,000-base targets searched as independent assembly groups,
  published, and replay-verified; and
- 1,000 exact 1,000-base targets submitted as one assembly group and rejected
  before the prohibited large search state is created.

Run them with:

```bash
uv run pytest -q src/dnadesign/junction/tests/scenarios/test_scale_dogfood.py
```

Together, these scenarios test schema expressivity, deterministic planning,
guarded resource behavior, and selected artifact round trips. They do not
validate the search policy, thermodynamics, oligo synthesis, pooled assembly,
or experimental performance for those target counts or lengths.

## Limits and failure behavior

V1 has independent ceilings for:

- request bytes and integer declarations;
- assembly groups, targets, input bases, loci, and candidates;
- encoded sequence state and distance caches;
- dynamic-programming cells and distance lookups;
- barcode-generation work;
- sampled search state and matching substring work; and
- projected and realized artifact bytes.

The [request contract](../reference/request.md#request-wide-resource-envelope)
summarizes public request limits. Source-level constants and formulas live in
`design/resources/limits.py` and `design/resources/estimates.py`.

When a guard fails, lower a scientifically reviewable budget, reduce the
request, or divide only independent assembly groups across requests. Preserve
the rejected request rather than mutating it invisibly.

## Candidate exhaustion is not a resource error

A request can fit every resource limit and still fail to find enough barcode
candidates under its declared GC, homopolymer, and substring constraints.
`junction` does not weaken those constraints automatically.

The pooled paper describes a different procedure that adjusts its substring
thresholds when candidate generation is insufficient. V1 keeps
`barcode_toehold_k` and `barcode_pair_k` fixed. A reviewed follow-up request may
change them, but it is a new input and normally a new plan identity.

## Memory behavior during publication

Bundle publication and replay render one artifact at a time. This removes the
former all-payload retention peak, but each JSON or TSV renderer still creates
one complete artifact in memory. Treat the 256 MiB per-file ceilings as hard
safety bounds, not recommended worker sizing.
