---
doc_id: junction-getting-started
title: Getting started with junction
type: tutorial
audience: first-time users
owner: dnadesign-maintainers
status: active
last_verified: 2026-08-10
---

# Getting started

This tutorial builds and verifies one 705 bp example target. The request
uses 96 nt nominal fragment geometry, 22 nt barcodes, 10 nt toeholds, and a
15 nt locus search range. Those dimensions follow the starting profile in the
pooled Sidewinder paper. The search policy and example sequence are software
examples, not a validated laboratory protocol or purchasing advice.

## 1. Prepare the repository

From the repository root:

```bash
uv sync --locked
uv run junction --help
```

## 2. Copy the complete request

The checked-in request explains each field in YAML comments:

```bash
cp src/dnadesign/junction/examples/gene-scale/request.yaml request.yaml
```

The target is an exact uppercase 5′→3′ `ACGT` string. Recovery primers are
inputs: `junction` checks their terminal matches and preserves each optional
`five_prime_extension`. Primer optimization and PCR assessment happen
upstream. Order labels are copied into vendor-neutral output rows.

If your target already exists as a raw string, text file, or FASTA, compile it
into the same request contract instead of copying sequence text by hand:

```bash
uv run junction request \
  --base-request src/dnadesign/junction/examples/gene-scale/request.yaml \
  --input targets.fasta \
  --primer-binding-length 20 \
  > request.json
```

`--base-request` contributes only the seed, planning profile, and order policy;
its targets are replaced. FASTA record IDs become target IDs. For one
in-memory sequence, use `--sequence ACGT... --target-id target-a`. Input is
uppercased and whitespace is removed, while ambiguity codes and duplicate IDs
fail. The explicit binding length selects terminal primer-binding spans; it is
not a PCR-performance claim.

For this exact request, the deterministic plan contains 13 fragment pairs and
12 three-way junctions. Its 26 fragment orders range from 46 to 106 nt and are
checked against the request's declared 45-to-110-nt interval. Passing that
check does not establish synthesis feasibility.

`assembly_group_id` says which targets must be checked together. Use one group
when their fragments may encounter one another during the intended assembly.
The ID does not specify a study, sample, vendor pool, annealing tube, PCR
product, or biological condition.

## 3. Publish the bundle

`build` performs the complete search, writes to a new directory, verifies the
staged files, installs them, and replays the installed bundle. You do not need
to run `preflight` or `plan` first unless you want their no-file output.

```bash
junction_demo_root="$(mktemp -d)"
junction_demo_root="$(cd "$junction_demo_root" && pwd -P)"
uv run junction build request.yaml \
  --output "$junction_demo_root/design-demo" \
  --format json
```

Publication is create-only: the destination must not exist and no existing
directory is replaced. The `pwd -P` step resolves the temporary root to its
physical path because publication rejects symlinked destination components.

For retained work, place the bundle under the project that owns the target,
for example `<project>/outputs/junction/<bundle-id>`. New directories use mode
`0700` and files use `0600`. Those modes reduce accidental local disclosure;
they do not make sequences safe to commit, sync, or share.

## 4. Verify it again later

```bash
uv run junction verify "$junction_demo_root/design-demo" --format json
```

Verification checks the exact inventory and hashes, parses the recorded
request, reruns the deterministic algorithm, renders each expected artifact,
and compares it with the saved file. A changed, missing, extra, relocated, or
non-reproducible file fails verification.

## 5. Inspect the bundle

Start with:

1. `checks.json` for compact scoped results and explicit `not_run` states;
2. `plan.json` for selected loci, toeholds, barcodes, strands, and search
   receipts;
3. `orders/oligos.tsv` for the vendor-neutral order sheet;
4. `sequences/targets.fasta`, `sequences/oligos.fasta`, and
   `sequences/expected_pcr_products.fasta` for sequence-tool handoffs; and
5. `views/three_way_junction_review.v1.json` for one molecular review record
   per target; and
6. `views/junction_sequence_dissimilarity.v1.json` for one compact sequence
   comparison record per assembly group.

Render `views/three_way_junction_review.v1.json` with
[BaseRender](../../baserender/docs/integrations/junction.md) to inspect every
selected nucleotide, its aligned complement, fragment and toehold boundaries,
and three-arm geometry. Use Junction's
`plot_sequence_dissimilarity(...)` API for the optional pairwise string-metric
view. Read complete order rows, primer sequences, search receipts, and check
states in the bundle rather than inferring them from a plot.

A successful verification establishes deterministic string construction and
bundle integrity. Thermodynamic, synthesis, and laboratory review are separate
acceptance steps. Continue with [How `junction`
works](explanation/how-junction-works.md), then [prepare a real
request](guides/prepare-a-request.md).
