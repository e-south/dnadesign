---
doc_id: junction-getting-started
title: Getting started with junction
type: tutorial
audience: first-time users
owner: dnadesign-maintainers
status: active
last_verified: 2026-08-09
---

# Getting started

This tutorial publishes and verifies one synthetic 705 bp target. The request
uses 96 nt nominal fragment geometry, 22 nt barcodes, 10 nt toeholds, and a
15 nt locus search range. Those dimensions start from the pooled Sidewinder
paper's method profile. The search policy and synthetic sequence are junction
examples, not a paper-validated laboratory protocol or purchasing advice.

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

The target is an exact uppercase 5′→3′ `ACGT` string. The caller supplies the
terminal primer strings, including an explicit empty `five_prime_extension`.
`junction` checks those strings against the target but does not design the
primers or predict PCR behavior. The order labels are copied into the output;
the tool does not choose a supplier or submit an order.

For this exact request, the deterministic plan contains 13 fragment pairs and
12 three-way junctions. Its 26 fragment orders range from 46 to 106 nt and are
checked against the request's declared 45-to-110-nt interval. The contract does
not decide whether an allowed oligo is synthesizable.

`assembly_group_id` is the boundary across which `junction` compares candidate
sequences. Put targets in the same group when their fragments must be designed
against one another because they may encounter one another during the intended
three-way-junction assembly. The ID does not specify a study, sample, vendor
pool, annealing tube, PCR product, or biological condition.

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

## 5. Review what was proved

Start with:

1. `checks.json` for compact scoped results and explicit `not_run` states;
2. `plan.json` for selected loci, toeholds, barcodes, strands, and search
   receipts;
3. `orders/oligos.tsv` for complete vendor-neutral sequences; and
4. `views/three_way_junction_review.v1.json` for one review record per target.

Render that last artifact with
[BaseRender](../../baserender/docs/integrations/junction.md) to inspect every
selected nucleotide, its aligned complement, fragment and toehold boundaries,
and a three-arm junction schematic. Read complete order rows, primer
sequences, search receipts, and check states in the bundle rather than
inferring them from a plot.

A successful verification establishes deterministic string construction and
file integrity within the documented software checks. It does not establish
thermodynamic folding, synthesis quality, phosphorylation, ligation, PCR,
cloning, or experimental success. Continue with [How `junction`
works](explanation/how-junction-works.md), then [prepare a real
request](guides/prepare-a-request.md).
