# Getting started

**Type:** tutorial
**Audience:** first-time users
**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-03

This tutorial publishes and verifies one small synthetic design. Its short
oligos, 8-nt toeholds, loose composition bounds, and small search budgets make
the example quick to run. They are software-demonstration values, not a
paper-validated laboratory profile or purchasing advice.

## 1. Prepare the repository

From the repository root:

```bash
uv sync --locked
uv run junction --help
```

## 2. Save a complete request

Save this as `request.yaml`:

```yaml
schema: dnadesign.junction.request.v2
seed: 17
planning:
  nominal_fragment_oligo_length: 46
  barcode_length: 16
  toehold_length: 8
  search_range: 2
  toehold_search_iterations: 40
  barcode_pool_factor: 5
  barcode_generation_attempts: 100000
  barcode_toehold_k: 4
  barcode_pair_k: 5
  barcode_subset_iterations: 40
  matching_iterations: 100
  barcode_gc_min: 0.25
  barcode_gc_max: 0.75
  barcode_max_homopolymer: 3
targets:
  - id: target-a
    assembly_group_id: assembly-a
    sequence: ACGATTCGGTACCTGATGCACTGAACGATTCGGTACCTGATGCACTGAACGATTCGGTACCTGATGCACTGA
    recovery_primers:
      mode: target_specific
      forward:
        binding_sequence: ACGATTCG
        five_prime_extension: ""
      reverse:
        binding_sequence: TCAGTGCA
        five_prime_extension: ""
order_policy:
  synthesis_scale: example-scale
  barcode_bearing_purification: example-purification
  complement_purification: example-purification
  primer_purification: example-purification
  complement_end_preparation: vendor_5_prime_phosphate
  minimum_fragment_oligo_length: 1
  max_oligo_length: 64
```

The target is an exact uppercase 5′→3′ `ACGT` string. The caller supplies the
terminal primer strings, including an explicit empty `five_prime_extension`.
`junction` checks those strings against the target but does not design the
primers or predict PCR behavior. The order labels are copied into the output;
the tool does not choose a supplier or submit an order.

`nominal_fragment_oligo_length` sets the planner's locus geometry. It is not a
promise that every fragment order has that length. The maximum is the larger
of the offset-expanded length `L + R - 1` and the terminal-complement length
`L - b + t`, where `L`, `R`, `b`, and `t` are the nominal length, search range,
barcode length, and toehold length. A terminal fragment can also be shorter.
The tutorial sets
`minimum_fragment_oligo_length: 1` only so its small synthetic example is easy
to run. Choose and review a real minimum for synthesis work. `junction` checks
the declared minimum and maximum as string lengths; it does not judge whether
an oligo is synthesizable.

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

A successful verification establishes deterministic string construction and
file integrity within the documented software checks. It does not establish
thermodynamic folding, synthesis quality, phosphorylation, ligation, PCR,
cloning, or experimental success. Continue with [How `junction`
works](explanation/how-junction-works.md), then [prepare a real
request](guides/prepare-a-request.md).
