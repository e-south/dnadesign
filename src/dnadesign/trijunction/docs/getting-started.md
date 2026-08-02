# Getting Started

**Type:** tutorial
**Audience:** operators evaluating one synthetic planning request
**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-02

This path validates, plans, publishes, and verifies one small synthetic
example. Replace every example sequence and order-policy label before using a
bundle outside a software review.

This tutorial uses one target in one physical pool. Before preparing pooled or
multi-pool work, choose the intended [`pool_id` request
shape](guides/request-shapes.md); the lifecycle and commands remain the same.

## 1. Prepare the Environment

From the repository root:

```bash
uv sync --locked
uv run trijunction --help
```

## 2. Save an Explicit Request

Save the following as `request.yaml`. The target is synthetic; the order
policy values are labels for exercising the contract, not purchasing advice.

```yaml
schema: dnadesign.trijunction.request.v1
seed: 17
planning:
  oligo_length: 46
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
    pool_id: pool-a
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
  max_oligo_length: 64
```

Requests are strict: every target is exact linear uppercase `ACGT` DNA.
Lowercase, RNA, ambiguous bases, circular topology, unknown fields, invalid
primer geometry, duplicate target identities, targets without a complete
junction locus, or infeasible search budgets fail before publication. JSON and
YAML request files are capped at 16 MiB.

Both primer objects require an explicit `five_prime_extension`; use `""` when
there is none. A non-empty extension is preserved verbatim before the binding
sequence in the order row. TriJunction does not infer or interpret restriction
sites, spacers, adapters, or Type IIS cleavage behavior from that sequence.

## 3. Preflight and Inspect

```bash
uv run trijunction preflight request.yaml --format json
uv run trijunction plan request.yaml --format json
```

`preflight` runs the same design checks as `plan` but returns a compact receipt.
Neither command writes a durable bundle. A successful receipt says
`status: planned`, `validation_scope: string_only`, and
`thermodynamic_screening: not_run`; it does not say the design is ready for the
laboratory.

## 4. Publish and Verify

Use a new directory. The publisher never overwrites an existing bundle.

```bash
trijunction_demo_root="$(mktemp -d)"
trijunction_demo_root="$(cd "$trijunction_demo_root" && pwd -P)"
uv run trijunction build request.yaml \
  --output "$trijunction_demo_root/design-v1" \
  --format json
uv run trijunction verify "$trijunction_demo_root/design-v1" --format json
```

The physical-path step matters on systems where a temporary-directory prefix
is itself a symlink. TriJunction rejects symlinked destination components.

For durable work, use a study- or project-owned output root, for example
`<study-workspace>/outputs/trijunction/<bundle-id>`. The publisher creates
private bundle directories (`0700`) and files (`0600`). These modes reduce
accidental local disclosure; they do not make sequence data safe to commit,
sync, or share. Keep the request with its owning project or study rather than
turning TriJunction into a study registry.

## 5. Review the Bundle

Read `checks.json` first, then `plan.json` and `orders/oligos.tsv`. Use
`views/three_way_junction_review.v1.json` for a compact target-by-target QA
projection or as the input to the optional BaseRender view. Treat the TSV as a
vendor-neutral projection that still requires scientific, thermodynamic,
synthesis, and ordering review. See the [contract
reference](reference/contracts.md) for the full artifact contract and [sources
and scope](reference/sources.md) for the evidence boundary. For larger request
shapes and the separate optional BaseRender review boundary, continue to
[scale and quality review](guides/scale-and-review.md).
