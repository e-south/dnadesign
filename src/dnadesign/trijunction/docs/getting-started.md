# Getting Started

**Type:** tutorial
**Audience:** first-time users running one synthetic request
**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-02

This tutorial validates, plans, publishes, and verifies one small synthetic
example. Replace the example sequences and ordering labels before using the
request for real work.

The example has one target in one physical pool. For pooled work, first decide
which targets should share a [`pool_id`](guides/request-shapes.md). The commands
do not change.

## 1. Prepare the Environment

From the repository root:

```bash
uv sync --locked
uv run trijunction --help
```

## 2. Save a Request

Save the following as `request.yaml`. The target is synthetic. The ordering
values only exercise the file format; they are not purchasing advice.

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

Every target must be an uppercase `ACGT` sequence that you have already
linearized. V1 has no topology field and does not plan circular targets.
TriJunction rejects
lowercase or ambiguous sequences, RNA, unknown fields, invalid primer geometry,
duplicate target identities, targets that cannot fit one complete junction,
and requests that exceed the search limits. JSON and
YAML request files are capped at 16 MiB. Target and pool IDs use a restricted
ASCII alphabet and may contain at most 128 characters. Free-text ordering
labels are capped at 128 UTF-8 bytes, which may be fewer than 128 non-ASCII
characters.

Both primer objects require `five_prime_extension`; use `""` when there is no
extension. A non-empty value is placed unchanged before the binding sequence
in the order row. TriJunction does not infer restriction sites, spacers,
adapters, or Type IIS cleavage behavior from that sequence.

## 3. Preflight and Inspect

```bash
uv run trijunction preflight request.yaml --format json
uv run trijunction plan request.yaml --format json
```

`preflight` runs the same design checks as `plan` but returns a short summary.
Neither command writes a bundle. A successful summary says
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

The `pwd -P` step resolves the temporary directory to its physical path.
TriJunction rejects destinations reached through a symlink.

For work you want to keep, save the bundle under the project that owns the
sequences, for example `<project>/outputs/trijunction/<bundle-id>`. New
directories use mode `0700` and new files use `0600`. Those permissions reduce
accidental local disclosure, but they do not make sequence data safe to commit,
sync, or share. Keep the request with its project or study.

## 5. Review the Bundle

Read `checks.json` first, then `plan.json` and `orders/oligos.tsv`.
`views/three_way_junction_review.v1.json` contains one compact review record
per target and can feed the optional BaseRender image. The TSV is
vendor-neutral and still needs scientific, thermodynamic, synthesis, and
ordering review. See the [contract
reference](reference/contracts.md) for the bundle file rules and [sources and
scope](reference/sources.md) for what the checks do and do not prove. For larger
request shapes and optional BaseRender images, continue to
[scale and quality review](guides/scale-and-review.md).
