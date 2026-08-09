---
doc_id: junction-request-contract
title: junction request contract
type: reference
scope: dnadesign.junction.request.v2
owner: dnadesign-maintainers
status: active
last_verified: 2026-08-08
---

# Request contract

Requests are strict JSON or YAML objects. Unknown fields, missing fields,
duplicate mapping keys, YAML anchors, and YAML aliases fail parsing. Files must
use `.json`, `.yaml`, or `.yml`, contain UTF-8 text, and fit within 16 MiB.

## Top-level fields

| Field | Type | Meaning |
| --- | --- | --- |
| `schema` | string | Must equal `dnadesign.junction.request.v2`. |
| `seed` | integer | Deterministic request seed in `0..2^64-1`. |
| `planning` | object | Exact geometry, search, and barcode policy. |
| `targets` | non-empty list | Exact target, assembly-group, and recovery declarations. |
| `order_policy` | object | Vendor-neutral labels, end preparation, and allowed order-length interval. |

## Planning fields

The `planning` object contains exactly:

- geometry: `nominal_fragment_oligo_length`, `barcode_length`, `toehold_length`, and
  `search_range`;
- search budgets: `toehold_search_iterations`,
  `barcode_generation_attempts`, `barcode_subset_iterations`, and
  `matching_iterations`;
- barcode set size: `barcode_pool_factor`;
- substring exclusions: `barcode_toehold_k` and `barcode_pair_k`; and
- composition bounds: `barcode_gc_min`, `barcode_gc_max`, and
  `barcode_max_homopolymer`.

Let `L`, `b`, `t`, and `R` denote the first four geometry fields. The v2
request requires:

- `t >= 2`;
- `barcode_pool_factor >= 5`;
- `barcode_toehold_k <= min(b, t)`;
- `barcode_pair_k <= b` and `barcode_pair_k > barcode_toehold_k`;
- `L > 2b + t + R - 1`;
- GC fractions in `[0, 1]`, with minimum no greater than maximum;
- `barcode_max_homopolymer <= b`; and
- `order_policy.max_oligo_length >= max(L + R - 1, L - b + t)`;
- `order_policy.minimum_fragment_oligo_length <=
  order_policy.max_oligo_length`.

The four iteration/attempt ceilings are 100,000; 10,000,000; 100,000; and
100,000 respectively, in the order listed above. `barcode_length` is capped at
65,534 by the method-v1 `uint16` distance-cache representation.

These are software validity bounds, not validated laboratory defaults. The
tool never changes them or relaxes `barcode_toehold_k` and `barcode_pair_k`
silently.

`nominal_fragment_oligo_length` controls locus spacing. It is not the Nature
paper's physical input-oligo length and does not promise equal-length orders.
The current geometry can emit a fragment order as long as
`max(L + R - 1, L - b + t)`. The first term covers offset-expanded strands;
the second covers a terminal complement strand. A terminal fragment can be
shorter than `L`. Exact planned lengths remain subject to the caller's
order-policy interval.

## Target fields

Each target contains exactly:

| Field | Meaning |
| --- | --- |
| `id` | Unique request-local identity. |
| `assembly_group_id` | Boundary for targets whose candidate sequences must be compared because their fragments may encounter one another during the intended three-way-junction assembly. |
| `sequence` | Complete linear 5′→3′ uppercase `ACGT` target. |
| `recovery_primers` | Mode plus caller-supplied forward and reverse primers. |

IDs start with an ASCII alphanumeric character, continue with alphanumerics,
`.`, `_`, or `-`, and fit within 128 bytes. Target IDs are unique. Duplicate
sequences within one assembly group are rejected; the same sequence may appear
under different IDs in different groups.

Every target must contain at least one complete locus and a nonempty terminal
target domain for every candidate offset. This requires
`len(sequence) >= L - b + R`. A shorter target fails; `junction` does not
switch it to direct synthesis.

Each `recovery_primers` object contains `mode`, `forward`, and `reverse`.
Supported modes are `target_specific` and `universal`. Each primer contains a
non-empty uppercase `binding_sequence` and an uppercase or empty
`five_prime_extension`. The forward binding string must match the target
prefix; the reverse must match the reverse complement of the target suffix.

The v2 request permits one recovery mode per assembly group. Universal mode
requires one identical complete pair across the group. Target-specific mode
rejects a pair that exactly resolves another declared target in the group.
These are string checks and an output rule, not primer design or PCR
validation.

## Order-policy fields

The `order_policy` object contains exactly:

- `synthesis_scale`;
- `barcode_bearing_purification`;
- `complement_purification`;
- `primer_purification`;
- `complement_end_preparation`;
- `minimum_fragment_oligo_length`; and
- `max_oligo_length`.

The four text labels are non-empty, exclude control characters and spreadsheet
formula prefixes, and fit within 128 UTF-8 bytes. End preparation is
`vendor_5_prime_phosphate` or `downstream_phosphorylation`. The tool records
these caller choices; it does not recommend a supplier, purification, scale,
or chemical workflow.

`minimum_fragment_oligo_length` and `max_oligo_length` are positive integers,
and the minimum must not exceed the maximum. The minimum applies only to the
barcode-bearing and complement strands that form assembly fragments. The
maximum applies to every fragment and primer order row. The caller must choose
the minimum; `junction` does not infer a paper profile, vendor rule, or
laboratory-valid threshold. Passing these checks establishes string length
only. No thermodynamic or synthesis validation runs.

The minimum is a path constraint, not a late output check. For each target,
the first domain, the spans between adjacent selected junctions, and the
terminal domain must jointly yield fragment strands at or above the declared
floor. The bounded seeded search ranks only feasible sampled paths and always
includes the lexicographically first feasible path. It does not enumerate the
Cartesian product of candidate loci.

## Request-wide resource envelope

Before materializing candidates, the current planner predicts loci, derived
file sizes, and search work under
`dnadesign.junction.request-workload.v1`.

| Counted input | Maximum |
| --- | ---: |
| Assembly groups | 4,096 |
| Targets | 100,000 |
| Input bases | 268,435,456 |
| Loci | 250,000 |
| Toehold candidates | 1,000,000 |
| Barcode candidates | 4,000,000 |

Per-assembly-group ceilings bound encoded bases, distance caches, dynamic-
programming cells, lookups, generated-barcode work, substring work, and sampled
state. Request-wide ceilings are four times the corresponding group limit.
These deterministic estimates are safety envelopes, not wall-clock or peak-RAM
forecasts. An exceeded limit fails before the guarded allocation.

For exact formulas and constants, use
[`design/resources/limits.py`](../../design/resources/limits.py) and
[`design/resources/estimates.py`](../../design/resources/estimates.py). For
operator guidance, see [Scale](../guides/scale.md).
