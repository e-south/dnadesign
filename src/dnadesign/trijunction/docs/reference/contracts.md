# TriJunction Contract Reference

**Type:** reference
**Scope:** TriJunction request parsing, string-level design, bundle
publication, and verification
**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-02

## Commands and Python API

The CLI and Python API provide the same four operations:

1. `preflight(request)` parses and designs without writing files.
2. `plan(request)` returns the complete deterministic in-memory plan.
3. `build(request, destination=...)` designs and publishes a verified bundle
   in a new directory.
4. `verify(bundle)` recomputes the plan and verifies every declared bundle file.

The CLI mirrors these names. Request files may use JSON or YAML. Published
bundles use canonical JSON plus one stable UTF-8 TSV file. CLI summaries can
render text or JSON. `plan --format json` prints the complete plan;
`plan --format text` prints only its status and plan ID. A successful preflight
summary reports `status: planned`, `validation_scope: string_only`, and
`thermodynamic_screening: not_run`; it is not a laboratory-readiness claim.

### JSON error response

After command dispatch, TriJunction configuration, design, and bundle failures
exit with status `1`. With `--format json`, the CLI writes this response to
standard error:

```json
{
  "status": "error",
  "error": {
    "code": "design_error",
    "message": "human-readable failure detail",
    "retryable": false
  }
}
```

`error.code` is `config_error`, `design_error`, `bundle_error`, or the fallback
`trijunction_error`. The message supplies specific context and
`retryable: false` means the identical request should not be retried without
changing the request or its environment. Filesystem argument checks and other
CLI argument or option errors are handled by Typer before TriJunction dispatch,
so they use Typer's own error text rather than this JSON response.

## Request Schema

`dnadesign.trijunction.request.v1` has exactly five top-level fields:
`schema`, `seed`, `planning`, `targets`, and `order_policy`. Unknown or missing
fields fail parsing. JSON and YAML duplicate mapping keys are rejected. YAML
anchors and aliases are also rejected so the reviewed document is the complete
request rather than an expanded representation with hidden reuse. Request
files are limited to 16 MiB and must use `.json`, `.yaml`, or `.yml`.

`pool_id` identifies a physical pool used for joint sequence design. See [request
shapes](../guides/request-shapes.md) for the supported single-target,
shared-pool, and independent-pool arrangements. It does not encode biological
meaning.

### Planning Invariants

Let `L` be `oligo_length`, `b` `barcode_length`, `t` `toehold_length`, and `R`
`search_range`.

- All request integers are at most `2^64 - 1`; integer budgets are positive
  except `seed`, which is nonnegative. The limit preserves the full unsigned
  64-bit seed range and gives every integer one portable representation bound.
- `t >= 2`.
- `barcode_pool_factor >= 5`.
- `barcode_toehold_k <= min(b, t)`.
- `barcode_pair_k <= b` and `barcode_pair_k > barcode_toehold_k`.
- `L > 2b + t + R - 1`.
- `barcode_gc_min` and `barcode_gc_max` are fractions in `[0, 1]`, with the
  minimum no greater than the maximum.
- `barcode_max_homopolymer <= b`.
- `order_policy.max_oligo_length >= L + R - 1`.

The request must contain at least one target. Every target is planned through
three-way junctions and must provide at least one complete candidate locus,
which requires length `>= L - b + R - 1`. A shorter target fails with an error
that points to direct synthesis outside TriJunction. It never switches assembly
methods automatically.

Target IDs are unique. IDs and pool IDs use alphanumerics plus `.`, `_`, or
`-`, starting with an alphanumeric character, and are capped at 128 ASCII
characters. Target and primer-binding sequences are non-empty uppercase
`ACGT`; primer extensions are uppercase `ACGT` or an explicit empty string.
You must supply each exact DNA sequence already linearized. V1 has no topology
field and does not accept RNA, IUPAC ambiguity codes, or lowercase
normalization. Duplicate sequences inside one physical pool are rejected. The
same sequence may appear under distinct globally unique target IDs only when
those targets use different physical pools.

`barcode_toehold_k` and `barcode_pair_k` are request fields, not values chosen
at runtime. Paper-inspired starting values are `floor(t / 2)` and
`max(floor(b / 4), barcode_toehold_k + 1)`, respectively. You must review and
declare them for your design context. Candidate exhaustion under the
declared values is an error; TriJunction never increments either value
automatically.

### Declared and Derived Resource Ceilings

Schema-v1 declaration ceilings reject oversized integers before design:

| Request field | Maximum |
| --- | ---: |
| `toehold_search_iterations` | 100,000 |
| `barcode_generation_attempts` | 10,000,000 |
| `barcode_subset_iterations` | 100,000 |
| `matching_iterations` | 100,000 |

V1's compact barcode-distance cache also limits `barcode_length` to 65,534
bases. The cache reserves the next `uint16` value to mark distances it has not
computed. TriJunction checks this limit before allocating the distance cache.

The request-wide policy `dnadesign.trijunction.request-workload.v1` also limits
the request before TriJunction starts designing its physical pools:

| Dimension | Maximum |
| --- | ---: |
| Physical pools | 4,096 |
| Targets | 100,000 |
| Input bases | 268,435,456 |
| Loci | 250,000 |
| Toehold candidates | 1,000,000 |
| Barcode candidates | 4,000,000 |

TriJunction measures the request exactly against the input limit. Before
sequence search, it counts the expected loci and estimates the largest possible
derived files. A request fails if an estimate exceeds the same limit used by
publication and offline verification. These are upper bounds, not predictions
of final file size.

| Artifact | Limit |
| --- | ---: |
| `request.json` | 16 MiB (exact) |
| `plan.json` | 256 MiB (upper bound) |
| `checks.json` | 16 MiB (upper bound) |
| `orders/oligos.tsv` | 256 MiB (upper bound) |
| `views/three_way_junction_review.v1.json` | 256 MiB (upper bound) |

TriJunction next estimates search work for each pool and for the complete
request:

| Modeled dimension | Per physical pool | Complete request |
| --- | ---: | ---: |
| Toehold encoded bases | 67,108,864 | 268,435,456 |
| Toehold distance-cache bytes | 268,435,456 | 1,073,741,824 |
| Toehold distance lookups | 2,000,000,000 | 8,000,000,000 |
| Toehold dynamic-programming cells | 2,000,000,000 | 8,000,000,000 |
| Toehold sampled-state bytes | 67,108,864 | 268,435,456 |
| Barcode-generation base visits | 250,000,000 | 1,000,000,000 |
| Barcode-generation state bytes | 134,217,728 | 536,870,912 |
| Barcode encoded bases | 134,217,728 | 536,870,912 |
| Barcode distance-cache bytes | 134,217,728 | 536,870,912 |
| Barcode-subset lookups | 100,000,000 | 400,000,000 |
| Barcode dynamic-programming cells | 1,000,000,000 | 4,000,000,000 |
| Barcode sampled-state bytes | 67,108,864 | 268,435,456 |
| Matching substring character visits | 750,000,000 | 3,000,000,000 |
| Matching sampled-state bytes | 67,108,864 | 268,435,456 |

These are deterministic software limits, not forecasts of wall-clock time
or available host memory. Toehold lookup totals include both streamed search
passes: one discovers each trial's maximum and one reads the resulting cached
distances to form exact fixed-point choice weights. Dynamic-programming cells
count only pairs that can require computation; the second pass is cache-only.
Substring character visits conservatively count both
copying and hashing every materialized character, including the two fixed
lower-bound scans. TriJunction stops before creating the large search data
associated with the first exceeded limit. Lower a declared budget or split
only physical pools that are genuinely independent. Splitting one intended
shared pool would omit the required cross-target sequence checks. See [scale
and quality review](../guides/scale-and-review.md).

### Recovery Invariants

Each target declares `target_specific` or `universal` recovery primers. Forward
and reverse primer objects each contain exactly `binding_sequence` and
`five_prime_extension`. The forward binding sequence must equal the target
prefix; the reverse binding sequence must equal the reverse complement of the
target suffix. The complete 5-prime-to-3-prime order sequence is
`five_prime_extension + binding_sequence`.

A physical pool uses one recovery mode. Universal recovery requires one exact
primer pair—including both binding sequences and both extensions—for the
complete pool. Target-specific binding pairs must not resolve a second target
in the same pool. The order table lists a universal primer pair once per
pool and records every consuming target in `target_ids`.

Extensions are carried as exact sequence, not interpreted as functional
annotations. A later cloning workflow must handle Type IIS sites, spacers,
adapters, cleavage geometry, and cloning policy.

### Ordering Invariants

The order policy contains exactly `synthesis_scale`,
`barcode_bearing_purification`, `complement_purification`,
`primer_purification`, `complement_end_preparation`, and `max_oligo_length`.
Its text is explicit, non-empty plain text and cannot begin with a spreadsheet
formula marker. Each of the four free-text fields is capped at 128 bytes after
UTF-8 encoding; this is a byte limit, not a character-count limit.
Complement-strand ends declare exactly one supported state:
`vendor_5_prime_phosphate` or `downstream_phosphorylation`. TriJunction records
these provided choices; it does not select a supplier or submit an order.
Every complete order sequence—including a recovery primer's extension and
binding sequence—must fit `max_oligo_length`.

## Design Contract

Planning is deterministic for the complete request, including `seed`. Targets
and pools are canonicalized before search, and per-pool search streams are
derived from the request seed. Changing inputs or the seed changes the request
identity and normally the plan identity.

For each physical pool, TriJunction:

1. enumerates complete candidate toehold windows at each junction locus;
2. selects a maximin toehold path under the declared search budget;
3. generates barcode candidates under declared GC, homopolymer, and shared
   substring constraints;
4. selects a maximin barcode subset and matches barcodes to toeholds by the
   worst pairwise longest-common-substring score;
5. composes first, internal, and last fragment roles; and
6. proves exact target reconstruction, reverse-complement complement-strand
   ligation, terminal recovery geometry, and the synthesis-length ceiling.

The exact geometry, strand formulas, objective functions, and differences from
the papers are specified in the [method reference](method.md).

Search exhaustion is an error. Constraints are never relaxed silently. The
plan records `thermodynamic_screening: not_run`; string checks are not a claim
of thermodynamic orthogonality or experimental performance.

## Plan and Bundle Schemas

The plan uses `dnadesign.trijunction.plan.v1` with algorithm identifier
`dnadesign.trijunction.string.v1`. It contains the request digest, content-based
plan ID, pool search receipts, target reconstruction evidence, vendor-neutral
order rows, and machine-readable checks.

The four search stages share an internal typed interface. Their implementations
can change without changing request parsing, strand composition, publication,
or rendering. V1 does not let a request choose an arbitrary backend. A future
optimizer that changes search meaning or scale behavior must use a new
algorithm identifier and bring its own equivalence, resource, and regression
evidence. It may not silently change an existing plan identity.

The bundle manifest uses `dnadesign.trijunction.bundle.v1` and records exact
paths, byte lengths, and SHA-256 identities for five artifacts:

| Artifact | Purpose |
| --- | --- |
| `request.json` | Canonical request used to reproduce the design. |
| `plan.json` | Complete plan and evidence. |
| `checks.json` | Compact machine-readable invariant results, each with an exact `pool` or `target` subject. |
| `orders/oligos.tsv` | Lossless vendor-neutral order table with target-use sets. |
| `views/three_way_junction_review.v1.json` | Validated review records with target and pool IDs but no dedicated study fields, one per target. |

`manifest.json` is canonical JSON and is outside its own artifact inventory.
Publication verifies the staged bundle, installs it at a destination that does
not already exist, and verifies it again. Destination components must resolve
through a physical, non-symlink path. New bundle directories use mode `0700`;
new files use mode `0600` because requests, plans, and order rows can contain
sensitive sequences. Filesystem modes do not replace repository, backup,
cloud-sync, or sharing policy.

Offline verification requires the exact manifest inventory and rejects missing
or extra manifest entries, relocated or symlinked declared artifacts,
undeclared filesystem entries, non-canonical payloads, digest mismatches, and
non-reproducible plans.

The review file is derived from the reproduced plan during both publication
and verification. It carries exact fragment coordinates, strand and recovery
sequences, pool search receipts, and structurally scoped checks. It is a stable
input for optional BaseRender jobs; rendered images remain separate BaseRender
outputs and never alter this bundle.

## Not Covered

TriJunction does not own target discovery, study identity, biological meaning,
experimental protocols, thermodynamic simulation, supplier APIs, purchasing,
or laboratory acceptance criteria. Keep those decisions in the study or other
process responsible for using the design.
