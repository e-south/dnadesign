# Request Shapes

**Type:** guide
**Audience:** users deciding which targets should be designed together
**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-02

Every request contains a list of exact DNA targets. Each target has a
`pool_id`. Targets with the same `pool_id` are designed together because their
oligos are intended to mix in one physical pool. The field does not describe a
biological replicate, condition, study, or sample.

Target IDs must be unique within the request. The same sequence may appear in
different physical pools, but it may not appear twice in one pool.

The same `preflight` → `plan` → `build` → `verify` commands work for every
shape below. The [getting-started tutorial](../getting-started.md) has a full
runnable example. The snippets here omit required sequences, primers, planning
settings, and ordering settings.

## Choose a Shape

| Intent | Request shape | Design behavior |
| --- | --- | --- |
| Fragment one gene or other exact DNA target | One target with one `pool_id` | Designs the junction oligos for that target. |
| Plan several targets for one shared physical pool | Several targets with the same `pool_id` | Selects barcodes and toeholds across all targets in the pool. |
| Plan independent physical pools together | Targets use two or more `pool_id` values | Designs each pool separately and returns one plan. |

A target can be a gene, fragment, cassette, control, or another exact DNA
product. The project or study that supplies it owns that biological meaning.

## One Target in One Pool

Use one target when one exact sequence is the complete design scope:

```yaml
targets:
  - id: exact-target-a
    pool_id: pool-a
    # sequence and recovery_primers are required
```

This shape works for a single gene or any other exact DNA sequence. `pool-a`
only identifies the physical pool.

## Multiple Targets in One Shared Pool

Assign the same `pool_id` when the resulting oligos are intended to coexist in
one physical pool:

```yaml
targets:
  - id: exact-target-a
    pool_id: shared-pool
  - id: exact-target-b
    pool_id: shared-pool
```

TriJunction evaluates barcode and toehold choices across all loci in the pool.
Do not split one intended physical pool into several requests to bypass a
limit. Separate requests would omit the cross-target sequence checks.

## Multiple Independent Pools in One Request

Use distinct `pool_id` values for targets that will not share a physical pool:

```yaml
targets:
  - id: exact-target-a
    pool_id: pool-a
  - id: exact-target-b
    pool_id: pool-b
```

TriJunction designs each pool separately with a seed derived from the request
seed. The command still returns one plan. A request may mix one-target and
multi-target pools.

## Recovery Primers Separate Binding from Exact Extensions

Every target declares terminally matching recovery primers and one recovery
mode. Each forward and reverse primer has two required fields:

```yaml
recovery_primers:
  mode: target_specific
  forward:
    binding_sequence: ACGTACGT
    five_prime_extension: ""
  reverse:
    binding_sequence: TGCATGCA
    five_prime_extension: ""
```

`binding_sequence` is the part that matches the end of the target.
`five_prime_extension` is uppercase `ACGT` DNA or an empty string. The ordered
primer is `five_prime_extension + binding_sequence`. TriJunction preserves the
extension but does not interpret adapters, restriction sites, spacers, Type
IIS enzymes, or later cloning steps.

A physical pool must use one recovery mode:

- `universal` means every target in that pool declares the same primer pair.
  Equality includes both binding sequences and both 5-prime extensions. The
  order table lists that pair once and records all consuming target IDs.
- `target_specific` means the binding pair for one target must not also resolve
  another target in the same pool.

You choose the primer sequences, extensions, and recovery mode. TriJunction
checks their target matches and pool consistency; it does not choose them or
infer their experimental purpose.

## What Fails Before Publication

Every target must fit at least one complete three-way-junction locus under the
requested geometry. A shorter target fails with an error. TriJunction does not
switch to direct synthesis or publish a different kind of oligo.

Search budgets and resource limits are part of the request contract. A request
that is too large or cannot satisfy its constraints fails before publication.
TriJunction does not relax the constraints. See [scale and quality
review](scale-and-review.md) before checking large targets, shared pools, or
many independent pools.

The [contract reference](../reference/contracts.md) remains authoritative for
request, recovery, budget, and failure invariants.

Keep sample identity, conditions, biological replication, objective functions,
ranking rules, and acceptance criteria in the owning study. Do not encode them
in `pool_id`. Retain the TriJunction request and bundle as the sequence-design
record.
