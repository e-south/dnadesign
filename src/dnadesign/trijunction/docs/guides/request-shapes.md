# Request Shapes

**Type:** guide
**Audience:** callers deciding how targets should share a design request
**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-02

TriJunction accepts one neutral request shape: a list of exact DNA targets.
Each target has a `pool_id` that declares which targets will share one physical
pool. The pool is the string-design boundary; it is not a biological replicate,
condition, study, or sample classification.

Target IDs are unique across the complete request. The same exact sequence may
appear in distinct physical pools, but duplicate sequences within one pool are
rejected because they do not identify independent joint-search subjects.

Use the same `preflight` → `plan` → `build` → `verify` lifecycle for every
shape below. The [getting-started tutorial](../getting-started.md) provides the
complete runnable procedure. These examples show structure only and omit
required sequences, primers, planning policy, and order policy.

## Choose a Shape

| Intent | Request shape | Design behavior |
| --- | --- | --- |
| Fragment one gene or other exact DNA target | One target with one `pool_id` | Designs the junction oligos for that target. |
| Plan several targets for one shared physical pool | Several targets with the same `pool_id` | Jointly selects barcode and toehold assignments across every target in that pool. |
| Plan independent physical pools together | Targets use two or more `pool_id` values | Derives a separate deterministic design stream for each pool and returns one request-level plan. |

TriJunction is sequence-oriented. The exact target can represent a gene,
fragment, cassette, control, or another caller-defined DNA product. That
meaning remains with the calling project or study.

## One Target in One Pool

Use one target when one exact sequence is the complete design scope:

```yaml
targets:
  - id: exact-target-a
    pool_id: pool-a
    # sequence and recovery_primers are required
```

This shape is suitable for fragmenting a single gene or another exact DNA
sequence. `pool-a` identifies the physical design pool; it does not assert what
the target means or how it will be used.

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

TriJunction designs the pool jointly. Barcode and toehold choices are evaluated
across the combined loci rather than by planning each target in isolation and
concatenating the results. Do not split one intended physical pool into several
requests to bypass a resource failure: doing so removes the joint string checks
that the pool requires.

## Multiple Independent Pools in One Request

Use distinct `pool_id` values for targets that will not share a physical pool:

```yaml
targets:
  - id: exact-target-a
    pool_id: pool-a
  - id: exact-target-b
    pool_id: pool-b
```

Each pool receives its own deterministic design stream derived from the request
seed. The result remains one request and one plan; separate pool IDs do not
create a batch API or an alternate lifecycle. A request may combine one-target
and multi-target pools.

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

`binding_sequence` is the target-matching DNA used for terminal geometry.
`five_prime_extension` is exact uppercase `ACGT` DNA, or an explicit empty
string. The complete order sequence is `five_prime_extension +
binding_sequence`. TriJunction preserves that sequence but does not interpret
adapters, restriction sites, spacers, Type IIS enzymes, or downstream cloning
intent.

A physical pool must use one recovery mode:

- `universal` means every target in that pool declares the same primer pair.
  Equality includes both binding sequences and both 5-prime extensions. The
  order projection emits that pair once and records all consuming target IDs.
- `target_specific` means the binding pair for one target must not also resolve
  another target in the same pool.

Primer sequences, extensions, and the decision to use universal or
target-specific recovery are caller-owned design inputs. TriJunction validates
the declared geometry and pool consistency; it does not infer experimental
intent.

## Fail-Closed Boundaries

Every target must be long enough to contain at least one complete
three-way-junction locus under the declared geometry. A shorter target fails
design explicitly. TriJunction does not silently switch to direct synthesis or
publish a different oligo type.

Planning also uses explicit, bounded search budgets. Oversized or infeasible
requests fail before publication rather than relaxing constraints or consuming
unbounded work. See [scale and quality review](scale-and-review.md) before
preflighting large single targets, shared pools, or many independent pools.
The [contract reference](../reference/contracts.md) remains authoritative for
request, recovery, budget, and failure invariants.

Study-owned concepts—including sample identity, condition, biological
replication, objective functions, ranking policy, and acceptance criteria—do
not belong in `pool_id` or the TriJunction request contract. Keep those in the
owning study and retain the TriJunction request and bundle as design evidence.
