# Prepare a request

**Type:** guide
**Audience:** users turning exact targets into a reviewed design request
**Owner:** dnadesign-maintainers
**Last verified:** 2026-08-02

`junction` does not infer a complete design from a sequence alone. Before
planning, supply five things:

1. the v2 request-schema identifier and one deterministic seed;
2. a planning profile;
3. exact linear DNA targets;
4. caller-chosen recovery primers; and
5. vendor-neutral order labels and an oligo-length ceiling.

Start from the complete [tutorial request](../getting-started.md), then replace
every demonstration value deliberately. The [request contract](../reference/request.md)
lists the exact fields and limits.

## Choose assembly groups

Every target has an `assembly_group_id`. It is the boundary across which
`junction` compares candidate sequences. Put targets in the same group when
their fragments must be designed against one another because they may
encounter one another during the intended three-way-junction assembly.
Different assembly-group IDs define independent searches.

An assembly group does not identify:

- a vendor oligo pool;
- a fragment-annealing or phosphorylation tube;
- a PCR product or recovery aliquot;
- a study, sample, condition, plate, or biological replicate.

Do not split a true joint assembly merely to pass a software limit; doing so
removes cross-target checks.

| Intended design | Target declarations |
| --- | --- |
| One exact gene or other linear target | One target with one `assembly_group_id`. |
| Several targets whose fragments may encounter one another during the intended assembly | Give those targets the same `assembly_group_id`. |
| Independent assemblies in one request | Give each independent set a distinct `assembly_group_id`. |

```yaml
targets:
  - id: target-a
    assembly_group_id: assembly-shared
    sequence: ACGT...
    recovery_primers: ...
  - id: target-b
    assembly_group_id: assembly-shared
    sequence: TGCA...
    recovery_primers: ...
  - id: independent-target
    assembly_group_id: assembly-independent
    sequence: GATC...
    recovery_primers: ...
```

Target IDs are globally unique within a request. The same sequence may appear
in different assembly groups under different target IDs, but it cannot appear
twice in one group.

## Supply exact targets

Each `sequence` is the complete linear 5′→3′ uppercase `ACGT` string that the
planner must reconstruct before any later cleavage or cloning step. The
request contract does not accept circular topology, RNA, ambiguity codes,
degenerate positions, or lowercase normalization.

If a target contains universal priming regions, buffers, Type IIS sites, or
adapters, `junction` treats them as part of the submitted target. It has no
payload-span or post-cleavage-product field.

## Supply recovery primers

Each primer separates its target-binding sequence from an exact 5′ extension:

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

The forward binding string must match the target prefix. The reverse binding
string must match the reverse complement of the target suffix. The order row
is `five_prime_extension + binding_sequence`.

The request requires one recovery mode per assembly group:

- `target_specific` rejects a binding pair that also exactly resolves another
  declared target in the same group.
- `universal` requires one identical forward/reverse pair, including
  extensions, across the group and emits that pair once with all consuming
  target IDs.

This one-mode rule is a software restriction, not a physical claim that an
assembled mixture cannot be divided for different recovery strategies.
`junction` does not design primers, compute melting temperatures, search the
broader genome, predict PCR, or implement the pooled paper's buffer-equalized
universal/Type-IIS workflow.

## Declare the planning profile

The planning profile controls geometry, search budgets, barcode composition,
and substring exclusions. There is no production preset. The small tutorial
values only exercise the software. The papers report useful experimental and
algorithmic contexts, but `junction` has not validated a drop-in laboratory
profile.

Review at least:

- the nominal fragment-oligo geometry, barcode length, and toehold length;
- how many candidate offsets each locus should expose;
- the amount of seeded search work;
- barcode GC and homopolymer bounds; and
- barcode-to-toehold and barcode-to-barcode forbidden substring lengths.

The tool never relaxes these values automatically. Candidate exhaustion fails
with the original request preserved.

`nominal_fragment_oligo_length` is a coordinate parameter, not a physical
length guarantee. The current method can emit fragment orders as long as
`nominal_fragment_oligo_length + search_range - 1`; terminal fragment orders
can be shorter than the nominal value. This differs from the Nature paper's
use of `L` for the physical input-oligo length. Inspect the planned order
lengths instead of treating the field name as a purchasing specification.

## Declare order metadata

The order policy records the caller's labels for synthesis scale,
purification, complement-strand end preparation, a minimum fragment-oligo
length, and a maximum length for every order row. These values are copied and
checked; they are not recommendations. The minimum applies to barcode-bearing
and complement strands, not recovery primers. It is required because short
terminal fragments can be valid under the coordinate model while still being
unsuitable for a caller's synthesis route. Supported end-preparation
declarations are `vendor_5_prime_phosphate` and
`downstream_phosphorylation`.

No default minimum is inferred from either paper. Declare the boundary that
your downstream process has reviewed. Passing it proves only that the emitted
fragment strings lie inside the caller's length interval; it does not validate
synthesis, folding, annealing, ligation, or amplification.

An arbitrary 5′ primer extension may contain a caller-supplied adapter or Type
IIS sequence. `junction` does not identify enzymes, design spacers, model
cleavage, validate overhangs, or plan later cloning.

## Check without publishing

Choose one no-file view:

```bash
uv run junction preflight request.yaml --format json
uv run junction plan request.yaml --format json
```

Both commands run the complete design search. `preflight` returns a short
summary; `plan` returns the full plan. They are alternatives, not sequential
stages. When ready to retain the result, run `build` directly with a new
destination.
