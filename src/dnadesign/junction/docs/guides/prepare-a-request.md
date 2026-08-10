---
doc_id: junction-prepare-request
title: Prepare a junction request
type: guide
audience: users turning exact targets into a reviewed design request
owner: dnadesign-maintainers
status: active
last_verified: 2026-08-10
---

# Prepare a request

A complete request supplies five things:

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

Choose one recovery mode per assembly group:

| Mode | Use it when | Contract |
| --- | --- | --- |
| `target_specific` | Each target is recovered with its own terminal pair. The papers call this construct-specific recovery. | A pair must not also resolve another declared target in the group. |
| `universal` | Every target already carries the same terminal binding regions. | The complete pair, including extensions, must match across the group and is emitted once for all consuming target IDs. |

An assembled mixture may still be split for different downstream operations;
the one-mode rule keeps one request unambiguous. Primer selection,
melting-temperature analysis, broader off-target search, and the pooled
paper's buffer-equalized universal/Type-IIS design remain upstream tasks.

## Declare the planning profile

The planning profile controls geometry, search budgets, barcode composition,
and substring exclusions. There is no production preset. The checked-in
gene-scale example starts from dimensions reported in the pooled paper, but
its search settings remain explicit `junction` policy. The papers provide useful
experimental and algorithmic context; `junction` has not validated a drop-in
laboratory profile.

Review at least:

- the nominal fragment-oligo geometry, barcode length, and toehold length;
- how many candidate offsets each locus should expose;
- the amount of seeded search work;
- barcode GC and homopolymer bounds; and
- barcode-to-toehold and barcode-to-barcode forbidden substring lengths.

The Nature study found effective ligation when the nick was at least about six
bases from the barcode helix and standardized its tested assemblies on a
10-base toehold. The pooled study also used `t = 10` unless stated otherwise.
The checked-in examples therefore use 10 nt. The request contract accepts other
positive lengths for computational work, but the papers do not establish a
general benefit for making the toehold longer than 10 nt. Treat a different
value as a new reviewed method choice, not a routine scaling control.

The tool never relaxes these values automatically. Candidate exhaustion fails
with the original request preserved.

`nominal_fragment_oligo_length` is a coordinate parameter, not a physical
length guarantee. The longest possible fragment order is the larger of:

- `nominal_fragment_oligo_length + search_range - 1`, for an offset-expanded
  strand; and
- `nominal_fragment_oligo_length - barcode_length + toehold_length`, for a
  terminal complement strand.

Terminal fragment orders can also be shorter than the nominal value. This
differs from the Nature paper's use of `L` for the physical input-oligo length.
Inspect the planned order lengths instead of treating the field name as a
purchasing specification.

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

Balance the lengths of the **ordered strands**, not the target domains named
F1, F2, and F3. A first, internal, or last fragment carries a different
combination of target, toehold, and barcode sequence, so equally sized target
domains do not produce equally sized orders. `junction` keeps sequence
separation as the search objective and does not hide a second length-balancing
score inside it. Use the published order table and BaseRender subtitle to
review the minimum, maximum, and median. If the spread is unsuitable, revise
the target length or declared geometry and rerun the complete search.

Supplier limits change, and product lines from one supplier can have different
windows. As of 2026-08-10, representative official specifications are:

| Pool product | Stated oligo-length window |
| --- | ---: |
| [IDT oPools](https://www.idtdna.com/pages/products/custom-dna-rna/dna-oligos/custom-dna-oligos/opools-oligo-pools) | 40–350 bases |
| [Twist Oligo Pools](https://www.twistbioscience.com/faq/oligo-pools/what-maximum-length-can-be-ordered-oligo-pool) | 20–350 nt |
| [Agilent SurePrint G7636A](https://www.agilent.com/store/en_US/Prod-G7636A/G7636A) | 30–110 nt |

These links are purchasing context, not built-in profiles or endorsements.
Set `minimum_fragment_oligo_length` and `max_oligo_length` to the exact route
you intend to use. If several routes must remain possible, declare their
reviewed intersection. The planner rejects a candidate path that would emit a
fragment below the minimum, and it rejects any fragment or recovery primer
above the maximum before publication.

A 5′ primer extension may carry a caller-supplied adapter or Type IIS sequence.
`junction` preserves that sequence exactly; enzyme choice, spacers, cleavage,
overhangs, and later cloning remain part of the upstream design.

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
