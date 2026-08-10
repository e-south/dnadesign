---
doc_id: junction-how-it-works
title: How junction works
type: explanation
audience: readers new to three-way-junction DNA assembly
owner: dnadesign-maintainers
status: active
last_verified: 2026-08-10
---

# How `junction` works

`junction` plans the oligo sequences for a Sidewinder-style assembly. The
software works from exact target strings; it does not infer a target or assess
primer performance. A request may supply recovery primers directly or extract
terminal binding strings at an explicit length.

[![Input target, fragment oligos, pre-ligation junctions, and the expected PCR duplex](../assets/assembly-process.svg)](../assets/assembly-process.svg)

## From target to product

The submitted target may start as request JSON or YAML, a raw sequence, a text
file, or FASTA. Raw, text, and FASTA input first compile to the same canonical
request used by every planning command.

### 1. Split the target into paired oligos

Selected toeholds divide the target into fragments. Each fragment produces two
orderable strands: a barcode-bearing strand and its shifted complement. The
temporary barcode arms sit outside the target sequence.

### 2. Anneal each fragment pair

[![Exact annealing for three fragment pairs](../assets/annealed-fragments.svg)](../assets/annealed-fragments.svg)

Target-derived bases pair vertically within each fragment. The barcode and
toehold arms remain exposed so neighboring fragments can recognize one
another. Every base uses the same physical spacing in this view.

### 3. Form a three-way junction

[![Nucleotide-level details for both three-way junctions](../assets/junction-detail.svg)](../assets/junction-detail.svg)

The barcode arms of neighboring fragments pair into the perpendicular helix.
The target-derived toehold pairs along the horizontal helix and places a nick
between adjacent complement strands. The detail view shows strand direction,
base pairing, and the exact target coordinate for each junction.

### 4. Ligate and recover the target

Ligation is intended to seal the complement-strand nicks. Recovery PCR is
intended to remove the temporary barcode strands and produce the exact duplex
specified by the target and request's terminal primers. The process view
above shows the complete pre-ligation assembly and expected PCR product.

## Sequence model

For one junction, the sequence model is:

```text
submitted target                 D0 · t1 · D1

fragment 0  barcode-bearing      D0 · t1 · b1
            complement           rc(D0)

fragment 1  barcode-bearing      rc(b1) · D1
            complement           rc(D1) · rc(t1)

reconstructed complement         rc(D1) · rc(t1) · rc(D0)
                                = rc(submitted target)
```

`D0` and `D1` are target domains, `t1` is the target-derived toehold, `b1`
is the temporary external barcode, and `rc(...)` means reverse complement.

## What the planner does

1. Read the exact targets, assembly groups, recovery primers, search settings,
   and order labels.
2. Enumerate candidate loci and target-derived toeholds.
3. Select separated toeholds and temporary barcodes within each assembly group.
4. Compose both strands of every fragment and check exact target
   reconstruction, primer matches, and order-length limits.
5. Write the request, plan, checks, order table, sequence FASTAs, plot record,
   and manifest to a new bundle, then verify it by replay.

The search is deterministic for the complete request and seed. Adding a target
to an assembly group can change the group's selected toeholds and barcodes.

## Terms used here

| Term | Meaning in `junction` |
| --- | --- |
| **Target** | The exact linear 5′→3′ DNA string expected before any later cleavage or cloning step. |
| **Assembly group** | Boundary across which `junction` compares candidate sequences. Targets belong together when their fragments must be designed against one another because they may encounter one another during the intended three-way-junction assembly. The field does not identify procurement, annealing, PCR, a study, or a biological condition. |
| **Locus** | A planner-defined decision site containing several possible target-derived toehold windows. |
| **Selected junction** | The software record that binds one chosen toehold to one external barcode. It is not the complete physical three-way-junction complex. |
| **Toehold** | A short sequence copied from the target at a locus. It remains part of the reconstructed target. |
| **Barcode** | An externally generated assembly sequence assigned to a junction. It is absent from the reconstructed target. This is not a sequencing index. |
| **Domain** | Target sequence between neighboring selected toeholds. |
| **Fragment** | One paired design unit containing two orderable oligos. It is not a single target substring or a single order row. |
| **Barcode-bearing strand** | The fragment oligo carrying target-derived sequence and the external barcode arms. The papers call this the barcode oligo. |
| **Complement strand** | The paired, shifted oligo whose sequence contributes to the ligated complement. The papers call this the coding oligo, even when a submitted target is not a coding sequence. |
| **Recovery primer** | A terminal primer supplied in the request or extracted from a target at an explicit length. Its target-complementary span is the primer-binding site; it may also carry an optional uninterpreted 5′ extension. |

## What the software verifies

| Check | Result |
| --- | --- |
| Request schema and declared limits | Checked before planning. |
| Target and reverse-complement reconstruction | Checked from the composed fragment strings. |
| Primer matches and order lengths | Checked against the complete request. |
| Search seed, scores, budgets, and selected assignments | Recorded in the bundle. |
| Thermodynamic structure and crosstalk | Recorded as `not_run`. |

The bundle verifies string construction and file integrity. Primer efficiency,
secondary structure, synthesis, ligation, amplification, and yield require
separate evidence.

For formulas and exact algorithm choices, read [Method
v1](../reference/method-v1.md). For the paper terminology, reported dimensions,
implementation differences, and validation boundaries, read [Sources and
scope](../reference/sources.md).
