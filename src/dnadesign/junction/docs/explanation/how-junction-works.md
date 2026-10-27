---
doc_id: junction-how-it-works
title: junction method overview
type: explanation
audience: readers new to three-way-junction DNA assembly
owner: dnadesign-maintainers
status: active
last_verified: 2026-08-10
---

# Method overview

`junction` follows the sequence model described by Robinson *et al.* for
three-way-junction assembly and its pooled-oligo extension. It starts from
exact linear DNA targets and supplied PCR primers.

## The planned sequence path

1. **Choose target-derived toeholds.** Candidate windows near each planned
   boundary are compared within an assembly group. One window is selected at
   each locus.
2. **Assign temporary barcodes.** Generated barcode strings are filtered and
   compared with the selected toeholds and with one another.
3. **Compose fragment pairs.** Every fragment has a barcode-bearing strand and
   a shifted complement strand. The target-derived regions are checked for
   exact pairing and reconstruction.
4. **Describe the interfaces.** Neighboring barcode arms form the vertical
   helix; a target-derived toehold forms part of the horizontal helix and
   places the complement-strand nick.
5. **Define the PCR product.** The expected duplex is derived from the submitted
   target and the exact terminal primer extensions in the request.

The [gene-scale tutorial](../getting-started.md) shows each stage with the
deterministic records and figures produced by the software.

## One-interface sequence model

```text
target                         D0 · t1 · D1

fragment 0, barcode-bearing   D0 · t1 · b1
fragment 0, complement        rc(D0)

fragment 1, barcode-bearing   rc(b1) · D1
fragment 1, complement        rc(D1) · rc(t1)

reconstructed complement      rc(D1) · rc(t1) · rc(D0)
```

`D0` and `D1` are target domains, `t1` is the selected target-derived
toehold, `b1` is the temporary barcode, and `rc(...)` is reverse complement.
The barcode is absent from the reconstructed target.

## Terms

| Term | Meaning |
| --- | --- |
| Target | Exact linear 5′→3′ DNA sequence to reconstruct. |
| Assembly group | Targets whose candidate strings must be compared together. |
| Locus | Candidate target windows for one planned interface. |
| Toehold | Selected target-derived sequence retained in the target. |
| Barcode | Temporary external sequence assigned to an interface; not a sequencing index. |
| Fragment | One paired design unit with two orderable oligos. |
| Barcode-bearing strand | Oligo carrying target-derived sequence and external barcode arms. |
| Complement strand | Shifted paired oligo that contributes to the ligated complement. |
| PCR primer | Supplied terminal primer, recorded under `recovery_primers`, with an optional exact 5′ extension. |

The papers call complement strands “coding oligos.” `junction` uses the more
general term because a target need not encode a protein.

## Software boundary

`junction` checks request validity, bounded search work, exact strand
composition, target reconstruction, primer terminal matches, order lengths,
bundle identities, and replay. Thermodynamic screening remains an explicit
`not_run` state. Synthesis, annealing, ligation, amplification, yield, and
experimental acceptance need separate evidence.

For formulas and deterministic choices, use [Method
v1](../reference/method-v1.md). For the primary papers, reported dimensions,
and differences from the published workflows, use [Sources and
scope](../reference/sources.md).
