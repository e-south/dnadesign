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

`junction` is a sequence planner. It turns a complete request into oligo
sequences, exact reconstruction evidence, and a reproducible review bundle.

## The physical idea

The Sidewinder method represents each target fragment with two paired oligos.
The barcode oligo carries target sequence and one or two external barcode arms,
depending on the fragment's position. The coding oligo is mostly complementary
to the target-facing part and leaves a short target-derived toehold exposed.
When intended neighbors meet, barcode and toehold pairing form a three-way
junction. Ligase joins the nick between adjacent coding oligos. Polymerase
extension, often within PCR, then displaces or destroys the temporary barcode
oligos and restores ordinary double-stranded target DNA.

The external barcode is an assembly aid. It is not part of the submitted target
that `junction` reconstructs. “Sequence-independent” in the paper refers to
that removable assembly code; it does not mean every target, parameter set, or
reaction is guaranteed to work.

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
These checks establish the sequence composition. Physical outcomes require
separate experimental evidence.

## From a request to order rows

1. **Read the caller's decisions.** The request supplies exact targets,
   assembly groups, recovery primers, search settings, and order labels.
2. **Enumerate loci.** For each target, the planner finds regularly spaced
   decision sites. Each locus contains several target-derived toehold windows.
3. **Select toeholds.** Within one assembly group, the planner selects one
   candidate per locus while favoring string separation between selected
   sequences.
4. **Generate and assign barcodes.** It creates external barcode candidates,
   filters them with declared string and composition rules, selects a separated
   subset, and assigns one barcode to each selected junction.
5. **Compose fragments.** Selected toeholds divide each target into domains.
   Every fragment produces a barcode-bearing strand and a complement strand in
   explicit 5′→3′ order.
6. **Check strings and limits.** The target domains and toeholds must reconstruct
   the submitted target exactly. Reverse-order complement strands must
   reconstruct its reverse complement. Primer strings must match the target
   termini, every fragment order must meet the caller's declared minimum, and
   every order row must fit the declared maximum.
7. **Publish evidence.** `build` writes the request, plan, checks, order rows,
   review records, and manifest to a new directory, then verifies the installed
   bundle by replay.

The search is deterministic for the complete request and seed. Adding a target
to an assembly group can change the group's selected toeholds and barcodes.

## Vocabulary

| Term | Meaning in `junction` |
| --- | --- |
| **Target** | The exact linear 5′→3′ DNA string expected before any later cleavage or cloning step. |
| **Assembly group** | Boundary across which `junction` compares candidate sequences. Targets belong together when their fragments must be designed against one another because they may encounter one another during the intended three-way-junction assembly. The field does not identify procurement, annealing, PCR, a study, or a biological condition. |
| **Locus** | A planner-defined decision site containing several possible target-derived toehold windows. |
| **Nominal fragment-oligo length** | A locus-spacing parameter. It does not promise an equal physical length for every emitted order. |
| **Selected junction** | The software record that binds one chosen toehold to one external barcode. It is not the complete physical three-way-junction complex. |
| **Toehold** | A short sequence copied from the target at a locus. It remains part of the reconstructed target. |
| **Barcode** | An externally generated assembly sequence assigned to a junction. It is absent from the reconstructed target. This is not a sequencing index. |
| **Domain** | Target sequence between neighboring selected toeholds. |
| **Fragment** | One paired design unit containing two orderable oligos. It is not a single target substring or a single order row. |
| **Barcode-bearing strand** | The fragment oligo carrying target-derived sequence and the external barcode arms. The papers call this the barcode oligo. |
| **Complement strand** | The paired, shifted oligo whose sequence contributes to the ligated complement. The papers call this the coding oligo, even when a submitted target is not a coding sequence. |
| **Recovery primer** | A caller-supplied terminal primer string with an optional uninterpreted 5′ extension. |
| **Plan** | The in-memory targets, selected assignments, strands, order rows, search receipts, and checks. |
| **Bundle** | A create-only directory containing a reproducible design record. External mutation is possible; later verification detects it. |
| **Review record or image** | A compact inspection view. JSON review records belong to the verified bundle; optional BaseRender images do not change plan identity. |

## Paper terms and software terms

| Primary literature | `junction` | Note |
| --- | --- | --- |
| target construct or CDS | target | `junction` accepts any exact linear DNA target, not only coding sequences. |
| Sidewinder fragment | fragment | Both mean the paired heteroduplex design unit. |
| barcode oligo | barcode-bearing strand | Neutral name for coding and non-coding targets. |
| coding oligo | complement strand | Neutral name; it does not imply gene coding sequence. |
| Sidewinder barcode or helix | barcode and its complement | Temporary external assembly code. |
| 3WJ assembly | modeled pre-ligation junctions | Expected strand layout before the recorded nick is ligated. |
| restored 2WJ | expected recovered duplex | Exact primer-extended target and complement strings. |
| construct-specific PCR | `target_specific` recovery | The spelling is generic; the caller still supplies the primers. |
| universal PCR | `universal` recovery | Method v1 only checks one shared exact primer pair for the group. |

## What the checks mean

| Question | Status |
| --- | --- |
| Does the request match the v2 schema and declared limits? | Checked. |
| Do selected target domains and toeholds reconstruct the submitted target string? | Checked. |
| Do the complement strings reconstruct `rc(target)` in the modeled order? | Checked. |
| Are the selected barcode and toehold assignments one-to-one within the assembly group? | Checked as strings. |
| Do supplied primer strings match the submitted target termini? | Checked. |
| Are order rows within the caller's length ceiling? | Checked. |
| Were deterministic search scores and seeds recorded? | Recorded. |
| Was thermodynamic structure or crosstalk screened? | `not_run`. |
| Are nick chemistry, phosphorylation, ligase docking, or reaction conditions ready? | Not evaluated. |
| Will primers amplify efficiently or avoid off-targets beyond exact declared matches? | Not evaluated. |
| Will a supplier synthesize the oligos or will the assembly work? | Not evaluated. |

## Where this differs from the papers

The Nature study standardized 10-nt toeholds after observing effective ligation
when the nick was at least six bases from the barcode helix, tested 15-to-21-nt
barcodes, and mainly used 120-mer oligos. The pooled preprint's usual profile
was `L=96`, `b=22`, `t=10`, and `R=15`, yielding stated final oligo lengths of
82 to 110 nt. These are reported experimental or algorithmic contexts, not
validated `junction` presets.

The two sources use `L` differently. In the Nature paper, `L` is the physical
input-oligo length and the coding capacity is `L - 2b`. In the pooled
preprint's generator, `L` is a nominal coordinate parameter and candidate
offsets can change the emitted length. `junction` follows the latter geometry
and names the field `nominal_fragment_oligo_length` to keep that distinction
visible. Its terminal-locus rule can also produce short terminal fragment
orders, so the caller must declare a separate minimum fragment-oligo length.

The current planner accepts a broader software geometry, uses deterministic
string objectives, adds fixed GC and homopolymer filters, and fails instead of
relaxing declared substring constraints. It also selects toeholds jointly
across an assembly group, whereas the pooled paper describes selecting
toeholds target by target before global barcode design and matching.

The `universal` recovery mode requires one exact caller-supplied primer pair
across an assembly group and consolidates the shared order rows. Shared priming
regions, length-equalizing buffers, internal Type IIS sites, and later
hierarchical assembly must already be designed upstream.

Universal recovery also has a source-observed risk that the request mode does
not evaluate. The preprint notes that PCR can favor shorter products, so a
truncated misassembly with both terminal priming regions can be preferentially
amplified. Under the reported conditions, the universal experiment observed a
junction misconnection rate of 1 in 217,985, while the highlighted
construct-specific condition observed 1 in 10,048,851. Those counts describe
different experiments in that preprint; they are not a general error rate or a
prediction for a `junction` plan.

Laboratory preparation also differs between the two sources. The Nature paper
phosphorylates coding oligos before annealing fragment pairs; the pooled
preprint phosphorylates and anneals the pooled oligos together. `junction`
records the caller's complement-end preparation choice without selecting a
complete reaction protocol.

## Current boundaries

- Requests are complete JSON or YAML documents; bare-sequence and FASTA
  onboarding are not yet available.
- Primer choice, melting-temperature analysis, broader off-target search, and
  thermodynamic screening remain upstream review steps.
- Post-Type-IIS payload modeling and degenerate-library compilation are outside
  method v1. Concrete library members can be submitted as exact targets.
- `assembly_group_id` controls joint sequence search. It is not a vendor-pool
  allocation or reaction recipe.
- Search receipts retain selected results and aggregate scores rather than a
  full rejected-candidate trace or live progress stream.
- Comparative software validation and laboratory validation of
  `junction`-generated designs remain future evidence work.

For formulas and exact algorithm choices, read [Method
v1](../reference/method-v1.md). For source status and claim boundaries, read
[Sources and scope](../reference/sources.md).
