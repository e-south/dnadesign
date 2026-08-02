## ADR 0003: TriJunction as a peer three-way-junction planner

**Status:** accepted
**Date:** 2026-08-01
**Owner:** dnadesign-maintainers

### Context

The Sidewinder papers describe DNA construction through temporary three-way
junctions. Target-derived toeholds and external barcode pairs guide coding
strand ligation, in the papers' terminology; the barcode helices are later
removed during recovery. The later bioRxiv preprint describes pooled
construction and a stochastic string-search procedure named PyWinder.

DNA Design needs a local planner for exact targets without making a study,
Construct, or Cruncher own this molecular lifecycle. The repository owner has
authorized an independently derived implementation and public PR review. That
engineering authorization is not a representation of institutional, patent,
trademark, or method-use clearance.

### Decision

Add `dnadesign.trijunction` as a peer tool with one public `trijunction`
command. TriJunction compiles exact DNA targets into checked
three-way-junction oligo plans and vendor-neutral order rows. It is an
independent, Sidewinder-inspired implementation, not the paper-described
PyWinder procedure and not an official Sidewinder product.

TriJunction owns strict request parsing, bounded deterministic search, fragment
and strand composition, method-specific checks, recovery geometry, neutral order
rows, and create-only local bundle publication. It uses sibling public contracts
only. Its executable schemas, algorithms, tests, and operator guide live with
the feature code in a separate reviewed PR; this ADR does not repeat them.

Public contracts use `target_specific`, `barcode_bearing_strand`,
`complement_strand`, and `assembled_complement`, with corresponding
`complement_*` end-preparation, nick-geometry, and purification fields. The
paper terms "construct-specific", "barcode strand", and "coding strand" appear
only when describing the cited method. No legacy aliases or compatibility shims
are introduced: "coding" would be false for arbitrary noncoding targets.

### Primary-source boundary

The implementation is derived from these two primary sources:

- Robinson, N. E. et al., [Construction of complex and diverse DNA sequences
  using DNA three-way junctions](https://doi.org/10.1038/s41586-025-10006-0),
  *Nature* **651**, 491-500 (2026).
- Robinson, N. E. et al., [One-pot parallel Sidewinder construction from oligo
  pools](https://doi.org/10.64898/2026.05.01.722326), bioRxiv preprint, posted
  2 May 2026.

The Nature article establishes the peer-reviewed molecular method. The
preprint demonstrates both target-specific (its term is "construct-specific")
and universal amplification after pooled assembly and describes the PyWinder
search procedure, but it is not peer reviewed and states that reuse requires
permission. Its later Type IIS/Golden Gate processing removes universal flanks
for hierarchical assembly; that is a downstream use of recovered amplicons,
not a Sidewinder junction-planning contract. Adjacent draft schemas,
specifications, scripts, and examples in the workspace's
`resources/sidewinder-papers/` directory are excluded from normative evidence.
No unreleased PyWinder source is copied or treated as a compatibility target.

The papers leave complete seed, tie-break, coordinate, and serialization
behavior underspecified. TriJunction declares those choices under its own
versioned contracts and does not claim bit-for-bit PyWinder compatibility.
Unlike the preprint's described barcode-pool search, the first release does not
silently relax sequence-symmetry constraints when a search is infeasible.

### Molecular and pool invariants

All targets assigned to one physical reaction pool are designed together.
Toehold selection, barcode selection, barcode-to-toehold matching, and
cross-reactivity heuristics cover every junction that will co-react in that
pool before assignments are partitioned back to targets. Barcode reuse is
permitted only across explicitly isolated pools.

Every checked plan must prove:

- actual cognate `t/t*` and `b/b*` reverse complementarity with explicit
  antiparallel strand orientation;
- explicit first, internal, and last fragment roles;
- the intended complement-strand adjacency, nick polarity, and sequence
  geometry;
- external barcode-domain sequence is absent from the recovered target, while
  target-derived domains carried on the same barcode-bearing strand remain
  distinguishable;
- ligated complement strands yield `assembled_complement`, exactly equal to the
  reverse complement of the submitted target; and
- canonical request replay reproduces the serialized plan and order rows.

The published recovery mechanism displaces or destroys the physical
barcode-bearing strand and uses the assembled complement as the polymerase
template. TriJunction checks the expected recovered sequence and domain
disposition; it does not claim that physical removal occurred in an experiment.

The order policy must declare complement-strand 5-prime end preparation as
either a vendor-applied phosphate or a phosphorylation precondition to be
fulfilled downstream. The plan and each complement-strand order preserve that
distinction. The paper's laboratory phosphorylation, ligase, buffer, and
temperature conditions are not silently promoted to a TriJunction protocol.
Order rows are a purchasing projection, not evidence that chemistry
preconditions were fulfilled or that a laboratory protocol was approved.

Recovery primers may be `target_specific` or `universal`. Each primer separates
its target-annealing `binding_sequence` from an exact, possibly empty,
`five_prime_extension`. Binding and orientation checks use only the binding
sequence. The full ordered primer is the exact 5-prime-to-3-prime concatenation
of extension and binding sequence; the plan, checks, and order rows preserve
both components and the full sequence without normalization or inference.

Target-specific primer pairs may differ by target. Universal recovery requires
one identical forward/reverse pair across every target in a physical pool,
including identical binding sequences and 5-prime extensions. The shared
binding sequences must anneal to common declared terminal regions in each
target. Extensions are not treated as target-binding sequence and do not alter
the assembly proofs for `target` or `assembled_complement`.

Recovery evidence separately proves the extension-bearing duplex. Written
5-prime to 3-prime, its target-oriented strand is
`forward_extension + target + reverse_complement(reverse_extension)`; its
complement-oriented strand is
`reverse_extension + assembled_complement + reverse_complement(forward_extension)`.
Primer binding sequences already occur at the target termini and are not
duplicated in either formula. Both exact recovered strands are persisted and
verified independently from the unextended assembly target.

Five-prime extensions are semantically opaque payloads for later adapters,
Type IIS sites, or other caller-owned processing. The first release does not
select an enzyme or validate recognition sites, strand orientation, digestion,
cut offsets, or resulting overhangs. A downstream tool or study that assigns
those meanings owns their design and verification.

### Verification boundary

TriJunction's edit-distance, shared-substring, GC, homopolymer, reconstruction,
and recovery checks verify the software plan only. They do not prove molecular
orthogonality, ligation fidelity, primer specificity, assembly yield, or
experimental success.

Thermodynamic screening is explicitly `not_run` in the first release. The tool
does not add a private NUPACK wrapper or present string heuristics as
thermodynamic evidence. A future multi-strand DNA screen requires a separate
Folding public-contract decision, named backend and version, persisted inputs
and outputs, and a TriJunction-owned acceptance policy.

### Ownership and non-goals

- Studies own target identity, selection rationale, constraints, and
  experimental state.
- Construct owns realization of declared sequence compositions, not junction
  search or orthogonality policy.
- Cruncher keeps its current sequence-optimization families and does not absorb
  TriJunction's molecular-plan lifecycle.
- Folding owns reusable thermodynamic execution and receipts, not
  junction-specific interpretation.
- TriJunction does not approve orders, execute laboratory protocols, or
  interpret assay evidence.

Automatic primer design, hierarchical assembly planning, empirical read
classification, automatic target mutation, vendor submission, remote jobs, and
laboratory-protocol execution are outside the first release. Type IIS
recognition, orientation, digestion, and overhang validation are also outside
the boundary even when an exact primer extension contains such a site.

### Security and publication

- Planning is local and makes no network, vendor, or ordering calls.
- Targets, identifiers, and order metadata are validated before durable writes;
  output text cannot carry spreadsheet formulas or control characters.
- Input sizes and search budgets are explicit and fail closed.
- Preflight and planning perform no durable writes.
- Final publication uses DNA Design's create-only staged directory primitive
  and installs a new destination only when that path is absent and the staged
  bundle has passed validation.
- The manifest binds the canonical request, plan, checks, order rows, and
  study-neutral review records by digest. Verification reads a stable
  descriptor-held snapshot, replays the plan, and rejects tampering or a
  concurrent filesystem change.
- TriJunction does not copy study records into its bundle or transmit target
  sequences. An operator must explicitly move or submit order data.

### Rights and naming boundary

Documentation and package metadata describe TriJunction as
"Sidewinder-inspired" and link both primary papers. `Sidewinder` and
`PyWinder` are not used as DNA Design's package, command, or schema identity.
The TriJunction name has not received trademark clearance.

The Nature article is open access under the publisher's stated license. The
bioRxiv preprint is permission restricted. Both papers disclose patent
activity, and the preprint discloses author relationships with Genyro. Public
source review authorized by the repository owner does not grant patent or
method-use rights. Release packaging, commercial distribution, ordering, and
laboratory use require the owner's applicable institutional and legal review.

### Consequences

- Three-way-junction planning gains a coherent peer boundary instead of
  expanding Cruncher or Construct.
- Pool-wide design, exact reverse-complement reconstruction, end-state policy,
  recovery geometry, and create-only publication are enforceable contracts.
- Fail-closed search can reject requests the paper-described procedure might
  satisfy through constraint relaxation.
- A string-verified plan remains explicitly weaker than thermodynamic or
  experimental evidence.
- Supporting new chemistry, thermodynamics, recovery design, remote services,
  or ordering authority requires a new decision rather than implicit scope
  growth.

### Links

- Archived proposal:
  `docs/dev/plans/archive/2026-08-01-trijunction-boundary-proposal.md`
- Documentation decision PR: [#64](https://github.com/e-south/dnadesign/pull/64)
- Feature implementation: [#67](https://github.com/e-south/dnadesign/pull/67)
