## ADR 0003: `junction` as a peer three-way-junction planner

**Status:** accepted
**Date:** 2026-08-01
**Owner:** dnadesign-maintainers

### Context

The Sidewinder papers describe DNA construction through temporary three-way
junctions. Target-derived toeholds and external barcode pairs guide coding
strand ligation, in the papers' terminology; the barcode helices are later
removed during recovery. The later bioRxiv preprint describes pooled
construction and a stochastic string-search procedure named PyWinder.

DNA Design needs a local planner for exact targets. Study code, Construct, and
Cruncher should not absorb this method-specific planning. The repository owner
authorized an independently developed implementation and public PR review. That
engineering decision does not grant institutional, patent, trademark, or
method-use clearance.

### Decision

Add `dnadesign.junction` as a peer tool with one public `junction` command.
`junction` turns exact DNA targets into checked three-way-junction oligo plans
and vendor-neutral order rows. It is an independent, Sidewinder-inspired
implementation, not the paper-described PyWinder procedure or an official
Sidewinder product.

`junction` owns strict request parsing, bounded deterministic search, fragment
and strand composition, method-specific checks, recovery geometry,
vendor-neutral order rows, and local bundles that never overwrite an existing
destination. It uses only public contracts from sibling tools. Its schemas,
algorithms, tests, and user guide live with the feature code; this ADR does
not repeat them.

The versioned request, method, and artifact contracts live with the tool and
its tests. This ADR chooses ownership and scope; it is not a second technical
specification. No import, command, schema, or file-format alias is retained.

### Sources and evidence

The implementation is derived from these two primary sources:

- Robinson, N. E. et al., [Construction of complex and diverse DNA sequences
  using DNA three-way junctions](https://doi.org/10.1038/s41586-025-10006-0),
  *Nature* **651**, 491-500 (2026).
- Robinson, N. E. et al., [One-pot parallel Sidewinder construction from oligo
  pools](https://doi.org/10.64898/2026.05.01.722326), bioRxiv preprint, posted
  2 May 2026.

The Nature article reports the peer-reviewed molecular method and its
experimental evaluation. The inspected bioRxiv preprint reports
construct-specific and universal amplification after pooled assembly and
describes the PyWinder search procedure. It is not peer reviewed and states
that reuse requires permission. Its later Type IIS/Golden Gate processing
removes universal flanks for hierarchical assembly; that is a downstream use
of recovered amplicons, not a Sidewinder junction-planning contract. Adjacent
draft schemas, specifications, scripts, and examples in the workspace's
`resources/sidewinder-papers/` directory are not used as sources.
No PyWinder source is copied or treated as a compatibility target.

The papers leave complete seed, tie-break, coordinate, and serialization
behavior underspecified. `junction` declares those choices under its own
versioned contracts and does not claim bit-for-bit PyWinder compatibility.
Unlike the preprint's described barcode-pool search, the first release does not
silently relax sequence-symmetry constraints when a search is infeasible.

### Stable boundaries

- Studies own target identity, selection rationale, experimental state, and
  downstream interpretation.
- An assembly group is only the boundary across which `junction` compares
  candidate sequences. Targets belong together when their fragments may
  encounter one another during the intended three-way-junction assembly. The
  ID is not a vendor pool, tube, PCR product, study, sample, or condition.
- `junction` owns deterministic string planning, explicit sequence checks,
  vendor-neutral order rows, and replayable local bundles.
- Construct owns realization of declared sequence compositions. Cruncher keeps
  its existing optimization families. Folding owns reusable thermodynamic
  execution and result records.
- String checks do not establish thermodynamic orthogonality, primer
  performance, ligation fidelity, assembly yield, or experimental success.
- Automatic primer design, thermodynamic screening, Type IIS design,
  hierarchical assembly, vendor submission, ordering, and laboratory execution
  are outside this decision.

Exact fields, formulas, limits, and evidence meanings are defined once in the
tool-owned [request](../../../src/dnadesign/junction/docs/reference/request.md),
[method](../../../src/dnadesign/junction/docs/reference/method-v1.md), and
[artifact](../../../src/dnadesign/junction/docs/reference/artifacts-api-and-errors.md)
references.

### Security and publication

Planning is local and makes no network, vendor, ordering, or laboratory calls.
Inputs and workloads are bounded before publication. Bundles are create-only,
content-identified, and replay-verified; later verification detects mutation
but does not make files immutable. `junction` does not copy study records into
its bundle or transmit target sequences.

### Naming and rights

Documentation and package metadata describe `junction` as
"Sidewinder-inspired" and link both primary papers. `Sidewinder` and
`PyWinder` are not used as DNA Design's package, command, or schema identity.
The `junction` name has not received trademark clearance.

The Nature article is open access under the publisher's stated license. The
bioRxiv preprint is permission restricted. Both papers disclose patent
activity, and the preprint discloses author relationships with Genyro and
Syntaxa. Public source review authorized by the repository owner does not grant
patent or method-use rights. Release packaging, commercial distribution,
ordering, and laboratory use require the owner's applicable institutional and
legal review.

### Consequences

- Three-way-junction planning has a separate tool instead of expanding Cruncher
  or Construct.
- Assembly-group-wide design, exact reverse-complement reconstruction,
  end-state policy, recovery geometry, and no-overwrite publication are
  enforceable contracts.
- Search fails instead of relaxing constraints, so it can reject requests that
  the procedure described in the paper might satisfy.
- A string-verified plan remains explicitly weaker than thermodynamic or
  experimental evidence.
- Supporting new chemistry, thermodynamics, recovery design, remote services,
  or ordering authority requires a new decision rather than implicit scope
  growth.

### Links

- [Archived proposal](../../dev/plans/archive/2026-08-01-junction-boundary-proposal.md)
- Documentation decision PR: [#64](https://github.com/e-south/dnadesign/pull/64)
- Feature implementation: [#67](https://github.com/e-south/dnadesign/pull/67)
