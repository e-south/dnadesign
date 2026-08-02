## TriJunction Boundary Proposal

**Status:** superseded by
[ADR 0003](../../../architecture/decisions/adr-0003-trijunction-peer-tool.md)
**Date:** 2026-08-01
**Owner:** dnadesign-maintainers

### Question

Where should a planner for Sidewinder-inspired three-way-junction oligos live,
and what should its first boundary include?

The proposal evaluated the two primary papers, DNA Design's existing tool
boundaries, local publication mechanics, and the adjacent workspace drafts.
The drafts were excluded from normative evidence because they were exploratory
and not independently verified.

### Options considered

1. Add the planner to Cruncher. Rejected because junction planning has a
   molecular-plan, recovery, verification, and ordering lifecycle distinct from
   Cruncher's current sequence-optimization families.
2. Add it to Construct. Rejected because Construct realizes declared sequence
   compositions and should not own junction search or orthogonality policy.
3. Add a peer tool. Preferred because it gives the method one explicit public
   contract while allowing studies and sibling tools to pass exact sequences
   through public boundaries.
4. Name the package Sidewinder or PyWinder. Rejected because those names belong
   to the published method and paper-described procedure and would overstate
   provenance or compatibility.

### Evidence that shaped the proposal

- The peer-reviewed paper requires actual `t/t*` and `b/b*` complementarity,
  ligation of what it calls coding strands, barcode-helix removal, and exact
  product recovery. TriJunction names those oligos complement strands because
  arbitrary targets need not encode proteins.
- The later preprint designs one global toehold and barcode set for targets that
  co-react before partitioning assignments back to targets.
- Complement-strand phosphorylation is a chemistry precondition and cannot be
  omitted from ordering semantics.
- Target-specific and universal recovery have different terminal geometry.
- The preprint's later Type IIS/Golden Gate step is downstream processing, so a
  generic 5-prime primer extension can preserve caller intent without making
  enzyme or overhang semantics part of TriJunction.
- String-distance checks and optional thermodynamic ranking are different
  evidence classes; neither proves experimental assembly success.
- DNA Design already provides a create-only staged publication primitive for
  immutable local bundles.

### Resolution

The repository owner authorized an independently derived implementation and
public PR review. ADR 0003 accepts `dnadesign.trijunction` as a peer tool and
records the stable ownership, pool, molecular, recovery, security,
thermodynamic, naming, and rights boundaries.

The feature's schemas, formulas, algorithms, commands, and examples remain
with its implementation and tests. They are intentionally absent here so this
archived proposal cannot become a competing specification.
